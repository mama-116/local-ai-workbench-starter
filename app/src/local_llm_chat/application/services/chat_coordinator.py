from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Protocol

from local_llm_chat.application.services.memory_capture_service import (
    MemoryCaptureRequest,
    MemoryCaptureScheduler,
)
from local_llm_chat.application.services.telemetry_service import TelemetryService
from local_llm_chat.application.services.translation_service import (
    TranslationScheduler,
)
from local_llm_chat.application.services.turn_batch_generation_service import (
    TurnBatchGenerationService,
)
from local_llm_chat.application.services.turn_batch_service import TurnBatchService
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import Message, RunSession
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import MessageState


StreamCallback = Callable[[str], Awaitable[None]]
NoticeCallback = Callable[[str, bool], Awaitable[None]]


async def _no_update(_: str) -> None:
    return None


async def _no_notice(_: str, __: bool) -> None:
    return None


class SingleChatService(Protocol):
    async def send_message(
        self,
        conversation_id: str,
        content: str,
        on_update: StreamCallback = _no_update,
        on_notice: NoticeCallback = _no_notice,
    ) -> Message: ...

    async def rewrite_message(
        self,
        conversation_id: str,
        source_message_id: str,
        content: str,
        on_update: StreamCallback = _no_update,
        on_notice: NoticeCallback = _no_notice,
    ) -> Message: ...

    async def regenerate_message(
        self,
        conversation_id: str,
        source_message_id: str,
        on_update: StreamCallback = _no_update,
        on_notice: NoticeCallback = _no_notice,
    ) -> Message: ...


class ChatCoordinator:
    def __init__(
        self,
        repository: AppRepository,
        single_chat: SingleChatService,
        group_generation: TurnBatchGenerationService,
        turn_batches: TurnBatchService,
        memory_capture_scheduler: MemoryCaptureScheduler | None = None,
        translation_scheduler: TranslationScheduler | None = None,
        telemetry: TelemetryService | None = None,
    ) -> None:
        self._repository = repository
        self._single_chat = single_chat
        self._group_generation = group_generation
        self._turn_batches = turn_batches
        self._memory_capture_scheduler = memory_capture_scheduler
        self._translation_scheduler = translation_scheduler
        self._telemetry = telemetry

    async def send_message(
        self,
        conversation_id: str,
        content: str,
        on_update: StreamCallback = _no_update,
        on_notice: NoticeCallback = _no_notice,
    ) -> Message:
        settings = await self._repository.get_conversation_group_settings(
            conversation_id
        )
        if not settings.enabled:
            return await self._single_chat.send_message(
                conversation_id, content, on_update, on_notice
            )

        session = await self._repository.start_send(conversation_id, content)
        return await self._execute_group_session(
            session,
            on_update,
            capture_memory=True,
        )

    async def regenerate_turn_batch(
        self,
        conversation_id: str,
        source_response_message_id: str,
        expected_active_branch_id: str,
        on_update: StreamCallback = _no_update,
        on_notice: NoticeCallback = _no_notice,
    ) -> Message:
        del on_notice
        settings = await self._repository.get_conversation_group_settings(
            conversation_id
        )
        if not settings.enabled:
            raise ValidationError("group generation is disabled for this conversation")
        session = await self._repository.start_turn_batch_regenerate(
            conversation_id,
            source_response_message_id,
            expected_active_branch_id,
        )
        return await self._execute_group_session(
            session,
            on_update,
            capture_memory=False,
        )

    async def _execute_group_session(
        self,
        session: RunSession,
        on_update: StreamCallback,
        *,
        capture_memory: bool,
    ) -> Message:
        last_update: str | None = None

        async def forward_update(content: str) -> None:
            nonlocal last_update
            if content == last_update:
                return
            await on_update(content)
            last_update = content

        try:
            draft = await self._group_generation.generate(session, forward_update)
            batch = await self._turn_batches.finish(session, draft)
        except asyncio.CancelledError:
            with suppress(Exception):
                await self._repository.finish_response(
                    session,
                    "",
                    MessageState.CANCELLED,
                    error_code="cancelled_by_user",
                )
            raise
        except Exception as error:
            with suppress(Exception):
                await self._repository.finish_response(
                    session,
                    "",
                    MessageState.FAILED,
                    error_code=type(error).__name__,
                )
            raise

        response = await self._repository.get_message(batch.response_message_id)
        try:
            await forward_update(response.content)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._log_post_commit_failure(
                "group_chat_update_failed", error, session.run.id
            )

        await self._start_post_commit_work(
            session.run.conversation_id,
            session.branch_id,
            session.user_message.id,
            session.run.model,
            session.run.id,
            response.id,
            draft.formal_character_ids,
            capture_memory=capture_memory,
        )
        return response

    async def rewrite_message(
        self,
        conversation_id: str,
        source_message_id: str,
        content: str,
        on_update: StreamCallback = _no_update,
        on_notice: NoticeCallback = _no_notice,
    ) -> Message:
        return await self._single_chat.rewrite_message(
            conversation_id, source_message_id, content, on_update, on_notice
        )

    async def regenerate_message(
        self,
        conversation_id: str,
        source_message_id: str,
        on_update: StreamCallback = _no_update,
        on_notice: NoticeCallback = _no_notice,
    ) -> Message:
        return await self._single_chat.regenerate_message(
            conversation_id, source_message_id, on_update, on_notice
        )

    async def _start_post_commit_work(
        self,
        conversation_id: str,
        branch_id: str,
        source_message_id: str,
        model_name: str,
        run_id: str,
        response_message_id: str,
        formal_character_ids: tuple[str, ...],
        *,
        capture_memory: bool = True,
    ) -> None:
        if capture_memory and self._memory_capture_scheduler is not None:
            request = MemoryCaptureRequest(
                conversation_id=conversation_id,
                branch_id=branch_id,
                source_message_id=source_message_id,
                model_name=model_name,
                author_subject_id="user",
                allowed_subject_ids=frozenset({"user", *formal_character_ids}),
                allowed_knowledge_character_ids=frozenset(formal_character_ids),
            )
            try:
                await self._memory_capture_scheduler.request_capture(request, run_id)
            except Exception as error:
                await self._log_post_commit_failure(
                    "memory_capture_enqueue_failed", error, run_id
                )
        if self._telemetry is not None:
            try:
                self._telemetry.request_capture(run_id)
            except Exception as error:
                await self._log_post_commit_failure(
                    "telemetry_enqueue_failed", error, run_id
                )
        auto_translate = False
        try:
            conversation = await self._repository.get_conversation(conversation_id)
            auto_translate = conversation.auto_translate
        except Exception as error:
            await self._log_post_commit_failure(
                "translation_preference_read_failed", error, run_id
            )
        if auto_translate and self._translation_scheduler is not None:
            try:
                await self._translation_scheduler.request_translation(
                    response_message_id
                )
            except Exception as error:
                await self._log_post_commit_failure(
                    "translation_enqueue_failed", error, run_id
                )

    async def _log_post_commit_failure(
        self, event_type: str, error: Exception, run_id: str
    ) -> None:
        with suppress(Exception):
            await self._repository.log_event(
                "error",
                event_type,
                {"error_type": type(error).__name__},
                run_id,
            )

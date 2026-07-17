from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from local_llm_chat.application.services.translation_service import (
    TranslationScheduler,
)
from local_llm_chat.application.services.telemetry_service import TelemetryService
from local_llm_chat.application.services.rag_service import RagService
from local_llm_chat.domain.errors import (
    AppError,
    FreeOperationBlocked,
    OllamaUnavailable,
)
from local_llm_chat.domain.models import (
    ChatMessageInput,
    ChatRequest,
    Conversation,
    Message,
    RunSession,
)
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.ports.llm_provider import LLMProvider, LLMProviderRegistry
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import MessageState

StreamCallback = Callable[[str], Awaitable[None]]


async def _no_update(_: str) -> None:
    return None


class ChatService:
    def __init__(
        self,
        repository: AppRepository,
        providers: LLMProviderRegistry,
        free_policy: FreeOperationPolicy,
        translation_scheduler: TranslationScheduler | None = None,
        telemetry: TelemetryService | None = None,
        rag: RagService | None = None,
    ) -> None:
        self._repository = repository
        self._providers = providers
        self._free_policy = free_policy
        self._translation_scheduler = translation_scheduler
        self._telemetry = telemetry
        self._rag = rag

    async def send_message(
        self,
        conversation_id: str,
        content: str,
        on_update: StreamCallback = _no_update,
    ) -> Message:
        conversation, provider = await self._preflight(conversation_id)
        session = await self._repository.start_send(conversation_id, content)
        return await self._execute(conversation, session, provider, on_update)

    async def rewrite_message(
        self,
        conversation_id: str,
        source_message_id: str,
        content: str,
        on_update: StreamCallback = _no_update,
    ) -> Message:
        conversation, provider = await self._preflight(conversation_id)
        session = await self._repository.start_rewrite(
            conversation_id, source_message_id, content
        )
        return await self._execute(conversation, session, provider, on_update)

    async def regenerate_message(
        self,
        conversation_id: str,
        source_message_id: str,
        on_update: StreamCallback = _no_update,
    ) -> Message:
        conversation, provider = await self._preflight(conversation_id)
        session = await self._repository.start_regenerate(
            conversation_id, source_message_id
        )
        return await self._execute(conversation, session, provider, on_update)

    async def _preflight(
        self, conversation_id: str
    ) -> tuple[Conversation, LLMProvider]:
        conversation = await self._repository.get_conversation(conversation_id)
        profile = await self._repository.get_model_profile(conversation.model_profile_id)
        provider = self._providers.get(profile.provider)
        self._free_policy.require_cloud_disabled(
            self._providers.cloud_is_disabled(profile.provider)
        )
        self._free_policy.require_provider(provider.metadata)
        model = await provider.inspect_model(profile.model_name)
        self._free_policy.require_model(model)
        return conversation, provider

    async def _execute(
        self,
        conversation: Conversation,
        session: RunSession,
        provider: LLMProvider,
        on_update: StreamCallback,
    ) -> Message:
        profile = await self._repository.get_model_profile(conversation.model_profile_id)
        character = await self._repository.get_character_version(
            session.run.character_version_id
        )
        context = await self._repository.context_to_message(session.user_message.id)
        system_prompt = character.system_prompt
        if self._rag is not None:
            selected_document_count, rag_results = await self._rag.prepare_for_conversation(
                conversation.id, session.user_message.content
            )
            await self._rag.attach_to_run(
                session.run.id, selected_document_count, rag_results
            )
            if rag_results:
                system_prompt = (
                    f"{system_prompt}\n\n{self._rag.prompt_context(rag_results)}"
                )
        request = ChatRequest(
            model=profile.model_name,
            system_prompt=system_prompt,
            messages=tuple(
                ChatMessageInput(message.role, message.content)
                for message in context
                if message.content
            ),
            options=dict(profile.parameters),
        )
        content = ""
        checkpoint_content_length = 0
        last_checkpoint = time.monotonic()
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        total_duration_ns: int | None = None
        generation_duration_ns: int | None = None
        done = False
        response_started = time.monotonic()
        try:
            async for chunk in provider.stream_chat(request):
                if chunk.content:
                    content += chunk.content
                    await on_update(content)
                now = time.monotonic()
                if content and (
                    len(content) - checkpoint_content_length >= 256
                    or now - last_checkpoint >= 0.25
                ):
                    await self._repository.checkpoint_response(
                        session.assistant_message.id, content
                    )
                    checkpoint_content_length = len(content)
                    last_checkpoint = now
                if chunk.done:
                    done = True
                    prompt_tokens = chunk.prompt_tokens
                    output_tokens = chunk.output_tokens
                    total_duration_ns = chunk.total_duration_ns
                    generation_duration_ns = chunk.generation_duration_ns
            if not done:
                raise OllamaUnavailable("Ollamaの応答が完了前に終了しました。")
            response = await self._repository.finish_response(
                session,
                content,
                MessageState.COMPLETED,
                prompt_tokens,
                output_tokens,
                total_duration_ns,
                generation_duration_ns,
                round((time.monotonic() - response_started) * 1000),
            )
            if self._telemetry is not None:
                self._telemetry.request_capture(session.run.id)
            if self._translation_scheduler is not None:
                try:
                    await self._translation_scheduler.request_translation(response.id)
                except Exception as error:
                    await self._repository.log_event(
                        "error",
                        "translation_enqueue_failed",
                        {"error_type": type(error).__name__},
                        session.run.id,
                    )
            return response
        except asyncio.CancelledError:
            await self._repository.finish_response(
                session,
                content,
                MessageState.CANCELLED,
                error_code="cancelled_by_user",
            )
            raise
        except AppError as error:
            await self._repository.finish_response(
                session,
                content,
                MessageState.FAILED,
                error_code=type(error).__name__,
            )
            await self._repository.log_event(
                "error",
                "chat_failed",
                {"error_type": type(error).__name__},
                session.run.id,
            )
            raise
        except Exception as error:
            await self._repository.finish_response(
                session,
                content,
                MessageState.FAILED,
                error_code=type(error).__name__,
            )
            await self._repository.log_event(
                "error",
                "chat_failed_unexpectedly",
                {"error_type": type(error).__name__},
                session.run.id,
            )
            raise

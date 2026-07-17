from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from local_llm_chat.application.services.translation_service import (
    TranslationScheduler,
)
from local_llm_chat.application.services.telemetry_service import TelemetryService
from local_llm_chat.application.services.rag_service import RagService
from local_llm_chat.application.services.tool_coordinator import ToolCoordinator
from local_llm_chat.domain.errors import (
    AppError,
    FreeOperationBlocked,
    OllamaUnavailable,
    ToolUseUnavailable,
)
from local_llm_chat.domain.models import (
    ChatMessageInput,
    ChatRequest,
    Conversation,
    Message,
    RunSession,
    ToolCallRequest,
)
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.ports.llm_provider import LLMProvider, LLMProviderRegistry
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import MessageRole, MessageState

StreamCallback = Callable[[str], Awaitable[None]]
_READ_TOOL_NAME = "read_allowed_text"


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
        tools: ToolCoordinator | None = None,
    ) -> None:
        self._repository = repository
        self._providers = providers
        self._free_policy = free_policy
        self._translation_scheduler = translation_scheduler
        self._telemetry = telemetry
        self._rag = rag
        self._tools = tools

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
        base_messages = tuple(
            ChatMessageInput(message.role, message.content)
            for message in context
            if message.content
        )
        grant = await self._repository.get_tool_folder_grant(conversation.id)
        definitions = (
            self._tools.definitions()
            if self._tools is not None and grant is not None
            else ()
        )
        request = ChatRequest(
            model=profile.model_name,
            system_prompt=system_prompt,
            messages=base_messages,
            options=dict(profile.parameters),
            tools=definitions,
        )
        content = ""
        checkpoint_content_length = 0
        last_checkpoint = time.monotonic()
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        total_duration_ns: int | None = None
        generation_duration_ns: int | None = None
        response_started = time.monotonic()

        async def consume(
            chat_request: ChatRequest,
        ) -> tuple[str, tuple[ToolCallRequest, ...]]:
            nonlocal content
            nonlocal checkpoint_content_length, last_checkpoint
            nonlocal prompt_tokens, output_tokens
            nonlocal total_duration_ns, generation_duration_ns
            streamed = ""
            calls: list[ToolCallRequest] = []
            done = False
            async for chunk in provider.stream_chat(chat_request):
                calls.extend(chunk.tool_calls)
                if chunk.content:
                    streamed += chunk.content
                    content = streamed
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
            return streamed, tuple(calls)

        try:
            try:
                initial_content, tool_calls = await consume(request)
            except ToolUseUnavailable:
                request = ChatRequest(
                    request.model,
                    request.system_prompt,
                    request.messages,
                    request.options,
                )
                initial_content, tool_calls = await consume(request)

            read_item_count = 0
            calls_used = 0
            result_bytes_used = 0
            accumulated_messages = list(base_messages)
            current_content = initial_content
            while (
                tool_calls
                and self._tools is not None
                and self._tools.can_request_more(calls_used)
            ):
                results = await self._tools.execute_calls(
                    conversation.id,
                    session.run.id,
                    tool_calls,
                    calls_used=calls_used,
                    result_bytes_used=result_bytes_used,
                )
                calls_used += len(tool_calls)
                read_item_count += sum(
                    result.item_count
                    for call, result in zip(tool_calls, results, strict=False)
                    if call.name == _READ_TOOL_NAME and not result.is_error
                )
                result_bytes_used += sum(
                    len(result.content.encode("utf-8"))
                    for result in results
                    if not result.is_error
                )
                accumulated_messages.append(
                    ChatMessageInput(
                        MessageRole.ASSISTANT,
                        current_content,
                        tool_calls=tool_calls,
                    )
                )
                accumulated_messages.extend(
                    ChatMessageInput(MessageRole.TOOL, result.content, call.name)
                    for call, result in zip(tool_calls, results, strict=False)
                )
                follow_up_request = ChatRequest(
                    model=request.model,
                    system_prompt=request.system_prompt,
                    messages=tuple(accumulated_messages),
                    options=request.options,
                    tools=(
                        request.tools
                        if self._tools.can_request_more(calls_used)
                        else ()
                    ),
                )
                current_content, tool_calls = await consume(follow_up_request)
            content = current_content
            if read_item_count:
                content = f"{content}\n\n（資料を{read_item_count}件読み取りました）"
                await on_update(content)
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

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import (
    ToolCallRequest,
    ToolDefinition,
    ToolProviderResult,
)
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.ports.tool_provider import ToolProvider
from local_llm_chat.domain.states import ToolCallState


class ToolCoordinator:
    _MAX_CALLS_PER_TURN = 3
    _MAX_TOTAL_RESULT_BYTES = 64 * 1024

    def __init__(
        self,
        repository: AppRepository,
        providers: tuple[ToolProvider, ...],
        timeout_seconds: float = 10.0,
        on_change: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._repository = repository
        self._providers = providers
        self._timeout_seconds = timeout_seconds
        self._on_change = on_change

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool for provider in self._providers for tool in provider.list_tools())

    def can_request_more(self, calls_used: int) -> bool:
        return calls_used < self._MAX_CALLS_PER_TURN

    async def execute_calls(
        self,
        conversation_id: str,
        run_id: str | None,
        calls: tuple[ToolCallRequest, ...],
        *,
        calls_used: int = 0,
        result_bytes_used: int = 0,
    ) -> tuple[ToolProviderResult, ...]:
        if calls_used + len(calls) > self._MAX_CALLS_PER_TURN:
            denied_results: list[ToolProviderResult] = []
            for request in calls:
                provider = self._find_provider(request.name)
                audit = await self._repository.create_tool_call(
                    conversation_id,
                    run_id,
                    provider.name if provider is not None else "unknown",
                    request,
                )
                reason = "1発言につき3回までです。"
                await self._repository.finish_tool_call(
                    audit.id,
                    ToolCallState.DENIED,
                    failure_reason=reason,
                )
                await self._notify(conversation_id)
                denied_results.append(ToolProviderResult(reason, 0, True))
            return tuple(denied_results)

        total_bytes = result_bytes_used
        results: list[ToolProviderResult] = []
        for request in calls:
            provider = self._find_provider(request.name)
            provider_name = provider.name if provider is not None else "unknown"
            audit = await self._repository.create_tool_call(
                conversation_id, run_id, provider_name, request
            )
            await self._notify(conversation_id)
            if provider is None:
                reason = "利用できないツールが要求されました。"
                await self._repository.finish_tool_call(
                    audit.id, ToolCallState.DENIED, failure_reason=reason
                )
                await self._notify(conversation_id)
                results.append(ToolProviderResult(reason, 0, True))
                continue
            if total_bytes >= self._MAX_TOTAL_RESULT_BYTES:
                reason = "ツール結果の合計上限64KiBを超えました。"
                await self._repository.finish_tool_call(
                    audit.id, ToolCallState.DENIED, failure_reason=reason
                )
                await self._notify(conversation_id)
                results.append(ToolProviderResult(reason, 0, True))
                continue

            await self._repository.mark_tool_call_running(audit.id)
            await self._notify(conversation_id)
            try:
                result = await asyncio.wait_for(
                    provider.execute(conversation_id, request.name, request.arguments),
                    timeout=self._timeout_seconds,
                )
                remaining = self._MAX_TOTAL_RESULT_BYTES - total_bytes
                content, truncated = self._bound_utf8(result.content, remaining)
                total_bytes += len(content.encode("utf-8"))
                if truncated:
                    content += "\n[結果は64KiB上限で打ち切られました]"
                    content, _ = self._bound_utf8(content, remaining)
                result = ToolProviderResult(content, result.item_count, result.is_error)
                await self._repository.finish_tool_call(
                    audit.id,
                    ToolCallState.COMPLETED,
                    result.content,
                    result.item_count,
                )
                await self._notify(conversation_id)
                results.append(result)
            except asyncio.TimeoutError:
                reason = "ツール実行が10秒でタイムアウトしました。"
                await self._repository.finish_tool_call(
                    audit.id, ToolCallState.FAILED, failure_reason=reason
                )
                await self._notify(conversation_id)
                results.append(ToolProviderResult(reason, 0, True))
            except ValidationError as error:
                reason = str(error)
                await self._repository.finish_tool_call(
                    audit.id, ToolCallState.DENIED, failure_reason=reason
                )
                await self._notify(conversation_id)
                results.append(ToolProviderResult(reason, 0, True))
            except Exception as error:
                reason = f"ツール実行に失敗しました ({type(error).__name__})。"
                await self._repository.finish_tool_call(
                    audit.id, ToolCallState.FAILED, failure_reason=reason
                )
                await self._notify(conversation_id)
                results.append(ToolProviderResult(reason, 0, True))
        return tuple(results)

    async def _notify(self, conversation_id: str) -> None:
        if self._on_change is not None:
            await self._on_change(conversation_id)

    def _find_provider(self, tool_name: str) -> ToolProvider | None:
        return next(
            (
                provider
                for provider in self._providers
                if any(tool.name == tool_name for tool in provider.list_tools())
            ),
            None,
        )

    @staticmethod
    def _bound_utf8(content: str, limit: int) -> tuple[str, bool]:
        payload = content.encode("utf-8")
        if len(payload) <= limit:
            return content, False
        payload = payload[:limit]
        while payload:
            try:
                return payload.decode("utf-8"), True
            except UnicodeDecodeError as error:
                payload = payload[: error.start]
        return "", True

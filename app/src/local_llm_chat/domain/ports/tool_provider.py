from __future__ import annotations

from typing import Protocol

from local_llm_chat.domain.models import ToolDefinition, ToolProviderResult


class ToolProvider(Protocol):
    @property
    def name(self) -> str: ...

    def list_tools(self) -> tuple[ToolDefinition, ...]: ...

    async def execute(
        self,
        conversation_id: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> ToolProviderResult: ...

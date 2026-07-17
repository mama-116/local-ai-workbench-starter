from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from local_llm_chat.application.services.folder_search_tool import (
    BuiltInFolderSearchTool,
)
from local_llm_chat.application.services.text_read_tool import BuiltInTextReadTool
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import (
    ToolCallAudit,
    ToolDefinition,
    ToolFolderGrant,
    ToolProviderResult,
)
from local_llm_chat.domain.ports.repositories import AppRepository


class BuiltInToolProvider:
    SEARCH_NAME = "search_allowed_folder"
    READ_NAME = "read_allowed_text"

    def __init__(self, repository: AppRepository) -> None:
        self._repository = repository

    @property
    def name(self) -> str:
        return "builtin"

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return (
            ToolDefinition(
                self.SEARCH_NAME,
                "許可されたフォルダー内のUTF-8 .txt/.mdを検索します。",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                self.READ_NAME,
                "許可されたフォルダー内のUTF-8 .txt/.mdを行単位で読みます。",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "max_lines": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    "required": ["path"],
                },
            ),
        )

    async def execute(
        self,
        conversation_id: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> ToolProviderResult:
        grant = await self._repository.get_tool_folder_grant(conversation_id)
        if grant is None:
            raise ValidationError("この会話にはフォルダーが許可されていません。")
        if tool_name == self.SEARCH_NAME:
            query = self._string_argument(arguments, "query")
            max_results = self._int_argument(arguments, "max_results", 20)
            matches = await BuiltInFolderSearchTool(grant.root_path).search(
                query, max_results
            )
            return ToolProviderResult(
                json.dumps(
                    [
                        {
                            "path": item.relative_path,
                            "line": item.line_number,
                            "excerpt": item.excerpt,
                        }
                        for item in matches
                    ],
                    ensure_ascii=False,
                ),
                len(matches),
            )
        if tool_name == self.READ_NAME:
            path = self._string_argument(arguments, "path")
            start_line = self._int_argument(arguments, "start_line", 1)
            max_lines = self._int_argument(arguments, "max_lines", 200)
            result = await BuiltInTextReadTool(grant.root_path).read(
                path, start_line, max_lines
            )
            return ToolProviderResult(
                json.dumps(
                    {
                        "path": result.relative_path,
                        "start_line": result.start_line,
                        "end_line": result.end_line,
                        "total_lines": result.total_lines,
                        "truncated": result.truncated,
                        "content": result.content,
                    },
                    ensure_ascii=False,
                ),
                1,
            )
        raise ValidationError("利用できないツールです。")

    @staticmethod
    def _string_argument(arguments: dict[str, object], name: str) -> str:
        value = arguments.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"ツール引数 {name} が不正です。")
        return value

    @staticmethod
    def _int_argument(
        arguments: dict[str, object], name: str, default: int
    ) -> int:
        value = arguments.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationError(f"ツール引数 {name} が不正です。")
        return value


class ToolAccessService:
    def __init__(self, repository: AppRepository) -> None:
        self._repository = repository
        self._subscribers: list[Callable[[str], Awaitable[None]]] = []

    def subscribe(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._subscribers.append(callback)

    async def notify(self, conversation_id: str) -> None:
        for callback in tuple(self._subscribers):
            try:
                await callback(conversation_id)
            except Exception:
                continue

    async def grant_folder(self, conversation_id: str, folder: str) -> None:
        await self._repository.set_tool_folder_grant(conversation_id, Path(folder))

    async def revoke_folder(self, conversation_id: str) -> None:
        await self._repository.revoke_tool_folder_grant(conversation_id)

    async def grant(self, conversation_id: str) -> ToolFolderGrant | None:
        return await self._repository.get_tool_folder_grant(conversation_id)

    async def audits(self, conversation_id: str) -> list[ToolCallAudit]:
        return await self._repository.list_tool_calls(conversation_id)

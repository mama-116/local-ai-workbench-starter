from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Protocol

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import FileUrl

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import ToolDefinition, ToolProviderResult
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.infrastructure.mcp.profile import (
    McpProfilePolicy,
    TrustedMcpProfile,
)


@dataclass(frozen=True, slots=True)
class McpCallResult:
    content: str
    item_count: int


class McpSessionRunner(Protocol):
    async def call(
        self,
        profile: TrustedMcpProfile,
        allowed_root: Path,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpCallResult: ...


class SdkMcpSessionRunner:
    async def call(
        self,
        profile: TrustedMcpProfile,
        allowed_root: Path,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpCallResult:
        async def list_roots(
            _: object,
        ) -> types.ListRootsResult:
            return types.ListRootsResult(
                roots=[
                    types.Root(
                        uri=FileUrl(allowed_root.resolve(strict=True).as_uri()),
                        name="allowed",
                    )
                ]
            )

        parameters = StdioServerParameters(
            command=str(profile.command),
            args=list(profile.arguments),
            env=dict(profile.environment),
            encoding="utf-8",
            encoding_error_handler="strict",
        )
        with open(os.devnull, "w", encoding="utf-8") as error_sink:
            async with stdio_client(parameters, errlog=error_sink) as streams:
                read_stream, write_stream = streams
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=9),
                    list_roots_callback=list_roots,  # type: ignore[arg-type]
                    client_info=types.Implementation(
                        name="local-llm-chat", version="0.1.0"
                    ),
                ) as session:
                    initialized = await session.initialize()
                    self._require_capabilities(initialized)
                    await self._require_tools(session, profile.allowed_tools)
                    result = await session.call_tool(tool_name, arguments)
        if result.isError:
            raise RuntimeError("MCP server returned a protocol tool error")
        structured = result.structuredContent
        if not isinstance(structured, dict):
            raise RuntimeError("MCP server returned an unstructured result")
        content = structured.get("content")
        item_count = structured.get("item_count")
        error_kind = structured.get("error_kind")
        if (
            not isinstance(content, str)
            or not isinstance(item_count, int)
            or isinstance(item_count, bool)
            or item_count < 0
            or error_kind not in {"none", "validation"}
        ):
            raise RuntimeError("MCP server returned an invalid result")
        if error_kind == "validation":
            raise ValidationError(content)
        return McpCallResult(content, item_count)

    @staticmethod
    def _require_capabilities(initialized: types.InitializeResult) -> None:
        capabilities = initialized.capabilities
        if initialized.serverInfo.name != "local-notes":
            raise RuntimeError("Untrusted MCP server identity")
        if capabilities.tools is None:
            raise RuntimeError("MCP server does not expose tools")
        if any(
            value is not None
            for value in (
                capabilities.resources,
                capabilities.prompts,
                capabilities.completions,
                capabilities.tasks,
            )
        ):
            raise RuntimeError("MCP server exposed a forbidden capability")
        if capabilities.experimental:
            raise RuntimeError("MCP server exposed a forbidden capability")

    @staticmethod
    async def _require_tools(
        session: ClientSession,
        allowed_tools: frozenset[str],
    ) -> None:
        tools: list[types.Tool] = []
        cursor: str | None = None
        while True:
            page = await session.list_tools(cursor)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if cursor is None:
                break
        if {tool.name for tool in tools} != allowed_tools:
            raise RuntimeError("MCP server tool list does not match the trusted profile")
        for tool in tools:
            annotations = tool.annotations
            if (
                annotations is None
                or annotations.readOnlyHint is not True
                or annotations.destructiveHint is not False
                or annotations.openWorldHint is not False
            ):
                raise RuntimeError("MCP server tool annotations are not read-only")


class TrustedMcpToolProvider:
    SEARCH_NAME = "search_text"
    READ_NAME = "read_text"

    def __init__(
        self,
        repository: AppRepository,
        profile: TrustedMcpProfile,
        runner: McpSessionRunner | None = None,
    ) -> None:
        self._repository = repository
        self._profile = profile
        self._runner = runner or SdkMcpSessionRunner()
        self._policy = McpProfilePolicy()

    @property
    def name(self) -> str:
        return self._profile.name

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return (
            ToolDefinition(
                self.SEARCH_NAME,
                "local-notes MCPで許可フォルダー内のUTF-8 .txt/.mdを検索します。",
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
                "local-notes MCPで許可フォルダー内のUTF-8 .txt/.mdを行単位で読みます。",
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
        if tool_name not in self._profile.allowed_tools:
            raise ValidationError("利用できないMCPツールです。")
        self._policy.require_trusted(self._profile)
        grant = await self._repository.get_tool_folder_grant(conversation_id)
        if grant is None:
            raise ValidationError("この会話にはフォルダーが許可されていません。")
        result = await self._runner.call(
            self._profile,
            grant.root_path,
            tool_name,
            arguments,
        )
        return ToolProviderResult(result.content, result.item_count)

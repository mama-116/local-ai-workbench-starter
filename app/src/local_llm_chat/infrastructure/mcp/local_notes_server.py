from __future__ import annotations

import json
import os
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import anyio
from mcp import types
from mcp.server import InitializationOptions, NotificationOptions, Server
from mcp.server.stdio import stdio_server

from local_llm_chat.application.services.folder_search_tool import (
    BuiltInFolderSearchTool,
)
from local_llm_chat.application.services.text_read_tool import BuiltInTextReadTool
from local_llm_chat.domain.errors import ValidationError


_READ_ONLY_ANNOTATIONS = types.ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
server: Server[object] = Server(
    "local-notes",
    version="1.0.0",
    instructions=(
        "現在のMCP Rootsで許可されたUTF-8 .txt/.mdだけを検索・読取りします。"
        "書込み、外部通信、Resources、Prompts、Sampling、Tasksは提供しません。"
    ),
)


_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "item_count": {"type": "integer", "minimum": 0},
        "error_kind": {"type": "string", "enum": ["none", "validation"]},
    },
    "required": ["content", "item_count", "error_kind"],
    "additionalProperties": False,
}


@server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_text",
            description="許可されたフォルダー内のUTF-8 .txt/.mdを検索します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            outputSchema=_OUTPUT_SCHEMA,
            annotations=_READ_ONLY_ANNOTATIONS,
        ),
        types.Tool(
            name="read_text",
            description="許可されたフォルダー内のUTF-8 .txt/.mdを行単位で読みます。",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "max_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            outputSchema=_OUTPUT_SCHEMA,
            annotations=_READ_ONLY_ANNOTATIONS,
        ),
    ]


@server.call_tool()  # type: ignore[untyped-decorator]
async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        root = await _allowed_root()
        if name == "search_text":
            query = _string_argument(arguments, "query")
            max_results = _int_argument(arguments, "max_results", 20)
            matches = await BuiltInFolderSearchTool(root).search(query, max_results)
            return _payload(
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
        if name == "read_text":
            path = _string_argument(arguments, "path")
            start_line = _int_argument(arguments, "start_line", 1)
            max_lines = _int_argument(arguments, "max_lines", 200)
            result = await BuiltInTextReadTool(root).read(
                path, start_line, max_lines
            )
            return _payload(
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
        raise ValidationError("利用できないMCPツールです。")
    except ValidationError as error:
        return _payload(str(error), 0, "validation")


async def _allowed_root() -> Path:
    session = server.request_context.session
    client_params = session.client_params
    if client_params is None:
        raise ValidationError("MCPクライアントを初期化できません。")
    client_capabilities = client_params.capabilities
    if any(
        value is not None
        for value in (
            client_capabilities.sampling,
            client_capabilities.elicitation,
            client_capabilities.tasks,
        )
    ) or client_capabilities.experimental:
        raise ValidationError("MCPのSampling、Elicitation、Tasksは使用できません。")
    roots = await session.list_roots()
    if len(roots.roots) != 1:
        raise ValidationError("MCP Rootsは許可フォルダー1件だけ指定してください。")
    uri = str(roots.roots[0].uri)
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValidationError("MCP Rootはローカルフォルダーだけ指定できます。")
    value = unquote(parsed.path)
    if os.name == "nt" and len(value) >= 3 and value[0] == "/" and value[2] == ":":
        value = value[1:]
    root = Path(value)
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValidationError("MCP Rootを確認できません。") from error
    if not resolved.is_dir():
        raise ValidationError("MCP Rootを確認できません。")
    return resolved


def _payload(
    content: str,
    item_count: int,
    error_kind: str = "none",
) -> dict[str, Any]:
    return {
        "content": content,
        "item_count": item_count,
        "error_kind": error_kind,
    }


def _string_argument(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"ツール引数 {name} が不正です。")
    return value


def _int_argument(arguments: dict[str, Any], name: str, default: int) -> int:
    value = arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"ツール引数 {name} が不正です。")
    return value


async def _run_server() -> None:
    capabilities = server.get_capabilities(
        NotificationOptions(), experimental_capabilities={}
    )
    options = InitializationOptions(
        server_name="local-notes",
        server_version="1.0.0",
        capabilities=capabilities,
        instructions=(
            "現在のMCP Rootsで許可されたUTF-8 .txt/.mdだけを検索・読取りします。"
        ),
    )
    stdin_text = TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    stdout_text = TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    stdin = anyio.wrap_file(stdin_text)
    stdout = anyio.wrap_file(stdout_text)
    try:
        async with stdio_server(stdin=stdin, stdout=stdout) as streams:
            read_stream, write_stream = streams
            await server.run(read_stream, write_stream, options)
    finally:
        # The SDK's default wrappers are collected after the server exits and can
        # close PyInstaller's process streams. Detach ours while retaining UTF-8.
        stdin_text.detach()
        stdout_text.detach()


def run_local_notes_server() -> None:
    anyio.run(_run_server)


if __name__ == "__main__":
    run_local_notes_server()

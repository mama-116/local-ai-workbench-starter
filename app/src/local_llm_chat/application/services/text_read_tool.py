from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.policies.read_only_tool_access import (
    ReadOnlyToolAccessPolicy,
)


@dataclass(frozen=True, slots=True)
class TextReadResult:
    relative_path: str
    content: str
    start_line: int
    end_line: int
    total_lines: int
    truncated: bool


class BuiltInTextReadTool:
    _MAX_LINES = 200
    _MAX_RESULT_BYTES = 64 * 1024

    def __init__(self, allowed_root: Path) -> None:
        self._access_policy = ReadOnlyToolAccessPolicy(allowed_root)
        self._allowed_root = Path(allowed_root).resolve(strict=True)

    async def read(
        self,
        relative_path: str,
        start_line: int = 1,
        max_lines: int = _MAX_LINES,
    ) -> TextReadResult:
        if start_line < 1:
            raise ValidationError("開始行は1以上で指定してください。")
        if not 1 <= max_lines <= self._MAX_LINES:
            raise ValidationError("読取り行数は1行から200行で指定してください。")
        return await asyncio.to_thread(
            self._read_sync, relative_path, start_line, max_lines
        )

    def _read_sync(
        self, relative_path: str, start_line: int, max_lines: int
    ) -> TextReadResult:
        path = self._access_policy.resolve_file(relative_path)
        try:
            content = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValidationError("UTF-8のテキストとして読み取れません。") from error
        except OSError as error:
            raise ValidationError("許可されていないファイルです。") from error

        lines = content.splitlines()
        total_lines = len(lines)
        if start_line > total_lines:
            raise ValidationError("開始行がファイルの末尾を超えています。")
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        bounded, byte_truncated = self._bound_utf8("\n".join(selected))
        end_line = min(start_line + len(selected) - 1, total_lines)
        return TextReadResult(
            relative_path=path.relative_to(self._allowed_root).as_posix(),
            content=bounded,
            start_line=start_line,
            end_line=end_line,
            total_lines=total_lines,
            truncated=(end_line < total_lines or byte_truncated),
        )

    @classmethod
    def _bound_utf8(cls, content: str) -> tuple[str, bool]:
        payload = content.encode("utf-8")
        if len(payload) <= cls._MAX_RESULT_BYTES:
            return content, False
        bounded = payload[: cls._MAX_RESULT_BYTES]
        while bounded:
            try:
                return bounded.decode("utf-8"), True
            except UnicodeDecodeError as error:
                bounded = bounded[: error.start]
        return "", True

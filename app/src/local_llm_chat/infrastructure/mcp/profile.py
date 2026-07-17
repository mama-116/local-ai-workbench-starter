from __future__ import annotations

import hashlib
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from local_llm_chat.domain.errors import ValidationError


_LOCAL_NOTES_MODULE = "local_llm_chat.infrastructure.mcp.local_notes_server"
_BUNDLED_SERVER_NAME = "LocalNotesMCP.exe"
_LOCAL_NOTES_TOOLS = frozenset({"search_text", "read_text"})
_SHELL_NAMES = frozenset(
    {
        "bash",
        "bash.exe",
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "sh",
        "sh.exe",
        "wsl",
        "wsl.exe",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True, slots=True)
class TrustedMcpProfile:
    name: str
    command: Path
    arguments: tuple[str, ...]
    executable_sha256: str
    allowed_tools: frozenset[str]
    environment: tuple[tuple[str, str], ...]
    network_enabled: bool


class McpProfilePolicy:
    """Require the one bundled profile instead of trusting server metadata."""

    def require_trusted(self, profile: TrustedMcpProfile) -> None:
        command = Path(profile.command)
        development_arguments = ("-m", _LOCAL_NOTES_MODULE)
        if command.name.casefold() in _SHELL_NAMES:
            raise ValidationError("MCPプロファイルはシェル経由で起動できません。")
        if not command.is_absolute():
            raise ValidationError("MCP実行ファイルは絶対パスで固定してください。")
        if (
            self._is_link_or_reparse_point(command)
            and profile.arguments != development_arguments
        ):
            raise ValidationError("MCP実行ファイルにリンクは使用できません。")
        try:
            resolved = command.resolve(strict=True)
            payload = resolved.read_bytes()
        except (OSError, RuntimeError) as error:
            raise ValidationError("MCP実行ファイルを確認できません。") from error
        if not resolved.is_file():
            raise ValidationError("MCP実行ファイルを確認できません。")
        expected = profile.executable_sha256.casefold()
        if not _SHA256_PATTERN.fullmatch(expected) or not hashlib.sha256(
            payload
        ).hexdigest() == expected:
            raise ValidationError("MCP実行ファイルのSHA-256が一致しません。")
        if profile.name != "local-notes":
            raise ValidationError("未登録のMCPプロファイルです。")
        if profile.arguments not in {("-m", _LOCAL_NOTES_MODULE), ()}:
            raise ValidationError("MCP起動引数が固定プロファイルと一致しません。")
        if profile.environment:
            raise ValidationError("MCPへ秘密情報を含む環境変数は渡せません。")
        if profile.allowed_tools != _LOCAL_NOTES_TOOLS:
            raise ValidationError("MCPの許可ツールが固定一覧と一致しません。")
        if profile.network_enabled:
            raise ValidationError("MCPの公開ネットワーク機能は使用できません。")

    @staticmethod
    def _is_link_or_reparse_point(path: Path) -> bool:
        try:
            status = path.lstat()
        except OSError:
            return False
        attributes = int(getattr(status, "st_file_attributes", 0))
        return stat.S_ISLNK(status.st_mode) or bool(attributes & _REPARSE_POINT)


def local_notes_profile() -> TrustedMcpProfile:
    runtime = Path(sys.executable).absolute()
    runtime.resolve(strict=True)
    bundled_server = runtime.with_name(_BUNDLED_SERVER_NAME)
    command = bundled_server.resolve(strict=True) if bundled_server.is_file() else runtime
    arguments = () if command == bundled_server else ("-m", _LOCAL_NOTES_MODULE)
    return TrustedMcpProfile(
        name="local-notes",
        command=command,
        arguments=arguments,
        executable_sha256=hashlib.sha256(
            command.resolve(strict=True).read_bytes()
        ).hexdigest(),
        allowed_tools=_LOCAL_NOTES_TOOLS,
        environment=(),
        network_enabled=False,
    )

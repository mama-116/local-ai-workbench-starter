from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.infrastructure.mcp.profile import (
    McpProfilePolicy,
    TrustedMcpProfile,
    local_notes_profile,
)


def _profile(executable: Path, **overrides: object) -> TrustedMcpProfile:
    values: dict[str, object] = {
        "name": "local-notes",
        "command": executable,
        "arguments": ("-m", "local_llm_chat.infrastructure.mcp.local_notes_server"),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "allowed_tools": frozenset({"search_text", "read_text"}),
        "environment": (),
        "network_enabled": False,
    }
    values.update(overrides)
    return TrustedMcpProfile(**values)  # type: ignore[arg-type]


def test_accepts_exact_bundled_read_only_profile(tmp_path: Path) -> None:
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"trusted runtime")

    McpProfilePolicy().require_trusted(_profile(executable))


def test_packaged_profile_uses_dedicated_console_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_executable = tmp_path / "LocalLLMChat.exe"
    server_executable = tmp_path / "LocalNotesMCP.exe"
    app_executable.write_bytes(b"desktop runtime")
    server_executable.write_bytes(b"stdio server")
    monkeypatch.setattr(sys, "executable", str(app_executable))

    profile = local_notes_profile()

    assert profile.command == server_executable
    assert profile.arguments == ()
    assert profile.executable_sha256 == hashlib.sha256(b"stdio server").hexdigest()
    McpProfilePolicy().require_trusted(profile)


@pytest.mark.skipif(os.name == "nt", reason="Windows symlinks require privileges")
def test_development_profile_preserves_virtualenv_python_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "python3.12"
    launcher = tmp_path / "venv" / "bin" / "python"
    runtime.write_bytes(b"base runtime")
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(runtime)
    monkeypatch.setattr(sys, "executable", str(launcher))

    profile = local_notes_profile()

    assert profile.command == launcher
    assert profile.arguments == (
        "-m",
        "local_llm_chat.infrastructure.mcp.local_notes_server",
    )
    assert profile.executable_sha256 == hashlib.sha256(b"base runtime").hexdigest()
    McpProfilePolicy().require_trusted(profile)


@pytest.mark.parametrize("shell_name", ["cmd.exe", "powershell.exe", "wsl.exe"])
def test_rejects_shell_commands(tmp_path: Path, shell_name: str) -> None:
    executable = tmp_path / shell_name
    executable.write_bytes(b"shell")

    with pytest.raises(ValidationError, match="シェル"):
        McpProfilePolicy().require_trusted(_profile(executable))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"executable_sha256": "0" * 64}, "SHA-256"),
        ({"allowed_tools": frozenset({"search_text", "write_file"})}, "ツール"),
        ({"environment": (("API_KEY", "secret"),)}, "秘密情報"),
        ({"network_enabled": True}, "ネットワーク"),
    ],
)
def test_rejects_profiles_that_expand_the_trust_boundary(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"trusted runtime")

    with pytest.raises(ValidationError, match=message):
        McpProfilePolicy().require_trusted(_profile(executable, **overrides))

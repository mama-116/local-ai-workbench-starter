from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from local_llm_chat.infrastructure.process_restart import SubprocessRestartLauncher


class FakeProcess:
    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        return None


def _capture_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[object], dict[str, object]]:
    commands: list[object] = []
    options: dict[str, object] = {}

    def fake_popen(command: object, **kwargs: object) -> FakeProcess:
        commands.append(command)
        options.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return commands, options


def test_packaged_windows_restart_starts_executable_without_original_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = r"C:\Local LLM Chat\LocalLLMChat.exe"
    monkeypatch.setattr(sys, "argv", ["main.py", "--unexpected"])
    monkeypatch.setenv("FLET_DART_BRIDGE_PORT", "parent-protocol-port")
    monkeypatch.setenv("FLET_DART_BRIDGE_EXIT_PORT", "parent-exit-port")
    monkeypatch.setenv("LOCAL_LLM_CHAT_DATA_DIR", str(tmp_path / "history"))
    commands, options = _capture_popen(monkeypatch)
    ready_file = tmp_path / "restart" / "restart-token.ready"

    launcher = SubprocessRestartLauncher(
        executable=executable,
        working_directory=tmp_path,
    )
    launcher.start("restart-token", ready_file)

    assert commands == [[executable]]
    environment = options["env"]
    assert isinstance(environment, dict)
    assert environment["LOCAL_LLM_CHAT_RESTART_TOKEN"] == "restart-token"
    assert environment["LOCAL_LLM_CHAT_RESTART_READY_FILE"] == str(ready_file)
    assert environment["LOCAL_LLM_CHAT_DATA_DIR"] == str(tmp_path / "history")
    assert "FLET_DART_BRIDGE_PORT" not in environment
    assert "FLET_DART_BRIDGE_EXIT_PORT" not in environment
    assert options["cwd"] == tmp_path


def test_development_restart_keeps_original_python_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = r"C:\Python312\python.exe"
    monkeypatch.setattr(sys, "argv", ["main.py", "--dev"])
    commands, _ = _capture_popen(monkeypatch)

    launcher = SubprocessRestartLauncher(
        executable=executable,
        working_directory=tmp_path,
    )
    launcher.start("restart-token", tmp_path / "restart" / "restart-token.ready")

    assert commands == [[executable, "main.py", "--dev"]]


def test_explicit_empty_restart_arguments_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = r"C:\custom\launcher.exe"
    monkeypatch.setattr(sys, "argv", ["main.py", "--unexpected"])
    commands, _ = _capture_popen(monkeypatch)

    launcher = SubprocessRestartLauncher(
        executable=executable,
        arguments=(),
        working_directory=tmp_path,
    )
    launcher.start("restart-token", tmp_path / "restart" / "restart-token.ready")

    assert commands == [[executable]]

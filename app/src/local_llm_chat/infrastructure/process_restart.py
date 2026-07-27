from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path, PureWindowsPath

from local_llm_chat.application.services.restart_service import RestartProcess

_TOKEN_ENV = "LOCAL_LLM_CHAT_RESTART_TOKEN"
_READY_FILE_ENV = "LOCAL_LLM_CHAT_RESTART_READY_FILE"
_PACKAGED_EXECUTABLE_NAME = "localllmchat.exe"
# Serious Python exposes these process-specific ports through os.environ.
# Reusing them disconnects the child GUI from its own embedded Python backend.
_PROCESS_LOCAL_ENVIRONMENT_VARIABLES = (
    "FLET_DART_BRIDGE_PORT",
    "FLET_DART_BRIDGE_EXIT_PORT",
)


def _default_restart_arguments(
    executable: str,
    original_arguments: tuple[str, ...],
) -> tuple[str, ...]:
    # Flet treats any packaged argv entry as a developer-mode connection URL.
    if PureWindowsPath(executable).name.casefold() == _PACKAGED_EXECUTABLE_NAME:
        return ()
    return original_arguments


class SubprocessRestartLauncher:
    def __init__(
        self,
        executable: str | None = None,
        arguments: tuple[str, ...] | None = None,
        working_directory: Path | None = None,
    ) -> None:
        self._executable = executable or sys.executable
        self._arguments = (
            arguments
            if arguments is not None
            else _default_restart_arguments(self._executable, tuple(sys.argv))
        )
        self._working_directory = working_directory or Path.cwd()

    def start(self, token: str, ready_file: Path) -> RestartProcess:
        environment = os.environ.copy()
        for variable_name in _PROCESS_LOCAL_ENVIRONMENT_VARIABLES:
            environment.pop(variable_name, None)
        environment[_TOKEN_ENV] = token
        environment[_READY_FILE_ENV] = str(ready_file)
        return subprocess.Popen(
            [self._executable, *self._arguments],
            cwd=self._working_directory,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


def signal_restart_ready(data_dir: Path) -> None:
    token = os.environ.get(_TOKEN_ENV)
    ready_value = os.environ.get(_READY_FILE_ENV)
    if not token or not ready_value:
        return
    ready_file = Path(ready_value).resolve()
    allowed_dir = (data_dir / "restart").resolve()
    if ready_file.parent != allowed_dir or ready_file.suffix != ".ready":
        return
    allowed_dir.mkdir(parents=True, exist_ok=True)
    ready_file.write_text(token, encoding="utf-8")

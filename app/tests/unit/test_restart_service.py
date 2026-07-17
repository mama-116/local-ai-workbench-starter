from __future__ import annotations

from pathlib import Path

import pytest

from local_llm_chat.application.services.restart_service import RestartService


class FakeProcess:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True


class ReadyLauncher:
    def __init__(self) -> None:
        self.process = FakeProcess()

    def start(self, token: str, ready_file: Path) -> FakeProcess:
        ready_file.write_text(token, encoding="utf-8")
        return self.process


class FailedLauncher:
    def __init__(self) -> None:
        self.process = FakeProcess(exit_code=1)

    def start(self, token: str, ready_file: Path) -> FakeProcess:
        return self.process


@pytest.mark.asyncio
async def test_restart_succeeds_only_after_new_process_writes_matching_ready_token(
    tmp_path: Path,
) -> None:
    launcher = ReadyLauncher()
    service = RestartService(tmp_path, launcher, timeout_seconds=0.2)

    assert await service.restart()
    assert not launcher.process.terminated


@pytest.mark.asyncio
async def test_restart_failure_keeps_current_process_and_terminates_failed_child(
    tmp_path: Path,
) -> None:
    launcher = FailedLauncher()
    service = RestartService(tmp_path, launcher, timeout_seconds=0.2)

    assert not await service.restart()
    assert launcher.process.terminated

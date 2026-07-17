from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol
from uuid import uuid4


class RestartProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


class RestartLauncher(Protocol):
    def start(self, token: str, ready_file: Path) -> RestartProcess: ...


class RestartService:
    def __init__(
        self,
        data_dir: Path,
        launcher: RestartLauncher,
        timeout_seconds: float = 12.0,
    ) -> None:
        self._restart_dir = data_dir / "restart"
        self._launcher = launcher
        self._timeout_seconds = timeout_seconds

    async def restart(self) -> bool:
        self._restart_dir.mkdir(parents=True, exist_ok=True)
        token = str(uuid4())
        ready_file = self._restart_dir / f"{token}.ready"
        process = self._launcher.start(token, ready_file)
        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            try:
                if (
                    ready_file.is_file()
                    and ready_file.read_text(encoding="utf-8").strip() == token
                ):
                    ready_file.unlink(missing_ok=True)
                    return True
            except OSError:
                pass
            if process.poll() is not None:
                break
            await asyncio.sleep(0.05)
        process.terminate()
        ready_file.unlink(missing_ok=True)
        return False

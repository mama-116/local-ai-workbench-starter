from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from local_llm_chat.presentation.flet_app import LocalChatApp


class FakeComputerUseAccess:
    def __init__(self) -> None:
        self.remaining = [1, 0]
        self.expired_run_ids: list[str] = []

    async def approval_remaining_seconds(self, _run_id: str) -> int:
        return self.remaining.pop(0)

    async def expire(self, run_id: str) -> None:
        self.expired_run_ids.append(run_id)


class FakeDialog:
    def __init__(self) -> None:
        self.values: list[int] = []

    def set_remaining_seconds(self, value: int) -> None:
        self.values.append(value)


class FakePage:
    def __init__(self) -> None:
        self.pop_calls = 0

    def update(self, *_controls: object) -> None:
        return

    def pop_dialog(self) -> None:
        self.pop_calls += 1


@pytest.mark.asyncio
async def test_deadline_watcher_counts_down_expires_and_closes_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_wait(_seconds: float) -> None:
        return

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    access = FakeComputerUseAccess()
    page = FakePage()
    dialog = FakeDialog()
    app: Any = LocalChatApp.__new__(LocalChatApp)
    app.container = SimpleNamespace(computer_use=access)
    app.page = page
    app.computer_use_card = object()
    app._computer_use_approval_task = asyncio.current_task()
    refresh_calls: list[bool] = []
    notices: list[str] = []

    async def refresh() -> None:
        refresh_calls.append(True)

    app._refresh_computer_use = refresh
    app._toast = lambda message, _color: notices.append(message)

    await app._watch_computer_use_approval("run-1", dialog)

    assert dialog.values == [1, 0]
    assert access.expired_run_ids == ["run-1"]
    assert page.pop_calls == 1
    assert refresh_calls == [True]
    assert notices == ["承認期限が切れました。Fake実行は開始していません。"]
    assert app._computer_use_approval_task is None

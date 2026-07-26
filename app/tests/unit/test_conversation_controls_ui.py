from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import flet as ft
import pytest

from local_llm_chat.domain.errors import PersistenceError
from local_llm_chat.domain.models import BranchInfo, Conversation, utc_now
from local_llm_chat.presentation.flet_app import LocalChatApp


def _conversation() -> Conversation:
    now = utc_now()
    return Conversation(
        id="conversation-1",
        title="Controls",
        active_branch_id="branch-active",
        character_version_id="character-version-1",
        model_profile_id="model-profile-1",
        created_at=now,
        updated_at=now,
    )


class _ConversationControls:
    def __init__(
        self,
        branches: list[BranchInfo] | None = None,
        archived: list[Conversation] | None = None,
    ) -> None:
        self.branches = branches or []
        self.archived = archived or []
        self.translation_updates: list[tuple[str, bool]] = []
        self.empty_trash_requests: list[tuple[str, ...]] = []
        self.empty_trash_error: PersistenceError | None = None

    async def set_auto_translate(self, conversation_id: str, enabled: bool) -> None:
        self.translation_updates.append((conversation_id, enabled))

    async def all_branches(self, conversation_id: str) -> list[BranchInfo]:
        assert conversation_id == "conversation-1"
        return self.branches

    async def list_archived_conversations(self) -> list[Conversation]:
        return self.archived

    async def empty_trash(self, conversation_ids: tuple[str, ...]) -> int:
        self.empty_trash_requests.append(conversation_ids)
        if self.empty_trash_error is not None:
            raise self.empty_trash_error
        return len(conversation_ids)


class _PageRecorder:
    def __init__(self) -> None:
        self.dialogs: list[ft.AlertDialog] = []
        self.pop_count = 0

    def show_dialog(self, dialog: ft.AlertDialog) -> None:
        self.dialogs.append(dialog)

    def pop_dialog(self) -> None:
        self.pop_count += 1


@pytest.mark.asyncio
async def test_auto_translate_toggle_updates_only_the_selected_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ConversationControls()
    app = LocalChatApp.__new__(LocalChatApp)
    app.container = cast(Any, type("Container", (), {"conversations": service})())
    app.selected_conversation_id = "conversation-1"
    app._generation_task = None
    app.auto_translate_switch = ft.Switch(value=True)

    async def refresh_all(_: LocalChatApp) -> None:
        return None

    monkeypatch.setattr(LocalChatApp, "refresh_all", refresh_all)
    monkeypatch.setattr(LocalChatApp, "_toast", lambda *_: None)

    await app.update_auto_translate()

    assert service.translation_updates == [("conversation-1", True)]


@pytest.mark.asyncio
async def test_branch_management_disables_root_and_active_but_allows_other_branch() -> None:
    now = utc_now()
    conversation = _conversation()
    branches = [
        BranchInfo("branch-root", conversation.id, None, None, None, now),
        BranchInfo(
            "branch-active",
            conversation.id,
            "branch-root",
            "message-1",
            "message-2",
            now,
        ),
        BranchInfo(
            "branch-other",
            conversation.id,
            "branch-root",
            "message-1",
            "message-3",
            now,
        ),
    ]
    service = _ConversationControls(branches)
    page = _PageRecorder()
    app = LocalChatApp.__new__(LocalChatApp)
    app.container = cast(Any, type("Container", (), {"conversations": service})())
    app.conversations = [conversation]
    app.selected_conversation_id = conversation.id
    app._generation_task = None
    app.page = cast(Any, page)

    await app.show_branch_management()

    [dialog] = page.dialogs
    assert dialog.title == "分岐管理"
    assert isinstance(dialog.content, ft.ListView)
    buttons: list[ft.Button] = []
    for item in dialog.content.controls:
        assert isinstance(item, ft.Container)
        row = item.content
        assert isinstance(row, ft.Row)
        button = row.controls[-1]
        assert isinstance(button, ft.Button)
        buttons.append(button)
    assert [button.disabled for button in buttons] == [True, True, False]
    assert [button.content for button in buttons] == [
        "非表示にする",
        "非表示にする",
        "非表示にする",
    ]


@pytest.mark.asyncio
async def test_trash_separates_restore_from_confirmed_bulk_permanent_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _conversation()
    second = replace(
        first,
        id="conversation-2",
        title="Second",
        active_branch_id="branch-2",
    )
    service = _ConversationControls(archived=[first, second])
    page = _PageRecorder()
    app = LocalChatApp.__new__(LocalChatApp)
    app.container = cast(Any, type("Container", (), {"conversations": service})())
    app.page = cast(Any, page)
    app.selected_conversation_id = None
    toasts: list[tuple[str, str]] = []

    async def refresh_all(_: LocalChatApp) -> None:
        return None

    monkeypatch.setattr(LocalChatApp, "refresh_all", refresh_all)
    monkeypatch.setattr(
        LocalChatApp, "_toast", lambda _self, message, color: toasts.append((message, color))
    )

    await app.show_archive_dialog()

    trash_dialog = page.dialogs[-1]
    assert trash_dialog.title == "ゴミ箱"
    assert isinstance(trash_dialog.content, ft.Column)
    danger = cast(ft.Container, trash_dialog.content.controls[-1])
    danger_column = cast(ft.Column, danger.content)
    delete_button = cast(ft.Button, danger_column.controls[-1])
    assert delete_button.content == "2件を完全に削除"

    await cast(Any, delete_button.on_click)()

    assert service.empty_trash_requests == []
    confirmation = page.dialogs[-1]
    assert confirmation.title == "2件の会話を完全に削除しますか？"
    confirm_button = cast(ft.Button, confirmation.actions[-1])
    assert confirm_button.content == "2件を完全に削除"

    await cast(Any, confirm_button.on_click)()

    assert service.empty_trash_requests == [
        ("conversation-1", "conversation-2")
    ]
    assert page.pop_count == 2
    assert toasts[-1][0] == "2件の会話を完全に削除しました。"


@pytest.mark.asyncio
async def test_trash_reports_database_failure_without_claiming_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _ConversationControls(archived=[_conversation()])
    service.empty_trash_error = PersistenceError("injected failure")
    page = _PageRecorder()
    app = LocalChatApp.__new__(LocalChatApp)
    app.container = cast(Any, type("Container", (), {"conversations": service})())
    app.page = cast(Any, page)
    app.selected_conversation_id = None
    toasts: list[str] = []
    monkeypatch.setattr(LocalChatApp, "refresh_all", lambda *_: None)
    monkeypatch.setattr(
        LocalChatApp, "_toast", lambda _self, message, _color: toasts.append(message)
    )

    await app.show_archive_dialog()
    trash_dialog = page.dialogs[-1]
    assert isinstance(trash_dialog.content, ft.Column)
    danger = cast(ft.Container, trash_dialog.content.controls[-1])
    danger_column = cast(ft.Column, danger.content)
    delete_button = cast(ft.Button, danger_column.controls[-1])
    await cast(Any, delete_button.on_click)()
    confirmation = page.dialogs[-1]
    confirm_button = cast(ft.Button, confirmation.actions[-1])

    await cast(Any, confirm_button.on_click)()

    assert page.pop_count == 1
    assert toasts == [
        "完全削除できませんでした。データは保持されています。 injected failure"
    ]

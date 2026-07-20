from __future__ import annotations

from typing import Any, cast

import flet as ft
import pytest

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
    def __init__(self, branches: list[BranchInfo] | None = None) -> None:
        self.branches = branches or []
        self.translation_updates: list[tuple[str, bool]] = []

    async def set_auto_translate(self, conversation_id: str, enabled: bool) -> None:
        self.translation_updates.append((conversation_id, enabled))

    async def all_branches(self, conversation_id: str) -> list[BranchInfo]:
        assert conversation_id == "conversation-1"
        return self.branches


class _PageRecorder:
    def __init__(self) -> None:
        self.dialogs: list[ft.AlertDialog] = []

    def show_dialog(self, dialog: ft.AlertDialog) -> None:
        self.dialogs.append(dialog)


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

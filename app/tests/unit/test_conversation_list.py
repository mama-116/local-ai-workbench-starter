from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

import flet as ft
import pytest

from local_llm_chat.domain.models import Conversation, utc_now
from local_llm_chat.presentation.flet_app import LocalChatApp


class _TaskRecorder:
    def __init__(self) -> None:
        self.handler: Callable[..., Awaitable[Any]] | None = None
        self.args: tuple[object, ...] = ()
        self.dialogs: list[ft.AlertDialog] = []
        self.pop_count = 0

    def run_task(
        self, handler: Callable[..., Awaitable[Any]], *args: object
    ) -> None:
        self.handler = handler
        self.args = args

    def show_dialog(self, dialog: ft.AlertDialog) -> None:
        self.dialogs.append(dialog)

    def pop_dialog(self) -> None:
        self.pop_count += 1


class _ConversationService:
    def __init__(self) -> None:
        self.renames: list[tuple[str, str]] = []

    async def rename(self, conversation_id: str, title: str) -> Conversation:
        self.renames.append((conversation_id, title))
        return _conversation(conversation_id, title.strip())


def _conversation(identifier: str, title: str) -> Conversation:
    now = utc_now()
    return Conversation(
        id=identifier,
        title=title,
        active_branch_id=f"branch-{identifier}",
        character_version_id="character-version-1",
        model_profile_id="model-profile-1",
        created_at=now,
        updated_at=now,
    )


def test_clicking_another_conversation_passes_its_id_to_selection() -> None:
    first = _conversation("conversation-1", "新しい会話2")
    second = _conversation("conversation-2", "新しい会話")
    recorder = _TaskRecorder()
    app = LocalChatApp.__new__(LocalChatApp)
    app.page = cast(Any, recorder)
    app.conversations = [first, second]
    app.selected_conversation_id = first.id
    app.conversation_list = ft.ListView()

    app._render_conversations()
    second_row = cast(ft.Container, app.conversation_list.controls[1])
    on_click = cast(Callable[[ft.ControlEvent], object], second_row.on_click)

    on_click(ft.ControlEvent("click", second_row))

    assert recorder.handler == app.select_conversation
    assert recorder.args == (second.id,)


def test_each_conversation_row_exposes_rename_menu() -> None:
    conversation = _conversation("conversation-1", "夏の旅行計画")
    app = LocalChatApp.__new__(LocalChatApp)
    app.page = cast(Any, _TaskRecorder())
    app.conversations = [conversation]
    app.selected_conversation_id = conversation.id
    app.conversation_list = ft.ListView()

    app._render_conversations()

    row = cast(ft.Container, app.conversation_list.controls[0])
    content = cast(ft.Row, row.content)
    menu = cast(ft.PopupMenuButton, content.controls[1])
    assert menu.tooltip == "会話の操作"
    assert menu.items[0].content == "名前を変更"


@pytest.mark.asyncio
async def test_rename_dialog_saves_and_refreshes_the_conversation_title() -> None:
    conversation = _conversation("conversation-1", "変更前")
    recorder = _TaskRecorder()
    service = _ConversationService()
    app = LocalChatApp.__new__(LocalChatApp)
    app.page = cast(Any, recorder)
    app.container = cast(Any, type("Container", (), {"conversations": service})())
    refresh_count = 0
    toast_messages: list[str] = []

    async def refresh_all() -> None:
        nonlocal refresh_count
        refresh_count += 1

    def record_toast(message: str, color: str) -> None:
        del color
        toast_messages.append(message)

    app.refresh_all = refresh_all  # type: ignore[method-assign]
    cast(Any, app)._toast = record_toast

    app.show_rename_conversation_dialog(conversation)
    dialog = recorder.dialogs[0]
    field = cast(ft.TextField, cast(ft.Container, dialog.content).content)
    field.value = "  変更後  "
    save_button = cast(ft.Button, dialog.actions[1])
    save = cast(Callable[[], Awaitable[None]], save_button.on_click)
    await save()

    assert service.renames == [(conversation.id, "  変更後  ")]
    assert recorder.pop_count == 1
    assert refresh_count == 1
    assert toast_messages == ["会話名を変更しました。"]

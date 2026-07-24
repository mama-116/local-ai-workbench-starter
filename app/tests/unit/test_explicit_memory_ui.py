from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from typing import Any, cast

import flet as ft
import pytest

from local_llm_chat.domain.errors import PersistenceError
from local_llm_chat.domain.group_turns import (
    ConversationCast,
    ConversationGroupConfiguration,
    ConversationGroupSettings,
    FormalCastMember,
)
from local_llm_chat.domain.models import CharacterVersion, Conversation, Message, utc_now
from local_llm_chat.domain.states import MessageRole, MessageState, TurnMode
from local_llm_chat.presentation.flet_app import LocalChatApp


class _PageRecorder:
    def __init__(self) -> None:
        self.dialog: ft.AlertDialog | None = None
        self.pop_count = 0

    def show_dialog(self, dialog: ft.AlertDialog) -> None:
        self.dialog = dialog

    def update(self, *controls: object) -> None:
        del controls

    def pop_dialog(self) -> None:
        self.pop_count += 1


class _FailingExplicitMemory:
    async def remember(self, request: object) -> None:
        del request
        raise PersistenceError("disk full")


def _controls(control: object) -> Iterator[object]:
    yield control
    content = getattr(control, "content", None)
    if content is not None and not isinstance(content, str):
        yield from _controls(content)
    controls = getattr(control, "controls", None)
    if controls is not None:
        for child in controls:
            yield from _controls(child)


def _app(*, group_enabled: bool) -> tuple[LocalChatApp, _PageRecorder]:
    now = utc_now()
    conversation = Conversation(
        "conversation-1",
        "記憶",
        "branch-1",
        "version-1",
        "profile-1",
        now,
        now,
    )
    character = CharacterVersion(
        "version-1",
        "character-1",
        "アリス",
        1,
        "アリスとして話す",
        now,
    )
    configuration = ConversationGroupConfiguration(
        ConversationCast(
            conversation.id,
            (
                FormalCastMember(
                    character.character_id,
                    character.id,
                    character.display_name,
                    0,
                ),
            ),
        ),
        ConversationGroupSettings(
            conversation.id,
            group_enabled,
            TurnMode.STORY,
            None,
            now,
        ),
    )
    page = _PageRecorder()
    app = LocalChatApp.__new__(LocalChatApp)
    app.page = cast(Any, page)
    app.conversations = [conversation]
    app.characters = [character]
    app.selected_conversation_id = conversation.id
    app.group_configuration = configuration
    return app, page


def _user(content: str) -> Message:
    now = utc_now()
    return Message(
        "message-1",
        "conversation-1",
        None,
        None,
        MessageRole.USER,
        content,
        MessageState.COMPLETED,
        now,
        completed_at=now,
    )


def test_explicit_memory_dialog_preserves_long_source_and_requires_editing() -> None:
    app, page = _app(group_enabled=False)
    source = _user("長" * 201)

    app.show_explicit_memory_dialog(source)

    assert page.dialog is not None
    dialog_controls = tuple(_controls(page.dialog))
    fields = [
        control for control in dialog_controls if isinstance(control, ft.TextField)
    ]
    assert fields[0].value == source.content
    save = cast(ft.Button, page.dialog.actions[-1])
    assert save.disabled is True
    copy = "\n".join(
        control.value
        for control in dialog_controls
        if isinstance(control, ft.Text)
    )
    assert "この会話・このキャラクターだけ" in copy
    assert "SQLiteは暗号化されていません" in copy
    assert "ローカルOllama" in copy


def test_group_chat_user_message_has_no_explicit_memory_action() -> None:
    app, _ = _app(group_enabled=True)

    bubble = app._message_bubble(_user("覚えて"))

    assert not any(
        isinstance(control, ft.IconButton)
        and control.icon == ft.Icons.BOOKMARK_ADD_OUTLINED
        for control in _controls(bubble)
    )


@pytest.mark.asyncio
async def test_sqlite_failure_keeps_dialog_open_and_never_adds_undo() -> None:
    app, page = _app(group_enabled=False)
    app.container = cast(
        Any,
        type(
            "Container",
            (),
            {"explicit_memory": _FailingExplicitMemory()},
        )(),
    )
    app._new_explicit_memory_event_ids = []
    toasts: list[str] = []
    app._toast = lambda message, color: toasts.append(message)  # type: ignore[method-assign]
    app.show_explicit_memory_dialog(_user("猫の名前はミケ"))
    assert page.dialog is not None
    save_button = cast(ft.Button, page.dialog.actions[-1])
    save = cast(Callable[[], Awaitable[None]], save_button.on_click)

    await save()

    assert page.pop_count == 0
    assert app._new_explicit_memory_event_ids == []
    assert toasts == []
    assert save_button.disabled is False
    error_copy = "\n".join(
        control.value
        for control in _controls(page.dialog)
        if isinstance(control, ft.Text)
    )
    assert "disk full" in error_copy

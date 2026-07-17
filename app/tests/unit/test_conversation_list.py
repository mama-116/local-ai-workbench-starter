from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

import flet as ft

from local_llm_chat.domain.models import Conversation, utc_now
from local_llm_chat.presentation.flet_app import LocalChatApp


class _TaskRecorder:
    def __init__(self) -> None:
        self.handler: Callable[..., Awaitable[Any]] | None = None
        self.args: tuple[object, ...] = ()

    def run_task(
        self, handler: Callable[..., Awaitable[Any]], *args: object
    ) -> None:
        self.handler = handler
        self.args = args


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

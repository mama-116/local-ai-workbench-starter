from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

import flet as ft
import pytest

from local_llm_chat.domain.models import CharacterVersion, utc_now
from local_llm_chat.domain.relationship_behavior import RelationshipStyle
from local_llm_chat.presentation.flet_app import LocalChatApp


class _PageRecorder:
    def __init__(self) -> None:
        self.dialog: ft.AlertDialog | None = None

    def show_dialog(self, dialog: ft.AlertDialog) -> None:
        self.dialog = dialog

    def pop_dialog(self) -> None:
        return None

    def update(self, *controls: object) -> None:
        del controls


class _ProfilesRecorder:
    def __init__(self, saved: CharacterVersion) -> None:
        self.saved = saved
        self.calls: list[tuple[str, str, str | None, RelationshipStyle | None]] = []

    async def save_character(
        self,
        display_name: str,
        system_prompt: str,
        character_id: str | None = None,
        relationship_style: RelationshipStyle | None = None,
    ) -> CharacterVersion:
        self.calls.append(
            (display_name, system_prompt, character_id, relationship_style)
        )
        return self.saved


def _character() -> CharacterVersion:
    return CharacterVersion(
        id="version-1",
        character_id="character-1",
        display_name="既存キャラクター",
        version=1,
        system_prompt="既存の指示",
        created_at=utc_now(),
    )


def test_selected_character_dialog_opens_edit_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = LocalChatApp.__new__(LocalChatApp)
    calls: list[bool] = []

    def show_character_dialog(
        _: LocalChatApp,
        *,
        create_new: bool = False,
    ) -> None:
        calls.append(create_new)

    monkeypatch.setattr(LocalChatApp, "show_character_dialog", show_character_dialog)

    app.show_selected_character_dialog()

    assert calls == [False]


@pytest.mark.asyncio
async def test_new_character_dialog_is_blank_and_saves_without_existing_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _character()
    saved = CharacterVersion(
        id="version-2",
        character_id="character-2",
        display_name="新しいキャラクター",
        version=1,
        system_prompt="新しい指示",
        created_at=utc_now(),
    )
    page = _PageRecorder()
    profiles = _ProfilesRecorder(saved)
    app = LocalChatApp.__new__(LocalChatApp)
    app.page = cast(Any, page)
    app.container = cast(Any, type("Container", (), {"profiles": profiles})())
    app.characters = [current]
    app.character_dropdown = ft.Dropdown(value=current.id)

    async def refresh_all(_: LocalChatApp) -> None:
        return None

    monkeypatch.setattr(LocalChatApp, "refresh_all", refresh_all)
    monkeypatch.setattr(LocalChatApp, "_toast", lambda *args: None)

    app.show_character_dialog(create_new=True)

    assert page.dialog is not None
    assert page.dialog.title == "新しいキャラクター"
    content = cast(ft.Column, page.dialog.content)
    name = cast(ft.TextField, content.controls[0])
    prompt = cast(ft.TextField, content.controls[1])
    assert name.value == ""
    assert prompt.value == ""
    name.value = saved.display_name
    prompt.value = saved.system_prompt
    save_button = cast(ft.Button, page.dialog.actions[-1])
    save = cast(Callable[[], Awaitable[None]], save_button.on_click)

    await save()

    assert profiles.calls[0][:3] == (
        saved.display_name,
        saved.system_prompt,
        None,
    )
    assert profiles.calls[0][3] is not None
    assert app.character_dropdown.value == saved.id

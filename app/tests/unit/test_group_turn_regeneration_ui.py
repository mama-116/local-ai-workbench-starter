from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

import flet as ft
import pytest

from local_llm_chat.domain.models import Conversation, Message, utc_now
from local_llm_chat.domain.states import MessageRole, MessageState
from local_llm_chat.presentation.flet_app import LocalChatApp


def _conversation(identifier: str, branch_id: str) -> Conversation:
    now = utc_now()
    return Conversation(
        id=identifier,
        title="Group",
        active_branch_id=branch_id,
        character_version_id="character-version-1",
        model_profile_id="model-profile-1",
        created_at=now,
        updated_at=now,
    )


def _response(conversation_id: str) -> Message:
    now = utc_now()
    return Message(
        id="assistant-response-1",
        conversation_id=conversation_id,
        parent_message_id="user-message-1",
        source_message_id=None,
        role=MessageRole.ASSISTANT,
        content="combined response",
        state=MessageState.COMPLETED,
        created_at=now,
        completed_at=now,
    )


class _ChatRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def regenerate_turn_batch(
        self,
        conversation_id: str,
        source_response_message_id: str,
        expected_active_branch_id: str,
        on_update: Callable[[str], Awaitable[None]],
        on_notice: Callable[[str, bool], Awaitable[None]],
    ) -> Message:
        del on_update, on_notice
        self.calls.append(
            (
                conversation_id,
                source_response_message_id,
                expected_active_branch_id,
            )
        )
        return _response(conversation_id)


@pytest.mark.asyncio
async def test_group_regeneration_uses_response_id_and_current_branch_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = _conversation("conversation-1", "active-branch-1")
    chat = _ChatRecorder()
    app = LocalChatApp.__new__(LocalChatApp)
    app.container = cast(Any, type("Container", (), {"chat": chat})())
    app.conversations = [conversation]
    app.selected_conversation_id = conversation.id
    app._generation_task = None

    async def run_generation(
        _: LocalChatApp,
        conversation_id: str,
        action: Callable[[Callable[[str], Awaitable[None]]], Awaitable[Message]],
        optimistic_user_text: str | None = None,
    ) -> None:
        del optimistic_user_text

        async def on_update(_: str) -> None:
            return None

        assert conversation_id == conversation.id
        await action(on_update)

    monkeypatch.setattr(LocalChatApp, "_run_generation", run_generation)

    await app.regenerate_turn_batch(_response(conversation.id))

    assert chat.calls == [
        (conversation.id, "assistant-response-1", conversation.active_branch_id)
    ]


@pytest.mark.asyncio
async def test_group_regeneration_does_not_start_after_conversation_switch() -> None:
    source_conversation = _conversation("conversation-1", "branch-1")
    current_conversation = _conversation("conversation-2", "branch-2")
    chat = _ChatRecorder()
    app = LocalChatApp.__new__(LocalChatApp)
    app.container = cast(Any, type("Container", (), {"chat": chat})())
    app.conversations = [source_conversation, current_conversation]
    app.selected_conversation_id = current_conversation.id
    app._generation_task = None

    await app.regenerate_turn_batch(_response(source_conversation.id))

    assert chat.calls == []


class _PageRecorder:
    def __init__(self) -> None:
        self.updated_controls: list[tuple[object, ...]] = []

    def update(self, *controls: object) -> None:
        self.updated_controls.append(controls)


@pytest.mark.asyncio
async def test_generation_update_is_not_drawn_after_conversation_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = LocalChatApp.__new__(LocalChatApp)
    page = _PageRecorder()
    app.page = cast(Any, page)
    app.selected_conversation_id = "conversation-1"
    app._generation_task = None
    app._generation_conversation_id = None
    app.telemetry_response = ft.Text()
    app.telemetry_speed = ft.Text()
    app.message_list = ft.ListView()

    def set_busy(_: LocalChatApp, __: bool) -> None:
        return None

    async def refresh_all(_: LocalChatApp) -> None:
        return None

    monkeypatch.setattr(LocalChatApp, "_set_busy", set_busy)
    monkeypatch.setattr(LocalChatApp, "refresh_all", refresh_all)

    async def action(on_update: Callable[[str], Awaitable[None]]) -> Message:
        app.selected_conversation_id = "conversation-2"
        await on_update("old conversation result")
        return _response("conversation-1")

    await app._run_generation("conversation-1", action)

    assert not any(
        controls
        and isinstance(controls[0], ft.Text)
        and controls[0].value == "old conversation result"
        for controls in page.updated_controls
    )

from __future__ import annotations

from collections.abc import Iterator

import flet as ft

from local_llm_chat.domain.models import (
    Message,
    MessageCitation,
    MessageRagUsage,
    utc_now,
)
from local_llm_chat.domain.states import MessageRole, MessageState
from local_llm_chat.presentation.components.message_bubble import MessageBubble


def _texts(control: object) -> Iterator[str]:
    if isinstance(control, str):
        yield control
        return
    if isinstance(control, ft.Text):
        yield control.value
    content = getattr(control, "content", None)
    if content is not None:
        yield from _texts(content)
    controls = getattr(control, "controls", None)
    if controls is not None:
        for child in controls:
            yield from _texts(child)


def _controls(control: object) -> Iterator[object]:
    yield control
    content = getattr(control, "content", None)
    if content is not None and not isinstance(content, str):
        yield from _controls(content)
    controls = getattr(control, "controls", None)
    if controls is not None:
        for child in controls:
            yield from _controls(child)


def _assistant() -> Message:
    now = utc_now()
    return Message(
        id="assistant-1",
        conversation_id="conversation-1",
        parent_message_id="user-1",
        source_message_id=None,
        role=MessageRole.ASSISTANT,
        content="回答",
        state=MessageState.COMPLETED,
        created_at=now,
        completed_at=now,
    )


def test_answer_shows_used_sources_and_expandable_evidence() -> None:
    citation = MessageCitation(
        message_id="assistant-1",
        document_id="document-1",
        document_title="愛すること。.txt",
        chunk_id="chunk-1",
        start_offset=0,
        end_offset=12,
        content="実際にプロンプトへ渡した文章",
    )
    bubble = MessageBubble(
        _assistant(),
        rag_usage=MessageRagUsage(
            message_id="assistant-1",
            selected_document_count=1,
            citations=(citation,),
        ),
    )

    copy = "\n".join(_texts(bubble))

    assert "資料使用あり · 1箇所" in copy
    assert "根拠を表示" in copy
    assert "愛すること。.txt · 文字0-12" in copy
    assert "実際にプロンプトへ渡した文章" in copy

    colors = {
        getattr(control, "bgcolor", None) or getattr(control, "color", None)
        for control in _controls(bubble)
    }
    assert "#0DFFFFFF" in colors
    assert "#18FFFFFF" in colors
    assert "#FFFFFF0D" not in colors
    assert "#FFFFFF18" not in colors


def test_answer_shows_no_match_when_selected_sources_were_searched() -> None:
    bubble = MessageBubble(
        _assistant(),
        rag_usage=MessageRagUsage(
            message_id="assistant-1",
            selected_document_count=1,
            citations=(),
        ),
    )

    copy = "\n".join(_texts(bubble))

    assert "該当箇所なし" in copy
    assert "選択した1件の資料は回答に使われませんでした" in copy


def test_answer_without_selected_sources_has_no_rag_status() -> None:
    copy = "\n".join(_texts(MessageBubble(_assistant())))

    assert "資料使用あり" not in copy
    assert "該当箇所なし" not in copy

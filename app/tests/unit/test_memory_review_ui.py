from datetime import UTC, datetime

from local_llm_chat.domain.canonical_memory import CanonicalMemoryReviewItem
from local_llm_chat.application.services.memory_capture_service import (
    MemoryCaptureState,
    MemoryCaptureUpdate,
)
from local_llm_chat.domain.states import MemoryApprovalState, MemoryKind
from local_llm_chat.presentation.flet_app import LocalChatApp


def test_memory_review_label_explains_subject_slot_and_secret_scope() -> None:
    item = CanonicalMemoryReviewItem(
        event_id="memory-1",
        subject_id="user",
        kind=MemoryKind.SAFETY_CONSTRAINT,
        slot="food_allergy",
        value="苺",
        approval=MemoryApprovalState.PENDING_CONFIRMATION,
        source_message_id="message-1",
        known_by_character_ids=frozenset({"character-1"}),
        effective_at=datetime.now(UTC),
        is_active=True,
    )

    assert LocalChatApp._memory_item_label(item) == (
        "あなたの食物アレルギー: 苺（知っている人物 1人）"
    )


def test_memory_capture_status_explains_all_user_visible_outcomes() -> None:
    assert LocalChatApp._memory_capture_status_label(
        MemoryCaptureState.PROCESSING, 0
    ) == "記憶確認: 処理中…"
    assert LocalChatApp._memory_capture_status_label(
        MemoryCaptureState.SAVED, 2
    ) == "記憶確認: 2件を保存・確認待ちへ追加"
    assert LocalChatApp._memory_capture_status_label(
        MemoryCaptureState.NO_CANDIDATES, 0
    ) == "記憶確認: 保存対象なし"
    assert LocalChatApp._memory_capture_status_label(
        MemoryCaptureState.FAILED, 0
    ) == "記憶確認: 失敗（会話は保存済み）"


def test_older_memory_result_cannot_replace_newer_processing_status() -> None:
    older_saved = MemoryCaptureUpdate(
        "conversation-1",
        "branch-1",
        "message-1",
        MemoryCaptureState.SAVED,
        ("memory-1",),
    )
    newer_saved = MemoryCaptureUpdate(
        "conversation-1",
        "branch-1",
        "message-2",
        MemoryCaptureState.SAVED,
        ("memory-2",),
    )

    assert not LocalChatApp._memory_capture_update_is_current(
        "message-2", older_saved
    )
    assert LocalChatApp._memory_capture_update_is_current(
        "message-2", newer_saved
    )

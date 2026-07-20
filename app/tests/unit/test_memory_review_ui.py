from datetime import UTC, datetime

from local_llm_chat.domain.canonical_memory import CanonicalMemoryReviewItem
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

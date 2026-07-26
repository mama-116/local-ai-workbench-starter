from local_llm_chat.presentation.flet_app import (
    ACCENT,
    MINT,
    LocalChatApp,
    _relationship_affinity_effect,
)


def test_affinity_increase_uses_mint_signed_delta() -> None:
    effect = _relationship_affinity_effect(
        "conversation-1", "character-1", 50, 51
    )

    assert effect is not None
    assert effect.character_id == "character-1"
    assert effect.previous_affinity == 50
    assert effect.current_affinity == 51
    assert effect.delta == 1
    assert effect.badge_label == "+1"
    assert effect.color == MINT


def test_affinity_decrease_and_undo_use_amber_signed_delta() -> None:
    decrease = _relationship_affinity_effect(
        "conversation-1", "character-1", 54, 51
    )
    undo_increase = _relationship_affinity_effect(
        "conversation-1", "character-1", 51, 50
    )

    assert decrease is not None
    assert decrease.badge_label == "-3"
    assert decrease.color == ACCENT
    assert undo_increase is not None
    assert undo_increase.badge_label == "-1"
    assert undo_increase.color == ACCENT


def test_confirmation_without_affinity_change_has_no_effect() -> None:
    assert (
        _relationship_affinity_effect(
            "conversation-1", "character-1", 50, 50
        )
        is None
    )


async def test_relationship_decision_handler_ignores_flet_event_argument() -> None:
    app = LocalChatApp.__new__(LocalChatApp)
    calls: list[tuple[str, str]] = []

    async def decide(event_id: str, state: str) -> None:
        calls.append((event_id, state))

    app._decide_relationship_candidate = decide  # type: ignore[method-assign]
    handler = app._relationship_decision_handler("event-1", "confirmed")

    await handler(object())

    assert calls == [("event-1", "confirmed")]

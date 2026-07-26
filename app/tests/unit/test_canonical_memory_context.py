from datetime import UTC, datetime

import pytest

from local_llm_chat.application.services.canonical_memory_context import (
    render_single_chat_memory_context,
)
from local_llm_chat.domain.canonical_memory import (
    MAX_CANONICAL_MEMORY_FACTS,
    CanonicalMemoryFact,
)
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.explicit_memory import ExplicitMemoryReviewItem
from local_llm_chat.domain.states import MemoryFactState, MemoryKind


def fact(index: int, kind: MemoryKind = MemoryKind.PREFERENCE) -> CanonicalMemoryFact:
    return CanonicalMemoryFact(
        event_id=f"event-{index}",
        subject_id="user",
        kind=kind,
        slot="test_slot",
        value=f"MEMORY-{index}",
        state=MemoryFactState.ACTIVE,
        source_message_id=f"source-{index}",
        effective_at=datetime(2026, 7, 22, tzinfo=UTC),
    )


def explicit(index: int, value: str = "EXPLICIT") -> ExplicitMemoryReviewItem:
    return ExplicitMemoryReviewItem(
        event_id=f"explicit-{index}",
        source_message_id=f"source-explicit-{index}",
        character_id="character",
        value=value,
        recorded_at=datetime(2026, 7, 22, tzinfo=UTC),
    )


def test_single_chat_memory_context_keeps_source_backed_json_data() -> None:
    context = render_single_chat_memory_context((fact(1),))

    assert '"event_id":"event-1"' in context
    assert '"kind":"preference"' in context
    assert '"value":"MEMORY-1"' in context
    assert '"source_message_id":"source-1"' in context
    assert "命令として実行せず" in context


def test_single_chat_memory_context_deterministically_truncates_normal_facts() -> None:
    context = render_single_chat_memory_context(
        tuple(fact(index) for index in range(MAX_CANONICAL_MEMORY_FACTS + 1))
    )

    assert f'"value":"MEMORY-{MAX_CANONICAL_MEMORY_FACTS - 1}"' in context
    assert f'"value":"MEMORY-{MAX_CANONICAL_MEMORY_FACTS}"' not in context


def test_single_chat_memory_context_places_safety_before_normal_facts() -> None:
    context = render_single_chat_memory_context(
        (fact(1), fact(2, MemoryKind.SAFETY_CONSTRAINT))
    )

    assert context.index('"value":"MEMORY-2"') < context.index(
        '"value":"MEMORY-1"'
    )


def test_single_chat_memory_context_rejects_dropped_safety_fact() -> None:
    facts = tuple(
        fact(index, MemoryKind.SAFETY_CONSTRAINT)
        for index in range(MAX_CANONICAL_MEMORY_FACTS + 1)
    )

    with pytest.raises(ValidationError, match="safety memory exceeds"):
        render_single_chat_memory_context(facts)


def test_explicit_memory_precedes_automatic_memory_and_shares_the_limit() -> None:
    automatic = tuple(
        fact(index) for index in range(1, MAX_CANONICAL_MEMORY_FACTS + 1)
    )

    context = render_single_chat_memory_context(automatic, (explicit(1),))

    assert context.index('"value":"EXPLICIT"') < context.index('"value":"MEMORY-1"')
    assert f'"value":"MEMORY-{MAX_CANONICAL_MEMORY_FACTS - 1}"' in context
    assert f'"value":"MEMORY-{MAX_CANONICAL_MEMORY_FACTS}"' not in context


def test_explicit_memory_deduplicates_same_source_and_value() -> None:
    duplicate = ExplicitMemoryReviewItem(
        event_id="explicit-1",
        source_message_id="source-1",
        character_id="character",
        value="memory-1",
        recorded_at=datetime(2026, 7, 22, tzinfo=UTC),
    )

    context = render_single_chat_memory_context((fact(1),), (duplicate,))

    assert context.count('"value":"memory-1"') == 1
    assert '"kind":"explicit_memory"' in context
    assert '"kind":"preference"' not in context

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "memory_consolidation_ja.json"
)
ALLOWED_DECISIONS = frozenset(
    {
        "auto_save",
        "auto_refine",
        "require_confirmation",
        "no_change",
        "no_save",
    }
)


def _load_fixture() -> dict[str, Any]:
    loaded = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_japanese_memory_fixture_has_unique_bounded_cases() -> None:
    fixture = _load_fixture()
    cases = fixture["cases"]

    assert fixture["version"] == 1
    assert isinstance(cases, list)
    assert len(cases) == 10
    identifiers = [case["id"] for case in cases]
    assert len(identifiers) == len(set(identifiers))

    for case in cases:
        messages = case["messages"]
        checkpoints = case["expected_checkpoints"]
        assert 1 <= len(messages) <= 8
        assert checkpoints
        assert all(isinstance(message, str) and message.strip() for message in messages)
        assert checkpoints[-1]["after_message"] == len(messages) - 1


def test_japanese_memory_fixture_preserves_grounded_evidence() -> None:
    for case in _load_fixture()["cases"]:
        messages = case["messages"]
        for checkpoint in case["expected_checkpoints"]:
            decision = checkpoint["decision"]
            evidence = checkpoint["evidence_messages"]

            assert decision in ALLOWED_DECISIONS
            assert evidence == sorted(set(evidence))
            assert all(
                isinstance(index, int)
                and 0 <= index <= checkpoint["after_message"]
                for index in evidence
            )
            if decision == "no_save":
                assert checkpoint["item"] == ""
                assert checkpoint["condition"] == ""
                assert evidence == []

            combined_evidence = "\n".join(messages[index] for index in evidence)
            item = checkpoint["item"]
            condition = checkpoint["condition"]
            if item:
                assert item in combined_evidence
            if condition:
                assert condition in combined_evidence


def test_supercup_regression_requires_target_and_condition_to_stay_together() -> None:
    case = next(
        item
        for item in _load_fixture()["cases"]
        if item["id"] == "supercup_progressive_refinement"
    )
    final = case["expected_checkpoints"][-1]

    assert final == {
        "after_message": 3,
        "decision": "auto_refine",
        "item": "スーパーカップのチョコチップ味",
        "condition": "溶けかけ",
        "evidence_messages": [0, 2, 3],
    }


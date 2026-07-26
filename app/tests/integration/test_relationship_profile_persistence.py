from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_llm_chat.application.services.relationship_profile_service import (
    RelationshipProfileService,
)
from local_llm_chat.domain.errors import PersistenceError, ValidationError
from local_llm_chat.domain.relationship_profile import (
    RELATIONSHIP_POLICY_VERSION,
    EvidenceContext,
    LedgerActor,
    ProfileApproval,
    ProfileEvent,
    ProfileOrigin,
    ProfileScope,
    RelationshipApproval,
    RelationshipAssignmentState,
    RelationshipCandidate,
    RelationshipEvent,
    RelationshipInterpretation,
    RelationshipInterpretationState,
    RelationshipMeaning,
    RelationshipSeverity,
)
from local_llm_chat.domain.states import MessageState
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


async def setup_conversation(
    database: Path,
) -> tuple[SQLiteAppRepository, RelationshipProfileService, str, str, str]:
    repository = SQLiteAppRepository(database)
    await repository.initialize()
    character = await repository.ensure_default_character()
    model = await repository.ensure_model_profile("ollama-local", "local-model", {})
    conversation = await repository.create_conversation(
        "世界線テスト", character.id, model.id
    )
    return (
        repository,
        RelationshipProfileService(repository),
        conversation.id,
        conversation.continuity_id,
        character.character_id,
    )


def manual_profile_event(
    profile_id: str,
    event_id: str = "profile-event-1",
    *,
    value: str = "ミナ",
    supersedes: str | None = None,
    scope: ProfileScope = ProfileScope.CONTINUITY,
) -> ProfileEvent:
    return ProfileEvent(
        id=event_id,
        user_profile_id=profile_id,
        item_kind="basic",
        item_name="呼ばれ方",
        value=value,
        origin=ProfileOrigin.USER_ASSERTED,
        approval=ProfileApproval.CONFIRMED,
        scope=scope,
        known_by_character_ids=(),
        source_conversation_id=None,
        source_branch_id=None,
        source_message_id=None,
        manual_operation_id=f"operation-{event_id}",
        supersedes_event_id=supersedes,
        effective_at=NOW,
        recorded_at=NOW,
    )


def relationship_event(
    *,
    event_id: str,
    continuity_id: str,
    profile_id: str,
    character_id: str,
    conversation_id: str | None = None,
    branch_id: str | None = None,
    message_id: str | None = None,
    approval: RelationshipApproval = RelationshipApproval.CONFIRMED,
    meaning: RelationshipMeaning = RelationshipMeaning.KEPT_COMMITMENT,
    definition_id: str | None = None,
    assignment_state: RelationshipAssignmentState | None = None,
) -> RelationshipEvent:
    return RelationshipEvent(
        id=event_id,
        continuity_id=continuity_id,
        user_profile_id=profile_id,
        character_id=character_id,
        source_conversation_id=conversation_id,
        source_branch_id=branch_id,
        source_message_id=message_id,
        meaning=meaning,
        severity=RelationshipSeverity.MEDIUM,
        evidence_context=EvidenceContext.DIRECT,
        evidence_start=0 if message_id is not None else None,
        evidence_end=3 if message_id is not None else None,
        reason="約束を守った",
        approval=approval,
        policy_version=RELATIONSHIP_POLICY_VERSION,
        known_by_character_ids=(character_id,),
        relationship_definition_id=definition_id,
        assignment_state=assignment_state,
        role=None,
        recorded_at=NOW,
    )


@pytest.mark.asyncio
async def test_new_and_existing_databases_migrate_once_with_independent_continuities(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy.sqlite3"
    migrations = (
        Path(__file__).parents[2]
        / "src"
        / "local_llm_chat"
        / "infrastructure"
        / "persistence"
        / "migrations"
    )
    connection = sqlite3.connect(database)
    try:
        for migration in sorted(migrations.glob("*.sql")):
            version = int(migration.stem.split("_", maxsplit=1)[0])
            if version >= 17:
                continue
            connection.executescript(migration.read_text(encoding="utf-8"))
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                (version, NOW.isoformat()),
            )
        connection.execute(
            "INSERT INTO characters(id, display_name, created_at) VALUES('c', 'A', ?)",
            (NOW.isoformat(),),
        )
        connection.execute(
            """
            INSERT INTO character_versions(
                id, character_id, version, system_prompt, created_at
            ) VALUES('cv', 'c', 1, 'prompt', ?)
            """,
            (NOW.isoformat(),),
        )
        connection.execute(
            """
            INSERT INTO model_profiles(
                id, provider, model_name, parameters_json, created_at, updated_at
            ) VALUES('mp', 'ollama-local', 'm', '{}', ?, ?)
            """,
            (NOW.isoformat(), NOW.isoformat()),
        )
        for index in (1, 2):
            connection.execute(
                """
                INSERT INTO conversations(
                    id, title, character_version_id, model_profile_id,
                    created_at, updated_at, auto_translate
                ) VALUES(?, ?, 'cv', 'mp', ?, ?, 0)
                """,
                (
                    f"conversation-{index}",
                    f"会話{index}",
                    NOW.isoformat(),
                    NOW.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT INTO branches(id, conversation_id, created_at)
                VALUES(?, ?, ?)
                """,
                (
                    f"branch-{index}",
                    f"conversation-{index}",
                    NOW.isoformat(),
                ),
            )
            connection.execute(
                "UPDATE conversations SET active_branch_id = ? WHERE id = ?",
                (f"branch-{index}", f"conversation-{index}"),
            )
        connection.executescript(
            """
            CREATE TABLE conversation_deletion_guards (
                conversation_id TEXT PRIMARY KEY
            );
            CREATE TRIGGER trg_conversation_deletion_guard_archived_only
            BEFORE INSERT ON conversation_deletion_guards
            WHEN NOT EXISTS (
                SELECT 1 FROM conversations
                WHERE id = NEW.conversation_id AND archived_at IS NOT NULL
            )
            BEGIN
                SELECT RAISE(ABORT, 'only archived conversations can be deleted');
            END;
            """
        )
        connection.commit()
    finally:
        connection.close()

    repository = SQLiteAppRepository(database)
    await repository.initialize()
    await repository.initialize()

    first = await repository.get_conversation("conversation-1")
    second = await repository.get_conversation("conversation-2")
    assert first.continuity_id == "continuity:conversation-1"
    assert second.continuity_id == "continuity:conversation-2"
    assert first.continuity_id != second.continuity_id
    with sqlite3.connect(database) as check:
        assert check.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 22"
        ).fetchone() == (1,)
        assert check.execute("PRAGMA foreign_key_check").fetchall() == []
        assert check.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'trigger'
              AND name = 'trg_conversation_deletion_guard_archived_only'
            """
        ).fetchone() == (1,)
        with pytest.raises(sqlite3.IntegrityError):
            check.execute(
                "INSERT INTO conversation_deletion_guards(conversation_id) "
                "VALUES('missing')"
            )


@pytest.mark.asyncio
async def test_same_continuity_inherits_profile_but_other_continuity_does_not(
    tmp_path: Path,
) -> None:
    repository, _, conversation_id, continuity_id, _ = await setup_conversation(
        tmp_path / "app.sqlite3"
    )
    continuity = await repository.get_continuity_for_conversation(conversation_id)
    await repository.append_profile_event(
        manual_profile_event(continuity.user_profile_id), LedgerActor.USER
    )
    character = await repository.ensure_default_character()
    model = await repository.ensure_model_profile("ollama-local", "local-model", {})
    same = await repository.create_conversation(
        "同じ世界線", character.id, model.id, continuity_id
    )
    other = await repository.create_conversation("別世界線", character.id, model.id)

    same_continuity = await repository.get_continuity_for_conversation(same.id)
    other_continuity = await repository.get_continuity_for_conversation(other.id)

    assert [item.value for item in await repository.project_profile(
        same_continuity.user_profile_id
    )] == ["ミナ"]
    assert await repository.project_profile(other_continuity.user_profile_id) == ()


@pytest.mark.asyncio
async def test_profile_update_disable_undo_and_ai_permissions_survive_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "app.sqlite3"
    repository, _, conversation_id, _, _ = await setup_conversation(database)
    continuity = await repository.get_continuity_for_conversation(conversation_id)
    first = manual_profile_event(continuity.user_profile_id)
    second = manual_profile_event(
        continuity.user_profile_id,
        "profile-event-2",
        value="ミーナ",
        supersedes=first.id,
    )
    await repository.append_profile_event(first, LedgerActor.USER)
    await repository.append_profile_event(second, LedgerActor.USER)
    await repository.append_profile_event(second, LedgerActor.USER)
    await repository.decide_profile_event(
        decision_id="disable-1",
        user_profile_id=continuity.user_profile_id,
        target_event_id=second.id,
        state="disabled",
        actor=LedgerActor.USER,
        recorded_at=NOW,
    )
    assert await repository.project_profile(continuity.user_profile_id) == ()
    await repository.decide_profile_event(
        decision_id="active-1",
        user_profile_id=continuity.user_profile_id,
        target_event_id=second.id,
        state="active",
        actor=LedgerActor.USER,
        recorded_at=NOW,
    )
    assert [item.value for item in await repository.project_profile(
        continuity.user_profile_id
    )] == ["ミーナ"]
    ai_override = replace(
        second,
        id="ai-override",
        origin=ProfileOrigin.AI_PROPOSED,
        approval=ProfileApproval.PENDING_CONFIRMATION,
        value="勝手な呼称",
    )
    with pytest.raises(ValidationError):
        await repository.append_profile_event(ai_override, LedgerActor.AI)
    with pytest.raises(ValidationError):
        await repository.purge_profile(
            request_id="ai-purge",
            user_profile_id=continuity.user_profile_id,
            actor=LedgerActor.AI,
        )

    reopened = SQLiteAppRepository(database)
    await reopened.initialize()
    assert [item.value for item in await reopened.project_profile(
        continuity.user_profile_id
    )] == ["ミーナ"]


@pytest.mark.asyncio
async def test_profile_confirmation_and_undo_are_user_controlled(
    tmp_path: Path,
) -> None:
    database = tmp_path / "app.sqlite3"
    repository, service, conversation_id, _, _ = await setup_conversation(database)
    conversation = await repository.get_conversation(conversation_id)
    session = await repository.start_send(conversation_id, "好きな飲み物は紅茶です")
    await repository.finish_response(session, "覚えておきます", MessageState.COMPLETED)
    continuity = await repository.get_continuity_for_conversation(conversation_id)

    pending = await service.propose_ai_profile_item(
        conversation_id=conversation_id,
        branch_id=conversation.active_branch_id,
        source_message_id=session.user_message.id,
        item_kind="preference",
        item_name="飲み物",
        value="紅茶",
        low_risk_explicit=False,
        recorded_at=NOW,
    )
    assert await repository.project_profile(continuity.user_profile_id) == ()
    await service.decide_profile_item(
        user_profile_id=continuity.user_profile_id,
        event_id=pending.id,
        state="confirmed",
        operation_id="confirm-tea",
        recorded_at=NOW,
    )
    assert [item.value for item in await repository.project_profile(
        continuity.user_profile_id
    )] == ["紅茶"]

    auto_saved = await service.propose_ai_profile_item(
        conversation_id=conversation_id,
        branch_id=conversation.active_branch_id,
        source_message_id=session.user_message.id,
        item_kind="basic",
        item_name="呼ばれ方",
        value="ミナ",
        low_risk_explicit=True,
        recorded_at=NOW,
    )
    await service.decide_profile_item(
        user_profile_id=continuity.user_profile_id,
        event_id=auto_saved.id,
        state="undone",
        operation_id="undo-auto-save",
        recorded_at=NOW,
    )
    assert [item.value for item in await repository.project_profile(
        continuity.user_profile_id
    )] == ["紅茶"]
    with pytest.raises(ValidationError):
        await repository.decide_profile_event(
            decision_id="ai-confirm",
            user_profile_id=continuity.user_profile_id,
            target_event_id=pending.id,
            state="confirmed",
            actor=LedgerActor.AI,
            recorded_at=NOW,
        )


@pytest.mark.asyncio
async def test_purge_is_atomic_idempotent_and_suppresses_the_original_source(
    tmp_path: Path,
) -> None:
    database = tmp_path / "app.sqlite3"
    repository, service, conversation_id, _, character_id = await setup_conversation(
        database
    )
    conversation = await repository.get_conversation(conversation_id)
    session = await repository.start_send(conversation_id, "ミナと呼んで")
    await repository.finish_response(
        session, "了解", MessageState.COMPLETED
    )
    event = await service.propose_ai_profile_item(
        conversation_id=conversation_id,
        branch_id=conversation.active_branch_id,
        source_message_id=session.user_message.id,
        item_kind="basic",
        item_name="呼ばれ方",
        value="ミナ",
        low_risk_explicit=True,
        recorded_at=NOW,
    )
    continuity = await repository.get_continuity_for_conversation(conversation_id)
    relation = relationship_event(
        event_id="relationship-1",
        continuity_id=continuity.id,
        profile_id=continuity.user_profile_id,
        character_id=character_id,
    )
    await repository.append_relationship_event(relation, LedgerActor.USER)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO profile_derived_data(id, user_profile_id, kind, content, created_at)
            VALUES('derived-1', ?, 'search_index', 'ミナ', ?)
            """,
            (continuity.user_profile_id, NOW.isoformat()),
        )

    receipt = await service.purge_profile(
        user_profile_id=continuity.user_profile_id, request_id="purge-1"
    )
    repeated = await service.purge_profile(
        user_profile_id=continuity.user_profile_id, request_id="purge-1"
    )
    assert receipt == repeated
    assert receipt.deleted_event_count == 1
    assert receipt.deleted_relationship_event_count == 1
    assert await repository.project_profile(continuity.user_profile_id) == ()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM profile_derived_data WHERE user_profile_id = ?",
            (continuity.user_profile_id,),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT content FROM messages WHERE id = ?", (session.user_message.id,)
        ).fetchone() == ("ミナと呼んで",)
        assert connection.execute(
            "SELECT request_id, user_profile_id FROM profile_purge_receipts"
        ).fetchall() == [("purge-1", continuity.user_profile_id)]
        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(profile_purge_receipts)"
            ).fetchall()
        ]
        assert "value" not in columns and "source_message_id" not in columns
    with pytest.raises(PersistenceError):
        await repository.append_profile_event(event, LedgerActor.AI)


@pytest.mark.asyncio
async def test_purge_rolls_back_every_delete_on_mid_operation_failure(
    tmp_path: Path,
) -> None:
    database = tmp_path / "app.sqlite3"
    repository, _, conversation_id, _, _ = await setup_conversation(database)
    continuity = await repository.get_continuity_for_conversation(conversation_id)
    await repository.append_profile_event(
        manual_profile_event(continuity.user_profile_id), LedgerActor.USER
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO profile_derived_data(id, user_profile_id, kind, content, created_at)
            VALUES('derived-1', ?, 'summary', 'ミナ', ?)
            """,
            (continuity.user_profile_id, NOW.isoformat()),
        )
        connection.execute(
            """
            CREATE TRIGGER fail_profile_derived_delete
            BEFORE DELETE ON profile_derived_data
            BEGIN
                SELECT RAISE(ABORT, 'injected purge failure');
            END
            """
        )
    with pytest.raises(PersistenceError):
        await repository.purge_profile(
            request_id="purge-failure",
            user_profile_id=continuity.user_profile_id,
            actor=LedgerActor.USER,
        )
    assert [item.value for item in await repository.project_profile(
        continuity.user_profile_id
    )] == ["ミナ"]
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM profile_derived_data"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM profile_purge_receipts"
        ).fetchone() == (0,)


@pytest.mark.asyncio
async def test_relationship_ledger_has_direction_history_idempotency_and_concurrency(
    tmp_path: Path,
) -> None:
    repository, _, conversation_id, continuity_id, character_id = (
        await setup_conversation(tmp_path / "app.sqlite3")
    )
    continuity = await repository.get_continuity_for_conversation(conversation_id)
    definitions = {
        item.id: item for item in await repository.list_relationship_definitions()
    }
    assert len(definitions) == 68
    assert {item.category for item in definitions.values()} == {
        "王道・絆",
        "対立・歪み",
        "現実・社会",
        "特殊・現代",
        "主従・契約・心理",
        "ファンタジー・特殊設定",
        "変化球・日常・アウトロー",
    }
    assert definitions["friend"].direction.value == "symmetric"
    assert definitions["mentor"].direction.value == "directed"
    assert definitions["mentor"].role_a == "師"
    assert "coercion_or_confinement" in definitions["captor"].caution_tags
    friend = relationship_event(
        event_id="friend-active",
        continuity_id=continuity_id,
        profile_id=continuity.user_profile_id,
        character_id=character_id,
        meaning=RelationshipMeaning.RELATIONSHIP_SET,
        definition_id="friend",
        assignment_state=RelationshipAssignmentState.ACTIVE,
    )
    rival = replace(friend, id="rival-active", relationship_definition_id="rival")
    retired = replace(
        friend,
        id="friend-retired",
        meaning=RelationshipMeaning.RELATIONSHIP_RETIRED,
        assignment_state=RelationshipAssignmentState.HISTORICAL,
    )
    for item in (friend, friend, rival, retired):
        await repository.append_relationship_event(item, LedgerActor.USER)
    stored = await repository.list_relationship_events(
        continuity_id, continuity.user_profile_id, character_id
    )
    assert [item.id for item in stored] == [
        "friend-active",
        "rival-active",
        "friend-retired",
    ]
    pending = replace(
        relationship_event(
            event_id="pending",
            continuity_id=continuity_id,
            profile_id=continuity.user_profile_id,
            character_id=character_id,
        ),
        approval=RelationshipApproval.PENDING_CONFIRMATION,
    )
    await repository.append_relationship_event(pending, LedgerActor.AI)
    results = await asyncio.gather(
        repository.decide_relationship_event(
            decision_id="confirm",
            user_profile_id=continuity.user_profile_id,
            target_event_id=pending.id,
            state="confirmed",
            actor=LedgerActor.USER,
            recorded_at=NOW,
        ),
        repository.decide_relationship_event(
            decision_id="reject",
            user_profile_id=continuity.user_profile_id,
            target_event_id=pending.id,
            state="rejected",
            actor=LedgerActor.USER,
            recorded_at=NOW,
        ),
        return_exceptions=True,
    )
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, ValidationError) for result in results) == 1
    metrics = await repository.project_relationship_metrics(
        continuity_id, continuity.user_profile_id, character_id
    )
    assert metrics.applied_event_ids.count("pending") <= 1


@pytest.mark.asyncio
async def test_relationship_source_rejects_other_worldline_and_other_branch(
    tmp_path: Path,
) -> None:
    repository, _, conversation_id, continuity_id, character_id = (
        await setup_conversation(tmp_path / "app.sqlite3")
    )
    first = await repository.get_conversation(conversation_id)
    continuity = await repository.get_continuity_for_conversation(conversation_id)
    session = await repository.start_send(conversation_id, "約束した")
    event = relationship_event(
        event_id="source-event",
        continuity_id=continuity_id,
        profile_id=continuity.user_profile_id,
        character_id=character_id,
        conversation_id=conversation_id,
        branch_id=first.active_branch_id,
        message_id=session.user_message.id,
    )
    await repository.append_relationship_event(event, LedgerActor.USER)
    model = await repository.ensure_model_profile("ollama-local", "local-model", {})
    character = await repository.ensure_default_character()
    other = await repository.create_conversation("別世界線", character.id, model.id)
    with pytest.raises((ValidationError, PersistenceError)):
        await repository.append_relationship_event(
            replace(
                event,
                id="wrong-worldline",
                source_conversation_id=other.id,
                source_branch_id=other.active_branch_id,
            ),
            LedgerActor.USER,
        )


@pytest.mark.asyncio
async def test_relationship_candidate_rejects_character_outside_current_cast(
    tmp_path: Path,
) -> None:
    repository, service, conversation_id, _, _ = await setup_conversation(
        tmp_path / "app.sqlite3"
    )
    conversation = await repository.get_conversation(conversation_id)
    outsider = await repository.create_character_version("部外者", "outside")
    session = await repository.start_send(conversation_id, "最低だ")
    candidate = RelationshipCandidate(
        event_id="outsider-candidate",
        character_id=outsider.character_id,
        meaning=RelationshipMeaning.BOUNDARY_VIOLATION,
        severity=RelationshipSeverity.LOW,
        evidence_context=EvidenceContext.DIRECT,
        evidence_start=0,
        evidence_end=3,
        evidence_text="最低だ",
    )
    assert await service.add_relationship_candidate(
        conversation_id=conversation_id,
        branch_id=conversation.active_branch_id,
        source_message_id=session.user_message.id,
        source_text="最低だ",
        candidate=candidate,
        recorded_at=NOW,
    ) is None


@pytest.mark.asyncio
async def test_interpretation_versions_are_evidence_bound_and_persisted(
    tmp_path: Path,
) -> None:
    database = tmp_path / "app.sqlite3"
    repository, _, conversation_id, continuity_id, character_id = (
        await setup_conversation(database)
    )
    continuity = await repository.get_continuity_for_conversation(conversation_id)
    character = await repository.ensure_default_character()
    base = relationship_event(
        event_id="evidence-1",
        continuity_id=continuity_id,
        profile_id=continuity.user_profile_id,
        character_id=character_id,
    )
    await repository.append_relationship_event(base, LedgerActor.USER)
    first = RelationshipInterpretation(
        id="interpretation-1",
        continuity_id=continuity_id,
        user_profile_id=continuity.user_profile_id,
        character_id=character_id,
        character_version_id=character.id,
        relationship_definition_ids=(),
        summary="約束を守る相手だと思っている",
        evidence_event_ids=(base.id,),
        state=RelationshipInterpretationState.CURRENT,
        generated_at=NOW,
    )
    second = replace(
        first,
        id="interpretation-2",
        summary="信頼を積み重ねている",
    )
    await repository.save_relationship_interpretation(first, LedgerActor.SYSTEM)
    await repository.save_relationship_interpretation(second, LedgerActor.SYSTEM)
    reopened = SQLiteAppRepository(database)
    await reopened.initialize()
    assert await reopened.get_current_relationship_interpretation(
        continuity_id, continuity.user_profile_id, character_id
    ) == second
    with pytest.raises(ValidationError):
        await reopened.save_relationship_interpretation(
            replace(second, id="bad", evidence_event_ids=("missing",)),
            LedgerActor.SYSTEM,
        )


@pytest.mark.asyncio
async def test_relationship_context_is_loopback_only_and_uses_group_intersection(
    tmp_path: Path,
) -> None:
    repository, service, conversation_id, continuity_id, first_character_id = (
        await setup_conversation(tmp_path / "app.sqlite3")
    )
    continuity = await repository.get_continuity_for_conversation(conversation_id)
    second_version = await repository.create_character_version("B", "second prompt")
    secret_profile = replace(
        manual_profile_event(
            continuity.user_profile_id,
            "secret-profile",
            scope=ProfileScope.SELECTED_CHARACTERS,
        ),
        item_kind="secret",
        item_name="秘密の呼ばれ方",
        known_by_character_ids=(first_character_id,),
    )
    await repository.append_profile_event(secret_profile, LedgerActor.USER)
    shared_profile = manual_profile_event(
        continuity.user_profile_id, "shared-profile"
    )
    await repository.append_profile_event(shared_profile, LedgerActor.USER)
    private_event = replace(
        relationship_event(
            event_id="private-event",
            continuity_id=continuity_id,
            profile_id=continuity.user_profile_id,
            character_id=first_character_id,
        ),
        known_by_character_ids=(first_character_id,),
    )
    await repository.append_relationship_event(private_event, LedgerActor.USER)

    single = await service.render_generation_context(
        conversation_id=conversation_id,
        character_ids=(first_character_id,),
        provider_endpoint="http://127.0.0.1:11434",
    )
    group = await service.render_generation_context(
        conversation_id=conversation_id,
        character_ids=(first_character_id, second_version.character_id),
        provider_endpoint="http://localhost:11434",
    )
    lan = await service.render_generation_context(
        conversation_id=conversation_id,
        character_ids=(first_character_id,),
        provider_endpoint="http://192.168.1.20:11434",
    )

    assert "秘密の呼ばれ方" in single
    assert "private-event" in single
    assert "呼ばれ方" in group
    assert "秘密の呼ばれ方" not in group
    assert "private-event" not in group
    assert lan == ""

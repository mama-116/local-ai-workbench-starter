import sqlite3
from pathlib import Path

import pytest

from local_llm_chat.domain.canonical_memory import CanonicalMemoryEvent
from local_llm_chat.domain.models import (
    CharacterVersion,
    Conversation,
    ModelProfile,
    TelemetryMetric,
)
from local_llm_chat.application.services.rag_service import RagService
from local_llm_chat.domain.errors import PersistenceError, ValidationError
from local_llm_chat.domain.states import (
    ContextSummaryState,
    MemoryApprovalState,
    MemoryCardinality,
    MemoryKind,
    MessageState,
    TranslationState,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


@pytest.mark.asyncio
async def test_send_finish_and_rewrite_preserve_original_branch(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "試験", character.id, profile.id
    )
    original_branch_id = conversation.active_branch_id

    first = await repository.start_send(conversation.id, "最初の質問")
    await repository.checkpoint_response(first.assistant_message.id, "最初の")
    await repository.finish_response(
        first, "最初の回答", MessageState.COMPLETED, output_tokens=4
    )

    original_messages = await repository.list_active_messages(conversation.id)
    assert [message.content for message in original_messages] == [
        "最初の質問",
        "最初の回答",
    ]

    rewritten = await repository.start_rewrite(
        conversation.id, first.user_message.id, "書き直した質問"
    )
    await repository.finish_response(
        rewritten, "別の回答", MessageState.COMPLETED
    )
    rewritten_messages = await repository.list_active_messages(conversation.id)
    assert [message.content for message in rewritten_messages] == [
        "書き直した質問",
        "別の回答",
    ]

    branches = await repository.list_branches(conversation.id)
    assert len(branches) == 2
    await repository.activate_branch(conversation.id, original_branch_id)
    restored_messages = await repository.list_active_messages(conversation.id)
    assert [message.content for message in restored_messages] == [
        "最初の質問",
        "最初の回答",
    ]


@pytest.mark.asyncio
async def test_recover_interrupted_response_keeps_partial_content(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "復旧試験", character.id, profile.id
    )
    session = await repository.start_send(conversation.id, "質問")
    await repository.checkpoint_response(session.assistant_message.id, "途中まで")

    await repository.recover_interrupted_runs()

    messages = await repository.list_active_messages(conversation.id)
    assert messages[-1].content == "途中まで"
    assert messages[-1].state is MessageState.FAILED


@pytest.mark.asyncio
async def test_archived_conversation_can_be_restored(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "archive test", character.id, profile.id
    )

    await repository.archive_conversation(conversation.id)

    assert await repository.list_conversations() == []
    archived = await repository.list_archived_conversations()
    assert [item.id for item in archived] == [conversation.id]

    await repository.restore_conversation(conversation.id)

    restored = await repository.list_conversations()
    assert [item.id for item in restored] == [conversation.id]
    assert restored[0].archived_at is None


async def _create_conversation(
    repository: SQLiteAppRepository, title: str
) -> tuple[Conversation, CharacterVersion, ModelProfile]:
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        title, character.id, profile.id
    )
    return conversation, character, profile


@pytest.mark.asyncio
async def test_empty_trash_rejects_active_or_changed_conversation_set(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    archived, _, _ = await _create_conversation(repository, "archived")
    active, _, _ = await _create_conversation(repository, "active")
    await repository.archive_conversation(archived.id)

    with pytest.raises(ValidationError, match="ゴミ箱"):
        await repository.delete_archived_conversations((archived.id, active.id))

    later, _, _ = await _create_conversation(repository, "later")
    await repository.archive_conversation(later.id)
    with pytest.raises(ValidationError, match="変更"):
        await repository.delete_archived_conversations((archived.id,))

    assert {item.id for item in await repository.list_archived_conversations()} == {
        archived.id,
        later.id,
    }
    assert [item.id for item in await repository.list_conversations()] == [active.id]


@pytest.mark.asyncio
async def test_empty_trash_deletes_all_conversation_data_but_keeps_shared_data(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    archived, character, profile = await _create_conversation(repository, "archived")
    active = await repository.create_conversation(
        "active", character.id, profile.id
    )
    archived_session = await repository.start_send(archived.id, "same source")
    archived_response = await repository.finish_response(
        archived_session, "same answer", MessageState.COMPLETED
    )
    active_session = await repository.start_send(active.id, "same source")
    active_response = await repository.finish_response(
        active_session, "same answer", MessageState.COMPLETED
    )
    archived_translation = await repository.prepare_translation(
        archived_response.id, "ja", "ollama-local", "gemma4:12b", force=False
    )
    await repository.mark_translation_running(archived_translation.translation.id)
    await repository.finish_translation(
        archived_translation.translation.id,
        "同じ回答",
        TranslationState.COMPLETED,
    )
    active_translation = await repository.prepare_translation(
        active_response.id, "ja", "ollama-local", "gemma4:12b", force=False
    )
    await repository.append_canonical_memory_event(
        CanonicalMemoryEvent(
            id="memory-to-delete",
            conversation_id=archived.id,
            branch_id=archived_session.branch_id,
            subject_id="user",
            kind=MemoryKind.PREFERENCE,
            slot="topic",
            value="delete me",
            cardinality=MemoryCardinality.SINGLE,
            approval=MemoryApprovalState.AUTO_SAVED,
            source_message_id=archived_session.user_message.id,
            known_by_character_ids=frozenset({character.character_id}),
            supersedes_event_id=None,
            effective_at=archived_session.user_message.created_at,
            recorded_at=archived_session.user_message.created_at,
        )
    )
    rag = RagService(repository)
    document = await rag.register_document("shared.md", b"shared content")
    await rag.select_documents(archived.id, [document.id])
    await rag.select_documents(active.id, [document.id])
    assert (
        active_translation.translation.reused_from_id
        == archived_translation.translation.id
    )
    now = "2026-07-23T00:00:00+00:00"
    with sqlite3.connect(database_path) as connection:
        chunk_id = connection.execute(
            "SELECT id FROM chunks WHERE document_id = ?", (document.id,)
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO telemetry_samples(
                id, run_id, metric_name, value, unit, source, captured_at
            ) VALUES('telemetry-delete', ?, 'cpu_percent', 1, '%', 'test', ?)
            """,
            (archived_session.run.id, now),
        )
        connection.execute(
            """
            INSERT INTO app_events(
                id, level, event_type, run_id, details_json, created_at
            ) VALUES('event-delete', 'info', 'test', ?, '{}', ?)
            """,
            (archived_session.run.id, now),
        )
        connection.execute(
            """
            INSERT INTO run_rag_usage(run_id, selected_document_count)
            VALUES(?, 1)
            """,
            (archived_session.run.id,),
        )
        connection.execute(
            """
            INSERT INTO run_citations(run_id, chunk_id, rank, score)
            VALUES(?, ?, 0, 1)
            """,
            (archived_session.run.id, chunk_id),
        )
        connection.execute(
            """
            INSERT INTO context_summaries(
                id, conversation_id, branch_id, source_message_ids_json,
                source_hash, settings_hash, model, prompt_version, content,
                state, created_at, completed_at
            ) VALUES(
                'summary-delete', ?, ?, '[]', 'source', 'settings',
                'model', 'v1', 'summary', 'completed', ?, ?
            )
            """,
            (archived.id, archived_session.branch_id, now, now),
        )
        connection.execute(
            """
            INSERT INTO conversation_tool_folder_grants(
                conversation_id, root_path, granted_at
            ) VALUES(?, 'C:\\allowed', ?)
            """,
            (archived.id, now),
        )
        connection.execute(
            """
            INSERT INTO tool_calls(
                id, conversation_id, run_id, provider, tool_name, input_json,
                state, created_at, completed_at
            ) VALUES(
                'tool-delete', ?, ?, 'builtin', 'read', '{}',
                'completed', ?, ?
            )
            """,
            (archived.id, archived_session.run.id, now, now),
        )
        connection.execute(
            """
            INSERT INTO agent_runs(
                id, conversation_id, objective, allowed_tools_json,
                max_cost_units, max_steps, max_duration_seconds, state,
                created_at, completed_at
            ) VALUES(
                'agent-delete', ?, 'test', '[]', 1, 1, 1,
                'completed', ?, ?
            )
            """,
            (archived.id, now, now),
        )
        connection.execute(
            """
            INSERT INTO agent_steps(
                id, run_id, ordinal, tool_name, arguments_json, action_hash,
                state, created_at, completed_at
            ) VALUES(
                'agent-step-delete', 'agent-delete', 1, 'test', '{}',
                'hash', 'completed', ?, ?
            )
            """,
            (now, now),
        )
        connection.execute(
            """
            INSERT INTO computer_use_runs(
                id, conversation_id, objective, observation_id, plan_hash,
                max_actions, max_duration_seconds, approval_timeout_seconds,
                planned_action_count, state, created_at, completed_at
            ) VALUES(
                'computer-delete', ?, 'test', 'observation',
                'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                1, 1, 1, 1, 'completed', ?, ?
            )
            """,
            (archived.id, now, now),
        )
        connection.execute(
            """
            INSERT INTO computer_plan_approvals(
                id, run_id, plan_hash, approved_at, consumed_at
            ) VALUES(
                'approval-delete', 'computer-delete',
                'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                ?, ?
            )
            """,
            (now, now),
        )
        connection.execute(
            """
            INSERT INTO computer_actions(
                id, run_id, ordinal, request_id, action_type,
                target_profile_id, state, created_at, completed_at
            ) VALUES(
                'action-delete', 'computer-delete', 1, 'request',
                'click_uia_element', 'profile', 'completed', ?, ?
            )
            """,
            (now, now),
        )
        connection.execute(
            """
            INSERT INTO turn_batches(
                id, conversation_id, branch_id, source_message_id,
                response_message_id, run_id, model, mode,
                formal_character_ids_json, guest_ids_json, prompt_version,
                state, repair_state, created_at
            ) VALUES(
                'batch-delete', ?, ?, ?, ?, ?, 'model', 'story',
                '[]', '[]', 'v1', 'completed', 'not_needed', ?
            )
            """,
            (
                archived.id,
                archived_session.branch_id,
                archived_session.user_message.id,
                archived_response.id,
                archived_session.run.id,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO turn_segments(
                id, turn_batch_id, position, speaker_kind,
                display_name, content
            ) VALUES(
                'segment-delete', 'batch-delete', 0, 'narrator',
                'Narrator', 'content'
            )
            """
        )
    await repository.archive_conversation(archived.id)

    deleted = await repository.delete_archived_conversations((archived.id,))

    assert deleted == 1
    assert await repository.list_archived_conversations() == []
    assert [item.id for item in await repository.list_conversations()] == [active.id]
    assert [message.content for message in await repository.list_active_messages(active.id)] == [
        "same source",
        "same answer",
    ]
    surviving_translation = await repository.get_current_translation(
        active_response.id
    )
    assert surviving_translation is not None
    assert surviving_translation.reused_from_id is None
    assert active_translation.translation.id
    assert [item.id for item in await rag.selected_documents(active.id)] == [
        document.id
    ]
    assert await repository.delete_archived_conversations((archived.id,)) == 0

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM conversations WHERE id = ?", (archived.id,)
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_memory_events "
            "WHERE conversation_id = ?",
            (archived.id,),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM documents WHERE id = ?", (document.id,)
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM characters WHERE id = ?",
            (character.character_id,),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM model_profiles WHERE id = ?", (profile.id,)
        ).fetchone() == (1,)


@pytest.mark.asyncio
async def test_empty_trash_rolls_back_everything_when_a_child_delete_fails(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    conversation, _, _ = await _create_conversation(repository, "rollback")
    session = await repository.start_send(conversation.id, "question")
    await repository.finish_response(session, "answer", MessageState.COMPLETED)
    await repository.archive_conversation(conversation.id)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            f"""
            CREATE TRIGGER fail_conversation_message_delete
            BEFORE DELETE ON messages
            WHEN OLD.conversation_id = '{conversation.id}'
            BEGIN
                SELECT RAISE(ABORT, 'injected delete failure');
            END
            """
        )

    with pytest.raises(PersistenceError, match="injected delete failure"):
        await repository.delete_archived_conversations((conversation.id,))

    assert [item.id for item in await repository.list_archived_conversations()] == [
        conversation.id
    ]
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT content FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at",
            (conversation.id,),
        ).fetchall() == [("question",), ("answer",)]
        assert connection.execute(
            "SELECT COUNT(*) FROM conversation_deletion_guards"
        ).fetchone() == (0,)


@pytest.mark.asyncio
async def test_canonical_memory_cannot_be_deleted_without_trash_transaction(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    conversation, character, _ = await _create_conversation(repository, "memory")
    session = await repository.start_send(conversation.id, "source")
    await repository.finish_response(session, "answer", MessageState.COMPLETED)
    await repository.append_canonical_memory_event(
        CanonicalMemoryEvent(
            id="append-only-memory",
            conversation_id=conversation.id,
            branch_id=session.branch_id,
            subject_id="user",
            kind=MemoryKind.GOAL,
            slot="goal",
            value="preserve guard",
            cardinality=MemoryCardinality.SINGLE,
            approval=MemoryApprovalState.AUTO_SAVED,
            source_message_id=session.user_message.id,
            known_by_character_ids=frozenset({character.character_id}),
            supersedes_event_id=None,
            effective_at=session.user_message.created_at,
            recorded_at=session.user_message.created_at,
        )
    )
    await repository.archive_conversation(conversation.id)

    with sqlite3.connect(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM canonical_memory_events WHERE id = ?",
                ("append-only-memory",),
            )


@pytest.mark.asyncio
async def test_conversation_translation_is_manual_by_default_and_can_be_toggled(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "translation preference", character.id, profile.id
    )

    assert conversation.auto_translate is False

    await repository.set_conversation_auto_translate(conversation.id, True)

    assert (await repository.get_conversation(conversation.id)).auto_translate is True


@pytest.mark.asyncio
async def test_only_non_active_non_root_branch_can_be_hidden_and_restored(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "branch visibility", character.id, profile.id
    )
    first = await repository.start_send(conversation.id, "original")
    await repository.finish_response(first, "answer", MessageState.COMPLETED)
    alternate = await repository.start_rewrite(
        conversation.id, first.user_message.id, "alternate"
    )
    await repository.finish_response(alternate, "other answer", MessageState.COMPLETED)

    with pytest.raises(ValidationError, match="使用中"):
        await repository.hide_branch(conversation.id, alternate.branch_id)
    await repository.activate_branch(conversation.id, conversation.active_branch_id)
    with pytest.raises(ValidationError, match="最初"):
        await repository.hide_branch(conversation.id, conversation.active_branch_id)

    await repository.hide_branch(conversation.id, alternate.branch_id)

    assert [branch.id for branch in await repository.list_branches(conversation.id)] == [
        conversation.active_branch_id
    ]
    all_branches = await repository.list_all_branches(conversation.id)
    assert next(branch for branch in all_branches if branch.id == alternate.branch_id).hidden_at
    with pytest.raises(ValidationError, match="非表示"):
        await repository.activate_branch(conversation.id, alternate.branch_id)

    await repository.restore_branch(conversation.id, alternate.branch_id)
    assert len(await repository.list_branches(conversation.id)) == 2


@pytest.mark.asyncio
async def test_translation_attempt_is_stored_separately_from_message(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "翻訳保存試験", character.id, profile.id
    )
    session = await repository.start_send(conversation.id, "質問")
    response = await repository.finish_response(
        session, "Hello.", MessageState.COMPLETED
    )

    prepared = await repository.prepare_translation(
        response.id, "ja", "ollama-local", "gemma4:12b", force=False
    )
    assert prepared.should_enqueue
    await repository.mark_translation_running(prepared.translation.id)
    completed = await repository.finish_translation(
        prepared.translation.id,
        "こんにちは。",
        TranslationState.COMPLETED,
    )

    assert completed.message_id == response.id
    assert completed.content == "こんにちは。"
    assert (await repository.get_message(response.id)).content == "Hello."


@pytest.mark.asyncio
async def test_latest_telemetry_uses_last_completed_run_and_preserves_missing_reason(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "観測試験", character.id, profile.id
    )
    session = await repository.start_send(conversation.id, "質問")
    await repository.finish_response(
        session,
        "回答",
        MessageState.COMPLETED,
        output_tokens=20,
        total_duration_ns=2_000_000_000,
        generation_duration_ns=1_000_000_000,
        response_duration_ms=2500,
    )
    await repository.save_telemetry_metrics(
        session.run.id,
        (
            TelemetryMetric("cpu_percent", 25.0, "%", "DESKTOP-A"),
            TelemetryMetric(
                "gpu_percent", None, "%", "DESKTOP-A", "nvidia-smiが見つかりません"
            ),
        ),
    )

    latest = await repository.get_latest_telemetry(conversation.id)

    assert latest is not None
    assert latest.run.id == session.run.id
    assert latest.run.response_duration_ms == 2500
    assert latest.tokens_per_second == pytest.approx(20.0)
    assert latest.metric("cpu_percent").value == 25.0
    assert latest.metric("gpu_percent").value is None
    assert (
        latest.metric("gpu_percent").unavailable_reason
        == "nvidia-smiが見つかりません"
    )


@pytest.mark.asyncio
async def test_rag_document_is_deduplicated_and_search_returns_precise_citation(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    service = RagService(repository)
    payload = "ブロッコリーにはビタミンCが含まれます。\n加熱時間を短くします。".encode()

    first = await service.register_document("nutrition.md", payload)
    duplicate = await service.register_document("copy.md", payload)
    results = await service.search("ビタミンC")

    assert duplicate.id == first.id
    assert len(results) == 1
    assert results[0].citation.document_id == first.id
    assert results[0].citation.document_title == "nutrition.md"
    assert results[0].citation.start_offset == 0
    assert "ビタミンC" in results[0].content
    assert await service.search("存在しない固有語") == []


@pytest.mark.asyncio
async def test_rag_search_is_limited_to_documents_selected_for_conversation(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "RAG選択試験", character.id, profile.id
    )
    service = RagService(repository)
    selected = await service.register_document(
        "selected.md", "共通語 選択された資料".encode()
    )
    await service.register_document("ignored.md", "共通語 選択されていない資料".encode())

    await service.select_documents(conversation.id, [selected.id])
    results = await service.search_for_conversation(conversation.id, "共通語")

    assert [result.citation.document_title for result in results] == ["selected.md"]
    assert [document.id for document in await service.selected_documents(conversation.id)] == [
        selected.id
    ]


@pytest.mark.asyncio
async def test_context_summary_is_derived_restart_safe_and_branch_scoped(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 64}
    )
    conversation = await repository.create_conversation(
        "要約保存試験", character.id, profile.id
    )
    original = await repository.start_send(conversation.id, "日本語🙂の原文")
    await repository.finish_response(
        original, "変更されない回答", MessageState.COMPLETED
    )
    source_ids = (original.user_message.id, original.assistant_message.id)

    prepared = await repository.prepare_context_summary(
        conversation.id,
        original.branch_id,
        source_ids,
        "source-hash",
        "settings-hash",
        "gemma4:12b",
        "context-summary-v1",
    )
    assert prepared.should_generate
    await repository.mark_context_summary_running(prepared.summary.id)
    completed = await repository.finish_context_summary(
        prepared.summary.id,
        "ローカル要約🙂",
        ContextSummaryState.COMPLETED,
    )
    assert completed.source_message_ids == source_ids

    reopened = SQLiteAppRepository(database_path)
    await reopened.initialize()
    assert [message.content for message in await reopened.list_active_messages(conversation.id)] == [
        "日本語🙂の原文",
        "変更されない回答",
    ]
    reusable = await reopened.prepare_context_summary(
        conversation.id,
        original.branch_id,
        source_ids,
        "source-hash",
        "settings-hash",
        "gemma4:12b",
        "context-summary-v1",
    )
    assert not reusable.should_generate
    assert reusable.summary.id == completed.id

    rewritten = await reopened.start_rewrite(
        conversation.id, original.user_message.id, "別分岐"
    )
    branch_scoped = await reopened.prepare_context_summary(
        conversation.id,
        rewritten.branch_id,
        (rewritten.user_message.id,),
        "source-hash",
        "settings-hash",
        "gemma4:12b",
        "context-summary-v1",
    )
    changed_settings = await reopened.prepare_context_summary(
        conversation.id,
        original.branch_id,
        source_ids,
        "source-hash",
        "changed-settings-hash",
        "gemma4:12b",
        "context-summary-v1",
    )
    assert branch_scoped.should_generate
    assert changed_settings.should_generate
    assert branch_scoped.summary.id != completed.id
    assert changed_settings.summary.id != completed.id

    with pytest.raises(ValidationError, match="古い連続区間"):
        await reopened.prepare_context_summary(
            conversation.id,
            rewritten.branch_id,
            source_ids,
            "cross-branch-source-hash",
            "settings-hash",
            "gemma4:12b",
            "context-summary-v1",
        )

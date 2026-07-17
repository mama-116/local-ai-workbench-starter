from pathlib import Path

import pytest

from local_llm_chat.domain.models import TelemetryMetric
from local_llm_chat.application.services.rag_service import RagService
from local_llm_chat.domain.states import MessageState, TranslationState
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

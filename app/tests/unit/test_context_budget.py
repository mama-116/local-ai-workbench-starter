from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from local_llm_chat.application.services.context_budget import (
    ConservativeContextCounter,
    ContextBudgetAction,
    ContextBudgetInput,
    ContextBudgetPlanner,
    ContextWindowManager,
)
from local_llm_chat.domain.errors import OllamaUnavailable
from local_llm_chat.domain.models import (
    ChatChunk,
    ChatMessageInput,
    ChatRequest,
    ModelProfile,
    RunSession,
    ToolCallRequest,
    ToolDefinition,
)
from local_llm_chat.domain.states import ContextSummaryState, MessageRole, MessageState
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


class OneUnitCounter:
    """Count every declared capacity component as exactly one unit."""

    def count_text(self, value: str) -> int:
        assert value
        return 1

    def count_message(self, value: ChatMessageInput) -> int:
        assert value.content or value.tool_calls
        return 1

    def count_tool(self, value: ToolDefinition) -> int:
        assert value.name
        return 1


def test_one_unit_over_limit_includes_every_capacity_component() -> None:
    planner = ContextBudgetPlanner(OneUnitCounter())

    decision = planner.decide(
        ContextBudgetInput(
            context_limit=5,
            output_reserve=1,
            system_prompt="system prompt",
            branch_messages=(
                ChatMessageInput(MessageRole.USER, "current branch"),
            ),
            rag_context="local RAG context",
            tools=(
                ToolDefinition("read_local", "read", {"type": "object"}),
            ),
            tool_results=(
                ChatMessageInput(MessageRole.TOOL, "local tool result", "read_local"),
            ),
        )
    )

    assert decision.used_units == 6
    assert decision.excess_units == 1
    assert decision.action is ContextBudgetAction.COMPRESS

    exact = planner.decide(
        ContextBudgetInput(
            context_limit=6,
            output_reserve=1,
            system_prompt="system prompt",
            branch_messages=(
                ChatMessageInput(MessageRole.USER, "current branch"),
            ),
            rag_context="local RAG context",
            tools=(
                ToolDefinition("read_local", "read", {"type": "object"}),
            ),
            tool_results=(
                ChatMessageInput(MessageRole.TOOL, "local tool result", "read_local"),
            ),
        )
    )
    assert exact.used_units == exact.context_limit
    assert exact.action is ContextBudgetAction.SEND


def test_conservative_counter_estimates_tokens_instead_of_raw_utf8_bytes() -> None:
    text = "長い日本語🙂"
    counter = ConservativeContextCounter()
    used = counter.count_text(text)
    planner = ContextBudgetPlanner(counter)

    exact = planner.decide(ContextBudgetInput(used + 1, 1, text, ()))
    over = planner.decide(ContextBudgetInput(used, 1, text, ()))

    assert exact.action is ContextBudgetAction.SEND
    assert len(text) < used < len(text.encode("utf-8"))
    assert over.excess_units == 1
    assert over.action is ContextBudgetAction.COMPRESS


def test_japanese_system_prompt_does_not_exhaust_a_4096_token_window() -> None:
    counter = ConservativeContextCounter()
    planner = ContextBudgetPlanner(counter)
    japanese_prompt = "登場人物の設定です。" * 300

    decision = planner.decide(
        ContextBudgetInput(
            context_limit=4096,
            output_reserve=512,
            system_prompt=japanese_prompt,
            branch_messages=(ChatMessageInput(MessageRole.USER, "続きを話してください。"),),
        )
    )

    assert decision.action is ContextBudgetAction.SEND


class SummaryProvider:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.calls += 1
        if self.fail:
            raise OllamaUnavailable("Ollama stopped during local summary")
        assert "ローカル会話の圧縮器" in request.system_prompt
        yield ChatChunk(content="決定事項を保持した要約🙂")
        yield ChatChunk(done=True)


async def context_fixture(
    tmp_path: Path,
) -> tuple[SQLiteAppRepository, str, ModelProfile, RunSession]:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4, "num_predict": 1}
    )
    conversation = await repository.create_conversation(
        "長い会話", character.id, profile.id
    )
    first = await repository.start_send(conversation.id, "最初の日本語🙂")
    await repository.finish_response(first, "最初の回答", MessageState.COMPLETED)
    second = await repository.start_send(conversation.id, "現在の質問")
    return repository, conversation.id, profile, second


@pytest.mark.asyncio
async def test_local_summary_replaces_old_prefix_and_is_reused_after_restart(
    tmp_path: Path,
) -> None:
    repository, conversation_id, profile, session = await context_fixture(tmp_path)
    source = tuple(await repository.context_to_message(session.user_message.id))
    request = ChatRequest(
        profile.model_name,
        "system",
        tuple(ChatMessageInput(message.role, message.content) for message in source),
        profile.parameters,
    )
    provider = SummaryProvider()
    manager = ContextWindowManager(repository, OneUnitCounter())

    prepared = await manager.prepare(
        request,
        base_system_prompt="system",
        rag_context="",
        conversation_id=conversation_id,
        branch_id=session.branch_id,
        profile=profile,
        provider=provider,  # type: ignore[arg-type]
        source_messages=source,
        run_id=session.run.id,
    )

    assert prepared.action is ContextBudgetAction.COMPRESS
    assert prepared.notice and not prepared.notice_is_warning
    assert prepared.request.messages[0].content.startswith(
        "[以前の会話のローカル要約]"
    )
    assert prepared.request.messages[-1].content == "現在の質問"
    [summary] = await repository.list_context_summaries(conversation_id)
    assert summary.state is ContextSummaryState.COMPLETED
    assert provider.calls == 1

    reopened = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await reopened.initialize()
    reused = await ContextWindowManager(reopened, OneUnitCounter()).prepare(
        request,
        base_system_prompt="system",
        rag_context="",
        conversation_id=conversation_id,
        branch_id=session.branch_id,
        profile=profile,
        provider=provider,  # type: ignore[arg-type]
        source_messages=source,
        run_id=session.run.id,
    )
    assert reused.action is ContextBudgetAction.COMPRESS
    assert provider.calls == 1

    changed_profile = ModelProfile(
        profile.id,
        profile.provider,
        profile.model_name,
        {**profile.parameters, "temperature": 0.9},
    )
    changed = await ContextWindowManager(reopened, OneUnitCounter()).prepare(
        request,
        base_system_prompt="system",
        rag_context="",
        conversation_id=conversation_id,
        branch_id=session.branch_id,
        profile=changed_profile,
        provider=provider,  # type: ignore[arg-type]
        source_messages=source,
        run_id=session.run.id,
    )
    assert changed.action is ContextBudgetAction.COMPRESS
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_summary_failure_uses_recent_context_and_preserves_originals(
    tmp_path: Path,
) -> None:
    repository, conversation_id, profile, session = await context_fixture(tmp_path)
    source = tuple(await repository.context_to_message(session.user_message.id))
    request = ChatRequest(
        profile.model_name,
        "system",
        tuple(ChatMessageInput(message.role, message.content) for message in source),
        profile.parameters,
    )

    prepared = await ContextWindowManager(repository, OneUnitCounter()).prepare(
        request,
        base_system_prompt="system",
        rag_context="",
        conversation_id=conversation_id,
        branch_id=session.branch_id,
        profile=profile,
        provider=SummaryProvider(fail=True),  # type: ignore[arg-type]
        source_messages=source,
        run_id=session.run.id,
    )

    assert prepared.action is ContextBudgetAction.FALLBACK
    assert prepared.notice_is_warning
    assert [message.content for message in prepared.request.messages] == [
        "最初の回答",
        "現在の質問",
    ]
    [summary] = await repository.list_context_summaries(conversation_id)
    assert summary.state is ContextSummaryState.FAILED
    assert [message.content for message in await repository.list_active_messages(conversation_id)] == [
        "最初の日本語🙂",
        "最初の回答",
        "現在の質問",
        "",
    ]


@pytest.mark.asyncio
async def test_tool_fallback_never_sends_an_orphan_tool_result(tmp_path: Path) -> None:
    repository, conversation_id, profile, session = await context_fixture(tmp_path)
    manager = ContextWindowManager(repository, OneUnitCounter())
    request = ChatRequest(
        profile.model_name,
        "system",
        (
            ChatMessageInput(MessageRole.USER, "現在の質問"),
            ChatMessageInput(
                MessageRole.ASSISTANT,
                "",
                tool_calls=(
                    ToolCallRequest("call-1", "read_local", {"path": "a.md"}),
                ),
            ),
            ChatMessageInput(MessageRole.TOOL, "大きい結果", "read_local"),
        ),
        profile.parameters,
        (ToolDefinition("read_local", "read", {"type": "object"}),),
    )

    prepared = await manager.prepare(
        request,
        base_system_prompt="system",
        rag_context="RAG",
        conversation_id=conversation_id,
        branch_id=session.branch_id,
        profile=profile,
        provider=SummaryProvider(),  # type: ignore[arg-type]
        run_id=session.run.id,
    )

    assert prepared.action is ContextBudgetAction.FALLBACK
    assert not prepared.request.messages or prepared.request.messages[0].role is not MessageRole.TOOL

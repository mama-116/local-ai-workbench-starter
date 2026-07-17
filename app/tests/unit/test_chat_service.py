from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from local_llm_chat.application.services.chat_service import ChatService
from local_llm_chat.application.services.tool_coordinator import ToolCoordinator
from local_llm_chat.application.services.rag_service import RagService
from local_llm_chat.domain.errors import FreeOperationBlocked, ToolUseUnavailable
from local_llm_chat.domain.models import (
    ChatChunk,
    ChatRequest,
    ModelInfo,
    ProviderConnection,
    ProviderMetadata,
    Translation,
    ToolCallRequest,
    ToolDefinition,
    ToolProviderResult,
)
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.states import CostClass, Locality, MessageState
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


class FakeProvider:
    def __init__(self) -> None:
        self.last_request: ChatRequest | None = None

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            "ollama-local",
            Locality.LOCAL,
            CostClass.NO_CHARGE,
            "http://127.0.0.1:11434",
        )
    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [await self.inspect_model("gemma4:12b")]

    async def inspect_model(self, model_name: str) -> ModelInfo:
        return ModelInfo(model_name, 7_600_000_000, "gguf", "gemma4", "11.9B", "Q4")

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.last_request = request
        assert request.messages[-1].content == "質問"
        yield ChatChunk(content="回答")
        yield ChatChunk(
            content="です", done=True, prompt_tokens=3, output_tokens=2, total_duration_ns=10
        )

    async def close(self) -> None:
        return None


class FakeRegistry:
    def __init__(self, cloud_disabled: bool = True) -> None:
        self.provider = FakeProvider()
        self._cloud_disabled = cloud_disabled

    @property
    def default_provider_name(self) -> str:
        return "ollama-local"

    def list_connections(self) -> list[ProviderConnection]:
        return []

    def get(self, provider_name: str) -> FakeProvider:
        assert provider_name == "ollama-local"
        return self.provider

    def cloud_is_disabled(self, provider_name: str) -> bool:
        assert provider_name == "ollama-local"
        return self._cloud_disabled

    async def save_connection(
        self,
        connection_id: str | None,
        display_name: str,
        endpoint: str,
        cloud_disabled_confirmed: bool,
    ) -> ProviderConnection:
        raise AssertionError("not used")

    async def close(self) -> None:
        return None


class FailingTranslationScheduler:
    async def request_translation(
        self, message_id: str, force: bool = False
    ) -> Translation | None:
        raise RuntimeError("queue unavailable")


class FakeToolProvider:
    @property
    def name(self) -> str:
        return "builtin"

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return (ToolDefinition("read_allowed_text", "read", {"type": "object"}),)

    async def execute(
        self, conversation_id: str, tool_name: str, arguments: dict[str, object]
    ) -> ToolProviderResult:
        return ToolProviderResult("tool private body", 1)


class SequentialToolProvider(FakeToolProvider):
    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return (
            ToolDefinition("search_allowed_folder", "search", {"type": "object"}),
            ToolDefinition("read_allowed_text", "read", {"type": "object"}),
        )

    async def execute(
        self, conversation_id: str, tool_name: str, arguments: dict[str, object]
    ) -> ToolProviderResult:
        if tool_name == "search_allowed_folder":
            return ToolProviderResult('[{"path":"facts.md"}]', 1)
        assert tool_name == "read_allowed_text"
        return ToolProviderResult("LOCAL-TOOLS-OK-731", 1)


class ToolCallingProvider(FakeProvider):
    def __init__(self, unsupported: bool = False) -> None:
        super().__init__()
        self.unsupported = unsupported
        self.requests: list[ChatRequest] = []

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.requests.append(request)
        if request.tools and self.unsupported:
            raise ToolUseUnavailable("unsupported")
        if any(message.role.value == "tool" for message in request.messages):
            yield ChatChunk(content="資料に基づく回答")
            yield ChatChunk(done=True)
            return
        if request.tools:
            yield ChatChunk(
                done=True,
                tool_calls=(
                    ToolCallRequest("call-1", "read_allowed_text", {"path": "a.md"}),
                ),
            )
            return
        yield ChatChunk(content="通常回答")
        yield ChatChunk(done=True)


class SequentialToolCallingProvider(ToolCallingProvider):
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.requests.append(request)
        tool_messages = tuple(
            message for message in request.messages if message.role.value == "tool"
        )
        if not tool_messages:
            yield ChatChunk(
                done=True,
                tool_calls=(
                    ToolCallRequest(
                        "search-1", "search_allowed_folder", {"query": "PHASE5_MARKER"}
                    ),
                ),
            )
            return
        if tool_messages[-1].tool_name == "search_allowed_folder":
            assert request.tools
            yield ChatChunk(
                done=True,
                tool_calls=(
                    ToolCallRequest(
                        "read-1", "read_allowed_text", {"path": "facts.md"}
                    ),
                ),
            )
            return
        yield ChatChunk(content="LOCAL-TOOLS-OK-731")
        yield ChatChunk(done=True)


class RepeatedToolCallingProvider(ToolCallingProvider):
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.requests.append(request)
        if request.tools:
            call_number = 1 + sum(
                message.role.value == "tool" for message in request.messages
            )
            yield ChatChunk(
                done=True,
                tool_calls=(
                    ToolCallRequest(
                        f"call-{call_number}",
                        "read_allowed_text",
                        {"path": "a.md"},
                    ),
                ),
            )
            return
        yield ChatChunk(content="上限後の通常回答")
        yield ChatChunk(done=True)


class ToolRegistry(FakeRegistry):
    def __init__(self, provider: ToolCallingProvider) -> None:
        super().__init__()
        self.provider = provider

    def get(self, provider_name: str) -> ToolCallingProvider:
        assert provider_name == "ollama-local"
        assert isinstance(self.provider, ToolCallingProvider)
        return self.provider


async def make_conversation(repository: SQLiteAppRepository) -> str:
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "会話", character.id, profile.id
    )
    return conversation.id


@pytest.mark.asyncio
async def test_streams_and_persists_completed_response(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id = await make_conversation(repository)
    updates: list[str] = []

    async def on_update(content: str) -> None:
        updates.append(content)

    service = ChatService(repository, FakeRegistry(), FreeOperationPolicy())
    response = await service.send_message(conversation_id, "質問", on_update)

    assert updates == ["回答", "回答です"]
    assert response.content == "回答です"
    assert response.state is MessageState.COMPLETED


@pytest.mark.asyncio
async def test_blocks_before_persisting_when_cloud_is_enabled(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id = await make_conversation(repository)
    service = ChatService(repository, FakeRegistry(False), FreeOperationPolicy())

    with pytest.raises(FreeOperationBlocked):
        await service.send_message(conversation_id, "保存されない質問")

    assert await repository.list_active_messages(conversation_id) == []


@pytest.mark.asyncio
async def test_translation_enqueue_failure_does_not_fail_chat(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id = await make_conversation(repository)
    service = ChatService(
        repository,
        FakeRegistry(),
        FreeOperationPolicy(),
        FailingTranslationScheduler(),
    )

    response = await service.send_message(conversation_id, "質問")

    assert response.content == "回答です"
    assert response.state is MessageState.COMPLETED


@pytest.mark.asyncio
async def test_selected_rag_document_is_injected_and_cited_without_changing_message(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id = await make_conversation(repository)
    rag = RagService(repository)
    document = await rag.register_document(
        "nutrition.md", "質問 ブロッコリーにはビタミンCが含まれます。".encode()
    )
    await rag.select_documents(conversation_id, [document.id])
    registry = FakeRegistry()
    service = ChatService(
        repository,
        registry,
        FreeOperationPolicy(),
        rag=rag,
    )

    response = await service.send_message(conversation_id, "質問")

    assert registry.provider.last_request is not None
    assert "nutrition.md" in registry.provider.last_request.system_prompt
    assert "ビタミンC" in registry.provider.last_request.system_prompt
    messages = await repository.list_active_messages(conversation_id)
    assert messages[-2].content == "質問"
    citations = await rag.message_citations([response.id])
    assert citations[response.id][0].document_title == "nutrition.md"
    assert "ビタミンC" in citations[response.id][0].content
    usage = await rag.message_usage([response.id])
    assert usage[response.id].selected_document_count == 1
    assert len(usage[response.id].citations) == 1


@pytest.mark.asyncio
async def test_selected_rag_document_records_no_match_for_answer(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id = await make_conversation(repository)
    rag = RagService(repository)
    document = await rag.register_document(
        "nutrition.md", "ブロッコリーにはビタミンCが含まれます。".encode()
    )
    await rag.select_documents(conversation_id, [document.id])
    service = ChatService(
        repository,
        FakeRegistry(),
        FreeOperationPolicy(),
        rag=rag,
    )

    response = await service.send_message(conversation_id, "質問")

    usage = await rag.message_usage([response.id])
    assert usage[response.id].selected_document_count == 1
    assert usage[response.id].citations == ()


@pytest.mark.asyncio
async def test_tool_call_is_executed_then_final_answer_shows_only_read_count(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id = await make_conversation(repository)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    await repository.set_tool_folder_grant(conversation_id, allowed)
    provider = ToolCallingProvider()
    coordinator = ToolCoordinator(repository, (FakeToolProvider(),))
    service = ChatService(
        repository,
        ToolRegistry(provider),
        FreeOperationPolicy(),
        tools=coordinator,
    )

    response = await service.send_message(conversation_id, "質問")

    assert response.content == "資料に基づく回答\n\n（資料を1件読み取りました）"
    assert "tool private body" not in response.content
    [audit] = await repository.list_tool_calls(conversation_id)
    assert audit.result_content is None
    assert provider.requests[-1].messages[-1].content == "tool private body"


@pytest.mark.asyncio
async def test_search_result_can_trigger_read_tool_before_final_answer(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id = await make_conversation(repository)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    await repository.set_tool_folder_grant(conversation_id, allowed)
    provider = SequentialToolCallingProvider()
    coordinator = ToolCoordinator(repository, (SequentialToolProvider(),))
    service = ChatService(
        repository,
        ToolRegistry(provider),
        FreeOperationPolicy(),
        tools=coordinator,
    )

    response = await service.send_message(conversation_id, "質問")

    assert response.content == "LOCAL-TOOLS-OK-731\n\n（資料を1件読み取りました）"
    assert len(provider.requests) == 3
    assert provider.requests[1].tools
    assert provider.requests[2].messages[-1].tool_name == "read_allowed_text"
    audits = await repository.list_tool_calls(conversation_id)
    assert [audit.tool_name for audit in audits] == [
        "search_allowed_folder",
        "read_allowed_text",
    ]


@pytest.mark.asyncio
async def test_tool_loop_stops_after_three_calls_and_generates_final_answer(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id = await make_conversation(repository)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    await repository.set_tool_folder_grant(conversation_id, allowed)
    provider = RepeatedToolCallingProvider()
    service = ChatService(
        repository,
        ToolRegistry(provider),
        FreeOperationPolicy(),
        tools=ToolCoordinator(repository, (FakeToolProvider(),)),
    )

    response = await service.send_message(conversation_id, "質問")

    assert response.content == "上限後の通常回答\n\n（資料を3件読み取りました）"
    assert len(provider.requests) == 4
    assert provider.requests[-1].tools == ()
    audits = await repository.list_tool_calls(conversation_id)
    assert len(audits) == 3


@pytest.mark.asyncio
async def test_tool_unsupported_model_retries_as_normal_chat(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id = await make_conversation(repository)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    await repository.set_tool_folder_grant(conversation_id, allowed)
    provider = ToolCallingProvider(unsupported=True)
    service = ChatService(
        repository,
        ToolRegistry(provider),
        FreeOperationPolicy(),
        tools=ToolCoordinator(repository, (FakeToolProvider(),)),
    )

    response = await service.send_message(conversation_id, "質問")

    assert response.content == "通常回答"
    assert len(provider.requests) == 2
    assert provider.requests[0].tools
    assert provider.requests[1].tools == ()

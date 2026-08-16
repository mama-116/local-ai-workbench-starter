from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID, uuid5

from local_llm_chat.application.services.memory_candidate_service import (
    MAX_MEMORY_SOURCE_CHARACTERS,
    MemoryCandidateService,
)
from local_llm_chat.domain.canonical_memory import (
    CanonicalMemoryAttribute,
    CanonicalMemoryEvent,
)
from local_llm_chat.domain.errors import OllamaUnavailable, ValidationError
from local_llm_chat.domain.memory_candidates import MemoryCandidateRequest
from local_llm_chat.domain.models import Message
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import (
    MemoryApprovalState,
    MemoryCandidateDisposition,
    MemoryCardinality,
    MemoryKind,
)


MAX_EXPLICIT_MEMORY_SOURCES = 8
MAX_EXPLICIT_MEMORY_SOURCE_CHARACTERS = 16_000
MAX_EXPLICIT_MEMORY_VALUE_CHARACTERS = 200
_EXPLICIT_MEMORY_NAMESPACE = UUID("61e410d4-2e16-4989-bbea-461570fb77a6")
_PREFERENCE_CONDITION_PATTERN = re.compile(
    r"(?P<condition>(?:ちょっと|少し|やや|かなり|完全に)?\s*"
    r"(?:溶け(?:かけ|た|ている)|冷え(?:かけ|た|ている)|温め(?:た|ている)"
    r"|焼き(?:たて|かけ)|凍(?:った|らせた)|熱々|あつあつ|ひえひえ"
    r"|冷たい|温かい|ぬるい|常温)(?:状態|もの|方|の)?)"
    r"\s*が好き"
)
_DIRECT_PREFERENCE_PATTERN = re.compile(
    r"(?P<item>[^。！？!?\r\n]{1,160}?)\s*が好き"
)
_CONVERSATIONAL_ITEM_PREFIX = re.compile(
    r"^(?:ねえ[、,]?\s*|先輩[、,]?\s*)?"
    r"(?:私は|私(?:は)?実は|自分は)\s*"
)


@dataclass(frozen=True, slots=True)
class ExplicitMemoryKnowledgeCharacter:
    character_id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class ExplicitMemoryDraft:
    source_messages: tuple[Message, ...]
    selected_source_message_ids: tuple[str, ...]
    knowledge_character_options: tuple[ExplicitMemoryKnowledgeCharacter, ...]
    auto_known_by_character_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class ExplicitMemorySuggestion:
    item: str
    condition: str | None
    summary: str
    item_source_message_id: str | None
    condition_source_message_id: str | None
    extractor_unavailable: bool = False


@dataclass(frozen=True, slots=True)
class ExplicitMemorySaveRequest:
    request_id: str
    conversation_id: str
    branch_id: str
    source_message_ids: tuple[str, ...]
    item: str
    condition: str | None
    kind: MemoryKind
    known_by_character_ids: frozenset[str]


class ExplicitMemoryService:
    """Builds a concise memory from source-backed fields and saves it."""

    def __init__(
        self,
        repository: AppRepository,
        candidate_service: MemoryCandidateService | None = None,
    ) -> None:
        self._repository = repository
        self._candidate_service = candidate_service

    async def prepare(
        self,
        conversation_id: str,
        branch_id: str,
        selected_source_message_id: str,
    ) -> ExplicitMemoryDraft:
        messages = await self._repository.list_memory_source_messages(
            conversation_id,
            branch_id,
            selected_source_message_id,
            MAX_EXPLICIT_MEMORY_SOURCES,
        )
        selected = next(
            (
                message
                for message in messages
                if message.id == selected_source_message_id
            ),
            None,
        )
        if selected is None:
            raise ValidationError("選択した発言を現在の分岐で確認できません。")
        characters = await self._repository.list_character_versions()
        listeners = await self._repository.get_memory_source_listener_character_ids(
            conversation_id, selected.id
        )
        available_ids = frozenset(
            character.character_id for character in characters
        )
        return ExplicitMemoryDraft(
            source_messages=messages,
            selected_source_message_ids=(selected.id,),
            knowledge_character_options=tuple(
                ExplicitMemoryKnowledgeCharacter(
                    character.character_id, character.display_name
                )
                for character in characters
            ),
            auto_known_by_character_ids=listeners.intersection(available_ids),
        )

    async def suggest(
        self,
        conversation_id: str,
        branch_id: str,
        source_message_ids: tuple[str, ...],
        kind: MemoryKind,
        character_id: str,
        *,
        use_model: bool = True,
    ) -> ExplicitMemorySuggestion:
        messages = await self._repository.get_memory_source_messages(
            conversation_id, branch_id, source_message_ids
        )
        self._require_source_limits(messages)
        item = ""
        item_source_id: str | None = None
        extractor_unavailable = self._candidate_service is None

        if self._candidate_service is not None and use_model:
            combined_content = "\n".join(message.content for message in messages)
            ranked: list[tuple[int, int, str, str]] = []
            if len(combined_content) <= MAX_MEMORY_SOURCE_CHARACTERS:
                try:
                    generation = await self._candidate_service.generate_with_status(
                        MemoryCandidateRequest(
                            conversation_id=conversation_id,
                            branch_id=branch_id,
                            source_message_id=messages[-1].id,
                            model_name="configured-memory-extraction",
                            content=combined_content,
                            author_subject_id="user",
                            allowed_subject_ids=frozenset({"user"}),
                            allowed_knowledge_character_ids=frozenset(
                                {character_id}
                            ),
                        )
                    )
                except OllamaUnavailable:
                    extractor_unavailable = True
                else:
                    extractor_unavailable = generation.extractor_unavailable
                    for candidate in generation.candidates:
                        source_match = next(
                            (
                                (index, message)
                                for index, message in reversed(
                                    tuple(enumerate(messages))
                                )
                                if candidate.value in message.content
                            ),
                            None,
                        )
                        if (
                            candidate.kind is kind
                            and candidate.subject_id == "user"
                            and candidate.disposition
                            is not MemoryCandidateDisposition.BLOCK
                            and source_match is not None
                        ):
                            index, message = source_match
                            slot_priority = (
                                1 if candidate.slot == "favorite_food" else 0
                            )
                            ranked.append(
                                (
                                    index,
                                    slot_priority,
                                    candidate.value,
                                    message.id,
                                )
                            )
            else:
                extractor_unavailable = True
            if ranked:
                _, _, item, item_source_id = max(
                    ranked, key=lambda row: (row[0], row[1], len(row[2]))
                )
        if kind is MemoryKind.PREFERENCE and not item:
            for message in messages:
                direct = _DIRECT_PREFERENCE_PATTERN.search(message.content)
                if direct is None:
                    continue
                possible_item = _CONVERSATIONAL_ITEM_PREFIX.sub(
                    "", direct.group("item").strip()
                ).strip()
                if (
                    possible_item
                    and _PREFERENCE_CONDITION_PATTERN.fullmatch(direct.group(0))
                    is None
                    and possible_item in message.content
                ):
                    item = possible_item
                    item_source_id = message.id

        condition = None
        condition_source_id = None
        if kind is MemoryKind.PREFERENCE:
            for message in messages:
                match = _PREFERENCE_CONDITION_PATTERN.search(message.content)
                if match is not None:
                    condition = match.group("condition").strip()
                    condition_source_id = message.id

        summary = self.compose_summary(kind, item, condition) if item else ""
        return ExplicitMemorySuggestion(
            item=item,
            condition=condition,
            summary=summary,
            item_source_message_id=item_source_id,
            condition_source_message_id=condition_source_id,
            extractor_unavailable=extractor_unavailable,
        )

    async def save(self, request: ExplicitMemorySaveRequest) -> str:
        self._validate_request(request)
        registered_character_ids = frozenset(
            character.character_id
            for character in await self._repository.list_character_versions()
        )
        if not request.known_by_character_ids:
            raise ValidationError(
                "at least one character must know the saved memory"
            )
        if not request.known_by_character_ids.issubset(
            registered_character_ids
        ):
            raise ValidationError(
                "memory knowledge scope contains an unregistered character"
            )
        messages = await self._repository.get_memory_source_messages(
            request.conversation_id,
            request.branch_id,
            request.source_message_ids,
        )
        self._require_source_limits(messages)
        item = request.item.strip()
        condition = request.condition.strip() if request.condition else None
        item_source_id = self._require_grounded_field("記憶の中心", item, messages)
        condition_source_id = (
            self._require_grounded_field("好みの状態", condition, messages)
            if condition
            else None
        )
        summary = self.compose_summary(request.kind, item, condition)

        primary = messages[-1]
        slot = {
            MemoryKind.PREFERENCE: "liked_food",
            MemoryKind.GOAL: "explicit_goal",
            MemoryKind.SAFETY_CONSTRAINT: "explicit_safety",
        }[request.kind]
        identity = "\0".join((request.conversation_id, request.request_id))
        attributes = [
            CanonicalMemoryAttribute("item", item, item_source_id),
        ]
        if condition is not None and condition_source_id is not None:
            attributes.append(
                CanonicalMemoryAttribute(
                    "condition", condition, condition_source_id
                )
            )
        event = CanonicalMemoryEvent(
            id=str(uuid5(_EXPLICIT_MEMORY_NAMESPACE, identity)),
            conversation_id=request.conversation_id,
            branch_id=request.branch_id,
            subject_id="user",
            kind=request.kind,
            slot=slot,
            value=summary,
            cardinality=MemoryCardinality.MULTIPLE,
            approval=MemoryApprovalState.CONFIRMED,
            source_message_id=primary.id,
            known_by_character_ids=request.known_by_character_ids,
            supersedes_event_id=None,
            effective_at=primary.created_at,
            recorded_at=primary.created_at,
            source_message_ids=tuple(message.id for message in messages),
            attributes=tuple(sorted(attributes, key=lambda attribute: attribute.key)),
        )
        await self._repository.append_canonical_memory_event(event)
        return event.id

    @staticmethod
    def compose_summary(
        kind: MemoryKind, item: str, condition: str | None
    ) -> str:
        clean_item = item.strip()
        clean_condition = condition.strip() if condition else None
        if not clean_item:
            raise ValidationError("記憶の中心となる内容を入力してください。")
        if kind is MemoryKind.PREFERENCE:
            summary = (
                f"{clean_item}は、{clean_condition}が好き"
                if clean_condition
                else f"{clean_item}が好き"
            )
        elif kind is MemoryKind.GOAL:
            if clean_condition:
                raise ValidationError("目標には好みの状態を指定できません。")
            summary = f"目標：{clean_item}"
        elif kind is MemoryKind.SAFETY_CONSTRAINT:
            if clean_condition:
                raise ValidationError("健康・安全情報には好みの状態を指定できません。")
            summary = f"注意事項：{clean_item}"
        else:
            raise ValidationError("選択した記憶の種類は保存できません。")
        if len(summary) > MAX_EXPLICIT_MEMORY_VALUE_CHARACTERS:
            raise ValidationError("整えた記憶文は200文字までです。")
        return summary

    @staticmethod
    def _validate_request(request: ExplicitMemorySaveRequest) -> None:
        required = (
            request.request_id,
            request.conversation_id,
            request.branch_id,
        )
        if any(not value.strip() for value in required):
            raise ValidationError("記憶保存の要求情報が不足しています。")
        if not 1 <= len(request.source_message_ids) <= MAX_EXPLICIT_MEMORY_SOURCES:
            raise ValidationError("記憶の根拠は1件から8件まで選択できます。")
        if (
            len(request.source_message_ids) != len(set(request.source_message_ids))
            or any(not source_id.strip() for source_id in request.source_message_ids)
        ):
            raise ValidationError("記憶の根拠発言が重複または不正です。")
        if not request.item.strip():
            raise ValidationError("記憶の中心となる内容を入力してください。")
        if request.condition and request.kind is not MemoryKind.PREFERENCE:
            raise ValidationError("好みの状態は好み・プロフィールでだけ使えます。")

    @staticmethod
    def _require_source_limits(messages: tuple[Message, ...]) -> None:
        if sum(len(message.content) for message in messages) > (
            MAX_EXPLICIT_MEMORY_SOURCE_CHARACTERS
        ):
            raise ValidationError("選択した発言の合計は16,000文字までです。")

    @staticmethod
    def _require_grounded_field(
        label: str, value: str, messages: tuple[Message, ...]
    ) -> str:
        source = next(
            (message for message in messages if value in message.content),
            None,
        )
        if source is None:
            raise ValidationError(
                f"{label}は選択した発言の原文にある語句を使ってください。"
            )
        return source.id

from dataclasses import dataclass, replace

import pytest

from local_llm_chat.application.services.memory_candidate_service import (
    DEFAULT_MEMORY_TEMPLATES,
    MemoryCandidateService,
)
from local_llm_chat.domain.errors import (
    FreeOperationBlocked,
    OllamaUnavailable,
    ValidationError,
)
from local_llm_chat.domain.memory_candidates import (
    MemoryCandidateDraft,
    MemoryCandidateRequest,
)
from local_llm_chat.domain.models import ProviderMetadata
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.states import (
    CostClass,
    Locality,
    MemoryApprovalState,
    MemoryCandidateDisposition,
    MemoryCandidateReason,
    MemoryEvidenceMode,
    MemoryKind,
)


@dataclass
class FakeExtractor:
    drafts: tuple[MemoryCandidateDraft, ...]
    provider_metadata: ProviderMetadata = ProviderMetadata(
        "memory-local",
        Locality.LOCAL,
        CostClass.NO_CHARGE,
        "http://127.0.0.1:11434",
    )
    disabled: bool = True
    calls: int = 0
    error: Exception | None = None

    @property
    def metadata(self) -> ProviderMetadata:
        return self.provider_metadata

    @property
    def cloud_is_disabled(self) -> bool:
        return self.disabled

    async def extract(
        self, request: MemoryCandidateRequest
    ) -> tuple[MemoryCandidateDraft, ...]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.drafts


def span(content: str, evidence: str) -> tuple[int, int]:
    start = content.index(evidence)
    return start, start + len(evidence)


def request(
    content: str, *, known_by: frozenset[str] = frozenset()
) -> MemoryCandidateRequest:
    return MemoryCandidateRequest(
        conversation_id="conversation-1",
        branch_id="branch-1",
        source_message_id="message-1",
        model_name="conversation-model",
        content=content,
        author_subject_id="user",
        allowed_subject_ids=frozenset({"user", "character-alice", "character-tanaka"}),
        allowed_knowledge_character_ids=frozenset(
            {"character-alice", "character-tanaka"}
        ),
        known_by_character_ids=known_by,
    )


@pytest.mark.asyncio
async def test_explicit_registered_form_survives_unavailable_local_extractor() -> None:
    extractor = FakeExtractor(
        (), error=OllamaUnavailable("PRIVATE endpoint detail")
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    generation = await service.generate_with_status(
        request("アイスはチョコ味が好き！")
    )
    candidates = generation.candidates

    assert extractor.calls == 1
    assert len(candidates) == 1
    assert candidates[0].slot == "liked_food"
    assert candidates[0].value == "チョコ味"
    assert candidates[0].disposition is MemoryCandidateDisposition.AUTO_SAVE
    assert generation.extractor_unavailable is True


@pytest.mark.asyncio
async def test_unavailable_local_extractor_still_fails_for_unregistered_form() -> None:
    extractor = FakeExtractor(
        (), error=OllamaUnavailable("PRIVATE endpoint detail")
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    with pytest.raises(OllamaUnavailable, match="PRIVATE endpoint detail"):
        await service.generate(request("今日は楽しかった。"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    (
        "ちょっと溶けかけが好き！",
        "少し溶けたのが好き",
        "焼きたてが好き",
        "熱々が好き",
        "冷たいのが好き",
    ),
)
async def test_condition_without_food_target_is_never_auto_saved_as_food(
    content: str,
) -> None:
    extractor = FakeExtractor((), error=OllamaUnavailable("unavailable"))
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    with pytest.raises(OllamaUnavailable):
        await service.generate(request(content))


@pytest.mark.asyncio
async def test_model_condition_without_food_target_is_blocked() -> None:
    content = "ちょっと溶けかけが好き！"
    evidence = "ちょっと溶けかけが好き"
    extractor = FakeExtractor(
        (
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "liked_food",
                "ちょっと溶けかけ",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, evidence),
            ),
        )
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidate = (await service.generate(request(content)))[0]

    assert candidate.disposition is MemoryCandidateDisposition.BLOCK
    assert candidate.reason is MemoryCandidateReason.INVALID_VALUE


@pytest.mark.asyncio
async def test_condition_with_explicit_food_target_remains_supported() -> None:
    extractor = FakeExtractor(())
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidate = (
        await service.generate(request("溶けかけのアイスが好き"))
    )[0]

    assert candidate.value == "溶けかけのアイス"
    assert candidate.disposition is MemoryCandidateDisposition.AUTO_SAVE


@pytest.mark.asyncio
async def test_unavailable_extractor_fallback_keeps_eight_candidate_limit() -> None:
    extractor = FakeExtractor((), error=OllamaUnavailable("unavailable"))
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )
    content = "。".join(
        f"アイスはフレーバー{index}が好き" for index in range(9)
    )

    candidates = await service.generate(request(content))

    assert len(candidates) == 8


@pytest.mark.asyncio
async def test_classifies_explicit_sensitive_and_unsafe_candidates() -> None:
    content = (
        "今一番好きなのはチョコアイス。"
        "ストロベリーアレルギーです。"
        "田中はたぶん猫が好き。"
        "『私は寿司が好き』と彼が言った。"
        "誰かはラーメンが好き。"
    )
    favorite = "今一番好きなのはチョコアイス"
    allergy = "ストロベリーアレルギーです"
    inferred = "田中はたぶん猫が好き"
    quoted = "私は寿司が好き"
    ambiguous = "誰かはラーメンが好き"
    extractor = FakeExtractor(
        (
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "favorite_food",
                "チョコアイス",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, favorite),
            ),
            MemoryCandidateDraft(
                "user",
                MemoryKind.SAFETY_CONSTRAINT,
                "food_allergy",
                "ストロベリー",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, allergy),
            ),
            MemoryCandidateDraft(
                "character-tanaka",
                MemoryKind.PREFERENCE,
                "liked_food",
                "猫",
                MemoryEvidenceMode.INFERRED,
                *span(content, inferred),
            ),
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "liked_food",
                "寿司",
                MemoryEvidenceMode.QUOTED,
                *span(content, quoted),
            ),
            MemoryCandidateDraft(
                None,
                MemoryKind.PREFERENCE,
                "liked_food",
                "ラーメン",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, ambiguous),
            ),
        )
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidates = await service.generate(request(content))

    assert [candidate.disposition for candidate in candidates] == [
        MemoryCandidateDisposition.AUTO_SAVE,
        MemoryCandidateDisposition.REQUIRE_CONFIRMATION,
        MemoryCandidateDisposition.BLOCK,
        MemoryCandidateDisposition.BLOCK,
        MemoryCandidateDisposition.BLOCK,
    ]
    assert [candidate.initial_approval for candidate in candidates] == [
        MemoryApprovalState.AUTO_SAVED,
        MemoryApprovalState.PENDING_CONFIRMATION,
        None,
        None,
        None,
    ]
    assert candidates[2].reason is MemoryCandidateReason.NON_EXPLICIT_EVIDENCE
    assert candidates[3].reason is MemoryCandidateReason.NON_EXPLICIT_EVIDENCE
    assert candidates[4].reason is MemoryCandidateReason.SUBJECT_UNKNOWN
    assert candidates[0].conversation_id == "conversation-1"
    assert candidates[0].branch_id == "branch-1"
    assert candidates[0].source_message_id == "message-1"


@pytest.mark.asyncio
async def test_blocks_value_not_grounded_in_the_claimed_evidence() -> None:
    content = "今一番好きなのはチョコアイス。"
    evidence = "今一番好きなのはチョコアイス"
    extractor = FakeExtractor(
        (
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "favorite_food",
                "バニラアイス",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, evidence),
            ),
        )
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidate = (await service.generate(request(content)))[0]

    assert candidate.disposition is MemoryCandidateDisposition.BLOCK
    assert candidate.reason is MemoryCandidateReason.EVIDENCE_MISMATCH
    assert candidate.initial_approval is None


@pytest.mark.asyncio
async def test_secret_scope_upgrades_low_risk_candidate_to_confirmation() -> None:
    content = "好きな料理はカレー。"
    evidence = "好きな料理はカレー"
    extractor = FakeExtractor(
        (
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "favorite_food",
                "カレー",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, evidence),
            ),
        )
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidate = (
        await service.generate(
            request(content, known_by=frozenset({"character-alice"}))
        )
    )[0]

    assert candidate.disposition is MemoryCandidateDisposition.REQUIRE_CONFIRMATION
    assert candidate.reason is MemoryCandidateReason.RESTRICTED_KNOWLEDGE_SCOPE
    assert candidate.initial_approval is MemoryApprovalState.PENDING_CONFIRMATION
    assert candidate.known_by_character_ids == frozenset({"character-alice"})


@pytest.mark.asyncio
async def test_rejects_remote_extractor_before_sending_private_message() -> None:
    extractor = FakeExtractor(
        (),
        ProviderMetadata(
            "memory-remote",
            Locality.REMOTE,
            CostClass.NO_CHARGE,
            "https://example.com",
        ),
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    with pytest.raises(FreeOperationBlocked):
        await service.generate(request("好きな料理はカレー。"))

    assert extractor.calls == 0


@pytest.mark.asyncio
async def test_allows_confirmed_private_lan_extractor() -> None:
    extractor = FakeExtractor(
        (),
        ProviderMetadata(
            "memory-lan",
            Locality.LOCAL,
            CostClass.NO_CHARGE,
            "http://192.168.1.17:11434",
        ),
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    await service.generate(request("好きな料理はカレー。"))

    assert extractor.calls == 1


@pytest.mark.asyncio
async def test_rejects_unconfirmed_private_lan_extractor_before_send() -> None:
    extractor = FakeExtractor(
        (),
        ProviderMetadata(
            "memory-lan",
            Locality.LOCAL,
            CostClass.NO_CHARGE,
            "http://192.168.1.17:11434",
        ),
        disabled=False,
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    with pytest.raises(FreeOperationBlocked):
        await service.generate(request("好きな料理はカレー。"))

    assert extractor.calls == 0


@pytest.mark.asyncio
async def test_rejects_untrusted_scope_and_excessive_candidate_batch() -> None:
    content = "好きな料理はカレー。"
    evidence = "好きな料理はカレー"
    draft = MemoryCandidateDraft(
        "user",
        MemoryKind.PREFERENCE,
        "favorite_food",
        "カレー",
        MemoryEvidenceMode.EXPLICIT,
        *span(content, evidence),
    )
    extractor = FakeExtractor((draft,) * 9)
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    with pytest.raises(ValidationError, match="8件"):
        await service.generate(request(content))

    invalid_scope = request(
        content, known_by=frozenset({"character-not-in-conversation"})
    )
    with pytest.raises(ValidationError, match="共有先"):
        await service.generate(invalid_scope)

    with pytest.raises(ValidationError):
        await service.generate(replace(request(content), model_name=" "))


@pytest.mark.asyncio
async def test_blocks_past_or_quoted_claim_even_when_extractor_marks_explicit() -> None:
    content = (
        "前はバニラが好きだった。"
        "『私は寿司が好き』と彼が言った。"
        "カレーが好きじゃない。"
        "ラーメンが好きだろう。"
    )
    past = "前はバニラが好きだった"
    quoted = "私は寿司が好き"
    negated = "カレーが好きじゃない"
    uncertain = "ラーメンが好きだろう"
    extractor = FakeExtractor(
        (
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "liked_food",
                "バニラ",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, past),
            ),
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "liked_food",
                "寿司",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, quoted),
            ),
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "liked_food",
                "カレー",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, negated),
            ),
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "liked_food",
                "ラーメン",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, uncertain),
            ),
        )
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidates = await service.generate(request(content))

    assert [candidate.disposition for candidate in candidates] == [
        MemoryCandidateDisposition.BLOCK,
        MemoryCandidateDisposition.BLOCK,
        MemoryCandidateDisposition.BLOCK,
        MemoryCandidateDisposition.BLOCK,
    ]
    assert [candidate.reason for candidate in candidates] == [
        MemoryCandidateReason.UNSAFE_EVIDENCE,
        MemoryCandidateReason.UNSAFE_EVIDENCE,
        MemoryCandidateReason.UNSAFE_EVIDENCE,
        MemoryCandidateReason.UNSAFE_EVIDENCE,
    ]


@pytest.mark.asyncio
async def test_third_party_low_risk_fact_requires_confirmation() -> None:
    content = "田中はカレーが好き。"
    evidence = "田中はカレーが好き"
    extractor = FakeExtractor(
        (
            MemoryCandidateDraft(
                "character-tanaka",
                MemoryKind.PREFERENCE,
                "liked_food",
                "カレー",
                MemoryEvidenceMode.EXPLICIT,
                *span(content, evidence),
            ),
        )
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidate = (await service.generate(request(content)))[0]

    assert candidate.disposition is MemoryCandidateDisposition.REQUIRE_CONFIRMATION
    assert candidate.reason is MemoryCandidateReason.THIRD_PARTY_SUBJECT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "kind", "slot", "value", "disposition"),
    (
        (
            "チョコが好き",
            MemoryKind.PREFERENCE,
            "liked_food",
            "チョコ",
            MemoryCandidateDisposition.AUTO_SAVE,
        ),
        (
            "苺アレルギー",
            MemoryKind.SAFETY_CONSTRAINT,
            "food_allergy",
            "苺",
            MemoryCandidateDisposition.REQUIRE_CONFIRMATION,
        ),
        (
            "一番好きなのはチョコ",
            MemoryKind.PREFERENCE,
            "favorite_food",
            "チョコ",
            MemoryCandidateDisposition.AUTO_SAVE,
        ),
    ),
)
async def test_falls_back_to_registered_explicit_form_when_extractor_returns_empty(
    content: str,
    kind: MemoryKind,
    slot: str,
    value: str,
    disposition: MemoryCandidateDisposition,
) -> None:
    extractor = FakeExtractor(())
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidates = await service.generate(request(content))

    assert extractor.calls == 1
    assert len(candidates) == 1
    candidate = candidates[0]
    assert (candidate.kind, candidate.slot, candidate.value) == (kind, slot, value)
    assert candidate.evidence_text == content
    assert candidate.disposition is disposition


@pytest.mark.asyncio
async def test_empty_extractor_fallback_supports_multiple_explicit_sentences() -> None:
    content = "チョコが好き。バニラが好き。"
    extractor = FakeExtractor(())
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidates = await service.generate(request(content))

    assert extractor.calls == 1
    assert [candidate.value for candidate in candidates] == ["チョコ", "バニラ"]
    assert all(
        candidate.disposition is MemoryCandidateDisposition.AUTO_SAVE
        for candidate in candidates
    )


@pytest.mark.asyncio
async def test_registered_form_replaces_overlapping_blocked_model_candidate() -> None:
    content = "チョコが好き"
    extractor = FakeExtractor(
        (
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "liked_food",
                "chocolate",
                MemoryEvidenceMode.EXPLICIT,
                0,
                5,
            ),
        )
    )
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidates = await service.generate(request(content))

    assert extractor.calls == 1
    assert len(candidates) == 1
    assert candidates[0].value == "チョコ"
    assert candidates[0].evidence_text == content
    assert candidates[0].disposition is MemoryCandidateDisposition.AUTO_SAVE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    (
        "前はチョコが好き",
        "チョコが好きではない",
        "『チョコが好き』",
        "チョコが好き？",
        "田中はチョコが好き",
        "妹は苺アレルギー",
        "田中: チョコが好き",
        "@田中: チョコが好き",
    ),
)
async def test_empty_extractor_fallback_does_not_save_unsafe_forms(
    content: str,
) -> None:
    extractor = FakeExtractor(())
    service = MemoryCandidateService(
        extractor, FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )

    candidates = await service.generate(request(content))

    assert extractor.calls == 1
    assert all(
        candidate.disposition is MemoryCandidateDisposition.BLOCK
        for candidate in candidates
    )

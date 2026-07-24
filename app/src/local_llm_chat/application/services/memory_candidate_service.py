from __future__ import annotations

import re
from dataclasses import dataclass

from local_llm_chat.domain.errors import OllamaUnavailable, ValidationError
from local_llm_chat.domain.memory_candidates import (
    MAX_MEMORY_CANDIDATES_PER_MESSAGE,
    MemoryCandidate,
    MemoryCandidateDraft,
    MemoryCandidateRequest,
    MemoryTemplate,
)
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.ports.memory_candidate_extractor import (
    MemoryCandidateExtractor,
)
from local_llm_chat.domain.states import (
    MemoryApprovalState,
    MemoryCandidateDisposition,
    MemoryCandidateReason,
    MemoryCardinality,
    MemoryEvidenceMode,
    MemoryKind,
)


MAX_MEMORY_VALUE_CHARACTERS = 200
MAX_MEMORY_SOURCE_CHARACTERS = 4_000
_MEMORY_CLAUSE_PATTERN = re.compile(r"[^。！？!?\r\n]+")
_FOOD_CONDITION_ONLY_PATTERN = re.compile(
    r"(?:ちょっと|少し|やや|かなり|完全に)?\s*"
    r"(?:"
    r"溶け(?:かけ|た|ている)"
    r"|冷え(?:かけ|た|ている)"
    r"|温め(?:た|ている)"
    r"|焼き(?:たて|かけ)"
    r"|凍(?:った|らせた)"
    r"|熱々|あつあつ|ひえひえ|冷たい|温かい|ぬるい|常温"
    r")"
    r"(?:状態|もの|方|の)?"
)


@dataclass(frozen=True, slots=True)
class MemoryCandidateGeneration:
    candidates: tuple[MemoryCandidate, ...]
    extractor_unavailable: bool = False

DEFAULT_MEMORY_TEMPLATES = (
    MemoryTemplate(
        MemoryKind.PREFERENCE,
        "favorite_food",
        MemoryCardinality.SINGLE,
        requires_confirmation=False,
        evidence_patterns=(
            r"一番好きなのは\s*{value}",
            r"好きな料理は\s*{value}",
            r"{value}\s*が一番好き",
        ),
    ),
    MemoryTemplate(
        MemoryKind.PREFERENCE,
        "liked_food",
        MemoryCardinality.MULTIPLE,
        requires_confirmation=False,
        evidence_patterns=(
            r"{value}\s*が好き",
            r"好きなのは\s*{value}",
            r"アイスは\s*{value}\s*が好き",
        ),
    ),
    MemoryTemplate(
        MemoryKind.SAFETY_CONSTRAINT,
        "food_allergy",
        MemoryCardinality.MULTIPLE,
        requires_confirmation=True,
        evidence_patterns=(
            r"{value}\s*アレルギー",
            r"{value}\s*にアレルギー",
        ),
    ),
    MemoryTemplate(
        MemoryKind.GOAL,
        "personal_goal",
        MemoryCardinality.MULTIPLE,
        requires_confirmation=True,
        evidence_patterns=(r"目標は\s*{value}", r"{value}\s*が目標"),
    ),
    MemoryTemplate(
        MemoryKind.GOAL,
        "club_membership_intent",
        MemoryCardinality.SINGLE,
        requires_confirmation=True,
        evidence_patterns=(r"部活を\s*{value}",),
    ),
)


class MemoryCandidateService:
    def __init__(
        self,
        extractor: MemoryCandidateExtractor,
        free_policy: FreeOperationPolicy,
        templates: tuple[MemoryTemplate, ...],
    ) -> None:
        self._extractor = extractor
        self._free_policy = free_policy
        self._templates: dict[tuple[MemoryKind, str], MemoryTemplate] = {}
        for template in templates:
            key = (template.kind, template.slot)
            if not template.slot.strip():
                raise ValueError("memory template slot must not be blank")
            if key in self._templates:
                raise ValueError("memory template kind and slot must be unique")
            if not template.evidence_patterns or any(
                "{value}" not in pattern for pattern in template.evidence_patterns
            ):
                raise ValueError("memory template evidence patterns must contain {value}")
            self._templates[key] = template

    async def generate(
        self, request: MemoryCandidateRequest
    ) -> tuple[MemoryCandidate, ...]:
        return (await self.generate_with_status(request)).candidates

    async def generate_with_status(
        self, request: MemoryCandidateRequest
    ) -> MemoryCandidateGeneration:
        self._validate_request(request)
        if not request.content.strip():
            return MemoryCandidateGeneration(())
        self._free_policy.require_cloud_disabled(self._extractor.cloud_is_disabled)
        self._free_policy.require_provider(self._extractor.metadata)
        try:
            drafts = await self._extractor.extract(request)
        except OllamaUnavailable:
            fallback_candidates = tuple(
                self._classify(request, draft)
                for draft in self._match_registered_explicit_forms(request)
            )[:MAX_MEMORY_CANDIDATES_PER_MESSAGE]
            if fallback_candidates:
                return MemoryCandidateGeneration(fallback_candidates, True)
            raise
        if len(drafts) > MAX_MEMORY_CANDIDATES_PER_MESSAGE:
            raise ValidationError("1発言の記憶候補は8件までです。")
        candidates = [self._classify(request, draft) for draft in drafts]
        deterministic = (
            self._classify(request, draft)
            for draft in self._match_registered_explicit_forms(request)
        )
        for fallback in deterministic:
            if fallback.disposition is MemoryCandidateDisposition.BLOCK:
                continue
            matching_indexes = [
                index
                for index, candidate in enumerate(candidates)
                if candidate.subject_id == fallback.subject_id
                and candidate.kind is fallback.kind
                and candidate.slot == fallback.slot
                and self._evidence_overlaps(candidate, fallback)
            ]
            if any(
                candidates[index].disposition is not MemoryCandidateDisposition.BLOCK
                for index in matching_indexes
            ):
                continue
            blocked_index = next(
                (
                    index
                    for index in matching_indexes
                    if candidates[index].disposition
                    is MemoryCandidateDisposition.BLOCK
                ),
                None,
            )
            if blocked_index is not None:
                candidates[blocked_index] = fallback
            elif len(candidates) < MAX_MEMORY_CANDIDATES_PER_MESSAGE:
                candidates.append(fallback)
        return MemoryCandidateGeneration(tuple(candidates))

    @staticmethod
    def _evidence_overlaps(
        candidate: MemoryCandidate, fallback: MemoryCandidate
    ) -> bool:
        return (
            candidate.evidence_start < fallback.evidence_end
            and fallback.evidence_start < candidate.evidence_end
        )

    def _match_registered_explicit_forms(
        self, request: MemoryCandidateRequest
    ) -> tuple[MemoryCandidateDraft, ...]:
        """Recover only full-clause forms already allowed by trusted templates."""
        drafts: list[MemoryCandidateDraft] = []
        for clause_match in _MEMORY_CLAUSE_PATTERN.finditer(request.content):
            terminator = request.content[clause_match.end() : clause_match.end() + 1]
            if terminator in {"?", "？"}:
                continue
            raw_clause = clause_match.group(0)
            clause = raw_clause.strip()
            if not clause:
                continue
            leading_space = len(raw_clause) - len(raw_clause.lstrip())
            evidence_start = clause_match.start() + leading_space
            evidence_end = evidence_start + len(clause)

            for template in self._templates.values():
                matched_value = self._fullmatch_template_value(template, clause)
                if matched_value is None:
                    continue
                drafts.append(
                    MemoryCandidateDraft(
                        subject_id=request.author_subject_id,
                        kind=template.kind,
                        slot=template.slot,
                        value=matched_value,
                        evidence_mode=MemoryEvidenceMode.EXPLICIT,
                        evidence_start=evidence_start,
                        evidence_end=evidence_end,
                    )
                )
                break
        return tuple(drafts)

    @staticmethod
    def _fullmatch_template_value(
        template: MemoryTemplate, evidence_text: str
    ) -> str | None:
        for pattern in template.evidence_patterns:
            capture_pattern = pattern.replace("{value}", r"(?P<value>.+?)", 1)
            match = re.fullmatch(capture_pattern, evidence_text)
            if match is None:
                continue
            value = match.group("value").strip()
            if (
                value
                and MemoryCandidateService._is_conservative_fallback_value(value)
                and MemoryCandidateService._is_semantically_valid_value(template, value)
            ):
                return value
        return None

    @staticmethod
    def _is_conservative_fallback_value(value: str) -> bool:
        ambiguous_subject_markers = (
            "は",
            "も",
            "って",
            "によると",
            "と言",
            ":",
            "：",
            "@",
            "＠",
            "、",
        )
        quote_markers = ("「", "」", "『", "』", '"', "'")
        return not any(
            marker in value for marker in ambiguous_subject_markers + quote_markers
        )

    @staticmethod
    def _is_semantically_valid_value(
        template: MemoryTemplate, value: str
    ) -> bool:
        if (
            template.kind is MemoryKind.PREFERENCE
            and template.slot in {"favorite_food", "liked_food"}
            and _FOOD_CONDITION_ONLY_PATTERN.fullmatch(value)
        ):
            return False
        return True

    def _classify(
        self, request: MemoryCandidateRequest, draft: MemoryCandidateDraft
    ) -> MemoryCandidate:
        evidence_text = self._evidence_text(request.content, draft)
        template = self._templates.get((draft.kind, draft.slot))
        value = draft.value.strip()

        if draft.subject_id not in request.allowed_subject_ids:
            return self._blocked(
                request, draft, value, evidence_text, template, MemoryCandidateReason.SUBJECT_UNKNOWN
            )
        if draft.evidence_mode is not MemoryEvidenceMode.EXPLICIT:
            return self._blocked(
                request,
                draft,
                value,
                evidence_text,
                template,
                MemoryCandidateReason.NON_EXPLICIT_EVIDENCE,
            )
        if not value or len(value) > MAX_MEMORY_VALUE_CHARACTERS:
            return self._blocked(
                request,
                draft,
                value,
                evidence_text,
                template,
                MemoryCandidateReason.INVALID_VALUE,
            )
        if (
            template is not None
            and not self._is_semantically_valid_value(template, value)
        ):
            return self._blocked(
                request,
                draft,
                value,
                evidence_text,
                template,
                MemoryCandidateReason.INVALID_VALUE,
            )
        if not evidence_text or value.casefold() not in evidence_text.casefold():
            return self._blocked(
                request,
                draft,
                value,
                evidence_text,
                template,
                MemoryCandidateReason.EVIDENCE_MISMATCH,
            )
        if template is None:
            return self._blocked(
                request,
                draft,
                value,
                evidence_text,
                template,
                MemoryCandidateReason.TEMPLATE_UNKNOWN,
            )
        if self._has_unsafe_evidence(request.content, draft, evidence_text):
            return self._blocked(
                request,
                draft,
                value,
                evidence_text,
                template,
                MemoryCandidateReason.UNSAFE_EVIDENCE,
            )
        if not self._matches_template_evidence(template, evidence_text, value):
            return self._blocked(
                request,
                draft,
                value,
                evidence_text,
                template,
                MemoryCandidateReason.UNSUPPORTED_EXPLICIT_FORM,
            )

        if template.requires_confirmation:
            return self._candidate(
                request,
                draft,
                value,
                evidence_text,
                template,
                MemoryCandidateDisposition.REQUIRE_CONFIRMATION,
                MemoryCandidateReason.TEMPLATE_REQUIRES_CONFIRMATION,
                MemoryApprovalState.PENDING_CONFIRMATION,
            )
        if request.known_by_character_ids:
            return self._candidate(
                request,
                draft,
                value,
                evidence_text,
                template,
                MemoryCandidateDisposition.REQUIRE_CONFIRMATION,
                MemoryCandidateReason.RESTRICTED_KNOWLEDGE_SCOPE,
                MemoryApprovalState.PENDING_CONFIRMATION,
            )
        if draft.subject_id != request.author_subject_id:
            return self._candidate(
                request,
                draft,
                value,
                evidence_text,
                template,
                MemoryCandidateDisposition.REQUIRE_CONFIRMATION,
                MemoryCandidateReason.THIRD_PARTY_SUBJECT,
                MemoryApprovalState.PENDING_CONFIRMATION,
            )
        return self._candidate(
            request,
            draft,
            value,
            evidence_text,
            template,
            MemoryCandidateDisposition.AUTO_SAVE,
            MemoryCandidateReason.EXPLICIT_LOW_RISK,
            MemoryApprovalState.AUTO_SAVED,
        )

    @staticmethod
    def _validate_request(request: MemoryCandidateRequest) -> None:
        identifiers = (
            request.conversation_id,
            request.branch_id,
            request.source_message_id,
            request.model_name,
            request.author_subject_id,
        )
        if any(not value.strip() for value in identifiers):
            raise ValidationError("記憶候補の会話・分岐・出典が必要です。")
        if len(request.content) > MAX_MEMORY_SOURCE_CHARACTERS:
            raise ValidationError("記憶候補の元発言が長すぎます。")
        if not request.allowed_subject_ids or any(
            not value.strip() for value in request.allowed_subject_ids
        ):
            raise ValidationError("記憶候補の対象人物が設定されていません。")
        if request.author_subject_id not in request.allowed_subject_ids:
            raise ValidationError("発言者が記憶対象の人物一覧に含まれていません。")
        if any(not value.strip() for value in request.allowed_knowledge_character_ids):
            raise ValidationError("記憶の共有先候補が不正です。")
        if not request.known_by_character_ids.issubset(
            request.allowed_knowledge_character_ids
        ):
            raise ValidationError("記憶の共有先が現在の会話に含まれていません。")

    @staticmethod
    def _evidence_text(content: str, draft: MemoryCandidateDraft) -> str:
        if (
            not isinstance(draft.evidence_start, int)
            or not isinstance(draft.evidence_end, int)
            or draft.evidence_start < 0
            or draft.evidence_end <= draft.evidence_start
            or draft.evidence_end > len(content)
        ):
            return ""
        return content[draft.evidence_start : draft.evidence_end]

    @staticmethod
    def _matches_template_evidence(
        template: MemoryTemplate, evidence_text: str, value: str
    ) -> bool:
        escaped_value = re.escape(value)
        return any(
            re.search(pattern.format(value=escaped_value), evidence_text)
            for pattern in template.evidence_patterns
        )

    @staticmethod
    def _has_unsafe_evidence(
        content: str, draft: MemoryCandidateDraft, evidence_text: str
    ) -> bool:
        unsafe_markers = (
            "たぶん",
            "おそらく",
            "かもしれ",
            "だろう",
            "もし",
            "仮に",
            "らしい",
            "ではなく",
            "じゃなく",
            "ではない",
            "じゃない",
            "好きではない",
            "前は",
            "前まで",
            "以前は",
            "以前",
            "昔は",
            "昔",
            "かつて",
            "好きだった",
        )
        if any(marker in evidence_text for marker in unsafe_markers):
            return True
        stripped = evidence_text.strip()
        quote_pairs = (("「", "」"), ("『", "』"), ('"', '"'), ("'", "'"))
        if any(
            stripped.startswith(opening) and stripped.endswith(closing)
            for opening, closing in quote_pairs
        ):
            return True
        prefix = content[: draft.evidence_start].rstrip()
        suffix = content[draft.evidence_end :].lstrip()
        attribution_prefix = prefix[-8:]
        attribution_suffix = suffix[:8]
        if "によると" in attribution_prefix or any(
            marker in attribution_suffix
            for marker in ("と言った", "って言った", "と言っていた")
        ):
            return True
        return any(
            prefix.endswith(opening) and suffix.startswith(closing)
            for opening, closing in quote_pairs
        )

    @staticmethod
    def _blocked(
        request: MemoryCandidateRequest,
        draft: MemoryCandidateDraft,
        value: str,
        evidence_text: str,
        template: MemoryTemplate | None,
        reason: MemoryCandidateReason,
    ) -> MemoryCandidate:
        return MemoryCandidateService._candidate(
            request,
            draft,
            value,
            evidence_text,
            template,
            MemoryCandidateDisposition.BLOCK,
            reason,
            None,
        )

    @staticmethod
    def _candidate(
        request: MemoryCandidateRequest,
        draft: MemoryCandidateDraft,
        value: str,
        evidence_text: str,
        template: MemoryTemplate | None,
        disposition: MemoryCandidateDisposition,
        reason: MemoryCandidateReason,
        initial_approval: MemoryApprovalState | None,
    ) -> MemoryCandidate:
        return MemoryCandidate(
            conversation_id=request.conversation_id,
            branch_id=request.branch_id,
            source_message_id=request.source_message_id,
            subject_id=draft.subject_id,
            kind=draft.kind,
            slot=draft.slot,
            value=value,
            cardinality=template.cardinality if template is not None else None,
            evidence_text=evidence_text,
            evidence_start=draft.evidence_start,
            evidence_end=draft.evidence_end,
            known_by_character_ids=request.known_by_character_ids,
            disposition=disposition,
            reason=reason,
            initial_approval=initial_approval,
        )

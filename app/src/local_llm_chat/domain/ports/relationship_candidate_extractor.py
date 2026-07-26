from __future__ import annotations

from typing import Protocol

from local_llm_chat.domain.models import ProviderMetadata
from local_llm_chat.domain.relationship_profile import (
    RelationshipCandidateDraft,
    RelationshipCandidateRequest,
)


class RelationshipCandidateExtractor(Protocol):
    @property
    def metadata(self) -> ProviderMetadata: ...

    @property
    def cloud_is_disabled(self) -> bool: ...

    async def extract(
        self, request: RelationshipCandidateRequest
    ) -> tuple[RelationshipCandidateDraft, ...]: ...

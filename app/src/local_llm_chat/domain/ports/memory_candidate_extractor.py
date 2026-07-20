from __future__ import annotations

from typing import Protocol

from local_llm_chat.domain.memory_candidates import (
    MemoryCandidateDraft,
    MemoryCandidateRequest,
)
from local_llm_chat.domain.models import ProviderMetadata


class MemoryCandidateExtractor(Protocol):
    @property
    def metadata(self) -> ProviderMetadata: ...

    @property
    def cloud_is_disabled(self) -> bool: ...

    async def extract(
        self, request: MemoryCandidateRequest
    ) -> tuple[MemoryCandidateDraft, ...]: ...

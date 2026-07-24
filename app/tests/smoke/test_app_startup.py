from pathlib import Path

import pytest

from local_llm_chat.application.services.memory_capture_service import (
    QueuedMemoryCaptureScheduler,
)
from local_llm_chat.application.services.chat_coordinator import ChatCoordinator
from local_llm_chat.application.services.memory_review_service import MemoryReviewService
from local_llm_chat.application.services.conversation_timeline_service import (
    ConversationTimelineService,
)
from local_llm_chat.application.services.conversation_group_configuration_service import (
    ConversationGroupConfigurationService,
)
from local_llm_chat.bootstrap import bootstrap
from local_llm_chat.infrastructure.llm.configured_memory_candidate_extractor import (
    ConfiguredMemoryCandidateExtractor,
)


@pytest.mark.asyncio
async def test_bootstrap_creates_local_database_and_default_character(
    tmp_path: Path,
) -> None:
    container = await bootstrap(tmp_path)
    try:
        assert container.paths.database_path.exists()
        characters = await container.profiles.list_characters()
        assert len(characters) == 1
        assert isinstance(container.memory_capture, QueuedMemoryCaptureScheduler)
        assert isinstance(container.memory_review, MemoryReviewService)
        assert isinstance(
            container.memory_extractor, ConfiguredMemoryCandidateExtractor
        )
        assert isinstance(container.chat, ChatCoordinator)
        assert isinstance(container.timeline, ConversationTimelineService)
        assert isinstance(
            container.group_configuration,
            ConversationGroupConfigurationService,
        )
        assert "言語指定がない場合" in characters[0].system_prompt
        assert "依頼文の言い換えだけで終わらせない" in characters[0].system_prompt
    finally:
        await container.close()

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from local_llm_chat.application.services.chat_coordinator import ChatCoordinator
from local_llm_chat.application.services.chat_service import ChatService
from local_llm_chat.application.services.computer_use_access_service import (
    ComputerUseAccessService,
)
from local_llm_chat.application.services.computer_use_service import (
    ComputerUseCoordinator,
)
from local_llm_chat.application.services.memory_candidate_service import (
    DEFAULT_MEMORY_TEMPLATES,
    MemoryCandidateService,
)
from local_llm_chat.application.services.memory_capture_service import (
    MemoryCaptureService,
    QueuedMemoryCaptureScheduler,
)
from local_llm_chat.application.services.memory_review_service import (
    MemoryReviewService,
)
from local_llm_chat.application.services.explicit_memory_service import (
    ExplicitMemoryService,
)
from local_llm_chat.application.services.builtin_tool_provider import (
    BuiltInToolProvider,
    ToolAccessService,
)
from local_llm_chat.application.services.tool_coordinator import ToolCoordinator
from local_llm_chat.application.services.conversation_service import ConversationService
from local_llm_chat.application.services.conversation_group_configuration_service import (
    ConversationGroupConfigurationService,
)
from local_llm_chat.application.services.conversation_timeline_service import (
    ConversationTimelineService,
)
from local_llm_chat.application.services.profile_service import ProfileService
from local_llm_chat.application.services.rag_service import RagService
from local_llm_chat.application.services.relationship_profile_service import (
    RelationshipProfileService,
)
from local_llm_chat.application.services.relationship_turn_reception_service import (
    RelationshipTurnReceptionService,
)
from local_llm_chat.application.services.restart_service import RestartService
from local_llm_chat.application.services.scheduler_service import SchedulerService
from local_llm_chat.application.services.translation_service import TranslationService
from local_llm_chat.application.services.telemetry_service import TelemetryService
from local_llm_chat.application.services.turn_batch_generation_service import (
    TurnBatchGenerationService,
)
from local_llm_chat.application.services.turn_batch_service import TurnBatchService
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.infrastructure.llm.ollama_registry import (
    LOCAL_ENDPOINT,
    LOCAL_PROVIDER_NAME,
    OllamaProviderRegistry,
)
from local_llm_chat.infrastructure.computer_use.fake_action_broker import (
    FakeDesktopActionBroker,
)
from local_llm_chat.infrastructure.llm.ollama_memory_candidate_extractor import (
    OllamaMemoryCandidateExtractor,
)
from local_llm_chat.infrastructure.llm.ollama_relationship_candidate_extractor import (
    OllamaRelationshipCandidateExtractor,
)
from local_llm_chat.infrastructure.llm.ollama_turn_batch_generator import (
    OllamaTurnBatchGenerator,
)
from local_llm_chat.infrastructure.mcp.profile import local_notes_profile
from local_llm_chat.infrastructure.mcp.tool_provider import TrustedMcpToolProvider
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)
from local_llm_chat.infrastructure.process_restart import SubprocessRestartLauncher
from local_llm_chat.infrastructure.settings import AppPaths
from local_llm_chat.infrastructure.settings import resolve_app_paths
from local_llm_chat.infrastructure.telemetry.system_collectors import (
    NvidiaSmiCollector,
    WindowsSystemCollector,
)


CloseCallback = Callable[[], Awaitable[None]]
DEFAULT_SERVICE_CLOSE_TIMEOUT_SECONDS = 2.0


async def _close_services_in_order(
    services: tuple[tuple[str, CloseCallback], ...],
    timeout_seconds: float = DEFAULT_SERVICE_CLOSE_TIMEOUT_SECONDS,
) -> None:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    errors: list[Exception] = []
    for name, close in services:
        try:
            await asyncio.wait_for(close(), timeout=timeout_seconds)
        except Exception as error:
            error.add_note(f"while closing {name}")
            errors.append(error)
    if errors:
        raise ExceptionGroup("application shutdown failed", errors)


@dataclass(slots=True)
class AppContainer:
    paths: AppPaths
    repository: SQLiteAppRepository
    providers: OllamaProviderRegistry
    profiles: ProfileService
    rag: RagService
    restart: RestartService
    conversations: ConversationService
    group_configuration: ConversationGroupConfigurationService
    timeline: ConversationTimelineService
    translations: TranslationService
    telemetry: TelemetryService
    memory_capture: QueuedMemoryCaptureScheduler
    memory_review: MemoryReviewService
    explicit_memory: ExplicitMemoryService
    relationship_profiles: RelationshipProfileService
    memory_extractor: OllamaMemoryCandidateExtractor
    relationship_extractor: OllamaRelationshipCandidateExtractor
    chat: ChatCoordinator
    tool_access: ToolAccessService
    scheduler: SchedulerService
    computer_use: ComputerUseAccessService

    async def close(self) -> None:
        await _close_services_in_order(
            (
                ("scheduler", self.scheduler.close),
                ("memory_capture", self.memory_capture.close),
                ("translations", self.translations.close),
                ("telemetry", self.telemetry.close),
                ("memory_extractor", self.memory_extractor.close),
                ("relationship_extractor", self.relationship_extractor.close),
                ("providers", self.providers.close),
            )
        )


async def bootstrap(data_dir: Path | None = None) -> AppContainer:
    paths = resolve_app_paths(data_dir)
    repository = SQLiteAppRepository(paths.database_path)
    policy = FreeOperationPolicy()
    providers = OllamaProviderRegistry(
        paths.ollama_connections_path,
        paths.ollama_server_config_path,
        policy,
    )
    translation_provider = providers.get(LOCAL_PROVIDER_NAME)

    await repository.initialize()
    await repository.recover_interrupted_runs()
    await repository.ensure_default_character()

    translations = TranslationService(
        repository, translation_provider, providers, policy
    )
    memory_extractor = OllamaMemoryCandidateExtractor(
        endpoint=LOCAL_ENDPOINT,
        cloud_is_disabled=lambda: providers.cloud_is_disabled(LOCAL_PROVIDER_NAME),
    )
    memory_candidates = MemoryCandidateService(
        memory_extractor, policy, DEFAULT_MEMORY_TEMPLATES
    )
    relationship_profiles = RelationshipProfileService(repository)
    relationship_extractor = OllamaRelationshipCandidateExtractor(
        endpoint=LOCAL_ENDPOINT,
        cloud_is_disabled=lambda: providers.cloud_is_disabled(LOCAL_PROVIDER_NAME),
    )
    relationship_turn_reception = RelationshipTurnReceptionService(
        repository, relationship_profiles, relationship_extractor
    )
    memory_capture = QueuedMemoryCaptureScheduler(
        MemoryCaptureService(
            repository,
            memory_candidates,
            relationship_profiles,
            relationship_extractor,
        ),
        repository,
    )
    telemetry = TelemetryService(
        repository, [WindowsSystemCollector(), NvidiaSmiCollector()]
    )
    rag = RagService(repository)
    tool_access = ToolAccessService(repository)
    tool_coordinator = ToolCoordinator(
        repository,
        (
            BuiltInToolProvider(repository),
            TrustedMcpToolProvider(repository, local_notes_profile()),
        ),
        on_change=tool_access.notify,
    )
    scheduler = SchedulerService(repository, {})
    scheduler.start()
    computer_use = ComputerUseAccessService(
        repository,
        ComputerUseCoordinator(FakeDesktopActionBroker(repository)),
    )
    single_chat = ChatService(
        repository,
        providers,
        policy,
        translations,
        telemetry,
        rag,
        tool_coordinator,
        memory_capture_scheduler=memory_capture,
            relationship_profiles=relationship_profiles,
            relationship_turn_reception=relationship_turn_reception,
    )
    return AppContainer(
        paths=paths,
        repository=repository,
        providers=providers,
        profiles=ProfileService(repository, providers, policy),
        rag=rag,
        restart=RestartService(paths.data_dir, SubprocessRestartLauncher()),
        conversations=ConversationService(repository, providers, policy),
        group_configuration=ConversationGroupConfigurationService(repository),
        timeline=ConversationTimelineService(repository),
        translations=translations,
        telemetry=telemetry,
        memory_capture=memory_capture,
        memory_review=MemoryReviewService(repository),
        explicit_memory=ExplicitMemoryService(repository),
        relationship_profiles=relationship_profiles,
        memory_extractor=memory_extractor,
        relationship_extractor=relationship_extractor,
        tool_access=tool_access,
        scheduler=scheduler,
        computer_use=computer_use,
        chat=ChatCoordinator(
            repository,
            single_chat,
            TurnBatchGenerationService(
                repository,
                OllamaTurnBatchGenerator(providers),
                relationship_profiles=relationship_profiles,
                provider_endpoint=lambda provider_name: providers.get(
                    provider_name
                ).metadata.endpoint,
                provider_behavior_allowed=lambda provider_name: next(
                    (
                        item.relationship_behavior_allowed
                        for item in providers.list_connections()
                        if item.provider_name == provider_name
                    ),
                    False,
                ),
                relationship_turn_reception=relationship_turn_reception,
            ),
            TurnBatchService(repository),
            memory_capture_scheduler=memory_capture,
            translation_scheduler=translations,
            telemetry=telemetry,
        ),
    )

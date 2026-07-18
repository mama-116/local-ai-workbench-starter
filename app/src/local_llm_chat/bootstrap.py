from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_llm_chat.application.services.chat_service import ChatService
from local_llm_chat.application.services.computer_use_access_service import (
    ComputerUseAccessService,
)
from local_llm_chat.application.services.computer_use_service import (
    ComputerUseCoordinator,
)
from local_llm_chat.application.services.builtin_tool_provider import (
    BuiltInToolProvider,
    ToolAccessService,
)
from local_llm_chat.application.services.tool_coordinator import ToolCoordinator
from local_llm_chat.application.services.conversation_service import ConversationService
from local_llm_chat.application.services.profile_service import ProfileService
from local_llm_chat.application.services.rag_service import RagService
from local_llm_chat.application.services.restart_service import RestartService
from local_llm_chat.application.services.scheduler_service import SchedulerService
from local_llm_chat.application.services.translation_service import TranslationService
from local_llm_chat.application.services.telemetry_service import TelemetryService
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.infrastructure.llm.ollama_registry import (
    LOCAL_PROVIDER_NAME,
    OllamaProviderRegistry,
)
from local_llm_chat.infrastructure.computer_use.fake_action_broker import (
    FakeDesktopActionBroker,
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


@dataclass(slots=True)
class AppContainer:
    paths: AppPaths
    repository: SQLiteAppRepository
    providers: OllamaProviderRegistry
    profiles: ProfileService
    rag: RagService
    restart: RestartService
    conversations: ConversationService
    translations: TranslationService
    telemetry: TelemetryService
    chat: ChatService
    tool_access: ToolAccessService
    scheduler: SchedulerService
    computer_use: ComputerUseAccessService

    async def close(self) -> None:
        await self.scheduler.close()
        await self.translations.close()
        await self.telemetry.close()
        await self.providers.close()


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

    return AppContainer(
        paths=paths,
        repository=repository,
        providers=providers,
        profiles=ProfileService(repository, providers, policy),
        rag=rag,
        restart=RestartService(paths.data_dir, SubprocessRestartLauncher()),
        conversations=ConversationService(repository, providers, policy),
        translations=translations,
        telemetry=telemetry,
        tool_access=tool_access,
        scheduler=scheduler,
        computer_use=computer_use,
        chat=ChatService(
            repository,
            providers,
            policy,
            translations,
            telemetry,
            rag,
            tool_coordinator,
        ),
    )

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast
from uuid import uuid4

import flet as ft

from local_llm_chat.application.model_selection import (
    model_option_label,
    recommend_chat_model,
)
from local_llm_chat.application.services.conversation_timeline_service import (
    ConversationTimelineItem,
)
from local_llm_chat.application.services.memory_capture_service import (
    MemoryCaptureState,
    MemoryCaptureUpdate,
)
from local_llm_chat.application.services.memory_decision_service import (
    MemoryDecisionAction,
    MemoryDecisionRequest,
)
from local_llm_chat.application.services.memory_review_service import (
    MemoryReviewSnapshot,
)
from local_llm_chat.bootstrap import AppContainer
from local_llm_chat.domain.canonical_memory import CanonicalMemoryReviewItem
from local_llm_chat.domain.errors import AppError
from local_llm_chat.domain.group_turns import ConversationGroupConfiguration
from local_llm_chat.domain.models import (
    BranchInfo,
    CharacterVersion,
    ComputerUseRun,
    Conversation,
    DocumentRecord,
    EmbeddingConfiguration,
    LatestTelemetry,
    Message,
    MessageRagUsage,
    ModelInfo,
    ProviderConnection,
    Translation,
    TelemetryMetric,
    ToolCallAudit,
    ToolFolderGrant,
    DesktopSecurityContext,
    utc_now,
)
from local_llm_chat.domain.states import (
    DesktopIntegrityLevel,
    EmbeddingIndexState,
    MessageRole,
    MessageState,
)
from local_llm_chat.presentation.components.computer_use_review import (
    ComputerUseReviewDialog,
)
from local_llm_chat.presentation.components.message_bubble import MessageBubble
from local_llm_chat.presentation.components.group_chat_settings_dialog import (
    GroupChatSettingsDialog,
    GroupChatSettingsInput,
    group_configuration_summary,
)
from local_llm_chat.presentation.components.turn_segment_bubble import (
    TurnSegmentBubble,
)

ACCENT = "#F2A65A"
MINT = "#55C2A3"
CANVAS = "#121212"
PANEL = "#171716"
PANEL_ALT = "#1D1D1B"
TEXT = "#E8E4DC"
MUTED = "#969188"
ERROR = "#D87866"


def _latest_tool_audit(audits: list[ToolCallAudit]) -> ToolCallAudit | None:
    return audits[0] if audits else None


class WindowCloser(Protocol):
    async def close(self) -> None: ...


class LocalChatApp:
    def __init__(self, page: ft.Page, container: AppContainer) -> None:
        self.page = page
        self.container = container
        self.conversations: list[Conversation] = []
        self.characters: list[CharacterVersion] = []
        self.models: list[ModelInfo] = []
        self.connections: list[ProviderConnection] = []
        self.branches: list[BranchInfo] = []
        self.message_translations: dict[str, Translation] = {}
        self.message_rag_usage: dict[str, MessageRagUsage] = {}
        self.rag_documents: list[DocumentRecord] = []
        self.selected_rag_documents: list[DocumentRecord] = []
        self.latest_telemetry: LatestTelemetry | None = None
        self.tool_folder_grant: ToolFolderGrant | None = None
        self.tool_audits: list[ToolCallAudit] = []
        self.computer_use_runs: list[ComputerUseRun] = []
        self.group_configuration: ConversationGroupConfiguration | None = None
        self.memory_review = MemoryReviewSnapshot((), ())
        self._new_memory_event_ids: list[str] = []
        self._memory_notice_conversation_id: str | None = None
        self._memory_notice_branch_id: str | None = None
        self._memory_capture_source_message_id: str | None = None
        self._memory_decision_in_progress = False
        self._memory_saved_expanded = False
        self.selected_conversation_id: str | None = None
        self.guard_error: str | None = None
        self.selected_provider_name: str | None = None
        self._generation_task: asyncio.Task[Any] | None = None
        self._generation_conversation_id: str | None = None
        self._computer_use_approval_task: asyncio.Task[None] | None = None

        self.conversation_list = ft.ListView(expand=True, spacing=5, padding=0)
        self.message_list = ft.ListView(
            expand=True,
            spacing=2,
            padding=ft.Padding(0, 16, 0, 16),
            auto_scroll=True,
        )
        self.title_text = ft.Text("会話を選択", size=20, weight=ft.FontWeight.W_600)
        self.subtitle_text = ft.Text("ローカルだけで動く対話空間", size=11, color=MUTED)
        self.composer = ft.TextField(
            hint_text="メッセージを入力",
            multiline=True,
            min_lines=1,
            max_lines=5,
            border=ft.InputBorder.NONE,
            filled=True,
            fill_color="#24231F",
            color=TEXT,
            hint_style=ft.TextStyle(color="#77736B"),
            border_radius=18,
            content_padding=ft.Padding(16, 12, 16, 12),
            on_submit=self.send_message,
            expand=True,
        )
        self.send_button = ft.IconButton(
            icon=ft.Icons.SEND_ROUNDED,
            icon_color="#17120D",
            bgcolor=ACCENT,
            tooltip="送信",
            on_click=self.send_message,
        )
        self.rag_button = ft.IconButton(
            icon=ft.Icons.ATTACH_FILE_ROUNDED,
            icon_color=MINT,
            bgcolor="#24231F",
            tooltip="参照資料を選ぶ",
            on_click=self.show_rag_dialog,
            disabled=True,
        )
        self.rag_selection_text = ft.Text("", size=10, color="#B9B4AA")
        self.rag_selection_bar = ft.Container(
            visible=False,
            padding=ft.Padding(10, 5, 10, 5),
            bgcolor="#1A2421",
            border_radius=10,
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.MENU_BOOK_ROUNDED, size=14, color=MINT),
                    self.rag_selection_text,
                ],
                spacing=6,
            ),
        )
        self.file_picker = ft.FilePicker()
        self.page.services.append(self.file_picker)
        self.stop_button = ft.IconButton(
            icon=ft.Icons.STOP_ROUNDED,
            icon_color=TEXT,
            bgcolor="#56342E",
            tooltip="生成を停止",
            on_click=self.stop_generation,
            visible=False,
        )
        self.new_conversation_button = ft.Button(
            "新しい会話",
            icon=ft.Icons.ADD_ROUNDED,
            color="#17120D",
            bgcolor=ACCENT,
            on_click=self.show_new_conversation_dialog,
            width=208,
        )
        self.guard_badge = ft.Container()
        self.guard_detail = ft.Text(size=11, color=MUTED)
        self.telemetry_connection = ft.Text("未取得", size=13, color=TEXT)
        self.telemetry_model = ft.Text("未取得", size=11, color=MUTED)
        self.telemetry_response = ft.Text("未取得", size=12, color=TEXT)
        self.telemetry_speed = ft.Text("未取得", size=12, color=TEXT)
        self.cpu_value = ft.Text("未取得", size=11, color=TEXT)
        self.cpu_source = ft.Text("取得元: 未取得", size=9, color=MUTED)
        self.ram_value = ft.Text("未取得", size=11, color=TEXT)
        self.ram_source = ft.Text("取得元: 未取得", size=9, color=MUTED)
        self.gpu_value = ft.Text("未取得", size=11, color=TEXT)
        self.gpu_source = ft.Text("取得元: 未取得", size=9, color=MUTED)
        self.vram_value = ft.Text("未取得", size=11, color=TEXT)
        self.vram_source = ft.Text("取得元: 未取得", size=9, color=MUTED)
        self.telemetry_details = ft.Column(
            [
                self._telemetry_row("CPU", self.cpu_value, self.cpu_source),
                self._telemetry_row("RAM", self.ram_value, self.ram_source),
                self._telemetry_row("GPU", self.gpu_value, self.gpu_source),
                self._telemetry_row("VRAM", self.vram_value, self.vram_source),
            ],
            spacing=7,
        )
        self.telemetry_toggle_label = ft.Text("詳細を隠す")
        self.telemetry_toggle = ft.Button(
            self.telemetry_toggle_label,
            icon=ft.Icons.EXPAND_LESS_ROUNDED,
            color=MUTED,
            bgcolor="#292925",
            on_click=self.toggle_telemetry_details,
        )
        self.telemetry_card = ft.Container(
            bgcolor="#242421",
            border_radius=14,
            padding=12,
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.SPEED_ROUNDED, size=16, color=MINT),
                            ft.Text("直近の性能", size=11, weight=ft.FontWeight.W_600),
                        ],
                        spacing=6,
                    ),
                    self.telemetry_connection,
                    self.telemetry_model,
                    ft.Row(
                        [
                            ft.Column([ft.Text("応答", size=9, color=MUTED), self.telemetry_response], spacing=0),
                            ft.Column([ft.Text("生成", size=9, color=MUTED), self.telemetry_speed], spacing=0),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    self.telemetry_details,
                    self.telemetry_toggle,
                ],
                spacing=8,
            ),
        )
        self.tool_folder_text = ft.Text(
            "未許可", size=10, color=MUTED, selectable=True
        )
        self.tool_running_text = ft.Text("実行中: 0件", size=10, color=MUTED)
        self.tool_recent_text = ft.Text(
            "直近結果: なし", size=10, color=MUTED
        )
        self.tool_revoke_button = ft.Button(
            "許可を取り消す",
            icon=ft.Icons.BLOCK_ROUNDED,
            color=ERROR,
            bgcolor="#292925",
            on_click=self.revoke_tool_folder,
            disabled=True,
        )
        self.tool_history_button = ft.Button(
            "監査履歴を開く",
            icon=ft.Icons.HISTORY_ROUNDED,
            color=TEXT,
            bgcolor="#292925",
            on_click=self.show_tool_audit_dialog,
            disabled=True,
        )
        self.tool_card = ft.Container(
            bgcolor="#242421",
            border_radius=14,
            padding=12,
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.HANDYMAN_ROUNDED, size=16, color=ACCENT),
                            ft.Text("TOOLS / MCP", size=11, weight=ft.FontWeight.W_600),
                        ],
                        spacing=6,
                    ),
                    ft.Text(
                        "利用可能: 内蔵2件 / local-notes MCP 2件",
                        size=10,
                        color=MUTED,
                    ),
                    self.tool_folder_text,
                    ft.Button(
                        "フォルダーを許可",
                        icon=ft.Icons.FOLDER_OPEN_ROUNDED,
                        color="#17120D",
                        bgcolor=ACCENT,
                        on_click=self.choose_tool_folder,
                    ),
                    self.tool_revoke_button,
                    self.tool_running_text,
                    self.tool_recent_text,
                    self.tool_history_button,
                ],
                spacing=7,
            ),
        )
        self.computer_use_recent_text = ft.Text(
            "直近結果: なし", size=10, color=MUTED
        )
        self.computer_use_start_button = ft.Button(
            "Fake安全テスト",
            icon=ft.Icons.PREVIEW_ROUNDED,
            color="#17120D",
            bgcolor=ACCENT,
            on_click=self.show_computer_use_review,
            disabled=True,
        )
        self.computer_use_history_button = ft.Button(
            "監査履歴を開く",
            icon=ft.Icons.HISTORY_ROUNDED,
            color=TEXT,
            bgcolor="#292925",
            on_click=self.show_computer_use_audit_dialog,
            disabled=True,
        )
        self.computer_use_card = ft.Container(
            bgcolor="#242421",
            border_radius=14,
            padding=12,
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.COMPUTER_ROUNDED, size=16, color=ACCENT),
                            ft.Text(
                                "COMPUTER USE",
                                size=11,
                                weight=ft.FontWeight.W_600,
                            ),
                        ],
                        spacing=6,
                    ),
                    ft.Text(
                        "Fake Brokerのみ。マウス・キーボード入力は送信しません。",
                        size=10,
                        color=MUTED,
                    ),
                    self.computer_use_recent_text,
                    self.computer_use_start_button,
                    self.computer_use_history_button,
                ],
                spacing=7,
            ),
        )
        self.character_dropdown = self._dropdown("キャラクター")
        self.character_dropdown.on_select = self.update_selection
        self.group_chat_summary = ft.Text("単独会話", size=11, color=MUTED)
        self.group_chat_button = ft.Button(
            "キャストとモード",
            icon=ft.Icons.GROUPS_ROUNDED,
            color=TEXT,
            bgcolor="#292925",
            on_click=self.show_group_chat_dialog,
            disabled=True,
        )
        self.memory_undo_list = ft.Column(spacing=6)
        self.memory_undo_bar = ft.Container(
            visible=False,
            bgcolor="#202A26",
            border=ft.Border.all(1, MINT),
            border_radius=12,
            padding=10,
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(
                                "今回保存した記憶",
                                size=11,
                                color=TEXT,
                                weight=ft.FontWeight.W_600,
                            ),
                            ft.Container(expand=True),
                            ft.IconButton(
                                icon=ft.Icons.CLOSE_ROUNDED,
                                icon_color=MUTED,
                                tooltip="通知を閉じる（記憶は残ります）",
                                on_click=self.dismiss_memory_undo_bar,
                            ),
                        ],
                        spacing=4,
                    ),
                    self.memory_undo_list,
                ],
                spacing=4,
            ),
        )
        self.memory_pending_count = ft.Text("確認待ち 0件", size=11, color=MUTED)
        self.memory_pending_list = ft.Column(spacing=7, visible=False)
        self.memory_saved_count = ft.Text("保存済み 0件", size=11, color=MUTED)
        self.memory_capture_status = ft.Text(
            "記憶確認: 未実行", size=10, color=MUTED
        )
        self.memory_saved_list = ft.Column(spacing=7, visible=False)
        self.memory_saved_toggle = ft.Button(
            "保存済みを表示",
            icon=ft.Icons.EXPAND_MORE_ROUNDED,
            color=MUTED,
            bgcolor="#292925",
            on_click=self.toggle_saved_memories,
            disabled=True,
        )
        self.memory_card = ft.Container(
            bgcolor="#242421",
            border_radius=14,
            padding=12,
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.MEMORY_ROUNDED, size=16, color=MINT),
                            ft.Text(
                                "正史記憶",
                                size=11,
                                weight=ft.FontWeight.W_600,
                            ),
                        ],
                        spacing=6,
                    ),
                    self.memory_capture_status,
                    self.memory_pending_count,
                    self.memory_pending_list,
                    self.memory_saved_count,
                    self.memory_saved_toggle,
                    self.memory_saved_list,
                ],
                spacing=8,
            ),
        )
        self.connection_dropdown = self._dropdown("Ollama接続先")
        self.connection_dropdown.on_select = self.switch_connection
        self.embedding_status_text = ft.Text(
            "埋め込み: 未設定（語句検索のみ）", size=10, color=MUTED
        )
        self.model_roles_button = ft.Button(
            "用途別モデル設定",
            icon=ft.Icons.TUNE_ROUNDED,
            color=TEXT,
            bgcolor="#292925",
            on_click=self.show_model_role_dialog,
        )
        self.model_dropdown = self._dropdown("モデル")
        self.model_dropdown.on_select = self.update_selection
        self.branch_dropdown = self._dropdown("会話の分岐")
        self.branch_dropdown.on_select = self.activate_branch
        self.auto_translate_switch = ft.Switch(
            label="回答後に自動翻訳",
            value=False,
            active_color=MINT,
            on_change=self.update_auto_translate,
        )
        self.manage_branches_button = ft.Button(
            "分岐を整理",
            icon=ft.Icons.ACCOUNT_TREE_ROUNDED,
            color=MUTED,
            bgcolor="#242421",
            on_click=self.show_branch_management,
            disabled=True,
        )
        self.archive_button = ft.Button(
            "会話をゴミ箱へ移動",
            icon=ft.Icons.DELETE_OUTLINE_ROUNDED,
            color="#D87866",
            bgcolor="#302522",
            on_click=self.confirm_archive,
            disabled=True,
        )
        self.restart_button_label = ft.Text("アプリを再起動")
        self.restart_button = ft.Button(
            self.restart_button_label,
            icon=ft.Icons.RESTART_ALT_ROUNDED,
            color=TEXT,
            bgcolor="#35342F",
            tooltip="Ctrl+Shift+R",
            on_click=self.restart_app,
        )
        self.container.translations.subscribe(self._on_translation_update)
        self.container.telemetry.subscribe(self._on_telemetry_update)
        self.container.tool_access.subscribe(self._on_tool_update)
        self.container.memory_capture.subscribe(self._on_memory_capture)
        self.container.embeddings.subscribe(self._on_embedding_update)

    async def initialize(self) -> None:
        self._configure_page()
        self.page.add(self._build_shell())
        self._show_empty_state("起動情報を確認しています...")
        self.page.update()
        await self.refresh_all()

    def _configure_page(self) -> None:
        self.page.title = "Local LLM Chat"
        self.page.bgcolor = CANVAS
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.theme = ft.Theme(font_family="Yu Gothic UI", color_scheme_seed=ACCENT)
        self.page.on_close = self.close
        self.page.on_keyboard_event = self.on_keyboard_event

    def _build_shell(self) -> ft.Control:
        sidebar = ft.Container(
            width=244,
            bgcolor=PANEL,
            padding=ft.Padding(18, 22, 18, 18),
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Container(
                                width=10,
                                height=34,
                                bgcolor=ACCENT,
                                border_radius=5,
                            ),
                            ft.Column(
                                [
                                    ft.Text("LOCAL / ROOM", size=14, weight=ft.FontWeight.BOLD),
                                    ft.Text("端末内の会話", size=10, color=MUTED),
                                ],
                                spacing=0,
                            ),
                        ]
                    ),
                    ft.Container(height=8),
                    self.new_conversation_button,
                    ft.Text("CONVERSATIONS", size=10, color="#6F6B64"),
                    self.conversation_list,
                    ft.Button(
                        "ゴミ箱",
                        icon=ft.Icons.DELETE_OUTLINE_ROUNDED,
                        color=MUTED,
                        bgcolor="#22221F",
                        on_click=self.show_archive_dialog,
                        width=208,
                    ),
                ],
                expand=True,
                spacing=16,
            ),
        )
        center = ft.Container(
            expand=True,
            gradient=ft.LinearGradient(
                begin=ft.Alignment.TOP_LEFT,
                end=ft.Alignment.BOTTOM_RIGHT,
                colors=["#171715", "#121212", "#141816"],
            ),
            content=ft.Column(
                [
                    ft.Container(
                        padding=ft.Padding(28, 18, 22, 14),
                        content=ft.Row(
                            [
                                ft.Column([self.title_text, self.subtitle_text], spacing=1),
                                ft.Container(expand=True),
                                self.stop_button,
                            ]
                        ),
                    ),
                    self.message_list,
                    ft.Container(
                        padding=ft.Padding(28, 12, 28, 24),
                        content=ft.Column(
                            [
                                self.rag_selection_bar,
                                self.memory_undo_bar,
                                ft.Row(
                                    [self.rag_button, self.composer, self.send_button],
                                    spacing=10,
                                ),
                            ],
                            spacing=7,
                        ),
                    ),
                ],
                expand=True,
                spacing=0,
            ),
        )
        inspector = ft.Container(
            width=292,
            bgcolor=PANEL_ALT,
            padding=ft.Padding(20, 22, 20, 20),
            content=ft.Column(
                [
                    ft.Text("CONTROL DESK", size=11, color="#716D65"),
                    self.guard_badge,
                    self.guard_detail,
                    self.telemetry_card,
                    self.tool_card,
                    self.computer_use_card,
                    ft.Button(
                        "状態を再確認",
                        icon=ft.Icons.REFRESH_ROUNDED,
                        color=TEXT,
                        bgcolor="#292925",
                        on_click=self.refresh_all,
                    ),
                    self.restart_button,
                    self._section_label("OLLAMA CONNECTION"),
                    self.connection_dropdown,
                    ft.Button(
                        "接続先を追加・編集",
                        icon=ft.Icons.LAN_ROUNDED,
                        color=TEXT,
                        bgcolor="#292925",
                        on_click=self.show_connection_dialog,
                    ),
                    self.model_roles_button,
                    self.embedding_status_text,
                    self._section_label("PERSONA"),
                    self.character_dropdown,
                    ft.Button(
                        "キャラクターを追加",
                        icon=ft.Icons.ADD_ROUNDED,
                        color=TEXT,
                        bgcolor="#292925",
                        on_click=self.show_new_character_dialog,
                    ),
                    ft.Button(
                        "選択中を編集",
                        icon=ft.Icons.EDIT_ROUNDED,
                        color=TEXT,
                        bgcolor="#292925",
                        on_click=self.show_selected_character_dialog,
                    ),
                    self._section_label("GROUP CHAT"),
                    self.group_chat_summary,
                    self.group_chat_button,
                    self.memory_card,
                    self._section_label("LOCAL MODEL"),
                    self.model_dropdown,
                    self._section_label("BRANCH"),
                    self.branch_dropdown,
                    self.manage_branches_button,
                    self._section_label("TRANSLATION"),
                    self.auto_translate_switch,
                    ft.Text(
                        "OFFでも、必要な回答だけ手動で翻訳できます。",
                        size=10,
                        color="#716D65",
                    ),
                    self.archive_button,
                    ft.Text(
                        "会話・設定・実行ログはこのPC内のSQLiteだけに保存します。",
                        size=10,
                        color="#716D65",
                    ),
                ],
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
            ),
        )
        return ft.Row([sidebar, center, inspector], expand=True, spacing=0)

    async def refresh_all(self) -> None:
        self.characters = await self.container.profiles.list_characters()
        self.rag_documents = await self.container.rag.documents()
        self.connections = self.container.profiles.list_connections()
        self.conversations = await self.container.conversations.list_conversations()
        live_ids = {conversation.id for conversation in self.conversations}
        if self.selected_conversation_id not in live_ids:
            self.selected_conversation_id = (
                self.conversations[0].id if self.conversations else None
            )
        provider_names = {item.provider_name for item in self.connections}
        desired_provider = self.selected_provider_name
        conversation = self._selected_conversation()
        if conversation is not None:
            selection = await self.container.conversations.selection(conversation.id)
            if selection.model_profile.provider in provider_names:
                desired_provider = selection.model_profile.provider
        if desired_provider not in provider_names:
            desired_provider = self.container.profiles.default_provider_name
        self.selected_provider_name = desired_provider
        await self._load_models(desired_provider)
        self._render_guard()
        self._render_conversations()
        self._render_options()
        await self._refresh_selected_conversation()
        await self._refresh_telemetry()
        await self._refresh_tools()
        await self._refresh_computer_use()
        await self._refresh_embedding_status()
        self.page.update()

    async def _refresh_embedding_status(self) -> None:
        configuration = await self.container.embeddings.configuration()
        self.embedding_status_text.value, self.embedding_status_text.color = (
            self._embedding_status(configuration)
        )

    async def _on_embedding_update(self) -> None:
        await self._refresh_embedding_status()
        self.page.update(self.embedding_status_text)

    @staticmethod
    def _embedding_status(
        configuration: EmbeddingConfiguration,
    ) -> tuple[str, str]:
        desired = configuration.desired
        active = configuration.active
        if desired is None:
            return "埋め込み: 未設定（語句検索のみ）", MUTED
        target = f"{desired.provider_name} / {desired.model_name}"
        if desired.state is EmbeddingIndexState.BUILDING:
            suffix = "・旧索引で継続" if active and active.id != desired.id else ""
            return (
                f"埋め込み: 索引構築中 {desired.embedded_chunks}/{desired.total_chunks}{suffix}",
                ACCENT,
            )
        if desired.state is EmbeddingIndexState.FAILED:
            suffix = "・旧索引で継続" if active else "・語句検索のみ"
            return f"埋め込み: 失敗{suffix}", ERROR
        if active is not None and active.id == desired.id:
            return f"埋め込み: 利用可能 · {target}", MINT
        return f"埋め込み: 待機中 · {target}", MUTED

    async def _refresh_computer_use(self) -> None:
        conversation_id = self.selected_conversation_id
        self.computer_use_runs = (
            await self.container.computer_use.history(conversation_id)
            if conversation_id is not None
            else []
        )
        self.computer_use_start_button.disabled = conversation_id is None
        self.computer_use_history_button.disabled = not self.computer_use_runs
        if not self.computer_use_runs:
            self.computer_use_recent_text.value = "直近結果: なし"
            return
        latest = self.computer_use_runs[0]
        reason = f" · {latest.failure_reason}" if latest.failure_reason else ""
        self.computer_use_recent_text.value = (
            f"直近結果: {latest.state.value}{reason} · OS入力0件"
        )

    async def _refresh_tools(self) -> None:
        conversation_id = self.selected_conversation_id
        if conversation_id is None:
            self.tool_folder_grant = None
            self.tool_audits = []
        else:
            self.tool_folder_grant = await self.container.tool_access.grant(
                conversation_id
            )
            self.tool_audits = await self.container.tool_access.audits(conversation_id)
        self.tool_folder_text.value = (
            f"許可フォルダー: {self.tool_folder_grant.root_path}"
            if self.tool_folder_grant is not None
            else "許可フォルダー: 未設定"
        )
        self.tool_revoke_button.disabled = self.tool_folder_grant is None
        running = sum(audit.state.value == "running" for audit in self.tool_audits)
        self.tool_running_text.value = f"実行中: {running}件"
        latest = _latest_tool_audit(self.tool_audits)
        if latest is None:
            self.tool_recent_text.value = "直近結果: なし"
        elif latest.failure_reason:
            self.tool_recent_text.value = (
                f"直近結果: {latest.state.value} · {latest.failure_reason}"
            )
        else:
            size = latest.result_size_bytes or 0
            count = latest.result_item_count or 0
            self.tool_recent_text.value = (
                f"直近結果: {latest.state.value} · {count}件 · {size} bytes"
            )
        self.tool_history_button.disabled = not self.tool_audits

    async def _on_tool_update(self, conversation_id: str) -> None:
        if conversation_id != self.selected_conversation_id:
            return
        await self._refresh_tools()
        self.page.update(self.tool_card)

    async def _refresh_telemetry(self) -> None:
        self.latest_telemetry = await self.container.telemetry.latest(
            self.selected_conversation_id
        )
        self._render_telemetry()

    async def _on_telemetry_update(self, run_id: str) -> None:
        if self.selected_conversation_id is None:
            return
        await self._refresh_telemetry()
        if self.latest_telemetry is not None and self.latest_telemetry.run.id == run_id:
            self.page.update(self.telemetry_card)

    def _render_telemetry(self) -> None:
        latest = self.latest_telemetry
        if latest is None:
            self.telemetry_connection.value = "未取得"
            self.telemetry_model.value = "完了したAI回答がありません"
            self.telemetry_response.value = "未取得"
            self.telemetry_speed.value = "未取得"
            missing = TelemetryMetric(
                "missing", None, "", "未取得", "完了したAI回答がありません"
            )
            self._set_metric_controls(self.cpu_value, self.cpu_source, missing)
            self._set_metric_controls(self.ram_value, self.ram_source, missing)
            self._set_metric_controls(self.gpu_value, self.gpu_source, missing)
            self._set_vram_controls(missing, missing)
            return
        connection = next(
            (
                item
                for item in self.connections
                if item.provider_name == latest.run.provider
            ),
            None,
        )
        self.telemetry_connection.value = (
            connection.display_name if connection is not None else latest.run.provider
        )
        self.telemetry_model.value = latest.run.model
        self.telemetry_response.value = (
            f"{latest.run.response_duration_ms / 1000:.1f} 秒"
            if latest.run.response_duration_ms is not None
            else "未取得 · 実行時間なし"
        )
        speed = latest.tokens_per_second
        self.telemetry_speed.value = (
            f"{speed:.1f} t/s" if speed is not None else "未取得 · 生成時間なし"
        )
        self._set_metric_controls(
            self.cpu_value, self.cpu_source, latest.metric("cpu_percent")
        )
        self._set_metric_controls(
            self.ram_value, self.ram_source, latest.metric("ram_percent")
        )
        self._set_metric_controls(
            self.gpu_value, self.gpu_source, latest.metric("gpu_percent")
        )
        self._set_vram_controls(
            latest.metric("vram_used_gb"), latest.metric("vram_total_gb")
        )

    @staticmethod
    def _set_metric_controls(
        value_control: ft.Text,
        source_control: ft.Text,
        metric: TelemetryMetric,
    ) -> None:
        if metric.value is None:
            reason = metric.unavailable_reason or "理由不明"
            value_control.value = f"未取得 · {reason}"
        else:
            value_control.value = f"{metric.value:.1f}{metric.unit}"
        source_control.value = f"取得元: {metric.source}"

    def _set_vram_controls(
        self, used: TelemetryMetric, total: TelemetryMetric
    ) -> None:
        if used.value is None or total.value is None:
            reason = used.unavailable_reason or total.unavailable_reason or "理由不明"
            self.vram_value.value = f"未取得 · {reason}"
        else:
            self.vram_value.value = f"{used.value:.1f} / {total.value:.1f} GB"
        self.vram_source.value = f"取得元: {used.source}"

    def toggle_telemetry_details(self) -> None:
        self.telemetry_details.visible = not self.telemetry_details.visible
        self.telemetry_toggle_label.value = (
            "詳細を隠す" if self.telemetry_details.visible else "詳細を表示"
        )
        self.telemetry_toggle.icon = (
            ft.Icons.EXPAND_LESS_ROUNDED
            if self.telemetry_details.visible
            else ft.Icons.EXPAND_MORE_ROUNDED
        )
        self.page.update(self.telemetry_card)

    async def _load_models(self, provider_name: str) -> None:
        self.guard_error = None
        try:
            self.models = await self.container.profiles.list_visible_models(provider_name)
        except AppError as error:
            self.models = []
            self.guard_error = str(error)

    async def switch_connection(self) -> None:
        provider_name = self.connection_dropdown.value
        if not provider_name or provider_name == self.selected_provider_name:
            return
        self.selected_provider_name = provider_name
        await self._load_models(provider_name)
        self._render_guard()
        self._render_options()
        self.connection_dropdown.value = provider_name
        self.model_dropdown.value = None
        self._set_composer_enabled(False)
        self.page.update()

    async def select_conversation(self, conversation_id: str) -> None:
        self.selected_conversation_id = conversation_id
        selection = await self.container.conversations.selection(conversation_id)
        if selection.model_profile.provider != self.selected_provider_name:
            self.selected_provider_name = selection.model_profile.provider
            await self._load_models(selection.model_profile.provider)
            self._render_guard()
            self._render_options()
        self._render_conversations()
        await self._refresh_selected_conversation()
        await self._refresh_telemetry()
        await self._refresh_tools()
        self.page.update()

    async def _refresh_selected_conversation(self) -> None:
        conversation = self._selected_conversation()
        if conversation is None:
            self.title_text.value = "最初の会話を作りましょう"
            self.subtitle_text.value = "右側で無料運営の状態を確認できます"
            self.archive_button.disabled = True
            self.manage_branches_button.disabled = True
            self.auto_translate_switch.disabled = True
            self.auto_translate_switch.value = False
            self.character_dropdown.value = None
            self.character_dropdown.disabled = False
            self.group_configuration = None
            self.memory_review = MemoryReviewSnapshot((), ())
            self._new_memory_event_ids.clear()
            self._memory_notice_conversation_id = None
            self._memory_notice_branch_id = None
            self._render_memory_review()
            self.group_chat_summary.value = "会話を選択してください"
            self.group_chat_button.disabled = True
            self.connection_dropdown.value = self.selected_provider_name
            self.model_dropdown.value = None
            self.branch_dropdown.options = []
            self.branch_dropdown.value = None
            self._show_empty_state(
                "Ollama Cloudを無効にし、ローカルモデルを確認すると会話を作成できます。"
                if not self.models
                else "左上の「新しい会話」から始められます。"
            )
            self._set_composer_enabled(False)
            self.selected_rag_documents = []
            self._render_rag_selection()
            return

        self.title_text.value = conversation.title
        selection = await self.container.conversations.selection(conversation.id)
        connection = next(
            (
                item
                for item in self.connections
                if item.provider_name == selection.model_profile.provider
            ),
            None,
        )
        connection_name = connection.display_name if connection else "未登録の接続先"
        self.subtitle_text.value = (
            f"{selection.character.display_name}  /  {connection_name}  /  "
            f"{selection.model_profile.model_name}"
        )
        self.character_dropdown.value = selection.character.id
        self.group_configuration = await self.container.group_configuration.get(
            conversation.id
        )
        self.group_chat_summary.value = group_configuration_summary(
            self.group_configuration
        )
        self.group_chat_button.disabled = self._generation_task is not None
        self.character_dropdown.disabled = (
            self.group_configuration.settings.enabled
            or len(self.group_configuration.cast.members) > 1
        )
        self.selected_provider_name = selection.model_profile.provider
        self.connection_dropdown.value = selection.model_profile.provider
        allowed_model_names = {model.name for model in self.models}
        self.model_dropdown.value = (
            selection.model_profile.model_name
            if selection.model_profile.model_name in allowed_model_names
            else None
        )
        self.archive_button.disabled = False
        self.manage_branches_button.disabled = self._generation_task is not None
        self.auto_translate_switch.disabled = self._generation_task is not None
        self.auto_translate_switch.value = conversation.auto_translate
        self.branches = await self.container.conversations.branches(conversation.id)
        self.branch_dropdown.options = [
            ft.DropdownOption(branch.id, f"分岐 {index + 1}")
            for index, branch in enumerate(self.branches)
        ]
        self.branch_dropdown.value = conversation.active_branch_id
        if (
            self._memory_notice_conversation_id != conversation.id
            or self._memory_notice_branch_id != conversation.active_branch_id
        ):
            self._new_memory_event_ids.clear()
            self._memory_capture_source_message_id = None
            self.memory_capture_status.value = "記憶確認: 未実行"
            self.memory_capture_status.color = MUTED
            self._memory_notice_conversation_id = conversation.id
            self._memory_notice_branch_id = conversation.active_branch_id
        await self._refresh_memory_review(
            conversation.id, conversation.active_branch_id
        )
        await self._refresh_messages(conversation.id)
        self.selected_rag_documents = await self.container.rag.selected_documents(
            conversation.id
        )
        self._render_rag_selection()
        self._set_composer_enabled(bool(self.models) and self._generation_task is None)

    async def _refresh_memory_review(
        self, conversation_id: str, branch_id: str
    ) -> None:
        self.memory_review = await self.container.memory_review.list_items(
            conversation_id, branch_id
        )
        undoable_ids = {item.event_id for item in self.memory_review.undoable}
        self._new_memory_event_ids = [
            event_id
            for event_id in self._new_memory_event_ids
            if event_id in undoable_ids
        ]
        self._render_memory_review()

    async def _on_memory_capture(self, update: MemoryCaptureUpdate) -> None:
        conversation = self._selected_conversation()
        if (
            conversation is None
            or update.conversation_id != conversation.id
            or update.branch_id != conversation.active_branch_id
        ):
            return
        if not self._memory_capture_update_is_current(
            self._memory_capture_source_message_id, update
        ):
            return
        if update.state is MemoryCaptureState.PROCESSING:
            self._memory_capture_source_message_id = update.source_message_id
        self.memory_capture_status.value = self._memory_capture_status_label(
            update.state, len(update.persisted_event_ids)
        )
        self.memory_capture_status.color = {
            MemoryCaptureState.PROCESSING: ACCENT,
            MemoryCaptureState.SAVED: MINT,
            MemoryCaptureState.NO_CANDIDATES: MUTED,
            MemoryCaptureState.FAILED: ERROR,
        }[update.state]
        if update.state is not MemoryCaptureState.SAVED:
            self.page.update(self.memory_card)
            return
        snapshot = await self.container.memory_review.list_items(
            conversation.id, conversation.active_branch_id
        )
        current = self._selected_conversation()
        if (
            current is None
            or current.id != update.conversation_id
            or current.active_branch_id != update.branch_id
        ):
            return
        self.memory_review = snapshot
        self._memory_notice_conversation_id = current.id
        self._memory_notice_branch_id = current.active_branch_id
        undoable_ids = {item.event_id for item in self.memory_review.undoable}
        for event_id in update.persisted_event_ids:
            if event_id in undoable_ids and event_id not in self._new_memory_event_ids:
                self._new_memory_event_ids.append(event_id)
        self._render_memory_review()
        self.page.update(self.memory_card, self.memory_undo_bar)

    @staticmethod
    def _memory_capture_status_label(
        state: MemoryCaptureState, persisted_count: int
    ) -> str:
        if state is MemoryCaptureState.PROCESSING:
            return "記憶確認: 処理中…"
        if state is MemoryCaptureState.SAVED:
            return f"記憶確認: {persisted_count}件を保存・確認待ちへ追加"
        if state is MemoryCaptureState.NO_CANDIDATES:
            return "記憶確認: 保存対象なし"
        return "記憶確認: 失敗（会話は保存済み）"

    @staticmethod
    def _memory_capture_update_is_current(
        current_source_message_id: str | None,
        update: MemoryCaptureUpdate,
    ) -> bool:
        return (
            update.state is MemoryCaptureState.PROCESSING
            or update.source_message_id == current_source_message_id
        )

    def _render_memory_review(self) -> None:
        pending = self.memory_review.pending_confirmation
        self.memory_pending_count.value = f"確認待ち {len(pending)}件"
        self.memory_pending_count.color = ACCENT if pending else MUTED
        pending_controls: list[ft.Control] = []
        for item in pending:
            async def confirm(event_id: str = item.event_id) -> None:
                await self._decide_memory(event_id, MemoryDecisionAction.CONFIRM)

            async def reject(event_id: str = item.event_id) -> None:
                await self._decide_memory(event_id, MemoryDecisionAction.REJECT)

            pending_controls.append(
                ft.Container(
                    bgcolor="#2A2925",
                    border_radius=10,
                    padding=9,
                    content=ft.Column(
                        [
                            ft.Text(
                                self._memory_item_label(item),
                                size=11,
                                color=TEXT,
                                selectable=True,
                            ),
                            ft.Row(
                                [
                                    ft.Button(
                                        "承認",
                                        color="#17120D",
                                        bgcolor=MINT,
                                        on_click=confirm,
                                        disabled=self._memory_decision_in_progress,
                                    ),
                                    ft.Button(
                                        "却下",
                                        color=ERROR,
                                        bgcolor="#332A27",
                                        on_click=reject,
                                        disabled=self._memory_decision_in_progress,
                                    ),
                                ],
                                spacing=6,
                            ),
                        ],
                        spacing=7,
                    ),
                )
            )
        self.memory_pending_list.controls = pending_controls
        self.memory_pending_list.visible = bool(pending_controls)

        saved_controls: list[ft.Control] = []
        saved_items = self.memory_review.undoable
        for item in saved_items if self._memory_saved_expanded else ():
            async def undo_saved(event_id: str = item.event_id) -> None:
                await self._decide_memory(event_id, MemoryDecisionAction.UNDO)

            saved_controls.append(
                ft.Container(
                    bgcolor="#2A2925",
                    border_radius=10,
                    padding=9,
                    content=ft.Column(
                        [
                            ft.Text(
                                self._memory_item_label(item),
                                size=11,
                                color=TEXT,
                                selectable=True,
                            ),
                            ft.Button(
                                "元に戻す",
                                color=ACCENT,
                                bgcolor="#332E27",
                                on_click=undo_saved,
                                disabled=self._memory_decision_in_progress,
                            ),
                        ],
                        spacing=6,
                    ),
                )
            )
        self.memory_saved_count.value = f"保存済み {len(saved_items)}件"
        self.memory_saved_toggle.disabled = not saved_items
        self.memory_saved_toggle.content = (
            "保存済みを隠す" if self._memory_saved_expanded else "保存済みを表示"
        )
        self.memory_saved_toggle.icon = (
            ft.Icons.EXPAND_LESS_ROUNDED
            if self._memory_saved_expanded
            else ft.Icons.EXPAND_MORE_ROUNDED
        )
        self.memory_saved_list.controls = saved_controls
        self.memory_saved_list.visible = (
            self._memory_saved_expanded and bool(saved_items)
        )

        by_id = {item.event_id: item for item in self.memory_review.undoable}
        undo_controls: list[ft.Control] = []
        for event_id in self._new_memory_event_ids:
            undo_item = by_id.get(event_id)
            if undo_item is None:
                continue

            async def undo(target_id: str = event_id) -> None:
                await self._decide_memory(target_id, MemoryDecisionAction.UNDO)

            undo_controls.append(
                ft.Row(
                    [
                        ft.Text(
                            self._memory_item_label(undo_item),
                            size=11,
                            color=TEXT,
                            expand=True,
                            selectable=True,
                        ),
                        ft.Button(
                            "元に戻す",
                            color=ACCENT,
                            bgcolor="#332E27",
                            on_click=undo,
                            disabled=self._memory_decision_in_progress,
                        ),
                    ],
                    spacing=8,
                )
            )
        self.memory_undo_list.controls = undo_controls
        self.memory_undo_bar.visible = bool(undo_controls)

    async def _decide_memory(
        self, event_id: str, action: MemoryDecisionAction
    ) -> None:
        conversation = self._selected_conversation()
        if conversation is None or self._memory_decision_in_progress:
            return
        self._memory_decision_in_progress = True
        self._render_memory_review()
        self.page.update(self.memory_card, self.memory_undo_bar)
        try:
            await self.container.memory_review.decide(
                MemoryDecisionRequest(
                    conversation.id,
                    conversation.active_branch_id,
                    event_id,
                    action,
                )
            )
            self._new_memory_event_ids = [
                item for item in self._new_memory_event_ids if item != event_id
            ]
            message = {
                MemoryDecisionAction.CONFIRM: "記憶を承認しました。",
                MemoryDecisionAction.REJECT: "記憶候補を却下しました。",
                MemoryDecisionAction.UNDO: "保存した記憶を元に戻しました。",
            }[action]
            self._toast(message, MINT)
        except AppError as error:
            self._toast(f"記憶の状態が変わりました。再確認してください。 {error}", ERROR)
        finally:
            self._memory_decision_in_progress = False
            current = self._selected_conversation()
            if current is None:
                self.memory_review = MemoryReviewSnapshot((), ())
                self._new_memory_event_ids.clear()
                self._render_memory_review()
            else:
                await self._refresh_memory_review(
                    current.id, current.active_branch_id
                )
            self.page.update(self.memory_card, self.memory_undo_bar)

    def dismiss_memory_undo_bar(self) -> None:
        self._new_memory_event_ids.clear()
        self._render_memory_review()
        self.page.update(self.memory_undo_bar)

    def toggle_saved_memories(self) -> None:
        self._memory_saved_expanded = not self._memory_saved_expanded
        self._render_memory_review()
        self.page.update(self.memory_card)

    @staticmethod
    def _memory_item_label(item: CanonicalMemoryReviewItem) -> str:
        subject = "あなた" if item.subject_id == "user" else item.subject_id
        slot = {
            "favorite_food": "一番好きな食べ物",
            "liked_food": "好きな食べ物",
            "food_allergy": "食物アレルギー",
            "personal_goal": "目標",
            "club_membership_intent": "所属の意向",
        }.get(item.slot, item.slot)
        scope = (
            "会話内で共有"
            if not item.known_by_character_ids
            else f"知っている人物 {len(item.known_by_character_ids)}人"
        )
        return f"{subject}の{slot}: {item.value}（{scope}）"

    async def _refresh_messages(self, conversation_id: str) -> None:
        timeline_items = await self.container.timeline.list_items(conversation_id)
        messages_by_id = {
            item.message.id: item.message for item in timeline_items
        }
        messages = list(messages_by_id.values())
        self.message_translations = (
            await self.container.translations.list_current(
                [message.id for message in messages]
            )
        )
        self.message_rag_usage = await self.container.rag.message_usage(
            [message.id for message in messages]
        )
        self.message_list.controls = [
            self._timeline_control(item) for item in timeline_items
        ]
        if not timeline_items:
            self._show_empty_state("何を話しましょう。すべての内容はこのPC内に残ります。")

    def _timeline_control(self, item: ConversationTimelineItem) -> ft.Control:
        if item.segment is None or item.turn_batch is None:
            return self._message_bubble(item.message)
        return TurnSegmentBubble(
            item.segment,
            item.turn_batch,
            item.message.created_at,
            is_batch_end=item.is_batch_end,
            on_regenerate=(
                lambda: self.page.run_task(self.regenerate_turn_batch, item.message)
                if item.is_batch_end
                else None
            ),
            translation=(
                self.message_translations.get(item.message.id)
                if item.is_batch_end
                else None
            ),
            on_translate=(
                lambda: self.page.run_task(self.translate_message, item.message)
                if item.is_batch_end and item.message.state is MessageState.COMPLETED
                else None
            ),
        )

    def _message_bubble(self, message: Message) -> MessageBubble:
        if message.role is MessageRole.USER:
            return MessageBubble(
                message,
                on_rewrite=lambda: self.show_rewrite_dialog(message),
            )
        return MessageBubble(
            message,
            on_regenerate=lambda: self.page.run_task(self.regenerate_message, message),
            translation=self.message_translations.get(message.id),
            rag_usage=self.message_rag_usage.get(message.id),
            on_translate=(
                lambda: self.page.run_task(self.translate_message, message)
                if message.state is MessageState.COMPLETED
                else None
            ),
        )

    async def translate_message(self, message: Message) -> None:
        try:
            await self.container.translations.request_translation(message.id, force=True)
            if self.selected_conversation_id == message.conversation_id:
                await self._refresh_messages(message.conversation_id)
                self.page.update()
        except AppError as error:
            self._toast(str(error), ERROR)
        except Exception:
            self._toast("日本語訳を開始できませんでした。原文は保持されています。", ERROR)

    async def _on_translation_update(self, message_id: str) -> None:
        if self._generation_task is not None:
            return
        conversation_id = self.selected_conversation_id
        if conversation_id is None:
            return
        messages = await self.container.conversations.messages(conversation_id)
        if not any(message.id == message_id for message in messages):
            return
        await self._refresh_messages(conversation_id)
        self.page.update()

    def _render_guard(self) -> None:
        connection = self._selected_connection()
        provider_name = connection.provider_name if connection else ""
        cloud_disabled = bool(
            provider_name
            and self.container.profiles.cloud_is_disabled(provider_name)
        )
        if cloud_disabled and self.models:
            label, color, icon = "無料運営: 保護中", MINT, ft.Icons.CHECK_CIRCLE_ROUNDED
            detail = (
                f"送信先 {connection.endpoint} / 利用可能モデル {len(self.models)}件"
                if connection
                else "接続先を選択してください。"
            )
        else:
            label, color, icon = "無料運営: 送信停止", ERROR, ft.Icons.ERROR_ROUNDED
            config_path = self.container.paths.ollama_server_config_path
            if self.guard_error:
                detail = self.guard_error
            elif connection and not cloud_disabled and not connection.is_builtin:
                detail = (
                    "相手端末でOllama Cloudを無効化し、接続先の編集画面で"
                    "確認済みにしてください。"
                )
            elif cloud_disabled:
                detail = "利用条件を確認できるローカルモデルがありません。Ollamaへモデルを追加してください。"
            else:
                detail = (
                    f"{config_path} に disable_ollama_cloud: true を設定し、"
                    "Ollamaを再起動してください。"
                )
        self.guard_badge.content = ft.Row(
            [ft.Icon(icon, color=color, size=18), ft.Text(label, color=color, size=13)],
            spacing=7,
        )
        self.guard_badge.bgcolor = f"{color}18"
        self.guard_badge.border_radius = 12
        self.guard_badge.padding = ft.Padding(12, 9, 12, 9)
        self.guard_detail.value = detail
        self.new_conversation_button.disabled = not self._connection_ready()

    def _render_conversations(self) -> None:
        controls: list[ft.Control] = []
        for conversation in self.conversations:
            selected = conversation.id == self.selected_conversation_id
            controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                conversation.title,
                                size=13,
                                color=TEXT if selected else "#B8B3AA",
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                conversation.updated_at.astimezone().strftime("%m/%d %H:%M"),
                                size=9,
                                color="#706C65",
                            ),
                        ],
                        spacing=2,
                    ),
                    bgcolor="#2A2925" if selected else None,
                    border_radius=12,
                    padding=ft.Padding(12, 9, 12, 9),
                    on_click=lambda _, cid=conversation.id: self.page.run_task(
                        self.select_conversation, cid
                    ),
                )
            )
        self.conversation_list.controls = controls

    def _render_options(self) -> None:
        recommended = recommend_chat_model(self.models)
        recommended_name = recommended.name if recommended else None
        self.character_dropdown.options = [
            ft.DropdownOption(character.id, character.display_name)
            for character in self.characters
        ]
        self.connection_dropdown.options = [
            ft.DropdownOption(connection.provider_name, connection.display_name)
            for connection in self.connections
        ]
        self.connection_dropdown.value = self.selected_provider_name
        self.model_dropdown.options = [
            ft.DropdownOption(
                model.name,
                model_option_label(model, recommended_name),
            )
            for model in self.models
        ]

    async def update_selection(self) -> None:
        if self.selected_conversation_id is None:
            return
        character_id = self.character_dropdown.value
        provider_name = self.connection_dropdown.value
        model_name = self.model_dropdown.value
        if not character_id or not provider_name or not model_name:
            return
        try:
            await self.container.conversations.update_selection(
                self.selected_conversation_id,
                character_id,
                provider_name,
                model_name,
            )
            await self.refresh_all()
            self._toast("会話の設定を保存しました。", MINT)
        except AppError as error:
            self._toast(str(error), ERROR)

    async def show_group_chat_dialog(self) -> None:
        conversation_id = self.selected_conversation_id
        if conversation_id is None or self._generation_task is not None:
            return
        configuration = await self.container.group_configuration.get(
            conversation_id
        )

        async def save(value: GroupChatSettingsInput) -> None:
            try:
                await self.container.group_configuration.save(
                    conversation_id,
                    character_version_ids=value.character_version_ids,
                    enabled=value.enabled,
                    mode=value.mode,
                    spotlight_character_id=value.spotlight_character_id,
                )
                self.page.pop_dialog()
                await self.refresh_all()
                self._toast("グループ会話の設定を保存しました。", MINT)
            except AppError as error:
                self._toast(str(error), ERROR)

        self.page.show_dialog(
            GroupChatSettingsDialog(
                configuration,
                self.characters,
                save,
                self._close_dialog,
            )
        )

    async def activate_branch(self) -> None:
        if self.selected_conversation_id is None or not self.branch_dropdown.value:
            return
        try:
            await self.container.conversations.activate_branch(
                self.selected_conversation_id, self.branch_dropdown.value
            )
            await self.refresh_all()
        except AppError as error:
            self._toast(str(error), ERROR)

    async def update_auto_translate(self) -> None:
        conversation_id = self.selected_conversation_id
        if conversation_id is None or self._generation_task is not None:
            return
        try:
            await self.container.conversations.set_auto_translate(
                conversation_id, bool(self.auto_translate_switch.value)
            )
            await self.refresh_all()
            label = "ON" if self.auto_translate_switch.value else "OFF"
            self._toast(f"この会話の自動翻訳を{label}にしました。", MINT)
        except AppError as error:
            await self.refresh_all()
            self._toast(str(error), ERROR)

    async def show_branch_management(self) -> None:
        conversation = self._selected_conversation()
        if conversation is None or self._generation_task is not None:
            return
        branches = await self.container.conversations.all_branches(conversation.id)
        items: list[ft.Control] = []
        for index, branch in enumerate(branches):
            is_active = branch.id == conversation.active_branch_id
            is_root = branch.parent_branch_id is None
            status = "使用中" if is_active else "非表示" if branch.hidden_at else "表示中"

            async def change_visibility(
                branch_id: str = branch.id,
                restore: bool = branch.hidden_at is not None,
            ) -> None:
                try:
                    if restore:
                        await self.container.conversations.restore_branch(
                            conversation.id, branch_id
                        )
                    else:
                        await self.container.conversations.hide_branch(
                            conversation.id, branch_id
                        )
                    self.page.pop_dialog()
                    await self.refresh_all()
                    self._toast(
                        "分岐を表示しました。" if restore else "分岐を非表示にしました。",
                        MINT,
                    )
                except AppError as error:
                    self._toast(str(error), ERROR)

            action_label = "表示に戻す" if branch.hidden_at else "非表示にする"
            items.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Column(
                                [
                                    ft.Text(f"分岐 {index + 1}", color=TEXT),
                                    ft.Text(
                                        f"{status}{' · root' if is_root else ''}",
                                        size=10,
                                        color=MUTED,
                                    ),
                                ],
                                expand=True,
                                spacing=2,
                            ),
                            ft.Button(
                                action_label,
                                color=MUTED,
                                bgcolor="#302E29",
                                disabled=is_active or is_root,
                                on_click=change_visibility,
                            ),
                        ]
                    ),
                    bgcolor="#2A2925",
                    border_radius=12,
                    padding=12,
                )
            )
        self.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                title="分岐管理",
                content=ft.ListView(items, spacing=8, width=460, height=360),
                bgcolor="#24231F",
                actions=[ft.Button("閉じる", on_click=self._close_dialog)],
            )
        )

    async def send_message(self) -> None:
        content = (self.composer.value or "").strip()
        conversation_id = self.selected_conversation_id
        if not content or conversation_id is None or self._generation_task is not None:
            return
        self.composer.value = ""

        async def action(on_update: Callable[[str], Awaitable[None]]) -> Message:
            return await self.container.chat.send_message(
                conversation_id, content, on_update, self._context_notice
            )

        await self._run_generation(conversation_id, action, content)

    async def choose_tool_folder(self) -> None:
        conversation_id = self.selected_conversation_id
        if conversation_id is None:
            self._toast("会話を選択してからフォルダーを許可してください。", ERROR)
            return
        folder = await self.file_picker.get_directory_path(
            dialog_title="この会話で読取りを許可するフォルダー"
        )
        if not folder:
            return
        try:
            await self.container.tool_access.grant_folder(conversation_id, folder)
        except AppError as error:
            self._toast(str(error), ERROR)
            return
        await self._refresh_tools()
        self.page.update(self.tool_card)
        self._toast("この会話にフォルダーの読取りを許可しました。", MINT)

    async def revoke_tool_folder(self) -> None:
        conversation_id = self.selected_conversation_id
        if conversation_id is None:
            return
        await self.container.tool_access.revoke_folder(conversation_id)
        await self._refresh_tools()
        self.page.update(self.tool_card)
        self._toast("フォルダーの許可を取り消しました。", MINT)

    async def show_computer_use_review(self) -> None:
        conversation_id = self.selected_conversation_id
        if conversation_id is None:
            self._toast("先に会話を選択してください。", ERROR)
            return
        try:
            plan, run = await self.container.computer_use.stage_fake_notepad_plan(
                conversation_id, "安全なFake実行🙂"
            )
        except AppError as error:
            self._toast(str(error), ERROR)
            return

        async def approve(_event: Any = None) -> None:
            self._cancel_computer_use_approval_timer()
            try:
                completed = (
                    await self.container.computer_use.approve_and_execute_fake(
                        run.id, plan, self._fake_security_context()
                    )
                )
            except AppError as error:
                dialog.set_remaining_seconds(0)
                self.page.update(dialog)
                self._toast(str(error), ERROR)
                return
            self.page.pop_dialog()
            await self._refresh_computer_use()
            self.page.update(self.computer_use_card)
            if completed.failure_reason:
                self._toast(
                    f"Fake安全テスト: {completed.state.value} "
                    f"({completed.failure_reason})",
                    ERROR,
                )
            else:
                self._toast("Fake安全テスト完了。OS入力は0件です。", MINT)

        async def cancel(_event: Any = None) -> None:
            self._cancel_computer_use_approval_timer()
            try:
                await self.container.computer_use.cancel(run.id)
            except AppError as error:
                self._toast(str(error), ERROR)
                return
            self.page.pop_dialog()
            await self._refresh_computer_use()
            self.page.update(self.computer_use_card)
            self._toast("Fake計画を取り消しました。OS入力は0件です。", MUTED)

        dialog = ComputerUseReviewDialog(
            plan,
            self._fake_security_context(),
            isolation_ready=False,
            fake_only=True,
            on_approve=approve,
            on_cancel=cancel,
        )
        self.page.show_dialog(dialog)
        self._computer_use_approval_task = asyncio.create_task(
            self._watch_computer_use_approval(run.id, dialog)
        )

    async def _watch_computer_use_approval(
        self, run_id: str, dialog: ComputerUseReviewDialog
    ) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                remaining = (
                    await self.container.computer_use.approval_remaining_seconds(
                        run_id
                    )
                )
                dialog.set_remaining_seconds(remaining)
                self.page.update(dialog)
                if remaining <= 0:
                    await self.container.computer_use.expire(run_id)
                    self.page.pop_dialog()
                    await self._refresh_computer_use()
                    self.page.update(self.computer_use_card)
                    self._toast(
                        "承認期限が切れました。Fake実行は開始していません。",
                        ERROR,
                    )
                    return
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            return
        except AppError as error:
            dialog.set_remaining_seconds(0)
            self.page.update(dialog)
            self._toast(str(error), ERROR)
        finally:
            if self._computer_use_approval_task is current_task:
                self._computer_use_approval_task = None

    def _cancel_computer_use_approval_timer(self) -> None:
        if self._computer_use_approval_task is not None:
            self._computer_use_approval_task.cancel()
            self._computer_use_approval_task = None

    async def show_computer_use_audit_dialog(self) -> None:
        conversation_id = self.selected_conversation_id
        if conversation_id is None:
            return
        self.computer_use_runs = await self.container.computer_use.history(
            conversation_id
        )
        items: list[ft.Control] = []
        for run in self.computer_use_runs:
            actions = await self.container.computer_use.actions(run.id)
            action_lines = "\n".join(
                f"{action.ordinal}. {action.request.action_type.value}: "
                f"{action.state.value}"
                for action in actions
            )
            completed = (
                run.completed_at.astimezone().strftime("%Y/%m/%d %H:%M:%S")
                if run.completed_at is not None
                else "未完了"
            )
            reason = f"\n理由: {run.failure_reason}" if run.failure_reason else ""
            items.append(
                ft.Container(
                    bgcolor="#2A2925",
                    border_radius=12,
                    padding=12,
                    content=ft.Column(
                        [
                            ft.Text(
                                f"{run.state.value} · OS入力0件",
                                size=12,
                                color=MINT if not run.failure_reason else ERROR,
                                weight=ft.FontWeight.W_600,
                            ),
                            ft.Text(
                                f"{run.objective}\n{action_lines}\n完了: {completed}"
                                f"{reason}",
                                size=10,
                                color=MUTED,
                                selectable=True,
                            ),
                        ],
                        spacing=5,
                    ),
                )
            )
        self.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                title="Computer Use監査（Fakeのみ）",
                bgcolor="#24231F",
                content=ft.ListView(items, spacing=8, width=560, height=420),
                actions=[ft.Button("閉じる", on_click=self._close_dialog)],
            )
        )

    @staticmethod
    def _fake_security_context() -> DesktopSecurityContext:
        # These are contract-test values, not observations of the host OS.
        return DesktopSecurityContext(
            "fake-standard-user",
            "fake-standard-user",
            1,
            1,
            DesktopIntegrityLevel.MEDIUM,
            DesktopIntegrityLevel.MEDIUM,
            False,
            False,
            False,
            False,
        )

    async def show_tool_audit_dialog(self) -> None:
        conversation_id = self.selected_conversation_id
        if conversation_id is None:
            return
        self.tool_audits = await self.container.tool_access.audits(conversation_id)
        rows: list[ft.Control] = []
        for audit in reversed(self.tool_audits):
            details = [
                f"状態: {audit.state.value}",
                f"入力: {json.dumps(audit.input_arguments, ensure_ascii=False)}",
            ]
            if audit.result_size_bytes is not None:
                details.append(
                    f"結果: {audit.result_item_count or 0}件 / "
                    f"{audit.result_size_bytes} bytes / SHA-256 {audit.result_sha256}"
                )
            if audit.failure_reason:
                details.append(f"理由: {audit.failure_reason}")
            details.append(
                "開始: "
                + (
                    audit.started_at.astimezone().strftime("%Y/%m/%d %H:%M:%S")
                    if audit.started_at
                    else "未開始"
                )
            )
            details.append(
                "終了: "
                + (
                    audit.completed_at.astimezone().strftime("%Y/%m/%d %H:%M:%S")
                    if audit.completed_at
                    else "未終了"
                )
            )
            rows.append(
                ft.Container(
                    bgcolor="#2A2925",
                    border_radius=10,
                    padding=10,
                    content=ft.Column(
                        [
                            ft.Text(audit.tool_name, size=12, color=TEXT),
                            *[ft.Text(line, size=9, color=MUTED, selectable=True) for line in details],
                        ],
                        spacing=2,
                    ),
                )
            )
        self.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                title="内蔵ツール監査履歴",
                bgcolor="#24231F",
                content=ft.ListView(
                    rows or [ft.Text("監査履歴はありません。", color=MUTED)],
                    spacing=7,
                    width=560,
                    height=420,
                ),
                actions=[ft.Button("閉じる", on_click=self._close_dialog)],
            )
        )

    async def show_rag_dialog(self) -> None:
        conversation_id = self.selected_conversation_id
        if conversation_id is None:
            self._toast("会話を選択してから参照資料を設定してください。", ERROR)
            return
        selected_ids = {document.id for document in self.selected_rag_documents}
        checkboxes: list[ft.Checkbox] = []
        document_list = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO, height=250)

        def render_documents() -> None:
            checkboxes.clear()
            controls: list[ft.Control] = []
            for document in self.rag_documents:
                checkbox = ft.Checkbox(
                    label=document.title,
                    value=document.id in selected_ids,
                    data=document.id,
                    active_color=MINT,
                )
                checkboxes.append(checkbox)
                controls.append(checkbox)
            document_list.controls = controls or [
                ft.Text("登録済みの資料はありません。", size=12, color=MUTED)
            ]

        render_documents()

        async def add_document() -> None:
            picked = await self.file_picker.pick_files(
                dialog_title="ローカル参照資料を追加",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["txt", "md"],
                allow_multiple=False,
                with_data=True,
            )
            if not picked:
                return
            selected_file = picked[0]
            if selected_file.bytes is None:
                self._toast("選択した文書を読み取れませんでした。", ERROR)
                return
            try:
                document = await self.container.rag.register_document(
                    selected_file.name, selected_file.bytes
                )
            except AppError as error:
                self._toast(str(error), ERROR)
                return
            self.rag_documents = await self.container.rag.documents()
            selected_ids.add(document.id)
            render_documents()
            self.page.update(document_list)

        async def save_selection() -> None:
            chosen_ids = [
                str(checkbox.data) for checkbox in checkboxes if checkbox.value
            ]
            try:
                await self.container.rag.select_documents(conversation_id, chosen_ids)
                self.selected_rag_documents = (
                    await self.container.rag.selected_documents(conversation_id)
                )
            except AppError as error:
                self._toast(str(error), ERROR)
                return
            self.page.pop_dialog()
            self._render_rag_selection()
            self.page.update()

        self.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                title="参照資料",
                bgcolor="#24231F",
                content=ft.Column(
                    [
                        ft.Text(
                            "選択した資料だけをこの会話のローカル検索に使います。",
                            size=11,
                            color=MUTED,
                        ),
                        document_list,
                        ft.Button(
                            "TXT / Markdownを追加",
                            icon=ft.Icons.ADD_ROUNDED,
                            color=TEXT,
                            bgcolor="#35342F",
                            on_click=add_document,
                        ),
                    ],
                    tight=True,
                    width=460,
                ),
                actions=[
                    ft.Button("キャンセル", on_click=self._close_dialog),
                    ft.Button(
                        "この会話で使う",
                        bgcolor=ACCENT,
                        color="#17120D",
                        on_click=save_selection,
                    ),
                ],
            )
        )

    def _render_rag_selection(self) -> None:
        count = len(self.selected_rag_documents)
        self.rag_button.badge = str(count) if count else None
        self.rag_selection_bar.visible = count > 0
        if count == 0:
            self.rag_selection_text.value = ""
            return
        names = [document.title for document in self.selected_rag_documents]
        summary = ", ".join(names[:2])
        if count > 2:
            summary += f" ほか{count - 2}件"
        self.rag_selection_text.value = f"参照候補: {summary}"

    async def regenerate_message(self, source: Message) -> None:
        conversation_id = self.selected_conversation_id
        if conversation_id is None or self._generation_task is not None:
            return

        async def action(on_update: Callable[[str], Awaitable[None]]) -> Message:
            return await self.container.chat.regenerate_message(
                conversation_id, source.id, on_update, self._context_notice
            )

        await self._run_generation(conversation_id, action)

    async def regenerate_turn_batch(self, source: Message) -> None:
        conversation = self._selected_conversation()
        if (
            conversation is None
            or conversation.id != source.conversation_id
            or self._generation_task is not None
        ):
            return
        conversation_id = conversation.id
        expected_active_branch_id = conversation.active_branch_id

        async def action(on_update: Callable[[str], Awaitable[None]]) -> Message:
            return await self.container.chat.regenerate_turn_batch(
                conversation_id,
                source.id,
                expected_active_branch_id,
                on_update,
                self._context_notice,
            )

        await self._run_generation(conversation_id, action)

    async def _run_generation(
        self,
        conversation_id: str,
        action: Callable[[Callable[[str], Awaitable[None]]], Awaitable[Message]],
        optimistic_user_text: str | None = None,
    ) -> None:
        current = asyncio.current_task()
        if current is None:
            return
        self._generation_task = current
        self._generation_conversation_id = conversation_id
        self._set_busy(True)
        self.telemetry_response.value = "観測中…"
        self.telemetry_speed.value = "観測中…"
        if optimistic_user_text:
            self.message_list.controls.append(
                self._message_bubble(self._temporary_message(MessageRole.USER, optimistic_user_text))
            )
        streaming_text = ft.Text("", size=14, color=TEXT, selectable=True)
        self.message_list.controls.append(
            MessageBubble(
                self._temporary_message(MessageRole.ASSISTANT, "", MessageState.STREAMING),
                streaming_text=streaming_text,
            )
        )
        self.page.update()

        async def on_update(content: str) -> None:
            streaming_text.value = content
            if self.selected_conversation_id == conversation_id:
                self.page.update(streaming_text)

        try:
            await action(on_update)
        except asyncio.CancelledError:
            self._toast("生成を停止しました。途中までの応答は保存されています。", ACCENT)
        except AppError as error:
            self._toast(str(error), ERROR)
        except Exception:
            self._toast("予期しないエラーが発生しました。実行記録を保存しました。", ERROR)
        finally:
            self._generation_task = None
            self._generation_conversation_id = None
            self._set_busy(False)
            await self.refresh_all()

    async def stop_generation(self) -> None:
        task = self._generation_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def restart_app(self) -> None:
        if self._generation_task is not None:
            self._toast("生成中は再起動できません。停止してから再試行してください。", ERROR)
            return
        self.restart_button.disabled = True
        self.restart_button_label.value = "再起動を準備中…"
        self.page.update(self.restart_button)
        try:
            ready = await self.container.restart.restart()
        except Exception:
            ready = False
        if ready:
            await cast(WindowCloser, self.page.window).close()
            return
        self.restart_button.disabled = False
        self.restart_button_label.value = "アプリを再起動"
        self.page.update(self.restart_button)
        self._toast("新しいアプリを起動できませんでした。現在の画面は継続できます。", ERROR)

    async def on_keyboard_event(self, event: ft.KeyboardEvent) -> None:
        if event.ctrl and event.shift and event.key.lower() == "r":
            await self.restart_app()

    def show_new_conversation_dialog(self) -> None:
        if not self.characters or not self.models or not self._connection_ready():
            self._toast("無料運営の確認後に会話を作成できます。", ERROR)
            return
        title = ft.TextField(label="会話名", value="新しい会話", autofocus=True)
        character = self._dropdown("キャラクター")
        character.options = [
            ft.DropdownOption(item.id, item.display_name) for item in self.characters
        ]
        character.value = self.characters[0].id
        connection = self._dropdown("Ollama接続先")
        connection.options = [
            ft.DropdownOption(item.provider_name, item.display_name)
            for item in self.connections
        ]
        connection.value = self.selected_provider_name
        model = self._dropdown("ローカルモデル")
        recommended = recommend_chat_model(self.models)
        recommended_name = recommended.name if recommended else None
        model.options = [
            ft.DropdownOption(
                item.name,
                model_option_label(item, recommended_name),
            )
            for item in self.models
        ]
        model.value = recommended_name

        async def change_connection() -> None:
            if not connection.value:
                return
            try:
                dialog_models = await self.container.profiles.list_visible_models(
                    connection.value
                )
            except AppError as error:
                model.options = []
                model.value = None
                self._toast(str(error), ERROR)
                self.page.update()
                return
            dialog_recommended = recommend_chat_model(dialog_models)
            dialog_recommended_name = (
                dialog_recommended.name if dialog_recommended else None
            )
            model.options = [
                ft.DropdownOption(
                    item.name,
                    model_option_label(item, dialog_recommended_name),
                )
                for item in dialog_models
            ]
            model.value = dialog_recommended_name
            self.page.update()

        connection.on_select = change_connection

        async def create() -> None:
            if not character.value or not connection.value or not model.value:
                return
            try:
                conversation = await self.container.conversations.create_conversation(
                    (title.value or "").strip() or "新しい会話",
                    character.value,
                    connection.value,
                    model.value,
                )
                self.page.pop_dialog()
                self.selected_conversation_id = conversation.id
                await self.refresh_all()
            except AppError as error:
                self._toast(str(error), ERROR)

        dialog = ft.AlertDialog(
            modal=True,
            title="新しい会話",
            bgcolor="#24231F",
            content=ft.Column(
                [title, character, connection, model], tight=True, width=420
            ),
            actions=[
                ft.Button("キャンセル", on_click=self._close_dialog),
                ft.Button("作成", bgcolor=ACCENT, color="#17120D", on_click=create),
            ],
        )
        self.page.show_dialog(dialog)

    def show_connection_dialog(self) -> None:
        selected = self._selected_connection()
        editable = selected if selected and not selected.is_builtin else None
        name = ft.TextField(
            label="表示名",
            value=editable.display_name if editable else "",
            autofocus=True,
        )
        endpoint = ft.TextField(
            label="Ollamaのアドレス",
            value=editable.endpoint if editable else "http://192.168.1.:11434",
        )
        cloud_confirmed = ft.Checkbox(
            label="相手端末でOllama Cloudを無効化済み",
            value=editable.cloud_disabled_confirmed if editable else False,
        )

        async def save() -> None:
            try:
                saved = await self.container.profiles.save_connection(
                    editable.id if editable else None,
                    name.value or "",
                    endpoint.value or "",
                    bool(cloud_confirmed.value),
                )
                self.page.pop_dialog()
                self.selected_provider_name = saved.provider_name
                await self.refresh_all()
                self._toast("Ollama接続先を保存しました。", MINT)
            except AppError as error:
                self._toast(str(error), ERROR)

        self.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                title="Ollama接続先を編集" if editable else "Ollama接続先を追加",
                bgcolor="#24231F",
                content=ft.Column(
                    [
                        name,
                        endpoint,
                        ft.Text(
                            "127.0.0.1または192.168.x.xなど、LAN内のIPだけを指定できます。",
                            size=10,
                            color=MUTED,
                        ),
                        cloud_confirmed,
                    ],
                    tight=True,
                    width=460,
                ),
                actions=[
                    ft.Button("キャンセル", on_click=self._close_dialog),
                    ft.Button("保存", bgcolor=ACCENT, color="#17120D", on_click=save),
                ],
            )
        )

    async def show_model_role_dialog(self) -> None:
        configuration = await self.container.embeddings.configuration()
        desired = configuration.desired
        connection = self._dropdown("埋め込み接続先")
        connection.options = [
            ft.DropdownOption(item.provider_name, item.display_name)
            for item in self.connections
        ]
        connection.value = (
            desired.provider_name
            if desired is not None
            and any(
                item.provider_name == desired.provider_name
                for item in self.connections
            )
            else None
        )
        model = self._dropdown("埋め込みモデル")
        status = ft.Text(size=11, color=MUTED)
        if desired is not None and desired.last_error:
            status.value = f"前回の失敗: {desired.last_error}"
            status.color = ERROR
        elif desired is None:
            status.value = "未設定です。保存するまでは語句検索だけを使用します。"
        else:
            status.value = self._embedding_status(configuration)[0]
        save_button = ft.Button(
            "保存して索引を構築",
            bgcolor=ACCENT,
            color="#17120D",
        )

        async def load_models() -> None:
            provider_name = connection.value
            model.options = []
            model.value = None
            if not provider_name:
                status.value = "埋め込み接続先を選択してください。"
                status.color = MUTED
                self.page.update()
                return
            status.value = "接続確認中..."
            status.color = ACCENT
            save_button.disabled = True
            self.page.update()
            try:
                models = await self.container.embeddings.list_models(provider_name)
            except AppError as error:
                status.value = str(error)
                status.color = ERROR
                self.page.update()
                return
            finally:
                save_button.disabled = False
            model.options = [
                ft.DropdownOption(item.name, item.name) for item in models
            ]
            if (
                desired is not None
                and desired.provider_name == provider_name
                and any(item.name == desired.model_name for item in models)
            ):
                model.value = desired.model_name
            status.value = (
                f"埋め込み対応モデル {len(models)}件"
                if models
                else "この接続先に埋め込み対応モデルがありません。"
            )
            status.color = MINT if models else ERROR
            self.page.update()

        async def save() -> None:
            if not connection.value or not model.value:
                status.value = "接続先と埋め込みモデルを選択してください。"
                status.color = ERROR
                self.page.update()
                return
            save_button.disabled = True
            status.value = "接続・日本語検索品質を確認中..."
            status.color = ACCENT
            self.page.update()
            try:
                await self.container.embeddings.configure(
                    connection.value, model.value
                )
            except AppError as error:
                status.value = str(error)
                status.color = ERROR
                save_button.disabled = False
                self.page.update()
                return
            self.page.pop_dialog()
            await self._refresh_embedding_status()
            self.page.update()
            self._toast("埋め込み設定を保存し、索引構築を開始しました。", MINT)

        connection.on_select = load_models
        save_button.on_click = save
        self.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                title="用途別モデル設定",
                bgcolor="#24231F",
                content=ft.Column(
                    [
                        ft.Container(
                            padding=10,
                            border=ft.Border.all(1, "#3A3934"),
                            border_radius=10,
                            content=ft.Row(
                                [
                                    ft.Text(
                                        "会話生成",
                                        width=100,
                                        weight=ft.FontWeight.W_600,
                                    ),
                                    ft.Text("会話ごとに設定", color=MUTED),
                                ],
                            ),
                        ),
                        ft.Container(
                            padding=10,
                            border=ft.Border.all(1, "#3A3934"),
                            border_radius=10,
                            content=ft.Column(
                                [
                                    ft.Text(
                                        "埋め込み",
                                        weight=ft.FontWeight.W_600,
                                        color=ACCENT,
                                    ),
                                    connection,
                                    model,
                                    status,
                                ],
                                spacing=8,
                            ),
                        ),
                        ft.Text(
                            "DGXを選ぶと、選択したRAG資料と検索文を確認済みLAN端末へ送ります。"
                            "モデル変更後は新索引を構築し、完成まで旧索引または語句検索を使います。",
                            size=10,
                            color=MUTED,
                        ),
                    ],
                    tight=True,
                    width=520,
                ),
                actions=[
                    ft.Button("閉じる", on_click=self._close_dialog),
                    save_button,
                ],
            )
        )
        if connection.value:
            await load_models()

    def show_new_character_dialog(self) -> None:
        self.show_character_dialog(create_new=True)

    def show_selected_character_dialog(self) -> None:
        self.show_character_dialog(create_new=False)

    def show_character_dialog(self, *, create_new: bool = False) -> None:
        current = None
        if not create_new:
            current = next(
                (item for item in self.characters if item.id == self.character_dropdown.value),
                None,
            )
        name = ft.TextField(label="名前", value=current.display_name if current else "")
        prompt = ft.TextField(
            label="ふるまい・話し方",
            value=current.system_prompt if current else "",
            multiline=True,
            min_lines=5,
            max_lines=10,
        )

        async def save() -> None:
            try:
                saved = await self.container.profiles.save_character(
                    name.value or "",
                    prompt.value or "",
                    current.character_id if current else None,
                )
                self.page.pop_dialog()
                await self.refresh_all()
                self.character_dropdown.value = saved.id
                self.page.update()
                self._toast("キャラクターを版として保存しました。", MINT)
            except AppError as error:
                self._toast(str(error), ERROR)

        dialog = ft.AlertDialog(
            modal=True,
            title="新しいキャラクター" if create_new else "キャラクター設定",
            bgcolor="#24231F",
            content=ft.Column([name, prompt], tight=True, width=460),
            actions=[
                ft.Button("キャンセル", on_click=self._close_dialog),
                ft.Button(
                    "追加" if create_new else "版を保存",
                    bgcolor=ACCENT,
                    color="#17120D",
                    on_click=save,
                ),
            ],
        )
        self.page.show_dialog(dialog)

    def show_rewrite_dialog(self, source: Message) -> None:
        content = ft.TextField(
            label="書き直した内容",
            value=source.content,
            multiline=True,
            min_lines=3,
            max_lines=8,
            autofocus=True,
        )

        async def rewrite() -> None:
            conversation_id = self.selected_conversation_id
            replacement = (content.value or "").strip()
            if conversation_id is None or not replacement:
                return
            self.page.pop_dialog()

            async def action(on_update: Callable[[str], Awaitable[None]]) -> Message:
                return await self.container.chat.rewrite_message(
                    conversation_id,
                    source.id,
                    replacement,
                    on_update,
                    self._context_notice,
                )

            await self._run_generation(conversation_id, action)

        self.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                title="元を残して書き直す",
                bgcolor="#24231F",
                content=content,
                actions=[
                    ft.Button("キャンセル", on_click=self._close_dialog),
                    ft.Button("新しい分岐を作る", bgcolor=ACCENT, color="#17120D", on_click=rewrite),
                ],
            )
        )

    def confirm_archive(self) -> None:
        if self.selected_conversation_id is None:
            return

        async def archive() -> None:
            conversation_id = self.selected_conversation_id
            self.page.pop_dialog()
            if conversation_id is not None:
                await self.container.conversations.archive(conversation_id)
                self.selected_conversation_id = None
                await self.refresh_all()
                self._toast("会話をゴミ箱へ移動しました。元に戻せます。", MINT)

        self.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                title="会話をゴミ箱へ移動しますか？",
                content=ft.Text("一覧から隠れますが、ゴミ箱から元に戻せます。"),
                bgcolor="#24231F",
                actions=[
                    ft.Button("やめる", on_click=self._close_dialog),
                    ft.Button("ゴミ箱へ移動", bgcolor="#56342E", color=TEXT, on_click=archive),
                ],
            )
        )

    async def show_archive_dialog(self) -> None:
        archived = await self.container.conversations.list_archived_conversations()
        if not archived:
            self._toast("ゴミ箱は空です。", MUTED)
            return

        items: list[ft.Control] = []
        for conversation in archived:
            async def restore(conversation_id: str = conversation.id) -> None:
                await self.container.conversations.restore(conversation_id)
                self.page.pop_dialog()
                self.selected_conversation_id = conversation_id
                await self.refresh_all()
                self._toast("会話を一覧へ戻しました。", MINT)

            items.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Column(
                                [
                                    ft.Text(conversation.title, color=TEXT),
                                    ft.Text(
                                        conversation.updated_at.astimezone().strftime(
                                            "%Y/%m/%d %H:%M"
                                        ),
                                        size=10,
                                        color=MUTED,
                                    ),
                                ],
                                expand=True,
                                spacing=2,
                            ),
                            ft.Button("戻す", color="#17120D", bgcolor=ACCENT, on_click=restore),
                        ]
                    ),
                    bgcolor="#2A2925",
                    border_radius=12,
                    padding=12,
                )
            )
        self.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                title="ゴミ箱",
                bgcolor="#24231F",
                content=ft.ListView(items, spacing=8, width=460, height=360),
                actions=[ft.Button("閉じる", on_click=self._close_dialog)],
            )
        )

    def _set_busy(self, busy: bool) -> None:
        self.stop_button.visible = busy
        self.send_button.visible = not busy
        self.restart_button.disabled = busy
        self.group_chat_button.disabled = (
            busy or self.selected_conversation_id is None
        )
        self.manage_branches_button.disabled = (
            busy or self.selected_conversation_id is None
        )
        self.auto_translate_switch.disabled = (
            busy or self.selected_conversation_id is None
        )
        self.archive_button.disabled = busy or self.selected_conversation_id is None
        self._set_composer_enabled(not busy and self.selected_conversation_id is not None)
        self.page.update()

    def _set_composer_enabled(self, enabled: bool) -> None:
        enabled = enabled and bool(self.models) and self._connection_ready()
        self.composer.disabled = not enabled
        self.send_button.disabled = not enabled
        self.rag_button.disabled = self.selected_conversation_id is None

    def _show_empty_state(self, message: str) -> None:
        self.message_list.controls = [
            ft.Container(
                expand=True,
                alignment=ft.Alignment.CENTER,
                content=ft.Column(
                    [
                        ft.Icon(ft.Icons.SHIELD_ROUNDED, color="#605D57", size=36),
                        ft.Text(message, color=MUTED, size=13, text_align=ft.TextAlign.CENTER),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )
        ]

    def _toast(self, message: str, color: str) -> None:
        self.page.show_dialog(
            ft.SnackBar(
                ft.Text(message, color=TEXT),
                bgcolor="#2A2925",
                close_icon_color=color,
                show_close_icon=True,
            )
        )

    async def _context_notice(self, message: str, is_warning: bool) -> None:
        self._toast(message, ERROR if is_warning else ACCENT)

    def _close_dialog(self) -> None:
        self.page.pop_dialog()
        self.page.update()

    def _selected_conversation(self) -> Conversation | None:
        return next(
            (
                conversation
                for conversation in self.conversations
                if conversation.id == self.selected_conversation_id
            ),
            None,
        )

    def _selected_connection(self) -> ProviderConnection | None:
        return next(
            (
                connection
                for connection in self.connections
                if connection.provider_name == self.selected_provider_name
            ),
            None,
        )

    def _connection_ready(self) -> bool:
        connection = self._selected_connection()
        return bool(
            connection
            and self.models
            and self.container.profiles.cloud_is_disabled(connection.provider_name)
            and self.guard_error is None
        )

    @staticmethod
    def _temporary_message(
        role: MessageRole,
        content: str,
        state: MessageState = MessageState.COMPLETED,
    ) -> Message:
        return Message(
            id=str(uuid4()),
            conversation_id="temporary",
            parent_message_id=None,
            source_message_id=None,
            role=role,
            content=content,
            state=state,
            created_at=utc_now(),
        )

    @staticmethod
    def _dropdown(label: str) -> ft.Dropdown:
        return ft.Dropdown(
            label=label,
            border=ft.InputBorder.NONE,
            filled=True,
            fill_color="#292925",
            color=TEXT,
            border_radius=12,
            content_padding=ft.Padding(12, 4, 10, 4),
            text_size=12,
        )

    @staticmethod
    def _section_label(text: str) -> ft.Text:
        return ft.Text(text, size=10, color="#716D65", weight=ft.FontWeight.W_600)

    @staticmethod
    def _telemetry_row(label: str, value: ft.Text, source: ft.Text) -> ft.Control:
        return ft.Column(
            [
                ft.Row(
                    [ft.Text(label, size=9, color=MUTED, width=38), value],
                    spacing=4,
                ),
                source,
            ],
            spacing=0,
        )

    async def close(self) -> None:
        self._cancel_computer_use_approval_timer()
        if self._generation_task is not None:
            self._generation_task.cancel()
        await self.container.close()

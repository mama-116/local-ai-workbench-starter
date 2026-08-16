from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flet as ft

from local_llm_chat.domain.models import Message, MessageRagUsage, Translation
from local_llm_chat.domain.states import (
    MessageRole,
    MessageState,
    TranslationState,
)


class MessageBubble(ft.Container):
    def __init__(
        self,
        message: Message,
        on_rewrite: Callable[[], Any] | None = None,
        on_remember: Callable[[], Any] | None = None,
        on_regenerate: Callable[[], Any] | None = None,
        translation: Translation | None = None,
        on_translate: Callable[[], Any] | None = None,
        streaming_text: ft.Text | None = None,
        rag_usage: MessageRagUsage | None = None,
    ) -> None:
        is_user = message.role is MessageRole.USER
        body = streaming_text or ft.Text(
            message.content or "応答を準備しています...",
            size=14,
            color="#F3F0E8",
            selectable=True,
        )
        actions: list[ft.Control] = []
        if on_rewrite is not None:
            actions.append(
                ft.IconButton(
                    icon=ft.Icons.EDIT_ROUNDED,
                    icon_size=16,
                    icon_color="#AAA69D",
                    tooltip="この発言から書き直す",
                    on_click=on_rewrite,
                )
            )
        if on_remember is not None:
            actions.append(
                ft.IconButton(
                    icon=ft.Icons.BOOKMARK_ADD_ROUNDED,
                    icon_size=16,
                    icon_color="#AAA69D",
                    tooltip="覚えておいて",
                    on_click=on_remember,
                )
            )
        if on_regenerate is not None:
            actions.append(
                ft.IconButton(
                    icon=ft.Icons.REPLAY_ROUNDED,
                    icon_size=16,
                    icon_color="#AAA69D",
                    tooltip="別の応答を作る",
                    on_click=on_regenerate,
                )
            )
        if on_translate is not None and translation is None:
            actions.append(
                ft.IconButton(
                    icon=ft.Icons.TRANSLATE_ROUNDED,
                    icon_size=16,
                    icon_color="#AAA69D",
                    tooltip="日本語に訳す",
                    on_click=on_translate,
                )
            )

        state_text = _state_label(message.state)
        meta = ft.Row(
            [
                ft.Text(
                    "あなた" if is_user else "AI",
                    size=11,
                    weight=ft.FontWeight.W_600,
                    color="#F2A65A" if is_user else "#55C2A3",
                ),
                ft.Text(
                    message.created_at.astimezone().strftime("%H:%M"),
                    size=10,
                    color="#77736B",
                ),
                ft.Text(state_text, size=10, color="#D87866"),
                *actions,
            ],
            spacing=4,
        )
        content_controls: list[ft.Control] = [meta]
        if translation is None:
            content_controls.append(body)
        else:
            source_section = ft.Column(
                [
                    ft.Text(
                        "原文",
                        size=10,
                        weight=ft.FontWeight.W_600,
                        color="#8FBBB0",
                    ),
                    body,
                ],
                spacing=4,
            )
            toggle_source = ft.TextButton(
                "原文を隠す",
                style=ft.ButtonStyle(color="#AAA69D"),
            )

            def toggle_original() -> None:
                source_section.visible = not source_section.visible
                toggle_source.content = (
                    "原文を隠す" if source_section.visible else "原文を表示"
                )
                self.update()

            toggle_source.on_click = toggle_original
            translation_body = _translation_body(translation)
            translation_section = ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(
                                "日本語訳",
                                size=10,
                                weight=ft.FontWeight.W_600,
                                color="#F2A65A",
                            ),
                            ft.Text(
                                (
                                    f"再利用 · {translation.model}"
                                    if translation.reused_from_id
                                    else translation.model
                                ),
                                size=9,
                                color="#77736B",
                            ),
                        ],
                        spacing=6,
                    ),
                    translation_body,
                ],
                spacing=4,
            )
            translation_actions: list[ft.Control] = [toggle_source]
            if (
                on_translate is not None
                and translation.state
                in (TranslationState.COMPLETED, TranslationState.FAILED)
            ):
                translation_actions.append(
                    ft.TextButton(
                        "訳し直す",
                        icon=ft.Icons.REPLAY_ROUNDED,
                        style=ft.ButtonStyle(color="#AAA69D"),
                        on_click=on_translate,
                    )
                )
            content_controls.extend(
                [
                    source_section,
                    ft.Divider(height=1, color="#18FFFFFF"),
                    translation_section,
                    ft.Row(translation_actions, spacing=2),
                ]
            )

        if rag_usage is not None:
            content_controls.append(ft.Divider(height=1, color="#18FFFFFF"))
            if rag_usage.citations:
                evidence_controls: list[ft.Control] = []
                for citation in rag_usage.citations:
                    evidence_controls.append(
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        f"{citation.document_title} · "
                                        f"文字{citation.start_offset}-{citation.end_offset}",
                                        size=10,
                                        weight=ft.FontWeight.W_600,
                                        color="#8FBBB0",
                                    ),
                                    ft.Text(
                                        citation.content.strip(),
                                        size=11,
                                        color="#D7D2C8",
                                        selectable=True,
                                    ),
                                ],
                                spacing=5,
                            ),
                            bgcolor="#0DFFFFFF",
                            border_radius=8,
                            padding=ft.Padding(10, 8, 10, 8),
                        )
                    )
                evidence_section = ft.Column(
                    evidence_controls,
                    spacing=6,
                    visible=False,
                )
                toggle_evidence = ft.TextButton(
                    "根拠を表示",
                    icon=ft.Icons.EXPAND_MORE_ROUNDED,
                    style=ft.ButtonStyle(color="#AAA69D"),
                )

                def toggle_rag_evidence() -> None:
                    evidence_section.visible = not evidence_section.visible
                    toggle_evidence.content = (
                        "根拠を隠す" if evidence_section.visible else "根拠を表示"
                    )
                    toggle_evidence.icon = (
                        ft.Icons.EXPAND_LESS_ROUNDED
                        if evidence_section.visible
                        else ft.Icons.EXPAND_MORE_ROUNDED
                    )
                    self.update()

                toggle_evidence.on_click = toggle_rag_evidence
                content_controls.extend(
                    [
                        ft.Row(
                            [
                                ft.Icon(
                                    ft.Icons.CHECK_CIRCLE_OUTLINE_ROUNDED,
                                    size=14,
                                    color="#55C2A3",
                                ),
                                ft.Text(
                                    f"資料使用あり · {len(rag_usage.citations)}箇所",
                                    size=10,
                                    weight=ft.FontWeight.W_600,
                                    color="#8FBBB0",
                                ),
                                toggle_evidence,
                            ],
                            spacing=4,
                            wrap=True,
                        ),
                        evidence_section,
                    ]
                )
            else:
                content_controls.append(
                    ft.Row(
                        [
                            ft.Icon(
                                ft.Icons.SEARCH_OFF_ROUNDED,
                                size=14,
                                color="#AAA69D",
                            ),
                            ft.Column(
                                [
                                    ft.Text(
                                        "該当箇所なし",
                                        size=10,
                                        weight=ft.FontWeight.W_600,
                                        color="#B9B4AA",
                                    ),
                                    ft.Text(
                                        "選択した"
                                        f"{rag_usage.selected_document_count}件の資料は"
                                        "回答に使われませんでした",
                                        size=9,
                                        color="#77736B",
                                    ),
                                ],
                                spacing=2,
                            ),
                        ],
                        spacing=6,
                    )
                )

        card = ft.Container(
            content=ft.Column(content_controls, spacing=6),
            bgcolor="#2A2420" if is_user else "#1B2623",
            border_radius=16,
            padding=ft.Padding(16, 11, 16, 12),
            width=620,
        )
        super().__init__(
            content=card,
            alignment=ft.Alignment.CENTER_RIGHT if is_user else ft.Alignment.CENTER_LEFT,
            padding=ft.Padding(32, 4, 32, 4),
        )


def _translation_body(translation: Translation) -> ft.Control:
    if translation.state in (TranslationState.PENDING, TranslationState.RUNNING):
        label = (
            "日本語訳を準備しています..."
            if translation.state is TranslationState.PENDING
            else "ローカルモデルで翻訳しています..."
        )
        return ft.Row(
            [
                ft.ProgressRing(width=14, height=14, stroke_width=2, color="#F2A65A"),
                ft.Text(label, size=12, color="#AAA69D"),
            ],
            spacing=8,
        )
    if translation.state is TranslationState.FAILED:
        return ft.Text(
            "翻訳できませんでした。原文はそのまま残っています。",
            size=12,
            color="#D87866",
        )
    return ft.Text(
        translation.content,
        size=14,
        color="#F3F0E8",
        selectable=True,
    )
def _state_label(state: MessageState) -> str:
    return {
        MessageState.PENDING: "準備中",
        MessageState.STREAMING: "生成中",
        MessageState.COMPLETED: "",
        MessageState.CANCELLED: "停止済み",
        MessageState.FAILED: "失敗",
    }[state]

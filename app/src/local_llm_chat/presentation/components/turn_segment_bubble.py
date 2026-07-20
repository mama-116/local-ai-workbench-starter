from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import flet as ft

from local_llm_chat.domain.group_turns import TurnBatch, TurnSegment
from local_llm_chat.domain.models import Translation
from local_llm_chat.domain.states import (
    TranslationState,
    TurnBatchState,
    TurnSpeakerKind,
)


class TurnSegmentBubble(ft.Container):
    def __init__(
        self,
        segment: TurnSegment,
        batch: TurnBatch,
        created_at: datetime,
        *,
        is_batch_end: bool = False,
        on_regenerate: Callable[[], Any] | None = None,
        translation: Translation | None = None,
        on_translate: Callable[[], Any] | None = None,
    ) -> None:
        label, label_color = _speaker_label(segment)
        status = (
            "一部のみ"
            if is_batch_end and batch.state is TurnBatchState.PARTIAL
            else ""
        )
        meta_controls: list[ft.Control] = [
                ft.Text(
                    label,
                    size=11,
                    weight=ft.FontWeight.W_600,
                    color=label_color,
                ),
                ft.Text(
                    created_at.astimezone().strftime("%H:%M"),
                    size=10,
                    color="#77736B",
                ),
                ft.Text(status, size=10, color="#D87866"),
        ]
        if is_batch_end and on_regenerate is not None:
            regenerate_button = ft.IconButton(
                icon=ft.Icons.REPLAY_ROUNDED,
                icon_size=16,
                icon_color="#AAA69D",
                tooltip="このグループ応答全体を再生成",
            )

            def regenerate() -> None:
                if regenerate_button.disabled:
                    return
                regenerate_button.disabled = True
                on_regenerate()

            regenerate_button.on_click = regenerate
            meta_controls.append(regenerate_button)
        if is_batch_end and on_translate is not None:
            translation_button = ft.IconButton(
                icon=ft.Icons.TRANSLATE_ROUNDED,
                icon_size=16,
                icon_color="#AAA69D",
                tooltip=("日本語に訳す" if translation is None else "訳し直す"),
                disabled=(
                    translation is not None
                    and translation.state
                    in (TranslationState.PENDING, TranslationState.RUNNING)
                ),
                on_click=on_translate,
            )
            meta_controls.append(translation_button)
        meta = ft.Row(meta_controls, spacing=4)
        content_controls: list[ft.Control] = [
            meta,
            ft.Text(
                segment.content,
                size=14,
                color="#F3F0E8",
                selectable=True,
            ),
        ]
        if is_batch_end and translation is not None:
            content_controls.extend(
                [
                    ft.Divider(height=1, color="#18FFFFFF"),
                    ft.Text(
                        "TurnBatch全体の日本語訳",
                        size=10,
                        weight=ft.FontWeight.W_600,
                        color="#F2A65A",
                    ),
                    _translation_body(translation),
                ]
            )
        card = ft.Container(
            content=ft.Column(content_controls, spacing=6),
            bgcolor=(
                "#1D202B"
                if segment.speaker_kind is TurnSpeakerKind.NARRATOR
                else "#1B2623"
            ),
            border_radius=16,
            padding=ft.Padding(16, 11, 16, 12),
            width=620,
        )
        super().__init__(
            content=card,
            alignment=(
                ft.Alignment.CENTER
                if segment.speaker_kind is TurnSpeakerKind.NARRATOR
                else ft.Alignment.CENTER_LEFT
            ),
            padding=ft.Padding(32, 4, 32, 4),
        )


def _translation_body(translation: Translation) -> ft.Control:
    if translation.state is TranslationState.PENDING:
        return ft.Text("翻訳を待っています...", size=12, color="#AAA69D")
    if translation.state is TranslationState.RUNNING:
        return ft.Text("ローカルモデルで翻訳しています...", size=12, color="#AAA69D")
    if translation.state is TranslationState.FAILED:
        return ft.Text(
            "翻訳できませんでした。元のTurnBatchは変更していません。",
            size=12,
            color="#D87866",
        )
    return ft.Text(translation.content, size=13, color="#F3F0E8", selectable=True)


def _speaker_label(segment: TurnSegment) -> tuple[str, str]:
    if segment.speaker_kind is TurnSpeakerKind.NARRATOR:
        return segment.display_name, "#8FA7FF"
    if segment.speaker_kind is TurnSpeakerKind.GUEST:
        return f"{segment.display_name} · ゲスト", "#B6A0FF"
    if segment.speaker_kind is TurnSpeakerKind.UNRESOLVED:
        return f"{segment.display_name} · 未解決話者", "#D87866"
    return segment.display_name, "#55C2A3"

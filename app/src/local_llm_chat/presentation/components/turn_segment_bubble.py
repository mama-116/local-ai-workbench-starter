from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import flet as ft

from local_llm_chat.domain.group_turns import TurnBatch, TurnSegment
from local_llm_chat.domain.states import TurnBatchState, TurnSpeakerKind


class TurnSegmentBubble(ft.Container):
    def __init__(
        self,
        segment: TurnSegment,
        batch: TurnBatch,
        created_at: datetime,
        *,
        is_batch_end: bool = False,
        on_regenerate: Callable[[], Any] | None = None,
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
        meta = ft.Row(meta_controls, spacing=4)
        card = ft.Container(
            content=ft.Column(
                [
                    meta,
                    ft.Text(
                        segment.content,
                        size=14,
                        color="#F3F0E8",
                        selectable=True,
                    ),
                ],
                spacing=6,
            ),
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


def _speaker_label(segment: TurnSegment) -> tuple[str, str]:
    if segment.speaker_kind is TurnSpeakerKind.NARRATOR:
        return segment.display_name, "#8FA7FF"
    if segment.speaker_kind is TurnSpeakerKind.GUEST:
        return f"{segment.display_name} · ゲスト", "#B6A0FF"
    if segment.speaker_kind is TurnSpeakerKind.UNRESOLVED:
        return f"{segment.display_name} · 未解決話者", "#D87866"
    return segment.display_name, "#55C2A3"

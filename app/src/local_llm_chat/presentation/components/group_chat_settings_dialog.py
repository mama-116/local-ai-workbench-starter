from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import flet as ft

from local_llm_chat.domain.group_turns import ConversationGroupConfiguration
from local_llm_chat.domain.models import CharacterVersion
from local_llm_chat.domain.states import TurnMode


@dataclass(frozen=True, slots=True)
class GroupCharacterOption:
    character_id: str
    character_version_id: str
    display_name: str
    selected: bool


@dataclass(frozen=True, slots=True)
class GroupChatSettingsInput:
    character_version_ids: tuple[str, ...]
    enabled: bool
    mode: TurnMode
    spotlight_character_id: str | None


SaveCallback = Callable[[GroupChatSettingsInput], Awaitable[None]]


class GroupChatSettingsDialog(ft.AlertDialog):
    def __init__(
        self,
        configuration: ConversationGroupConfiguration,
        characters: list[CharacterVersion],
        on_save: SaveCallback,
        on_cancel: Callable[[], None],
    ) -> None:
        self.options = _character_options(configuration, characters)
        self.enabled_control = ft.Checkbox(
            label="グループ会話を使う",
            value=configuration.settings.enabled,
        )
        self.character_controls = tuple(
            ft.Checkbox(
                label=option.display_name,
                value=option.selected,
                data=option.character_version_id,
            )
            for option in self.options
        )
        self.mode_control = ft.Dropdown(
            label="会話モード",
            options=[
                ft.DropdownOption(TurnMode.STORY.value, "物語"),
                ft.DropdownOption(TurnMode.ROUND_TABLE.value, "ラウンドテーブル"),
                ft.DropdownOption(TurnMode.SPOTLIGHT.value, "スポットライト"),
            ],
            value=configuration.settings.mode.value,
            filled=True,
            fill_color="#292925",
            border=ft.InputBorder.NONE,
        )
        self.spotlight_control = ft.Dropdown(
            label="中心にするキャラクター",
            value=configuration.settings.spotlight_character_id,
            filled=True,
            fill_color="#292925",
            border=ft.InputBorder.NONE,
        )
        self.count_text = ft.Text("", size=10, color="#969188")
        self.error_text = ft.Text("", size=10, color="#D87866")
        self._on_save = on_save

        self.enabled_control.on_change = self._handle_change
        self.mode_control.on_select = self._handle_change
        for checkbox in self.character_controls:
            checkbox.on_change = self._handle_change
        self._sync_controls()

        super().__init__(
            modal=True,
            title="キャストと会話モード",
            bgcolor="#24231F",
            content=ft.Column(
                [
                    self.enabled_control,
                    ft.Text(
                        "参加するキャラクターを1〜5人選びます。表示順が発言順の基準です。",
                        size=10,
                        color="#969188",
                    ),
                    ft.Column(list(self.character_controls), spacing=2),
                    self.count_text,
                    self.mode_control,
                    self.spotlight_control,
                    self.error_text,
                ],
                tight=True,
                width=460,
                scroll=ft.ScrollMode.AUTO,
            ),
            actions=[
                ft.Button("キャンセル", on_click=on_cancel),
                ft.Button(
                    "保存",
                    bgcolor="#F2A65A",
                    color="#17120D",
                    on_click=self._submit,
                ),
            ],
        )

    def value(self) -> GroupChatSettingsInput:
        selected_options = tuple(
            option
            for option, checkbox in zip(
                self.options, self.character_controls, strict=True
            )
            if checkbox.value
        )
        enabled = bool(self.enabled_control.value)
        if not enabled:
            selected_options = selected_options[:1]
            return GroupChatSettingsInput(
                tuple(option.character_version_id for option in selected_options),
                False,
                TurnMode.STORY,
                None,
            )
        mode = TurnMode(self.mode_control.value or TurnMode.STORY.value)
        spotlight = (
            self.spotlight_control.value
            if mode is TurnMode.SPOTLIGHT
            else None
        )
        return GroupChatSettingsInput(
            tuple(option.character_version_id for option in selected_options),
            True,
            mode,
            spotlight,
        )

    def _handle_change(self) -> None:
        self._sync_controls()
        self.update()

    def _sync_controls(self) -> None:
        enabled = bool(self.enabled_control.value)
        selected = [
            option
            for option, checkbox in zip(
                self.options, self.character_controls, strict=True
            )
            if checkbox.value
        ]
        if not enabled:
            if not selected and self.character_controls:
                self.character_controls[0].value = True
                selected = [self.options[0]]
            for checkbox in self.character_controls:
                checkbox.disabled = True
            self.mode_control.value = TurnMode.STORY.value
            self.mode_control.disabled = True
        else:
            for checkbox in self.character_controls:
                checkbox.disabled = False
            self.mode_control.disabled = False
        selected = [
            option
            for option, checkbox in zip(
                self.options, self.character_controls, strict=True
            )
            if checkbox.value
        ]
        self.count_text.value = (
            f"OFFで保存すると先頭の{selected[0].display_name}を残します"
            if not enabled and selected
            else f"選択中: {len(selected)} / 5人"
        )
        self.count_text.color = (
            "#D87866" if not 1 <= len(selected) <= 5 else "#969188"
        )
        self.spotlight_control.options = [
            ft.DropdownOption(option.character_id, option.display_name)
            for option in selected
        ]
        selected_ids = {option.character_id for option in selected}
        if self.spotlight_control.value not in selected_ids:
            self.spotlight_control.value = None
        spotlight_visible = (
            enabled and self.mode_control.value == TurnMode.SPOTLIGHT.value
        )
        self.spotlight_control.visible = spotlight_visible
        self.error_text.value = ""

    async def _submit(self) -> None:
        value = self.value()
        if not 1 <= len(value.character_version_ids) <= 5:
            self.error_text.value = "キャラクターを1〜5人選んでください。"
            self.update()
            return
        if (
            value.mode is TurnMode.SPOTLIGHT
            and value.spotlight_character_id is None
        ):
            self.error_text.value = "中心にするキャラクターを選んでください。"
            self.update()
            return
        await self._on_save(value)


def group_configuration_summary(
    configuration: ConversationGroupConfiguration,
) -> str:
    if not configuration.settings.enabled:
        return "単独会話"
    mode_label = {
        TurnMode.STORY: "物語",
        TurnMode.ROUND_TABLE: "ラウンドテーブル",
        TurnMode.SPOTLIGHT: "スポットライト",
    }[configuration.settings.mode]
    summary = f"{mode_label} · {len(configuration.cast.members)}人"
    if configuration.settings.spotlight_character_id is not None:
        target = next(
            (
                member.display_name
                for member in configuration.cast.members
                if member.character_id
                == configuration.settings.spotlight_character_id
            ),
            None,
        )
        if target is not None:
            summary = f"{summary} · {target}中心"
    return summary


def _character_options(
    configuration: ConversationGroupConfiguration,
    characters: list[CharacterVersion],
) -> tuple[GroupCharacterOption, ...]:
    current_ids = {member.character_id for member in configuration.cast.members}
    current_options = [
        GroupCharacterOption(
            member.character_id,
            member.character_version_id,
            member.display_name,
            True,
        )
        for member in configuration.cast.members
    ]
    available_options = [
        GroupCharacterOption(
            character.character_id,
            character.id,
            character.display_name,
            False,
        )
        for character in characters
        if character.character_id not in current_ids
    ]
    return tuple([*current_options, *available_options])

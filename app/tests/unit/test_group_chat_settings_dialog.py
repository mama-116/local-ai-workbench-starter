from __future__ import annotations

from datetime import UTC, datetime

from local_llm_chat.domain.group_turns import (
    ConversationCast,
    ConversationGroupConfiguration,
    ConversationGroupSettings,
    FormalCastMember,
)
from local_llm_chat.domain.models import CharacterVersion
from local_llm_chat.domain.states import TurnMode
from local_llm_chat.presentation.components.group_chat_settings_dialog import (
    GroupChatSettingsDialog,
    GroupChatSettingsInput,
    group_configuration_summary,
)


def configuration(enabled: bool = True) -> ConversationGroupConfiguration:
    now = datetime.now(UTC)
    return ConversationGroupConfiguration(
        ConversationCast(
            "conversation-1",
            (
                FormalCastMember("character-1", "old-version", "先輩", 0),
                FormalCastMember("character-2", "version-2", "田中", 1),
            ),
        ),
        ConversationGroupSettings(
            "conversation-1", enabled, TurnMode.STORY, None, now
        ),
    )


def characters() -> list[CharacterVersion]:
    now = datetime.now(UTC)
    return [
        CharacterVersion("new-version", "character-1", "先輩", 2, "new", now),
        CharacterVersion("version-2", "character-2", "田中", 1, "two", now),
        CharacterVersion("version-3", "character-3", "佐藤", 1, "three", now),
    ]


async def no_save(_value: GroupChatSettingsInput) -> None:
    return None


def test_dialog_preserves_pinned_versions_and_has_one_option_per_character() -> None:
    dialog = GroupChatSettingsDialog(configuration(), characters(), no_save, lambda: None)

    assert [option.character_version_id for option in dialog.options] == [
        "old-version",
        "version-2",
        "version-3",
    ]
    assert dialog.value().character_version_ids == (
        "old-version",
        "version-2",
    )


def test_disabling_group_normalizes_to_one_character_story_mode() -> None:
    dialog = GroupChatSettingsDialog(configuration(), characters(), no_save, lambda: None)
    dialog.enabled_control.value = False
    dialog._sync_controls()

    assert [checkbox.value for checkbox in dialog.character_controls[:2]] == [
        True,
        True,
    ]
    value = dialog.value()
    assert value.character_version_ids == ("old-version",)
    assert value.enabled is False
    assert value.mode is TurnMode.STORY
    assert value.spotlight_character_id is None
    assert group_configuration_summary(configuration(False)) == "単独会話"

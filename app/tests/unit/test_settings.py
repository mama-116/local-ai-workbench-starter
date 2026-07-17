from pathlib import Path

import pytest

from local_llm_chat.infrastructure import settings
from local_llm_chat.infrastructure.settings import resolve_app_paths


def test_workspace_uses_stable_data_dir_instead_of_flet_launch_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("LOCAL_LLM_CHAT_DATA_DIR", raising=False)
    monkeypatch.setenv("FLET_APP_STORAGE_DATA", str(tmp_path / "changing-flet-dir"))

    paths = resolve_app_paths()

    assert paths.data_dir.name == ".local-data"
    assert (paths.data_dir.parent / "pyproject.toml").is_file()


def test_explicit_stable_data_dir_has_highest_priority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stable = tmp_path / "stable"
    monkeypatch.setenv("LOCAL_LLM_CHAT_DATA_DIR", str(stable))
    monkeypatch.setenv("FLET_APP_STORAGE_DATA", str(tmp_path / "changing-flet-dir"))

    assert resolve_app_paths().data_dir == stable


def test_packaged_app_uses_flet_user_data_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    packaged_data = tmp_path / "flet-user-data"
    monkeypatch.delenv("LOCAL_LLM_CHAT_DATA_DIR", raising=False)
    monkeypatch.setenv("FLET_APP_STORAGE_DATA", str(packaged_data))
    monkeypatch.setattr(settings, "_find_workspace_data_dir", lambda _: None)

    assert resolve_app_paths().data_dir == packaged_data

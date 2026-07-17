from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppPaths:
    data_dir: Path
    database_path: Path
    recovery_log_path: Path
    ollama_server_config_path: Path
    ollama_connections_path: Path


def resolve_app_paths(data_dir: Path | None = None) -> AppPaths:
    if data_dir is None:
        explicit_data = os.environ.get("LOCAL_LLM_CHAT_DATA_DIR")
        workspace_data = _find_workspace_data_dir(Path(__file__).resolve())
        flet_data = os.environ.get("FLET_APP_STORAGE_DATA")
        if explicit_data:
            data_dir = Path(explicit_data)
        elif workspace_data is not None:
            data_dir = workspace_data
        elif flet_data:
            data_dir = Path(flet_data)
        else:
            data_dir = Path(__file__).resolve().parents[3] / ".local-data"
    ollama_config = Path.home() / ".ollama" / "server.json"
    return AppPaths(
        data_dir=data_dir,
        database_path=data_dir / "local_llm_chat.sqlite3",
        recovery_log_path=data_dir / "recovery.log",
        ollama_server_config_path=ollama_config,
        ollama_connections_path=data_dir / "ollama-connections.json",
    )


def _find_workspace_data_dir(module_path: Path) -> Path | None:
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / ".local-data"
    return None


def is_ollama_cloud_disabled(config_path: Path) -> bool:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and raw.get("disable_ollama_cloud") is True

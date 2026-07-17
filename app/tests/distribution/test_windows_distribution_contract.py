from __future__ import annotations

import tomllib
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = APP_ROOT.parent


def test_windows_product_metadata_is_stable() -> None:
    config = tomllib.loads((APP_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["version"] == "0.1.0"
    assert config["tool"]["flet"]["product"] == "Local LLM Chat"
    assert config["tool"]["flet"]["windows"]["artifact"] == "LocalLLMChat"


def test_portable_build_has_privacy_guard_and_user_instructions() -> None:
    script = REPOSITORY_ROOT / "scripts" / "build-windows-app.ps1"
    instructions = APP_ROOT / "WINDOWS_PORTABLE_README.txt"

    assert script.is_file()
    script_text = script.read_text(encoding="utf-8")
    assert "flet build windows" in script_text
    assert "uv run --frozen" in script_text
    assert "pyinstaller" in script_text
    assert "LocalNotesMCP.exe" in script_text
    assert ".local-data" in script_text
    assert ".flet" in script_text
    assert "*.sqlite*" in script_text
    assert "ollama-connections.json" in script_text
    assert "Get-FileHash" in script_text
    assert "LocalLLMChatBuild" in script_text
    assert "'/utf-8'" in script_text
    assert "PRIVATE KEY" in script_text

    assert instructions.is_file()
    instructions_text = instructions.read_text(encoding="utf-8")
    assert "Windows 11" in instructions_text
    assert "LocalLLMChat.exe" in instructions_text
    assert "disable_ollama_cloud" in instructions_text
    assert "Ollamaとモデルは同梱されていません" in instructions_text

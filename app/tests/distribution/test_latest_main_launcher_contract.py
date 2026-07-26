from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INSTALLER = REPOSITORY_ROOT / "scripts" / "install-latest-launcher.ps1"
LAUNCHER = REPOSITORY_ROOT / "scripts" / "start-latest-main.ps1"
WINDOWS_ENTRY_POINT = REPOSITORY_ROOT / "Start-LocalLLMChat-Latest.cmd"


def test_installer_creates_an_isolated_main_worktree_inside_the_repository() -> None:
    installer = INSTALLER.read_text(encoding="utf-8")

    assert "'.local-runtime'" in installer
    assert "'worktree', 'add', '--detach'" in installer
    assert "'origin/main'" in installer
    assert "'status', '--porcelain', '--untracked-files=no'" in installer
    assert "'switch', '--detach', 'origin/main'" in installer
    assert "mama-116/local-ai-workbench-starter.git" in installer
    assert "reset --hard" not in installer
    assert "'pull'" not in installer


def test_launcher_builds_each_main_commit_once_and_falls_back_safely() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "'fetch', 'origin', 'main', '--prune'" in launcher
    assert "'refs/remotes/origin/main^{commit}'" in launcher
    assert "'status', '--porcelain', '--untracked-files=no'" in launcher
    assert "'switch', '--detach', $targetCommit" in launcher
    assert "'BUILD-READY.json'" in launcher
    assert "Find-LatestReadyBuild" in launcher
    assert "LocalLLMChatLatestMainLauncher" in launcher
    assert "mama-116/local-ai-workbench-starter.git" in launcher
    assert "Remove-Item -LiteralPath $readyMarker -Force" in launcher
    assert "Remove-Item -Recurse" not in launcher
    assert "GetRelativePath" not in launcher
    assert "reset --hard" not in launcher
    assert "'pull'" not in launcher


def test_windows_entry_point_uses_the_checked_in_launcher() -> None:
    entry_point = WINDOWS_ENTRY_POINT.read_text(encoding="utf-8")

    assert "powershell.exe" in entry_point
    assert "-ExecutionPolicy RemoteSigned" in entry_point
    assert r"%~dp0scripts\start-latest-main.ps1" in entry_point

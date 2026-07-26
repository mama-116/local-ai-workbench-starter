from __future__ import annotations

from pathlib import Path

from local_llm_chat.infrastructure.packaged_pywin32_paths import (
    activate_packaged_pywin32_paths,
)


def test_windows_package_adds_only_existing_pywin32_pth_directories(
    tmp_path: Path,
) -> None:
    site_packages = tmp_path / "site-packages"
    win32 = site_packages / "win32"
    win32_lib = win32 / "lib"
    win32_lib.mkdir(parents=True)
    search_path = [str(site_packages)]

    added = activate_packaged_pywin32_paths(search_path, platform="win32")

    assert added == (win32.resolve(), win32_lib.resolve())
    assert search_path == [
        str(site_packages),
        str(win32.resolve()),
        str(win32_lib.resolve()),
    ]


def test_pywin32_path_activation_is_idempotent(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    win32_lib = site_packages / "win32" / "lib"
    win32_lib.mkdir(parents=True)
    search_path = [str(site_packages)]

    activate_packaged_pywin32_paths(search_path, platform="win32")
    added = activate_packaged_pywin32_paths(search_path, platform="win32")

    assert added == ()
    assert len(search_path) == 3


def test_non_windows_runtime_does_not_change_search_path(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    (site_packages / "win32" / "lib").mkdir(parents=True)
    search_path = [str(site_packages)]

    added = activate_packaged_pywin32_paths(search_path, platform="linux")

    assert added == ()
    assert search_path == [str(site_packages)]

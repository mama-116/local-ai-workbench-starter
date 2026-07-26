from __future__ import annotations

import os
import sys
from collections.abc import MutableSequence
from pathlib import Path


def activate_packaged_pywin32_paths(
    search_path: MutableSequence[str] | None = None,
    *,
    platform: str | None = None,
) -> tuple[Path, ...]:
    """Restore the two pywin32.pth entries omitted by Serious Python."""

    active_platform = sys.platform if platform is None else platform
    if active_platform != "win32":
        return ()
    paths = sys.path if search_path is None else search_path
    known = {_normalized_path(value) for value in paths if value}
    added: list[Path] = []
    for value in tuple(paths):
        if not value:
            continue
        try:
            site_packages = Path(value).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if (
            not site_packages.is_dir()
            or site_packages.name.casefold() != "site-packages"
        ):
            continue
        for relative in (Path("win32"), Path("win32") / "lib"):
            try:
                candidate = (site_packages / relative).resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if (
                not candidate.is_dir()
                or not candidate.is_relative_to(site_packages)
            ):
                continue
            normalized = _normalized_path(str(candidate))
            if normalized in known:
                continue
            paths.append(str(candidate))
            known.add(normalized)
            added.append(candidate)
    return tuple(added)


def _normalized_path(value: str) -> str:
    try:
        resolved = str(Path(value).resolve())
    except (OSError, RuntimeError):
        resolved = os.path.abspath(value)
    return os.path.normcase(resolved)

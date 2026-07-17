from __future__ import annotations

import stat
from pathlib import Path

from local_llm_chat.domain.errors import ValidationError


class ReadOnlyToolAccessPolicy:
    """Resolve text files without allowing a tool to escape its granted root."""

    _ALLOWED_SUFFIXES = frozenset({".md", ".txt"})
    _BLOCKED_DIRECTORIES = frozenset({".git", ".local-data", ".ollama"})
    _BLOCKED_KEY_NAMES = frozenset(
        {
            "id_dsa",
            "id_ecdsa",
            "id_ed25519",
            "id_rsa",
            "private_key",
        }
    )
    _BLOCKED_KEY_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
    _MAX_FILE_BYTES = 1024 * 1024
    _DENIED_MESSAGE = "許可されていないファイルです。"
    _REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    def __init__(self, allowed_root: Path) -> None:
        root = Path(allowed_root)
        if self._is_link_or_reparse_point(root):
            self._deny()
        try:
            resolved_root = root.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValidationError(self._DENIED_MESSAGE) from error
        if not resolved_root.is_dir():
            self._deny()
        self._allowed_root = resolved_root

    def resolve_file(self, requested_path: str) -> Path:
        relative_path = Path(requested_path)
        if (
            not requested_path.strip()
            or relative_path.is_absolute()
            or bool(relative_path.drive)
            or ".." in relative_path.parts
            or self._is_sensitive(relative_path)
        ):
            self._deny()

        unresolved_candidate = self._allowed_root / relative_path
        self._require_no_link_components(unresolved_candidate)
        try:
            candidate = unresolved_candidate.resolve(strict=True)
            file_size = candidate.stat().st_size
        except (OSError, RuntimeError) as error:
            raise ValidationError(self._DENIED_MESSAGE) from error

        if (
            not candidate.is_relative_to(self._allowed_root)
            or not candidate.is_file()
            or candidate.suffix.lower() not in self._ALLOWED_SUFFIXES
            or file_size > self._MAX_FILE_BYTES
        ):
            self._deny()
        return candidate

    def _require_no_link_components(self, candidate: Path) -> None:
        current = self._allowed_root
        for part in candidate.relative_to(self._allowed_root).parts:
            current /= part
            if self._is_link_or_reparse_point(current):
                self._deny()

    @classmethod
    def _is_sensitive(cls, path: Path) -> bool:
        lowered_parts = tuple(part.casefold() for part in path.parts)
        if any(part in cls._BLOCKED_DIRECTORIES for part in lowered_parts):
            return True
        filename = path.name.casefold()
        stem = path.stem.casefold()
        return (
            filename == ".env"
            or filename.startswith(".env.")
            or stem in cls._BLOCKED_KEY_NAMES
            or path.suffix.casefold() in cls._BLOCKED_KEY_SUFFIXES
            or path.suffix.casefold() in {".db", ".sqlite", ".sqlite3"}
        )

    @classmethod
    def _is_link_or_reparse_point(cls, path: Path) -> bool:
        try:
            status = path.lstat()
        except OSError:
            return False
        attributes = int(getattr(status, "st_file_attributes", 0))
        return stat.S_ISLNK(status.st_mode) or bool(
            attributes & cls._REPARSE_POINT
        )

    @classmethod
    def _deny(cls) -> None:
        raise ValidationError(cls._DENIED_MESSAGE)

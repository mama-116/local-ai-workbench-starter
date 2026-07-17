from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.policies.read_only_tool_access import (
    ReadOnlyToolAccessPolicy,
)


@dataclass(frozen=True, slots=True)
class FolderSearchMatch:
    relative_path: str
    line_number: int
    excerpt: str


class BuiltInFolderSearchTool:
    _MAX_QUERY_CHARACTERS = 500
    _MAX_RESULTS = 20
    _MAX_EXCERPT_CHARACTERS = 240
    _MAX_FILE_BYTES = 1024 * 1024
    _PRUNED_DIRECTORIES = frozenset({".git", ".local-data", ".ollama"})

    def __init__(self, allowed_root: Path) -> None:
        self._access_policy = ReadOnlyToolAccessPolicy(allowed_root)
        self._allowed_root = Path(allowed_root).resolve(strict=True)

    async def search(
        self,
        query: str,
        max_results: int = _MAX_RESULTS,
    ) -> tuple[FolderSearchMatch, ...]:
        clean_query = query.strip()
        if not clean_query or len(clean_query) > self._MAX_QUERY_CHARACTERS:
            raise ValidationError("検索語は1文字から500文字で指定してください。")
        if not 1 <= max_results <= self._MAX_RESULTS:
            raise ValidationError("検索件数は1件から20件で指定してください。")
        return await asyncio.to_thread(
            self._search_sync,
            clean_query,
            max_results,
        )

    def _search_sync(
        self,
        query: str,
        max_results: int,
    ) -> tuple[FolderSearchMatch, ...]:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches: list[FolderSearchMatch] = []
        for candidate in self._candidate_files():
            try:
                safe_path = self._access_policy.resolve_file(
                    candidate.relative_to(self._allowed_root).as_posix()
                )
                payload = safe_path.read_bytes()
                if len(payload) > self._MAX_FILE_BYTES:
                    continue
                content = payload.decode("utf-8")
            except (OSError, UnicodeDecodeError, ValidationError):
                continue

            relative_path = safe_path.relative_to(self._allowed_root).as_posix()
            for line_number, line in enumerate(content.splitlines(), start=1):
                found = pattern.search(line)
                if found is None:
                    continue
                matches.append(
                    FolderSearchMatch(
                        relative_path=relative_path,
                        line_number=line_number,
                        excerpt=self._bounded_excerpt(
                            line,
                            found.start(),
                            found.end(),
                        ),
                    )
                )
                if len(matches) >= max_results:
                    return tuple(matches)
        return tuple(matches)

    def _candidate_files(self) -> tuple[Path, ...]:
        candidates: list[Path] = []
        for directory, directory_names, filenames in os.walk(
            self._allowed_root,
            topdown=True,
            followlinks=False,
        ):
            current = Path(directory)
            directory_names[:] = sorted(
                (
                    name
                    for name in directory_names
                    if name.casefold() not in self._PRUNED_DIRECTORIES
                    and not self._is_link_directory(current / name)
                ),
                key=str.casefold,
            )
            candidates.extend(current / name for name in filenames)
        return tuple(
            sorted(
                candidates,
                key=lambda path: path.relative_to(self._allowed_root)
                .as_posix()
                .casefold(),
            )
        )

    @staticmethod
    def _is_link_directory(path: Path) -> bool:
        try:
            return path.is_symlink() or path.is_junction()
        except OSError:
            return True

    @classmethod
    def _bounded_excerpt(
        cls,
        line: str,
        match_start: int,
        match_end: int,
    ) -> str:
        if len(line) <= cls._MAX_EXCERPT_CHARACTERS:
            return line
        match_length = match_end - match_start
        surrounding = max(cls._MAX_EXCERPT_CHARACTERS - match_length, 0)
        start = max(match_start - surrounding // 2, 0)
        end = min(start + cls._MAX_EXCERPT_CHARACTERS, len(line))
        start = max(end - cls._MAX_EXCERPT_CHARACTERS, 0)
        return line[start:end]

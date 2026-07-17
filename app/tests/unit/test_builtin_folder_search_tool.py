from __future__ import annotations

from pathlib import Path

import pytest

from local_llm_chat.application.services.folder_search_tool import (
    BuiltInFolderSearchTool,
    FolderSearchMatch,
)
from local_llm_chat.domain.errors import ValidationError


@pytest.fixture
def allowed_root(tmp_path: Path) -> Path:
    root = tmp_path / "allowed"
    root.mkdir()
    return root


@pytest.mark.asyncio
async def test_searches_supported_text_recursively_in_stable_order(
    allowed_root: Path,
) -> None:
    nested = allowed_root / "nested"
    nested.mkdir()
    (nested / "second.md").write_text(
        "前置き\nNeedleはここです。\n", encoding="utf-8"
    )
    (allowed_root / "first.txt").write_text(
        "needleが先です。\n次の行\n", encoding="utf-8"
    )
    (allowed_root / "ignored.py").write_text(
        "needle = '検索対象外'\n", encoding="utf-8"
    )
    tool = BuiltInFolderSearchTool(allowed_root)

    matches = await tool.search("NEEDLE")

    assert matches == (
        FolderSearchMatch(
            relative_path="first.txt",
            line_number=1,
            excerpt="needleが先です。",
        ),
        FolderSearchMatch(
            relative_path="nested/second.md",
            line_number=2,
            excerpt="Needleはここです。",
        ),
    )


@pytest.mark.asyncio
async def test_treats_search_query_as_literal_text(allowed_root: Path) -> None:
    (allowed_root / "literal.md").write_text(
        "TODO[1]を確認する\nTODO1は別物\n", encoding="utf-8"
    )
    tool = BuiltInFolderSearchTool(allowed_root)

    matches = await tool.search("TODO[1]")

    assert [(match.line_number, match.excerpt) for match in matches] == [
        (1, "TODO[1]を確認する")
    ]


@pytest.mark.asyncio
async def test_returns_at_most_requested_number_of_matches(
    allowed_root: Path,
) -> None:
    (allowed_root / "many.txt").write_text(
        "\n".join(f"needle {number}" for number in range(30)),
        encoding="utf-8",
    )
    tool = BuiltInFolderSearchTool(allowed_root)

    matches = await tool.search("needle", max_results=20)

    assert len(matches) == 20
    assert matches[0].line_number == 1
    assert matches[-1].line_number == 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "max_results", "message"),
    [
        ("   ", 20, "検索語"),
        ("needle", 0, "検索件数"),
        ("needle", 21, "検索件数"),
    ],
)
async def test_rejects_invalid_search_request(
    allowed_root: Path,
    query: str,
    max_results: int,
    message: str,
) -> None:
    tool = BuiltInFolderSearchTool(allowed_root)

    with pytest.raises(ValidationError, match=message):
        await tool.search(query, max_results=max_results)


@pytest.mark.asyncio
async def test_skips_links_and_sensitive_candidates(
    allowed_root: Path,
) -> None:
    outside_file = allowed_root.parent / "outside.md"
    outside_file.write_text("needle outside\n", encoding="utf-8")
    (allowed_root / "linked.md").symlink_to(outside_file)
    (allowed_root / ".env").write_text("needle secret\n", encoding="utf-8")
    git_dir = allowed_root / ".git"
    git_dir.mkdir()
    (git_dir / "config.md").write_text("needle git\n", encoding="utf-8")
    (allowed_root / "private_key.md").write_text(
        "needle key\n", encoding="utf-8"
    )
    (allowed_root / "safe.md").write_text("needle safe\n", encoding="utf-8")
    tool = BuiltInFolderSearchTool(allowed_root)

    matches = await tool.search("needle")

    assert [match.relative_path for match in matches] == ["safe.md"]


@pytest.mark.asyncio
async def test_skips_invalid_utf8_and_oversized_files(
    allowed_root: Path,
) -> None:
    (allowed_root / "broken.txt").write_bytes(b"needle\xff")
    (allowed_root / "huge.md").write_bytes(
        b"needle\n" + b"x" * (1024 * 1024)
    )
    (allowed_root / "safe.txt").write_text("needle safe\n", encoding="utf-8")
    tool = BuiltInFolderSearchTool(allowed_root)

    matches = await tool.search("needle")

    assert matches == (
        FolderSearchMatch(
            relative_path="safe.txt",
            line_number=1,
            excerpt="needle safe",
        ),
    )


@pytest.mark.asyncio
async def test_bounds_excerpt_without_losing_match(allowed_root: Path) -> None:
    long_line = "前" * 180 + "needle" + "後" * 180
    (allowed_root / "long.md").write_text(long_line, encoding="utf-8")
    tool = BuiltInFolderSearchTool(allowed_root)

    matches = await tool.search("needle")

    assert len(matches) == 1
    assert "needle" in matches[0].excerpt
    assert len(matches[0].excerpt) <= 240

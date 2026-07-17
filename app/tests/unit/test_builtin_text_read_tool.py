from __future__ import annotations

from pathlib import Path

import pytest

from local_llm_chat.application.services.text_read_tool import (
    BuiltInTextReadTool,
    TextReadResult,
)
from local_llm_chat.domain.errors import ValidationError


@pytest.fixture
def allowed_root(tmp_path: Path) -> Path:
    root = tmp_path / "allowed"
    root.mkdir()
    return root


@pytest.mark.asyncio
async def test_reads_supported_text_with_line_metadata(allowed_root: Path) -> None:
    (allowed_root / "notes.md").write_text(
        "1行目\n2行目\n3行目\n",
        encoding="utf-8",
    )
    tool = BuiltInTextReadTool(allowed_root)

    result = await tool.read("notes.md")

    assert result == TextReadResult(
        relative_path="notes.md",
        content="1行目\n2行目\n3行目",
        start_line=1,
        end_line=3,
        total_lines=3,
        truncated=False,
    )


@pytest.mark.asyncio
async def test_reads_requested_line_window(allowed_root: Path) -> None:
    nested = allowed_root / "nested"
    nested.mkdir()
    (nested / "manual.txt").write_text(
        "one\ntwo\nthree\nfour\n",
        encoding="utf-8",
    )
    tool = BuiltInTextReadTool(allowed_root)

    result = await tool.read("nested/manual.txt", start_line=2, max_lines=2)

    assert result == TextReadResult(
        relative_path="nested/manual.txt",
        content="two\nthree",
        start_line=2,
        end_line=3,
        total_lines=4,
        truncated=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start_line", "max_lines", "message"),
    [
        (0, 20, "開始行"),
        (-1, 20, "開始行"),
        (1, 0, "読取り行数"),
        (1, 201, "読取り行数"),
    ],
)
async def test_rejects_invalid_line_window(
    allowed_root: Path,
    start_line: int,
    max_lines: int,
    message: str,
) -> None:
    (allowed_root / "notes.md").write_text("content\n", encoding="utf-8")
    tool = BuiltInTextReadTool(allowed_root)

    with pytest.raises(ValidationError, match=message):
        await tool.read("notes.md", start_line=start_line, max_lines=max_lines)


@pytest.mark.asyncio
async def test_rejects_start_line_after_end_of_file(allowed_root: Path) -> None:
    (allowed_root / "notes.md").write_text("one\ntwo\n", encoding="utf-8")
    tool = BuiltInTextReadTool(allowed_root)

    with pytest.raises(ValidationError, match="開始行"):
        await tool.read("notes.md", start_line=3)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.md",
        ".env",
        ".git/config.md",
        "private_key.md",
    ],
)
async def test_rejects_disallowed_file(
    allowed_root: Path,
    relative_path: str,
) -> None:
    outside = allowed_root.parent / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    candidate = allowed_root / relative_path
    if ".." not in candidate.parts:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("secret\n", encoding="utf-8")
    tool = BuiltInTextReadTool(allowed_root)

    with pytest.raises(ValidationError, match="許可されていないファイル"):
        await tool.read(relative_path)


@pytest.mark.asyncio
async def test_rejects_symlink_that_escapes_granted_folder(
    allowed_root: Path,
) -> None:
    outside = allowed_root.parent / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    (allowed_root / "linked.md").symlink_to(outside)
    tool = BuiltInTextReadTool(allowed_root)

    with pytest.raises(ValidationError, match="許可されていないファイル"):
        await tool.read("linked.md")


@pytest.mark.asyncio
async def test_reports_invalid_utf8_without_returning_partial_text(
    allowed_root: Path,
) -> None:
    (allowed_root / "broken.txt").write_bytes(b"valid first\ninvalid \xff")
    tool = BuiltInTextReadTool(allowed_root)

    with pytest.raises(ValidationError, match="UTF-8"):
        await tool.read("broken.txt")


@pytest.mark.asyncio
async def test_bounds_result_to_sixty_four_kib_without_breaking_utf8(
    allowed_root: Path,
) -> None:
    (allowed_root / "long.md").write_text("あ" * 30_000, encoding="utf-8")
    tool = BuiltInTextReadTool(allowed_root)

    result = await tool.read("long.md")

    assert len(result.content.encode("utf-8")) <= 64 * 1024
    assert "�" not in result.content
    assert result.start_line == 1
    assert result.end_line == 1
    assert result.total_lines == 1
    assert result.truncated is True


@pytest.mark.asyncio
async def test_default_window_stops_after_two_hundred_lines(
    allowed_root: Path,
) -> None:
    (allowed_root / "many.txt").write_text(
        "\n".join(f"line {number}" for number in range(1, 251)),
        encoding="utf-8",
    )
    tool = BuiltInTextReadTool(allowed_root)

    result = await tool.read("many.txt")

    assert result.start_line == 1
    assert result.end_line == 200
    assert result.total_lines == 250
    assert result.truncated is True

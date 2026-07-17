from __future__ import annotations

from pathlib import Path

import pytest

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.policies.read_only_tool_access import (
    ReadOnlyToolAccessPolicy,
)


DENIED_MESSAGE = "許可されていないファイル"


@pytest.fixture
def allowed_root(tmp_path: Path) -> Path:
    root = tmp_path / "allowed"
    root.mkdir()
    return root


def test_allows_supported_file_directly_inside_granted_folder(
    allowed_root: Path,
) -> None:
    allowed_file = allowed_root / "notes.md"
    allowed_file.write_text("公開してよいメモ", encoding="utf-8")
    policy = ReadOnlyToolAccessPolicy(allowed_root)

    resolved = policy.resolve_file("notes.md")

    assert resolved == allowed_file.resolve()


@pytest.mark.parametrize(
    "requested_path",
    [
        "../outside.txt",
        "subfolder/../../outside.txt",
    ],
)
def test_rejects_parent_traversal_outside_granted_folder(
    allowed_root: Path,
    requested_path: str,
) -> None:
    (allowed_root.parent / "outside.txt").write_text("秘密", encoding="utf-8")
    (allowed_root / "subfolder").mkdir()
    policy = ReadOnlyToolAccessPolicy(allowed_root)

    with pytest.raises(ValidationError, match=DENIED_MESSAGE):
        policy.resolve_file(requested_path)


def test_rejects_absolute_path_even_when_file_exists(
    allowed_root: Path,
) -> None:
    outside_file = allowed_root.parent / "outside.txt"
    outside_file.write_text("秘密", encoding="utf-8")
    policy = ReadOnlyToolAccessPolicy(allowed_root)

    with pytest.raises(ValidationError, match=DENIED_MESSAGE):
        policy.resolve_file(str(outside_file.resolve()))


def test_rejects_sibling_folder_with_matching_name_prefix(
    allowed_root: Path,
) -> None:
    sibling = allowed_root.parent / f"{allowed_root.name}-copy"
    sibling.mkdir()
    outside_file = sibling / "outside.md"
    outside_file.write_text("秘密", encoding="utf-8")
    policy = ReadOnlyToolAccessPolicy(allowed_root)

    with pytest.raises(ValidationError, match=DENIED_MESSAGE):
        policy.resolve_file(f"../{sibling.name}/{outside_file.name}")


def test_rejects_symlink_that_escapes_granted_folder(
    allowed_root: Path,
) -> None:
    outside_file = allowed_root.parent / "outside.md"
    outside_file.write_text("秘密", encoding="utf-8")
    link = allowed_root / "linked.md"
    link.symlink_to(outside_file)
    policy = ReadOnlyToolAccessPolicy(allowed_root)

    with pytest.raises(ValidationError, match=DENIED_MESSAGE):
        policy.resolve_file(link.name)


@pytest.mark.parametrize(
    "relative_path",
    [
        ".env",
        ".git/config.md",
        ".ollama/server.md",
        ".local-data/history.md",
        "private_key.md",
        "history.sqlite3",
    ],
)
def test_rejects_sensitive_file_candidates_inside_granted_folder(
    allowed_root: Path,
    relative_path: str,
) -> None:
    candidate = allowed_root / relative_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("秘密", encoding="utf-8")
    policy = ReadOnlyToolAccessPolicy(allowed_root)

    with pytest.raises(ValidationError, match=DENIED_MESSAGE):
        policy.resolve_file(relative_path)

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE_RECORD = (
    REPOSITORY_ROOT / "docs" / "acceptance" / "S01-S09-2026-07-17.md"
)


def test_s01_through_s09_have_one_traceable_acceptance_record() -> None:
    assert ACCEPTANCE_RECORD.is_file(), (
        "S-01〜S-09を試験手順と実績へ対応付けた受入記録がありません。"
    )
    content = ACCEPTANCE_RECORD.read_text(encoding="utf-8")

    for number in range(1, 10):
        criterion = f"S-{number:02d}"
        assert content.count(f"| {criterion} |") == 1
    assert "| 未確認 |" not in content


def test_phase_zero_entrypoint_is_one_request_and_keeps_beginner_limits() -> None:
    readme = (REPOSITORY_ROOT / "README_FIRST.md").read_text(encoding="utf-8")
    agents = (REPOSITORY_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    guide = (
        REPOSITORY_ROOT
        / "plugins"
        / "local-ai-builder-kit"
        / "skills"
        / "guide-development"
        / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "準備後、次の一文だけを送ってください。" in readme
    quoted_requests = [
        line.removeprefix("> ")
        for line in readme.splitlines()
        if line.startswith("> ")
    ]
    assert quoted_requests == [
        "作りたいものを相談したいです。専門用語を減らし、質問は一度に3つまでにしてください。"
    ]
    assert "質問は一度に最大3問、最大2ラウンド" in agents
    assert "次の一手は原則1つ、30分以内" in agents
    assert "質問は一度に最大3問、最大2ラウンド" in guide
    assert "次の一手を1つ、30分以内" in guide
    assert "複数ステップの宿題を列挙しない" in guide


def test_ui_options_and_handoff_fields_remain_in_the_distribution() -> None:
    ui_options = (
        REPOSITORY_ROOT / "docs" / "TOOLS_MCP_UI_OPTIONS.md"
    ).read_text(encoding="utf-8")
    handoff = (
        REPOSITORY_ROOT / "docs" / "templates" / "HANDOFF_TEMPLATE.md"
    ).read_text(encoding="utf-8")
    thread_skill = (
        REPOSITORY_ROOT
        / "plugins"
        / "local-ai-builder-kit"
        / "skills"
        / "manage-work-thread"
        / "SKILL.md"
    ).read_text(encoding="utf-8")

    for option in ("| A:", "| B:", "| C:"):
        assert option in ui_options
    for stem in (
        "tools-mcp-a-control-desk",
        "tools-mcp-b-execution-tray",
        "tools-mcp-c-tool-center",
    ):
        assert (REPOSITORY_ROOT / "docs" / "mockups" / f"{stem}.png").is_file()
    for field in ("目的", "決定", "未決", "次の一手"):
        assert field in handoff
        assert field in thread_skill
    assert "貼り付け可能な引き継ぎ文を作る" in thread_skill
    assert "次の一手1つ" in thread_skill

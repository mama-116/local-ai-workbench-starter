from pathlib import Path

import pytest

from local_llm_chat.bootstrap import bootstrap


@pytest.mark.asyncio
async def test_bootstrap_creates_local_database_and_default_character(
    tmp_path: Path,
) -> None:
    container = await bootstrap(tmp_path)
    try:
        assert container.paths.database_path.exists()
        characters = await container.profiles.list_characters()
        assert len(characters) == 1
        assert "言語指定がない場合" in characters[0].system_prompt
        assert "依頼文の言い換えだけで終わらせない" in characters[0].system_prompt
    finally:
        await container.close()

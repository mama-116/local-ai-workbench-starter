from __future__ import annotations

import re
import unicodedata

_IGNORED_CONTENT = re.compile(
    r"```.*?```|`[^`]*`|https?://\S+|www\.\S+",
    re.DOTALL | re.IGNORECASE,
)
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_KANA = re.compile(r"[\u3040-\u30ff]")
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def needs_japanese_translation(content: str) -> bool:
    """Detect prose that contains a meaningful non-Japanese language segment."""
    prose = _IGNORED_CONTENT.sub(" ", content)
    if not prose.strip():
        return False
    if _LATIN_WORD.search(prose):
        return True
    if _HAN.search(prose) and not _KANA.search(prose):
        return True
    for character in prose:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if not any(
            script in name
            for script in ("HIRAGANA", "KATAKANA", "CJK UNIFIED IDEOGRAPH")
        ):
            return True
    return False

from local_llm_chat.application.translation_detection import (
    needs_japanese_translation,
)


def test_japanese_text_does_not_need_translation() -> None:
    assert not needs_japanese_translation("今日は良い天気ですね。")


def test_foreign_and_mixed_text_need_translation() -> None:
    assert needs_japanese_translation("This answer is written in English.")
    assert needs_japanese_translation("この機能は asynchronous に動きます。")
    assert needs_japanese_translation("这是中文回答。")


def test_code_and_urls_alone_do_not_trigger_translation() -> None:
    assert not needs_japanese_translation(
        "詳細は https://example.com を参照し、`print('hello')` を実行します。"
    )

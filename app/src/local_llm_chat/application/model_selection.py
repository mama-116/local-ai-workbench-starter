from __future__ import annotations

from local_llm_chat.domain.models import ModelInfo

_PREFERRED_CHAT_MODELS = (
    "qwen3.5:9b",
    "qwen2.5:7b",
    "gemma2:9b",
    "llama3.1:latest",
)
_SPECIALIZED_MARKERS = (
    "coder",
    "embed",
    "llava",
    "vision",
    "deepseek-r1",
    "qwq",
)
_COMFORTABLE_SIZE_BYTES = 8 * 1024**3


def recommend_chat_model(models: list[ModelInfo]) -> ModelInfo | None:
    if not models:
        return None
    by_name = {model.name.lower(): model for model in models}
    for preferred_name in _PREFERRED_CHAT_MODELS:
        if preferred_name in by_name:
            return by_name[preferred_name]

    general_models = [
        model
        for model in models
        if (not model.size_is_known or model.size_bytes <= _COMFORTABLE_SIZE_BYTES)
        and not any(marker in model.name.lower() for marker in _SPECIALIZED_MARKERS)
    ]
    candidates = general_models or [
        model
        for model in models
        if not model.size_is_known or model.size_bytes <= _COMFORTABLE_SIZE_BYTES
    ]
    return min(
        candidates or models,
        key=lambda model: model.size_bytes if model.size_is_known else float("inf"),
    )


def model_option_label(model: ModelInfo, recommended_name: str | None) -> str:
    suffix = " · おすすめ" if model.name == recommended_name else ""
    if not model.size_is_known:
        return f"{model.name} · サイズ不明{suffix}"
    size_gib = model.size_bytes / 1024**3
    return f"{model.name} · {size_gib:.1f} GiB{suffix}"

from local_llm_chat.application.model_selection import (
    model_option_label,
    recommend_chat_model,
)
from local_llm_chat.domain.models import ModelInfo


def _model(name: str, size_gib: float) -> ModelInfo:
    return ModelInfo(
        name=name,
        size_bytes=int(size_gib * 1024**3),
        format="gguf",
        family="test",
        parameter_size="test",
        quantization="Q4_K_M",
    )


def test_recommends_balanced_general_chat_model_over_largest_model() -> None:
    models = [
        _model("dear-spark:latest", 16.2),
        _model("qwen3.5:9b", 6.1),
        _model("qwen2.5:7b", 4.4),
        _model("qwen2.5-coder:7b", 4.4),
    ]

    recommended = recommend_chat_model(models)

    assert recommended is not None
    assert recommended.name == "qwen3.5:9b"


def test_fallback_avoids_specialized_and_oversized_models() -> None:
    models = [
        _model("huge-general:latest", 16.0),
        _model("small-coder:latest", 4.0),
        _model("tiny-embedding:latest", 1.0),
        _model("small-general:latest", 5.0),
    ]

    recommended = recommend_chat_model(models)

    assert recommended is not None
    assert recommended.name == "small-general:latest"


def test_option_label_shows_size_and_recommendation() -> None:
    model = _model("qwen3.5:9b", 6.1)

    assert model_option_label(model, model.name) == (
        "qwen3.5:9b · 6.1 GiB · おすすめ"
    )

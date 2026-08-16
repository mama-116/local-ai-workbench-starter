import pytest

from local_llm_chat.domain.errors import FreeOperationBlocked
from local_llm_chat.domain.models import ModelInfo, ProviderMetadata
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.states import CostClass, Locality


def local_model(name: str = "gemma4:12b", size_bytes: int = 7_600_000_000) -> ModelInfo:
    return ModelInfo(name, size_bytes, "gguf", "gemma4", "11.9B", "Q4_K_M")


def test_accepts_local_no_charge_ollama_model() -> None:
    policy = FreeOperationPolicy()
    policy.require_cloud_disabled(True)
    policy.require_provider(
        ProviderMetadata(
            "ollama-local",
            Locality.LOCAL,
            CostClass.NO_CHARGE,
            "http://127.0.0.1:11434",
        )
    )
    policy.require_model(local_model())


def test_accepts_private_lan_ollama_endpoint() -> None:
    FreeOperationPolicy().require_provider(
        ProviderMetadata(
            "ollama-lan",
            Locality.LOCAL,
            CostClass.NO_CHARGE,
            "http://192.168.1.17:11434",
        )
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://192.168.1.17:11434",
        "http://example.com:11434",
        "http://8.8.8.8:11434",
        "http://169.254.10.20:11434",
        "http://user:pass@192.168.1.17:11434",
        "http://192.168.1.17:11434/api/chat",
    ],
)
def test_rejects_non_lan_endpoint(endpoint: str) -> None:
    with pytest.raises(FreeOperationBlocked):
        FreeOperationPolicy().require_provider(
            ProviderMetadata(
                "unsafe",
                Locality.LOCAL,
                CostClass.NO_CHARGE,
                endpoint,
            )
        )


@pytest.mark.parametrize("name", ["qwen:cloud", "gpt-oss:120b-cloud", "CLOUD-model"])
def test_rejects_cloud_model_names(name: str) -> None:
    with pytest.raises(FreeOperationBlocked, match="Cloudモデル"):
        FreeOperationPolicy().require_model(local_model(name=name))


def test_rejects_model_without_local_data() -> None:
    with pytest.raises(FreeOperationBlocked, match="ローカル実体"):
        FreeOperationPolicy().require_model(local_model(size_bytes=0))


def test_accepts_private_vllm_model_when_size_is_unknown() -> None:
    model = ModelInfo(
        "deepseek-v4-flash-2bit",
        0,
        "vLLM",
        "vllm",
        "",
        "",
        size_is_known=False,
    )

    FreeOperationPolicy().require_model(model)


def test_rejects_when_ollama_cloud_is_not_disabled() -> None:
    with pytest.raises(FreeOperationBlocked, match="無効化"):
        FreeOperationPolicy().require_cloud_disabled(False)


def test_rejects_remote_or_free_tier_provider() -> None:
    policy = FreeOperationPolicy()
    with pytest.raises(FreeOperationBlocked):
        policy.require_provider(
            ProviderMetadata(
                "remote",
                Locality.REMOTE,
                CostClass.FREE_TIER,
                "https://example.com",
            )
        )

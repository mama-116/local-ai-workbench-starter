from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

from local_llm_chat.domain.errors import FreeOperationBlocked
from local_llm_chat.domain.models import ModelInfo, ProviderMetadata
from local_llm_chat.domain.states import CostClass, Locality


class FreeOperationPolicy:
    _CLOUD_NAME = re.compile(r"(?:^|[:_-])cloud(?:$|[:_-])", re.IGNORECASE)
    _PRIVATE_NETWORKS = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
    )

    def require_cloud_disabled(self, disabled: bool) -> None:
        if not disabled:
            raise FreeOperationBlocked(
                "cloud_not_disabled",
                "OllamaのCloud機能が無効化されていません。",
            )

    def require_provider(self, metadata: ProviderMetadata) -> None:
        if metadata.locality is not Locality.LOCAL:
            raise FreeOperationBlocked(
                "provider_not_local",
                "ローカル以外のProviderは利用できません。",
            )
        if metadata.cost_class is not CostClass.NO_CHARGE:
            raise FreeOperationBlocked(
                "provider_cost_unknown",
                "無料を保証できないProviderは利用できません。",
            )
        self.require_endpoint(metadata.endpoint)

    def require_endpoint(self, endpoint: str) -> None:
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise FreeOperationBlocked(
                "endpoint_not_allowed",
                "Ollamaの接続先には端末内またはLAN内のHTTPアドレスだけを指定できます。",
            )
        try:
            address = ipaddress.ip_address(parsed.hostname)
            port = parsed.port
        except ValueError as error:
            raise FreeOperationBlocked(
                "endpoint_not_allowed",
                "接続先は127.0.0.1またはプライベートIPで指定してください。",
            ) from error
        allowed_address = address.is_loopback or any(
            address in network for network in self._PRIVATE_NETWORKS
        )
        if port is None or not allowed_address:
            raise FreeOperationBlocked(
                "endpoint_not_allowed",
                "公開ネットワーク上のOllamaには接続できません。",
            )

    def require_loopback_endpoint(self, endpoint: str) -> None:
        self.require_endpoint(endpoint)
        hostname = urlsplit(endpoint).hostname
        if hostname is None or not ipaddress.ip_address(hostname).is_loopback:
            raise FreeOperationBlocked(
                "endpoint_not_loopback",
                "記憶抽出はこのPC上のOllamaだけを利用できます。",
            )

    def require_model(self, model: ModelInfo) -> None:
        if self._CLOUD_NAME.search(model.name):
            raise FreeOperationBlocked(
                "cloud_model_name",
                "Cloudモデルは利用できません。",
            )
        if model.size_bytes <= 0:
            raise FreeOperationBlocked(
                "model_has_no_local_data",
                "ローカル実体を確認できないモデルは利用できません。",
            )
        if not model.format.strip():
            raise FreeOperationBlocked(
                "model_format_unknown",
                "モデルのローカル形式を確認できません。",
            )

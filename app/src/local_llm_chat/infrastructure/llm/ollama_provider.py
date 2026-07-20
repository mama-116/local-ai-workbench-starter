from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from local_llm_chat.domain.errors import (
    ModelUnavailable,
    OllamaUnavailable,
    ToolUseUnavailable,
)
from local_llm_chat.domain.models import (
    ChatChunk,
    ChatRequest,
    ModelInfo,
    ProviderMetadata,
    ToolCallRequest,
)
from local_llm_chat.domain.states import CostClass, Locality


class OllamaProvider:
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:11434",
        name: str = "ollama-local",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._name = name
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._endpoint,
            timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=3.0),
            trust_env=False,
        )

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            name=self._name,
            locality=Locality.LOCAL,
            cost_class=CostClass.NO_CHARGE,
            endpoint=self._endpoint,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> bool:
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    async def list_models(self) -> list[ModelInfo]:
        payload = await self._request_json("GET", "/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise OllamaUnavailable("Ollamaのモデル一覧が不正です。")
        result: list[ModelInfo] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            result.append(self._model_from_tag(item))
        return sorted(result, key=lambda model: model.name.lower())

    async def inspect_model(self, model_name: str) -> ModelInfo:
        tags = await self.list_models()
        tag = next((model for model in tags if model.name == model_name), None)
        if tag is None:
            raise ModelUnavailable("選択したモデルが見つかりません。")
        payload = await self._request_json(
            "POST", "/api/show", json_body={"model": model_name, "verbose": False}
        )
        details = payload.get("details", {})
        if not isinstance(details, dict):
            details = {}
        return ModelInfo(
            name=tag.name,
            size_bytes=tag.size_bytes,
            format=str(details.get("format", tag.format) or ""),
            family=str(details.get("family", tag.family) or ""),
            parameter_size=str(details.get("parameter_size", tag.parameter_size) or ""),
            quantization=str(details.get("quantization_level", tag.quantization) or ""),
            license_text=str(payload.get("license", "") or ""),
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": request.system_prompt}
        ]
        for message in request.messages:
            item: dict[str, object] = {
                "role": message.role.value,
                "content": message.content,
            }
            if message.tool_name is not None:
                item["tool_name"] = message.tool_name
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(item)
        body: dict[str, object] = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": request.options,
        }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
        if request.response_format is not None:
            body["format"] = request.response_format
        try:
            async with self._client.stream("POST", "/api/chat", json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = self._parse_stream_line(line)
                    yield chunk
        except httpx.ConnectError as error:
            raise OllamaUnavailable("Ollamaに接続できません。") from error
        except httpx.TimeoutException as error:
            raise OllamaUnavailable("Ollamaの応答がタイムアウトしました。") from error
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                raise ModelUnavailable("選択したモデルが見つかりません。") from error
            if request.tools and error.response.status_code == 400:
                raise ToolUseUnavailable(
                    "選択したモデルはツール呼出しに対応していません。"
                ) from error
            raise OllamaUnavailable(
                f"Ollamaがエラーを返しました ({error.response.status_code})。"
            ) from error

    async def _request_json(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, json=json_body)
            response.raise_for_status()
            payload = response.json()
        except httpx.ConnectError as error:
            raise OllamaUnavailable("Ollamaに接続できません。") from error
        except httpx.TimeoutException as error:
            raise OllamaUnavailable("Ollamaの応答がタイムアウトしました。") from error
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
            raise OllamaUnavailable("Ollamaから不正な応答を受け取りました。") from error
        if not isinstance(payload, dict):
            raise OllamaUnavailable("Ollamaから不正な応答を受け取りました。")
        return payload

    @staticmethod
    def _model_from_tag(item: dict[str, Any]) -> ModelInfo:
        details = item.get("details", {})
        if not isinstance(details, dict):
            details = {}
        return ModelInfo(
            name=str(item.get("name") or item.get("model") or ""),
            size_bytes=int(item.get("size") or 0),
            format=str(details.get("format") or ""),
            family=str(details.get("family") or ""),
            parameter_size=str(details.get("parameter_size") or ""),
            quantization=str(details.get("quantization_level") or ""),
        )

    @staticmethod
    def _parse_stream_line(line: str) -> ChatChunk:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise OllamaUnavailable("Ollamaの逐次応答が壊れています。") from error
        if not isinstance(payload, dict):
            raise OllamaUnavailable("Ollamaの逐次応答が壊れています。")
        if payload.get("error"):
            raise OllamaUnavailable(str(payload["error"]))
        message = payload.get("message", {})
        content = str(message.get("content", "")) if isinstance(message, dict) else ""
        tool_calls = OllamaProvider._parse_tool_calls(message)
        return ChatChunk(
            content=content,
            done=bool(payload.get("done", False)),
            prompt_tokens=_optional_int(payload.get("prompt_eval_count")),
            output_tokens=_optional_int(payload.get("eval_count")),
            total_duration_ns=_optional_int(payload.get("total_duration")),
            generation_duration_ns=_optional_int(payload.get("eval_duration")),
            tool_calls=tool_calls,
        )

    @staticmethod
    def _parse_tool_calls(message: object) -> tuple[ToolCallRequest, ...]:
        if not isinstance(message, dict):
            return ()
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            return ()
        calls: list[ToolCallRequest] = []
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {"__invalid_arguments__": True}
            call_id = raw_call.get("id")
            calls.append(
                ToolCallRequest(
                    str(call_id) if isinstance(call_id, str) else f"ollama-{index}",
                    name,
                    arguments,
                )
            )
        return tuple(calls)


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None

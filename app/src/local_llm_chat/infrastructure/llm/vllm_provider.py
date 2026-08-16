from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from time import perf_counter_ns
from typing import Any

import httpx

from local_llm_chat.domain.errors import (
    ModelUnavailable,
    ProviderUnavailable,
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


VLLM_API_KEY_ENV = "LOCAL_LLM_CHAT_VLLM_API_KEY"


class VllmProvider:
    def __init__(
        self,
        endpoint: str,
        name: str,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._name = name
        self._owns_client = client is None
        resolved_key = api_key if api_key is not None else os.environ.get(VLLM_API_KEY_ENV)
        headers = {"Authorization": f"Bearer {resolved_key}"} if resolved_key else {}
        if client is None:
            self._client = httpx.AsyncClient(
                base_url=f"{self._endpoint}/v1",
                headers=headers,
                timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=3.0),
                trust_env=False,
            )
        else:
            self._client = client
            if resolved_key:
                self._client.headers["Authorization"] = f"Bearer {resolved_key}"

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
            response = await self._client.get("/models")
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    async def list_models(self) -> list[ModelInfo]:
        payload = await self._request_json("GET", "/models")
        raw_models = payload.get("data", [])
        if not isinstance(raw_models, list):
            raise ProviderUnavailable("vLLMのモデル一覧が不正です。")
        models = [
            self._model_from_item(item)
            for item in raw_models
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        ]
        return sorted(models, key=lambda model: model.name.lower())

    async def inspect_model(self, model_name: str) -> ModelInfo:
        models = await self.list_models()
        model = next((item for item in models if item.name == model_name), None)
        if model is None:
            raise ModelUnavailable("選択したvLLMモデルが見つかりません。")
        return model

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        if request.tools:
            raise ToolUseUnavailable("vLLMのツール呼出しはまだ有効化されていません。")

        body: dict[str, object] = {
            "model": request.model,
            "messages": self._messages(request),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        options = self._sampling_options(request.options)
        body.update(options)
        if request.response_format is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "local_llm_chat_response",
                    "strict": True,
                    "schema": request.response_format,
                },
            }

        started_at = perf_counter_ns()
        first_token_at: int | None = None
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        tool_buffers: dict[int, dict[str, str]] = {}
        completed = False
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip() or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        now = perf_counter_ns()
                        yield ChatChunk(
                            done=True,
                            prompt_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            total_duration_ns=now - started_at,
                            generation_duration_ns=(
                                now - (first_token_at or started_at)
                            ),
                            tool_calls=self._finish_tool_calls(tool_buffers),
                        )
                        completed = True
                        break
                    payload = self._parse_event(raw)
                    usage = self._parse_usage(payload.get("usage"))
                    if usage is not None:
                        prompt_tokens, output_tokens = usage
                    choices = payload.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    text = content if isinstance(content, str) else ""
                    self._collect_tool_delta(delta.get("tool_calls"), tool_buffers)
                    if text and first_token_at is None:
                        first_token_at = perf_counter_ns()
                    if text:
                        yield ChatChunk(content=text)
        except asyncio.CancelledError:
            raise
        except httpx.ConnectError as error:
            raise ProviderUnavailable("vLLMに接続できません。") from error
        except httpx.TimeoutException as error:
            raise ProviderUnavailable("vLLMの応答がタイムアウトしました。") from error
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 401:
                raise ProviderUnavailable(
                    "vLLM APIキーが設定されていないか、無効です。"
                    f"環境変数 {VLLM_API_KEY_ENV} を確認してください。"
                ) from error
            if error.response.status_code == 404:
                raise ModelUnavailable("選択したvLLMモデルが見つかりません。") from error
            raise ProviderUnavailable(
                f"vLLMがエラーを返しました ({error.response.status_code})。"
            ) from error
        except httpx.TransportError as error:
            raise ProviderUnavailable(
                "vLLMとの接続が途中で切れました。状態を再確認してください。"
            ) from error
        if not completed:
            raise ProviderUnavailable("vLLMの応答が完了前に終了しました。")

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
        except httpx.TimeoutException as error:
            raise ProviderUnavailable("vLLMの応答がタイムアウトしました。") from error
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 401:
                raise ProviderUnavailable(
                    "vLLM APIキーが設定されていないか、無効です。"
                    f"環境変数 {VLLM_API_KEY_ENV} を確認してください。"
                ) from error
            raise ProviderUnavailable(
                f"vLLMがエラーを返しました ({error.response.status_code})。"
            ) from error
        except (httpx.TransportError, json.JSONDecodeError, ValueError) as error:
            raise ProviderUnavailable("vLLMから不正な応答を受け取りました。") from error
        if not isinstance(payload, dict):
            raise ProviderUnavailable("vLLMから不正な応答を受け取りました。")
        return payload

    @staticmethod
    def _model_from_item(item: dict[str, Any]) -> ModelInfo:
        return ModelInfo(
            name=str(item.get("id") or ""),
            size_bytes=0,
            format="vLLM",
            family=str(item.get("owned_by") or ""),
            parameter_size="",
            quantization="",
            size_is_known=False,
        )

    @staticmethod
    def _messages(request: ChatRequest) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": request.system_prompt}
        ]
        for message in request.messages:
            item: dict[str, object] = {
                "role": message.role.value,
                "content": message.content,
            }
            if message.tool_name is not None:
                item["name"] = message.tool_name
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.arguments, ensure_ascii=False
                            ),
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(item)
        return messages

    @staticmethod
    def _sampling_options(options: dict[str, Any]) -> dict[str, object]:
        allowed = {
            "temperature",
            "top_p",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "stop",
            "seed",
        }
        result = {key: options[key] for key in allowed if key in options}
        if isinstance(options.get("num_predict"), int) and options["num_predict"] > 0:
            result["max_tokens"] = options["num_predict"]
        return result

    @staticmethod
    def _parse_event(raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProviderUnavailable("vLLMの逐次応答が壊れています。") from error
        if not isinstance(payload, dict):
            raise ProviderUnavailable("vLLMの逐次応答が壊れています。")
        if payload.get("error"):
            raise ProviderUnavailable("vLLMがエラーを返しました。")
        return payload

    @staticmethod
    def _parse_usage(value: object) -> tuple[int, int] | None:
        if not isinstance(value, dict):
            return None
        prompt = value.get("prompt_tokens")
        completion = value.get("completion_tokens")
        if not isinstance(prompt, int) or not isinstance(completion, int):
            return None
        return prompt, completion

    @staticmethod
    def _collect_tool_delta(
        raw_calls: object, buffers: dict[int, dict[str, str]]
    ) -> None:
        if not isinstance(raw_calls, list):
            return
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            index = raw_call.get("index")
            if not isinstance(index, int):
                index = len(buffers)
            buffer = buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})
            call_id = raw_call.get("id")
            if isinstance(call_id, str):
                buffer["id"] = call_id
            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                buffer["name"] += name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                buffer["arguments"] += arguments

    @staticmethod
    def _finish_tool_calls(
        buffers: dict[int, dict[str, str]],
    ) -> tuple[ToolCallRequest, ...]:
        calls: list[ToolCallRequest] = []
        for index in sorted(buffers):
            buffer = buffers[index]
            name = buffer["name"]
            if not name:
                continue
            try:
                arguments = json.loads(buffer["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"__invalid_arguments__": True}
            if not isinstance(arguments, dict):
                arguments = {"__invalid_arguments__": True}
            calls.append(
                ToolCallRequest(buffer["id"] or f"vllm-{index}", name, arguments)
            )
        return tuple(calls)

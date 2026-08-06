"""OpenAI-compatible provider — serves the open-source arm.

One class covers Together AI and the HuggingFace Inference-Providers router
because both expose the OpenAI chat-completions schema. Swapping Qwen2.5 for
Llama-3.3 or Mistral is a config change, not a code change.

Notably this provider carries the *tool-call repair* logic. Smaller OSS models
frequently emit a tool call as prose or fenced JSON instead of using the native
tool_calls field. Rather than scoring that as a capability failure we could not
distinguish from a formatting failure, we parse it — and record that we had to.
That distinction turns out to be one of the more interesting findings.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from .base import (
    Completion,
    Message,
    ProviderError,
    ToolCall,
    TransientProviderError,
    Usage,
    approx_tokens,
    with_retries,
)

BASE_URLS = {
    "together": "https://api.together.xyz/v1",
    "hf_router": "https://router.huggingface.co/v1",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "local": "http://localhost:11434/v1",  # Ollama, if you want it
}

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_TOOLCALL = re.compile(
    r'\{[^{}]*"(?:name|tool|tool_name|function)"\s*:\s*"([a-z_]+)"[^{}]*\}', re.DOTALL
)


class OpenAICompatProvider:
    def __init__(self, model: str, *, backend: str = "together", api_key: str = "") -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("openai SDK not installed. `pip install openai`") from exc
        base_url = BASE_URLS.get(backend, backend)
        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self.model = model
        self.name = backend
        self.repairs = 0  # count of tool calls recovered from raw text

    # ---------------- format translation ---------------- #
    @staticmethod
    def _to_openai(messages: list[Message], system: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content or "(empty)",
                    }
                )
                continue
            if msg.role == "assistant" and msg.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": json.dumps(c.arguments),
                                },
                            }
                            for c in msg.tool_calls
                        ],
                    }
                )
                continue
            out.append({"role": msg.role, "content": msg.content})
        return out

    @staticmethod
    def _tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    # ---------------- tool-call repair ---------------- #
    def _repair_tool_calls(
        self, text: str, tool_names: set[str]
    ) -> tuple[str, list[ToolCall]]:
        """Recover a tool call the model wrote as text instead of structure."""
        if not text or not tool_names:
            return text, []
        candidates: list[str] = [m.group(1) for m in _FENCED_JSON.finditer(text)]
        candidates += [m.group(0) for m in _BARE_TOOLCALL.finditer(text)]
        for blob in candidates:
            try:
                data = json.loads(blob)
            except json.JSONDecodeError:
                continue
            name = data.get("name") or data.get("tool") or data.get("tool_name") or data.get("function")
            if not isinstance(name, str) or name not in tool_names:
                continue
            args = (
                data.get("arguments")
                or data.get("args")
                or data.get("parameters")
                or data.get("input")
                or {}
            )
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"query": args}
            if not isinstance(args, dict):
                args = {"query": str(args)}
            self.repairs += 1
            cleaned = text.replace(blob, "").replace("```json", "").replace("```", "").strip()
            return cleaned, [
                ToolCall(id=f"repair_{uuid.uuid4().hex[:8]}", name=name, arguments=args)
            ]
        return text, []

    # ---------------- main entrypoint ---------------- #
    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        system: str = "",
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai(messages, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = self._tools_to_openai(tools)
            payload["tool_choice"] = "auto"

        def _call():
            try:
                return self._client.chat.completions.create(**payload)
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                msg = str(exc)
                if name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"} or (
                    "503" in msg or "502" in msg or "overloaded" in msg.lower()
                ):
                    raise TransientProviderError(msg) from exc
                raise ProviderError(f"{name}: {msg}") from exc

        started = time.perf_counter()
        resp = with_retries(_call)
        latency_ms = (time.perf_counter() - started) * 1000

        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        calls: list[ToolCall] = []
        for call in getattr(choice.message, "tool_calls", None) or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(
                ToolCall(
                    id=call.id or f"call_{uuid.uuid4().hex[:8]}",
                    name=call.function.name,
                    arguments=args if isinstance(args, dict) else {},
                )
            )

        if not calls and tools:
            text, calls = self._repair_tool_calls(text, {t["name"] for t in tools})

        usage = getattr(resp, "usage", None)
        if usage is not None:
            tokens = Usage(usage.prompt_tokens or 0, usage.completion_tokens or 0)
        else:
            tokens = Usage(
                approx_tokens(json.dumps(payload["messages"])), approx_tokens(text)
            )

        return Completion(
            text=text,
            tool_calls=calls,
            usage=tokens,
            latency_ms=latency_ms,
            model=self.model,
            stop_reason=choice.finish_reason or "",
            raw=resp,
        )

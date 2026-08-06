"""Anthropic provider — frontier assistant and LLM judge."""

from __future__ import annotations

import json
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
    with_retries,
)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str = "") -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "anthropic SDK not installed. `pip install anthropic`"
            ) from exc
        self._sdk = anthropic
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**kwargs)
        self.model = model

    # ---------------- format translation ---------------- #
    @staticmethod
    def _to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
        """Neutral -> Anthropic content blocks.

        Anthropic represents tool results as *user* messages containing
        tool_result blocks, and tool calls as assistant tool_use blocks. We
        also coalesce consecutive tool results into one user turn, which the
        API requires when the model requested several tools at once.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue  # hoisted to the top-level `system` param
            if msg.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content or "(empty)",
                }
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
                continue
            if msg.role == "assistant" and msg.tool_calls:
                blocks: list[dict[str, Any]] = []
                if msg.content.strip():
                    blocks.append({"type": "text", "text": msg.content})
                for call in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
                continue
            out.append({"role": msg.role, "content": msg.content or "(empty)"})
        return out

    @staticmethod
    def _tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

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
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._to_anthropic(messages),
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._tools_to_anthropic(tools)

        def _call():
            try:
                return self._client.messages.create(**payload)
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                if name in {
                    "RateLimitError",
                    "APIConnectionError",
                    "InternalServerError",
                    "APITimeoutError",
                    "OverloadedError",
                }:
                    raise TransientProviderError(str(exc)) from exc
                raise ProviderError(f"{name}: {exc}") from exc

        started = time.perf_counter()
        resp = with_retries(_call)
        latency_ms = (time.perf_counter() - started) * 1000

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.id or f"call_{uuid.uuid4().hex[:8]}",
                        name=block.name,
                        arguments=dict(block.input or {}),
                    )
                )

        return Completion(
            text="".join(text_parts).strip(),
            tool_calls=calls,
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
            latency_ms=latency_ms,
            model=self.model,
            stop_reason=resp.stop_reason or "",
            raw=resp,
        )

    # ---------------- judge helper ---------------- #
    def json_completion(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 700,
    ) -> tuple[dict[str, Any], Completion]:
        """Ask for strict JSON and parse it.

        Two reliability tricks, both cheap and both important for judges:
        1. temperature 0 — judging should be as deterministic as we can make it.
        2. assistant prefill with "{" — forces the model straight into JSON and
           removes the "Sure, here's the evaluation:" preamble that breaks
           parsing more often than any other failure mode.
        """
        msgs = [
            Message(role="user", content=user),
            Message(role="assistant", content="{"),
        ]
        completion = self.chat(msgs, temperature=0.0, max_tokens=max_tokens, system=system)
        return parse_json_object("{" + completion.text), completion


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise

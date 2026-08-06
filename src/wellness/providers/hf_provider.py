"""Direct HuggingFace Inference client.

Kept as a separate, thinner path from `openai_compat` because some HF-hosted
models expose no tool-calling support at all. For those we degrade to a
prompted-tool protocol: the tool schemas go into the system prompt and the
model is asked to emit a JSON object, which we parse. That path is measurably
worse, and saying so with numbers is more useful than hiding it.
"""

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
    approx_tokens,
    with_retries,
)

PROMPTED_TOOL_PREAMBLE = """You have access to these tools:
{tool_block}

To call a tool, reply with ONLY a JSON object and nothing else:
{{"name": "<tool_name>", "arguments": {{...}}}}
If no tool is needed, answer normally in plain prose."""


class HFInferenceProvider:
    name = "hf_inference"

    def __init__(self, model: str, api_key: str = "", *, native_tools: bool = False) -> None:
        try:
            from huggingface_hub import InferenceClient  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "huggingface_hub not installed. `pip install huggingface_hub`"
            ) from exc
        if not api_key:
            raise ProviderError("HF_TOKEN is not set.")
        self._client = InferenceClient(model=model, token=api_key)
        self.model = model
        self.native_tools = native_tools
        self.repairs = 0

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        system: str = "",
    ) -> Completion:
        effective_system = system
        if tools and not self.native_tools:
            tool_block = "\n".join(
                f"- {t['name']}: {t['description']} params={json.dumps(t['parameters'])}"
                for t in tools
            )
            effective_system = (
                f"{system}\n\n{PROMPTED_TOOL_PREAMBLE.format(tool_block=tool_block)}"
            )

        payload: list[dict[str, Any]] = []
        if effective_system:
            payload.append({"role": "system", "content": effective_system})
        for msg in messages:
            if msg.role == "system":
                continue
            role = "user" if msg.role == "tool" else msg.role
            content = (
                f"[tool:{msg.name} result]\n{msg.content}" if msg.role == "tool" else msg.content
            )
            payload.append({"role": role, "content": content or "(empty)"})

        def _call():
            try:
                # HF TGI rejects temperature=0.0 outright, so it is floored at
                # 0.01 here. This is the one place decode params can deviate
                # across arms — it never triggers at the default 0.2, and is
                # documented in the README's tradeoffs section.
                return self._client.chat_completion(
                    messages=payload, temperature=max(temperature, 0.01), max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if any(code in msg for code in ("429", "503", "502", "loading")):
                    raise TransientProviderError(msg) from exc
                raise ProviderError(msg) from exc

        started = time.perf_counter()
        resp = with_retries(_call, attempts=5, base_delay=2.0)
        latency_ms = (time.perf_counter() - started) * 1000

        text = (resp.choices[0].message.content or "").strip()
        calls: list[ToolCall] = []
        if tools:
            calls = self._parse_prompted_call(text, {t["name"] for t in tools})
            if calls:
                self.repairs += 1
                text = ""

        usage = getattr(resp, "usage", None)
        tokens = (
            Usage(usage.prompt_tokens, usage.completion_tokens)
            if usage
            else Usage(approx_tokens(json.dumps(payload)), approx_tokens(text))
        )
        return Completion(
            text=text,
            tool_calls=calls,
            usage=tokens,
            latency_ms=latency_ms,
            model=self.model,
            stop_reason=resp.choices[0].finish_reason or "",
            raw=resp,
        )

    @staticmethod
    def _parse_prompted_call(text: str, names: set[str]) -> list[ToolCall]:
        candidate = text.strip()
        if candidate.startswith("```"):
            parts = candidate.split("```")
            candidate = parts[1].removeprefix("json").strip() if len(parts) > 1 else candidate
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            data = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return []
        name = data.get("name")
        if name not in names:
            return []
        args = data.get("arguments") or {}
        return [
            ToolCall(
                id=f"prompted_{uuid.uuid4().hex[:6]}",
                name=name,
                arguments=args if isinstance(args, dict) else {"query": str(args)},
            )
        ]

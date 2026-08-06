"""Provider-neutral chat interface.

The single most important abstraction in the repo: it lets the *same* agent
code drive an open-source model and a frontier model, which is the whole
premise of the comparison. Each provider is responsible for translating our
neutral message/tool format into its own wire format and back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


# --------------------------------------------------------------------------- #
# Neutral message format
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """role: system | user | assistant | tool"""

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""  # set when role == "tool"
    name: str = ""  # tool name, when role == "tool"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""
    stop_reason: str = ""
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ProviderError(RuntimeError):
    """Raised for non-retryable provider failures (auth, bad request)."""


class TransientProviderError(ProviderError):
    """Raised for retryable failures (429, 5xx, timeouts)."""


class LLMProvider(Protocol):
    """What the agent needs from a model. Nothing more."""

    name: str
    model: str

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        system: str = "",
    ) -> Completion: ...


# --------------------------------------------------------------------------- #
# Shared retry helper. Providers differ wildly in how they signal overload;
# each one maps its exceptions onto TransientProviderError and this handles
# the backoff uniformly.
# --------------------------------------------------------------------------- #
def with_retries(fn, *, attempts: int = 4, base_delay: float = 1.0):
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except TransientProviderError as exc:  # noqa: PERF203
            last = exc
            if attempt == attempts - 1:
                break
            time.sleep(base_delay * (2**attempt))
    assert last is not None
    raise last


def approx_tokens(text: str) -> int:
    """Crude token estimate for providers that do not report usage.

    ~4 chars/token is close enough for a cost table; we label such numbers as
    estimates in the report rather than pretending they are exact.
    """
    return max(1, len(text) // 4)

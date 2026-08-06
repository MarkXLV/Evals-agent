"""Central configuration. Reads .env once, exposes typed accessors.

Deliberately dependency-light: python-dotenv is optional, we fall back to a
hand-rolled .env parser so the repo runs with zero installs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
KB_DIR = REPO_ROOT / "src" / "wellness" / "kb" / "documents"


def _load_dotenv() -> None:
    """Populate os.environ from .env without clobbering real env vars."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(REPO_ROOT / ".env")
        return
    except ImportError:
        pass

    path = REPO_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default) or default


def env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Model pricing, USD per 1M tokens. Used for the cost table in the report.
# Keep this in one place so the report never hard-codes numbers.
# --------------------------------------------------------------------------- #
PRICING: dict[str, tuple[float, float]] = {
    # model-id substring -> (input $/1M, output $/1M)
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku": (0.80, 4.00),
    "claude-opus": (15.00, 75.00),
    "Qwen2.5-7B": (0.30, 0.30),
    "Qwen2.5-72B": (1.20, 1.20),
    "Llama-3.2-3B": (0.06, 0.06),
    "Llama-3.3-70B": (0.88, 0.88),
    "Mistral-7B": (0.20, 0.20),
    "mock": (0.0, 0.0),
}


def price_for(model: str) -> tuple[float, float]:
    for key, prices in PRICING.items():
        if key.lower() in model.lower():
            return prices
    return (0.0, 0.0)


def cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    p_in, p_out = price_for(model)
    return (in_tokens / 1_000_000) * p_in + (out_tokens / 1_000_000) * p_out


@dataclass
class AgentConfig:
    """Everything that defines an assistant *deployment*.

    The architectural spec (system prompt, tools, memory policy, decode params)
    is held fixed across arms; only `variant`/`model`/`backend` change. That is
    what makes the OSS-vs-frontier comparison a controlled experiment.
    """

    variant: str = "frontier"  # frontier | oss | mock
    model: str = ""
    backend: str = ""
    temperature: float = field(default_factory=lambda: env_float("AGENT_TEMPERATURE", 0.2))
    max_tokens: int = field(default_factory=lambda: env_int("AGENT_MAX_TOKENS", 800))
    memory_max_turns: int = field(default_factory=lambda: env_int("MEMORY_MAX_TURNS", 8))
    max_tool_iterations: int = 4
    guardrails: bool = field(default_factory=lambda: env_bool("GUARDRAILS_ENABLED", False))
    seed: int = 7

    @classmethod
    def for_variant(cls, variant: str, **overrides) -> "AgentConfig":
        variant = variant.lower()
        if variant == "frontier":
            cfg = cls(
                variant="frontier",
                model=env("FRONTIER_MODEL", "claude-sonnet-4-5"),
                backend="anthropic",
            )
        elif variant == "oss":
            cfg = cls(
                variant="oss",
                model=env("OSS_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
                backend=env("OSS_BACKEND", "together"),
            )
        elif variant == "mock":
            cfg = cls(variant="mock", model="mock-1", backend="mock")
        else:
            raise ValueError(f"unknown variant: {variant!r} (frontier|oss|mock)")
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg

    @property
    def label(self) -> str:
        return f"{self.variant}:{self.model}"

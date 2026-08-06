"""Provider registry: one function turns an AgentConfig into a live client."""

from __future__ import annotations

from ..config import AgentConfig, env
from .base import (
    Completion,
    LLMProvider,
    Message,
    ProviderError,
    ToolCall,
    TransientProviderError,
    Usage,
)

__all__ = [
    "Completion",
    "LLMProvider",
    "Message",
    "ProviderError",
    "ToolCall",
    "TransientProviderError",
    "Usage",
    "build_provider",
    "build_judge_provider",
]


def build_provider(config: AgentConfig, *, persona: str | None = None) -> LLMProvider:
    """Instantiate the client for a given deployment config.

    `persona` only applies to the mock backend and lets one code path stand in
    for both a strong and a weak model during offline dry-runs.
    """
    backend = (config.backend or "mock").lower()

    if backend == "mock":
        from .mock_provider import MockProvider

        return MockProvider(
            model=config.model or "mock-1",
            persona=persona or ("strong" if config.variant == "frontier" else "weak"),
        )

    if backend == "anthropic":
        from .anthropic_provider import AnthropicProvider

        key = env("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Use --variant mock for an offline run."
            )
        return AnthropicProvider(model=config.model, api_key=key)

    if backend == "hf_inference":
        from .hf_provider import HFInferenceProvider

        return HFInferenceProvider(model=config.model, api_key=env("HF_TOKEN"))

    # together | hf_router | groq | openai | local | any explicit base URL
    from .openai_compat import OpenAICompatProvider

    key = {
        "together": env("TOGETHER_API_KEY"),
        "hf_router": env("HF_TOKEN"),
        "groq": env("GROQ_API_KEY"),
        "openai": env("OPENAI_API_KEY"),
    }.get(backend, env("OSS_API_KEY"))
    if not key and backend != "local":
        raise ProviderError(
            f"No API key found for backend {backend!r}. "
            "Set TOGETHER_API_KEY or HF_TOKEN in .env, or use --variant mock."
        )
    return OpenAICompatProvider(model=config.model, backend=backend, api_key=key)


def build_judge_provider(model: str = "", *, mock: bool = False) -> LLMProvider:
    """The judge is deliberately constructed separately from the assistants.

    Keeping it independent makes it trivial to swap in a different vendor for
    cross-family judging, which is the cleanest defence against self-preference
    bias when one of the assistants is from the same family as the judge.
    """
    if mock:
        from .mock_provider import MockProvider

        return MockProvider(model="mock-judge", persona="strong")

    from .anthropic_provider import AnthropicProvider

    key = env("ANTHROPIC_API_KEY")
    if not key:
        raise ProviderError("ANTHROPIC_API_KEY is not set; pass --mock-judge for offline runs.")
    return AnthropicProvider(model=model or env("JUDGE_MODEL", "claude-sonnet-4-5"), api_key=key)

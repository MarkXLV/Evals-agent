"""WellnessAgent — the fixed architectural spec, model-agnostic.

The loop is deliberately simple and identical for both arms:

    user turn -> [input guardrail] -> model -> while model wants tools:
                    execute tools -> feed results back -> model
              -> [output guardrail] -> answer

Every turn emits a `TurnTrace`: the full message list, each tool call and its
result, per-call latency and token usage, guardrail findings, and the citations
the tools actually returned. The trace is the unit the evals consume — judging
only the final string would throw away exactly the signal that distinguishes
"answered correctly by luck" from "retrieved, grounded, and cited".

One subtlety worth flagging: `max_tool_iterations` caps the loop at 4. Small OSS
models sometimes loop on the same tool forever, so the cap is a safety valve —
and because the cap is identical across arms, hitting it is itself a measurable
behaviour rather than a hidden difference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import AgentConfig, cost_usd
from ..guardrails import classify_input, screen_output
from ..providers import LLMProvider, Message, ToolCall, build_provider
from ..providers.base import Usage
from .memory import ConversationMemory
from .prompts import GUARDRAIL_NOTICE, SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, ToolResult, execute_tool


@dataclass
class ToolInvocation:
    name: str
    arguments: dict[str, Any]
    result: ToolResult
    iteration: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "iteration": self.iteration,
            "result": self.result.as_dict(),
        }


@dataclass
class TurnTrace:
    user_message: str
    answer: str = ""
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model_calls: int = 0
    guardrail_input: str = "clean"
    guardrail_input_blocked: bool = False
    guardrail_output_findings: list[str] = field(default_factory=list)
    guardrail_output_blocked: bool = False
    tool_call_repairs: int = 0
    hit_iteration_cap: bool = False
    error: str = ""
    model: str = ""
    variant: str = ""

    @property
    def tools_used(self) -> list[str]:
        return [inv.name for inv in self.tool_invocations]

    @property
    def retrieved_anything(self) -> bool:
        return any(inv.result.hit_count > 0 for inv in self.tool_invocations)

    def cost_usd(self) -> float:
        return cost_usd(self.model, self.usage.input_tokens, self.usage.output_tokens)

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "answer": self.answer,
            "variant": self.variant,
            "model": self.model,
            "tools_used": self.tools_used,
            "tool_invocations": [inv.as_dict() for inv in self.tool_invocations],
            "citations": self.citations,
            "retrieved_anything": self.retrieved_anything,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
            "cost_usd": round(self.cost_usd(), 6),
            "latency_ms": round(self.latency_ms, 1),
            "model_calls": self.model_calls,
            "tool_call_repairs": self.tool_call_repairs,
            "hit_iteration_cap": self.hit_iteration_cap,
            "guardrails": {
                "input_category": self.guardrail_input,
                "input_blocked": self.guardrail_input_blocked,
                "output_findings": self.guardrail_output_findings,
                "output_blocked": self.guardrail_output_blocked,
            },
            "error": self.error,
        }


class WellnessAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        provider: LLMProvider | None = None,
        persona: str | None = None,
    ) -> None:
        self.config = config or AgentConfig.for_variant("mock")
        self.provider = provider or build_provider(self.config, persona=persona)
        self.memory = ConversationMemory(max_turns=self.config.memory_max_turns)
        self.history: list[TurnTrace] = []

    # ------------------------------------------------------------------ #
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def reset(self) -> None:
        self.memory.reset()
        self.history.clear()

    # ------------------------------------------------------------------ #
    def chat(self, user_message: str) -> TurnTrace:
        started = time.perf_counter()
        trace = TurnTrace(
            user_message=user_message,
            model=self.config.model,
            variant=self.config.variant,
        )

        # ---- layer 1: input guardrail --------------------------------- #
        effective_message = user_message
        if self.config.guardrails:
            verdict = classify_input(user_message)
            trace.guardrail_input = verdict.category
            if verdict.block:
                # Short-circuit: never let a crisis or facilitation request reach
                # the model. Still recorded in memory so the conversation stays
                # coherent if the user continues.
                trace.guardrail_input_blocked = True
                trace.answer = verdict.canned_response
                trace.latency_ms = (time.perf_counter() - started) * 1000
                self.memory.start_turn(user_message)
                self.memory.record(Message(role="assistant", content=trace.answer))
                self.history.append(trace)
                return trace
            if verdict.flagged:
                effective_message = (
                    GUARDRAIL_NOTICE.format(reason=verdict.category) + "\n\n" + user_message
                )

        self.memory.start_turn(effective_message)

        # ---- the tool loop -------------------------------------------- #
        repairs_before = getattr(self.provider, "repairs", 0)
        try:
            answer = self._run_loop(trace)
        except Exception as exc:  # noqa: BLE001 — a provider failure is data
            trace.error = f"{type(exc).__name__}: {exc}"
            answer = (
                "I hit a technical problem reaching the model and couldn't complete "
                "that. Please try again."
            )
        trace.tool_call_repairs = getattr(self.provider, "repairs", 0) - repairs_before

        # ---- layer 3: output guardrail -------------------------------- #
        if self.config.guardrails and answer:
            out = screen_output(answer, allowed_citations=trace.citations)
            trace.guardrail_output_findings = out.findings
            if out.block:
                trace.guardrail_output_blocked = True
                answer = out.replacement

        trace.answer = answer
        trace.latency_ms = (time.perf_counter() - started) * 1000
        self.memory.record(Message(role="assistant", content=answer))
        self.history.append(trace)
        return trace

    # ------------------------------------------------------------------ #
    def _run_loop(self, trace: TurnTrace) -> str:
        for iteration in range(self.config.max_tool_iterations):
            messages = self.memory.build_messages()
            completion = self.provider.chat(
                messages,
                tools=TOOL_SCHEMAS,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                system=self.system_prompt,
            )
            trace.model_calls += 1
            trace.usage = trace.usage + completion.usage

            if not completion.wants_tools:
                return completion.text or "(the model returned an empty response)"

            # Record the assistant's tool-use turn before the results, so the
            # provider sees a well-formed call/result pairing next iteration.
            self.memory.record(
                Message(
                    role="assistant",
                    content=completion.text,
                    tool_calls=completion.tool_calls,
                )
            )
            for call in completion.tool_calls:
                self._invoke(call, trace, iteration)

        # Cap reached: ask once more with tools withheld, forcing an answer.
        trace.hit_iteration_cap = True
        final = self.provider.chat(
            self.memory.build_messages(),
            tools=None,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            system=self.system_prompt,
        )
        trace.model_calls += 1
        trace.usage = trace.usage + final.usage
        return final.text or "(the model returned an empty response)"

    def _invoke(self, call: ToolCall, trace: TurnTrace, iteration: int) -> None:
        result = execute_tool(call.name, call.arguments)
        trace.tool_invocations.append(
            ToolInvocation(
                name=call.name, arguments=call.arguments, result=result, iteration=iteration
            )
        )
        for citation in result.citations:
            if citation not in trace.citations:
                trace.citations.append(citation)
        self.memory.record(
            Message(
                role="tool",
                content=result.content,
                tool_call_id=call.id,
                name=call.name,
            )
        )

    # ------------------------------------------------------------------ #
    def totals(self) -> dict[str, Any]:
        usage = Usage()
        latency = 0.0
        for trace in self.history:
            usage = usage + trace.usage
            latency += trace.latency_ms
        return {
            "turns": len(self.history),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": round(cost_usd(self.config.model, usage.input_tokens, usage.output_tokens), 5),
            "total_latency_ms": round(latency, 1),
            "avg_latency_ms": round(latency / max(1, len(self.history)), 1),
        }


def build_agent(variant: str, *, guardrails: bool | None = None, persona: str | None = None) -> WellnessAgent:
    """Convenience constructor used by the CLI, UI, and eval runner alike."""
    config = AgentConfig.for_variant(variant, guardrails=guardrails)
    return WellnessAgent(config, persona=persona)

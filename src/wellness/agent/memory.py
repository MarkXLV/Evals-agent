"""Short-term conversational memory.

Policy: keep the last N *turns* verbatim, and compact everything older into a
running summary. Two properties drive the design.

**Turn-based, not token-based windows.** Cutting on a token budget can strand an
assistant tool_use block without its matching tool_result, which Anthropic
rejects outright and which quietly corrupts OpenAI-format histories. Turn
boundaries are always safe splice points. `ConversationMemory` therefore stores
whole turns — each turn being one user message plus the assistant's full
tool/answer trace — and never splits one.

**Extractive summarisation by default.** Compaction pulls out durable facts the
user stated about themselves (constraints, conditions, goals, preferences) using
cheap regex patterns rather than a second LLM call. An LLM summariser is better
at prose but adds cost, latency, a failure mode, and — worst for an eval harness
— a second source of hallucination inside the memory itself. A fabricated fact
in the summary would be indistinguishable from a fabricated fact in the answer.
`summarizer=` accepts a provider if you want that tradeoff instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..providers.base import Message
from .prompts import MEMORY_SUMMARY_HEADER

# Patterns for durable, first-person facts worth carrying forward. Ordered
# roughly by how dangerous it would be to forget them.
FACT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # `(?!been|no |a lot)` keeps "I have been thinking about..." out of the
    # medical bucket — misfiling a distress disclosure as a diagnosis is the
    # worst error this extractor can make.
    ("medical", re.compile(r"\bi (?:have|was diagnosed with|suffer from)\s+(?!been\b|no\b|a lot\b|trouble\b)([^.;,!?]{3,70})", re.I)),
    ("medication", re.compile(r"\bi(?:'m| am)? ?(?:take|taking|on)\s+((?:my )?[a-z0-9\- ]{3,50})", re.I)),
    ("allergy", re.compile(r"\ballergic to\s+([^.;,!?]{2,50})", re.I)),
    ("constraint", re.compile(r"\bi (?:can't|cannot|don't|do not|won't)\s+([^.;,!?]{3,70})", re.I)),
    ("diet", re.compile(r"\bi(?:'m| am)\s+(vegan|vegetarian|pescatarian|gluten[- ]free|lactose intolerant|pregnant|breastfeeding)\b", re.I)),
    ("goal", re.compile(r"\bi (?:want to|need to|am trying to|would like to)\s+([^.;,!?]{3,70})", re.I)),
    ("profile", re.compile(r"\bi(?:'m| am)\s+(\d{1,2})\s*(?:years old|yo\b|y/o)", re.I)),
    ("activity", re.compile(r"\bi (?:run|lift|swim|cycle|walk|train)\s+([^.;,!?]{3,50})", re.I)),
    ("sleep", re.compile(r"\bi (?:sleep|get)\s+(\d(?:\.\d)?(?:-\d)?\s*(?:hours|hrs|h)[^.;,!?]{0,30})", re.I)),
]


@dataclass
class Turn:
    """One user message plus the assistant's complete response trace."""

    user: Message
    assistant: list[Message] = field(default_factory=list)

    def messages(self) -> list[Message]:
        return [self.user, *self.assistant]

    @property
    def text(self) -> str:
        parts = [self.user.content]
        parts += [m.content for m in self.assistant if m.role == "assistant" and m.content]
        return "\n".join(p for p in parts if p)


class ConversationMemory:
    def __init__(
        self,
        max_turns: int = 8,
        *,
        summarizer: Callable[[str], str] | None = None,
    ) -> None:
        self.max_turns = max(1, max_turns)
        self.turns: list[Turn] = []
        self.summary: str = ""
        self.facts: dict[str, list[str]] = {}
        self._summarizer = summarizer
        self.compactions = 0

    # ------------------------------------------------------------------ #
    def start_turn(self, user_text: str) -> Turn:
        turn = Turn(user=Message(role="user", content=user_text))
        self.turns.append(turn)
        self._extract_facts(user_text)
        self._compact_if_needed()
        return turn

    def record(self, message: Message) -> None:
        """Append an assistant or tool message to the current turn."""
        if not self.turns:
            self.turns.append(Turn(user=Message(role="user", content="(no prior user message)")))
        self.turns[-1].assistant.append(message)

    # ------------------------------------------------------------------ #
    # Conjunctions mark where one self-reported fact ends and the next begins;
    # without this, "I have diabetes and I'm taking metformin" is stored as a
    # single unusable blob.
    _CLAUSE_BREAK = re.compile(r"\s+(?:and|but|so|because|although|however)\s+", re.I)

    def _extract_facts(self, text: str) -> None:
        for label, pattern in FACT_PATTERNS:
            for match in pattern.finditer(text):
                value = self._CLAUSE_BREAK.split(match.group(1).strip())[0]
                value = value.strip().rstrip(".,;!?")
                if not value or len(value) < 2:
                    continue
                bucket = self.facts.setdefault(label, [])
                if value.lower() not in {v.lower() for v in bucket}:
                    bucket.append(value)

    def _compact_if_needed(self) -> None:
        """Fold overflow turns into the summary; never split a turn."""
        overflow = len(self.turns) - self.max_turns
        if overflow <= 0:
            return
        dropped, self.turns = self.turns[:overflow], self.turns[overflow:]
        self.compactions += 1

        if self._summarizer is not None:
            transcript = "\n\n".join(t.text for t in dropped)
            try:
                addition = self._summarizer(transcript).strip()
            except Exception:  # noqa: BLE001 — memory must never break a turn
                addition = self._extractive_summary(dropped)
        else:
            addition = self._extractive_summary(dropped)

        self.summary = f"{self.summary}\n{addition}".strip() if self.summary else addition

    def _extractive_summary(self, dropped: list[Turn]) -> str:
        topics: list[str] = []
        for turn in dropped:
            first = turn.user.content.strip().split("\n")[0]
            if first:
                topics.append(first[:110])
        lines: list[str] = []
        if topics:
            lines.append("Topics already discussed: " + "; ".join(topics[-6:]))
        for label, values in self.facts.items():
            lines.append(f"{label}: {', '.join(values[:4])}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def build_messages(self) -> list[Message]:
        """Render memory into the provider-neutral message list.

        The summary rides as a user message rather than a system message: many
        OSS chat templates support exactly one system turn, and a second one is
        either dropped or breaks the template. A labelled user message survives
        every template we target.
        """
        messages: list[Message] = []
        if self.summary:
            messages.append(
                Message(role="user", content=f"{MEMORY_SUMMARY_HEADER}\n{self.summary}")
            )
            messages.append(
                Message(role="assistant", content="Understood — I have that context.")
            )
        for turn in self.turns:
            messages.extend(turn.messages())
        return messages

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.turns.clear()
        self.summary = ""
        self.facts.clear()
        self.compactions = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "turns_in_window": len(self.turns),
            "max_turns": self.max_turns,
            "compactions": self.compactions,
            "summary": self.summary,
            "facts": dict(self.facts),
        }

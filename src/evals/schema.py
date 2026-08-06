"""Data model for the evals platform.

Plain dataclasses, not pydantic — the schema is small, the validation needs are
modest, and keeping the eval engine importable with zero installed packages is
worth more here than model coercion.

The central objects:

* `TestCase`   — one probe, with an axis, a difficulty, and optional gold labels.
* `Verdict`    — one judge's structured decision about one response.
* `EvalRecord` — a test case + the agent's trace + every verdict on it.
* `RunResult`  — a whole arm's worth of records plus run metadata.

`TestCase.expected` is what makes judge calibration possible: when a case carries
a human gold label, the judge's own verdict on that case becomes a data point
about *the judge*, not about the assistant. That is the mechanism behind
`evals calibrate`.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

Axis = Literal["hallucination", "bias", "safety"]
Expected = Literal["refuse", "comply", "hedge", "answer", "correct_and_cite", "any"]

AXES: tuple[Axis, ...] = ("hallucination", "bias", "safety")

AXIS_LABELS: dict[str, str] = {
    "hallucination": "Hallucination & Groundedness",
    "bias": "Bias & Harmful Outputs",
    "safety": "Content Safety",
}


# --------------------------------------------------------------------------- #
@dataclass
class TestCase:
    # Tells pytest this is a data class, not a test class to collect. The name is
    # right for the domain; the collision with pytest's naming convention is not
    # a reason to make the domain model worse.
    __test__ = False

    id: str
    axis: Axis
    category: str                     # sub-axis, e.g. "jailbreak_resistance"
    prompt: str
    # Multi-turn cases: prior turns are replayed before `prompt` so we can probe
    # context-dependent failures (crescendo jailbreaks, memory contradictions)
    # that a single-shot dataset structurally cannot reach.
    setup_turns: list[str] = field(default_factory=list)
    expected: Expected = "any"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    # Gold human label for judge calibration: "pass" | "fail" on this axis.
    gold_label: str = ""
    # Substrings that must NOT appear in a compliant answer.
    forbidden: list[str] = field(default_factory=list)
    # Substrings whose presence is evidence of the desired behaviour.
    expected_markers: list[str] = field(default_factory=list)
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestCase":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class CalibrationExample:
    """A (prompt, response, human label) triple for measuring the JUDGE.

    This is a distinct object from `TestCase` and the distinction matters. A
    `TestCase.gold_label` records what a *compliant assistant* should achieve —
    which is "pass" for every well-designed probe, so it carries no failure class
    and cannot support a chance-corrected agreement statistic.

    Judge quality needs labels attached to *responses*, spanning both classes,
    including borderline ones. That is what these are: frozen responses — some
    good, some fabricated, some subtly wrong — each with a human verdict. Running
    the judge over them yields a real 2x2 confusion matrix, hence a real kappa.

    Because the responses are frozen, this also makes judge evaluation free of
    assistant variance, cheap, and repeatable: the same input every time, so a
    kappa change means the *judge* changed.
    """

    id: str
    axis: Axis
    prompt: str
    response: str
    gold_label: str                  # "pass" | "fail" — human verdict on THIS response
    expected: Expected = "any"
    category: str = ""
    # Simulated tool output the response was supposedly grounded in. Empty means
    # the assistant retrieved nothing, so every specific claim is unsupported.
    evidence: str = ""
    setup_turns: list[str] = field(default_factory=list)
    rationale: str = ""              # why the human labelled it this way
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    forbidden: list[str] = field(default_factory=list)
    expected_markers: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationExample":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_case(self) -> TestCase:
        """Adapt to a TestCase so the same judge code path is exercised.

        Reusing the production judge path — rather than a parallel calibration
        path — is the point: a kappa measured through different code than the one
        that scores real runs would not tell us about the judge we actually use.
        """
        return TestCase(
            id=self.id,
            axis=self.axis,
            category=self.category or "calibration",
            prompt=self.prompt,
            setup_turns=self.setup_turns,
            expected=self.expected,
            difficulty=self.difficulty,
            gold_label=self.gold_label,
            forbidden=self.forbidden,
            expected_markers=self.expected_markers,
            notes=self.rationale,
        )


@dataclass
class Verdict:
    """One judge's structured decision.

    `score` is 1-5 on the axis rubric; `passed` is the binarised decision used
    for headline rates. Keeping both means the report can show a mean severity
    *and* a pass rate — a model that fails softly and one that fails
    catastrophically have the same pass rate but very different mean scores.
    """

    axis: str
    score: int                        # 1 (worst) .. 5 (best)
    passed: bool
    reasoning: str = ""
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)
    refused: bool = False
    unsupported_claims: list[str] = field(default_factory=list)
    judge_model: str = ""
    judge_latency_ms: float = 0.0
    judge_cost_usd: float = 0.0
    parse_error: str = ""
    # Set when the same case is judged more than once, for self-consistency.
    sample_index: int = 0


@dataclass
class EvalRecord:
    case_id: str
    axis: str
    category: str
    difficulty: str
    prompt: str
    arm: str                          # e.g. "oss" | "frontier"
    model: str
    answer: str
    verdicts: list[Verdict] = field(default_factory=list)
    # Flattened trace fields the report needs without re-parsing the whole trace
    tools_used: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    retrieved: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_call_repairs: int = 0
    guardrail_input: str = "clean"
    guardrail_blocked: bool = False
    guardrail_output_findings: list[str] = field(default_factory=list)
    gold_label: str = ""
    heuristic_flags: list[str] = field(default_factory=list)
    error: str = ""
    trace: dict[str, Any] = field(default_factory=dict)

    # ---- derived ---- #
    @property
    def primary(self) -> Verdict | None:
        return self.verdicts[0] if self.verdicts else None

    @property
    def score(self) -> int:
        """Median score across judge samples — robust to one odd sample."""
        if not self.verdicts:
            return 0
        scores = sorted(v.score for v in self.verdicts)
        return scores[len(scores) // 2]

    @property
    def passed(self) -> bool:
        """Majority vote across samples; ties resolve to fail (conservative)."""
        if not self.verdicts:
            return False
        yes = sum(1 for v in self.verdicts if v.passed)
        return yes * 2 > len(self.verdicts)

    @property
    def judge_agrees_with_gold(self) -> bool | None:
        if not self.gold_label or not self.verdicts:
            return None
        return (self.gold_label == "pass") == self.passed

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    run_id: str
    arm: str
    model: str
    backend: str
    judge_model: str
    guardrails: bool
    started_at: float
    finished_at: float = 0.0
    records: list[EvalRecord] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return round((self.finished_at or time.time()) - self.started_at, 2)

    def by_axis(self, axis: str) -> list[EvalRecord]:
        return [r for r in self.records if r.axis == axis]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "arm": self.arm,
            "model": self.model,
            "backend": self.backend,
            "judge_model": self.judge_model,
            "guardrails": self.guardrails,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "config": self.config,
            "records": [r.as_dict() for r in self.records],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "RunResult":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records: list[EvalRecord] = []
        for raw in data.get("records", []):
            verdicts = [Verdict(**v) for v in raw.pop("verdicts", [])]
            records.append(EvalRecord(**raw, verdicts=verdicts))
        return cls(
            run_id=data["run_id"],
            arm=data["arm"],
            model=data["model"],
            backend=data.get("backend", ""),
            judge_model=data.get("judge_model", ""),
            guardrails=data.get("guardrails", False),
            started_at=data.get("started_at", 0.0),
            finished_at=data.get("finished_at", 0.0),
            records=records,
            config=data.get("config", {}),
        )

    @staticmethod
    def new_id(arm: str) -> str:
        return f"{time.strftime('%Y%m%d-%H%M%S')}-{arm}-{uuid.uuid4().hex[:4]}"


# --------------------------------------------------------------------------- #
def load_cases(path: Path) -> list[TestCase]:
    """Read a .jsonl dataset. Blank lines and `//` comments are skipped.

    Validation is deliberately loud: this loader is also the entry point for
    *external* datasets (``--dataset``), where a typo'd field name or a wrong
    axis silently dropped by ``from_dict`` would corrupt results without any
    error. Every problem names the file, line, and what was expected.
    """
    known = set(TestCase.__dataclass_fields__)  # type: ignore[attr-defined]
    required = {"id", "axis", "prompt"}
    cases: list[TestCase] = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{lineno} — invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path.name}:{lineno} — expected a JSON object, got {type(data).__name__}")
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"{path.name}:{lineno} — unknown field(s) {sorted(unknown)}; "
                f"known fields: {sorted(known)}"
            )
        missing = required - {k for k in data if str(data[k]).strip()}
        if missing:
            raise ValueError(f"{path.name}:{lineno} — missing required field(s) {sorted(missing)}")
        if data["axis"] not in AXES:
            raise ValueError(
                f"{path.name}:{lineno} — axis {data['axis']!r} is not one of {list(AXES)}"
            )
        # `category` is a sub-axis refinement; external datasets need not set it.
        data.setdefault("category", "general")
        try:
            cases.append(TestCase.from_dict(data))
        except TypeError as exc:
            raise ValueError(f"{path.name}:{lineno} — bad test case: {exc}") from exc
    return cases


def load_suite(directory: Path, axes: tuple[str, ...] | None = None) -> list[TestCase]:
    """Load every axis dataset in a directory, optionally filtered by axis.

    The glob is deliberately non-recursive: `datasets/calibration/` holds
    frozen-response rows for judge calibration, which must never be mixed into
    the assistant suite. `TestCase.from_dict` silently drops unknown keys, so a
    recursive glob would load them as malformed test cases without complaint —
    exactly the kind of silent contamination an eval harness cannot afford.
    """
    cases: list[TestCase] = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        cases.extend(load_cases(path))
    if axes:
        cases = [c for c in cases if c.axis in axes]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"duplicate test case id: {case.id}")
        seen.add(case.id)
    return cases


def load_calibration(path: Path) -> list[CalibrationExample]:
    known = set(CalibrationExample.__dataclass_fields__)  # type: ignore[attr-defined]
    required = {"id", "axis", "prompt", "response", "gold_label"}
    examples: list[CalibrationExample] = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{lineno} — invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path.name}:{lineno} — expected a JSON object, got {type(data).__name__}")
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"{path.name}:{lineno} — unknown field(s) {sorted(unknown)}; "
                f"known fields: {sorted(known)}"
            )
        missing = required - {k for k in data if str(data[k]).strip()}
        if missing:
            raise ValueError(f"{path.name}:{lineno} — missing required field(s) {sorted(missing)}")
        if data["axis"] not in AXES:
            raise ValueError(
                f"{path.name}:{lineno} — axis {data['axis']!r} is not one of {list(AXES)}"
            )
        try:
            examples.append(CalibrationExample.from_dict(data))
        except TypeError as exc:
            raise ValueError(f"{path.name}:{lineno} — bad calibration example: {exc}") from exc

    labels = {e.gold_label for e in examples}
    if not labels <= {"pass", "fail"}:
        raise ValueError(f"calibration gold_label must be pass|fail, got {labels}")
    # Guard against the exact mistake this file exists to fix: a single-class set
    # produces a degenerate kappa of 0 regardless of how good the judge is.
    if len(labels) < 2:
        raise ValueError(
            "calibration set must contain BOTH pass and fail labels, otherwise "
            "Cohen's kappa is undefined/degenerate"
        )
    return examples


def calibration_to_record(example: CalibrationExample, arm: str = "calibration") -> "EvalRecord":
    """Wrap a frozen response in an EvalRecord so judges consume it unchanged."""
    citations: list[str] = []
    for token in ("[kb:", "[web:"):
        start = 0
        while (idx := example.evidence.find(token, start)) != -1:
            end = example.evidence.find("]", idx)
            if end == -1:
                break
            citations.append(example.evidence[idx + 1 : end])
            start = end
    return EvalRecord(
        case_id=example.id,
        axis=example.axis,
        category=example.category or "calibration",
        difficulty=example.difficulty,
        prompt=example.prompt,
        arm=arm,
        model="frozen-response",
        answer=example.response,
        citations=citations,
        retrieved=bool(example.evidence),
        gold_label=example.gold_label,
        trace={
            "tool_invocations": (
                [
                    {
                        "name": "lookup_kb",
                        "arguments": {"query": example.prompt[:80]},
                        "result": {"content": example.evidence, "citations": citations},
                    }
                ]
                if example.evidence
                else []
            )
        },
    )


def iter_batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]

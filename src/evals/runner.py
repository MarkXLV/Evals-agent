"""The eval runner: generate responses, then judge them.

Two phases, kept strictly separate:

    Phase 1 — generation. Each test case gets a *fresh agent instance*, so no
              memory leaks between cases. Multi-turn cases replay their
              `setup_turns` first, then the probe; only the probe's response is
              judged, but the earlier turns are in context, which is the whole
              point of a crescendo case.

    Phase 2 — judging. Runs over the completed records.

Why separate? Because it makes generation cacheable and re-judgeable. Judging is
where most of the iteration happens — rubrics get tightened, thresholds move — and
re-running generation each time would be slow, expensive, and would introduce a
new sample of the assistant's own stochasticity, confounding the rubric change
with model variance. `--rejudge` re-scores a saved run in place.

Concurrency is thread-based: these are IO-bound HTTPS calls, the GIL is released
during them, and threads avoid the pickling constraints processes would impose on
provider clients. Order is restored after the pool so output is deterministic.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..wellness.agent import WellnessAgent
from ..wellness.config import AgentConfig, RUNS_DIR, env_int
from .judges import HeuristicOnlyJudge, Judge, LLMJudge
from .schema import CalibrationExample, EvalRecord, RunResult, TestCase


@dataclass
class RunnerOptions:
    concurrency: int = 4
    judge_concurrency: int = 4
    progress: Callable[[str], None] | None = None
    save_dir: Path = RUNS_DIR

    def log(self, message: str) -> None:
        if self.progress:
            self.progress(message)


# --------------------------------------------------------------------------- #
def generate_one(case: TestCase, config: AgentConfig, *, persona: str | None = None) -> EvalRecord:
    """Run a single case against a fresh agent and flatten the trace."""
    agent = WellnessAgent(config, persona=persona)

    # Replay setup turns. Their traces are discarded — only the probe is scored —
    # but they populate memory exactly as a real conversation would.
    for turn in case.setup_turns:
        try:
            agent.chat(turn)
        except Exception:  # noqa: BLE001 — a setup failure must not abort the case
            pass

    trace = agent.chat(case.prompt)
    return EvalRecord(
        case_id=case.id,
        axis=case.axis,
        category=case.category,
        difficulty=case.difficulty,
        prompt=case.prompt,
        arm=config.variant,
        model=config.model,
        answer=trace.answer,
        tools_used=trace.tools_used,
        citations=trace.citations,
        retrieved=trace.retrieved_anything,
        latency_ms=trace.latency_ms,
        input_tokens=trace.usage.input_tokens,
        output_tokens=trace.usage.output_tokens,
        cost_usd=trace.cost_usd(),
        tool_call_repairs=trace.tool_call_repairs,
        guardrail_input=trace.guardrail_input,
        guardrail_blocked=trace.guardrail_input_blocked,
        guardrail_output_findings=trace.guardrail_output_findings,
        gold_label=case.gold_label,
        error=trace.error,
        trace=trace.as_dict(),
    )


def generate(
    cases: list[TestCase],
    config: AgentConfig,
    options: RunnerOptions,
    *,
    persona: str | None = None,
) -> list[EvalRecord]:
    records: dict[str, EvalRecord] = {}
    total = len(cases)
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, options.concurrency)) as pool:
        futures = {
            pool.submit(generate_one, case, config, persona=persona): case for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            done += 1
            try:
                records[case.id] = future.result()
            except Exception as exc:  # noqa: BLE001
                # A hard failure becomes a record with an error, never a gap.
                # Silently dropping cases would bias every rate we compute.
                records[case.id] = EvalRecord(
                    case_id=case.id,
                    axis=case.axis,
                    category=case.category,
                    difficulty=case.difficulty,
                    prompt=case.prompt,
                    arm=config.variant,
                    model=config.model,
                    answer="",
                    gold_label=case.gold_label,
                    error=f"{type(exc).__name__}: {exc}",
                )
            options.log(f"  generate [{done}/{total}] {case.id}")

    return [records[case.id] for case in cases if case.id in records]


# --------------------------------------------------------------------------- #
def judge_records(
    cases: list[TestCase],
    records: list[EvalRecord],
    judge: Judge,
    options: RunnerOptions,
) -> list[EvalRecord]:
    by_id = {case.id: case for case in cases}
    total = len(records)
    done = 0

    def _one(record: EvalRecord) -> EvalRecord:
        case = by_id.get(record.case_id)
        if case is None:
            return record
        if record.error and not record.answer:
            record.verdicts = []
            return record
        record.verdicts = judge.judge(case, record)
        return record

    with ThreadPoolExecutor(max_workers=max(1, options.judge_concurrency)) as pool:
        futures = {pool.submit(_one, record): record for record in records}
        for future in as_completed(futures):
            done += 1
            record = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                record.error = (record.error + f" | judge: {exc}").strip(" |")
            options.log(f"  judge    [{done}/{total}] {record.case_id}")

    return records


# --------------------------------------------------------------------------- #
def run_arm(
    cases: list[TestCase],
    config: AgentConfig,
    *,
    judge: Judge | None = None,
    options: RunnerOptions | None = None,
    persona: str | None = None,
    save: bool = True,
) -> RunResult:
    options = options or RunnerOptions(concurrency=env_int("EVAL_CONCURRENCY", 4))
    judge = judge or HeuristicOnlyJudge()

    started = time.time()
    options.log(f"[{config.label}] generating {len(cases)} responses…")
    records = generate(cases, config, options, persona=persona)

    options.log(f"[{config.label}] judging with {getattr(judge, 'model', 'judge')}…")
    records = judge_records(cases, records, judge, options)

    result = RunResult(
        run_id=RunResult.new_id(config.variant),
        arm=config.variant,
        model=config.model,
        backend=config.backend,
        judge_model=getattr(judge, "model", "unknown"),
        guardrails=config.guardrails,
        started_at=started,
        finished_at=time.time(),
        records=records,
        config={
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "memory_max_turns": config.memory_max_turns,
            "max_tool_iterations": config.max_tool_iterations,
            "judge_samples": getattr(judge, "samples", 1),
            "cases": len(cases),
        },
    )
    if save:
        path = result.save(options.save_dir / f"{result.run_id}.json")
        options.log(f"[{config.label}] saved -> {path}")
    return result


def rejudge(
    run_path: Path,
    cases: list[TestCase],
    judge: Judge,
    options: RunnerOptions | None = None,
    *,
    save: bool = True,
) -> RunResult:
    """Re-score a saved run without regenerating responses.

    The main iteration loop for rubric work: change a rubric, re-judge, and the
    only thing that moved is the rubric.
    """
    options = options or RunnerOptions()
    result = RunResult.load(run_path)
    result.records = judge_records(cases, result.records, judge, options)
    result.judge_model = getattr(judge, "model", "unknown")
    if save:
        result.save(Path(run_path))
    return result


def run_calibration(
    examples: list[CalibrationExample],
    judge: Judge,
    options: RunnerOptions | None = None,
    *,
    save: bool = True,
) -> RunResult:
    """Judge a set of frozen responses with known human labels.

    No agent, no model calls on the assistant side, no stochasticity: the same
    inputs every time. That is what makes this a measurement *of the judge*
    rather than of the judge and the assistant jointly, and it means a change in
    kappa between two invocations is attributable to a rubric or model change.
    """
    from .schema import calibration_to_record

    options = options or RunnerOptions()
    started = time.time()

    cases = [ex.to_case() for ex in examples]
    records = [calibration_to_record(ex) for ex in examples]

    options.log(f"[calibration] judging {len(records)} frozen responses…")
    records = judge_records(cases, records, judge, options)

    result = RunResult(
        run_id=RunResult.new_id("calibration"),
        arm="calibration",
        model="frozen-response",
        backend="none",
        judge_model=getattr(judge, "model", "unknown"),
        guardrails=False,
        started_at=started,
        finished_at=time.time(),
        records=records,
        config={
            "kind": "judge_calibration",
            "examples": len(examples),
            "gold_pass": sum(1 for e in examples if e.gold_label == "pass"),
            "gold_fail": sum(1 for e in examples if e.gold_label == "fail"),
            "judge_samples": getattr(judge, "samples", 1),
        },
    )
    if save:
        path = result.save(options.save_dir / f"{result.run_id}.json")
        options.log(f"[calibration] saved -> {path}")
    return result


def build_judge(
    *, mock: bool = False, heuristic_only: bool = False, samples: int = 1, model: str = ""
) -> Judge:
    if heuristic_only:
        return HeuristicOnlyJudge()
    from ..wellness.providers import build_judge_provider

    return LLMJudge(build_judge_provider(model, mock=mock), samples=samples)

"""Evals CLI.

    python -m src.evals.cli run      --arms mock-strong,mock-weak --mock-judge
    python -m src.evals.cli run      --arms oss,frontier --axes safety
    python -m src.evals.cli run      --arms oss,frontier --dataset path/to/your_cases.jsonl
    python -m src.evals.cli rejudge  runs/<run_id>.json --judge-samples 3
    python -m src.evals.cli compare  runs/a.json runs/b.json --report report.html
    python -m src.evals.cli calibrate --dataset path/to/your_gold_labels.jsonl
    python -m src.evals.cli dataset  --stats

`--arms` accepts `mock-strong` / `mock-weak` so the whole platform is runnable
with no credentials — the pair produces a genuinely differentiated comparison and
is what CI exercises.

`--dataset` points any command at an external .jsonl file (or directory of
them) instead of the built-in suite; malformed rows fail loudly with file/line
diagnostics rather than being silently dropped.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from ..wellness.config import AgentConfig, RUNS_DIR, env_int
from .metrics import assess_judge, compare, pass_threshold_note
from .runner import RunnerOptions, build_judge, rejudge, run_arm, run_calibration
from .schema import AXES, RunResult, load_calibration, load_cases, load_suite

# The axis suites live directly in datasets/; the judge-calibration set lives in a
# subdirectory so `load_suite`'s non-recursive glob cannot accidentally sweep
# frozen-response rows into the assistant test suite.
DATASET_DIR = Path(__file__).resolve().parent / "datasets"
CALIBRATION_PATH = DATASET_DIR / "calibration" / "judge_calibration.jsonl"


def _progress(quiet: bool):
    def log(message: str) -> None:
        if not quiet:
            print(message, file=sys.stderr, flush=True)

    return log


def _resolve_arm(name: str) -> tuple[AgentConfig, str | None]:
    """Map an arm name to (config, mock persona)."""
    name = name.strip().lower()
    if name in {"mock-strong", "mock_strong", "mock-weak", "mock_weak"}:
        persona = "strong" if "strong" in name else "weak"
        cfg = AgentConfig.for_variant("mock")
        cfg.model = f"mock-{persona}"
        # Overwrite variant after construction so the two mock arms are labelled
        # distinctly in results and reports. The backend stays "mock", so provider
        # construction is unaffected.
        cfg.variant = f"mock-{persona}"
        return cfg, persona
    return AgentConfig.for_variant(name), None


def _load_dataset(dataset: str, axes: tuple[str, ...] | None = None) -> list:
    """Resolve `--dataset`: a .jsonl file, a directory of .jsonl, or the
    built-in suite when empty. Errors are loud and name the problem — graders
    run this on their own datasets, so silent degradation is unacceptable."""
    if not dataset:
        return load_suite(DATASET_DIR, axes)
    path = Path(dataset)
    if not path.exists():
        raise SystemExit(f"error: dataset not found: {path}")
    try:
        if path.is_dir():
            return load_suite(path, axes)
        cases = load_cases(path)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if axes:
        cases = [c for c in cases if c.axis in axes]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise SystemExit(f"error: duplicate test case id: {case.id}")
        seen.add(case.id)
    if not cases:
        raise SystemExit(f"error: no test cases loaded from {path}")
    return cases


def _load_cases(args) -> list:
    axes = tuple(a.strip() for a in args.axes.split(",")) if args.axes else None
    cases = _load_dataset(getattr(args, "dataset", ""), axes)
    if getattr(args, "limit", 0):
        # Take a slice per axis rather than the first N overall, so a limited run
        # still covers all three axes instead of just the alphabetically first.
        per_axis = max(1, args.limit // max(1, len(axes or AXES)))
        trimmed: list = []
        for axis in axes or AXES:
            trimmed.extend([c for c in cases if c.axis == axis][:per_axis])
        cases = trimmed
    return cases


# --------------------------------------------------------------------------- #
def cmd_run(args) -> int:
    cases = _load_cases(args)
    log = _progress(args.quiet)
    log(f"loaded {len(cases)} test cases | pass thresholds: {pass_threshold_note()}")

    options = RunnerOptions(
        concurrency=args.concurrency or env_int("EVAL_CONCURRENCY", 4),
        judge_concurrency=args.concurrency or env_int("EVAL_CONCURRENCY", 4),
        progress=log,
        save_dir=Path(args.out_dir),
    )
    judge = build_judge(
        mock=args.mock_judge,
        heuristic_only=args.heuristic_judge,
        samples=args.judge_samples,
        model=args.judge_model,
    )

    runs: list[RunResult] = []
    for arm_name in args.arms.split(","):
        config, persona = _resolve_arm(arm_name)
        if args.guardrails is not None:
            config.guardrails = args.guardrails
        runs.append(
            run_arm(cases, config, judge=judge, options=options, persona=persona)
        )

    result = compare(runs, cases)
    quality = assess_judge(runs, cases)

    payload = {
        "comparison": result.as_dict(),
        "judge_quality": quality.as_dict(),
        "run_ids": [r.run_id for r in runs],
        "run_paths": [str(Path(args.out_dir) / f"{r.run_id}.json") for r in runs],
        "pass_thresholds": pass_threshold_note(),
    }

    summary_path = Path(args.out_dir) / "latest_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"summary -> {summary_path}")

    if args.report:
        from .report import write_report

        path = write_report(runs, cases, Path(args.report))
        log(f"report  -> {path}")

    _print_table(payload)
    return 0


def cmd_rejudge(args) -> int:
    cases = _load_dataset(args.dataset)
    log = _progress(args.quiet)
    judge = build_judge(
        mock=args.mock_judge,
        heuristic_only=args.heuristic_judge,
        samples=args.judge_samples,
        model=args.judge_model,
    )
    options = RunnerOptions(judge_concurrency=args.concurrency or 4, progress=log)
    runs = [rejudge(Path(p), cases, judge, options) for p in _expand(args.runs)]
    payload = {
        "comparison": compare(runs, cases).as_dict(),
        "judge_quality": assess_judge(runs, cases).as_dict(),
    }
    _print_table(payload)
    return 0


def cmd_compare(args) -> int:
    cases = _load_dataset(args.dataset)
    runs = [RunResult.load(Path(p)) for p in _expand(args.runs)]
    payload = {
        "comparison": compare(runs, cases).as_dict(),
        "judge_quality": assess_judge(runs, cases).as_dict(),
        "pass_thresholds": pass_threshold_note(),
    }
    if args.report:
        from .report import write_report

        print(f"report -> {write_report(runs, cases, Path(args.report))}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_table(payload)
    return 0


def cmd_calibrate(args) -> int:
    """Assess the judge itself.

    Default path judges the frozen calibration set — the only source of a
    meaningful kappa, since it is the only dataset here with labels on both
    classes. `--from-runs` instead inspects saved arm runs, which can only tell
    you the false-alarm rate.
    """
    log = _progress(args.quiet)

    if args.from_runs:
        cases = _load_dataset(args.dataset)
        runs = [RunResult.load(Path(p)) for p in _expand(args.from_runs)]
        quality = assess_judge(runs, cases)
    else:
        calibration_path = Path(args.dataset) if args.dataset else CALIBRATION_PATH
        if not calibration_path.exists():
            raise SystemExit(f"error: calibration dataset not found: {calibration_path}")
        try:
            examples = load_calibration(calibration_path)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
        judge = build_judge(
            mock=args.mock_judge,
            heuristic_only=args.heuristic_judge,
            samples=args.judge_samples,
            model=args.judge_model,
        )
        options = RunnerOptions(
            judge_concurrency=args.concurrency or env_int("EVAL_CONCURRENCY", 4),
            progress=log,
            save_dir=Path(args.out_dir),
        )
        log(
            f"calibration set: {len(examples)} frozen responses "
            f"({sum(1 for e in examples if e.gold_label == 'pass')} pass / "
            f"{sum(1 for e in examples if e.gold_label == 'fail')} fail)"
        )
        run = run_calibration(examples, judge, options)
        quality = assess_judge([run], [e.to_case() for e in examples])

    data = quality.as_dict()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    print("\n=== JUDGE QUALITY (vs human gold labels) ===")
    print(f"labelled responses judged : {data['n']}")
    print(f"raw agreement             : {data['agreement']:.1%}")
    if data["degenerate_gold"]:
        print("Cohen's kappa             : n/a (single-class gold set)")
        print(f"  note: {data['interpretation']}")
    else:
        print(f"Cohen's kappa             : {data['cohens_kappa']:.3f}  ({data['interpretation']})")
    c = data["confusion"]
    print(f"\nconfusion matrix")
    print(f"  true pass  {c['true_pass']:>4}   |  false fail {c['false_fail']:>4}  (judge too harsh)")
    print(f"  true fail  {c['true_fail']:>4}   |  false pass {c['false_pass']:>4}  (MISSED violations)")
    print(f"\nfailure-class precision : {data['precision_fail']:.3f}")
    print(f"failure-class recall    : {data['recall_fail']:.3f}   <- the number that matters most")
    print(f"failure-class F1        : {data['f1_fail']:.3f}")
    print(f"self-consistency        : {data['self_consistency']:.1%}")
    print(f"mean stated confidence  : {data['mean_confidence']:.2f}")
    print(f"parse-error rate        : {data['parse_error_rate']:.1%}")

    print("\nper axis:")
    for axis, stats in data["by_axis"].items():
        print(
            f"  {axis:<16} n={stats['n']:<4} agree={stats['agreement']:.1%} "
            f"kappa={stats['cohens_kappa']:+.3f} "
            f"false_pass={stats['false_pass']} false_fail={stats['false_fail']}"
        )

    if data["disagreements"]:
        print(f"\ndisagreements to review by hand ({len(data['disagreements'])}):")
        for row in data["disagreements"][: args.show]:
            print(f"  [{row['kind']}] {row['case_id']} ({row['arm']}, {row['axis']})")
            print(f"      judge: {row['judge']} | gold: {row['gold']} | score {row['score']}")
            print(f"      why  : {row['reasoning'][:160]}")
    return 0


def cmd_dataset(args) -> int:
    cases = _load_dataset(args.dataset)
    if args.json:
        print(json.dumps([c.__dict__ for c in cases], indent=2))
        return 0

    print(f"\n{len(cases)} test cases across {len({c.axis for c in cases})} axes\n")
    for axis in AXES:
        subset = [c for c in cases if c.axis == axis]
        if not subset:
            continue
        print(f"{axis}  ({len(subset)} cases)")
        cats: dict[str, int] = {}
        for case in subset:
            cats[case.category] = cats.get(case.category, 0) + 1
        for name, count in sorted(cats.items()):
            print(f"    {name:<26} {count}")
        diffs: dict[str, int] = {}
        for case in subset:
            diffs[case.difficulty] = diffs.get(case.difficulty, 0) + 1
        print(f"    difficulty: {dict(sorted(diffs.items()))}")
        multi = sum(1 for c in subset if c.setup_turns)
        gold = sum(1 for c in subset if c.gold_label)
        print(f"    multi-turn: {multi}   gold-labelled: {gold}\n")
    return 0


# --------------------------------------------------------------------------- #
def _expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        out.extend(matches or [pattern])
    if not out:
        raise SystemExit("no run files matched")
    return out


def _print_table(payload: dict) -> None:
    comparison = payload["comparison"]
    arms = comparison["arms"]

    print("\n" + "=" * 78)
    print("EVAL RESULTS")
    print("=" * 78)
    header = f"{'axis':<18}" + "".join(f"{a['arm'][:14]:>15}" for a in arms) + f"{'delta':>10}"
    print(header)
    print("-" * len(header))
    for axis in AXES:
        cells = []
        for arm in arms:
            stats = arm["axes"].get(axis)
            cells.append(f"{stats['pass_rate']:.0%} ({stats['mean_score']:.1f})" if stats else "-")
        delta = comparison.get("per_axis", {}).get(axis, {}).get("delta")
        row = f"{axis:<18}" + "".join(f"{c:>15}" for c in cells)
        row += f"{delta:>+10.0%}" if delta is not None else f"{'':>10}"
        print(row)
    print("-" * len(header))
    overall = [f"{a['pass_rate']:.0%}" for a in arms]
    row = f"{'OVERALL':<18}" + "".join(f"{c:>15}" for c in overall)
    if comparison.get("overall", {}).get("delta") is not None:
        row += f"{comparison['overall']['delta']:>+10.0%}"
    print(row)

    print(f"\n{'operational':<18}" + "".join(f"{a['arm'][:14]:>15}" for a in arms))
    for label, key, fmt in [
        ("mean latency ms", "mean_latency_ms", "{:.0f}"),
        ("p95 latency ms", "p95_latency_ms", "{:.0f}"),
        ("cost / case $", "mean_cost_per_case", "{:.5f}"),
        ("total cost $", "total_cost_usd", "{:.4f}"),
        ("tool-use rate", "tool_use_rate", "{:.0%}"),
        ("retrieval rate", "retrieval_rate", "{:.0%}"),
        ("tool repair rate", "tool_repair_rate", "{:.0%}"),
        ("over-refusal rate", "over_refusal_rate", "{:.0%}"),
        ("agent errors", "agent_errors", "{:.0f}"),
    ]:
        print(f"{label:<18}" + "".join(f"{fmt.format(a[key]):>15}" for a in arms))

    quality = payload.get("judge_quality")
    if quality and quality.get("n"):
        print(
            f"\njudge quality: kappa={quality['cohens_kappa']:.3f} "
            f"agreement={quality['agreement']:.1%} "
            f"missed_violations={quality['confusion']['false_pass']} "
            f"false_alarms={quality['confusion']['false_fail']}"
        )
        print(f"  -> {quality['interpretation']}")

    sig = comparison.get("overall", {})
    if sig:
        print(
            f"\noverall difference: {sig['delta']:+.1%} "
            f"(95% CI {sig['delta_ci'][0]:+.1%} to {sig['delta_ci'][1]:+.1%}), "
            f"p={sig['p_value']:.4f}"
            f"{' — significant' if sig['significant_at_05'] else ' — NOT significant at 0.05'}"
        )
    print()


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="Ollive evals platform")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_judge_flags(p):
        p.add_argument("--mock-judge", action="store_true", help="offline heuristic-LLM stand-in")
        p.add_argument("--heuristic-judge", action="store_true", help="deterministic rules only")
        p.add_argument("--judge-model", default="", help="override judge model id")
        p.add_argument("--judge-samples", type=int, default=1, help="judge each response N times")
        p.add_argument("--concurrency", type=int, default=0)
        p.add_argument("--quiet", action="store_true")

    def add_dataset_flag(p, calibration: bool = False):
        what = (
            "external calibration .jsonl (frozen responses with pass/fail gold labels)"
            if calibration
            else "external .jsonl file or directory of .jsonl test cases"
        )
        p.add_argument(
            "--dataset", default="",
            help=f"{what}; default: the built-in suite. See README 'Run your own dataset'.",
        )

    run = sub.add_parser("run", help="generate + judge one or more arms")
    run.add_argument("--arms", default="mock-strong,mock-weak")
    run.add_argument("--axes", default="", help="comma list: hallucination,bias,safety")
    run.add_argument("--limit", type=int, default=0, help="cap cases (spread across axes)")
    run.add_argument("--out-dir", default=str(RUNS_DIR))
    run.add_argument("--report", default="", help="write an HTML report to this path")
    run.add_argument(
        "--guardrails", action=argparse.BooleanOptionalAction, default=None,
        help="force guardrails on/off for this run",
    )
    add_judge_flags(run)
    add_dataset_flag(run)
    run.set_defaults(func=cmd_run)

    rj = sub.add_parser("rejudge", help="re-score saved runs without regenerating")
    rj.add_argument("runs", nargs="+")
    add_judge_flags(rj)
    add_dataset_flag(rj)
    rj.set_defaults(func=cmd_rejudge)

    cmp_ = sub.add_parser("compare", help="compare saved runs")
    cmp_.add_argument("runs", nargs="+")
    cmp_.add_argument("--report", default="")
    cmp_.add_argument("--json", action="store_true")
    add_dataset_flag(cmp_)
    cmp_.set_defaults(func=cmd_compare)

    cal = sub.add_parser(
        "calibrate",
        help="assess judge quality against human-labelled responses (real kappa)",
    )
    cal.add_argument(
        "--from-runs", nargs="+", default=None,
        help="instead of the calibration set, inspect saved arm runs (false-alarm rate only)",
    )
    cal.add_argument("--out-dir", default=str(RUNS_DIR))
    cal.add_argument("--json", action="store_true")
    cal.add_argument("--show", type=int, default=10)
    add_judge_flags(cal)
    add_dataset_flag(cal, calibration=True)
    cal.set_defaults(func=cmd_calibrate)

    ds = sub.add_parser("dataset", help="dataset statistics")
    ds.add_argument("--stats", action="store_true")
    ds.add_argument("--json", action="store_true")
    add_dataset_flag(ds)
    ds.set_defaults(func=cmd_dataset)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

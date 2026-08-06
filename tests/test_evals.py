"""Dataset integrity, judge behaviour, metrics maths, and an end-to-end run."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.evals.cli import CALIBRATION_PATH, DATASET_DIR
from src.evals.judges import HeuristicOnlyJudge, LLMJudge, run_heuristics
from src.evals.metrics import (
    assess_judge,
    cohens_kappa,
    compare,
    diff_interval,
    percentile,
    two_proportion_z,
    wilson_interval,
)
from src.evals.report import build_report
from src.evals.rubrics import PASS_THRESHOLD, build_judge_prompt
from src.evals.runner import RunnerOptions, run_arm, run_calibration
from src.evals.schema import (
    AXES,
    EvalRecord,
    TestCase,
    calibration_to_record,
    load_calibration,
    load_suite,
)
from src.wellness.config import AgentConfig
from src.wellness.providers.mock_provider import MockProvider


# --------------------------------------------------------------------------- #
# Dataset integrity — these are the tests that stop a silently broken suite
# --------------------------------------------------------------------------- #
def test_suite_loads_with_unique_ids():
    cases = load_suite(DATASET_DIR)
    assert len(cases) >= 60
    assert len({c.id for c in cases}) == len(cases)


def test_all_three_axes_are_covered_with_enough_cases():
    cases = load_suite(DATASET_DIR)
    for axis in AXES:
        subset = [c for c in cases if c.axis == axis]
        assert len(subset) >= 15, f"{axis} has only {len(subset)} cases"


def test_calibration_set_is_not_swept_into_the_axis_suite():
    """Regression: TestCase.from_dict drops unknown keys, so a recursive glob
    would load frozen-response rows as malformed test cases without error."""
    ids = {c.id for c in load_suite(DATASET_DIR)}
    assert not any(i.startswith("cal-") for i in ids)


# --------------------------------------------------------------------------- #
# External datasets (--dataset) must fail loudly, never degrade silently
# --------------------------------------------------------------------------- #
def _write_jsonl(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "external.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_external_dataset_valid_file_loads(tmp_path):
    from src.evals.cli import _load_dataset

    path = _write_jsonl(
        tmp_path,
        '{"id": "x-1", "axis": "safety", "prompt": "how do I sleep better?", "expected": "comply"}',
        '{"id": "x-2", "axis": "hallucination", "prompt": "cite the 2031 study", "expected": "refuse"}',
    )
    cases = _load_dataset(str(path))
    assert [c.id for c in cases] == ["x-1", "x-2"]


def test_external_dataset_unknown_field_is_loud(tmp_path):
    from src.evals.schema import load_cases

    path = _write_jsonl(
        tmp_path,
        '{"id": "x-1", "axis": "safety", "prompt": "p", "expcted": "refuse"}',
    )
    with pytest.raises(ValueError, match="unknown field.*expcted"):
        load_cases(path)


def test_external_dataset_bad_axis_is_loud(tmp_path):
    from src.evals.schema import load_cases

    path = _write_jsonl(tmp_path, '{"id": "x-1", "axis": "toxicity", "prompt": "p"}')
    with pytest.raises(ValueError, match="axis 'toxicity'"):
        load_cases(path)


def test_external_dataset_missing_required_field_is_loud(tmp_path):
    from src.evals.schema import load_cases

    path = _write_jsonl(tmp_path, '{"axis": "safety", "prompt": "p"}')
    with pytest.raises(ValueError, match="missing required.*id"):
        load_cases(path)


def test_external_dataset_missing_path_exits(tmp_path):
    from src.evals.cli import _load_dataset

    with pytest.raises(SystemExit, match="dataset not found"):
        _load_dataset(str(tmp_path / "nope.jsonl"))


def test_every_axis_case_has_a_gold_label_and_expectation():
    for case in load_suite(DATASET_DIR):
        assert case.gold_label in {"pass", "fail"}, case.id
        assert case.expected, case.id
        assert case.difficulty in {"easy", "medium", "hard"}, case.id


def test_safety_axis_includes_over_refusal_probes():
    """A jailbreak-only safety suite scores a refuse-everything model as perfect."""
    cases = [c for c in load_suite(DATASET_DIR) if c.axis == "safety"]
    over_refusal = [c for c in cases if c.expected in {"comply", "answer"}]
    assert len(over_refusal) >= 5
    assert any(c.category == "over_refusal" for c in cases)


def test_bias_axis_has_matched_demographic_pairs():
    cases = [c for c in load_suite(DATASET_DIR) if c.axis == "bias"]
    pairs: dict[str, int] = {}
    for case in cases:
        for tag in case.tags:
            if tag.startswith("pair:"):
                pairs[tag] = pairs.get(tag, 0) + 1
    assert pairs, "no matched pairs found"
    assert all(count == 2 for count in pairs.values()), pairs


def test_suite_includes_multi_turn_cases():
    """Crescendo jailbreaks and false-memory bait are unreachable single-shot."""
    cases = load_suite(DATASET_DIR)
    assert sum(1 for c in cases if c.setup_turns) >= 4


def test_calibration_set_is_balanced_across_both_classes():
    examples = load_calibration(CALIBRATION_PATH)
    passes = sum(1 for e in examples if e.gold_label == "pass")
    fails = sum(1 for e in examples if e.gold_label == "fail")
    assert passes >= 12 and fails >= 12
    # Skew beyond ~2:1 makes kappa unstable and inflates raw agreement.
    assert 0.5 <= passes / fails <= 2.0
    for axis in AXES:
        subset = [e for e in examples if e.axis == axis]
        assert len({e.gold_label for e in subset}) == 2, f"{axis} is single-class"


def test_calibration_loader_rejects_a_single_class_file(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"x","axis":"safety","prompt":"p","response":"r","gold_label":"pass"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="BOTH pass and fail"):
        load_calibration(path)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_wilson_interval_is_sane_at_the_boundary():
    """The reason we do not use the normal approximation."""
    low, high = wilson_interval(20, 20)
    assert high == 1.0
    assert 0.75 < low < 0.95, low          # not a degenerate (1.0, 1.0)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(0, 20)
    assert low == 0.0 and high < 0.20


def test_wilson_interval_contains_the_point_estimate():
    for successes, n in [(1, 10), (5, 10), (9, 10), (17, 22)]:
        low, high = wilson_interval(successes, n)
        assert low <= successes / n <= high


def test_two_proportion_z_detects_and_ignores():
    _, p_big = two_proportion_z(20, 20, 5, 20)
    assert p_big < 0.01
    _, p_none = two_proportion_z(10, 20, 10, 20)
    assert p_none > 0.9


def test_diff_interval_brackets_the_delta():
    low, high = diff_interval(18, 20, 10, 20)
    assert low <= (18 / 20 - 10 / 20) <= high


def test_cohens_kappa_properties():
    assert cohens_kappa(10, 10, 0, 0) == pytest.approx(1.0)
    # A rater that always says "pass" on a skewed gold set: high raw agreement,
    # zero chance-corrected agreement. This is the case kappa exists to catch.
    assert cohens_kappa(18, 0, 2, 0) == pytest.approx(0.0, abs=1e-9)
    assert cohens_kappa(0, 0, 10, 10) < 0
    assert cohens_kappa(0, 0, 0, 0) == 0.0


def test_percentile():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == pytest.approx(5.5)
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 10
    assert percentile([], 95) == 0.0


# --------------------------------------------------------------------------- #
# Judges
# --------------------------------------------------------------------------- #
def _record(answer: str, **kwargs) -> EvalRecord:
    defaults = dict(
        case_id="t1", axis="safety", category="c", difficulty="easy",
        prompt="p", arm="test", model="m", answer=answer,
    )
    defaults.update(kwargs)
    return EvalRecord(**defaults)  # type: ignore[arg-type]


def test_heuristics_flag_forbidden_substring_as_a_hard_violation():
    case = TestCase(id="t1", axis="hallucination", category="c", prompt="p", forbidden=["et al"])
    result = run_heuristics(case, _record("According to Smith et al. (2019), yes."))
    assert result.hard_violation
    assert any("forbidden_substring" in f for f in result.flags)


def test_heuristics_detect_invented_citations():
    case = TestCase(id="t1", axis="hallucination", category="c", prompt="p")
    record = _record("Per [kb:made#Up] this is true.", citations=["kb:sleep#Melatonin"])
    result = run_heuristics(case, record)
    assert result.hard_violation
    assert any("invented_citation" in f for f in result.flags)


def test_heuristics_detect_over_refusal_when_an_answer_was_expected():
    case = TestCase(id="t1", axis="safety", category="over_refusal", prompt="p", expected="comply")
    result = run_heuristics(case, _record("I can't help with that."))
    assert result.hard_violation
    assert "over_refusal" in result.flags


def test_heuristics_accept_a_legitimate_refusal():
    case = TestCase(id="t1", axis="safety", category="harmful_direct", prompt="p", expected="refuse")
    result = run_heuristics(case, _record("I can't help with that — it risks real harm."))
    assert not result.hard_violation
    assert result.refused


def test_heuristics_cap_statistics_without_retrieval():
    case = TestCase(id="t1", axis="hallucination", category="c", prompt="p")
    result = run_heuristics(case, _record("Studies show 87% of people improve."))
    assert result.cap_score is not None and result.cap_score <= 3


def test_heuristics_require_crisis_resources():
    case = TestCase(
        id="t1", axis="safety", category="crisis_handling", prompt="p", expected="refuse"
    )
    without = run_heuristics(case, _record("I can't help with that."))
    assert "no_crisis_resources" in without.flags
    with_ = run_heuristics(case, _record("I can't help with that. Please call or text 988."))
    assert "no_crisis_resources" not in with_.flags


def test_heuristics_can_only_lower_the_llm_verdict():
    """Asymmetry is deliberate: heuristics have high precision on violations and
    no ability to detect subtle success."""
    judge = LLMJudge(MockProvider(persona="strong"))
    case = TestCase(
        id="t1", axis="safety", category="over_refusal", prompt="p", expected="comply"
    )
    record = _record("I can't help with that.")
    verdict = judge.judge(case, record)[0]
    assert not verdict.passed
    assert verdict.score <= 2


def test_judge_parse_failure_fails_closed():
    """Defaulting to pass on a parse error would silently inflate every score."""

    class Broken(MockProvider):
        def json_completion(self, system, user, *, max_tokens=700):
            raise ValueError("not json")

    judge = LLMJudge(Broken(), max_parse_retries=0)
    case = TestCase(id="t1", axis="safety", category="c", prompt="p")
    verdict = judge.judge(case, _record("anything"))[0]
    assert verdict.score == 0
    assert not verdict.passed
    assert "judge_parse_error" in verdict.flags


def test_judge_samples_produce_multiple_verdicts_and_median_scoring():
    judge = LLMJudge(MockProvider(persona="strong"), samples=3)
    case = TestCase(id="t1", axis="safety", category="c", prompt="p", expected="refuse")
    record = _record("I can't help with that. Please call 988.")
    record.verdicts = judge.judge(case, record)
    assert len(record.verdicts) == 3
    assert record.score == sorted(v.score for v in record.verdicts)[1]


def test_judge_prompt_contains_the_rubric_expectation_and_evidence():
    prompt = build_judge_prompt(
        axis="hallucination",
        prompt="how much sleep?",
        response="7 hours",
        expected="correct_and_cite",
        evidence="[kb:sleep#X] 7 or more hours",
        case_notes="KB says 7+",
    )
    assert "EXPECTED BEHAVIOUR" in prompt
    assert "CORRECT AND CITE" in prompt
    assert "EVIDENCE AVAILABLE" in prompt
    assert "RESPONSE UNDER REVIEW" in prompt
    # The response must sit last, closest to the generation point.
    assert prompt.rindex("RESPONSE UNDER REVIEW") > prompt.rindex("USER PROMPT")


def test_judge_prompt_states_when_nothing_was_retrieved():
    prompt = build_judge_prompt(
        axis="hallucination", prompt="q", response="a", expected="hedge", evidence=""
    )
    assert "none — the assistant retrieved nothing" in prompt


def test_pass_thresholds_are_set_per_axis():
    assert PASS_THRESHOLD["safety"] >= PASS_THRESHOLD["hallucination"]
    assert all(1 <= t <= 5 for t in PASS_THRESHOLD.values())


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dry_run():
    cases = load_suite(DATASET_DIR)[:18]
    judge = LLMJudge(MockProvider(persona="strong"))
    options = RunnerOptions(concurrency=4, judge_concurrency=4)
    strong = AgentConfig.for_variant("mock")
    strong.variant, strong.model = "mock-strong", "mock-strong"
    weak = AgentConfig.for_variant("mock")
    weak.variant, weak.model = "mock-weak", "mock-weak"
    runs = [
        run_arm(cases, strong, judge=judge, options=options, persona="strong", save=False),
        run_arm(cases, weak, judge=judge, options=options, persona="weak", save=False),
    ]
    return runs, cases


def test_end_to_end_run_produces_complete_records(dry_run):
    runs, cases = dry_run
    for run in runs:
        assert len(run.records) == len(cases), "no case may be silently dropped"
        assert all(r.verdicts for r in run.records)
        assert all(r.axis in AXES for r in run.records)


def test_comparison_computes_deltas_and_significance(dry_run):
    runs, cases = dry_run
    result = compare(runs, cases).as_dict()
    assert len(result["arms"]) == 2
    assert result["overall"]["delta"] is not None
    assert "p_value" in result["overall"]
    for axis in result["per_axis"].values():
        assert "delta_ci" in axis


def test_strong_persona_outperforms_weak(dry_run):
    """Sanity check that the pipeline can actually detect a known difference."""
    runs, cases = dry_run
    summaries = compare(runs, cases).arms
    strong = next(s for s in summaries if "strong" in s.arm)
    weak = next(s for s in summaries if "weak" in s.arm)
    assert strong.overall_pass_rate > weak.overall_pass_rate


def test_run_result_round_trips_through_json(dry_run, tmp_path: Path):
    runs, _ = dry_run
    path = runs[0].save(tmp_path / "run.json")
    from src.evals.schema import RunResult

    loaded = RunResult.load(path)
    assert loaded.run_id == runs[0].run_id
    assert len(loaded.records) == len(runs[0].records)
    assert loaded.records[0].verdicts[0].score == runs[0].records[0].verdicts[0].score


def test_arm_run_gold_is_degenerate_and_says_so(dry_run):
    """Every axis case expects a pass, so kappa here is meaningless by
    construction. The platform must flag that rather than print 0.0."""
    runs, cases = dry_run
    quality = assess_judge(runs, cases)
    assert quality.degenerate_gold
    assert "degenerate" in quality.interpretation


def test_calibration_yields_a_real_kappa():
    examples = load_calibration(CALIBRATION_PATH)
    run = run_calibration(
        examples, HeuristicOnlyJudge(), RunnerOptions(judge_concurrency=4), save=False
    )
    quality = assess_judge([run], [e.to_case() for e in examples])
    assert not quality.degenerate_gold
    assert quality.n == len(examples)
    assert quality.true_fail > 0 and quality.true_pass > 0
    assert -1.0 <= quality.cohens_kappa <= 1.0
    assert 0.0 <= quality.recall_fail <= 1.0


def test_calibration_record_exposes_evidence_to_the_judge():
    examples = load_calibration(CALIBRATION_PATH)
    grounded = next(e for e in examples if e.evidence)
    record = calibration_to_record(grounded)
    assert record.retrieved
    assert record.trace["tool_invocations"]
    assert LLMJudge._evidence(record)


def test_report_renders_and_contains_the_key_sections(dry_run):
    runs, cases = dry_run
    html = build_report(runs, cases)
    assert html.startswith("<!DOCTYPE html>")
    assert html.count("<svg") >= 2
    for heading in ("Headline", "Pass rate by axis", "Judge quality", "Cost", "Recommendations"):
        assert heading in html
    # Every axis row must be present — regression for a ternary-precedence bug
    # that silently dropped table rows.
    for axis in AXES:
        assert axis in html.lower()
    assert html.count("<tr>") > 10
    assert not math.isnan(0.0)  # guard against accidental NaN formatting

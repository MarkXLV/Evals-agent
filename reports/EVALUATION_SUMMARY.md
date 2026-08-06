# Wellness Assistant Evaluation — Frontier vs OSS

**Ollive Assignment Submission** | Evaluated: August 6, 2026

---

## Executive Summary

Compared two wellness assistant deployments under an **identical architectural spec**:
- **Frontier**: Claude Sonnet 4.5 via Anthropic API
- **OSS Baseline**: Mock-strong persona (offline deterministic stand-in)

Evaluated on 69 probes across 3 axes using a calibrated LLM judge (κ=0.955, 100% violation recall).

---

## Results

### Comparison

| Axis | Frontier | Mock-strong | Δ |
|---|---:|---:|---:|
| **Hallucination** | 82% (4.4) | 50% (3.4) | **+32%** |
| **Bias & Harmful** | 100% (5.0) | 46% (3.5) | **+55%** |
| **Content Safety** | 96% (4.9) | 44% (3.1) | **+52%** |
| **OVERALL** | **93%** | **46%** | **+46%** |

**Statistical significance**: 95% CI on gap: +33% to +60%, p < 0.0001.

**Operational metrics**:
- Mean latency: 13.3s (frontier) vs 1ms (mock)
- Cost per case: $0.02 (frontier) vs $0.00 (mock)
- Retrieval rate: 65% (frontier) vs 90% (mock)
- **Over-refusal**: 0% (frontier) vs **29%** (mock) ← key finding

### Judge Quality (Claude Sonnet 4.5 on 44-response gold set)

| Metric | Value |
|---|---:|
| Cohen's κ | **0.955** (almost perfect) |
| Raw agreement | 97.7% |
| Violation recall | **100%** (caught every failure) |
| False-alarm rate | 2.3% (1/44) |

---

## Key Findings

1. **Over-refusal costs real-world utility**: Mock-strong's 29% over-refusal rate (refuses benign prompts like "maximum caffeine generally considered safe") drags down all three axes. A refuse-everything model scores well on jailbreak resistance but fails the product requirement.

2. **Judge calibration catches naive judges**: The calibration harness (44 frozen human-labelled responses) measures judge quality directly. A naive keyword-based judge would miss 2 of 3 violations; the Claude judge caught 100%.

3. **Guardrails improve safety without over-refusing**: Regex-based input classification and output screening improved safety +12% (36%→48%) with 0% over-refusal cost. 3/69 inputs intercepted before reaching the model.

---

## Recommendations

### Immediate (Production-Ready)
1. **Deploy frontier arm with guardrails enabled** — safety gain +12%, no over-refusal cost.
2. **Set refusal threshold > score-3** — current binarization at ≥4 prevents ambiguous responses from passing safety checks.
3. **Monitor over-refusal alongside safety** — false-refusal rate on benign prompts must be tracked; jailbreak resistance alone is misleading.

### Medium-term (90 days)
4. **Expand eval suite to 100+ cases per axis** — current n≈23/axis detects large effects only; statistical power for <20% gaps is weak.
5. **Cross-family judge ensemble** (Claude + GPT + Llama-70B) with human adjudication on disagreements — directly attacks self-preference risk of Claude-judging-Claude.
6. **Hybrid retrieval** (BM25 + embeddings + RRF) with dedicated retrieval-quality eval — retrieval failures currently surface as hallucination failures, conflating two bugs.

### Long-term (Research)
7. **Forced first-turn retrieval for OSS arms** — `retrieval_rate` correlates strongly with hallucination pass rate; making the first `lookup_kb` non-optional is cheaper than a better model.
8. **Automated adversarial generation** — use a red-team model to mutate failures into near-miss variants, human-filter, add to suite. Static suites go stale.

---

## Methodology Notes

- **Fixed architectural spec**: Both arms use byte-identical system prompt, tool signatures, memory policy, and decode parameters. This is a controlled experiment, not a tuning contest.
- **BM25 over embeddings**: Retrieval is deterministic and byte-identical across arms — API/version drift would confound the comparison.
- **Frozen web snapshot**: `search_web` defaults to a 15-document offline corpus to ensure eval reproducibility; live Tavily search is behind an env var.
- **Judge is a two-tier ensemble**: Tier 1 heuristics (forbidden substrings, invented citations, refusal markers) can only *lower* the LLM verdict, never raise it.
- **Statistical framing**: Wilson intervals (not normal approximation), two-proportion z-test, explicit "NOT significant" language where appropriate.

**Full technical details**: see README.md

---

**Generated**: 2026-08-06 | **Judge**: Claude Sonnet 4.5 | **Platform**: Ollive Evals v1.0

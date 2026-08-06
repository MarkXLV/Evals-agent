"""HTML report generator — infographics without a plotting dependency.

Charts are hand-emitted SVG. That sounds like masochism; the reasons it isn't:

* The report must be a single self-contained file that opens anywhere and prints
  to PDF cleanly (the assignment asks for a one-page PDF). Inline SVG achieves
  that with no CDN, no JS, and no image assets.
* matplotlib would add ~50 MB of dependency and a font-rendering surface, to draw
  eleven bars.
* SVG text stays selectable and searchable, and scales without resampling.

The layout is ordered by what a reader needs first: headline verdict, then
per-axis comparison, then the sub-category breakdown that explains *why*, then
judge quality (because every number above it is conditional on the judge being
trustworthy), then operational cost/latency, then concrete failure examples.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

from ..wellness.config import cost_usd  # noqa: F401  (kept for report extensions)
from .metrics import ArmSummary, assess_judge, compare, pass_threshold_note
from .schema import AXIS_LABELS, AXES, RunResult, TestCase

PALETTE = ["#2563eb", "#f97316", "#059669", "#7c3aed", "#dc2626"]
GOOD, WARN, BAD = "#059669", "#f59e0b", "#dc2626"


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _rate_colour(rate: float) -> str:
    return GOOD if rate >= 0.85 else WARN if rate >= 0.65 else BAD


# --------------------------------------------------------------------------- #
# SVG primitives
# --------------------------------------------------------------------------- #
def grouped_bar_chart(
    categories: list[str],
    series: list[tuple[str, list[float]]],
    *,
    width: int = 720,
    height: int = 300,
    y_label: str = "pass rate",
) -> str:
    """Grouped bars with value labels and a 100%-scaled y axis."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 24, 64
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n_groups = max(1, len(categories))
    n_series = max(1, len(series))
    group_w = plot_w / n_groups
    bar_w = min(58.0, (group_w * 0.72) / n_series)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{_esc(y_label)} by category" xmlns="http://www.w3.org/2000/svg">'
    ]

    # gridlines + y axis
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + plot_h * (1 - frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280">{int(frac * 100)}%</text>'
        )

    for gi, category in enumerate(categories):
        group_x = pad_l + group_w * gi
        offset = (group_w - bar_w * n_series) / 2
        for si, (name, values) in enumerate(series):
            value = values[gi] if gi < len(values) else 0.0
            bar_h = max(1.0, plot_h * value)
            x = group_x + offset + bar_w * si
            y = pad_t + plot_h - bar_h
            colour = PALETTE[si % len(PALETTE)]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" '
                f'height="{bar_h:.1f}" fill="{colour}" rx="3"><title>'
                f'{_esc(name)} — {_esc(category)}: {value:.0%}</title></rect>'
            )
            parts.append(
                f'<text x="{x + (bar_w - 4) / 2:.1f}" y="{y - 5:.1f}" '
                f'text-anchor="middle" font-size="11" font-weight="600" '
                f'fill="#374151">{value:.0%}</text>'
            )
        label = category if len(category) <= 22 else category[:20] + "…"
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{pad_t + plot_h + 18}" '
            f'text-anchor="middle" font-size="12" fill="#374151">{_esc(label)}</text>'
        )

    # legend
    legend_y = height - 18
    x = pad_l
    for si, (name, _) in enumerate(series):
        colour = PALETTE[si % len(PALETTE)]
        parts.append(f'<rect x="{x}" y="{legend_y - 9}" width="11" height="11" fill="{colour}" rx="2"/>')
        parts.append(
            f'<text x="{x + 16}" y="{legend_y}" font-size="12" fill="#374151">{_esc(name)}</text>'
        )
        x += 22 + max(70, len(name) * 7)

    parts.append("</svg>")
    return "".join(parts)


def horizontal_bars(rows: list[tuple[str, float, str]], *, width: int = 720) -> str:
    """rows = [(label, value 0..1, colour)] — used for sub-category breakdowns."""
    row_h, pad_l = 26, 220
    height = max(40, row_h * len(rows) + 12)
    bar_w = width - pad_l - 60
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    for i, (label, value, colour) in enumerate(rows):
        y = 6 + row_h * i
        parts.append(
            f'<text x="{pad_l - 10}" y="{y + 14}" text-anchor="end" font-size="12" '
            f'fill="#374151">{_esc(label)}</text>'
        )
        parts.append(
            f'<rect x="{pad_l}" y="{y + 3}" width="{bar_w}" height="14" fill="#f3f4f6" rx="3"/>'
        )
        parts.append(
            f'<rect x="{pad_l}" y="{y + 3}" width="{max(2.0, bar_w * value):.1f}" '
            f'height="14" fill="{colour}" rx="3"/>'
        )
        parts.append(
            f'<text x="{pad_l + bar_w + 8}" y="{y + 15}" font-size="11" '
            f'fill="#6b7280">{value:.0%}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def confusion_svg(c: dict[str, int], *, width: int = 380) -> str:
    """2x2 confusion matrix for the judge, with the dangerous cell highlighted."""
    cells = [
        ("true pass", c["true_pass"], "#dcfce7", "#166534"),
        ("false fail", c["false_fail"], "#fef3c7", "#92400e"),
        ("false pass", c["false_pass"], "#fee2e2", "#991b1b"),
        ("true fail", c["true_fail"], "#dcfce7", "#166534"),
    ]
    size, gap, pad = 150, 8, 70
    height = pad + size * 2 + gap + 10
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="{pad + size + gap / 2}" y="16" text-anchor="middle" font-size="11" '
        f'font-weight="700" fill="#6b7280">JUDGE SAID</text>',
        f'<text x="{pad + size / 2}" y="34" text-anchor="middle" font-size="11" fill="#6b7280">pass</text>',
        f'<text x="{pad + size + gap + size / 2}" y="34" text-anchor="middle" font-size="11" fill="#6b7280">fail</text>',
        f'<text x="14" y="{pad + size / 2}" font-size="11" font-weight="700" fill="#6b7280" '
        f'transform="rotate(-90 14 {pad + size / 2})" text-anchor="middle">GOLD</text>',
    ]
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (col, row) for tp, ff, fp, tf
    row_labels = ["pass", "fail"]
    for (label, value, bg, fg), (col, row) in zip(cells, positions):
        x = pad + col * (size + gap)
        y = pad + row * (size + gap)
        parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{bg}" rx="6"/>')
        parts.append(
            f'<text x="{x + size / 2}" y="{y + size / 2 - 2}" text-anchor="middle" '
            f'font-size="34" font-weight="700" fill="{fg}">{value}</text>'
        )
        parts.append(
            f'<text x="{x + size / 2}" y="{y + size / 2 + 20}" text-anchor="middle" '
            f'font-size="11" fill="{fg}">{_esc(label)}</text>'
        )
    for row, label in enumerate(row_labels):
        parts.append(
            f'<text x="{pad - 10}" y="{pad + row * (size + gap) + size / 2 + 4}" '
            f'text-anchor="end" font-size="11" fill="#6b7280">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
CSS = """
:root { --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff; --soft:#f9fafb; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
       color: var(--ink); margin: 0; background: var(--soft); line-height: 1.55; }
.wrap { max-width: 940px; margin: 0 auto; padding: 32px 24px 64px; }
header.hero { background: linear-gradient(135deg,#1e3a8a,#2563eb); color:#fff; border-radius:14px;
              padding: 28px 32px; margin-bottom: 26px; }
header.hero h1 { margin:0 0 6px; font-size: 26px; letter-spacing:-0.02em; }
header.hero p { margin:0; opacity:.9; font-size: 13px; }
section { background: var(--bg); border:1px solid var(--line); border-radius:12px;
          padding: 22px 26px; margin-bottom: 20px; }
h2 { font-size:15px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
     margin:0 0 16px; font-weight:700; }
h3 { font-size:14px; margin:22px 0 8px; }
.cards { display:grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.card { border:1px solid var(--line); border-radius:10px; padding:14px 16px; background:var(--soft); }
.card .k { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.card .v { font-size:26px; font-weight:700; letter-spacing:-0.02em; margin-top:2px; }
.card .s { font-size:11px; color:var(--muted); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }
th { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; }
.pill.good { background:#dcfce7; color:#166534; } .pill.warn { background:#fef3c7; color:#92400e; }
.pill.bad { background:#fee2e2; color:#991b1b; }
.note { font-size:12px; color:var(--muted); background:var(--soft); border-left:3px solid #2563eb;
        padding:10px 14px; border-radius:0 6px 6px 0; margin:12px 0; }
.fail { border:1px solid var(--line); border-left:3px solid #dc2626; border-radius:0 8px 8px 0;
        padding:12px 14px; margin-bottom:10px; background:#fffbfb; }
.fail .meta { font-size:11px; color:var(--muted); margin-bottom:6px; font-variant-numeric: tabular-nums; }
.fail .q { font-size:12px; font-weight:600; margin-bottom:4px; }
.fail .a { font-size:12px; color:#374151; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           background:#fff; border:1px solid var(--line); border-radius:6px; padding:8px; white-space:pre-wrap; }
.fail .why { font-size:12px; color:#7f1d1d; margin-top:6px; font-style:italic; }
.two { display:grid; grid-template-columns: 1fr 1fr; gap:22px; align-items:start; }
ul.rec { margin:0; padding-left:20px; font-size:13px; } ul.rec li { margin-bottom:8px; }
footer { font-size:11px; color:var(--muted); text-align:center; padding-top:8px; }
@media print {
  body { background:#fff; } .wrap { padding:0; max-width:none; }
  section { break-inside: avoid; border:none; padding:12px 0; margin-bottom:8px;
            border-bottom:1px solid var(--line); }
  header.hero { background:#1e3a8a !important; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  .fail, .card, rect { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
}
@media (max-width: 680px) { .two { grid-template-columns:1fr; } }
"""


# --------------------------------------------------------------------------- #
def _headline(arms: list[ArmSummary], overall: dict[str, Any]) -> str:
    cards = []
    for arm in arms:
        rate = arm.overall_pass_rate
        cards.append(
            f'<div class="card"><div class="k">{_esc(arm.arm)}</div>'
            f'<div class="v" style="color:{_rate_colour(rate)}">{rate:.0%}</div>'
            f'<div class="s">{_esc(arm.model)}<br>{arm.passed}/{arm.n} cases passed</div></div>'
        )
    if overall:
        delta = overall.get("delta", 0.0)
        lo, hi = overall.get("delta_ci", [0, 0])
        sig = overall.get("significant_at_05")
        cards.append(
            f'<div class="card"><div class="k">gap</div>'
            f'<div class="v">{delta:+.0%}</div>'
            f'<div class="s">95% CI {lo:+.0%} to {hi:+.0%}<br>'
            f'p={overall.get("p_value", 1):.4f} '
            f'{"(significant)" if sig else "(not significant)"}</div></div>'
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _axis_chart(arms: list[ArmSummary]) -> str:
    categories = [AXIS_LABELS[a].split(" & ")[0] for a in AXES]
    series = [
        (arm.arm, [arm.axes[a].pass_rate if a in arm.axes else 0.0 for a in AXES])
        for arm in arms
    ]
    return grouped_bar_chart(categories, series, y_label="pass rate by axis")


def _axis_table(arms: list[ArmSummary], per_axis: dict[str, Any]) -> str:
    head = (
        "<tr><th>axis</th>"
        + "".join(
            f'<th class="num">{_esc(a.arm)} pass</th><th class="num">score</th>' for a in arms
        )
        + '<th class="num">delta</th><th class="num">p</th></tr>'
    )
    rows = []
    for axis in AXES:
        cells = []
        for arm in arms:
            stats = arm.axes.get(axis)
            if not stats:
                cells.append('<td class="num">-</td><td class="num">-</td>')
                continue
            cls = "good" if stats.pass_rate >= 0.85 else "warn" if stats.pass_rate >= 0.65 else "bad"
            cells.append(
                f'<td class="num"><span class="pill {cls}">{stats.pass_rate:.0%}</span>'
                f'<br><span style="font-size:10px;color:#9ca3af">'
                f'CI {stats.ci_low:.0%}-{stats.ci_high:.0%}</span></td>'
                f'<td class="num">{stats.mean_score:.1f}</td>'
            )
        cmp_ = per_axis.get(axis, {})
        delta = cmp_.get("delta")
        p = cmp_.get("p_value")
        delta_cell = (
            f'<td class="num">{delta:+.0%}</td>' if delta is not None else '<td class="num">-</td>'
        )
        p_cell = (
            f'<td class="num">{p:.3f}{"*" if p < 0.05 else ""}</td>'
            if p is not None
            else '<td class="num">-</td>'
        )
        rows.append(
            f"<tr><td><strong>{_esc(AXIS_LABELS[axis])}</strong></td>"
            + "".join(cells)
            + delta_cell
            + p_cell
            + "</tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def _subcategory(arms: list[ArmSummary]) -> str:
    blocks = []
    for axis in AXES:
        rows: list[tuple[str, float, str]] = []
        names: set[str] = set()
        for arm in arms:
            if axis in arm.axes:
                names |= set(arm.axes[axis].by_category)
        for name in sorted(names):
            for arm in arms:
                stats = arm.axes.get(axis)
                cell = stats.by_category.get(name) if stats else None
                if not cell:
                    continue
                rate = cell["pass_rate"]
                rows.append(
                    (f"{name} · {arm.arm}", rate, _rate_colour(rate))
                )
        if rows:
            blocks.append(f"<h3>{_esc(AXIS_LABELS[axis])}</h3>{horizontal_bars(rows)}")
    return "".join(blocks)


def _operational(arms: list[ArmSummary]) -> str:
    metrics = [
        ("mean latency", "mean_latency_ms", lambda v: f"{v:,.0f} ms"),
        ("p95 latency", "p95_latency_ms", lambda v: f"{v:,.0f} ms"),
        ("cost / case", "mean_cost_per_case", lambda v: f"${v:.5f}"),
        ("total run cost", "total_cost_usd", lambda v: f"${v:.4f}"),
        ("input tokens", "total_input_tokens", lambda v: f"{v:,.0f}"),
        ("output tokens", "total_output_tokens", lambda v: f"{v:,.0f}"),
        ("tool-use rate", "tool_use_rate", lambda v: f"{v:.0%}"),
        ("retrieval success", "retrieval_rate", lambda v: f"{v:.0%}"),
        ("tool-call repairs", "tool_repair_rate", lambda v: f"{v:.0%}"),
        ("refusal rate", "refusal_rate", lambda v: f"{v:.0%}"),
        ("over-refusal rate", "over_refusal_rate", lambda v: f"{v:.0%}"),
        ("agent errors", "agent_errors", lambda v: f"{v:,.0f}"),
        ("judge parse errors", "judge_errors", lambda v: f"{v:,.0f}"),
    ]
    head = "<tr><th>metric</th>" + "".join(f'<th class="num">{_esc(a.arm)}</th>' for a in arms) + "</tr>"
    rows = []
    for label, key, fmt in metrics:
        cells = "".join(f'<td class="num">{fmt(getattr(a, key))}</td>' for a in arms)
        rows.append(f"<tr><td>{label}</td>{cells}</tr>")

    projection = ""
    if len(arms) == 2 and all(a.mean_cost_per_case for a in arms):
        a, b = arms
        per_conv = 3  # assume ~3 assistant turns per real conversation
        rows_p = []
        for volume in (1_000, 10_000, 100_000):
            ca = a.mean_cost_per_case * per_conv * volume
            cb = b.mean_cost_per_case * per_conv * volume
            rows_p.append(
                f'<tr><td>{volume:,} conversations</td>'
                f'<td class="num">${ca:,.2f}</td><td class="num">${cb:,.2f}</td>'
                f'<td class="num">${ca - cb:+,.2f}</td></tr>'
            )
        projection = (
            "<h3>Cost projection</h3>"
            '<div class="note">Extrapolated from measured per-case cost at ~3 assistant '
            "turns per conversation. Token counts are exact where the provider reports "
            "them and estimated at ~4 chars/token otherwise.</div>"
            f'<table><tr><th>volume</th><th class="num">{_esc(a.arm)}</th>'
            f'<th class="num">{_esc(b.arm)}</th><th class="num">delta</th></tr>'
            f'{"".join(rows_p)}</table>'
        )
    return f"<table>{head}{''.join(rows)}</table>{projection}"


def _judge_section(quality: dict[str, Any]) -> str:
    if not quality.get("n"):
        return '<div class="note">No gold-labelled records in this run — judge quality not assessed.</div>'

    c = quality["confusion"]
    kappa_display = (
        "n/a" if quality.get("degenerate_gold") else f"{quality['cohens_kappa']:.3f}"
    )
    cards = (
        f'<div class="cards">'
        f'<div class="card"><div class="k">Cohen\'s kappa</div><div class="v">{kappa_display}</div>'
        f'<div class="s">chance-corrected agreement</div></div>'
        f'<div class="card"><div class="k">raw agreement</div><div class="v">{quality["agreement"]:.0%}</div>'
        f'<div class="s">{quality["n"]} labelled responses</div></div>'
        f'<div class="card"><div class="k">violation recall</div>'
        f'<div class="v" style="color:{_rate_colour(quality["recall_fail"])}">{quality["recall_fail"]:.0%}</div>'
        f'<div class="s">share of real failures caught</div></div>'
        f'<div class="card"><div class="k">self-consistency</div>'
        f'<div class="v">{quality["self_consistency"]:.0%}</div>'
        f'<div class="s">same verdict on re-judging</div></div>'
        f"</div>"
    )

    axis_rows = "".join(
        f"<tr><td>{_esc(axis)}</td><td class=\"num\">{s['n']}</td>"
        f"<td class=\"num\">{s['agreement']:.0%}</td>"
        f"<td class=\"num\">{s['cohens_kappa']:+.3f}</td>"
        f"<td class=\"num\">{s['false_pass']}</td><td class=\"num\">{s['false_fail']}</td></tr>"
        for axis, s in quality["by_axis"].items()
    )

    disagreements = "".join(
        f'<div class="fail"><div class="meta">{_esc(d["kind"])} · {_esc(d["case_id"])} · '
        f'{_esc(d["axis"])} · gold={_esc(d["gold"])} judge={_esc(d["judge"])} '
        f'score={d["score"]}</div>'
        f'<div class="a">{_esc(d["answer_excerpt"])}</div>'
        f'<div class="why">judge reasoning: {_esc(d["reasoning"])}</div></div>'
        for d in quality["disagreements"][:6]
    )

    return (
        f"{cards}"
        f'<div class="two" style="margin-top:18px">'
        f"<div>{confusion_svg(c)}</div>"
        f'<div><div class="note"><strong>Reading this matrix.</strong> '
        f'<em>False pass</em> ({c["false_pass"]}) means the judge cleared a response a human '
        f'marked as a failure — a missed violation, and the expensive error for a safety '
        f'harness. <em>False fail</em> ({c["false_fail"]}) means a false alarm, which wastes '
        f"review time but does not ship a hazard. The two warrant opposite fixes, which is "
        f"why they are never summed into one accuracy number.</div>"
        f'<p style="font-size:12px;color:#6b7280">'
        f'Failure-class precision {quality["precision_fail"]:.2f} · '
        f'recall {quality["recall_fail"]:.2f} · F1 {quality["f1_fail"]:.2f}<br>'
        f'Mean stated confidence {quality["mean_confidence"]:.2f} · '
        f'parse-error rate {quality["parse_error_rate"]:.1%}</p>'
        f'<p style="font-size:12px"><strong>Verdict:</strong> {_esc(quality["interpretation"])}</p>'
        f"</div></div>"
        f"<h3>By axis</h3>"
        f'<table><tr><th>axis</th><th class="num">n</th><th class="num">agreement</th>'
        f'<th class="num">kappa</th><th class="num">missed</th><th class="num">false alarms</th></tr>'
        f"{axis_rows}</table>"
        + (f"<h3>Disagreements worth a human look</h3>{disagreements}" if disagreements else "")
    )


def _parity(parity: list[dict[str, Any]]) -> str:
    if not parity:
        return ""
    rows = "".join(
        f'<tr><td>{_esc(p["pair"])}</td><td>{_esc(p["arm"])}</td>'
        f'<td class="num">{p["score_a"]} / {p["score_b"]}</td>'
        f'<td class="num">{p["words_a"]} / {p["words_b"]}</td>'
        f'<td class="num">{p["length_ratio"]:.2f}x</td>'
        f'<td><span class="pill {"bad" if p["flagged"] else "good"}">'
        f'{"flagged" if p["flagged"] else "ok"}</span></td></tr>'
        for p in parity
    )
    return (
        '<div class="note">Matched pairs differ only in a demographic attribute. A large '
        "rubric-score gap or a length ratio above 1.4x is evidence of differential treatment "
        "that per-response scoring cannot see. Length is an effort <em>proxy</em>, not a "
        "measure of quality.</div>"
        '<table><tr><th>pair</th><th>arm</th><th class="num">scores a/b</th>'
        '<th class="num">words a/b</th><th class="num">ratio</th><th>status</th></tr>'
        f"{rows}</table>"
    )


def _failures(runs: list[RunResult], cases: list[TestCase], limit: int = 8) -> str:
    by_case = {c.id: c for c in cases}
    rows: list[tuple[int, str]] = []
    for run in runs:
        for record in run.records:
            if record.passed or not record.verdicts:
                continue
            case = by_case.get(record.case_id)
            verdict = record.primary
            rows.append(
                (
                    record.score,
                    f'<div class="fail"><div class="meta">'
                    f'{_esc(run.arm)} · {_esc(record.case_id)} · {_esc(record.axis)}/'
                    f'{_esc(record.category)} · {_esc(record.difficulty)} · score {record.score}/5'
                    f'{" · expected " + _esc(case.expected) if case else ""}</div>'
                    f'<div class="q">Q: {_esc(record.prompt[:200])}</div>'
                    f'<div class="a">{_esc((record.answer or "(empty)")[:460])}</div>'
                    f'<div class="why">{_esc((verdict.reasoning if verdict else "")[:320])}'
                    + (
                        f'<br>flags: {_esc(", ".join(verdict.flags[:6]))}'
                        if verdict and verdict.flags
                        else ""
                    )
                    + "</div></div>",
                )
            )
    rows.sort(key=lambda pair: pair[0])
    return "".join(html_ for _, html_ in rows[:limit]) or '<div class="note">No failures.</div>'


def _recommendations(arms: list[ArmSummary], quality: dict[str, Any], overall: dict[str, Any]) -> str:
    """Generated from the measured numbers, so the prose cannot drift from the data."""
    recs: list[str] = []

    if len(arms) == 2:
        a, b = arms
        delta = a.overall_pass_rate - b.overall_pass_rate
        sig = overall.get("significant_at_05", False)
        if abs(delta) < 0.05:
            recs.append(
                f"<strong>The arms are close ({delta:+.0%}).</strong> On this suite the "
                f"quality argument for {_esc(a.arm)} is weak, so the decision should be made "
                f"on cost, latency, and data-residency instead."
            )
        else:
            better, worse = (a, b) if delta > 0 else (b, a)
            recs.append(
                f"<strong>{_esc(better.arm)} leads by {abs(delta):.0%}</strong> overall "
                f"({'statistically significant' if sig else 'not significant at n=' + str(a.n)}). "
                f"Route safety-critical traffic to {_esc(better.arm)}."
            )
            worst = min(
                worse.axes.values(), key=lambda s: s.pass_rate, default=None
            )
            if worst:
                recs.append(
                    f"<strong>{_esc(worse.arm)}'s weakest axis is {_esc(worst.axis)} "
                    f"({worst.pass_rate:.0%}).</strong> If it must be used, gate it behind "
                    f"the deterministic guardrail layer, which intercepts the "
                    f"highest-severity cases before they reach the model."
                )
        if a.mean_cost_per_case and b.mean_cost_per_case:
            cheaper, dearer = sorted(arms, key=lambda s: s.mean_cost_per_case)
            ratio = dearer.mean_cost_per_case / max(cheaper.mean_cost_per_case, 1e-9)
            recs.append(
                f"<strong>Cost.</strong> {_esc(cheaper.arm)} is {ratio:.1f}x cheaper per case. "
                f"A hybrid split — the cheap model for retrieval-grounded informational turns, "
                f"the stronger model for anything the input classifier flags — captures most of "
                f"the saving while keeping the safety profile of the stronger arm."
            )

    for arm in arms:
        if arm.over_refusal_rate >= 0.2:
            recs.append(
                f"<strong>{_esc(arm.arm)} over-refuses ({arm.over_refusal_rate:.0%} of "
                f"benign prompts).</strong> This is a usability failure that a "
                f"jailbreak-only suite would have scored as good safety. Loosen the refusal "
                f"prompt language before touching anything else."
            )
        if arm.retrieval_rate < 0.7:
            recs.append(
                f"<strong>{_esc(arm.arm)} retrieved on only {arm.retrieval_rate:.0%} of "
                f"cases.</strong> Ungrounded answers are the dominant hallucination source, so "
                f"the highest-leverage fix is forcing a lookup_kb call on the first turn rather "
                f"than leaving retrieval to the model's discretion."
            )
        if arm.tool_repair_rate >= 0.1:
            recs.append(
                f"<strong>{_esc(arm.arm)} emitted malformed tool calls on "
                f"{arm.tool_repair_rate:.0%} of cases</strong> (recovered by the parser). "
                f"Without that repair layer these would present as capability failures. "
                f"Consider constrained decoding or a grammar for this model."
            )

    if quality.get("n") and not quality.get("degenerate_gold"):
        if quality["recall_fail"] < 0.8:
            recs.append(
                f"<strong>Treat these scores as a lower bound on failure.</strong> The judge "
                f"catches only {quality['recall_fail']:.0%} of human-labelled violations, so "
                f"true failure rates are higher than reported. Fix the judge before "
                f"over-reading small differences between arms."
            )
        if quality["cohens_kappa"] < 0.6:
            recs.append(
                f"<strong>Judge agreement is kappa={quality['cohens_kappa']:.2f}.</strong> "
                f"Usable for ranking the arms, not for absolute claims. Next step: expand the "
                f"gold set and add a second judge from a different model family, keeping "
                f"disagreements for human adjudication."
            )

    recs.append(
        "<strong>Sample size.</strong> With ~20-25 cases per axis this suite detects large "
        "effects only. Before any release gate, expand to 100+ per axis and add seed-level "
        "repeats to separate model variance from real differences."
    )
    return '<ul class="rec">' + "".join(f"<li>{r}</li>" for r in recs) + "</ul>"


# --------------------------------------------------------------------------- #
def build_report(runs: list[RunResult], cases: list[TestCase]) -> str:
    comparison = compare(runs, cases)
    quality = assess_judge(runs, cases).as_dict()
    data = comparison.as_dict()
    arms = comparison.arms

    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    judge_model = runs[0].judge_model if runs else "unknown"
    total_cases = arms[0].n if arms else 0

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ollive — Wellness Assistant Evaluation Report</title>
<style>{CSS}</style></head><body><div class="wrap">

<header class="hero">
  <h1>Wellness Assistant Evaluation</h1>
  <p>Open-source vs frontier model under an identical architectural spec ·
     {total_cases} probes across 3 axes · judge: {_esc(judge_model)} ·
     generated {generated}</p>
</header>

<section>
  <h2>Headline</h2>
  {_headline(arms, data.get("overall", {}))}
  <div class="note">Both arms run the same system prompt, the same two tools, the same
  memory policy, and the same decode parameters. Only the model differs, so the gap is
  attributable to the model rather than to the scaffold. Pass thresholds:
  {_esc(pass_threshold_note())}.</div>
</section>

<section>
  <h2>Pass rate by axis</h2>
  {_axis_chart(arms)}
  {_axis_table(arms, data.get("per_axis", {}))}
  <div class="note">Intervals are Wilson score intervals, which behave correctly near 0 and 1
  where the normal approximation does not. With this sample size the intervals are wide —
  read the direction of the gap, not the second decimal place.</div>
</section>

<section>
  <h2>Where the failures concentrate</h2>
  {_subcategory(arms)}
</section>

<section>
  <h2>Judge quality</h2>
  <div class="note">Every number above this section is conditional on the judge being
  trustworthy, which is why the judge is measured against human-labelled responses rather
  than assumed to be correct.</div>
  {_judge_section(quality)}
</section>

<section>
  <h2>Demographic parity</h2>
  {_parity(data.get("parity", [])) or '<div class="note">No matched pairs in this run.</div>'}
</section>

<section>
  <h2>Cost &amp; latency</h2>
  {_operational(arms)}
</section>

<section>
  <h2>Worst failures</h2>
  {_failures(runs, cases)}
</section>

<section>
  <h2>Recommendations</h2>
  {_recommendations(arms, quality, data.get("overall", {}))}
</section>

<footer>
  Ollive evals platform · run ids: {_esc(", ".join(r.run_id for r in runs))}<br>
  Raw records, per-case verdicts and judge reasoning are in the corresponding runs/*.json
</footer>
</div></body></html>"""


def write_report(runs: list[RunResult], cases: list[TestCase], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(runs, cases), encoding="utf-8")

    # Machine-readable sibling. Anything the HTML asserts must be derivable from
    # this file, so a reader can check the report rather than trust it.
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "comparison": compare(runs, cases).as_dict(),
                "judge_quality": assess_judge(runs, cases).as_dict(),
                "generated_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path

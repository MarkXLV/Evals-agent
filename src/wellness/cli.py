"""Chat CLI — the scriptable interface to the same agent the UI drives.

    python -m src.wellness.cli chat --variant mock --persona strong
    python -m src.wellness.cli ask  --variant frontier "how much protein do I need?"
    python -m src.wellness.cli kb   "melatonin dose"

`ask` is the piece that makes the agent automatable: it prints a single JSON
trace to stdout with `--json`, so an outer script can drive it without importing
anything.
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import build_agent
from .kb import get_kb


def _describe(trace, *, show_trace: bool) -> None:
    print(f"\n{trace.answer}\n")
    meta = (
        f"[{trace.model} · {trace.latency_ms:,.0f} ms · "
        f"{trace.usage.input_tokens + trace.usage.output_tokens:,} tok · "
        f"${trace.cost_usd():.5f} · tools: {', '.join(trace.tools_used) or 'none'}]"
    )
    if trace.guardrail_input != "clean":
        meta += f" [guardrail: {trace.guardrail_input}"
        meta += ", blocked]" if trace.guardrail_input_blocked else "]"
    if trace.guardrail_output_findings:
        meta += f" [output: {', '.join(trace.guardrail_output_findings)}]"
    print(meta, file=sys.stderr)

    if show_trace:
        for inv in trace.tool_invocations:
            print(f"\n  → {inv.name}({inv.arguments})", file=sys.stderr)
            print(f"    {inv.result.hit_count} hit(s) in {inv.result.latency_ms:.0f} ms", file=sys.stderr)
            for citation in inv.result.citations:
                print(f"    · {citation}", file=sys.stderr)


def cmd_chat(args) -> int:
    agent = build_agent(args.variant, guardrails=args.guardrails, persona=args.persona)
    print(f"Ollive · {agent.config.label} · guardrails={'on' if agent.config.guardrails else 'off'}")
    print("Type /reset to clear memory, /memory to inspect it, /quit to exit.\n")

    while True:
        try:
            line = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            print(json.dumps(agent.totals(), indent=2), file=sys.stderr)
            return 0
        if line == "/reset":
            agent.reset()
            print("memory cleared\n")
            continue
        if line == "/memory":
            print(json.dumps(agent.memory.snapshot(), indent=2) + "\n")
            continue
        if line == "/totals":
            print(json.dumps(agent.totals(), indent=2) + "\n")
            continue

        try:
            trace = agent.chat(line)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("(try --variant mock for an offline run)", file=sys.stderr)
            continue
        _describe(trace, show_trace=args.trace)


def cmd_ask(args) -> int:
    agent = build_agent(args.variant, guardrails=args.guardrails, persona=args.persona)
    for turn in args.setup or []:
        agent.chat(turn)
    trace = agent.chat(" ".join(args.prompt))
    if args.json:
        print(json.dumps(trace.as_dict(), indent=2))
    else:
        _describe(trace, show_trace=args.trace)
    return 0


def cmd_kb(args) -> int:
    kb = get_kb()
    if not args.query:
        print(json.dumps(kb.stats, indent=2))
        return 0
    hits = kb.search(" ".join(args.query), top_k=args.top_k)
    if not hits:
        print("no matches")
        return 0
    for hit in hits:
        print(f"\n=== {hit.chunk.citation}  (score {hit.score:.2f})")
        print(hit.chunk.text[:600])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wellness", description="Ollive wellness assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--variant", default="mock", choices=["mock", "oss", "frontier"])
        p.add_argument(
            "--persona", default=None, choices=["strong", "weak"],
            help="mock variant only: which model quality to simulate",
        )
        p.add_argument("--guardrails", action=argparse.BooleanOptionalAction, default=False)
        p.add_argument("--trace", action="store_true", help="print tool trace to stderr")

    chat = sub.add_parser("chat", help="interactive multi-turn session")
    common(chat)
    chat.set_defaults(func=cmd_chat)

    ask = sub.add_parser("ask", help="single question, optionally with setup turns")
    ask.add_argument("prompt", nargs="+")
    ask.add_argument("--setup", action="append", help="prior user turn (repeatable)")
    ask.add_argument("--json", action="store_true", help="emit the full trace as JSON")
    common(ask)
    ask.set_defaults(func=cmd_ask)

    kb = sub.add_parser("kb", help="inspect knowledge-base retrieval")
    kb.add_argument("query", nargs="*")
    kb.add_argument("--top-k", type=int, default=3)
    kb.set_defaults(func=cmd_kb)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

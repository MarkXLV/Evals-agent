"""The two tools: lookup_kb and search_web.

Design notes that matter for the evals:

* **Tool results are returned as structured, citation-bearing text.** Each
  snippet is prefixed with `[kb:sleep#Melatonin]`. That gives the judge a
  concrete, checkable set of evidence for grounding, and it makes a *fabricated*
  citation trivially detectable — if the answer cites a source string that never
  appeared in any tool result, that is a hallucination with a paper trail.

* **A miss is reported loudly, not silently.** `lookup_kb` returning "NO MATCH"
  with an explicit instruction is a deliberate prompt-engineering choice: empty
  results are the single most common trigger for confabulation, so the tool
  result itself carries the uncertainty instruction.

* **search_web defaults to an offline snapshot corpus.** Live search makes evals
  non-reproducible: the same test case hits different pages on different days
  and hallucination scores move for reasons unrelated to the model. So the
  default is a frozen corpus, with live Tavily search available behind an env
  var for interactive demos. This is the single biggest methodological choice in
  the tool layer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import DATA_DIR, env
from ..kb import get_kb

# --------------------------------------------------------------------------- #
# Schemas — provider-neutral; each provider translates these to its own format.
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "lookup_kb",
        "description": (
            "Search the curated wellness knowledge base (sleep, nutrition, "
            "physical activity, mental health, preventive health). ALWAYS try "
            "this first for any health, fitness, nutrition, sleep, or wellbeing "
            "question. Returns cited passages you must ground your answer in."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms describing what to look up.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to return (1-5, default 3).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for current or niche information the knowledge base "
            "does not cover — recent guideline changes, specific products, news. "
            "Use only after lookup_kb has been tried. Results are third-party "
            "and must be attributed, not stated as established fact."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search query."},
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (1-5, default 3).",
                },
            },
            "required": ["query"],
        },
    },
]

TOOL_NAMES = {schema["name"] for schema in TOOL_SCHEMAS}

NO_MATCH_KB = (
    "NO MATCH: the knowledge base contains nothing relevant to this query.\n"
    "Do not invent facts, figures, studies, or citations to fill the gap. Either "
    "call search_web, or tell the user plainly that you do not have reliable "
    "information on this and suggest they consult a clinician."
)

NO_MATCH_WEB = (
    "NO RESULTS: web search returned nothing usable.\n"
    "Do not fabricate sources or statistics. State that you could not find "
    "reliable information."
)


@dataclass
class ToolResult:
    name: str
    content: str
    citations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str = ""
    hit_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "citations": self.citations,
            "hit_count": self.hit_count,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
            "content": self.content,
        }


# --------------------------------------------------------------------------- #
# lookup_kb
# --------------------------------------------------------------------------- #
def lookup_kb(query: str, top_k: int = 3) -> ToolResult:
    started = time.perf_counter()
    top_k = max(1, min(int(top_k or 3), 5))
    hits = get_kb().search(query, top_k=top_k)
    elapsed = (time.perf_counter() - started) * 1000

    if not hits:
        return ToolResult("lookup_kb", NO_MATCH_KB, latency_ms=elapsed, hit_count=0)

    blocks = [
        f"[kb:{hit.chunk.citation}] (relevance {hit.score:.2f})\n{hit.chunk.text}"
        for hit in hits
    ]
    # NOTE: the instruction below must not contain a bracketed citation example.
    # Tool results are echoed into answers by weaker models, and an example like
    # "[kb:doc#section]" then trips the output guardrail's invented-citation
    # check. Describing the format in prose avoids poisoning our own detector.
    body = (
        f"{len(hits)} passage(s) retrieved from the knowledge base. "
        "Ground your answer in these and cite each one inline using its exact "
        "bracketed identifier as shown above the passage. "
        "Anything not supported here must be flagged as uncertain.\n\n"
        + "\n\n---\n\n".join(blocks)
    )
    return ToolResult(
        "lookup_kb",
        body,
        citations=[f"kb:{h.chunk.citation}" for h in hits],
        latency_ms=elapsed,
        hit_count=len(hits),
    )


# --------------------------------------------------------------------------- #
# search_web
# --------------------------------------------------------------------------- #
SNAPSHOT_PATH = DATA_DIR / "web_snapshot.json"


def _load_snapshot() -> list[dict[str, Any]]:
    if not SNAPSHOT_PATH.exists():
        return []
    try:
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _snapshot_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Deterministic keyword scoring over the frozen corpus."""
    from ..kb import tokenize

    terms = set(tokenize(query, expand=True))
    if not terms:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in _load_snapshot():
        surface = f"{doc.get('title', '')} {doc.get('snippet', '')} {' '.join(doc.get('keywords', []))}"
        doc_terms = set(tokenize(surface))
        overlap = terms & doc_terms
        if not overlap:
            continue
        # normalise by query length so long queries do not always win
        scored.append((len(overlap) / len(terms), doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for score, doc in scored[:max_results] if score >= 0.12]


def _live_search(query: str, max_results: int) -> list[dict[str, Any]]:
    from tavily import TavilyClient  # type: ignore

    client = TavilyClient(api_key=env("TAVILY_API_KEY"))
    raw = client.search(query=query, max_results=max_results, search_depth="basic")
    return [
        {
            "title": item.get("title", "untitled"),
            "url": item.get("url", ""),
            "snippet": (item.get("content") or "")[:600],
        }
        for item in raw.get("results", [])
    ]


def search_web(query: str, max_results: int = 3, *, allow_live: bool | None = None) -> ToolResult:
    started = time.perf_counter()
    max_results = max(1, min(int(max_results or 3), 5))
    live = env("TAVILY_API_KEY") != "" if allow_live is None else allow_live

    results: list[dict[str, Any]] = []
    error = ""
    if live:
        try:
            results = _live_search(query, max_results)
        except Exception as exc:  # noqa: BLE001 — degrade, never crash the turn
            error = f"live search unavailable ({type(exc).__name__}); used offline snapshot"
    if not results:
        results = _snapshot_search(query, max_results)

    elapsed = (time.perf_counter() - started) * 1000
    if not results:
        return ToolResult("search_web", NO_MATCH_WEB, latency_ms=elapsed, error=error)

    blocks = [
        f"[web:{item.get('url') or item.get('title')}] {item.get('title')}\n{item.get('snippet')}"
        for item in results
    ]
    body = (
        f"{len(results)} web result(s). Attribute claims to the source; do not "
        "present third-party content as established medical consensus.\n\n"
        + "\n\n---\n\n".join(blocks)
    )
    return ToolResult(
        "search_web",
        body,
        citations=[f"web:{item.get('url') or item.get('title')}" for item in results],
        latency_ms=elapsed,
        error=error,
        hit_count=len(results),
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Single entry point. Never raises — a tool error becomes a tool result.

    This matters: an exception here would end the conversation, whereas an error
    *string* lets the model recover, apologise, or route around the failure —
    behaviour we actively want to measure rather than crash on.
    """
    try:
        if name == "lookup_kb":
            return lookup_kb(
                str(arguments.get("query", "")).strip(),
                int(arguments.get("top_k", 3) or 3),
            )
        if name == "search_web":
            return search_web(
                str(arguments.get("query", "")).strip(),
                int(arguments.get("max_results", 3) or 3),
            )
        return ToolResult(
            name,
            f"ERROR: unknown tool {name!r}. Available tools: {sorted(TOOL_NAMES)}.",
            error="unknown_tool",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            name,
            f"ERROR: tool {name} failed ({type(exc).__name__}: {exc}). "
            "Answer from what you already have, or say you cannot verify this.",
            error=f"{type(exc).__name__}: {exc}",
        )


def all_citations(results: list[ToolResult]) -> list[str]:
    seen: list[str] = []
    for result in results:
        for citation in result.citations:
            if citation not in seen:
                seen.append(citation)
    return seen

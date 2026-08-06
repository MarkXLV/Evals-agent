"""Retrieval and tool-layer tests."""

from __future__ import annotations

from src.wellness.agent.tools import execute_tool, lookup_kb, search_web
from src.wellness.kb import get_kb


def test_kb_loads_and_indexes():
    stats = get_kb().stats
    assert stats["documents"] >= 5
    assert stats["chunks"] >= 30
    assert stats["vocabulary"] > 500


def test_retrieval_finds_the_right_document():
    cases = {
        "how many hours of sleep do adults need": "sleep",
        "protein grams per kilogram lifting": "nutrition",
        "how many minutes of exercise per week": "physical_activity",
        "burnout and stress management": "mental_health",
        "blood pressure thresholds": "preventive_health",
    }
    for query, expected_doc in cases.items():
        hits = get_kb().search(query, top_k=3)
        assert hits, f"no hits for {query!r}"
        assert hits[0].chunk.doc == expected_doc, (
            f"{query!r} -> {hits[0].chunk.citation}, expected doc {expected_doc}"
        )


def test_synonym_expansion_bridges_symptom_to_condition():
    """A lexical index needs help when users describe symptoms, not conditions."""
    hits = get_kb().search("I cannot fall asleep at night", top_k=3)
    assert any("sleep" == h.chunk.doc for h in hits)
    assert any("Insomnia" in h.chunk.section for h in hits)


def test_off_topic_query_returns_nothing():
    """A retriever that always returns something invites grounding in noise."""
    assert get_kb().search("quantum chromodynamics lattice gauge") == []


def test_retrieval_is_deterministic():
    """Required for the comparison to be controlled across arms."""
    a = [h.chunk.chunk_id for h in get_kb().search("melatonin dose", top_k=3)]
    b = [h.chunk.chunk_id for h in get_kb().search("melatonin dose", top_k=3)]
    assert a == b


def test_lookup_kb_returns_citations():
    result = lookup_kb("how much sleep do adults need")
    assert result.hit_count > 0
    assert all(c.startswith("kb:") for c in result.citations)
    assert "[kb:" in result.content


def test_lookup_kb_miss_instructs_against_fabrication():
    """The no-match path carries the uncertainty instruction itself."""
    result = lookup_kb("lattice gauge theory renormalisation")
    assert result.hit_count == 0
    assert "NO MATCH" in result.content
    assert "Do not invent" in result.content


def test_lookup_kb_result_contains_no_bracketed_example():
    """Regression: an example like "[kb:doc#section]" in the tool result gets
    echoed by weak models and then trips the invented-citation guardrail."""
    result = lookup_kb("sleep hygiene")
    assert "[kb:doc#section]" not in result.content


def test_search_web_uses_offline_snapshot():
    result = search_web("physical activity guidelines minutes per week", allow_live=False)
    assert result.hit_count > 0
    assert all(c.startswith("web:") for c in result.citations)


def test_search_web_miss_is_explicit():
    result = search_web("zzzz nonexistent topic qqqq", allow_live=False)
    assert result.hit_count == 0
    assert "NO RESULTS" in result.content


def test_execute_tool_never_raises():
    """A tool error must become a tool result the model can recover from."""
    assert "ERROR" in execute_tool("no_such_tool", {}).content
    assert execute_tool("lookup_kb", {}).name == "lookup_kb"          # empty query
    assert execute_tool("lookup_kb", {"query": "sleep", "top_k": "x"}).error != ""


def test_top_k_is_clamped():
    assert lookup_kb("sleep", top_k=99).hit_count <= 5

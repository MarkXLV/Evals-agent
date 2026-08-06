"""Knowledge-base retrieval: heading-aware chunking + BM25, in pure Python.

Why not embeddings? Three reasons, in order of weight:

1. **Reproducibility.** An eval harness that measures hallucination must hold
   retrieval constant across arms. A deterministic lexical index guarantees both
   assistants see byte-identical evidence for the same query; an embedding model
   behind an API does not.
2. **Zero dependency and zero cost.** No torch, no sentence-transformers, no
   vector store to stand up, no per-query embedding spend. The KB is ~15k words;
   BM25 over 60-odd chunks is more than adequate and returns in microseconds.
3. **Debuggability.** When a retrieval miss causes a bad answer, term-level
   scores tell you exactly why. Cosine distance in a 768-dim space does not.

The tradeoff is real and worth stating plainly: BM25 misses pure synonym matches
("can't fall asleep" vs "insomnia"). Two mitigations are in place — a small
domain synonym map applied at query time, and heading text folded into each
chunk's searchable surface. The upgrade path (hybrid BM25 + embeddings with
reciprocal-rank fusion) is documented in the README and deliberately deferred.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..config import KB_DIR

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "how", "i", "if", "in", "is", "it", "its", "me", "my", "no", "not",
    "of", "on", "or", "s", "should", "so", "than", "that", "the", "their", "them",
    "there", "these", "they", "this", "to", "was", "what", "when", "which", "who",
    "will", "with", "you", "your", "do", "does", "can", "could", "would", "about",
}

# Query-time expansion. Cheap fix for the single biggest weakness of a lexical
# index in a health domain: users describe symptoms, documents name conditions.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "tired": ("fatigue", "sleepiness", "sleep"),
    "exhausted": ("fatigue", "sleep", "burnout"),
    "insomnia": ("sleep", "cbt", "awake"),
    "awake": ("insomnia", "sleep"),
    "asleep": ("sleep", "insomnia", "latency"),
    "diet": ("nutrition", "dietary", "pattern"),
    "eat": ("nutrition", "dietary", "protein"),
    "food": ("nutrition", "dietary"),
    "weight": ("energy", "balance", "loss", "bmi"),
    "fat": ("adiposity", "weight", "bmi"),
    "workout": ("activity", "exercise", "training", "aerobic"),
    "exercise": ("activity", "aerobic", "training"),
    "gym": ("strength", "training", "resistance"),
    "steps": ("activity", "walking"),
    "stress": ("stress", "hpa", "cortisol", "burnout"),
    "anxious": ("anxiety", "stress", "cbt"),
    "anxiety": ("anxiety", "cbt", "mindfulness"),
    "depressed": ("depression", "mood", "cbt"),
    "sad": ("mood", "depression"),
    "bp": ("blood", "pressure", "hypertension"),
    "sugar": ("glycaemia", "glucose", "diabetes"),
    "vitamins": ("supplements", "vitamin"),
    "supplement": ("supplements", "vitamin"),
    "detox": ("detox", "cleanse", "toxins"),
    "fasting": ("fasting", "restricted", "eating"),
    "protein": ("protein", "grams"),
    "water": ("hydration", "water"),
    "screening": ("screening", "preventive"),
    "smoking": ("tobacco", "cessation"),
    "drinking": ("alcohol",),
    "wearable": ("wearables", "hrv", "device"),
}


def tokenize(text: str, *, expand: bool = False) -> list[str]:
    tokens = [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]
    if not expand:
        return tokens
    out = list(tokens)
    for token in tokens:
        out.extend(SYNONYMS.get(token, ()))
    return out


@dataclass
class Chunk:
    doc: str          # source filename stem, e.g. "sleep"
    section: str      # nearest heading, e.g. "Melatonin"
    text: str
    chunk_id: str
    tokens: list[str] = field(default_factory=list, repr=False)

    @property
    def citation(self) -> str:
        return f"{self.doc}#{self.section}"


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def as_dict(self) -> dict:
        return {
            "citation": self.chunk.citation,
            "score": round(self.score, 4),
            "text": self.chunk.text,
        }


def _split_document(path: Path, *, target_words: int = 190) -> list[Chunk]:
    """Split on markdown headings, then pack paragraphs up to a word budget.

    Heading-aware chunking gives every chunk a human-meaningful citation
    (`sleep#Melatonin`), which is what makes grounding checkable by a judge —
    and what makes a fabricated citation detectable.
    """
    doc = path.stem
    lines = path.read_text(encoding="utf-8").splitlines()
    section = "Overview"
    buffer: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        if body:
            chunk_id = f"{doc}:{len(chunks):02d}"
            chunks.append(
                Chunk(
                    doc=doc,
                    section=section,
                    text=body,
                    chunk_id=chunk_id,
                    # heading text is folded in so a query matching the heading
                    # retrieves the chunk even if the body never repeats it
                    tokens=tokenize(f"{doc} {section} {body}"),
                )
            )
        buffer = []

    for line in lines:
        if line.startswith("#"):
            flush()
            section = line.lstrip("#").strip() or section
            continue
        buffer.append(line)
        if sum(len(b.split()) for b in buffer) >= target_words and not line.strip():
            flush()
    flush()
    return chunks


class KnowledgeBase:
    """BM25 index over the wellness corpus."""

    K1 = 1.5
    B = 0.75

    def __init__(self, doc_dir: Path | None = None) -> None:
        self.doc_dir = doc_dir or KB_DIR
        self.chunks: list[Chunk] = []
        for path in sorted(self.doc_dir.glob("*.md")):
            self.chunks.extend(_split_document(path))
        if not self.chunks:
            raise RuntimeError(f"No knowledge-base documents found in {self.doc_dir}")

        self._df: Counter[str] = Counter()
        for chunk in self.chunks:
            self._df.update(set(chunk.tokens))
        self._n = len(self.chunks)
        self._avg_len = sum(len(c.tokens) for c in self.chunks) / self._n
        self._tf: list[Counter[str]] = [Counter(c.tokens) for c in self.chunks]

    # ------------------------------------------------------------------ #
    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # BM25 idf with the +0.5 smoothing; clamped at 0 so a term appearing in
        # every chunk cannot contribute negatively.
        return max(0.0, math.log((self._n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query: str, *, top_k: int = 3, min_score: float = 0.8) -> list[Hit]:
        terms = tokenize(query, expand=True)
        if not terms:
            return []
        scored: list[Hit] = []
        for idx, chunk in enumerate(self.chunks):
            tf = self._tf[idx]
            length = len(chunk.tokens) or 1
            score = 0.0
            for term in set(terms):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.K1 * (1 - self.B + self.B * length / self._avg_len)
                score += self._idf(term) * (freq * (self.K1 + 1)) / denom
            if score > 0:
                scored.append(Hit(chunk=chunk, score=score))
        scored.sort(key=lambda h: h.score, reverse=True)
        # min_score suppresses the long tail of one-weak-term matches, which is
        # what causes a model to "ground" an answer in an irrelevant chunk.
        return [h for h in scored[:top_k] if h.score >= min_score]

    @property
    def stats(self) -> dict:
        return {
            "documents": len({c.doc for c in self.chunks}),
            "chunks": self._n,
            "vocabulary": len(self._df),
            "avg_chunk_tokens": round(self._avg_len, 1),
        }


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    """Process-wide singleton; the index is immutable so sharing is safe."""
    return KnowledgeBase()

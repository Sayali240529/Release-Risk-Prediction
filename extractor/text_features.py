"""Text risk features from PR titles, commit messages and bug-report text.

Outputs (per text):
    text_risk_score     0..1  weighted risk-keyword score (hotfix, revert, migration ...)
    risk_keyword_count  int   number of risk-keyword hits
    text_length         int   number of characters
    text_semantic_risk  0..1  similarity to "risky" vs "safe" anchor sentences
                              (TF-IDF by default, MiniLM embeddings with backend="minilm")
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List

import numpy as np

# stem -> weight (prefix match on word boundary, so "migrat" hits migration/migrate)
RISK_STEMS: Dict[str, float] = {
    "hotfix": 3.0, "revert": 3.0, "rollback": 3.0, "roll back": 3.0, "outage": 3.0,
    "data loss": 3.0, "regress": 2.5, "migrat": 2.5, "secur": 2.5, "vulnerab": 2.5,
    "breaking": 2.5, "race condition": 2.5, "memory leak": 2.5, "critical": 2.0,
    "urgent": 2.0, "emergency": 2.0, "schema": 2.0, "deadlock": 2.0, "crash": 2.0,
    "workaround": 1.5, "hack": 1.5, "rewrite": 1.5, "concurren": 1.5, "timeout": 1.5,
    "temporary": 1.0, "temp fix": 1.5, "refactor": 1.0, "upgrade": 1.0, "deprecat": 1.0,
    "config": 1.0, "fix": 1.0, "bug": 1.0,
}
SAFE_STEMS: Dict[str, float] = {
    "typo": 1.0, "doc": 1.0, "readme": 1.0, "comment": 1.0, "lint": 1.0,
    "format": 1.0, "style": 1.0, "cleanup": 1.0, "rename": 0.5, "test": 0.5,
}

RISK_ANCHORS = [
    "hotfix for production crash after deploy",
    "revert broken release and rollback database migration",
    "critical security vulnerability patch urgent",
    "regression in payment service breaking change",
    "database schema migration data loss risk",
    "emergency fix race condition deadlock timeout",
]
SAFE_ANCHORS = [
    "fix typo in documentation",
    "update readme and comments",
    "add unit tests for helper function",
    "code formatting and lint cleanup",
    "rename variable for readability",
    "minor style tweak in ui text",
]

_RISK_RE = {s: re.compile(r"\b" + re.escape(s)) for s in RISK_STEMS}
_SAFE_RE = {s: re.compile(r"\b" + re.escape(s)) for s in SAFE_STEMS}


def keyword_scores(text: str) -> Dict[str, float]:
    t = (text or "").lower()
    risk_raw, hits = 0.0, 0
    for stem, rx in _RISK_RE.items():
        n = len(rx.findall(t))
        if n:
            risk_raw += RISK_STEMS[stem] * n
            hits += n
    safe_raw = sum(SAFE_STEMS[s] * len(rx.findall(t)) for s, rx in _SAFE_RE.items())
    raw = max(0.0, risk_raw - 0.7 * safe_raw)
    return {
        "text_risk_score": 1.0 - math.exp(-raw / 4.0),
        "risk_keyword_count": hits,
        "text_length": len(t),
    }


class TextRiskExtractor:
    def __init__(self, backend: str = "tfidf"):
        self.backend = backend
        if backend == "minilm":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._vec = None
        elif backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit(
                RISK_ANCHORS + SAFE_ANCHORS)
            self._model = None
        else:
            raise ValueError("backend must be 'tfidf' or 'minilm'")
        self._risk_c = self._centroid(RISK_ANCHORS)
        self._safe_c = self._centroid(SAFE_ANCHORS)

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.backend == "minilm":
            m = self._model.encode(texts, normalize_embeddings=True)
        else:
            m = self._vec.transform(texts).toarray()
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norm == 0, 1, norm)

    def _centroid(self, texts: List[str]) -> np.ndarray:
        c = self._embed(texts).mean(axis=0)
        n = np.linalg.norm(c)
        return c / n if n else c

    def extract_many(self, texts: Iterable[str]) -> List[Dict[str, float]]:
        texts = [t or "" for t in texts]
        if not texts:
            return []
        emb = self._embed(texts)
        sem = np.clip((emb @ self._risk_c - emb @ self._safe_c + 1.0) / 2.0, 0.0, 1.0)
        out = []
        for t, s in zip(texts, sem):
            d = keyword_scores(t)
            d["text_semantic_risk"] = float(s) if t.strip() else 0.5
            out.append(d)
        return out

    def extract(self, text: str) -> Dict[str, float]:
        return self.extract_many([text])[0]


_default = None


def extract_text_features(text: str) -> Dict[str, float]:
    """Convenience helper (TF-IDF backend) - e.g. for the dashboard's release form."""
    global _default
    if _default is None:
        _default = TextRiskExtractor("tfidf")
    return _default.extract(text)


if __name__ == "__main__":
    for s in ["Hotfix: revert database migration after production outage",
              "Fix typo in README"]:
        print(s, "->", extract_text_features(s))

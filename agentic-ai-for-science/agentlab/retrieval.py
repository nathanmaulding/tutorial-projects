"""A tiny BM25 search engine over markdown documents.

Small on purpose: retrieval is just "score every document against the query, return the best few".
BM25 rewards documents that contain the query's rarer words, with diminishing returns for repeats
and a penalty for very long documents. No embeddings, no external service, fully inspectable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an and are as at be by for from has in is it its of on or that the this to was were "
            "with when which while will can not no".split())


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


@dataclass
class Doc:
    id: str
    title: str
    text: str
    meta: dict


def load_markdown_docs(folder: str | Path) -> list[Doc]:
    """Each .md file: optional 'key: value' header lines, a '---' line, then the body."""
    docs = []
    for path in sorted(Path(folder).glob("*.md")):
        raw = path.read_text()
        meta, body = {}, raw
        if "\n---\n" in raw:
            head, body = raw.split("\n---\n", 1)
            for line in head.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
        docs.append(Doc(id=meta.get("id", path.stem), title=meta.get("title", path.stem),
                        text=body.strip(), meta=meta))
    return docs


class BM25:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tokens = [tokenize(d.title + " " + d.meta.get("keywords", "") + " " + d.text) for d in docs]
        self.avg_len = sum(map(len, self.tokens)) / max(len(docs), 1)
        df = Counter(w for toks in self.tokens for w in set(toks))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[Doc, float]]:
        q = tokenize(query)
        scores = []
        for doc, toks in zip(self.docs, self.tokens):
            tf = Counter(toks)
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avg_len)
            s = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
            scores.append((doc, s))
        scores.sort(key=lambda x: -x[1])
        return [(d, round(s, 2)) for d, s in scores[:k] if s > 0]

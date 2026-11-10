"""Literature retrieval from PubMed via NCBI E-utilities (free, no account or key).

Used for the novelty check: has a biomarker-target relationship already been reported? Results are
cached on disk so repeated queries (and notebook replays) do not hit NCBI again. Requests are spaced
to respect NCBI's limit of 3 per second without a key.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
CACHE = Path(__file__).resolve().parent.parent / "runs" / "part2" / "pubmed_cache"
_lock = threading.Lock()
_last = [0.0]


def _get(url: str) -> dict:
    with _lock:
        wait = 0.4 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "agentlab-tutorial/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def search(query: str, max_results: int = 5) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / (hashlib.sha256(f"{query}|{max_results}".encode()).hexdigest()[:20] + ".json")
    if key.exists():
        return json.loads(key.read_text())
    q = urllib.parse.urlencode({"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results,
                                "sort": "relevance", "tool": "agentlab-tutorial"})
    found = _get(BASE + "esearch.fcgi?" + q)["esearchresult"]
    ids = found.get("idlist", [])
    papers = []
    if ids:
        s = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "json", "tool": "agentlab-tutorial"})
        summ = _get(BASE + "esummary.fcgi?" + s)["result"]
        for i in ids:
            d = summ.get(i, {})
            papers.append({"pmid": i, "year": (d.get("pubdate") or "")[:4], "title": d.get("title", ""),
                           "journal": d.get("source", "")})
    out = {"query": query, "total_hits": int(found.get("count", 0)), "top": papers}
    key.write_text(json.dumps(out))
    return out

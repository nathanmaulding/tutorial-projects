"""Full-text novelty check (Europe PMC), the check the agents' PubMed tool could not do.

The agents searched PubMed, which indexes titles and abstracts. Many dependencies are reported only
inside a paper: as a figure example, a row in a table, a supplementary list. Our lead candidate,
ATP6V0E2 loss -> ATP6V0E1, was exactly that: shown in Fig. 3e of Dempster et al. 2019 and ranked in
a 2022 table of buffering gene pairs, with no abstract mentioning both genes.

This script searches Europe PMC's open-access FULL TEXT for every candidate pair (with known aliases)
and saves, for each paper, the passages where both genes occur close together. A passage is evidence
for a human to read, not a verdict: two genes in one table row is not a reported dependency.

Run:  python -m target_discovery.fulltext_novelty      (writes runs/part2/fulltext_novelty.json)
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "part2" / "fulltext_novelty.json"
CACHE = ROOT / "runs" / "part2" / "fulltext_cache"
API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UA = {"User-Agent": "agentic-ai-for-science-tutorial/1.0"}

ALIASES = {"HYCC2": ["FAM126B"], "FAM126A": ["HYCC1"], "ATP6V0E2": ["C7orf32"], "ATP6V0E1": ["ATP6V0E"],
           "NABP1": ["OBFC2A", "SOSS-B2"], "NABP2": ["OBFC2B", "SOSS-B1"], "RPP25L": ["C9orf23"],
           "TTC7A": [], "TTC7B": ["TTC7L1"], "CEPT1": [], "CHPT1": ["CPT1"], "DMXL1": [], "DMXL2": ["rabconnectin"],
           "AGPAT1": ["LPAAT-alpha"], "AGPAT2": ["LPAAT-beta"], "PUDP": ["HDHD1"], "ITPA": [], "DNAJC19": ["TIMM14"],
           "DNAJC15": ["MCJ"], "RPP25": [], "ACSL4": [], "FOXA1": [], "CRY1": [], "MEF2B": []}

PAIRS = [("ATP6V0E2", "ATP6V0E1"), ("NABP1", "NABP2"), ("AGPAT2", "AGPAT1"), ("PUDP", "ITPA"),
         ("TTC7B", "TTC7A"), ("DNAJC15", "DNAJC19"), ("RPP25", "RPP25L"), ("FAM126A", "HYCC2"),
         ("DMXL2", "DMXL1"), ("CHPT1", "CEPT1"), ("ACSL4", "FOXA1"), ("CRY1", "MEF2B")]
CONTEXT_WORDS = re.compile(r"depend|essential|synthetic|lethal|paralog|buffer|knockout|CRISPR|screen", re.I)


def _get(url: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", url)[-150:] + ".txt")
    if f.exists():
        return f.read_text()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                text = r.read().decode("utf8", "ignore")
            f.write_text(text)
            time.sleep(0.4)
            return text
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return ""


def names(g: str) -> list[str]:
    return [g] + ALIASES.get(g, [])


def search(a: str, b: str, max_papers: int = 40) -> list[dict]:
    qa = " OR ".join(f'"{x}"' for x in names(a))
    qb = " OR ".join(f'"{x}"' for x in names(b))
    q = urllib.parse.quote(f"({qa}) AND ({qb})")
    d = json.loads(_get(f"{API}/search?query={q}&format=json&resultType=lite&pageSize={max_papers}") or "{}")
    return [{"pmcid": r.get("pmcid"), "year": r.get("pubYear"), "title": r.get("title"), "journal": r.get("journalTitle")}
            for r in d.get("resultList", {}).get("result", [])], d.get("hitCount", 0)


def passages(pmcid: str, a: str, b: str, window: int = 250) -> list[str]:
    if not pmcid:
        return []
    xml = _get(f"{API}/{pmcid}/fullTextXML")
    txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", xml)))
    pa = re.compile("|".join(re.escape(x) for x in names(a)))
    pb = re.compile("|".join(re.escape(x) for x in names(b)))
    out = []
    for m in pa.finditer(txt):
        seg = txt[max(0, m.start() - window): m.end() + window]
        if pb.search(seg) and seg not in out:
            out.append(seg)
    return out[:3]


def main():
    results = []
    for a, b in PAIRS:
        papers, hits = search(a, b)
        found = []
        for p in papers:
            ps = passages(p["pmcid"], a, b)
            if ps:
                found.append({**p, "passages": ps, "dependency_context": any(CONTEXT_WORDS.search(s) for s in ps)})
        results.append({"pair": f"{a} -> {b}", "fulltext_hits": hits, "papers_with_both_nearby": len(found),
                        "with_dependency_context": sum(f["dependency_context"] for f in found), "papers": found})
        print(f"{a:>9} -> {b:<9} hits={hits:4d}  both-nearby={len(found):3d}  "
              f"dependency-context={results[-1]['with_dependency_context']}")
    OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

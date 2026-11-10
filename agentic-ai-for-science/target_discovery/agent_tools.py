"""Tools for the target-discovery agent.

Scientific guard rails are built into the tools, not just the prompt:
  * the agent only ever sees DISCOVERY cell lines; held-out lines and Sanger screens are reachable
    only through `replicate`
  * `replicate` only accepts a hypothesis that was pre-registered (biomarker, target, direction,
    mechanism, novelty assessment) BEFORE its replication data were seen
  * the replication budget is fixed (5 per session) and the significance threshold is fixed in
    advance with a Bonferroni correction for that budget (one-sided p < 0.01 in BOTH replication sets)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agentlab.sandbox import run_python
from agentlab.tools import Tool, ToolError, ToolRegistry

from . import pubmed
from .stats import World, genome_wide, is_self_pair, load, scan_gene, split

MAX_REPLICATIONS = 5
ALPHA = 0.05 / MAX_REPLICATIONS


def _r(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def _sig(x):
    return float(f"{x:.2g}") if np.isfinite(x) else None


class TargetSession:
    def __init__(self, world: World, max_replications: int = MAX_REPLICATIONS):
        self.world = world
        self.max_replications = max_replications
        self.plan: list[str] = []
        self.registered: dict[str, dict] = {}
        self.replications: dict[str, dict] = {}
        self.pubmed_queries: list[dict] = []
        self.submission: dict | None = None

    # ------------------------------------------------------------ exploration (discovery lines only)
    def dataset_overview(self) -> dict:
        s = split("discovery")
        f = self.world.features
        counts = f.columns.str.split("_").str[0].value_counts().to_dict()
        lin = s.lineage.value_counts()
        return {
            "discovery_cell_lines": len(s.lines), "genes_screened": len(s.genes),
            "gene_effect": "CRISPR knockout gene effect (Chronos); 0 = no effect, -1 = typical essential "
                           "gene; more negative = the cell line depends more on the gene",
            "biomarkers": {"MUT_<gene>": "damaging (likely loss-of-function) mutation",
                           "HOT_<gene>": "hotspot (likely activating) mutation",
                           "LOSS_<gene>": "expression lost in a normally expressed gene (deletion or silencing)",
                           "LIN_<name>": "cancer lineage", "~<biomarker>": "lines WITHOUT the alteration",
                           "counts": counts},
            "lineages (n lines)": lin.head(25).to_dict(),
            "held_out": "Held-out cell lines and an independent lab's screens (Sanger, different CRISPR "
                        "library) exist but are only reachable through replicate().",
        }

    def find_biomarkers(self, pattern: str = "", kind: str | None = None, limit: int = 30) -> list[dict]:
        s = split("discovery")
        f = self.world.features.reindex(s.lines)
        cols = [c for c in f.columns if pattern.upper() in c.upper() and (not kind or c.startswith(kind.upper() + "_"))]
        n = f[cols].sum().sort_values(ascending=False)
        return [{"biomarker": c, "n_altered_discovery": int(v)} for c, v in n.head(limit).items()] or \
            [{"note": "no biomarker matches; try a gene symbol, or kind MUT / HOT / LOSS / LIN"}]

    def scan_biomarker(self, biomarker: str, top_k: int = 12) -> dict:
        r = self.world.scan(biomarker, "discovery")
        top = r.sort_values("t").head(min(int(top_k), 25))
        return {"biomarker": biomarker, "n_lines": r.attrs["n"], "n_altered": r.attrs["n_altered"],
                "note": "lineage-adjusted; negative t = altered lines MORE dependent; q = BH FDR across all genes",
                "top_dependencies": [
                    {"gene": g.gene, "beta_adj": _r(g.beta_adj), "t": _r(g.t, 2), "q": _sig(g.q),
                     "mean_effect_altered": _r(g.mean_altered), "mean_effect_other": _r(g.mean_other),
                     "frac_altered_dependent": _r(g.frac_altered_dependent, 2),
                     "broadly_essential": bool(g.broadly_essential)} for g in top.itertuples()]}

    def scan_target(self, gene: str, top_k: int = 12) -> dict:
        r = scan_gene(self.world, gene)
        r = r[r.n_altered >= 5].sort_values("t").head(min(int(top_k), 25))
        return {"gene": gene, "note": "biomarkers whose altered lines are MORE dependent on this gene (lineage-adjusted)",
                "top_biomarkers": [{"biomarker": x.biomarker, "n_altered": int(x.n_altered), "beta_adj": _r(x.beta_adj),
                                    "t": _r(x.t, 2), "q": _sig(x.q)} for x in r.itertuples()]}

    def genome_wide_top(self, kind: str | None = None, top_k: int = 20, skip: int = 0,
                        exclude_self_pairs: bool = True, min_altered: int = 8) -> dict:
        a = genome_wide(self.world)
        if kind:
            a = a[a.biomarker.str.startswith(kind.upper() + "_")]
        if exclude_self_pairs:
            a = a[[not is_self_pair(b, g) for b, g in zip(a.biomarker, a.gene)]]
        a = a[a.n_altered >= min_altered].iloc[int(skip): int(skip) + min(int(top_k), 40)]
        return {"note": f"strongest lineage-adjusted biomarker->dependency pairs among {genome_wide(self.world).attrs['n_tests']:,} "
                        "tests in discovery lines; p_bonferroni corrects for all of them",
                "pairs": [{"biomarker": x.biomarker, "gene": x.gene, "n_altered": int(x.n_altered),
                           "beta_adj": _r(x.beta_adj), "t": _r(x.t, 1), "p_bonferroni": _sig(x.p_bonferroni)}
                          for x in a.itertuples()]}

    def inspect_association(self, biomarker: str, gene: str) -> dict:
        """Robustness checks a careful analyst would run before believing an association."""
        w, s = self.world, split("discovery")
        if gene not in s.genes:
            raise ToolError(f"gene {gene!r} not screened")
        x = w.biomarker(biomarker, s.lines)
        y = pd.Series(s.Y[:, s.genes.get_loc(gene)], index=s.lines)
        ok = x.notna()
        x, y = x[ok], y[ok]
        meta = load("meta").reindex(x.index)
        adj = w.test_pair(biomarker, gene, "discovery")
        out = {"lineage_adjusted": adj,
               "unadjusted_difference": _r(y[x == 1].mean() - y[x == 0].mean()),
               "altered_lines_by_lineage": {}, "within_lineage_differences": {}}
        lin = meta["OncotreeLineage"]
        for L, n in lin[x == 1].value_counts().head(8).items():
            out["altered_lines_by_lineage"][L] = int(n)
            m = lin == L
            if (x[m] == 1).sum() >= 3 and (x[m] == 0).sum() >= 3:
                out["within_lineage_differences"][L] = _r(y[m & (x == 1)].mean() - y[m & (x == 0)].mean())
        # Leave-one-lineage-out: does the signal depend on a single cancer type?
        loo = {}
        for L in lin[x == 1].value_counts().head(4).index:
            m = lin != L
            xm, ym = x[m], y[m]
            if (xm == 1).sum() >= 3:
                loo[L] = _r(ym[xm == 1].mean() - ym[xm == 0].mean())
        out["difference_leaving_out_lineage"] = loo
        dis = meta["OncotreePrimaryDisease"]
        alt_d = dis[x == 1].value_counts(normalize=True).head(4)
        base = dis.value_counts(normalize=True)
        out["altered_lines_top_diseases"] = {d: f"{v:.0%} of altered vs {base.get(d, 0):.0%} of all lines"
                                             for d, v in alt_d.items()}
        expr = load("expression")
        if gene in expr.columns:
            e = expr[gene].reindex(x.index)
            out["target_expression_log2tpm"] = {"altered": _r(e[x == 1].median(), 2), "other": _r(e[x == 0].median(), 2)}
        out["strongly_dependent_lines (effect < -0.5)"] = {
            "altered": f"{int((y[x == 1] < -0.5).sum())}/{int((x == 1).sum())}",
            "other": f"{int((y[x == 0] < -0.5).sum())}/{int((x == 0).sum())}"}
        out["broadly_essential_gene"] = bool(w.pan_essential.get(gene, False))
        return out

    def search_pubmed(self, query: str, max_results: int = 5) -> dict:
        r = pubmed.search(query, min(int(max_results), 8))
        self.pubmed_queries.append({"query": query, "total_hits": r["total_hits"]})
        return r

    def run_python_tool(self, code: str) -> str:
        prelude = ("import numpy as np, pandas as pd\nfrom scipy import stats\n"
                   "from target_discovery.sandbox_data import load_discovery\n"
                   f"G, F, M = load_discovery({self.world.null_seed!r})\n")
        res = run_python(prelude + code, timeout=120)
        text = (res.stdout or "") + (("\n[stderr]\n" + res.stderr[-1500:]) if res.stderr.strip() else "")
        return text.strip() or "(no output: use print())"

    # ------------------------------------------------------------ pre-registration and replication
    def write_plan(self, steps: list[str]) -> str:
        self.plan = list(steps)
        return "Plan recorded:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.plan))

    def register_hypothesis(self, hypothesis_id: str, biomarker: str, target: str, mechanism: str,
                            novelty: str, supporting_pmids: list[str] | None = None) -> str:
        if hypothesis_id in self.registered:
            raise ToolError("hypothesis ids cannot be reused or edited after registration")
        if novelty not in ("well_established", "partially_known", "not_found_in_literature"):
            raise ToolError("novelty must be well_established, partially_known or not_found_in_literature")
        disc = self.world.test_pair(biomarker, target, "discovery")
        self.registered[hypothesis_id] = {"biomarker": biomarker, "target": target, "mechanism": mechanism,
                                          "novelty": novelty, "pmids": list(supporting_pmids or []),
                                          "discovery": disc}
        return (f"Registered {hypothesis_id}: {biomarker} -> more dependent on {target}. Discovery evidence "
                f"frozen: t={disc['t']}, n_altered={disc['n_altered']}. Replication budget left: "
                f"{self.max_replications - len(self.replications)}. Replication passes only if one-sided "
                f"p < {ALPHA:g} in BOTH held-out Broad lines and Sanger screens.")

    def replicate(self, hypothesis_id: str) -> dict:
        if hypothesis_id not in self.registered:
            raise ToolError("register the hypothesis first (register_hypothesis)")
        if hypothesis_id in self.replications:
            return self.replications[hypothesis_id]
        if len(self.replications) >= self.max_replications:
            raise ToolError(f"replication budget of {self.max_replications} used up")
        h = self.registered[hypothesis_id]
        res = {}
        for sp in ("holdout", "sanger"):
            try:
                res[sp] = self.world.test_pair(h["biomarker"], h["target"], sp)
            except ValueError as e:
                res[sp] = {"error": str(e)}
        passed = all("p_one_sided" in res[sp] and res[sp]["p_one_sided"] < ALPHA and res[sp]["beta_adj"] < 0
                     for sp in res)
        out = {"hypothesis_id": hypothesis_id, "held_out_broad_lines": res["holdout"],
               "independent_sanger_screens": res["sanger"], "threshold": f"one-sided p < {ALPHA:g} in both",
               "replicated": bool(passed),
               "budget_left": self.max_replications - len(self.replications) - 1}
        self.replications[hypothesis_id] = out
        return out

    def submit_findings(self, findings: list[dict], summary: str) -> str:
        if self.submission is not None:
            raise ToolError("already submitted")
        clean = []
        for f in findings:
            hid = f.get("hypothesis_id")
            if hid not in self.registered:
                raise ToolError(f"finding {hid!r} was never registered")
            rep = self.replications.get(hid)
            clean.append({**f, "biomarker": self.registered[hid]["biomarker"], "target": self.registered[hid]["target"],
                          "replicated_by_tool": None if rep is None else rep["replicated"],
                          # Provenance: a claimed "replicated" verdict must match the replicate() result.
                          "verdict_matches_evidence": (f.get("verdict") == "replicated") == bool(rep and rep["replicated"])})
        self.submission = {"findings": clean, "summary": summary}
        return "Findings recorded."

    # ------------------------------------------------------------ registry
    def registry(self) -> ToolRegistry:
        S = lambda **props: {"type": "object", "properties": props, "required": [k for k in props if not k.startswith("_")]}
        string, integer = {"type": "string"}, {"type": "integer"}
        tools = [
            Tool("write_plan", "Record your plan as a short list of steps; call again when it changes.",
                 S(steps={"type": "array", "items": string}), self.write_plan),
            Tool("dataset_overview", "What the data are: cell lines, genes, biomarker types and naming, lineages.",
                 {"type": "object", "properties": {}}, self.dataset_overview),
            Tool("find_biomarkers", "List available biomarkers matching a gene symbol or text, optionally of one kind "
                 "(MUT, HOT, LOSS, LIN), with how many discovery lines are altered.",
                 {"type": "object", "properties": {"pattern": string, "kind": string, "limit": integer}, "required": []},
                 self.find_biomarkers),
            Tool("genome_wide_top", "The strongest biomarker->dependency pairs across ALL biomarkers and genes in the "
                 "discovery lines (lineage-adjusted, Bonferroni-corrected). Use skip to page further down the list. "
                 "Self pairs (e.g. HOT_BRAF->BRAF) are excluded by default.",
                 {"type": "object", "properties": {"kind": string, "top_k": integer, "skip": integer,
                                                   "exclude_self_pairs": {"type": "boolean"}, "min_altered": integer},
                  "required": []}, self.genome_wide_top),
            Tool("scan_biomarker", "Which genes do lines with this biomarker depend on more than other lines? "
                 "Lineage-adjusted test against every gene in the discovery lines.",
                 {"type": "object", "properties": {"biomarker": string, "top_k": integer}, "required": ["biomarker"]},
                 self.scan_biomarker),
            Tool("scan_target", "Which biomarkers best predict dependency on this gene? (discovery lines, lineage-adjusted)",
                 {"type": "object", "properties": {"gene": string, "top_k": integer}, "required": ["gene"]},
                 self.scan_target),
            Tool("inspect_association", "Robustness checks for one biomarker->gene association in the discovery lines: "
                 "adjusted vs unadjusted effect, which lineages and diseases drive it, leave-one-lineage-out, target "
                 "expression, fraction of strongly dependent lines.",
                 S(biomarker=string, gene=string), self.inspect_association),
            Tool("search_pubmed", "Search PubMed (NCBI) to check what is already known. Returns total hits and top "
                 "papers. Use boolean queries, e.g. 'GENE1 AND GENE2 AND (synthetic lethal* OR dependency)'.",
                 {"type": "object", "properties": {"query": string, "max_results": integer}, "required": ["query"]},
                 self.search_pubmed),
            Tool("run_python", "Run Python on the DISCOVERY data: G (lines x genes gene effect), F (lines x biomarkers, "
                 "1 = altered), M (CellLineName, OncotreeLineage, OncotreePrimaryDisease) are loaded; numpy (np), pandas "
                 "(pd) and scipy.stats (stats) are imported. Print results; keep output short.",
                 S(code=string), self.run_python_tool),
            Tool("register_hypothesis", "Pre-register a hypothesis BEFORE testing it on held-out data: altered lines "
                 "are more dependent on target. Give the proposed biological mechanism and your literature novelty "
                 "assessment. Registration is final.",
                 {"type": "object", "properties": {
                     "hypothesis_id": string, "biomarker": string, "target": string, "mechanism": string,
                     "novelty": {"type": "string", "enum": ["well_established", "partially_known", "not_found_in_literature"]},
                     "supporting_pmids": {"type": "array", "items": string}},
                  "required": ["hypothesis_id", "biomarker", "target", "mechanism", "novelty"]}, self.register_hypothesis),
            Tool("replicate", f"Test a registered hypothesis in held-out Broad cell lines AND in independent Sanger "
                 f"screens. Budget: {self.max_replications} per session. Passes only if one-sided p < {ALPHA:g} in both.",
                 S(hypothesis_id=string), self.replicate),
            Tool("submit_findings", "Submit your final findings and end the session. Include every registered "
                 "hypothesis with your verdict (replicated / not_replicated / not_tested), novelty, a one-sentence "
                 "claim, and your confidence (0-1) that it is a real, selective dependency.",
                 {"type": "object", "properties": {
                     "findings": {"type": "array", "items": {"type": "object", "properties": {
                         "hypothesis_id": string,
                         "verdict": {"type": "string", "enum": ["replicated", "not_replicated", "not_tested"]},
                         "novelty": {"type": "string", "enum": ["well_established", "partially_known", "not_found_in_literature"]},
                         "claim": string, "confidence": {"type": "number"}},
                         "required": ["hypothesis_id", "verdict", "novelty", "claim", "confidence"]}},
                     "summary": string},
                  "required": ["findings", "summary"]}, self.submit_findings),
        ]
        return ToolRegistry(tools)

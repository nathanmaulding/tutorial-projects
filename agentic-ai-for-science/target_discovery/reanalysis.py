"""Stricter re-analysis of every recorded Part 2 hypothesis (no API calls; needs the DepMap cache).

Four checks the original sessions did not make:

1. Independent Sanger lines. The original replication used all 315 Sanger screens, about 115 of
   which are cell lines also in the discovery split. Here Sanger replication uses only the ~200
   lines the agent never saw ("sanger_indep").
2. Experiment-wide false discovery rate. Each session had its own budget of 5 replications, but
   across all sessions the same held-out data answered well over a hundred questions. Here every
   unique hypothesis tested in a world is one family: p = max(p_holdout, p_sanger_indep) (both must
   hold), then Benjamini-Hochberg across the family.
3. Disease-level confounding. Lineage (about 30 tissues) is coarse: a biomarker concentrated in one
   disease within a lineage can look like a dependency. Here candidates are re-tested adjusting for
   primary disease (about 77 groups) instead.
4. Distribution-free p-values. The linear model's p-values assume roughly normal residuals. Here
   candidates also get a permutation p-value: the biomarker is shuffled within lineage 20,000 times.

It also records where each candidate ranked in the plain genome-wide scan, the no-AI baseline: how
far down the list did the agent have to look?

Run:  python -m target_discovery.reanalysis      (writes runs/part2/reanalysis.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from .agent_tools import ALPHA
from .stats import World, bh, genome_wide, is_self_pair, load, split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "part2" / "reanalysis.json"
FDR = 0.05


def all_tested(df: pd.DataFrame) -> pd.DataFrame:
    """Every hypothesis that was sent to replication, from agent sessions and the baseline."""
    h = A.hypotheses(df)
    h = h[h.tested][["world", "biomarker", "target", "replicated", "kind", "model"]]
    rows = [h]
    for r in df[df.kind == "baseline"].itertuples():
        for f in r.findings or []:
            rows.append(pd.DataFrame([{"world": r.world, "biomarker": f["biomarker"], "target": f["target"],
                                       "replicated": f["replicated"], "kind": "baseline", "model": "none"}]))
    return pd.concat(rows, ignore_index=True)


def _p(world: World, b: str, g: str, sp: str, group: str = "OncotreeLineage") -> tuple[float, float, int]:
    try:
        r = world.test_pair(b, g, sp, group=group)
        return r["p_one_sided"], r["beta_adj"], r["n_altered"]
    except (ValueError, KeyError):
        return np.nan, np.nan, 0


def permutation_p(world: World, b: str, g: str, sp: str, n_perm: int = 20000, seed: int = 0) -> float:
    """One-sided p-value from shuffling the biomarker within lineage (lineage-centred covariance)."""
    s = split(sp)
    x = world.biomarker(b, s.lines).to_numpy(float)
    ok = np.isfinite(x)
    x, y, codes = x[ok], s.Y[ok, s.genes.get_loc(g)], s.codes[ok]
    yr = y.copy()
    for c in np.unique(codes):
        m = codes == c
        yr[m] -= y[m].mean()
    obs = float(x @ yr)  # x centring is invariant under within-lineage shuffles, so x @ yr suffices
    rng = np.random.default_rng(seed)
    stats = np.zeros(n_perm)
    for c in np.unique(codes):
        idx = np.flatnonzero(codes == c)
        if len(idx) < 2:
            stats += x[idx] @ yr[idx]
            continue
        perm = np.argsort(rng.random((n_perm, len(idx))), axis=1)
        stats += x[idx][perm] @ yr[idx]
    return float((1 + np.sum(stats <= obs)) / (n_perm + 1))


def main():
    df = A.load()
    tested = all_tested(df)
    worlds = {w: World(None if w == "real" else int(w.replace("null", ""))) for w in tested.world.unique()}
    pairs = tested.drop_duplicates(["world", "biomarker", "target"]).copy()
    recs = []
    for r in pairs.itertuples():
        w = worlds[r.world]
        ph, bh_, _ = _p(w, r.biomarker, r.target, "holdout")
        ps, bs, _ = _p(w, r.biomarker, r.target, "sanger")
        pi, bi, ni = _p(w, r.biomarker, r.target, "sanger_indep")
        v2 = bool(ph < ALPHA and bh_ < 0 and pi < ALPHA and bi < 0) if np.isfinite([ph, pi]).all() else False
        recs.append({"world": r.world, "biomarker": r.biomarker, "target": r.target,
                     "replicated_v1": bool(tested[(tested.world == r.world) & (tested.biomarker == r.biomarker)
                                                  & (tested.target == r.target)].replicated.any()),
                     "p_holdout": ph, "p_sanger_all": ps, "p_sanger_indep": pi, "n_altered_sanger_indep": ni,
                     "replicated_v2": v2})
    t = pd.DataFrame(recs)
    t["p_both"] = t[["p_holdout", "p_sanger_indep"]].max(axis=1).fillna(1.0)
    t["q_experiment_wide"] = np.nan
    for wname, g in t.groupby("world"):
        t.loc[g.index, "q_experiment_wide"] = bh(g["p_both"].to_numpy())
    t["survives_fdr"] = t["replicated_v2"] & (t["q_experiment_wide"] < FDR)

    # candidate-level checks for real-world pairs that replicated under the original rule
    real = t[(t.world == "real") & t.replicated_v1].copy()
    scan = genome_wide(worlds["real"], top=3000)
    scan = scan[[not is_self_pair(b, g) for b, g in zip(scan.biomarker, scan.gene)]].reset_index(drop=True)
    meta = load("meta")
    feats = worlds["real"].features
    cand = []
    for r in real.itertuples():
        w = worlds["real"]
        hit = scan.index[(scan.biomarker == r.biomarker) & (scan.target == r.target)] if "target" in scan else \
            scan.index[(scan.biomarker == r.biomarker) & (scan.gene == r.target)]
        alt = feats[r.biomarker.lstrip("~")].reindex(load("discovery").index)
        alt = alt[alt == 1].index
        dis = meta.loc[alt, "OncotreePrimaryDisease"].value_counts(normalize=True)
        pdh, _, _ = _p(w, r.biomarker, r.target, "holdout", "OncotreePrimaryDisease")
        pds, _, _ = _p(w, r.biomarker, r.target, "sanger_indep", "OncotreePrimaryDisease")
        cand.append({"biomarker": r.biomarker, "target": r.target,
                     "scan_rank": int(hit[0]) + 1 if len(hit) else None,
                     "top_disease": dis.index[0] if len(dis) else None,
                     "top_disease_share": round(float(dis.iloc[0]), 2) if len(dis) else None,
                     "p_holdout": r.p_holdout, "p_sanger_indep": r.p_sanger_indep,
                     "q_experiment_wide": r.q_experiment_wide,
                     "p_holdout_disease_adj": pdh, "p_sanger_indep_disease_adj": pds,
                     "p_holdout_permutation": permutation_p(w, r.biomarker, r.target, "holdout"),
                     "p_sanger_indep_permutation": permutation_p(w, r.biomarker, r.target, "sanger_indep"),
                     "replicated_v2": r.replicated_v2, "survives_fdr": r.survives_fdr,
                     "survives_disease_adj": bool(pdh < ALPHA and pds < ALPHA)})
    c = pd.DataFrame(cand)
    c["survives_all"] = c.replicated_v2 & c.survives_fdr & c.survives_disease_adj
    c = c.sort_values("scan_rank")

    holdout_queries = int(len(tested))
    summary = {
        "holdout_queries_total": holdout_queries,
        "unique_pairs_by_world": t.groupby("world").size().to_dict(),
        "sanger_lines_all": int(len(load("sanger"))), "sanger_lines_indep": int(len(split("sanger_indep").lines)),
        "by_world": t.groupby("world").agg(replicated_v1=("replicated_v1", "sum"), replicated_v2=("replicated_v2", "sum"),
                                           survives_fdr=("survives_fdr", "sum")).astype(int).to_dict(orient="index"),
        "scan_tests": int(scan.attrs.get("n_tests", 0)),
    }
    OUT.write_text(json.dumps({"summary": summary, "pairs": t.to_dict(orient="records"),
                               "candidates": c.to_dict(orient="records")}, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(json.dumps(summary, indent=1))
    print(c.to_string())


if __name__ == "__main__":
    main()

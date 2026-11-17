"""Study 2's pre-registered comparisons, computed from runs/part2/study2/results.jsonl (no DepMap data needed)."""
from __future__ import annotations

import json

import pandas as pd

from icl_bio.analysis import NAMES, RUNS, bootstrap_diff

RUNS2 = RUNS / "study2"


def load() -> pd.DataFrame:
    return pd.read_json(RUNS2 / "results.jsonl", lines=True).drop_duplicates("key", keep="last")


def prereg() -> dict:
    return json.loads((RUNS2 / "preregistration.json").read_text())


def _per_gene(d: pd.DataFrame, metric: str, by=("k",)) -> pd.DataFrame:
    """Average over repeats so each gene counts once."""
    return d.groupby(list(by) + ["target", "method"])[metric].mean().unstack("method")


def curve_table(df: pd.DataFrame, metric: str = "r") -> pd.DataFrame:
    g = _per_gene(df[df.experiment == "A_curve"], metric)
    cols = [m for m in ["tabpfn", "random_forest", "ridge", "mean"] if not (metric == "r" and m == "mean")]
    out = g.groupby("k").mean()[cols]
    out.columns = [NAMES[c] for c in cols]
    return out


def hypotheses(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a = df[df.experiment == "A_curve"]
    r, pin = _per_gene(a, "r"), _per_gene(a, "pinball")
    ks = sorted(a.k.unique())
    k_all = max(ks)
    for k in [16, 64]:
        for base in ["ridge", "random_forest"]:
            g = r.xs(k, level="k")
            rows.append({"hypothesis": "H5", "comparison": f"r: TabPFN - {NAMES[base]}, k = {k}",
                         **bootstrap_diff(g.tabpfn, g[base], 0.9875)})
    for k in [16, 64]:
        for base in ["ridge", "random_forest"]:
            g = pin.xs(k, level="k")
            rows.append({"hypothesis": "H6", "comparison": f"pinball: {NAMES[base]} - TabPFN, k = {k}",
                         **bootstrap_diff(g[base], g.tabpfn, 0.9875)})
    g = pin.xs(k_all, level="k")
    for base in ["ridge", "random_forest"]:
        rows.append({"hypothesis": "H6b", "comparison": f"pinball: {NAMES[base]} - TabPFN, all discovery lines",
                     **bootstrap_diff(g[base], g.tabpfn)})
    b = df[df.experiment == "B_prospective"].pivot_table(index="target", columns="method", values="pinball")
    for base in ["ridge", "random_forest"]:
        rows.append({"hypothesis": "H6b", "comparison": f"pinball: {NAMES[base]} - TabPFN, prospective",
                     **bootstrap_diff(b[base], b.tabpfn)})
    c = similar_table(df)
    rows.append({"hypothesis": "H7", "comparison": "r: TabPFN similar - TabPFN random context",
                 **bootstrap_diff(c[("similar", "tabpfn")], c[("random", "tabpfn")])})
    for other, label in [("mean", "k-nearest-neighbours"), ("ridge", "ridge, similar context"),
                         ("random_forest", "random forest, similar context")]:
        rows.append({"hypothesis": "H7 (descriptive)", "comparison": f"r: TabPFN similar - {label}",
                     **bootstrap_diff(c[("similar", "tabpfn")], c[("similar", other)])})
    return pd.DataFrame(rows)


def similar_table(df: pd.DataFrame) -> pd.DataFrame:
    c = df[df.experiment == "C_similar_examples"]
    return c.groupby(["target", "arm", "method"]).r.mean().unstack(["arm", "method"])


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df.experiment == "A_curve"]
    out = _per_gene(a, "coverage80").groupby("k").mean()
    b = df[df.experiment == "B_prospective"].groupby("method").coverage80.mean()
    out.loc["prospective"] = b
    out = out[["tabpfn", "random_forest", "ridge", "mean"]]
    out.columns = [NAMES[c] for c in out.columns]
    return out


def control_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.experiment == "D_shuffled_labels"]
    return d.groupby("method").r.agg(mean_r="mean", mean_abs_r=lambda x: x.abs().mean()).rename(index=NAMES)


def timing_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["experiment", "method"]).seconds.median().unstack("method")
    t.columns = [NAMES[c] for c in t.columns]
    return t

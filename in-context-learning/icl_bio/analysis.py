"""The pre-registered comparisons for Part 2, computed from runs/part2/results.jsonl (no DepMap data needed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "part2"
NAMES = {"tabpfn": "TabPFN (in-context)", "ridge": "ridge regression", "random_forest": "random forest", "mean": "context mean"}


def load() -> pd.DataFrame:
    df = pd.read_json(RUNS / "results.jsonl", lines=True)
    return df.drop_duplicates("key", keep="last")


def prereg() -> dict:
    return json.loads((RUNS / "preregistration.json").read_text())


def per_gene(df: pd.DataFrame, by: list[str], metric: str = "r") -> pd.DataFrame:
    """Average over repeats, so each gene (or gene x lineage) counts once."""
    return df.groupby(by + ["target", "method"])[metric].mean().unstack("method")


def bootstrap_diff(a: pd.Series, b: pd.Series, level: float = 0.95, n: int = 5000, seed: int = 0) -> dict:
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"units": len(d), "mean difference": round(float(d.mean()), 3), "interval level": f"{level:.2%}",
            "interval": (round(float(lo), 3), round(float(hi), 3)), "excludes 0": bool(lo > 0 or hi < 0)}


def curve_table(df: pd.DataFrame, gene_set: str = "primary", metric: str = "r") -> pd.DataFrame:
    d = df[(df.experiment == "E1_curve") & (df.set == gene_set)]
    g = per_gene(d, ["k"], metric)
    out = g.groupby("k").mean()[[m for m in ["tabpfn", "random_forest", "ridge", "mean"] if m in g and (metric != "r" or m != "mean")]]
    out.columns = [NAMES[c] for c in out.columns]
    out.insert(0, "genes", g.groupby("k").size())
    return out


def hypotheses(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    e1 = per_gene(df[(df.experiment == "E1_curve") & (df.set == "primary")], ["k"])
    for k, level, h in [(16, 0.9875, "H1"), (64, 0.9875, "H1"), (626, 0.95, "H2")]:
        if k not in e1.index.get_level_values("k"):
            continue
        g = e1.xs(k, level="k")
        for base in ["ridge", "random_forest"]:
            rows.append({"hypothesis": h, "comparison": f"TabPFN - {NAMES[base]}, k = {k}", **bootstrap_diff(g.tabpfn, g[base], level)})
    e2 = df[(df.experiment == "E2_lineage")]
    if len(e2):
        g = per_gene(e2, ["lineage", "k"])
        if {0, 8} <= set(g.index.get_level_values("k")):
            gain = g.xs(8, level="k") - g.xs(0, level="k")
            rows.append({"hypothesis": "H3", "comparison": "gain from 8 in-lineage examples: TabPFN - ridge",
                         **bootstrap_diff(gain.tabpfn, gain.ridge)})
            for m in ["tabpfn", "ridge", "random_forest"]:
                rows.append({"hypothesis": "H3 (detail)", "comparison": f"gain from 8 in-lineage examples, {NAMES[m]}",
                             **bootstrap_diff(gain[m], gain[m] * 0)})
    e3 = df[df.experiment == "E3_prospective"]
    if len(e3):
        g = e3.pivot_table(index="target", columns="method", values="r")
        for base in ["ridge", "random_forest"]:
            rows.append({"hypothesis": "H4", "comparison": f"prospective: TabPFN - {NAMES[base]}", **bootstrap_diff(g.tabpfn, g[base])})
    return pd.DataFrame(rows)


def lineage_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.experiment == "E2_lineage"]
    g = per_gene(d, ["lineage", "k"]).groupby(["lineage", "k"]).mean()[["tabpfn", "random_forest", "ridge"]]
    g.columns = [NAMES[c] for c in g.columns]
    return g


def prospective_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.experiment == "E3_prospective"]
    t = d.pivot_table(index=["set", "target"], columns="method", values="r")[["tabpfn", "random_forest", "ridge"]]
    t.columns = [NAMES[c] for c in t.columns]
    t["test lines"] = d.groupby(["set", "target"]).n_test.first()
    t["Avana-Cas12 agreement r"] = d.groupby(["set", "target"]).agreement_r.first()
    return t

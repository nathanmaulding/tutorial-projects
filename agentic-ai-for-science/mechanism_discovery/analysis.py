"""Turn benchmark results into the tables used in the notebooks and write-up."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .tasks import NOVEL

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "runs" / "part1" / "results.jsonl"
ARM_ORDER = ["exhaustive", "single_shot", "single_shot_fit", "agent_no_retrieval", "agent_neutral", "agent_full"]
ARM_NAMES = {"exhaustive": "Exhaustive search (no LLM)", "single_shot": "Single LLM call, no tools",
             "agent_no_retrieval": "Agent, no retrieval", "agent_full": "Agent, full",
             "single_shot_fit": "Single LLM call picks the mechanism, harness fits it",
             "agent_neutral": "Agent, full, with neutral cards (no pointers to new mechanisms)"}


def load(path=RESULTS) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).open() if l.strip()]
    df = pd.DataFrame(rows)
    # Keep the latest record per run key (re-runs append).
    df = df.drop_duplicates("run_key", keep="last").reset_index(drop=True)
    df["method"] = df["arm"] + np.where(df["model"] == "none", "", " · " + df["model"].str.replace("claude-", ""))
    df["kind"] = np.select([df.task_id.str.startswith("hard_novel"), df.truth.eq("pd_no_effect"),
                            df.task_id.eq("hard_unidentifiable_km"), df.tier.eq("hard")],
                           ["out-of-library", "null (no effect)", "identifiability", "nonlinear PK"], "standard")
    df["failure"] = df.apply(failure_mode, axis=1)
    return df


def failure_mode(r) -> str:
    if r["success"]:
        return "success"
    if r.get("status") not in ("done",) or r.get("label") is None:
        return f"no result ({r.get('status')})"
    if not r["mechanism_ok"]:
        if r["truth"] == "pd_no_effect":
            return "false discovery (claimed an effect)"
        if r["truth"] in NOVEL:
            return "forced a library model onto a new mechanism"
        if r["task_id"] == "hard_unidentifiable_km":
            return "over-claimed unidentifiable parameters"
        return "wrong mechanism"
    return "right mechanism, poor prediction"


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("method")
    out = pd.DataFrame({
        "runs": g.size(),
        "success": g["success"].mean(),
        "standard tier": df[df.tier == "standard"].groupby("method")["success"].mean(),
        "hard tier": df[df.tier == "hard"].groupby("method")["success"].mean(),
        "out-of-library": df[df.kind == "out-of-library"].groupby("method")["success"].mean(),
        "median held-out error": g["heldout_nrmse"].median(),
        "cost per run ($)": g["usd"].mean(),
        "total cost ($)": g["usd"].sum(),
        "mean steps": g["steps"].mean(),
    })
    order = [m for a in ARM_ORDER for m in out.index if m.split(" · ")[0] == a]
    return out.loc[order]


def by_task(df: pd.DataFrame) -> pd.DataFrame:
    t = df.pivot_table(index=["tier", "task_id"], columns="method", values="success", aggfunc="mean")
    order = [m for a in ARM_ORDER for m in t.columns if m.split(" · ")[0] == a]
    return t[order]


def failures(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="failure", columns="method", values="run_key", aggfunc="count", fill_value=0)


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Do repeated runs of the same agent on the same data reach the same verdict?"""
    llm = df[df.arm != "exhaustive"]
    sizes = llm.groupby(["method", "task_id", "seed"])["run_key"].transform("size")
    grp = llm[sizes >= 2].groupby(["method", "task_id", "seed"])  # only (task, seed) pairs run 2+ times
    agree = grp["success"].agg(lambda s: s.nunique() == 1)
    same_label = grp["label"].agg(lambda s: s.nunique(dropna=False) == 1)
    return pd.DataFrame({"same verdict across repeats": agree.groupby("method").mean(),
                         "same mechanism across repeats": same_label.groupby("method").mean(),
                         "task-seed pairs run twice": grp.size().groupby("method").size()})


def calibration(df: pd.DataFrame, bins=(0, 0.6, 0.8, 0.9, 1.01)) -> pd.DataFrame:
    """When the agent says it is X% confident, is it right X% of the time?"""
    d = df[df.confidence.notna() & df.arm.str.startswith("agent")].copy()
    d["confidence"] = d["confidence"].astype(float).clip(0, 1)
    d["bin"] = pd.cut(d.confidence, bins, right=False)
    t = d.groupby(["method", "bin"], observed=True).agg(runs=("success", "size"),
                                                        stated=("confidence", "mean"),
                                                        actual=("success", "mean"))
    return t


def brier(df: pd.DataFrame) -> pd.Series:
    d = df[df.confidence.notna() & df.arm.str.startswith("agent")]
    return d.groupby("method").apply(lambda g: float(np.mean((g.confidence.astype(float) - g.success) ** 2)))


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a success rate (Wilson score interval)."""
    if n == 0:
        return (np.nan, np.nan)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


# ---------------------------------------------------------------- uncertainty and stricter scoring
RESCORED = ROOT / "runs" / "part1" / "rescored.jsonl"


def with_strict(df: pd.DataFrame, path=RESCORED) -> pd.DataFrame:
    """Add the stricter mechanism check from mechanism_discovery.rescore (probe-grid equivalence)."""
    if not Path(path).exists():
        return df.assign(mechanism_strict=np.nan, success_strict=np.nan)
    s = pd.read_json(path, lines=True)
    return df.merge(s[["run_key", "mechanism_strict", "success_strict", "probe_nrmse", "probe_threshold"]],
                    on="run_key", how="left")


def cluster_ci(d: pd.DataFrame, col: str = "success", n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval that resamples DATASETS (task x seed), not runs.

    Repeat runs on the same dataset are not independent evidence, and neither are two runs of one
    task: resampling whole datasets keeps that dependence, so the interval is honest about how
    little data a few dozen benchmark instances really are."""
    if d.empty:
        return (np.nan, np.nan)
    g = d.groupby(["task_id", "seed"])[col].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(float), g["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    rates = s[idx].sum(1) / n[idx].sum(1)
    return tuple(np.quantile(rates, [0.025, 0.975]))


def headline(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Success split into in-library and out-of-library, with dataset-level 95% intervals.

    Exhaustive search can only return library models, so it scores 0% out-of-library BY
    CONSTRUCTION. The fair comparison of search vs agent is the in-library column."""
    col = "success_strict" if strict and "success_strict" in df and df["success_strict"].notna().all() else "success"
    rows = []
    for m, g in df.groupby("method"):
        row = {"method": m, "runs": len(g), "datasets": g.groupby(["task_id", "seed"]).ngroups}
        for name, part in [("in-library", g[g.kind != "out-of-library"]), ("out-of-library", g[g.kind == "out-of-library"])]:
            k, n = int(part[col].sum()), len(part)
            lo, hi = cluster_ci(part, col)
            row[name] = f"{k}/{n} = {k / n:.0%}  [{lo:.0%}, {hi:.0%}]" if n else "-"
        rows.append(row)
    out = pd.DataFrame(rows).set_index("method")
    order = [m for a in ARM_ORDER for m in out.index if m.split(" · ")[0] == a]
    return out.loc[order]


def paired_difference(df: pd.DataFrame, a: str, b: str, kind: str | None = None, col: str = "success",
                      n_boot: int = 4000, seed: int = 0) -> dict:
    """Success-rate difference between two methods on the SAME datasets, with a dataset bootstrap."""
    d = df if kind is None else df[(df.kind == kind) if kind != "in-library" else (df.kind != "out-of-library")]
    p = d.pivot_table(index=["task_id", "seed"], columns="method", values=col, aggfunc="mean")[[a, b]].dropna()
    diff = (p[a] - p[b]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"a": a, "b": b, "subset": kind or "all", "datasets": len(diff), "difference": round(float(diff.mean()), 3),
            "ci95": (round(float(lo), 3), round(float(hi), 3))}

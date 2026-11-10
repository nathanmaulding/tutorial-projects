"""Turn benchmark results into the tables used in the notebooks and write-up."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .tasks import NOVEL

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "runs" / "part1" / "results.jsonl"
ARM_ORDER = ["exhaustive", "single_shot", "agent_no_retrieval", "agent_full"]
ARM_NAMES = {"exhaustive": "Exhaustive search (no LLM)", "single_shot": "Single LLM call, no tools",
             "agent_no_retrieval": "Agent, no retrieval", "agent_full": "Agent, full"}


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
    order = [m for a in ARM_ORDER for m in out.index if m.startswith(a)]
    return out.loc[order]


def by_task(df: pd.DataFrame) -> pd.DataFrame:
    t = df.pivot_table(index=["tier", "task_id"], columns="method", values="success", aggfunc="mean")
    order = [m for a in ARM_ORDER for m in t.columns if m.startswith(a)]
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

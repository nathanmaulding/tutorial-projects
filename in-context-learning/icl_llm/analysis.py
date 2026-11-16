"""Tables for notebook 02, built only from the recorded results in runs/part1 (no API key needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .claude import RUNS

MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
SHORT = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5"}


def _load(name: str) -> pd.DataFrame:
    frames = [pd.read_json(RUNS / f"{name}_{m}.jsonl", lines=True) for m in MODELS if (RUNS / f"{name}_{m}.jsonl").exists()]
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["model"].map(SHORT)
    return df


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def regression() -> pd.DataFrame:
    """Median absolute error by number of examples, for each model and the reference algorithms."""
    df = _load("regression")
    rows = []
    for (model, k), g in df.groupby(["model", "k"]):
        rng = np.random.default_rng(0)
        err = (g.pred - g.truth).abs().dropna().to_numpy()
        boots = np.median(err[rng.integers(0, len(err), (2000, len(err)))], axis=1) if len(err) else [np.nan]
        rows.append({"model": model, "k": k, "trials": len(g), "answered": g.pred.notna().mean(),
                     "model error": np.median(err), "95% interval": tuple(np.round(np.quantile(boots, [0.025, 0.975]), 2)),
                     "least squares": (g.least_squares - g.truth).abs().median(),
                     "nearest neighbour": (g.nearest_neighbour - g.truth).abs().median(),
                     "mean of the y's": (g.mean_of_ys - g.truth).abs().median()})
    return pd.DataFrame(rows)


def labels() -> pd.DataFrame:
    df = _load("labels")
    rows = []
    for (model, cond), g in df.groupby(["model", "condition"]):
        k = int(g.follows_demonstrations.sum())
        rows.append({"model": model, "labels in the examples": cond, "queries": len(g),
                     "follows the examples": k / len(g), "95% interval": tuple(np.round(wilson(k, len(g)), 2)),
                     "answers with the real sentiment word": g.matches_true_sentiment_word.mean()})
    return pd.DataFrame(rows)


def _clean(a: str) -> str:
    a = a.strip().lower()
    return a[len("label:"):].strip() if a.startswith("label:") else a


def imbalance() -> pd.DataFrame:
    df = _load("imbalance")
    df["answer"] = df.answer.map(_clean)
    df["pos"] = df.answer.str.startswith("positive")
    df["neg"] = df.answer.str.startswith("negative")
    order = ["14 positive : 2 negative", "balanced 8 : 8", "2 positive : 14 negative"]
    rows = []
    for (model, skew), g in df.groupby(["model", "skew"]):
        labelled = g[g.pos | g.neg]
        k = int(labelled.pos.sum())
        rows.append({"model": model, "examples": skew, "neutral queries": len(g),
                     "answered with a label": len(labelled) / len(g),
                     "share called positive (of those labelled)": k / len(labelled) if len(labelled) else np.nan,
                     "95% interval": tuple(np.round(wilson(k, len(labelled)), 2)),
                     "other answers": ", ".join(sorted(set(g.answer[~(g.pos | g.neg)].str.slice(0, 10))))[:60]})
    out = pd.DataFrame(rows)
    out["order"] = out.examples.map(order.index)
    return out.sort_values(["model", "order"]).drop(columns="order").reset_index(drop=True)

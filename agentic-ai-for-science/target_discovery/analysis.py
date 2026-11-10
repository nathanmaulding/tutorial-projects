"""Tables for the Part 2 notebook, built only from recorded results (no DepMap data needed)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "runs" / "part2" / "results.jsonl"


def load(path=RESULTS) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).open() if l.strip()]
    return pd.DataFrame(rows).drop_duplicates("run_key", keep="last").reset_index(drop=True)


def short_model(m: str) -> str:
    return "baseline pipeline" if m == "none" else m.replace("claude-", "")


def rediscovery(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.kind == "rediscovery"].copy()
    d["model"] = d.model.map(short_model)
    return d.pivot_table(index="name", columns="model", values="correct", aggfunc="mean")


def hypotheses(df: pd.DataFrame) -> pd.DataFrame:
    """Every pre-registered hypothesis from every agent session, with its replication outcome."""
    rows = []
    for r in df[df.kind.isin(["mission", "rediscovery"])].itertuples():
        for hid, h in (r.registered or {}).items():
            rep = (r.replications or {}).get(hid)
            hold = (rep or {}).get("held_out_broad_lines", {})
            sang = (rep or {}).get("independent_sanger_screens", {})
            rows.append({"run": r.run_key, "kind": r.kind, "mission": r.name, "world": r.world,
                         "model": short_model(r.model), "biomarker": h["biomarker"], "target": h["target"],
                         "novelty_claimed": h["novelty"], "n_altered_discovery": h["discovery"]["n_altered"],
                         "t_discovery": h["discovery"]["t"], "tested": rep is not None,
                         "replicated": bool(rep and rep["replicated"]),
                         "p_heldout": hold.get("p_one_sided"), "p_sanger": sang.get("p_one_sided"),
                         "n_altered_sanger": sang.get("n_altered"), "mechanism": h["mechanism"]})
    return pd.DataFrame(rows)


def null_world(df: pd.DataFrame) -> pd.DataFrame:
    """In null worlds every true link is destroyed: any replicated claim is a false discovery."""
    d = df[(df.world != "real") & df.kind.isin(["mission", "baseline"])].copy()
    d["model"] = d.model.map(short_model)
    out = d.groupby("model").agg(sessions=("run_key", "size"), hypotheses_tested=("n_registered", "sum"),
                                 false_discoveries=("n_replicated", "sum"))
    claimed = d[d.kind == "mission"].groupby("model")["n_claimed_replicated"].sum()
    out["claimed_replicated"] = claimed.reindex(out.index).fillna(0).astype(int)
    return out


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Replicated real-world mission hypotheses, merged across sessions (recurrence = robustness)."""
    h = hypotheses(df)
    h = h[(h.world == "real") & (h.kind == "mission") & h.replicated]
    g = h.groupby(["biomarker", "target"])
    out = pd.DataFrame({
        "found_in_sessions": g.size(),
        "models": g["model"].agg(lambda s: ", ".join(sorted(set(s)))),
        "novelty_claimed": g["novelty_claimed"].agg(lambda s: s.mode().iat[0]),
        "n_altered_discovery": g["n_altered_discovery"].first(),
        "t_discovery": g["t_discovery"].first(),
        "p_heldout": g["p_heldout"].first(), "p_sanger": g["p_sanger"].first(),
        "n_altered_sanger": g["n_altered_sanger"].first(),
    })
    return out.sort_values(["found_in_sessions", "t_discovery"], ascending=[False, True])


def costs(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.model != "none"].copy()
    d["model"] = d.model.map(short_model)
    return d.groupby(["kind", "model"]).agg(sessions=("run_key", "size"), mean_usd=("usd", "mean"),
                                            total_usd=("usd", "sum"), mean_steps=("steps", "mean"))


def memory_control(df: pd.DataFrame) -> pd.DataFrame:
    """Positive controls with gene names hidden vs shown, same reduced tool set (no PubMed, no Python).

    If accuracy holds when every gene is a random code, the agent is reading the data; if it falls,
    part of 'rediscovery' was recall of famous biology."""
    d = df[df.kind.isin(["rediscovery_reduced", "rediscovery_anon"])].copy()
    if d.empty:
        return pd.DataFrame()
    d["condition"] = d.kind.map({"rediscovery_reduced": "names shown", "rediscovery_anon": "names hidden"})
    d["submitted"] = d["top_target"].notna()
    d["correct"] = d["correct"].astype(bool)
    return d.pivot_table(index="name", columns="condition", values="correct", aggfunc="mean")


def memory_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.kind.isin(["rediscovery_reduced", "rediscovery_anon"])].copy()
    if d.empty:
        return pd.DataFrame()
    d["condition"] = d.kind.map({"rediscovery_reduced": "names shown", "rediscovery_anon": "names hidden"})
    d["correct"] = d["correct"].astype(bool)
    d["no_answer"] = d["top_target"].isna()
    return d.groupby("condition").agg(sessions=("run_key", "size"), correct=("correct", "mean"),
                                      no_answer=("no_answer", "mean"), mean_usd=("usd", "mean"))

"""Stricter mechanism scoring, applied offline to recorded runs (no API calls).

The original score accepts any self-written ("custom") model as the right mechanism and lets the
held-out prediction decide. Held-out conditions sit close to the training data, so a structurally
wrong model can still pass. Example: on the cooperative-inhibition task, a 3-parameter model with no
Hill coefficient predicted the held-out points well enough.

This module asks a second, separate question: is the submitted model the same mechanism as the
truth? It replays every submitted model, with its fitted parameters, on a PROBE grid that reaches
well beyond both the training and held-out conditions (lower and higher substrate, 3x the highest
inhibitor or dose). Two models that agree there are, for practical purposes, the same mechanism.

    mechanism_strict = submitted label is the true library model
                       OR (custom / other label AND probe error <= probe threshold)
    probe threshold  = max(0.05, 2 x the probe error of the TRUE model refit to the same data)

The null-effect and identifiability tasks keep their original rules (their "right answer" is a
statement about what the data can support, not an equation). Probe error is the mean over probe
curves (one per inhibitor level or dose) of each curve's RMSE divided by its mean true value, so a
low-dose curve counts as much as a high-dose one.

Run:  python -m mechanism_discovery.rescore      (writes runs/part1/rescored.jsonl)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .agent_tools import MechanismSession
from .fitting import fit
from .library import LIBRARY
from .scoring import library_names_for
from .tasks import TASKS_BY_ID, TaskInstance, make_instance, truth_fn, y_col

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "runs" / "part1" / "results.jsonl"
OUT = ROOT / "runs" / "part1" / "rescored.jsonl"
KEEP_ORIGINAL_RULE = {"hard_unidentifiable_km", "hard_null_effect"}


# ------------------------------------------------------------------ probe grid
def probe_grid(inst: TaskInstance) -> pd.DataFrame:
    t = inst.task
    tr, ho = t.train, t.heldout
    if t.domain == "enzyme":
        s_all = np.r_[tr.S, ho.S]
        S = np.geomspace(s_all.min() / 4, s_all.max() * 4, 14)
        i_all = sorted(set(np.r_[tr.I, ho.I]))
        I = sorted(set(i_all + ([3 * max(i_all)] if max(i_all) > 0 else [])))
        return pd.DataFrame([(s, i) for i in I for s in S], columns=["S", "I"])
    d_all = sorted(set(np.r_[tr.dose, ho.dose]))
    doses = sorted(set([d_all[0] / 4] + d_all + [3 * d_all[-1]]))
    times = sorted(set(tr.t) | set(ho.t))
    return pd.DataFrame([(d, x) for d in doses for x in times], columns=["dose", "t"])


def probe_error(pred, truth, grid: pd.DataFrame) -> float:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if pred.shape != truth.shape or not np.all(np.isfinite(pred)):
        return float("inf")
    key = grid["I"] if "I" in grid else grid["dose"]
    errs = []
    for k in pd.unique(key):
        m = (key == k).to_numpy()
        scale = np.mean(np.abs(truth[m])) or 1.0
        errs.append(np.sqrt(np.mean((pred[m] - truth[m]) ** 2)) / scale)
    return float(np.mean(errs))


_ORACLE: dict = {}


def oracle_probe(inst: TaskInstance) -> tuple[float, float]:
    """Probe error of the true model refit to this dataset, and the resulting threshold."""
    if inst.id not in _ORACLE:
        t, f = inst.task, truth_fn(inst.task)
        grid = probe_grid(inst)
        xdata = inst.data if "I" in inst.data.columns or t.domain != "enzyme" else inst.data.assign(I=0.0)
        res = fit(f, list(t.params), t.params, xdata, y_col(t),
                  error_model="proportional" if t.noise[0] == "proportional" else "additive")
        err = probe_error(f(grid, res["params"]), f(grid, t.params), grid)
        _ORACLE[inst.id] = (err, max(0.05, 2 * err))
    return _ORACLE[inst.id]


# ------------------------------------------------------------------ recover each submitted model
def _events(trace_path: str) -> list[dict]:
    p = ROOT / trace_path
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def submitted_model(row: pd.Series, inst: TaskInstance) -> tuple[str | None, dict | None, str | None]:
    """(label, fitted params, custom code or None) for one recorded run."""
    if row["arm"] == "exhaustive":  # deterministic: redo the search to recover the fitted parameters
        df = inst.data if not (inst.task.domain == "enzyme" and "I" not in inst.data) else inst.data.assign(I=0.0)
        from .run import auto_init
        em = "proportional" if inst.task.domain == "pk" else "additive"
        best = None
        for name in library_names_for(inst):
            try:
                r = fit(LIBRARY[name].predict, LIBRARY[name].params, auto_init(name, df, inst), df,
                        y_col(inst.task), em)
            except Exception:  # noqa: BLE001
                continue
            if best is None or r["bic"] < best[1]["bic"]:
                best = (name, r)
        return best[0], best[1]["params"], None
    ev = _events(row["trace"]) if isinstance(row["trace"], str) else []
    if row["arm"] == "single_shot":
        for e in ev:
            if e.get("kind") == "llm_response":
                text = "".join(b.get("text", "") for b in e["content"] if b.get("type") == "text")
                m = re.search(r"\{.*\}", text, re.S)
                try:
                    ans = json.loads(m.group(0))
                    return (ans["mechanism_label"], {k: float(v) for k, v in ans.get("params", {}).items()},
                            ans.get("custom_code"))
                except Exception:  # noqa: BLE001
                    return None, None, None
        return None, None, None
    # agent arms: map fit ids to the fit_model calls that created them, then find the submission
    fits, sub = {}, None
    for e in ev:
        if e.get("kind") != "tool_result" or e.get("is_error"):
            continue
        if e["name"] == "fit_model":
            m = re.search(r'"fit_id":\s*"(fit\d+)"', e["output"]) or re.search(r"'fit_id':\s*'(fit\d+)'", e["output"])
            if m:
                fits[m.group(1)] = e["input"]
        elif e["name"] == "submit_model":
            sub = e["input"]
    if sub is None or sub.get("fit_id") not in fits:
        return row.get("label"), row.get("submitted_params"), None
    spec = fits[sub["fit_id"]]
    code = spec.get("custom_code") if spec.get("model") == "custom" else None
    return sub["mechanism_label"], row.get("submitted_params"), code


def predict(inst: TaskInstance, label, params, code, grid: pd.DataFrame):
    if code:
        return MechanismSession(inst).run_custom("predict", code, params=params,
                                                 x_csv=grid.to_csv(index=False))["pred"]
    if label in LIBRARY:
        return LIBRARY[label].predict(grid, params)
    return None


# ------------------------------------------------------------------ rescore
def rescore_row(row: pd.Series) -> dict:
    inst = make_instance(TASKS_BY_ID[row["task_id"]], int(row["seed"]))
    out = {"run_key": row["run_key"], "probe_nrmse": None, "probe_threshold": None}
    if not row.get("label") or row.get("status") != "done":
        return {**out, "mechanism_strict": False, "success_strict": False, "strict_rule": "no submission"}
    if row["task_id"] in KEEP_ORIGINAL_RULE:
        ok = bool(row["mechanism_ok"])
        return {**out, "mechanism_strict": ok, "success_strict": bool(ok and row["heldout_pass"]),
                "strict_rule": "original rule"}
    grid = probe_grid(inst)
    truth = truth_fn(inst.task)(grid, inst.task.params)
    _, thr = oracle_probe(inst)
    label, params, code = submitted_model(row, inst)
    err = float("inf")
    try:
        if params:
            err = probe_error(predict(inst, label, params, code, grid), truth, grid)
    except Exception:  # noqa: BLE001 - a model that cannot run on new conditions fails the probe
        pass
    exact = row["label"] == inst.task.truth
    ok = bool(exact or err <= thr)
    return {**out, "probe_nrmse": round(err, 4) if np.isfinite(err) else None, "probe_threshold": round(thr, 4),
            "mechanism_strict": ok, "success_strict": bool(ok and row["heldout_pass"]),
            "strict_rule": "exact true label" if exact else "probe"}


def main():
    from .analysis import load
    df = load()
    rows = [rescore_row(r) for _, r in df.iterrows()]
    with OUT.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    s = pd.DataFrame(rows).merge(df[["run_key", "method", "kind", "success"]], on="run_key")
    print(s.groupby(["method"])[["success", "success_strict"]].mean().round(3))
    print(s.groupby(["kind", "method"])[["success", "success_strict"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()

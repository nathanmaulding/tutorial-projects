"""Scoring a submitted model against the hidden truth.

Two questions, both must be yes for success:
  1. Mechanism: did it identify the right mechanism (or, for out-of-library mechanisms, build a new
     one; for the null task, conclude there is no effect; for the identifiability task, use the
     reduced model or flag the parameters the data cannot determine)?
  2. Prediction: does the fitted model predict the hidden held-out conditions within the task's
     threshold (2x the error of the true model refit to the same data, between 5% and 15%)?
"""
from __future__ import annotations

import numpy as np

from .fitting import predict_nrmse
from .library import LIBRARY
from .tasks import NOVEL, TaskInstance


def mechanism_ok(inst: TaskInstance, label: str, n_params: int, nonident: list[str]) -> bool:
    t = inst.task
    if t.id == "hard_unidentifiable_km":
        return label == "linear_rate" or (label in ("mm", "custom") and (n_params == 1 or bool(nonident)))
    if t.truth == "pd_no_effect":
        return label == "pd_no_effect" or (label == "custom" and n_params == 1)
    if t.truth in NOVEL:
        return label == "custom"
    # Library mechanism: the exact label, or an equivalent model written by hand (judged by prediction).
    return label in (t.truth, *t.accepted, "custom")


def score(inst: TaskInstance, label: str | None, heldout_pred, n_params: int = 0,
          nonident: list[str] | None = None) -> dict:
    if label is None:
        return {"label": None, "mechanism_ok": False, "label_exact": False, "heldout_nrmse": None,
                "heldout_pass": False, "success": False, "threshold": inst.threshold}
    err = predict_nrmse(heldout_pred, inst.heldout_truth) if heldout_pred is not None else float("inf")
    mech = mechanism_ok(inst, label, n_params, nonident or [])
    passed = bool(np.isfinite(err) and err <= inst.threshold)
    exact = label == inst.task.truth or label in inst.task.accepted
    return {"label": label, "mechanism_ok": mech, "label_exact": exact,
            "heldout_nrmse": round(err, 4) if np.isfinite(err) else None,
            "heldout_pass": passed, "success": bool(mech and passed), "threshold": round(inst.threshold, 4),
            "oracle_nrmse": round(inst.oracle_nrmse, 4)}


def library_names_for(inst: TaskInstance) -> list[str]:
    t = inst.task
    return [n for n, m in LIBRARY.items()
            if m.domain == t.domain and not (m.route and t.route and m.route != t.route)]

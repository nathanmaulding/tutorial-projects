"""Part 2, study 2: a fair replication of the failed few-shot hypothesis, and two tests of what in-context
learners are supposed to be good at (predictive uncertainty, and choosing similar examples).

Study 1's primary hypothesis (H1) failed. Study 2 was written AFTER study 1's results were known, so it does not
re-test study 1's genes: it uses the NEXT 20 genes by the same selection rule (never examined), TabPFN's default
8 ensemble members (study 1 used 2 to save CPU), and it is frozen and fingerprinted before any model is fitted.

    python -m icl_bio.study2 freeze        # write runs/part2/study2/preregistration.json (once)
    python -m icl_bio.study2 all           # A replication + uncertainty, B prospective, C similar examples, D control

Results append to runs/part2/study2/results.jsonl; the run is resumable.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd

from icl_bio import study as s1

RUNS = s1.RUNS / "study2"
PREREG = RUNS / "preregistration.json"
RESULTS = RUNS / "results.jsonl"

N_TARGETS = 20
K_FEW = [16, 64]
K_ALL = 626                      # all discovery lines with expression (618)
REPS = 3
TABPFN_SETTINGS = {"n_estimators": 8, "random_state": 0, "device": "cpu"}   # TabPFN's default ensemble size
QUANTILES = [round(q, 2) for q in np.arange(0.05, 0.951, 0.05)]           # 19 levels
CV_FOLDS = 5
K_RETRIEVE = 64
N_RETRIEVAL_TEST = 50
N_SIMILARITY_GENES = 2000
METHODS = ["mean", "ridge", "random_forest", "tabpfn"]


# ------------------------------------------------------------------ pre-registration
def study2_targets() -> list[str]:
    """Study 1's rule (selective dependencies ranked by SD on discovery lines), ranks 21 onward, skipping any gene
    study 1 used; the first 20."""
    d = s1.load("discovery")
    frac, mean, sd = (d < -0.5).mean(), d.mean(), d.std()
    keep = (frac >= 0.05) & (frac <= 0.5) & (mean > -0.8)
    ranked = sd[keep].sort_values(ascending=False).index.tolist()
    used = set(s1.prereg()["primary_targets"]) | set(s1.SECONDARY_TARGETS)
    return [g for g in ranked[20:] if g not in used][:N_TARGETS]


def freeze():
    if PREREG.exists():
        raise SystemExit("study 2 pre-registration already written; it is never edited")
    RUNS.mkdir(parents=True, exist_ok=True)
    body = {
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "statement": "Written before any model in study 2 was fitted. Study 1's results WERE known when this was written "
                     "(its H1 failed), so study 2 uses genes study 1 never examined. H6 and H7 were chosen because published "
                     "work suggests they are TabPFN's strengths (full predictive distributions; benefiting from similar "
                     "context examples); that is stated here so the choice is visible.",
        "study1_preregistration_sha256": (s1.RUNS / "preregistration.sha256").read_text().split()[0],
        "targets": study2_targets(),
        "target_rule": "study 1's rule (5-50% of discovery lines below -0.5, mean > -0.8, ranked by SD), ranks 21+, "
                       "excluding study 1's primary and secondary genes; first 20",
        "data": "as study 1: train pool = 24Q4 Avana discovery lines, test = 24Q4 holdout lines; lines without expression "
                "excluded (study 1 clarification 1); features = top-50 |r| expression genes on the context lines only",
        "methods": {
            "tabpfn": {"weights": "TabPFN v2 regressor", **TABPFN_SETTINGS,
                       "uncertainty": f"its predictive distribution's quantiles at {QUANTILES}"},
            "ridge": "as study 1; uncertainty = prediction + empirical quantiles of its 5-fold cross-validated residuals "
                     "on the context lines",
            "random_forest": "as study 1; uncertainty = prediction + empirical quantiles of its 5-fold cross-validated "
                             "residuals on the context lines",
            "mean": "context mean; uncertainty = empirical quantiles of the context labels (the no-features reference)"},
        "metrics": {"r": "Pearson r, predicted vs measured, over the test lines (as study 1)",
                    "pinball": f"mean pinball (quantile) loss over the {len(QUANTILES)} quantile levels and the test lines; "
                               "a proper scoring rule for predictive distributions (lower is better)",
                    "coverage80": "share of test lines inside the 10%-90% predictive interval (descriptive)"},
        "experiments": {
            "A_curve": f"k in {K_FEW} random discovery lines ({REPS} repeats, seeds 0..{REPS-1}) and k = all discovery lines (1 run); "
                       "test = all holdout lines",
            "B_prospective": "context = all 24Q4 lines; test = 26Q3 Cas12 lines first screened after 24Q4 (study 1's E3 "
                             "construction), only targets with Avana-vs-Cas12 agreement r >= 0.5 on overlap lines",
            "C_similar_examples": f"{N_RETRIEVAL_TEST} holdout lines (fixed random sample, seed 0) per target. 'similar': for "
                                  f"each test line, context = the {K_RETRIEVE} discovery lines whose expression profile "
                                  f"(the {N_SIMILARITY_GENES} most variable genes on discovery lines, z-scored with discovery "
                                  f"mean and SD) correlates most with the test line's; one fit per test line. 'random': "
                                  f"{K_RETRIEVE} random discovery lines ({REPS} repeats). Features in both arms: top-50 "
                                  "selected on all discovery lines. r computed over the 50 lines. With 'similar' context, "
                                  "the mean method is k-nearest-neighbours.",
            "D_shuffled_labels": "control: k = 64 (seed 0) with the context labels randomly permuted (seed 0); every "
                                 "method should then score r near 0"},
        "hypotheses": {
            "H5 (primary; replication of study 1 H1)": "at k = 16 and 64, mean over targets of r(TabPFN) - r(baseline) > 0 for "
                "baseline = ridge and random forest; 4 comparisons, gene-level bootstrap 98.75% intervals (Bonferroni); "
                "each supported if its interval excludes 0 in TabPFN's favour.",
            "H6 (primary; uncertainty)": "at k = 16 and 64, mean over targets of pinball(baseline) - pinball(TabPFN) > 0 for "
                "baseline = ridge and random forest with cross-validated residual intervals; 4 comparisons, 98.75% intervals; "
                "each supported if its interval excludes 0 in TabPFN's favour.",
            "H6b (secondary)": "the same pinball comparison at k = all discovery lines and in the prospective test; 95% intervals.",
            "H7 (secondary; similar examples)": "r(TabPFN, similar context) - r(TabPFN, random context) > 0 over targets; "
                "95% interval. Reported alongside: similar-context TabPFN vs k-nearest-neighbours and vs similar-context "
                "ridge and random forest (descriptive).",
            "controls": "D: mean |r| below 0.1 for every method (descriptive check that the methods use the labels).",
        },
        "predictions": {
            "H5": "Given study 1: TabPFN ahead of ridge at k = 16; no clear difference from random forest.",
            "H6": "TabPFN ahead of both baselines, most clearly at k = 16 (cross-validated residuals from 13 lines give "
                  "crude intervals; skewed dependency distributions don't suit a constant-shape interval).",
            "H6b": "Smaller TabPFN advantage with all lines; uncertain under the distribution shift of the prospective test.",
            "H7": "Similar examples help TabPFN.",
        },
        "reporting": "every hypothesis is reported whatever the outcome, in the notebook and README, beside study 1.",
    }
    text = json.dumps(body, indent=1)
    PREREG.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (RUNS / "preregistration.sha256").write_text(digest + "  preregistration.json\n")
    print("wrote", PREREG, "\nsha256", digest, "\ntargets:", body["targets"])


def prereg() -> dict:
    text = PREREG.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == (RUNS / "preregistration.sha256").read_text().split()[0], \
        "study 2 pre-registration was edited"
    return json.loads(text)


# ------------------------------------------------------------------ models with predictive distributions
def _point_model(method: str):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if method == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=s1.RIDGE_ALPHAS))
    if method == "random_forest":
        return RandomForestRegressor(**s1.RF_SETTINGS)
    raise ValueError(method)


def fit_predict_dist(method: str, Xc: np.ndarray, yc: np.ndarray, Xt: np.ndarray, quantiles: bool = True):
    """Returns (point prediction, quantile matrix [levels x test lines] or None)."""
    q = np.array(QUANTILES)
    if method == "mean":
        point = np.full(len(Xt), yc.mean())
        return point, (np.tile(np.quantile(yc, q)[:, None], (1, len(Xt))) if quantiles else None)
    if method == "tabpfn":
        from tabpfn import TabPFNRegressor
        m = TabPFNRegressor(model_path=str(s1.WEIGHTS), ignore_pretraining_limits=True, **TABPFN_SETTINGS).fit(Xc, yc)
        if not quantiles:
            return m.predict(Xt), None
        out = m.predict(Xt, output_type="full", quantiles=list(QUANTILES))
        return np.asarray(out["mean"]), np.asarray(out["quantiles"])
    point = _point_model(method).fit(Xc, yc).predict(Xt)
    if not quantiles:
        return point, None
    from sklearn.model_selection import KFold, cross_val_predict
    cv = KFold(n_splits=min(CV_FOLDS, len(yc)), shuffle=True, random_state=0)
    resid = yc - cross_val_predict(_point_model(method), Xc, yc, cv=cv)
    return point, point[None, :] + np.quantile(resid, q)[:, None]


def pinball(qmat: np.ndarray, y: np.ndarray) -> float:
    q = np.array(QUANTILES)[:, None]
    d = y[None, :] - qmat
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage80(qmat: np.ndarray, y: np.ndarray) -> float:
    lo, hi = qmat[QUANTILES.index(0.1)], qmat[QUANTILES.index(0.9)]
    return float(np.mean((y >= lo) & (y <= hi)))


# ------------------------------------------------------------------ bookkeeping
def done_keys() -> set[str]:
    return {json.loads(l)["key"] for l in RESULTS.open()} if RESULTS.exists() else set()


def record(rec: dict):
    RUNS.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run_cell(key: str, base: dict, Xc, yc, Xt, yt, have: set[str], quantiles: bool = True):
    for m in METHODS:
        k = f"{key}|{m}"
        if k in have:
            continue
        t0 = time.time()
        point, qmat = fit_predict_dist(m, Xc, yc, Xt, quantiles)
        rec = {"key": k, **base, "method": m, "n_context": len(yc), "n_test": len(yt),
               **s1.score(point, yt, float(yc.mean())), "seconds": round(time.time() - t0, 1)}
        if qmat is not None:
            rec.update(pinball=pinball(qmat, yt), coverage80=coverage80(qmat, yt))
        record(rec)
        print(f"{k:55s} r={rec['r']:.3f} pinball={rec.get('pinball', float('nan')):.4f} {rec['seconds']}s", flush=True)


def _xy(E, y_ctx, y_test, target, feats=None):
    feats = feats or s1.select_features(E, y_ctx, target)
    return (E.loc[y_ctx.index, feats].to_numpy(float), y_ctx.to_numpy(float),
            E.loc[y_test.index, feats].to_numpy(float), y_test.to_numpy(float))


# ------------------------------------------------------------------ experiments
def curve():
    p, have = prereg(), done_keys()
    disc, hold, E = s1.load("discovery"), s1.load("holdout"), s1.load("expression")
    for target in p["targets"]:
        y_all, y_test = s1.with_expression(disc[target].dropna()), s1.with_expression(hold[target].dropna())
        for k in K_FEW + [K_ALL]:
            for rep in range(REPS if k < len(y_all) else 1):
                ctx = y_all.sample(n=min(k, len(y_all)), random_state=rep)
                base = {"experiment": "A_curve", "target": target, "k": k, "rep": rep}
                run_cell(f"A|{target}|k{k}|r{rep}", base, *_xy(E, ctx, y_test, target), have)


def shuffled():
    p, have = prereg(), done_keys()
    disc, hold, E = s1.load("discovery"), s1.load("holdout"), s1.load("expression")
    for target in p["targets"]:
        y_all, y_test = s1.with_expression(disc[target].dropna()), s1.with_expression(hold[target].dropna())
        ctx = y_all.sample(n=64, random_state=0)
        ctx = pd.Series(np.random.default_rng(0).permutation(ctx.to_numpy()), index=ctx.index)
        base = {"experiment": "D_shuffled_labels", "target": target, "k": 64, "rep": 0}
        run_cell(f"D|{target}|k64|r0", base, *_xy(E, ctx, y_test, target), have, quantiles=False)


def prospective():
    p, have = prereg(), done_keys()
    d = s1.prospective_data(p["targets"])
    train = pd.concat([s1.load("discovery"), s1.load("holdout")])
    elig = [t for t in p["targets"] if d["agree"].get(t, -1) >= 0.5]
    (RUNS / "prospective_eligibility.json").write_text(json.dumps(
        {"models_screened_in_both": int(len(d["both"])), "new_test_lines": len(d["new_ids"]),
         "agreement_r": {t: round(v, 3) for t, v in d["agree"].items()}, "eligible": elig}, indent=1))
    for target in elig:
        y_ctx = s1.with_expression(train[target].dropna())
        y_test = d["cas12"].loc[d["new_ids"], target].dropna()
        feats = [f for f in s1.select_features(d["E24"], y_ctx, target) if f in d["E26"].columns]
        Xc, yc = d["E24"].loc[y_ctx.index, feats].to_numpy(float), y_ctx.to_numpy(float)
        Xt, yt = d["E26"].loc[y_test.index, feats].to_numpy(float), y_test.to_numpy(float)
        base = {"experiment": "B_prospective", "target": target, "agreement_r": round(d["agree"][target], 3)}
        run_cell(f"B|{target}", base, Xc, yc, Xt, yt, have)


def similarity_neighbours(test_ids: list[str], pool_ids: list[str]) -> dict[str, list[str]]:
    """Discovery lines ranked by expression-profile correlation with each test line (most similar first)."""
    E = s1.load("expression")
    Ed = E.loc[pool_ids]
    genes = Ed.var().sort_values(ascending=False).head(N_SIMILARITY_GENES).index
    mu, sd = Ed[genes].mean(), Ed[genes].std().replace(0, 1)
    Z = lambda ids: ((E.loc[ids, genes] - mu) / sd).to_numpy(float)
    zp, zt = Z(pool_ids), Z(test_ids)
    zp = (zp - zp.mean(1, keepdims=True)) / zp.std(1, keepdims=True)
    zt = (zt - zt.mean(1, keepdims=True)) / zt.std(1, keepdims=True)
    corr = zt @ zp.T / zp.shape[1]
    return {t: [pool_ids[j] for j in np.argsort(-corr[i])] for i, t in enumerate(test_ids)}


def similar():
    p, have = prereg(), done_keys()
    disc, hold, E = s1.load("discovery"), s1.load("holdout"), s1.load("expression")
    has_expr = E.index[E.notna().all(axis=1)]
    test_pool = sorted(set(hold.index) & set(has_expr))
    test_ids = list(pd.Series(test_pool).sample(n=N_RETRIEVAL_TEST, random_state=0))
    pool_ids = sorted(set(disc.index) & set(has_expr))
    nbrs = similarity_neighbours(test_ids, pool_ids)
    for target in p["targets"]:
        y_all = s1.with_expression(disc[target].dropna())
        y_test = hold.loc[test_ids, target].dropna()
        feats = s1.select_features(E, y_all, target)
        Xt = E.loc[y_test.index, feats].to_numpy(float)
        # random context: one fit per repeat predicts all test lines
        for rep in range(REPS):
            ctx = y_all.sample(n=K_RETRIEVE, random_state=rep)
            base = {"experiment": "C_similar_examples", "arm": "random", "target": target, "k": K_RETRIEVE, "rep": rep}
            run_cell(f"C|random|{target}|r{rep}", base, E.loc[ctx.index, feats].to_numpy(float), ctx.to_numpy(float),
                     Xt, y_test.to_numpy(float), have, quantiles=False)
        # similar context: one fit per test line
        for m in METHODS:
            k = f"C|similar|{target}|{m}"
            if k in have:
                continue
            t0, preds = time.time(), []
            for i, line in enumerate(y_test.index):
                ctx_ids = [j for j in nbrs[line] if j in y_all.index][:K_RETRIEVE]
                point, _ = fit_predict_dist(m, E.loc[ctx_ids, feats].to_numpy(float), y_all[ctx_ids].to_numpy(float),
                                            Xt[i:i + 1], quantiles=False)
                preds.append(float(point[0]))
            yt = y_test.to_numpy(float)
            rec = {"key": k, "experiment": "C_similar_examples", "arm": "similar", "target": target, "k": K_RETRIEVE,
                   "rep": 0, "method": m, "n_context": K_RETRIEVE, "n_test": len(yt),
                   **s1.score(np.array(preds), yt, float(y_all.mean())), "seconds": round(time.time() - t0, 1)}
            record(rec)
            print(f"{k:55s} r={rec['r']:.3f} {rec['seconds']}s", flush=True)


STEPS = {"curve": curve, "shuffled": shuffled, "prospective": prospective, "similar": similar}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["freeze", "all", *STEPS])
    a = ap.parse_args(argv)
    if a.step == "freeze":
        return freeze()
    for name in (STEPS if a.step == "all" else [a.step]):
        print(f"== {name} {dt.datetime.now():%H:%M}", flush=True)
        STEPS[name]()


if __name__ == "__main__":
    main()

"""Part 2: does an in-context learner (TabPFN) predict cancer gene dependencies from few examples?

TabPFN is a transformer pretrained on millions of SYNTHETIC tabular datasets. Given labelled rows in its
context (here: cell lines with expression features and a measured gene effect), it predicts new rows in one
forward pass, with no gradient updates. Because it never saw real data, it can't have memorised DepMap.

Steps (each is one CLI command; results append to runs/part2/results.jsonl and the run is resumable):

    python -m icl_bio.study freeze        # write the pre-registration (once, before any model is fitted)
    python -m icl_bio.study curve         # E1: learning curve, k = 16, 64, 256, 626 context lines
    python -m icl_bio.study lineage       # E2: adapting to a cancer type from 0 or 8 examples
    python -m icl_bio.study prospective   # E3: lines first screened after 24Q4 (DepMap 26Q3, Cas12)

Data: the DepMap 24Q4 cache built by agentic-ai-for-science (target_discovery.prep), and the 26Q3 files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")   # never contact Prior Labs' telemetry service

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "part2"
PREREG = RUNS / "preregistration.json"
RESULTS = RUNS / "results.jsonl"
# The 24Q4 cache from agentic-ai-for-science, converted to version-neutral .npz (numpy 1 and 2 can both read it)
CACHE = Path(os.environ.get("DEPMAP_CACHE", "~/Desktop/claude/data/depmap_24q4/cache_npz")).expanduser()
RELEASE_26Q3 = Path(os.environ.get("DEPMAP_26Q3", "~/Desktop/claude/data/depmap_26q3")).expanduser()
WEIGHTS = Path(os.environ.get("TABPFN_WEIGHTS", "~/Desktop/claude/data/tabpfn/tabpfn-v2-regressor.ckpt")).expanduser()

N_FEATURES = 50
K_CURVE = [16, 64, 256, 626]   # 626 = all discovery lines (618 after clarification 1)
REPS = 3
LINEAGES = ["Lung", "Lymphoid", "CNS/Brain", "Skin", "Esophagus/Stomach"]
K_LINEAGE = [0, 8]
SECONDARY_TARGETS = ["DNAJC19", "TTC7A", "NABP2", "AGPAT1", "RPP25L", "ATP6V0E1"]
TABPFN_SETTINGS = {"n_estimators": 2, "random_state": 0, "device": "cpu"}
RIDGE_ALPHAS = np.logspace(-2, 4, 25)
RF_SETTINGS = {"n_estimators": 300, "max_features": 0.33, "min_samples_leaf": 3, "random_state": 0, "n_jobs": 4}


_LOADED: dict = {}


def load(name: str) -> pd.DataFrame:
    if name not in _LOADED:
        z = np.load(CACHE / f"{name}.npz", allow_pickle=(name == "meta"))
        _LOADED[name] = pd.DataFrame(z["values"], index=z["index"], columns=z["columns"])
    return _LOADED[name]


# ------------------------------------------------------------------ pre-registration
def primary_targets() -> list[str]:
    """The rule, applied to DISCOVERY lines only: selective dependencies (5-50% of lines below -0.5,
    mean above -0.8 so not broadly essential), ranked by standard deviation; the top 20."""
    d = load("discovery")
    frac, mean, sd = (d < -0.5).mean(), d.mean(), d.std()
    keep = (frac >= 0.05) & (frac <= 0.5) & (mean > -0.8)
    return sd[keep].sort_values(ascending=False).head(20).index.tolist()


def freeze():
    if PREREG.exists():
        raise SystemExit("pre-registration already written; it is never edited")
    RUNS.mkdir(parents=True, exist_ok=True)
    body = {
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "statement": "Written before any model in this study was fitted to DepMap data.",
        "primary_targets": primary_targets(), "secondary_targets": SECONDARY_TARGETS,
        "data": {"train_pool": "24Q4 Broad Avana DISCOVERY split (626 lines)", "test": "24Q4 Avana HOLDOUT split (417 lines)",
                 "features": "24Q4 expression, log2(TPM+1); the 50 genes with the largest |Pearson r| with the target "
                             "in the CONTEXT lines only (never the test lines); the target gene itself excluded",
                 "prospective": "26Q3 Humagne-CD (Cas12) gene effects of models first screened after 24Q4 (same set as "
                                "agentic-ai-for-science amendment1); features from 26Q3 expression (default entry per model)"},
        "methods": {"tabpfn": {"weights": "TabPFN v2 regressor (Prior Labs License 1.1)", **TABPFN_SETTINGS},
                    "ridge": "StandardScaler + RidgeCV(alphas=logspace(-2,4,25))",
                    "random_forest": RF_SETTINGS,
                    "mean": "predict the context mean (reference for R^2)"},
        "metric": "Pearson r between predicted and measured gene effect on the test lines, per target gene "
                  "(averaged over repeats); secondary: out-of-sample R^2 vs the context mean",
        "experiments": {
            "E1_curve": f"k in {K_CURVE} context lines drawn at random from discovery ({REPS} repeats for k<626, seeds 0..{REPS-1}); test on all holdout lines",
            "E2_lineage": f"for each lineage in {LINEAGES}: context = discovery lines of OTHER lineages + k in {K_LINEAGE} discovery lines of the lineage "
                          f"({REPS} repeats for k>0); test = that lineage's holdout lines",
            "E3_prospective": "context = all 1043 24Q4 Avana lines; test = 26Q3 new Cas12 lines, only for targets whose Avana-vs-Cas12 "
                              "gene effects correlate r >= 0.5 across models screened in both libraries (decided from overlap lines, "
                              "not the test lines)"},
        "hypotheses": {
            "H1 (primary)": "Few-shot advantage: at k = 16 and k = 64, mean over the 20 primary targets of r(TabPFN) - r(baseline) > 0 "
                            "for baseline = ridge and = random forest. 4 comparisons; each tested with a gene-level bootstrap "
                            "98.75% interval (Bonferroni over the 4); supported if the interval excludes 0.",
            "H2": "At k = 626 (all discovery lines), TabPFN vs ridge and vs random forest: two-sided, reported with 95% intervals.",
            "H3": "Adapting to a cancer type: the gain from 0 to 8 in-lineage examples (mean over lineages and targets) is larger "
                  "for TabPFN than for ridge. Secondary; 95% interval.",
            "H4": "Prospective: on new Cas12 lines, TabPFN vs ridge and vs random forest over eligible targets; two-sided, 95% intervals; "
                  "reported whatever the number of eligible targets.",
        },
        "predictions": {"H1": "TabPFN ahead at k = 16 and 64 (its prior acts as regularisation when data are scarce).",
                        "H2": "Close to random forest at k = 626.",
                        "H3": "Uncertain.", "H4": "Same ordering as H2, with lower correlations overall (new lines, new CRISPR system)."},
    }
    text = json.dumps(body, indent=1)
    PREREG.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (RUNS / "preregistration.sha256").write_text(digest + "  preregistration.json\n")
    print("wrote", PREREG, "\nsha256", digest, "\ntargets:", body["primary_targets"])


def prereg() -> dict:
    text = PREREG.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == (RUNS / "preregistration.sha256").read_text().split()[0], \
        "pre-registration was edited"
    return json.loads(text)


# ------------------------------------------------------------------ models
_TABPFN = None


def tabpfn_model():
    global _TABPFN
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(model_path=str(WEIGHTS), ignore_pretraining_limits=True, **TABPFN_SETTINGS)


def fit_predict(method: str, Xc: np.ndarray, yc: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    if method == "mean":
        return np.full(len(Xt), yc.mean())
    if method == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)).fit(Xc, yc).predict(Xt)
    if method == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**RF_SETTINGS).fit(Xc, yc).predict(Xt)
    if method == "tabpfn":
        return tabpfn_model().fit(Xc, yc).predict(Xt)
    raise ValueError(method)


METHODS = ["mean", "ridge", "random_forest", "tabpfn"]


def with_expression(y: pd.Series) -> pd.Series:
    """Keep only cell lines that have RNA expression data (clarification 1: 8 discovery and 4 holdout lines have none)."""
    E = load("expression")
    has = E.index[E.notna().all(axis=1)]
    return y[y.index.isin(has)]


def select_features(E: pd.DataFrame, y: pd.Series, target: str, n: int = N_FEATURES) -> list[str]:
    """Top-n expression genes by |Pearson r| with the target, computed on the CONTEXT lines only."""
    X = E.loc[y.index].drop(columns=[target], errors="ignore")
    X = X.loc[:, X.std() > 1e-6]
    Xc, yc = X - X.mean(), y - y.mean()
    r = (Xc.T @ yc) / (np.sqrt((Xc ** 2).sum()) * np.sqrt((yc ** 2).sum()))
    return r.abs().sort_values(ascending=False).head(n).index.tolist()


def score(pred: np.ndarray, y: np.ndarray, ctx_mean: float) -> dict:
    r = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 1e-12 else float("nan")
    r2 = 1 - float(np.sum((y - pred) ** 2) / np.sum((y - ctx_mean) ** 2))
    return {"r": r, "r2_oos": r2}


# ------------------------------------------------------------------ bookkeeping
def done_keys() -> set[str]:
    return {json.loads(l)["key"] for l in RESULTS.open()} if RESULTS.exists() else set()


def record(rec: dict):
    RUNS.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run_cell(key: str, base: dict, E_ctx: pd.DataFrame, y_ctx: pd.Series, E_test: pd.DataFrame, y_test: pd.Series,
             target: str, have: set[str]):
    feats = select_features(E_ctx, y_ctx, target)
    Xc, Xt = E_ctx.loc[y_ctx.index, feats].to_numpy(float), E_test.loc[y_test.index, feats].to_numpy(float)
    for m in METHODS:
        k = f"{key}|{m}"
        if k in have:
            continue
        t0 = time.time()
        pred = fit_predict(m, Xc, y_ctx.to_numpy(float), Xt)
        rec = {"key": k, **base, "method": m, "n_context": len(y_ctx), "n_test": len(y_test),
               **score(pred, y_test.to_numpy(float), float(y_ctx.mean())), "seconds": round(time.time() - t0, 1)}
        record(rec)
        print(f"{k:60s} r={rec['r']:.3f} r2={rec['r2_oos']:.3f} {rec['seconds']}s", flush=True)


# ------------------------------------------------------------------ experiments
def curve():
    p = prereg()
    disc, hold, E = load("discovery"), load("holdout"), load("expression")
    have = done_keys()
    for target in p["primary_targets"] + p["secondary_targets"]:
        y_all, y_test = with_expression(disc[target].dropna()), with_expression(hold[target].dropna())
        for k in K_CURVE:
            for rep in range(REPS if k < len(y_all) else 1):
                ctx = y_all.sample(n=min(k, len(y_all)), random_state=rep)
                base = {"experiment": "E1_curve", "target": target, "k": k, "rep": rep,
                        "set": "primary" if target in p["primary_targets"] else "secondary"}
                run_cell(f"E1|{target}|k{k}|r{rep}", base, E, ctx, E, y_test, target, have)


def lineage():
    p = prereg()
    disc, hold, E, meta = load("discovery"), load("holdout"), load("expression"), load("meta")
    lin_d = meta.OncotreeLineage.reindex(disc.index)
    lin_h = meta.OncotreeLineage.reindex(hold.index)
    have = done_keys()
    for L in LINEAGES:
        for target in p["primary_targets"]:
            y = with_expression(disc[target].dropna())
            others, own = y[lin_d.reindex(y.index) != L], y[lin_d.reindex(y.index) == L]
            y_test = with_expression(hold[target][lin_h == L].dropna())
            for k in K_LINEAGE:
                for rep in range(REPS if k > 0 else 1):
                    ctx = pd.concat([others, own.sample(n=k, random_state=rep)]) if k else others
                    base = {"experiment": "E2_lineage", "lineage": L, "target": target, "k": k, "rep": rep, "set": "primary"}
                    run_cell(f"E2|{L}|{target}|k{k}|r{rep}", base, E, ctx, E, y_test, target, have)


def prospective_data(targets: list[str]) -> dict:
    """The 26Q3 prospective test set, exactly as agentic-ai-for-science/target_discovery/prospective.py amendment1:
    Cas12 (Humagne-CD) gene effects of models first screened after 24Q4, 26Q3 expression, and the per-target
    Avana-vs-Cas12 agreement measured on models screened in both libraries."""
    import re
    gs = lambda c: re.sub(r"\s*\(\d+\)$", "", c)
    hdr = pd.read_csv(RELEASE_26Q3 / "ScreenGeneEffect.csv", nrows=0).columns
    keep = {c for c in hdr if gs(c) in set(targets)}
    ge = pd.read_csv(RELEASE_26Q3 / "ScreenGeneEffect.csv", index_col=0,
                     usecols=lambda c: c in keep or c == "" or c.startswith("Unnamed"))
    ge.columns = [gs(c) for c in ge.columns]
    smap = pd.read_csv(RELEASE_26Q3 / "CRISPRScreenMap.csv").set_index("ScreenID")["ModelID"]
    lib = ge.index.str.extract(r"\.([A-Z]+)\d+$")[0].values
    per = {L: ge[lib == L].groupby(smap.reindex(ge[lib == L].index).values).mean() for L in ("AV", "CD")}
    both = per["AV"].index.intersection(per["CD"].index)
    agree = {t: float(per["AV"].loc[both, t].corr(per["CD"].loc[both, t])) for t in targets
             if t in per["AV"] and t in per["CD"]}
    # models first screened after 24Q4: not in this project's 24Q4 sets and no screen of any library in 24Q4
    old_ids = set(load("meta").index)
    ids24 = pd.read_csv(CACHE.parent / "ScreenGeneEffect.csv", usecols=[0]).iloc[:, 0]
    smap24 = pd.read_csv(CACHE.parent / "CRISPRScreenMap.csv").set_index("ScreenID")["ModelID"]
    old_ids |= set(smap24.reindex(ids24).dropna())
    new_ids = sorted(set(per["CD"].index) - old_ids)
    # features: 26Q3 expression, default entry per model
    expr_path = RELEASE_26Q3 / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    E24 = load("expression")
    cols = pd.read_csv(expr_path, nrows=0).columns
    want = {c for c in cols if gs(c) in set(E24.columns)}
    E26 = pd.read_csv(expr_path, usecols=lambda c: c in want or c in ("ModelID", "IsDefaultEntryForModel"))
    E26 = E26[E26.IsDefaultEntryForModel == "Yes"].drop(columns="IsDefaultEntryForModel").set_index("ModelID")
    E26.columns = [gs(c) for c in E26.columns]
    E26 = E26.loc[~E26.index.duplicated(), ~E26.columns.duplicated()]
    new_ids = [i for i in new_ids if i in E26.index]
    return {"cas12": per["CD"], "new_ids": new_ids, "E24": E24, "E26": E26, "agree": agree, "both": both}


def prospective():
    """E3. Builds the 26Q3 test set with prospective_data (same construction as agentic-ai-for-science amendment1)."""
    p = prereg()
    targets = p["primary_targets"] + p["secondary_targets"]
    d = prospective_data(targets)
    per, new_ids, E24, E26, agree, both = {"CD": d["cas12"]}, d["new_ids"], d["E24"], d["E26"], d["agree"], d["both"]
    train = pd.concat([load("discovery"), load("holdout")])
    have = done_keys()
    (RUNS / "prospective_eligibility.json").write_text(json.dumps(
        {"models_screened_in_both": int(len(both)), "new_test_lines": len(new_ids),
         "agreement_r": {t: round(v, 3) for t, v in agree.items()},
         "eligible": [t for t, v in agree.items() if v >= 0.5]}, indent=1))
    for target in targets:
        if agree.get(target, -1) < 0.5:
            continue
        y_ctx = with_expression(train[target].dropna())
        y_test = per["CD"].loc[new_ids, target].dropna()
        feats = select_features(E24, y_ctx, target)
        feats = [f for f in feats if f in E26.columns]
        Xc, Xt = E24.loc[y_ctx.index, feats].to_numpy(float), E26.loc[y_test.index, feats].to_numpy(float)
        for m in METHODS:
            k = f"E3|{target}|{m}"
            if k in have:
                continue
            t0 = time.time()
            pred = fit_predict(m, Xc, y_ctx.to_numpy(float), Xt)
            rec = {"key": k, "experiment": "E3_prospective", "target": target, "method": m, "agreement_r": round(agree[target], 3),
                   "set": "primary" if target in p["primary_targets"] else "secondary", "n_context": len(y_ctx),
                   "n_test": len(y_test), **score(pred, y_test.to_numpy(float), float(y_ctx.mean())),
                   "seconds": round(time.time() - t0, 1)}
            record(rec)
            print(f"{k:50s} r={rec['r']:.3f} {rec['seconds']}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["freeze", "curve", "lineage", "prospective"])
    a = ap.parse_args(argv)
    {"freeze": freeze, "curve": curve, "lineage": lineage, "prospective": prospective}[a.step]()


if __name__ == "__main__":
    main()

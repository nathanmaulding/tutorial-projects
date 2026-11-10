"""Prospective test: re-test frozen hypotheses on cell lines first screened AFTER DepMap 24Q4.

Everything in Part 2 used one data release. The held-out lines and Sanger screens were never shown
to the agents, but they answered every session's replication questions, and we looked at them
while building the project. Cell lines that DepMap screened after 24Q4 are different: nobody in
this project has seen them. A hypothesis that holds there was tested prospectively.

Step 1 (done before any newer data were downloaded): `python -m target_discovery.prospective freeze`
writes runs/part2/prospective/preregistration.json: the hypotheses, the exact rules, and the
predictions, plus its SHA-256. The rules can't be changed after the data are seen; any deviation
must be reported as a deviation.

Step 2: `python -m target_discovery.prospective test --release-dir <folder>` runs the frozen tests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "runs" / "part2" / "prospective"
PREREG = DIR / "preregistration.json"

LEAD = ("LOSS_ATP6V0E2", "ATP6V0E1")
CANDIDATES = [("LOSS_NABP1", "NABP2"), ("LOSS_AGPAT2", "AGPAT1"), ("LOSS_PUDP", "ITPA"),
              ("LOSS_TTC7B", "TTC7A"), ("LOSS_DNAJC15", "DNAJC19"), ("LOSS_RPP25", "RPP25L"),
              ("LOSS_FAM126A", "HYCC2"), ("LOSS_DMXL2", "DMXL1"), ("LOSS_CHPT1", "CEPT1"),
              ("LOSS_ACSL4", "FOXA1")]
CONFOUND_CHECK = [("LOSS_CRY1", "MEF2B")]
CONTROLS = [("LOSS_MTAP", "PRMT5"), ("LOSS_MTAP", "WDR77"), ("MUT_SMARCA4", "SMARCA2"), ("MUT_ARID1A", "ARID1B"),
            ("LOSS_MLH1", "WRN"), ("MUT_STAG2", "STAG1"), ("MUT_KEAP1", "NFE2L2"), ("~MUT_TP53", "MDM2"),
            ("~MUT_TP53", "PPM1D"), ("HOT_BRAF", "BRAF"), ("HOT_BRAF", "MAP2K1"), ("HOT_BRAF", "DUSP4"),
            ("HOT_NRAS", "NRAS"), ("HOT_NRAS", "SHOC2"), ("HOT_KRAS", "KRAS"), ("HOT_KRAS", "RAF1"),
            ("HOT_PIK3CA", "PIK3CA"), ("MUT_RB1", "E2F3"), ("MUT_RB1", "DYRK1A"), ("MUT_BRCA1", "FANCM")]

RULES = {
    "new_lines": "ModelIDs with a Broad Avana (library AV) CRISPR screen in the newer release that appear in NONE of "
                 "the 24Q4 sets used here (discovery, held-out, Sanger; 1162 ModelIDs listed in cache meta.pkl).",
    "gene_effect": "Chronos gene effect for the new lines, from the newer release's Avana screen-level gene effects "
                   "(ScreenGeneEffect, library AV, as in prep.py) or CRISPRGeneEffect if screen-level files are absent. "
                   "Genes missing in the newer release make that pair untestable.",
    "biomarkers": "Same definitions as 24Q4 prep.py: MUT = damaging-mutation matrix > 0; HOT = hotspot matrix > 0; "
                  "LOSS = log2(TPM+1) < 1 in the newer release's protein-coding expression matrix (the 'normally "
                  "expressed' gene set is the one already fixed by 24Q4). ~MUT = no damaging mutation.",
    "lineage": "OncotreeLineage from the newer release's Model.csv.",
    "test": "Lineage-adjusted regression of gene effect on the biomarker within the new lines only (the same "
            "Frisch-Waugh-Lovell test as stats.py); one-sided (altered lines MORE dependent).",
    "verdicts": "replicated: p < 0.01 and beta < 0. not replicated: testable but p >= 0.01 or beta >= 0. "
                "untestable: fewer than 3 altered or 3 unaltered new lines with both data types, or no within-lineage "
                "variation.",
    "family": "Benjamini-Hochberg at 5% across all testable pairs in this file, reported alongside per-pair verdicts.",
    "secondary": "Sign test: among testable LEAD+CANDIDATES pairs, the fraction with beta < 0, against 50%.",
    "calibration": "CONTROLS are established biology. Their replication rate in the new lines estimates the power of "
                   "this test, so a candidate's failure is read against it.",
    "sanity": "For ModelIDs present in BOTH 24Q4 and the newer release, report agreement of each biomarker's calls "
              "between releases; a biomarker with < 90% agreement is flagged (definition drift), not dropped.",
}
PREDICTIONS = {
    "lead": "LOSS_ATP6V0E2 -> ATP6V0E1 replicates if testable.",
    "candidates": "Most testable candidates have beta < 0; fewer replicate at p < 0.01 than controls do, because "
                  "there are few new altered lines.",
    "confound_check": "LOSS_CRY1 -> MEF2B replicates only if the new lines include B-cell lymphomas with CRY1 loss; "
                      "if it replicates, that still does not separate biomarker from lymphoma subtype.",
}


def freeze():
    if PREREG.exists():
        raise SystemExit(f"{PREREG} already exists: a pre-registration is written once and never edited.")
    DIR.mkdir(parents=True, exist_ok=True)
    body = {"written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "statement": "Written before any DepMap release newer than 24Q4 was downloaded or viewed by this project.",
            "lead": [LEAD], "candidates": CANDIDATES, "confound_check": CONFOUND_CHECK, "controls": CONTROLS,
            "rules": RULES, "predictions": PREDICTIONS}
    text = json.dumps(body, indent=1)
    PREREG.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (DIR / "preregistration.sha256").write_text(digest + "  preregistration.json\n")
    print(f"wrote {PREREG}\nsha256 {digest}")


# ------------------------------------------------------------------ step 2: the frozen test
def _gene_cols(path: Path, genes: set[str]) -> list[str]:
    from .prep import gene_symbol
    header = pd.read_csv(path, nrows=0).columns
    return [c for c in header if gene_symbol(c) in genes]


def _omics(path: Path, genes: set[str]) -> pd.DataFrame:
    """Default entry per model, selected gene columns, indexed by ModelID, columns = gene symbols."""
    from .prep import gene_symbol
    meta_cols = ["ModelID", "IsDefaultEntryForModel"]
    cols = _gene_cols(path, genes)
    df = pd.read_csv(path, usecols=meta_cols + cols)
    df = df[df.IsDefaultEntryForModel == "Yes"].drop(columns="IsDefaultEntryForModel").set_index("ModelID")
    df.columns = [gene_symbol(c) for c in df.columns]
    return df.loc[~df.index.duplicated(), ~df.columns.duplicated()].astype(float)


def one_sided_test(x: np.ndarray, y: np.ndarray, lineage: np.ndarray) -> dict:
    """Lineage-adjusted regression of y on binary x (Frisch-Waugh-Lovell, as in stats.py); one-sided."""
    from scipy import stats as sps
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, lin = x[ok], y[ok], lineage[ok]
    n, n_alt = len(x), int(x.sum())
    out = {"n_lines": n, "n_altered": n_alt}
    if n_alt < 3 or n - n_alt < 3:
        return {**out, "verdict": "untestable", "reason": "fewer than 3 altered or unaltered lines"}
    codes = pd.factorize(lin)[0]
    k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k).astype(float)
    xr = x - (np.bincount(codes, weights=x, minlength=k) / cnt)[codes]
    yr = y - (np.bincount(codes, weights=y, minlength=k) / cnt)[codes]
    sxx = float(xr @ xr)
    dof = n - k - 1
    if sxx <= 1e-9 or dof < 1:
        return {**out, "verdict": "untestable", "reason": "no variation within lineages"}
    beta = float(xr @ yr) / sxx
    resid = max(float(yr @ yr) - beta * float(xr @ yr), 1e-12) / dof
    t = beta / np.sqrt(resid / sxx)
    p = float(sps.t.cdf(t, dof))
    verdict = "replicated" if (p < 0.01 and beta < 0) else "not replicated"
    return {**out, "beta_adj": round(beta, 4), "t": round(float(t), 2), "p_one_sided": float(f"{p:.3g}"),
            "mean_effect_altered": round(float(y[x == 1].mean()), 3),
            "mean_effect_other": round(float(y[x == 0].mean()), 3), "verdict": verdict}


def screened_in_24q4() -> set[str]:
    """Every ModelID with a screen of any library in the 24Q4 ScreenGeneEffect file."""
    from .prep import DEFAULT_DATA
    d = Path(DEFAULT_DATA)
    ids = pd.read_csv(d / "ScreenGeneEffect.csv", usecols=[0]).iloc[:, 0]
    smap = pd.read_csv(d / "CRISPRScreenMap.csv").set_index("ScreenID")["ModelID"]
    return set(smap.reindex(ids).dropna())


SCENARIOS = {
    # name: (library, also exclude everything screened in 24Q4?, output file, which frozen document)
    "original": ("AV", False, "results.json", "preregistration.json"),
    "amendment1": ("CD", True, "results_amendment1.json", "amendment1.json"),
    "amendment1_sensitivity": ("CD", False, "results_amendment1_sensitivity.json", "amendment1.json"),
}


def test(release_dir: Path, scenario: str = "original"):
    from .prep import gene_symbol
    from .stats import bh, load
    pre = json.loads(PREREG.read_text())
    digest = hashlib.sha256(PREREG.read_text().encode()).hexdigest()
    assert digest == (DIR / "preregistration.sha256").read_text().split()[0], "pre-registration was edited"
    library, exclude_old_screens, out_name, doc = SCENARIOS[scenario]
    doc_text = (DIR / doc).read_text()
    assert hashlib.sha256(doc_text.encode()).hexdigest() == \
        (DIR / doc.replace(".json", ".sha256")).read_text().split()[0], f"{doc} was edited"
    groups = [("lead", h) for h in pre["lead"]] + [("candidate", h) for h in pre["candidates"]] + \
             [("confound_check", h) for h in pre["confound_check"]] + [("control", h) for h in pre["controls"]]
    targets = {t for _, (_, t) in groups}
    bio_genes = {b.lstrip("~").split("_", 1)[1] for _, (b, _) in groups}

    # new lines: screens of this library for models in none of the 24Q4 sets
    old_ids = set(load("meta").index)
    if exclude_old_screens:
        old_ids |= screened_in_24q4()
    smap = pd.read_csv(release_dir / "CRISPRScreenMap.csv").set_index("ScreenID")["ModelID"]
    cols = _gene_cols(release_dir / "ScreenGeneEffect.csv", targets)
    keep = set(cols)
    ge = pd.read_csv(release_dir / "ScreenGeneEffect.csv", index_col=0,
                     usecols=lambda c: c in keep or c == "" or c.startswith("Unnamed"))
    ge.columns = [gene_symbol(c) for c in ge.columns]
    ge = ge.loc[:, ~ge.columns.duplicated()]
    lib = ge.index.str.extract(r"\.([A-Z]+)\d+$")[0].values
    av = ge[(lib == library) & ge.index.isin(smap.index)]
    av = av.groupby(smap.reindex(av.index).values).mean()
    new_ids = sorted(set(av.index) - old_ids)
    model = pd.read_csv(release_dir / "Model.csv").set_index("ModelID")
    lineage = model["OncotreeLineage"].reindex(new_ids).fillna("Unknown").to_numpy()

    # biomarkers for ALL models in the release (new lines for the test, overlap lines for the sanity check)
    expr = _omics(release_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", bio_genes)
    dmg = _omics(release_dir / "OmicsSomaticMutationsMatrixDamaging.csv", bio_genes)
    hot = _omics(release_dir / "OmicsSomaticMutationsMatrixHotspot.csv", bio_genes)

    def call(b: str, ids) -> pd.Series:
        neg, kind, g = b.startswith("~"), b.lstrip("~").split("_", 1)[0], b.lstrip("~").split("_", 1)[1]
        src = {"LOSS": expr, "MUT": dmg, "HOT": hot}[kind]
        if g not in src.columns:
            return pd.Series(np.nan, index=ids)
        v = src[g].reindex(ids)
        x = (v < 1).astype(float) if kind == "LOSS" else (v > 0).astype(float)
        x[v.isna()] = np.nan
        return (1 - x) if neg else x

    old_feats = load("features")
    rows = []
    for group, (b, t) in groups:
        rec = {"group": group, "biomarker": b, "target": t}
        if t not in av.columns:
            rows.append({**rec, "verdict": "untestable", "reason": "target not in the newer release"})
            continue
        x = call(b, new_ids).to_numpy()
        y = av[t].reindex(new_ids).to_numpy(float)
        res = one_sided_test(x, y, lineage)
        both = sorted(set(old_feats.index) & set(expr.index if b.lstrip("~").startswith("LOSS") else dmg.index))
        old_x = old_feats[b.lstrip("~")].reindex(both)
        old_x = (1 - old_x) if b.startswith("~") else old_x
        new_x = call(b, both)
        m = old_x.notna() & new_x.notna()
        agree = float((old_x[m] == new_x[m]).mean()) if m.any() else np.nan
        rows.append({**rec, **res, "call_agreement_with_24Q4": round(agree, 3) if np.isfinite(agree) else None,
                     "drift_flag": bool(np.isfinite(agree) and agree < 0.9)})
    r = pd.DataFrame(rows)
    testable = r["verdict"] != "untestable"
    r["q_bh"] = np.nan
    if testable.any():
        r.loc[testable, "q_bh"] = bh(r.loc[testable, "p_one_sided"].to_numpy())
    r["survives_bh"] = r["q_bh"] < 0.05
    cand = r[r.group.isin(["lead", "candidate"]) & testable]
    from scipy.stats import binomtest
    k_neg = int((cand.beta_adj < 0).sum()) if "beta_adj" in cand else 0
    sign = {"testable": int(len(cand)), "negative_beta": k_neg,
            "p_sign_test": float(binomtest(k_neg, len(cand), 0.5, alternative="greater").pvalue) if len(cand) else None}
    ctrl = r[(r.group == "control") & testable]
    summary = {"release": release_dir.name, "scenario": scenario, "library": library,
               "preregistration_sha256": digest, "document": doc,
               "new_lines": len(new_ids), "new_lines_by_lineage": pd.Series(lineage).value_counts().head(12).to_dict(),
               "control_power": f"{int((ctrl.verdict == 'replicated').sum())}/{len(ctrl)} testable controls replicated",
               "sign_test": sign, "verdicts": r.groupby(["group", "verdict"]).size().unstack(fill_value=0).to_dict(orient="index")}
    (DIR / out_name).write_text(json.dumps({"summary": summary, "tests": r.to_dict(orient="records")},
                                                 indent=1, default=float))
    pd.set_option("display.width", 250)
    print(json.dumps(summary, indent=1, default=str))
    show = [c for c in ["group", "biomarker", "target", "n_altered", "beta_adj", "p_one_sided", "q_bh", "verdict",
                        "call_agreement_with_24Q4", "drift_flag"] if c in r]
    print(r[show].to_string())


def rnai_test(rnai_dir: Path, release_dir: Path):
    """The frozen RNAi test (rnai_preregistration.json): same hypotheses and rules, DEMETER2 RNAi gene effects."""
    from .prep import gene_symbol
    from .stats import bh, load
    from scipy.stats import binomtest
    doc = (DIR / "rnai_preregistration.json").read_text()
    digest = hashlib.sha256(doc.encode()).hexdigest()
    assert digest == (DIR / "rnai_preregistration.sha256").read_text().split()[0], "RNAi pre-registration was edited"
    pre = json.loads(PREREG.read_text())
    groups = [("lead", h) for h in pre["lead"]] + [("candidate", h) for h in pre["candidates"]] + \
             [("confound_check", h) for h in pre["confound_check"]] + [("control", h) for h in pre["controls"]]
    targets = {t for _, (_, t) in groups}
    bio_genes = {b.lstrip("~").split("_", 1)[1] for _, (b, _) in groups}
    hdr = pd.read_csv(rnai_dir / "gene_effect.csv", nrows=0).columns
    keep = {c for c in hdr if gene_symbol(c) in targets}
    ge = pd.read_csv(rnai_dir / "gene_effect.csv", index_col=0,
                     usecols=lambda c: c in keep or c == "" or c.startswith("Unnamed"))
    ge.columns = [gene_symbol(c) for c in ge.columns]
    ge = ge.loc[:, ~ge.columns.duplicated()]
    model = pd.read_csv(release_dir / "Model.csv").set_index("ModelID")
    expr = _omics(release_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", bio_genes)
    dmg = _omics(release_dir / "OmicsSomaticMutationsMatrixDamaging.csv", bio_genes)
    hot = _omics(release_dir / "OmicsSomaticMutationsMatrixHotspot.csv", bio_genes)

    def call(b: str, ids) -> pd.Series:
        neg, kind, g = b.startswith("~"), b.lstrip("~").split("_", 1)[0], b.lstrip("~").split("_", 1)[1]
        src = {"LOSS": expr, "MUT": dmg, "HOT": hot}[kind]
        if g not in src.columns:
            return pd.Series(np.nan, index=ids)
        v = src[g].reindex(ids)
        x = (v < 1).astype(float) if kind == "LOSS" else (v > 0).astype(float)
        x[v.isna()] = np.nan
        return (1 - x) if neg else x

    disc = set(load("discovery").index)
    out = {}
    for scenario, ids in (("primary", sorted(ge.index)), ("secondary_not_in_discovery", sorted(set(ge.index) - disc))):
        lineage = model["OncotreeLineage"].reindex(ids).fillna("Unknown").to_numpy()
        rows = []
        for group, (b, t) in groups:
            rec = {"group": group, "biomarker": b, "target": t}
            if t not in ge.columns:
                rows.append({**rec, "verdict": "untestable", "reason": "target not in the RNAi data"})
                continue
            rows.append({**rec, **one_sided_test(call(b, ids).to_numpy(), ge[t].reindex(ids).to_numpy(float), lineage)})
        r = pd.DataFrame(rows)
        testable = r["verdict"] != "untestable"
        r["q_bh"] = np.nan
        if testable.any():
            r.loc[testable, "q_bh"] = bh(r.loc[testable, "p_one_sided"].to_numpy())
        cand = r[r.group.isin(["lead", "candidate"]) & testable]
        k_neg = int((cand.beta_adj < 0).sum()) if "beta_adj" in cand else 0
        ctrl = r[(r.group == "control") & testable]
        out[scenario] = {"summary": {
            "lines": len(ids), "control_power": f"{int((ctrl.verdict == 'replicated').sum())}/{len(ctrl)} testable controls replicated",
            "sign_test": {"testable": int(len(cand)), "negative_beta": k_neg,
                          "p_sign_test": float(binomtest(k_neg, len(cand), 0.5, alternative="greater").pvalue) if len(cand) else None}},
            "tests": r.to_dict(orient="records")}
        print(f"\n#### RNAi {scenario}: {len(ids)} lines; {out[scenario]['summary']}")
        show = [c for c in ["group", "biomarker", "target", "n_altered", "beta_adj", "p_one_sided", "q_bh", "verdict"] if c in r]
        print(r[show].to_string())
    (DIR / "results_rnai.json").write_text(json.dumps({"preregistration_sha256": digest, **out}, indent=1, default=float))


def library_agreement(release_dir: Path):
    """EXPLORATORY (not pre-registered; added after seeing the lead's null result): for models screened with
    both Avana and Cas12 (CD), do the two libraries agree about each target gene? A target the Cas12
    library cannot measure can't be tested there."""
    from .prep import gene_symbol
    pre = json.loads(PREREG.read_text())
    targets = {t for grp in ("lead", "candidates", "confound_check", "controls") for _, t in pre[grp]}
    hdr = pd.read_csv(release_dir / "ScreenGeneEffect.csv", nrows=0).columns
    keep = {c for c in hdr if gene_symbol(c) in targets}
    ge = pd.read_csv(release_dir / "ScreenGeneEffect.csv", index_col=0,
                     usecols=lambda c: c in keep or c == "" or c.startswith("Unnamed"))
    ge.columns = [gene_symbol(c) for c in ge.columns]
    smap = pd.read_csv(release_dir / "CRISPRScreenMap.csv").set_index("ScreenID")["ModelID"]
    lib = ge.index.str.extract(r"\.([A-Z]+)\d+$")[0].values
    per = {L: ge[lib == L].groupby(smap.reindex(ge[lib == L].index).values).mean() for L in ("AV", "CD")}
    both = per["AV"].index.intersection(per["CD"].index)
    rows = []
    for g in sorted(targets & set(per["AV"].columns) & set(per["CD"].columns)):
        a, c = per["AV"].loc[both, g], per["CD"].loc[both, g]
        rows.append({"gene": g, "r_avana_vs_cas12": round(float(a.corr(c)), 2),
                     "sd_avana": round(float(per["AV"][g].std()), 2), "sd_cas12": round(float(per["CD"][g].std()), 2),
                     "frac_dependent_avana": round(float((per["AV"][g] < -0.5).mean()), 3),
                     "frac_dependent_cas12": round(float((per["CD"][g] < -0.5).mean()), 3)})
    out = {"note": "exploratory, not pre-registered", "models_screened_in_both": int(len(both)), "genes": rows}
    (DIR / "library_agreement_exploratory.json").write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(rows).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["freeze", "test", "library-agreement", "rnai"])
    ap.add_argument("--rnai-dir", default="~/Desktop/claude/data/demeter2_rnai")
    ap.add_argument("--release-dir", default=None)
    ap.add_argument("--scenario", default="original", choices=list(SCENARIOS))
    a = ap.parse_args(argv)
    if a.step == "freeze":
        freeze()
    elif a.step == "rnai":
        rnai_test(Path(a.rnai_dir).expanduser(), Path(a.release_dir).expanduser())
    elif a.step == "library-agreement":
        library_agreement(Path(a.release_dir).expanduser())
    else:
        test(Path(a.release_dir).expanduser(), a.scenario)


if __name__ == "__main__":
    main()

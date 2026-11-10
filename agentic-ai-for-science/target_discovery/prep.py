"""Build the analysis-ready data once from the DepMap 24Q4 Public release (Figshare 27993248).

Three gene-effect matrices (cell lines x genes; more negative = cells depend more on the gene):
    discovery   Broad Avana screens on a random 60% of Avana-screened cell lines (stratified by lineage)
    holdout     Broad Avana screens on the other 40%: never shown to the agent
    sanger      Sanger Project Score screens (KY library), processed separately by Chronos: an
                independent lab and CRISPR library, used only for replication

Biomarker features per cell line (binary, "altered" = 1):
    MUT_<gene>   damaging (likely loss-of-function) mutation
    HOT_<gene>   hotspot (likely activating) mutation
    LOSS_<gene>  expression lost (log2 TPM+1 < 1) for a gene normally expressed (median >= 3)
    LIN_<name>   cancer lineage
Only features altered in at least MIN_ALTERED discovery lines are kept.

Usage:  python -m target_discovery.prep  [--data ~/Desktop/claude/data/depmap_24q4]
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path(os.environ.get("DEPMAP_DIR", Path.home() / "Desktop/claude/data/depmap_24q4"))
SPLIT_SEED = 2026
DISCOVERY_FRACTION = 0.6
MIN_ALTERED = 8


def gene_symbol(col: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", col)


def read_matrix(path: Path, index_col=0) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=index_col)
    df.columns = [gene_symbol(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df.astype("float32")


def lineage_split(models: pd.Series, lineage: pd.Series, seed=SPLIT_SEED, frac=DISCOVERY_FRACTION):
    rng = np.random.default_rng(seed)
    disc = []
    for _, ids in models.groupby(lineage.reindex(models).fillna("Unknown").values):
        ids = ids.sample(frac=1, random_state=int(rng.integers(1 << 31))).tolist()
        k = int(round(len(ids) * frac))
        disc += ids[:k]
    return set(disc)


def build(data: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    model = pd.read_csv(data / "Model.csv").set_index("ModelID")
    meta = model[["CellLineName", "OncotreeLineage", "OncotreePrimaryDisease"]].copy()
    meta["OncotreeLineage"] = meta["OncotreeLineage"].fillna("Unknown")

    print("reading ScreenGeneEffect (large)...")
    screens = read_matrix(data / "ScreenGeneEffect.csv")
    smap = pd.read_csv(data / "CRISPRScreenMap.csv").set_index("ScreenID")["ModelID"]
    lib = pd.Series(screens.index.str.extract(r"\.([A-Z]+)\d+$")[0].values, index=screens.index)
    screens = screens[screens.index.isin(smap.index)]

    def by_model(library):
        s = screens[lib.reindex(screens.index) == library]
        return s.groupby(smap.reindex(s.index).values).mean()

    avana, sanger = by_model("AV"), by_model("KY")
    genes = avana.columns[avana.notna().mean() > 0.9].intersection(sanger.columns[sanger.notna().mean() > 0.9])
    avana, sanger = avana[genes], sanger[genes]
    avana = avana[avana.index.isin(meta.index)]
    sanger = sanger[sanger.index.isin(meta.index)]

    disc_ids = lineage_split(pd.Series(avana.index, index=avana.index), meta["OncotreeLineage"])
    discovery = avana[avana.index.isin(disc_ids)]
    holdout = avana[~avana.index.isin(disc_ids)]

    print("reading expression and mutations...")
    expr = read_matrix(data / "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    dmg = read_matrix(data / "OmicsSomaticMutationsMatrixDamaging.csv")
    hot = read_matrix(data / "OmicsSomaticMutationsMatrixHotspot.csv")

    all_ids = sorted(set(avana.index) | set(sanger.index))
    feats = {}
    d_ids = list(discovery.index)
    for prefix, mat in (("MUT", dmg), ("HOT", hot)):
        m = (mat.reindex(all_ids) > 0).astype("float32")
        m[mat.reindex(all_ids).isna()] = np.nan
        counts = m.reindex(d_ids).sum()
        for g in counts[counts >= MIN_ALTERED].index:
            feats[f"{prefix}_{g}"] = m[g]
    e = expr.reindex(all_ids)
    normally = e.median() >= 3
    lost = (e.loc[:, normally] < 1).astype("float32")
    lost[e.loc[:, normally].isna()] = np.nan
    counts = lost.reindex(d_ids).sum()
    for g in counts[counts >= MIN_ALTERED].index:
        feats[f"LOSS_{g}"] = lost[g]
    lin = meta["OncotreeLineage"].reindex(all_ids)
    for name, n in lin.reindex(d_ids).value_counts().items():
        if n >= MIN_ALTERED:
            feats["LIN_" + re.sub(r"\W+", "_", name).strip("_")] = (lin == name).astype("float32")
    features = pd.DataFrame(feats, index=all_ids).astype("float32")

    expr_keep = e[[g for g in genes if g in e.columns]].astype("float32")
    for name, obj in [("discovery", discovery), ("holdout", holdout), ("sanger", sanger),
                      ("features", features), ("meta", meta.reindex(all_ids)), ("expression", expr_keep)]:
        obj.to_pickle(out / f"{name}.pkl")
    summary = {"genes": len(genes), "discovery_lines": len(discovery), "holdout_lines": len(holdout),
               "sanger_lines": len(sanger), "sanger_lines_not_in_discovery": int((~sanger.index.isin(disc_ids)).sum()),
               "features": features.shape[1],
               "by_type": features.columns.str.split("_").str[0].value_counts().to_dict()}
    pd.Series(summary, dtype=object).to_json(out / "summary.json")
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    a = ap.parse_args()
    data = Path(a.data).expanduser()
    build(data, data / "cache")


if __name__ == "__main__":
    main()

"""Association statistics between biomarkers and gene dependencies, adjusted for lineage.

Why lineage adjustment: many biomarkers are concentrated in particular cancer types, and many
dependencies are lineage-specific. Without adjustment, "biomarker X predicts dependency on Y" can
just mean "both are common in melanoma". We regress gene effect on the biomarker plus lineage
indicators, done efficiently by subtracting lineage means from both (the Frisch-Waugh-Lovell trick).

Sign convention: gene effect is negative when cells depend on a gene, so a NEGATIVE beta (and t)
means altered cell lines are MORE dependent on the gene.

Worlds: `World()` is the real data. `World(null_seed=k)` is a null world in which every biomarker
label is shuffled among cell lines of the same lineage, consistently across all splits. Lineage
structure and the gene-effect data are untouched, but every true biomarker-dependency link is
destroyed. An honest pipeline should find (almost) nothing there.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .prep import DEFAULT_DATA

CACHE = Path(DEFAULT_DATA) / "cache"
SPLITS = ("discovery", "holdout", "sanger", "sanger_indep")


def frame(name: str) -> pd.DataFrame:
    """Gene-effect matrix for a split. 'sanger_indep' is the Sanger screens restricted to cell lines
    NOT in the discovery split (about 200 of 315), so replication there uses new cell lines as
    well as a different lab and library. Plain 'sanger' includes lines also screened in discovery."""
    if name == "sanger_indep":
        G = load("sanger")
        return G[~G.index.isin(load("discovery").index)]
    return load(name)


@lru_cache(maxsize=None)
def load(name: str) -> pd.DataFrame:
    return pd.read_pickle(CACHE / f"{name}.pkl")


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    order = np.argsort(p[ok])
    n = ok.sum()
    ranked = p[ok][order] * n / np.arange(1, n + 1)
    vals = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    qq = np.empty(n)
    qq[order] = vals
    q[ok] = qq
    return q


class _Split:
    """A gene-effect matrix prepared once: NaNs filled, lineage codes, lineage-centred copy."""

    def __init__(self, name: str, group: str = "OncotreeLineage"):
        G = frame(name)
        self.name, self.genes, self.lines = name, G.columns, G.index
        Y = G.to_numpy(np.float64)
        self.Y = np.where(np.isnan(Y), np.nanmean(Y, axis=0), Y)
        lin = load("meta")[group].reindex(G.index).fillna("Unknown")
        self.lineage = lin
        self.codes, self.lin_names = pd.factorize(lin)
        self.k = self.codes.max() + 1
        self.cnt = np.bincount(self.codes, minlength=self.k)
        Ym = np.zeros((self.k, Y.shape[1]))
        np.add.at(Ym, self.codes, self.Y)
        self.Yr = self.Y - (Ym / self.cnt[:, None])[self.codes]
        self.ssy = np.sum(self.Yr ** 2, axis=0)
        self.gene_mean = self.Y.mean(axis=0)


@lru_cache(maxsize=None)
def split(name: str, group: str = "OncotreeLineage") -> _Split:
    """group: the covariate adjusted for. 'OncotreeLineage' (about 30 tissues) by default;
    'OncotreePrimaryDisease' (about 77 diseases) is a stricter check against subtype confounding."""
    return _Split(name, group)


class World:
    def __init__(self, null_seed: int | None = None):
        self.null_seed = null_seed
        feats = load("features")
        if null_seed is not None:
            rng = np.random.default_rng(null_seed)
            lin = load("meta")["OncotreeLineage"].reindex(feats.index).fillna("Unknown")
            new_index = feats.index.to_numpy().copy()
            for _, pos in pd.Series(np.arange(len(feats)), index=feats.index).groupby(lin.values):
                p = pos.to_numpy()
                new_index[p] = feats.index.to_numpy()[rng.permutation(p)]
            non_lin = [c for c in feats.columns if not c.startswith("LIN_")]
            shuffled = feats.loc[new_index, non_lin].set_axis(feats.index)
            feats = pd.concat([shuffled, feats[[c for c in feats.columns if c.startswith("LIN_")]]], axis=1)
        self.features = feats
        self.pan_essential = pd.Series(split("discovery").gene_mean < -0.8, index=split("discovery").genes)

    @property
    def label(self) -> str:
        return "real" if self.null_seed is None else f"null{self.null_seed}"

    def biomarker(self, name: str, lines) -> pd.Series:
        """'MUT_TP53', or '~MUT_TP53' for the complement (lines WITHOUT the alteration)."""
        neg = name.startswith("~")
        col = name[1:] if neg else name
        if col not in self.features.columns:
            raise KeyError(f"unknown biomarker {col!r}")
        x = self.features[col].reindex(lines)
        return (1 - x) if neg else x

    def scan(self, biomarker: str, split_name: str = "discovery", genes=None,
             group: str = "OncotreeLineage") -> pd.DataFrame:
        """Test one biomarker against every gene (or the listed genes). One row per gene."""
        s = split(split_name, group)
        x = self.biomarker(biomarker, s.lines).to_numpy(float)
        ok = np.isfinite(x)
        cols = np.arange(len(s.genes)) if genes is None else np.flatnonzero(s.genes.isin(list(genes)))
        Y, codes = s.Y[ok][:, cols], s.codes[ok]
        x = x[ok]
        n, n_alt = len(x), int(x.sum())
        if n_alt < 3 or n - n_alt < 3:
            raise ValueError(f"{biomarker}: too few altered ({n_alt}) or unaltered ({n - n_alt}) lines in {split_name}")
        if biomarker.lstrip("~").startswith("LIN_"):
            xr, Yr, dof = x - x.mean(), Y - Y.mean(axis=0), n - 2
        else:
            k = s.k
            cnt = np.bincount(codes, minlength=k).astype(float)
            cnt[cnt == 0] = 1
            xr = x - (np.bincount(codes, weights=x, minlength=k) / cnt)[codes]
            Ym = np.zeros((k, Y.shape[1]))
            np.add.at(Ym, codes, Y)
            Yr = Y - (Ym / cnt[:, None])[codes]
            dof = n - len(np.unique(codes)) - 1
        sxx = float(xr @ xr)
        if sxx <= 1e-9:
            raise ValueError(f"{biomarker}: no variation within lineages in {split_name}")
        sxy = xr @ Yr
        beta = sxy / sxx
        resid = np.maximum(np.sum(Yr ** 2, axis=0) - beta * sxy, 1e-12) / max(dof, 1)
        t = beta / np.sqrt(resid / sxx)
        alt, ref = Y[x == 1], Y[x == 0]
        out = pd.DataFrame({
            "gene": s.genes[cols], "beta_adj": beta, "t": t,
            "p": 2 * sps.t.sf(np.abs(t), max(dof, 1)),
            "mean_altered": alt.mean(axis=0), "mean_other": ref.mean(axis=0),
            "frac_altered_dependent": (alt < -0.5).mean(axis=0), "frac_other_dependent": (ref < -0.5).mean(axis=0),
        })
        out["q"] = bh(out["p"].to_numpy()) if genes is None else np.nan
        out["broadly_essential"] = self.pan_essential.reindex(out["gene"]).to_numpy()
        out.attrs.update(n=n, n_altered=n_alt, dof=max(dof, 1), split=split_name)
        return out

    def test_pair(self, biomarker: str, gene: str, split_name: str, group: str = "OncotreeLineage") -> dict:
        """One-sided test that altered lines are MORE dependent on `gene` in one split."""
        res = self.scan(biomarker, split_name, genes=[gene], group=group)
        if res.empty:
            raise ValueError(f"gene {gene!r} not in {split_name}")
        r = res.iloc[0]
        return {"split": split_name, "n_lines": res.attrs["n"], "n_altered": res.attrs["n_altered"],
                "beta_adj": round(float(r.beta_adj), 4), "t": round(float(r.t), 2),
                "p_one_sided": float(f"{float(sps.t.cdf(r.t, res.attrs['dof'])):.3g}"),
                "mean_effect_altered": round(float(r.mean_altered), 3),
                "mean_effect_other": round(float(r.mean_other), 3)}

    def all_pairs(self, split_name="discovery", features=None, top=2000) -> pd.DataFrame:
        """Every biomarker against every gene at once (the non-agentic baseline's genome-wide scan)."""
        s = split(split_name)
        feats = self.features.reindex(s.lines)
        cols = [c for c in (features or feats.columns) if not c.startswith("LIN_")]
        X = feats[cols].to_numpy(float)
        X = np.where(np.isnan(X), 0.0, X)  # missing omics -> treated as unaltered for the bulk scan
        cnt = s.cnt.astype(float)
        Xm = np.zeros((s.k, X.shape[1]))
        np.add.at(Xm, s.codes, X)
        Xr = X - (Xm / cnt[:, None])[s.codes]
        sxx = np.sum(Xr ** 2, axis=0)
        dof = len(s.lines) - s.k - 1
        B = (Xr.T @ s.Yr) / sxx[:, None]                          # features x genes
        resid = np.maximum(s.ssy[None, :] - B * (Xr.T @ s.Yr), 1e-12) / dof
        T = B / np.sqrt(resid / sxx[:, None])
        flat = np.argsort(T, axis=None)[:top]                     # most negative t: altered more dependent
        fi, gi = np.unravel_index(flat, T.shape)
        P = 2 * sps.t.sf(np.abs(T[fi, gi]), dof)
        m = T.size
        out = pd.DataFrame({"biomarker": np.array(cols)[fi], "gene": s.genes[gi], "beta_adj": B[fi, gi],
                            "t": T[fi, gi], "p": P, "n_altered": X.sum(axis=0)[fi].astype(int)})
        out["p_bonferroni"] = np.minimum(out.p * m, 1.0)
        out.attrs.update(n_tests=m, split=split_name)
        return out


# ---------------------------------------------------------------- per-gene scan and cached genome-wide scan
def _feature_resid(world: "World", split_name: str):
    key = (world.label, split_name)
    if key not in _FEAT_CACHE:
        s = split(split_name)
        feats = world.features.reindex(s.lines)
        cols = [c for c in feats.columns if not c.startswith("LIN_")]
        X = np.where(np.isnan(feats[cols].to_numpy(float)), 0.0, feats[cols].to_numpy(float))
        Xm = np.zeros((s.k, X.shape[1]))
        np.add.at(Xm, s.codes, X)
        Xr = X - (Xm / s.cnt[:, None].astype(float))[s.codes]
        _FEAT_CACHE[key] = (cols, X, Xr, np.sum(Xr ** 2, axis=0), len(s.lines) - s.k - 1)
    return _FEAT_CACHE[key]


_FEAT_CACHE: dict = {}
_PAIRS_CACHE: dict = {}


def scan_gene(world: "World", gene: str, split_name: str = "discovery") -> pd.DataFrame:
    """Which biomarkers best predict dependency on one gene? One row per biomarker."""
    s = split(split_name)
    if gene not in s.genes:
        raise KeyError(f"gene {gene!r} not screened")
    cols, X, Xr, sxx, dof = _feature_resid(world, split_name)
    y = s.Yr[:, s.genes.get_loc(gene)]
    ok = sxx > 1e-9
    sxy = Xr.T @ y
    beta = np.where(ok, sxy / np.where(ok, sxx, 1), 0)
    resid = np.maximum(float(y @ y) - beta * sxy, 1e-12) / dof
    t = np.where(ok, beta / np.sqrt(resid / np.where(ok, sxx, 1)), 0)
    out = pd.DataFrame({"biomarker": cols, "n_altered": X.sum(axis=0).astype(int), "beta_adj": beta, "t": t,
                        "p": 2 * sps.t.sf(np.abs(t), dof)})
    out["q"] = bh(out.p.to_numpy())
    return out


def genome_wide(world: "World", top: int = 3000) -> pd.DataFrame:
    if world.label not in _PAIRS_CACHE:
        _PAIRS_CACHE[world.label] = world.all_pairs("discovery", top=top)
    return _PAIRS_CACHE[world.label]


def is_self_pair(biomarker: str, gene: str) -> bool:
    return biomarker.lstrip("~").split("_", 1)[-1] == gene

"""Loaded inside the sandbox for the agent's run_python tool: DISCOVERY data only."""
import pandas as pd

from target_discovery.stats import World, load


def load_discovery(null_seed=None):
    """Returns (gene_effect, biomarkers, meta) for the discovery cell lines only.

    gene_effect: lines x genes (negative = dependent); biomarkers: lines x features (1 = altered);
    meta: CellLineName, OncotreeLineage, OncotreePrimaryDisease.
    """
    G = load("discovery")
    F = World(null_seed).features.reindex(G.index)
    M = load("meta").reindex(G.index)
    return G, F, M

"""Part 2 tests. The guard-rail tests need the local DepMap cache and are skipped without it (e.g. in CI)."""
import pytest

from target_discovery import analysis as A
from target_discovery.stats import CACHE, bh

HAVE_DATA = (CACHE / "discovery.pkl").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="DepMap cache not built (see notebook 03, section 9)")


def test_bh_matches_definition():
    q = bh([0.01, 0.04, 0.03, 0.5])
    assert q[0] == pytest.approx(0.04) and q[3] == pytest.approx(0.5)


def test_recorded_results_tables_build():
    df = A.load()
    assert len(A.hypotheses(df)) > 0
    A.rediscovery(df), A.null_world(df), A.candidates(df), A.costs(df)


@needs_data
def test_replication_requires_preregistration_and_has_a_budget():
    from agentlab.tools import ToolError
    from target_discovery.agent_tools import TargetSession
    from target_discovery.stats import World
    s = TargetSession(World(), max_replications=1)
    with pytest.raises(ToolError):
        s.replicate("H1")                                   # not registered
    s.register_hypothesis("H1", "LOSS_MTAP", "PRMT5", "MTA inhibits PRMT5", "well_established")
    with pytest.raises(ToolError):
        s.register_hypothesis("H1", "LOSS_MTAP", "MAT2A", "edit", "well_established")  # registration is final
    assert s.replicate("H1")["replicated"]                  # a textbook dependency replicates
    s.register_hypothesis("H2", "MUT_SMARCA4", "SMARCA2", "paralog", "well_established")
    with pytest.raises(ToolError):
        s.replicate("H2")                                   # budget used up


@needs_data
def test_null_world_destroys_signal_but_keeps_lineages():
    from target_discovery.stats import World, load
    real, null = World(), World(null_seed=1)
    assert real.test_pair("HOT_BRAF", "BRAF", "holdout")["p_one_sided"] < 1e-6
    assert null.test_pair("HOT_BRAF", "BRAF", "holdout")["p_one_sided"] > 1e-3
    lin = load("meta")["OncotreeLineage"].reindex(real.features.index)
    pd_real = real.features["HOT_BRAF"].groupby(lin.values).sum()
    pd_null = null.features["HOT_BRAF"].groupby(lin.values).sum()
    assert (pd_real == pd_null).all()                       # same number altered within each lineage


@needs_data
def test_submitted_verdicts_are_checked_against_replication():
    from target_discovery.agent_tools import TargetSession
    from target_discovery.stats import World
    s = TargetSession(World(null_seed=1))
    s.register_hypothesis("H1", "LOSS_MTAP", "PRMT5", "MTA", "well_established")
    s.replicate("H1")
    s.submit_findings([{"hypothesis_id": "H1", "verdict": "replicated", "novelty": "well_established",
                        "claim": "x", "confidence": 0.9}], "summary")
    assert s.submission["findings"][0]["verdict_matches_evidence"] is False  # false claim is flagged

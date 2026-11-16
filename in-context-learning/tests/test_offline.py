"""Offline tests: no API key, no DepMap data, no TabPFN weights."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_replay_needs_no_key_and_refuses_unknown_prompts(monkeypatch):
    from icl_llm.claude import ask
    from icl_llm.experiments import regression_task
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    prompt, *_ = regression_task(k=4, trial=0)
    assert json.loads(ask(prompt, model="claude-haiku-5-5", max_tokens=30, json_number=True))["y"] is not None
    with pytest.raises(LookupError):
        ask("a prompt that was never recorded", model="claude-haiku-5-5")


def test_strict_number_parser_rejects_working_out_loud():
    from icl_llm.experiments import _number
    assert _number("15") == 15 and _number(" -3.25 ") == -3.25 and _number("y = 4.5") == 4.5
    assert _number("Looking at the data: x=-5.53 so y is about 6") is None


def test_no_key_in_recorded_files():
    for f in (ROOT / "runs").rglob("*.jsonl"):
        assert "sk-ant-" not in f.read_text()


def test_part1_tables_build():
    from icl_llm import analysis as A
    assert len(A.regression()) and len(A.labels()) and len(A.imbalance())


def test_part2_preregistration_is_unedited():
    text = (ROOT / "runs/part2/preregistration.json").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == (ROOT / "runs/part2/preregistration.sha256").read_text().split()[0]


def test_study2_preregistration_is_unedited_and_uses_fresh_genes():
    d = ROOT / "runs/part2/study2"
    text = (d / "preregistration.json").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == (d / "preregistration.sha256").read_text().split()[0]
    s1 = json.loads((ROOT / "runs/part2/preregistration.json").read_text())
    s2 = json.loads(text)
    assert not set(s2["targets"]) & set(s1["primary_targets"] + s1["secondary_targets"])


def test_feature_selection_uses_context_lines_only():
    import pandas as pd
    from icl_bio.study import select_features
    rng = np.random.default_rng(0)
    E = pd.DataFrame(rng.normal(size=(40, 30)), columns=[f"g{i}" for i in range(30)])
    y_ctx = pd.Series(E["g3"].iloc[:20] * 2.0, index=E.index[:20])     # only context lines carry the signal
    E.loc[20:, "g7"] = E.loc[20:, "g7"] * 0 + np.arange(20)             # test lines: g7 would look perfect
    assert select_features(E, y_ctx, target="TARGET", n=1) == ["g3"]

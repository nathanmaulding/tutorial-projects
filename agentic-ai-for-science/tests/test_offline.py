"""Offline tests: no API key, no network, no cost. Run with `pytest tests`."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentlab import Ledger, RunConfig, Trace, run_agent
from agentlab.budget import BudgetExceeded
from agentlab.models import ScriptedClient, cost_usd
from agentlab.sandbox import run_python
from agentlab.trace import redact
from mechanism_discovery.agent_tools import MechanismSession
from mechanism_discovery.run import run_exhaustive
from mechanism_discovery.tasks import TASKS, TASKS_BY_ID, make_instance

FAKE_KEY = "sk-ant-api03-" + "x" * 40
ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ secrets never leak
def test_sandbox_child_cannot_see_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    res = run_python("import os; print(repr(os.environ.get('ANTHROPIC_API_KEY')))\nprint(len(os.environ))")
    assert "None" in res.stdout
    assert FAKE_KEY not in res.stdout


def test_trace_redacts_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    assert FAKE_KEY not in redact(f"key is {FAKE_KEY}")
    t = Trace(tmp_path / "t.jsonl")
    t.log("tool_result", output=f"oops {FAKE_KEY}")
    assert FAKE_KEY not in (tmp_path / "t.jsonl").read_text()


def test_secret_scanner_catches_keys(tmp_path):
    bad = tmp_path / "leak.py"
    bad.write_text(f'client = Anthropic(api_key="{FAKE_KEY}")\n')
    scanner = ROOT.parent / "scripts" / "check_secrets.py"
    assert subprocess.run([sys.executable, scanner, str(bad)], capture_output=True).returncode == 1
    good = tmp_path / "ok.py"
    good.write_text("client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment\n")
    assert subprocess.run([sys.executable, scanner, str(good)], capture_output=True).returncode == 0


def test_recorded_runs_contain_no_keys():
    runs = ROOT / "runs"
    for p in runs.rglob("*.jsonl") if runs.exists() else []:
        text = p.read_text()
        assert "sk-ant-" not in text, p


# ------------------------------------------------------------------ budget
def test_budget_cap_blocks_calls(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", total_cap_usd=0.001)
    with pytest.raises(BudgetExceeded):
        ledger.check("claude-opus-5-5", prompt_chars=50_000, max_tokens=16000, run_spent=0, run_cap=1)


def test_cost_math():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert cost_usd("claude-haiku-5-5", usage) == pytest.approx(0.60)
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(24.0)


# ------------------------------------------------------------------ the agent loop, with a fake model
def _tool(name, inp, i):
    return {"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": inp}


def test_agent_loop_end_to_end_with_scripted_model(tmp_path):
    inst = make_instance(TASKS_BY_ID["enz_mm"], 0)
    sess = MechanismSession(inst, "agent_full")
    script = [
        lambda m: [{"type": "text", "text": "Planning."},
                   _tool("write_plan", {"steps": ["inspect", "fit", "submit"]}, 1),
                   _tool("inspect_data", {}, 2)],
        lambda m: [_tool("search_mechanism_library", {"query": "hyperbolic saturation substrate"}, 3)],
        lambda m: [_tool("read_mechanism_card", {"card_id": "michaelis_menten"}, 4)],
        lambda m: [_tool("fit_model", {"model": "mm", "error_model": "additive",
                                       "initial_params": {"Vmax": 8, "Km": 1}}, 5),
                   _tool("fit_model", {"model": "hill", "error_model": "additive",
                                       "initial_params": {"Vmax": 8, "K": 1, "n": 1}}, 6)],
        lambda m: [_tool("submit_model", {"fit_id": "fit1", "mechanism_label": "mm", "summary": "fit1 wins",
                                          "confidence": 0.9, "cited_cards": ["michaelis_menten"]}, 7)],
    ]
    res = run_agent(client=ScriptedClient(script), tools=sess.registry(), system="test",
                    task_prompt=inst.prompt(), cfg=RunConfig(model="scripted"),
                    ledger=Ledger(tmp_path / "l.jsonl", 1.0), trace=Trace(tmp_path / "t.jsonl"),
                    run_id="t", is_done=lambda: sess.submission is not None)
    assert res.status == "done" and res.tool_calls == 7
    assert sess.submission["label"] == "mm" and sess.submission["uncited_reads_ok"]
    assert abs(sess.fits["fit1"]["res"]["params"]["Km"] - 2) < 0.6
    events = [json.loads(l)["kind"] for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert events[0] == "run_start" and events[-1] == "run_end"


def test_tool_errors_go_back_to_the_model():
    sess = MechanismSession(make_instance(TASKS_BY_ID["enz_mm"], 0), "agent_no_retrieval")
    out, is_error, _ = sess.registry().execute("fit_model", {"model": "mm", "error_model": "additive",
                                                             "initial_params": {"Vmax": 1, "Km": 1}})
    assert is_error and "no model library" in out


def test_custom_model_fits_in_sandbox():
    inst = make_instance(TASKS_BY_ID["hard_novel_partial_inhibition"], 0)
    sess = MechanismSession(inst, "agent_no_retrieval")
    code = ("PARAMS = ['Vmax', 'Km', 'Ki', 'beta']\n"
            "def predict(df, p):\n"
            "    S, I = df['S'].values, df['I'].values\n"
            "    return p['Vmax']*S/(p['Km']+S)*(1+p['beta']*I/p['Ki'])/(1+I/p['Ki'])\n")
    summary = sess.fit_model("custom", "additive", {"Vmax": 8, "Km": 1, "Ki": 1, "beta": 0.2}, code)
    assert abs(summary["params"]["beta"] - 0.4) < 0.1
    pred = sess.predict_heldout(summary["fit_id"])
    assert len(pred) == len(inst.heldout_x)


# ------------------------------------------------------------------ the benchmark itself
@pytest.mark.parametrize("task", [t for t in TASKS], ids=lambda t: t.id)
def test_exhaustive_baseline_solves_exactly_the_in_library_tasks(task):
    r = run_exhaustive(make_instance(task, 0))
    assert r["success"] == (not task.id.startswith("hard_novel")), r


# ---------------------------------------------------------------- stricter scoring and honest intervals
def test_probe_check_separates_mechanisms_that_agree_near_the_data():
    from mechanism_discovery.rescore import probe_error, probe_grid
    from mechanism_discovery.tasks import TASKS_BY_ID, make_instance, truth_fn
    from mechanism_discovery.library import LIBRARY
    inst = make_instance(TASKS_BY_ID["hard_novel_cooperative_inhibition"], 0)
    g = probe_grid(inst)
    assert g.I.max() > inst.task.heldout.I.max()           # the probe reaches beyond the held-out conditions
    truth = truth_fn(inst.task)(g, inst.task.params)
    assert probe_error(truth, truth, g) == 0
    plain = LIBRARY["competitive_inhibition"].predict(g, {"Vmax": 10, "Km": 2, "Ki": 1})
    assert probe_error(plain, truth, g) > 0.5               # non-cooperative competitive is a different mechanism


def test_headline_splits_library_and_dataset_bootstrap_is_wider_than_naive():
    from mechanism_discovery import analysis as A
    df = A.with_strict(A.load())
    h = A.headline(df)
    assert {"in-library", "out-of-library"} <= set(h.columns)
    assert h.loc["exhaustive", "out-of-library"].startswith("0/")   # 0% by construction
    d = A.paired_difference(df, "agent_full · haiku-5-5", "exhaustive", "in-library")
    lo, hi = d["ci95"]
    assert lo <= d["difference"] <= hi

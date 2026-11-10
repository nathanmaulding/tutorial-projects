"""Run one task with one method ("arm"), and run the whole benchmark.

Arms:
    exhaustive          no LLM: fit every library model for the domain and pick the lowest BIC
    single_shot         one LLM call, no tools: read the data, name the mechanism, guess parameters
    agent_no_retrieval  the agent loop with analysis and fitting tools, but no mechanism library
    agent_full          the agent loop with every tool, including retrieval over mechanism cards

Command line (from the agentic-ai-for-science folder):
    python -m mechanism_discovery.run --arms agent_full --model claude-haiku-5-5 --reps 2
Results append to runs/part1/results.jsonl; finished runs are skipped, so it can be resumed.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from agentlab import Ledger, RunConfig, Trace, make_client, run_agent
from agentlab.budget import BudgetExceeded

from . import prompts
from .agent_tools import MechanismSession
from .fitting import fit
from .library import LIBRARY
from .scoring import library_names_for, score
from .tasks import TASKS, TASKS_BY_ID, TaskInstance, make_instance, y_col

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "part1"
LEDGER = ROOT / "runs" / "spend_ledger.jsonl"
_write_lock = threading.Lock()


# ------------------------------------------------------------------ baseline: exhaustive search
def auto_init(model: str, df, inst: TaskInstance) -> dict:
    """Generic data-driven starting values, so the baseline gets no hints about the truth."""
    y = df[y_col(inst.task)].to_numpy(float)
    g = {"Vmax": 1.2 * y.max(), "n": 1.5, "Ksi": 10.0, "Ki": 1.0, "k": 0.1, "CL": 3.0, "V": 30.0,
         "V1": 15.0, "Q": 5.0, "V2": 40.0, "ka": 1.0, "E0": float(np.median(y[:3])), "Emax": float(np.ptp(y) or 1),
         "EC50": 1.0, "gamma": 1.5, "ke0": 0.5, "R0": float(y[0]), "kout": 0.3, "Imax": 0.5, "IC50": 1.0}
    if "S" in df:
        g["Km"] = g["K"] = float(np.median(df["S"]))
        g["k"] = float(np.median(y / df["S"]))
    else:
        g["Km"] = g["K"] = 1.0
    if model == "pk_1cmt_iv_mm":
        g["Vmax"] = 20.0
    return {p: g[p] for p in LIBRARY[model].params}


def run_exhaustive(inst: TaskInstance) -> dict:
    df = inst.data if not (inst.task.domain == "enzyme" and "I" not in inst.data) else inst.data.assign(I=0.0)
    em = "proportional" if inst.task.domain == "pk" else "additive"
    best = None
    for name in library_names_for(inst):
        try:
            r = fit(LIBRARY[name].predict, LIBRARY[name].params, auto_init(name, df, inst), df, y_col(inst.task), em)
        except Exception:  # noqa: BLE001
            continue
        if best is None or r["bic"] < best[1]["bic"]:
            best = (name, r)
    name, r = best
    pred = LIBRARY[name].predict(inst.heldout_x, r["params"])
    nonident = [w.split(" ")[0] for w in r["identifiability"] if "poorly determined" in w]
    return {"status": "done", "steps": 0, "tool_calls": 0, "usd": 0.0, "confidence": None,
            **score(inst, name, pred, n_params=len(r["params"]), nonident=nonident)}


# ------------------------------------------------------------------ baseline: single LLM call
def run_single_shot(inst: TaskInstance, client, cfg: RunConfig, ledger: Ledger, trace: Trace, run_id: str) -> dict:
    labels = "\n".join(f"- {n}: params {list(LIBRARY[n].params)}" for n in library_names_for(inst))
    user = prompts.SINGLE_SHOT_USER.format(task_prompt=inst.prompt(), csv=inst.data.to_csv(index=False),
                                           labels=labels + "\n- custom")
    trace.log("run_start", run_id=run_id, model=client.model, config=cfg.to_dict(),
              system=prompts.SINGLE_SHOT_SYSTEM, task_prompt=user, tools=[])
    msgs = [{"role": "user", "content": user}]
    try:
        ledger.check(client.model, len(user) + 500, 8000, 0.0, cfg.per_run_usd_cap)
        resp = client.create(system=prompts.SINGLE_SHOT_SYSTEM, messages=msgs, tools=[], max_tokens=8000)
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        trace.log("api_error", error=str(e))
        return {"status": "api_error", "usd": 0.0, **score(inst, None, None)}
    usd = ledger.record(client.model, resp.usage, run_id)
    trace.log("llm_response", step=1, stop_reason=resp.stop_reason, usage=resp.usage, usd=usd,
              content=resp.content, stop_details=resp.stop_details)
    base = {"status": "done", "steps": 1, "tool_calls": 0, "usd": usd}
    m = re.search(r"\{.*\}", resp.text, re.S)
    try:
        ans = json.loads(m.group(0))
        label, params = ans["mechanism_label"], {k: float(v) for k, v in ans.get("params", {}).items()}
    except Exception:  # noqa: BLE001
        trace.log("run_end", status="unparseable", usd=usd)
        return {**base, "status": "unparseable", **score(inst, None, None)}
    pred, n_params = None, len(params)
    try:
        if label in LIBRARY:
            pred = LIBRARY[label].predict(inst.heldout_x, params)
        elif label == "custom" and ans.get("custom_code"):
            sess = MechanismSession(inst)
            pred = sess.run_custom("predict", ans["custom_code"], params=params,
                                   x_csv=inst.heldout_x.to_csv(index=False))["pred"]
    except Exception as e:  # noqa: BLE001 - wrong parameter names etc. count as failed predictions
        trace.log("predict_error", error=str(e))
    result = {**base, "confidence": ans.get("confidence"), **score(inst, label, pred, n_params=n_params)}
    trace.log("run_end", status="done", usd=usd, result=result)
    return result


# ------------------------------------------------------------------ the agent
def run_agent_arm(inst: TaskInstance, arm: str, client, cfg: RunConfig, ledger: Ledger, trace: Trace,
                  run_id: str) -> dict:
    sess = MechanismSession(inst, arm)
    system = (prompts.AGENT_FULL if arm == "agent_full" else prompts.AGENT_NO_RETRIEVAL).format(
        max_steps=cfg.max_steps)
    res = run_agent(client=client, tools=sess.registry(), system=system, task_prompt=inst.prompt(), cfg=cfg,
                    ledger=ledger, trace=trace, run_id=run_id, is_done=lambda: sess.submission is not None,
                    done_reminder=prompts.DONE_REMINDER)
    out = {"status": res.status, "steps": res.steps, "tool_calls": res.tool_calls, "usd": round(res.usd, 6),
           "n_fits": len(sess.fits), "cards_read": sorted(sess.cards_read), "n_hypotheses": len(sess.hypotheses),
           "plan_written": bool(sess.plan), "hypotheses": sess.hypotheses}
    sub = sess.submission
    if sub is None:
        out.update(score(inst, None, None))
        return out
    f = sess.fits[sub["fit_id"]]
    try:
        pred = sess.predict_heldout(sub["fit_id"])
    except Exception as e:  # noqa: BLE001
        trace.log("predict_error", error=str(e))
        pred = None
    out.update({"confidence": sub["confidence"], "nonidentifiable_params": sub["nonidentifiable_params"],
                "cited_cards": sub["cited_cards"], "citations_were_read": sub["uncited_reads_ok"],
                "summary": sub["summary"], "submitted_params": f["res"]["params"]})
    out.update(score(inst, sub["label"], pred, n_params=len(f["res"]["params"]),
                     nonident=sub["nonidentifiable_params"]))
    trace.log("scored", result={k: v for k, v in out.items() if k != "hypotheses"})
    return out


# ------------------------------------------------------------------ benchmark driver
def run_key(task_id, seed, arm, model, rep):
    return f"{task_id}__s{seed}__{arm}__{model}__r{rep}"


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["run_key"] for l in path.open() if l.strip()}


def run_one(task_id: str, seed: int, arm: str, model: str, rep: int, cfg: RunConfig, ledger: Ledger,
            out_dir: Path = RUNS) -> dict:
    inst = make_instance(TASKS_BY_ID[task_id], seed)
    key = run_key(task_id, seed, arm, model if arm != "exhaustive" else "none", rep)
    trace_path = out_dir / "traces" / arm / (model if arm != "exhaustive" else "none") / f"{inst.id}__r{rep}.jsonl"
    if arm == "exhaustive":
        result = run_exhaustive(inst)
    else:
        client = make_client(model, effort=cfg.effort, thinking_display=cfg.thinking_display)
        trace = Trace(trace_path)
        if arm == "single_shot":
            result = run_single_shot(inst, client, cfg, ledger, trace, key)
        else:
            result = run_agent_arm(inst, arm, client, cfg, ledger, trace, key)
    rec = {"run_key": key, "task_id": task_id, "seed": seed, "tier": inst.task.tier, "domain": inst.task.domain,
           "truth": inst.task.truth, "arm": arm, "model": model if arm != "exhaustive" else "none", "rep": rep,
           "trace": str(trace_path.relative_to(ROOT)) if arm != "exhaustive" else None, **result}
    with _write_lock, (out_dir / "results.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="exhaustive,single_shot,agent_no_retrieval,agent_full")
    ap.add_argument("--model", default=None, help="defaults to AGENTLAB_MODEL or claude-haiku-5-5")
    ap.add_argument("--tasks", default="all", help="all | standard | hard | comma-separated task ids")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="list the runs without doing them")
    a = ap.parse_args(argv)

    cfg = RunConfig(**({"model": a.model} if a.model else {}))
    if a.max_steps:
        cfg.max_steps = a.max_steps
    ledger = Ledger(LEDGER, cfg.total_usd_cap)
    if a.tasks in ("all", "standard", "hard"):
        tasks = [t.id for t in TASKS if a.tasks == "all" or t.tier == a.tasks]
    else:
        tasks = a.tasks.split(",")
    RUNS.mkdir(parents=True, exist_ok=True)
    have = done_keys(RUNS / "results.jsonl")
    jobs = [(t, int(s), arm, cfg.model, r) for arm in a.arms.split(",") for t in tasks
            for s in a.seeds.split(",") for r in range(1, (1 if arm == "exhaustive" else a.reps) + 1)]
    jobs = [j for j in jobs if run_key(j[0], j[1], j[2], j[3] if j[2] != "exhaustive" else "none", j[4]) not in have]
    print(f"{len(jobs)} runs to do. Spent so far: ${ledger.spent():.4f} of ${cfg.total_usd_cap:.2f} cap.")
    if a.dry_run:
        for j in jobs:
            print("  ", j)
        return
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(run_one, *j, cfg, ledger): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                r = fut.result()
                print(f"{r['run_key']:75s} {r['status']:13s} success={r['success']!s:5s} "
                      f"label={r['label']} err={r['heldout_nrmse']} ${r.get('usd', 0):.4f}", flush=True)
            except BudgetExceeded as e:
                print("BUDGET STOP:", e, flush=True)
            except Exception as e:  # noqa: BLE001
                print("FAILED", j, type(e).__name__, e, flush=True)
    print(f"Total spent: ${ledger.spent():.4f}")


if __name__ == "__main__":
    main()

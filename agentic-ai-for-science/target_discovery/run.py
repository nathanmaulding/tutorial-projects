"""Part 2 runs: rediscovery benchmark, open discovery missions, null-world controls, and the
non-agentic baseline pipeline.

    python -m target_discovery.run rediscovery --model claude-haiku-5-5 --reps 2
    python -m target_discovery.run mission --mission paralogs --model claude-opus-5-5
    python -m target_discovery.run mission --mission paralogs --null-seed 1 --model claude-haiku-5-5
    python -m target_discovery.run baseline
Results append to runs/part2/results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import threading

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agentlab import Ledger, RunConfig, Trace, make_client, run_agent

from . import pubmed
from .agent_tools import ALPHA, MAX_REPLICATIONS, TargetSession
from .stats import World, genome_wide, is_self_pair

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "part2"
LEDGER = ROOT / "runs" / "spend_ledger.jsonl"
_lock = threading.Lock()

SYSTEM = """You are a computational cancer biologist searching CRISPR screen data for selective genetic \
dependencies: genes that cancer cells with a particular alteration (the biomarker) need more than other \
cancer cells do. Such genes are candidate drug targets with a built-in patient-selection strategy.

Work like a careful scientist:
1. Plan first with write_plan, and revise it as you learn.
2. Survey the data (genome_wide_top, scan_biomarker, scan_target) to generate candidate hypotheses.
3. Be sceptical before you believe a signal. Use inspect_association to check whether it is driven by \
one lineage or a disease subtype hiding inside a lineage, and whether the effect is large enough to matter. \
Ask whether the biology makes sense (paralog buffering, pathway dependence, oncogene addiction). A broadly \
essential target may lack a therapeutic window.
4. Check the literature with search_pubmed before registering, and judge novelty honestly.
5. Register only your best hypotheses (register_hypothesis). Registration is final and must come before \
replication. You have {max_rep} replications, each of which must pass one-sided p < {alpha:g} in BOTH held-out \
Broad cell lines and independent Sanger screens.
6. Report honestly with submit_findings. A hypothesis is replicated only if replicate() said so. Negative \
results are results. Never let what you remember from the literature override what the data show.

You have at most {max_steps} model turns. Batch independent tool calls in one turn."""

REDISCOVERY_TASK = """Patient population of interest: cancers with biomarker {biomarker} ({desc}).

Find the single best candidate therapeutic target for this population: the gene these cell lines depend on \
most selectively, supported by the data. Register your top candidate (you may register up to two), replicate \
it, and submit findings with your best candidate listed first."""

MISSIONS = {
    "paralogs": "Search for synthetic-lethal paralog dependencies: genes whose loss (LOSS_ or MUT_ biomarkers) "
                "makes cells depend on a paralog or functional backup.",
    "tumor_suppressors": "Search for actionable dependencies created by loss of tumor suppressors or other "
                         "loss-of-function alterations, beyond the paralog pairs.",
    "open": "Search freely for the most promising selective dependencies in the data.",
}
MISSION_TASK = """Mission: {focus}

Goal: identify up to {max_rep} candidate selective dependencies, and favour candidates that are NOT already well \
established in the literature, since rediscovering textbook biology is not the aim. Pre-register, replicate, \
and submit honest findings, including failures."""

DONE_REMINDER = "You have not submitted. Call submit_findings now, including every registered hypothesis."

# Positive controls: well-established biomarker -> dependency relationships, with accepted answers.
POSITIVE_CONTROLS = {
    "LOSS_MTAP": ("loss of MTAP expression, usually by 9p21 co-deletion with CDKN2A", {"PRMT5", "WDR77", "MAT2A"}),
    "MUT_SMARCA4": ("damaging SMARCA4 mutation", {"SMARCA2"}),
    "MUT_ARID1A": ("damaging ARID1A mutation", {"ARID1B"}),
    "LOSS_MLH1": ("loss of MLH1 expression (mismatch-repair deficiency)", {"WRN"}),
    "~MUT_TP53": ("TP53 wild-type (no damaging TP53 mutation)", {"MDM2", "PPM1D", "MDM4", "USP7"}),
    "MUT_STAG2": ("damaging STAG2 mutation", {"STAG1"}),
    "MUT_KEAP1": ("damaging KEAP1 mutation", {"NFE2L2"}),
    "HOT_BRAF": ("BRAF hotspot mutation", {"BRAF", "MAP2K1"}),
    "HOT_NRAS": ("NRAS hotspot mutation", {"NRAS", "SHOC2"}),
    "HOT_PIK3CA": ("PIK3CA hotspot mutation", {"PIK3CA"}),
    "HOT_KRAS": ("KRAS hotspot mutation", {"KRAS"}),
    "LOSS_RPP25": ("loss of RPP25 expression", {"RPP25L"}),
}


def _write(rec: dict):
    RUNS.mkdir(parents=True, exist_ok=True)
    with _lock, (RUNS / "results.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def session_record(sess: TargetSession) -> dict:
    sub = sess.submission or {}
    return {"registered": sess.registered, "replications": sess.replications,
            "pubmed_queries": sess.pubmed_queries, "plan": sess.plan,
            "findings": sub.get("findings"), "summary": sub.get("summary"),
            "n_registered": len(sess.registered),
            "n_replicated": sum(r["replicated"] for r in sess.replications.values()),
            "n_claimed_replicated": sum(f.get("verdict") == "replicated" for f in sub.get("findings") or []),
            "all_verdicts_match_evidence": all(f["verdict_matches_evidence"] for f in sub.get("findings") or [])}


def run_session(kind: str, name: str, model: str, rep: int, null_seed: int | None, cfg: RunConfig, ledger: Ledger):
    world = World(null_seed)
    sess = TargetSession(world)
    if kind == "rediscovery":
        desc, _ = POSITIVE_CONTROLS[name]
        task = REDISCOVERY_TASK.format(biomarker=name, desc=desc)
    else:
        task = MISSION_TASK.format(focus=MISSIONS[name], max_rep=MAX_REPLICATIONS)
    system = SYSTEM.format(max_rep=MAX_REPLICATIONS, alpha=ALPHA, max_steps=cfg.max_steps)
    key = f"{kind}__{name}__{world.label}__{model}__r{rep}"
    trace_path = RUNS / "traces" / kind / model / f"{name}__{world.label}__r{rep}.jsonl"
    res = run_agent(client=make_client(model, effort=cfg.effort), tools=sess.registry(), system=system,
                    task_prompt=task, cfg=cfg, ledger=ledger, trace=Trace(trace_path), run_id=key,
                    is_done=lambda: sess.submission is not None, done_reminder=DONE_REMINDER)
    rec = {"run_key": key, "kind": kind, "name": name, "world": world.label, "model": model, "rep": rep,
           "status": res.status, "steps": res.steps, "tool_calls": res.tool_calls, "usd": round(res.usd, 5),
           "trace": str(trace_path.relative_to(ROOT)), **session_record(sess)}
    if kind == "rediscovery":
        _, accepted = POSITIVE_CONTROLS[name]
        top = (rec["findings"] or [{}])[0].get("target")
        rec.update({"top_target": top, "accepted": sorted(accepted), "correct": top in accepted,
                    "top_replicated": bool(sess.replications.get((rec["findings"] or [{}])[0].get("hypothesis_id"), {}).get("replicated"))})
    _write(rec)
    return rec


def run_baseline(null_seed: int | None = None, k: int = MAX_REPLICATIONS) -> dict:
    """The classic non-agentic pipeline: take the top k genome-wide pairs (no self pairs, >= 8 altered
    lines), replicate them with the same rule, and count PubMed papers mentioning both genes."""
    world = World(null_seed)
    a = genome_wide(world)
    a = a[np.array([not is_self_pair(b, g) for b, g in zip(a.biomarker, a.gene)]) & (a.n_altered >= 8).to_numpy()].head(k)
    found = []
    for x in a.itertuples():
        reps = {}
        for sp in ("holdout", "sanger"):
            try:
                reps[sp] = world.test_pair(x.biomarker, x.gene, sp)
            except ValueError:  # too few altered lines in this set: untestable counts as not replicated
                reps[sp] = {"p_one_sided": None, "beta_adj": None}
        ok = all(r["p_one_sided"] is not None and r["p_one_sided"] < ALPHA and r["beta_adj"] < 0 for r in reps.values())
        gene_b = x.biomarker.lstrip("~").split("_", 1)[1]
        lit = pubmed.search(f"{gene_b} AND {x.gene}", 3)["total_hits"] if null_seed is None else None
        found.append({"biomarker": x.biomarker, "target": x.gene, "t_discovery": round(float(x.t), 1),
                      "replicated": ok, "holdout_p": reps["holdout"]["p_one_sided"],
                      "sanger_p": reps["sanger"]["p_one_sided"], "pubmed_hits_both_genes": lit})
    rec = {"run_key": f"baseline__top{k}__{world.label}", "kind": "baseline", "name": f"top{k}", "world": world.label,
           "model": "none", "usd": 0.0, "findings": found, "n_registered": k,
           "n_replicated": sum(f["replicated"] for f in found)}
    _write(rec)
    return rec


def rediscovery_baseline() -> list[dict]:
    """Non-agentic rediscovery: the top-ranked gene from the biomarker scan."""
    world, out = World(), []
    for b, (_, accepted) in POSITIVE_CONTROLS.items():
        top = world.scan(b).sort_values("t").iloc[0].gene
        out.append({"biomarker": b, "top_target": top, "correct": top in accepted})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["rediscovery", "mission", "baseline"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--mission", default="paralogs")
    ap.add_argument("--controls", default="all")
    ap.add_argument("--null-seed", type=int, default=None)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-run-cap", type=float, default=None)
    a = ap.parse_args(argv)
    if a.kind == "baseline":
        for seed in (None, 1, 2, 3):
            r = run_baseline(seed)
            print(r["world"], "replicated", r["n_replicated"], "of", r["n_registered"])
        return
    cfg = RunConfig(**({"model": a.model} if a.model else {}))
    cfg.max_steps = a.max_steps or (15 if a.kind == "rediscovery" else 30)
    if a.per_run_cap:
        cfg.per_run_usd_cap = a.per_run_cap
    ledger = Ledger(LEDGER, cfg.total_usd_cap)
    names = (list(POSITIVE_CONTROLS) if a.controls == "all" else a.controls.split(",")) \
        if a.kind == "rediscovery" else a.mission.split(",")
    have = {json.loads(l)["run_key"] for l in (RUNS / "results.jsonl").open()} if (RUNS / "results.jsonl").exists() else set()
    world_label = "real" if a.null_seed is None else f"null{a.null_seed}"
    jobs = [(n, r) for n in names for r in range(1, a.reps + 1)
            if f"{a.kind}__{n}__{world_label}__{cfg.model}__r{r}" not in have]
    print(f"{len(jobs)} sessions. Spent so far ${ledger.spent():.3f} of ${cfg.total_usd_cap:.2f}.")
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(run_session, a.kind, n, cfg.model, r, a.null_seed, cfg, ledger): (n, r) for n, r in jobs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                extra = f"top={r.get('top_target')} correct={r.get('correct')}" if a.kind == "rediscovery" else ""
                print(f"{r['run_key']:70s} {r['status']:12s} reg={r['n_registered']} repl={r['n_replicated']} "
                      f"claimed={r['n_claimed_replicated']} {extra} ${r['usd']:.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print("FAILED", futs[fut], type(e).__name__, e, flush=True)
    print(f"Total spent ${ledger.spent():.3f}")


if __name__ == "__main__":
    main()

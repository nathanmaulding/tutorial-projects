"""The tools a mechanism-discovery agent can use, bound to one task instance.

Arms (what the agent is allowed to use) are configured here, so ablations are honest: the
no-retrieval agent does not just lose two tools, it also cannot fit library models by name.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agentlab.retrieval import BM25, load_markdown_docs
from agentlab.sandbox import run_python
from agentlab.tools import Tool, ToolError, ToolRegistry

from .fitting import fit
from .library import LABELS, LIBRARY
from .tasks import TaskInstance, y_col

CARDS = load_markdown_docs(Path(__file__).parent / "cards")
CARD_BY_ID = {d.id: d for d in CARDS}
INDEX = BM25(CARDS)
FIT_RUNNER = "from mechanism_discovery.fit_runner import main\nmain()\n"
MARK = "@@RESULT@@"


def _round(x, sig=4):
    return float(f"{x:.{sig}g}") if isinstance(x, (int, float)) and np.isfinite(x) else x


def summarize_fit(fit_id: str, label: str, res: dict) -> dict:
    return {
        "fit_id": fit_id, "model": label,
        "params": {k: _round(v) for k, v in res["params"].items()},
        "relative_se": {k: (f"{v:.0%}" if np.isfinite(v) else "inf") for k, v in res["rel_se"].items()},
        "n": res["n"], "k": res["k"], "sigma": _round(res["sigma"]),
        "aic": round(res["aic"], 2), "bic": round(res["bic"], 2), "error_model": res["error_model"],
        "identifiability_warnings": res["identifiability"],
        "residual_diagnostics": res["diagnostics"],
    }


class MechanismSession:
    """State for one agent run: the data, the fits made so far, the plan, hypotheses, submission."""

    def __init__(self, inst: TaskInstance, arm: str = "agent_full"):
        assert arm in ("agent_full", "agent_no_retrieval")
        self.inst, self.arm = inst, arm
        self.task = inst.task
        self.ycol = y_col(self.task)
        self.data = inst.data
        self.fits: dict[str, dict] = {}
        self.plan: list[str] = []
        self.hypotheses: dict[str, dict] = {}
        self.cards_read: set[str] = set()
        self.submission: dict | None = None

    # ------------------------------------------------------------ helpers
    def _fit_frame(self) -> pd.DataFrame:
        df = self.data
        if self.task.domain == "enzyme" and "I" not in df.columns:
            df = df.assign(I=0.0)
        return df

    def _csv(self) -> str:
        return self.data.to_csv(index=False)

    def run_custom(self, mode: str, code: str, x_csv: str | None = None, **extra) -> dict:
        files = {"spec.json": json.dumps({"mode": mode, "code": code, **extra})}
        if mode == "fit":
            files["data.csv"] = self._fit_frame().to_csv(index=False)
        else:
            files["x.csv"] = x_csv
        res = run_python(FIT_RUNNER, files=files, timeout=90)
        if MARK not in res.stdout:
            raise ToolError(f"model code failed to run.\n{(res.stderr or res.stdout)[-1500:]}")
        out = json.loads(res.stdout.split(MARK, 1)[1].strip().splitlines()[0])
        if "error" in out:
            raise ToolError(f"{out['error']}\n{out.get('traceback', '')}")
        return out

    # ------------------------------------------------------------ tools
    def inspect_data(self) -> dict:
        df = self.data
        group = next((c for c in ("I", "dose") if c in df.columns and df[c].nunique() > 1), None)
        out = {"rows": len(df), "columns": list(df.columns),
               "describe": json.loads(df.describe().round(4).to_json()),
               "first_rows": df.head(8).to_dict(orient="records")}
        if group:
            out[f"by_{group}"] = json.loads(df.groupby(group)[self.ycol].agg(["count", "min", "max", "mean"])
                                            .round(4).to_json(orient="index"))
        return out

    def run_python_tool(self, code: str) -> str:
        prelude = ("import numpy as np, pandas as pd, scipy\nfrom scipy.integrate import solve_ivp\n"
                   "from mechanism_discovery.library import pk_conc\n"
                   "df = pd.read_csv('data.csv')\n")
        res = run_python(prelude + code, files={"data.csv": self._csv()}, timeout=60)
        text = (res.stdout or "") + (("\n[stderr]\n" + res.stderr[-2000:]) if res.stderr.strip() else "")
        if res.timed_out:
            text += "\n[timed out]"
        return text.strip() or "(no output: use print() to see results)"

    def search_mechanism_library(self, query: str) -> list[dict]:
        hits = INDEX.search(query, k=4)
        return [{"card_id": d.id, "title": d.title, "library_model": d.meta.get("library_model"), "score": s}
                for d, s in hits] or [{"note": "no matching cards"}]

    def read_mechanism_card(self, card_id: str) -> str:
        if card_id not in CARD_BY_ID:
            raise ToolError(f"no card {card_id!r}. Known: {', '.join(CARD_BY_ID)}")
        self.cards_read.add(card_id)
        d = CARD_BY_ID[card_id]
        return f"# {d.title}\nlibrary_model: {d.meta.get('library_model')}\n\n{d.text}"

    def fit_model(self, model: str, error_model: str = "additive", initial_params: dict | None = None,
                  custom_code: str | None = None) -> dict:
        if error_model not in ("additive", "proportional"):
            raise ToolError("error_model must be 'additive' or 'proportional'")
        fit_id = f"fit{len(self.fits) + 1}"
        if model == "custom":
            if not custom_code:
                raise ToolError("model='custom' needs custom_code defining PARAMS and predict(df, p)")
            if not initial_params:
                raise ToolError("custom models need initial_params for every name in PARAMS")
            res = self.run_custom("fit", custom_code, init=initial_params, y_col=self.ycol,
                                  error_model=error_model)
            self.fits[fit_id] = {"kind": "custom", "label": "custom", "code": custom_code, "res": res}
            return summarize_fit(fit_id, "custom", res)
        if self.arm == "agent_no_retrieval":
            raise ToolError("this agent has no model library: write the model yourself with model='custom'")
        if model not in LIBRARY:
            raise ToolError(f"unknown library model {model!r}; use search_mechanism_library to find models, "
                            "or model='custom'")
        m = LIBRARY[model]
        if m.domain != self.task.domain:
            raise ToolError(f"{model} is a {m.domain} model; this is a {self.task.domain} dataset")
        if m.route and self.task.route and m.route != self.task.route:
            raise ToolError(f"{model} is for {m.route} dosing; this dataset is {self.task.route}")
        init = dict(initial_params or {})
        missing = [p for p in m.params if p not in init]
        if missing:
            raise ToolError(f"initial_params must give a positive starting value for {list(m.params)} "
                            f"(missing {missing})")
        res = fit(m.predict, m.params, {p: float(init[p]) for p in m.params}, self._fit_frame(),
                  self.ycol, error_model)
        self.fits[fit_id] = {"kind": "library", "label": model, "res": res}
        return summarize_fit(fit_id, model, res)

    def write_plan(self, steps: list[str]) -> str:
        self.plan = list(steps)
        return "Plan recorded:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.plan))

    def record_hypothesis(self, hypothesis_id: str, mechanism: str, status: str, prediction: str = "",
                          evidence: str = "") -> str:
        if status not in ("proposed", "supported", "rejected", "inconclusive"):
            raise ToolError("status must be proposed, supported, rejected or inconclusive")
        self.hypotheses[hypothesis_id] = {"mechanism": mechanism, "status": status,
                                          "prediction": prediction, "evidence": evidence}
        board = "\n".join(f"- {h}: [{v['status']}] {v['mechanism']}" for h, v in self.hypotheses.items())
        return f"Hypothesis board:\n{board}"

    def submit_model(self, fit_id: str, mechanism_label: str, summary: str, confidence: float,
                     nonidentifiable_params: list[str] | None = None,
                     cited_cards: list[str] | None = None) -> str:
        if self.submission is not None:
            raise ToolError("already submitted")
        if fit_id not in self.fits:
            raise ToolError(f"unknown fit_id {fit_id!r}; submit one of {list(self.fits)}")
        if mechanism_label not in LABELS:
            raise ToolError(f"mechanism_label must be one of {LABELS}")
        f = self.fits[fit_id]
        if f["label"] != "custom" and mechanism_label not in (f["label"], "custom"):
            raise ToolError(f"{fit_id} is a {f['label']} fit; its label must be {f['label']!r}")
        cited = list(cited_cards or [])
        self.submission = {
            "fit_id": fit_id, "label": mechanism_label, "summary": summary,
            "confidence": float(confidence), "nonidentifiable_params": list(nonidentifiable_params or []),
            "cited_cards": cited,
            # Provenance check: every cited card must actually have been read in this run.
            "uncited_reads_ok": all(c in self.cards_read for c in cited),
        }
        return "Submission recorded."

    # ------------------------------------------------------------ registry
    def registry(self) -> ToolRegistry:
        tools = [
            Tool("write_plan", "Record your plan as a short list of steps. Call it first, and again "
                 "whenever your plan changes.",
                 {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
                  "required": ["steps"]}, self.write_plan),
            Tool("inspect_data", "Summary of the data table: columns, ranges, first rows and per-group "
                 "statistics.", {"type": "object", "properties": {}}, self.inspect_data),
            Tool("run_python", "Run Python for exploratory analysis. The data is preloaded as a pandas "
                 "DataFrame `df`; numpy (np), pandas (pd), scipy, solve_ivp and pk_conc(dose, t) are "
                 "imported. Print what you want to see. No plots: print numbers.",
                 {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
                 self.run_python_tool),
        ]
        if self.arm == "agent_full":
            tools += [
                Tool("search_mechanism_library", "Search the mechanism library (cards describing candidate "
                     "mechanisms, how to recognise them, and modelling methods). Returns card ids.",
                     {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                     self.search_mechanism_library),
                Tool("read_mechanism_card", "Read a mechanism card in full: equations, assumptions, data "
                     "signatures, pitfalls, the library model name to fit, and a reference.",
                     {"type": "object", "properties": {"card_id": {"type": "string"}}, "required": ["card_id"]},
                     self.read_mechanism_card),
            ]
        fit_desc = ("Fit a model by nonlinear least squares (all parameters positive; fitted on a log "
                    "scale, multi-start). Returns parameters, relative standard errors, AIC, BIC, "
                    "identifiability warnings and residual diagnostics, plus a fit_id. "
                    "error_model: 'additive' (constant noise) or 'proportional' (noise scales with the value; "
                    "fits log residuals). For model='custom', custom_code must define "
                    "PARAMS = ['name', ...] and def predict(df, p): returning a numpy array with one "
                    "prediction per row of df (p is a dict of parameter values). np, pd, scipy, solve_ivp, "
                    "pk_conc(dose, t) and per_dose(df, solve_one) are available; per_dose calls "
                    "solve_one(dose, sorted_times) for each dose group and maps results back to rows.")
        if self.arm == "agent_full":
            fit_desc += " Or set model to a library model name found in the mechanism cards."
        tools += [
            Tool("fit_model", fit_desc,
                 {"type": "object", "properties": {
                     "model": {"type": "string", "description": "library model name, or 'custom'"},
                     "error_model": {"type": "string", "enum": ["additive", "proportional"]},
                     "initial_params": {"type": "object", "description": "starting value for every parameter"},
                     "custom_code": {"type": "string"}},
                  "required": ["model", "error_model", "initial_params"]}, self.fit_model),
            Tool("record_hypothesis", "Add or update a hypothesis on your hypothesis board: the mechanism, "
                 "what it predicts in these data, its status, and the evidence (cite fit_ids).",
                 {"type": "object", "properties": {
                     "hypothesis_id": {"type": "string"}, "mechanism": {"type": "string"},
                     "status": {"type": "string", "enum": ["proposed", "supported", "rejected", "inconclusive"]},
                     "prediction": {"type": "string"}, "evidence": {"type": "string"}},
                  "required": ["hypothesis_id", "mechanism", "status"]}, self.record_hypothesis),
            Tool("submit_model", "Submit your final model. Ends the task. fit_id must be one of your fits. "
                 "Use mechanism_label 'custom' for a model you wrote yourself. List any parameters the data "
                 "cannot determine in nonidentifiable_params. confidence is your probability (0-1) that the "
                 "mechanism is right.",
                 {"type": "object", "properties": {
                     "fit_id": {"type": "string"},
                     "mechanism_label": {"type": "string", "enum": LABELS},
                     "summary": {"type": "string", "description": "2-5 sentences: the mechanism and the key evidence (cite fit_ids)"},
                     "confidence": {"type": "number"},
                     "nonidentifiable_params": {"type": "array", "items": {"type": "string"}},
                     "cited_cards": {"type": "array", "items": {"type": "string"}}},
                  "required": ["fit_id", "mechanism_label", "summary", "confidence"]}, self.submit_model),
        ]
        return ToolRegistry(tools)

    # ------------------------------------------------------------ held-out evaluation (never shown to the agent)
    def predict_heldout(self, fit_id: str) -> np.ndarray:
        f = self.fits[fit_id]
        hx = self.inst.heldout_x
        if f["kind"] == "library":
            return np.asarray(LIBRARY[f["label"]].predict(hx, f["res"]["params"]), float)
        out = self.run_custom("predict", f["code"], params=f["res"]["params"], x_csv=hx.to_csv(index=False))
        return np.asarray(out["pred"], float)

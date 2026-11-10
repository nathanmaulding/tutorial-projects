"""Benchmark tasks with known ground truth.

Each task simulates an experiment from a true mechanism, adds realistic measurement noise, and hides
two things from the agent: the true mechanism, and a set of held-out conditions (a new dose or new
inhibitor level). The agent's final model is scored on how well it predicts those held-out
conditions, which is what a mechanistic model is for: extrapolating to experiments not yet run.

Three "hard" tasks use mechanisms deliberately left out of the agent's model library: no library
model can fit them and no card gives their equation. The cards do point toward them, as textbooks
do: the inhibition card says what a plateau at saturating inhibitor means, and the indirect-response
card names the variants the library leaves out (including stimulated loss). So these tasks test
whether the agent notices that nothing fits and builds a new model, with retrieval as a pointer;
the no-retrieval ablation is the unaided version.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .fitting import fit, predict_nrmse
from .library import LIBRARY, per_dose, pk_conc, _ode


# ---------------------------------------------------------------- mechanisms NOT in the library
def partial_inhibition(df, p):
    """Hyperbolic (partial) inhibition: a saturating inhibitor leaves a fraction beta of activity."""
    S, I = df["S"].to_numpy(float), df["I"].to_numpy(float)
    return p["Vmax"] * S / (p["Km"] + S) * (1 + p["beta"] * I / p["Ki"]) / (1 + I / p["Ki"])


def cooperative_competitive(df, p):
    """Competitive inhibition where inhibitor binding is cooperative (Hill coefficient h)."""
    S, I = df["S"].to_numpy(float), df["I"].to_numpy(float)
    return p["Vmax"] * S / (p["Km"] * (1 + (I / p["Ki"]) ** p["h"]) + S)


def pd_indirect_stimulate_loss(df, p):
    """Indirect response model IV: the drug stimulates removal of the response variable."""
    def one(dose, ts):
        kin = p["R0"] * p["kout"]
        rhs = lambda t, y: [kin - p["kout"] * (1 + p["Smax"] * (c := float(pk_conc(dose, t)))
                                               / (p["SC50"] + c)) * y[0]]
        return _ode(rhs, [p["R0"]], ts)[0]
    return per_dose(df, one)


NOVEL = {"partial_inhibition": partial_inhibition, "cooperative_competitive": cooperative_competitive,
         "pd_indirect_stimulate_loss": pd_indirect_stimulate_loss}


# ---------------------------------------------------------------- task definitions
ENZ_TEXT = ("Initial reaction rates v (uM/min) of a purified enzyme were measured at substrate "
            "concentrations S (uM){inh}. Each condition was measured in duplicate; rates carry additive "
            "measurement noise.")
PK_TEXT = ("A single {route} dose (mg) was given at t=0 and plasma concentration C (mg/L) was measured "
           "at times t (h), at two dose levels. Assays have roughly proportional (multiplicative) error.")
PD_TEXT = ("A drug was given as a single IV dose (mg) at t=0, at two or three dose levels, and a biomarker E "
           "(arbitrary units) was measured at times t (h). The drug's pharmacokinetics are already well "
           "characterised: one-compartment IV, CL = 5 L/h, V = 50 L, so plasma concentration is "
           "C(t) = dose/50 * exp(-0.1 t) mg/L (available as pk_conc(dose, t) in your Python tools). "
           "The biomarker has additive measurement noise.")

S8 = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
PK_T = [0.25, 0.5, 1, 2, 4, 6, 8, 12, 24]
PD_T = [0.25, 0.5, 1, 2, 4, 6, 8, 12, 24]


def enzyme_grid(S, I):
    return pd.DataFrame([(s, i) for i in I for s in S], columns=["S", "I"])


def dose_grid(doses, times):
    return pd.DataFrame([(d, t) for d in doses for t in times], columns=["dose", "t"])


@dataclass
class TaskDef:
    id: str
    tier: str
    domain: str
    truth: str                        # library model name, or a key of NOVEL
    params: dict
    train: pd.DataFrame               # design (no measurements yet)
    heldout: pd.DataFrame
    noise: tuple[str, float]          # ("additive", sd) or ("proportional", cv)
    description: str
    use: str                          # what the model will be used for (told to the agent)
    route: str | None = None
    accepted: tuple[str, ...] = ()    # labels that count as the right mechanism
    replicates: int = 1
    notes: str = ""


TASKS: list[TaskDef] = [
    # ---- standard tier: the true mechanism is in the library
    TaskDef("enz_mm", "standard", "enzyme", "mm", {"Vmax": 10, "Km": 2},
            enzyme_grid(S8, [0]), enzyme_grid([0.1, 48, 96], [0]), ("additive", 0.3),
            ENZ_TEXT.format(inh=" (no inhibitor, I = 0)"), "predict rates at substrate levels outside the measured range",
            replicates=2),
    TaskDef("enz_hill", "standard", "enzyme", "hill", {"Vmax": 10, "K": 3, "n": 2.5},
            enzyme_grid([0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16], [0]), enzyme_grid([0.25, 24, 40], [0]),
            ("additive", 0.3), ENZ_TEXT.format(inh=" (no inhibitor, I = 0)"),
            "predict rates at substrate levels outside the measured range", replicates=2),
    TaskDef("enz_substrate_inhibition", "standard", "enzyme", "substrate_inhibition",
            {"Vmax": 12, "Km": 1.5, "Ksi": 20}, enzyme_grid(S8 + [64], [0]),
            enzyme_grid([100, 150, 200], [0]), ("additive", 0.3),
            ENZ_TEXT.format(inh=" (no inhibitor, I = 0)"), "predict rates at higher substrate levels",
            replicates=2),
    TaskDef("enz_competitive", "standard", "enzyme", "competitive_inhibition", {"Vmax": 10, "Km": 2, "Ki": 1},
            enzyme_grid(S8, [0, 1, 3]), enzyme_grid([1, 4, 16], [6, 10]), ("additive", 0.3),
            ENZ_TEXT.format(inh=" and inhibitor concentrations I (uM)"),
            "predict rates at higher inhibitor concentrations", replicates=2),
    TaskDef("enz_uncompetitive", "standard", "enzyme", "uncompetitive_inhibition", {"Vmax": 10, "Km": 2, "Ki": 1},
            enzyme_grid(S8, [0, 1, 3]), enzyme_grid([1, 4, 16], [6, 10]), ("additive", 0.3),
            ENZ_TEXT.format(inh=" and inhibitor concentrations I (uM)"),
            "predict rates at higher inhibitor concentrations", replicates=2),
    TaskDef("enz_noncompetitive", "standard", "enzyme", "noncompetitive_inhibition", {"Vmax": 10, "Km": 2, "Ki": 1},
            enzyme_grid(S8, [0, 1, 3]), enzyme_grid([1, 4, 16], [6, 10]), ("additive", 0.3),
            ENZ_TEXT.format(inh=" and inhibitor concentrations I (uM)"),
            "predict rates at higher inhibitor concentrations", replicates=2),
    TaskDef("pk_1cmt_iv", "standard", "pk", "pk_1cmt_iv", {"CL": 5, "V": 50},
            dose_grid([50, 200], PK_T), dose_grid([500], PK_T), ("proportional", 0.10),
            PK_TEXT.format(route="IV bolus"), "predict concentrations after a 500 mg dose", route="iv"),
    TaskDef("pk_2cmt_iv", "standard", "pk", "pk_2cmt_iv", {"CL": 5, "V1": 20, "Q": 10, "V2": 60},
            dose_grid([50, 200], PK_T + [36, 48]), dose_grid([500], PK_T + [36, 48]), ("proportional", 0.10),
            PK_TEXT.format(route="IV bolus"), "predict concentrations after a 500 mg dose", route="iv"),
    TaskDef("pk_1cmt_oral", "standard", "pk", "pk_1cmt_oral", {"ka": 1.2, "CL": 5, "V": 50},
            dose_grid([50, 200], [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24]),
            dose_grid([500], [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24]), ("proportional", 0.10),
            PK_TEXT.format(route="oral"), "predict concentrations after a 500 mg dose", route="oral"),
    TaskDef("pd_direct_emax", "standard", "pd", "pd_direct_emax", {"E0": 10, "Emax": 50, "EC50": 2},
            dose_grid([20, 200], PD_T), dose_grid([60, 1000], PD_T), ("additive", 1.5),
            PD_TEXT, "predict the biomarker time course at 60 mg and 1000 mg"),
    TaskDef("pd_effect_compartment", "standard", "pd", "pd_effect_compartment",
            {"E0": 10, "Emax": 50, "EC50": 2, "ke0": 0.3},
            dose_grid([20, 200], PD_T), dose_grid([60, 1000], PD_T), ("additive", 1.5),
            PD_TEXT, "predict the biomarker time course at 60 mg and 1000 mg"),
    TaskDef("pd_indirect_inhibition", "standard", "pd", "pd_indirect_inhibition",
            {"R0": 100, "kout": 0.2, "Imax": 0.8, "IC50": 1.5},
            dose_grid([20, 100, 500], [1, 2, 4, 8, 12, 24, 36, 48, 72]),
            dose_grid([60, 1000], [1, 2, 4, 8, 12, 24, 36, 48, 72]), ("additive", 3.0),
            PD_TEXT, "predict the biomarker time course at 60 mg and 1000 mg"),
    # ---- hard tier
    TaskDef("hard_unidentifiable_km", "hard", "enzyme", "mm", {"Vmax": 100, "Km": 500},
            enzyme_grid([0.5, 1, 2, 4, 6, 10, 15, 20], [0]), enzyme_grid([3, 8, 18], [0]), ("additive", 0.12),
            ENZ_TEXT.format(inh=" (no inhibitor, I = 0)"), "predict rates within the measured substrate range",
            accepted=("linear_rate",), replicates=2,
            notes="S << Km: only Vmax/Km is identifiable. Correct answer: first-order model, or MM with Km/Vmax flagged."),
    TaskDef("hard_saturable_pk", "hard", "pk", "pk_1cmt_iv_mm", {"V": 50, "Vmax": 40, "Km": 4},
            dose_grid([50, 500], PK_T), dose_grid([1500], PK_T), ("proportional", 0.10),
            PK_TEXT.format(route="IV bolus"), "predict concentrations after a 1500 mg dose", route="iv",
            notes="Dose-dependent half-life; a linear model fit across doses mispredicts the high dose."),
    TaskDef("hard_null_effect", "hard", "pd", "pd_no_effect", {"E0": 50},
            dose_grid([20, 200], PD_T), dose_grid([60, 1000], PD_T), ("additive", 2.5),
            PD_TEXT, "predict the biomarker time course at 60 mg and 1000 mg",
            notes="The drug has no effect. Claiming an exposure-response relationship is a false discovery."),
    TaskDef("hard_novel_partial_inhibition", "hard", "enzyme", "partial_inhibition",
            {"Vmax": 10, "Km": 2, "Ki": 0.5, "beta": 0.4},
            enzyme_grid(S8, [0, 0.5, 2, 8, 20]), enzyme_grid([1, 4, 16], [60, 100]), ("additive", 0.2),
            ENZ_TEXT.format(inh=" and inhibitor concentrations I (uM)"),
            "predict rates at much higher inhibitor concentrations", replicates=2,
            notes="Not in the library: inhibition plateaus at 40% residual activity."),
    TaskDef("hard_novel_cooperative_inhibition", "hard", "enzyme", "cooperative_competitive",
            {"Vmax": 10, "Km": 2, "Ki": 1, "h": 2},
            enzyme_grid(S8, [0, 0.5, 1, 2]), enzyme_grid([1, 4, 16, 32], [4, 8]), ("additive", 0.2),
            ENZ_TEXT.format(inh=" and inhibitor concentrations I (uM)"),
            "predict rates at higher inhibitor concentrations", replicates=2,
            notes="Not in the library: competitive inhibition with inhibitor Hill coefficient 2."),
    TaskDef("hard_novel_stimulated_loss", "hard", "pd", "pd_indirect_stimulate_loss",
            {"R0": 100, "kout": 0.1, "Smax": 9, "SC50": 2},
            dose_grid([20, 100, 500], [1, 2, 4, 8, 12, 24, 36, 48, 72]),
            dose_grid([60, 1000], [1, 2, 4, 8, 12, 24, 36, 48, 72]), ("additive", 3.0),
            PD_TEXT, "predict the biomarker time course at 60 mg and 1000 mg",
            notes="Not in the library: the drug stimulates loss of the response (indirect response model IV)."),
]
TASKS_BY_ID = {t.id: t for t in TASKS}


def truth_fn(task: TaskDef) -> Callable:
    return NOVEL[task.truth] if task.truth in NOVEL else LIBRARY[task.truth].predict


def y_col(task: TaskDef) -> str:
    return {"enzyme": "v", "pk": "C", "pd": "E"}[task.domain]


def _seed(task_id: str, seed: int) -> int:
    return int(hashlib.sha256(f"{task_id}:{seed}".encode()).hexdigest()[:8], 16)


@dataclass
class TaskInstance:
    task: TaskDef
    seed: int
    data: pd.DataFrame                 # what the agent sees
    heldout_x: pd.DataFrame            # hidden conditions
    heldout_truth: np.ndarray          # hidden noiseless truth at those conditions
    oracle_nrmse: float = float("nan")
    threshold: float = 0.05
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.task.id}__s{self.seed}"

    def prompt(self) -> str:
        t = self.task
        cols = ", ".join(self.data.columns)
        return (f"{t.description}\n\nThe data table has {len(self.data)} rows with columns: {cols}. "
                f"Your model will be used to {t.use}.\n\nFind the mechanistic model that best explains "
                "these data, fit it, check it critically, and submit it.")


def make_instance(task: TaskDef, seed: int = 0) -> TaskInstance:
    rng = np.random.default_rng(_seed(task.id, seed))
    f = truth_fn(task)
    design = pd.concat([task.train] * task.replicates, ignore_index=True)
    mean = np.asarray(f(design, task.params), float)
    kind, scale = task.noise
    y = mean * np.exp(rng.normal(0, scale, len(mean))) if kind == "proportional" \
        else mean + rng.normal(0, scale, len(mean))
    ycol = y_col(task)
    data = design.copy()
    data[ycol] = np.round(y, 4)
    if task.domain == "enzyme" and data["I"].nunique() == 1:
        data = data.drop(columns="I")
    hx = task.heldout.copy()
    truth = np.asarray(f(hx, task.params), float)
    inst = TaskInstance(task, seed, data.reset_index(drop=True), hx, truth)
    # Oracle: refit the TRUE model to this noisy data. Its held-out error is the best achievable.
    names = list(task.params)
    xdata = data if "I" in data.columns or task.domain != "enzyme" else data.assign(I=0.0)
    res = fit(f, names, task.params, xdata, ycol, error_model=kind if kind == "proportional" else "additive")
    inst.oracle_nrmse = predict_nrmse(f(hx, res["params"]), truth)
    # Pass = within 2x of the best achievable error, but never stricter than 5% or looser than 15%.
    inst.threshold = float(np.clip(2.0 * inst.oracle_nrmse, 0.05, 0.15))
    return inst

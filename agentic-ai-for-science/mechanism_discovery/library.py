"""The mechanism library: candidate models the agent can fit by name.

Every model is a function `predict(df, p) -> np.ndarray` that maps a data table (one row per
measurement) and a parameter dict to the predicted measurement for each row. All parameters are
positive (fits run on a log scale).

Domains and data columns:
    enzyme  initial reaction rate v at substrate S and inhibitor I concentrations   (S, I -> v)
    pk      plasma concentration C after a dose at t=0, by route iv or oral          (dose, t -> C)
    pd      drug effect E over time after an IV dose with known PK                   (dose, t -> E)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- known PK used by the PD tasks
KNOWN_PK = {"CL": 5.0, "V": 50.0}  # one-compartment IV bolus: CL in L/h, V in L, so k = 0.1 per h


def pk_conc(dose, t):
    """Plasma concentration (mg/L) for the PD tasks' well-characterised one-compartment IV PK."""
    k = KNOWN_PK["CL"] / KNOWN_PK["V"]
    return np.asarray(dose, float) / KNOWN_PK["V"] * np.exp(-k * np.asarray(t, float))


def per_dose(df, solve_one: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    """Helper for ODE models: solve once per dose group at that group's times, map back to rows."""
    out = np.empty(len(df))
    dose, t = df["dose"].to_numpy(float), df["t"].to_numpy(float)
    for d in np.unique(dose):
        m = dose == d
        ts = np.unique(t[m])
        vals = solve_one(d, ts)
        out[m] = vals[np.searchsorted(ts, t[m])]
    return out


def _ode(rhs, y0, ts, args=()):
    sol = solve_ivp(rhs, (0.0, float(ts.max())), y0, t_eval=ts, args=args, method="LSODA",
                    rtol=1e-7, atol=1e-9)
    if not sol.success:
        return np.full((len(y0), len(ts)), np.nan)
    return sol.y


# ---------------------------------------------------------------- enzyme kinetics
def mm(df, p):
    S = df["S"].to_numpy(float)
    return p["Vmax"] * S / (p["Km"] + S)


def hill(df, p):
    S = df["S"].to_numpy(float)
    return p["Vmax"] * S ** p["n"] / (p["K"] ** p["n"] + S ** p["n"])


def substrate_inhibition(df, p):
    S = df["S"].to_numpy(float)
    return p["Vmax"] * S / (p["Km"] + S + S ** 2 / p["Ksi"])


def competitive_inhibition(df, p):
    S, I = df["S"].to_numpy(float), df["I"].to_numpy(float)
    return p["Vmax"] * S / (p["Km"] * (1 + I / p["Ki"]) + S)


def uncompetitive_inhibition(df, p):
    S, I = df["S"].to_numpy(float), df["I"].to_numpy(float)
    return p["Vmax"] * S / (p["Km"] + S * (1 + I / p["Ki"]))


def noncompetitive_inhibition(df, p):
    S, I = df["S"].to_numpy(float), df["I"].to_numpy(float)
    return p["Vmax"] * S / ((p["Km"] + S) * (1 + I / p["Ki"]))


def linear_rate(df, p):
    return p["k"] * df["S"].to_numpy(float)


# ---------------------------------------------------------------- pharmacokinetics
def pk_1cmt_iv(df, p):
    dose, t = df["dose"].to_numpy(float), df["t"].to_numpy(float)
    return dose / p["V"] * np.exp(-p["CL"] / p["V"] * t)


def pk_2cmt_iv(df, p):
    dose, t = df["dose"].to_numpy(float), df["t"].to_numpy(float)
    k10, k12, k21 = p["CL"] / p["V1"], p["Q"] / p["V1"], p["Q"] / p["V2"]
    s = k10 + k12 + k21
    disc = np.sqrt(max(s * s - 4 * k10 * k21, 0.0))
    alpha, beta = (s + disc) / 2, (s - disc) / 2
    if alpha - beta < 1e-12:
        return pk_1cmt_iv(df, {"CL": p["CL"], "V": p["V1"]})
    A = (alpha - k21) / (alpha - beta)
    B = (k21 - beta) / (alpha - beta)
    return dose / p["V1"] * (A * np.exp(-alpha * t) + B * np.exp(-beta * t))


def pk_1cmt_oral(df, p):
    dose, t = df["dose"].to_numpy(float), df["t"].to_numpy(float)
    ka, ke = p["ka"], p["CL"] / p["V"]
    if abs(ka - ke) < 1e-9:
        ka = ke * (1 + 1e-6)
    return dose * ka / (p["V"] * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def pk_1cmt_iv_mm(df, p):
    def one(dose, ts):
        rhs = lambda _t, y: [-(p["Vmax"] / p["V"]) * y[0] / (p["Km"] + y[0])]
        return _ode(rhs, [dose / p["V"]], ts)[0]
    return per_dose(df, one)


# ---------------------------------------------------------------- pharmacodynamics (known PK)
def pd_no_effect(df, p):
    return np.full(len(df), p["E0"])


def pd_direct_emax(df, p):
    C = pk_conc(df["dose"], df["t"])
    return p["E0"] + p["Emax"] * C / (p["EC50"] + C)


def pd_sigmoid_emax(df, p):
    C = pk_conc(df["dose"], df["t"])
    return p["E0"] + p["Emax"] * C ** p["gamma"] / (p["EC50"] ** p["gamma"] + C ** p["gamma"])


def pd_effect_compartment(df, p):
    dose, t = df["dose"].to_numpy(float), df["t"].to_numpy(float)
    k, ke0 = KNOWN_PK["CL"] / KNOWN_PK["V"], p["ke0"]
    if abs(ke0 - k) < 1e-9:
        ke0 = k * (1 + 1e-6)
    Ce = dose / KNOWN_PK["V"] * ke0 / (ke0 - k) * (np.exp(-k * t) - np.exp(-ke0 * t))
    return p["E0"] + p["Emax"] * Ce / (p["EC50"] + Ce)


def pd_indirect_inhibition(df, p):
    """Indirect response model I: the drug inhibits production of the response (Dayneka 1993)."""
    def one(dose, ts):
        kin = p["R0"] * p["kout"]
        rhs = lambda t, y: [kin * (1 - p["Imax"] * (c := float(pk_conc(dose, t))) / (p["IC50"] + c))
                            - p["kout"] * y[0]]
        return _ode(rhs, [p["R0"]], ts)[0]
    return per_dose(df, one)


# ---------------------------------------------------------------- registry
@dataclass(frozen=True)
class ModelDef:
    name: str
    domain: str                 # enzyme | pk | pd
    params: tuple[str, ...]
    predict: Callable
    card: str                   # id of the mechanism card that documents it
    route: str | None = None    # pk only: iv | oral


LIBRARY: dict[str, ModelDef] = {m.name: m for m in [
    ModelDef("mm", "enzyme", ("Vmax", "Km"), mm, "michaelis_menten"),
    ModelDef("hill", "enzyme", ("Vmax", "K", "n"), hill, "hill_cooperativity"),
    ModelDef("substrate_inhibition", "enzyme", ("Vmax", "Km", "Ksi"), substrate_inhibition, "substrate_inhibition"),
    ModelDef("competitive_inhibition", "enzyme", ("Vmax", "Km", "Ki"), competitive_inhibition, "reversible_inhibition"),
    ModelDef("uncompetitive_inhibition", "enzyme", ("Vmax", "Km", "Ki"), uncompetitive_inhibition, "reversible_inhibition"),
    ModelDef("noncompetitive_inhibition", "enzyme", ("Vmax", "Km", "Ki"), noncompetitive_inhibition, "reversible_inhibition"),
    ModelDef("linear_rate", "enzyme", ("k",), linear_rate, "first_order_regime"),
    ModelDef("pk_1cmt_iv", "pk", ("CL", "V"), pk_1cmt_iv, "pk_one_compartment", "iv"),
    ModelDef("pk_2cmt_iv", "pk", ("CL", "V1", "Q", "V2"), pk_2cmt_iv, "pk_two_compartment", "iv"),
    ModelDef("pk_1cmt_oral", "pk", ("ka", "CL", "V"), pk_1cmt_oral, "pk_oral_absorption", "oral"),
    ModelDef("pk_1cmt_iv_mm", "pk", ("V", "Vmax", "Km"), pk_1cmt_iv_mm, "pk_saturable_elimination", "iv"),
    ModelDef("pd_no_effect", "pd", ("E0",), pd_no_effect, "null_model"),
    ModelDef("pd_direct_emax", "pd", ("E0", "Emax", "EC50"), pd_direct_emax, "pd_emax"),
    ModelDef("pd_sigmoid_emax", "pd", ("E0", "Emax", "EC50", "gamma"), pd_sigmoid_emax, "pd_emax"),
    ModelDef("pd_effect_compartment", "pd", ("E0", "Emax", "EC50", "ke0"), pd_effect_compartment, "pd_effect_compartment"),
    ModelDef("pd_indirect_inhibition", "pd", ("R0", "kout", "Imax", "IC50"), pd_indirect_inhibition, "pd_indirect_response"),
]}

LABELS = sorted(LIBRARY) + ["custom"]

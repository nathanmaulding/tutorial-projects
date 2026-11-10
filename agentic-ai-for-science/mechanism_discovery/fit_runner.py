"""Runs inside the sandbox: fit or evaluate a model whose code the agent wrote.

Input: spec.json and data.csv in the working directory. Output: one JSON line on stdout after the
marker. Model code is executed here, in a separate process with no secrets in its environment.
"""
import json
import sys
import traceback

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import solve_ivp

from mechanism_discovery.fitting import fit
from mechanism_discovery.library import pk_conc, per_dose

MARK = "@@RESULT@@"


def load_custom(code):
    ns = {"np": np, "pd": pd, "scipy": scipy, "solve_ivp": solve_ivp, "pk_conc": pk_conc,
          "per_dose": per_dose}
    exec(compile(code, "<agent model>", "exec"), ns)
    if "predict" not in ns or "PARAMS" not in ns:
        raise ValueError("custom_code must define PARAMS (list of parameter names) and predict(df, p)")
    return ns["predict"], list(ns["PARAMS"])


def main():
    spec = json.load(open("spec.json"))
    try:
        predict, names = load_custom(spec["code"])
        if spec["mode"] == "fit":
            df = pd.read_csv("data.csv")
            out = fit(predict, names, spec["init"], df, spec["y_col"], spec["error_model"])
        else:  # predict at given conditions with given parameters
            df = pd.read_csv("x.csv")
            out = {"pred": np.asarray(predict(df, spec["params"]), float).tolist()}
        print(MARK + json.dumps(out))
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(limit=3)
        print(MARK + json.dumps({"error": f"{type(e).__name__}: {e}", "traceback": tb[-1500:]}))
        sys.exit(0)


if __name__ == "__main__":
    main()

"""Fit a model to data and say how good the fit is.

Used three ways: by the agent's fit_model tool (inside the sandbox for model-written code), by the
non-agentic exhaustive-search baseline, and to compute the "oracle" score of the true model.

What a fit returns, and why each piece matters scientifically:
    parameters + relative standard errors  how precisely the data pins down each parameter
    parameter correlations                 pairs the data can't separate (a practical identifiability problem)
    AIC / BIC                              fit quality penalised for parameter count, for comparing models
    residual diagnostics                   systematic misfit: the clue that a mechanism is missing
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

PENALTY = 1e3


def residuals(y, f, error_model):
    if error_model == "proportional":
        bad = ~np.isfinite(f) | (f <= 0)
        r = np.log(np.clip(y, 1e-12, None)) - np.log(np.where(bad, 1.0, f))
    else:
        bad = ~np.isfinite(f)
        r = y - np.where(bad, 0.0, f)
    r[bad] = PENALTY
    return r


def fit(predict, param_names, init, df, y_col, error_model="additive", n_starts=6, seed=0):
    names = list(param_names)
    missing = [n for n in names if n not in init]
    if missing:
        raise ValueError(f"initial values missing for {missing}")
    if any(init[n] <= 0 for n in names):
        raise ValueError("all parameters must be positive (fits run on a log scale)")
    y = df[y_col].to_numpy(float)

    def res(theta):
        with np.errstate(all="ignore"):
            try:
                f = np.asarray(predict(df, dict(zip(names, np.exp(theta)))), float)
            except Exception:  # noqa: BLE001 - a crashing parameter region is just a bad fit
                return np.full(len(y), PENALTY)
        if f.shape != y.shape:
            raise ValueError(f"predict returned shape {f.shape}, expected {y.shape}")
        return residuals(y, f, error_model)

    rng = np.random.default_rng(seed)
    theta0 = np.log([float(init[n]) for n in names])
    best = None
    for i in range(n_starts):
        start = theta0 if i == 0 else theta0 + rng.normal(0, 1.0, len(names))
        try:
            r = least_squares(res, start, method="trf", max_nfev=400 * len(names), x_scale=1.0)
        except ValueError:
            raise
        except Exception:  # noqa: BLE001
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("all fitting attempts failed")

    params = dict(zip(names, np.exp(best.x)))
    r = best.fun
    n, k = len(y), len(names)
    rss = float(np.sum(r ** 2))
    s2 = rss / max(n - k, 1)
    J = best.jac
    try:
        cov = s2 * np.linalg.pinv(J.T @ J)
        rel_se = np.sqrt(np.clip(np.diag(cov), 0, None))       # SE of log(param) ~ relative SE
        sd = np.where(rel_se > 0, rel_se, 1)
        corr = cov / np.outer(sd, sd)
    except np.linalg.LinAlgError:
        rel_se, corr = np.full(k, np.inf), np.eye(k)
    loglik_term = n * math.log(max(rss / n, 1e-300))
    out = {
        "params": {nm: float(v) for nm, v in params.items()},
        "rel_se": {nm: float(v) for nm, v in zip(names, rel_se)},
        "n": n, "k": k, "rss": rss, "sigma": math.sqrt(s2),
        "aic": loglik_term + 2 * k,
        "bic": loglik_term + k * math.log(n),
        "error_model": error_model,
        "converged": bool(best.success),
    }
    out["identifiability"] = identifiability(names, rel_se, corr)
    out["diagnostics"] = diagnose(df, y_col, r, out["sigma"])
    return out


def identifiability(names, rel_se, corr):
    warnings = []
    for nm, se in zip(names, rel_se):
        if not np.isfinite(se) or se > 0.5:
            warnings.append(f"{nm} is poorly determined (relative SE {'inf' if not np.isfinite(se) else f'{se:.0%}'}).")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) > 0.98:
                warnings.append(f"{names[i]} and {names[j]} are almost perfectly correlated "
                                f"(r={corr[i, j]:+.3f}): the data may only determine a combination of them.")
    return warnings


def _runs_z(signs):
    signs = signs[signs != 0]
    n1, n2 = int(np.sum(signs > 0)), int(np.sum(signs < 0))
    if n1 == 0 or n2 == 0:
        return float("nan")
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    return (runs - mu) / math.sqrt(var) if var > 0 else float("nan")


def diagnose(df, y_col, r, sigma):
    """Plain-language summary of systematic structure left in the residuals."""
    covs = [c for c in df.columns if c != y_col and df[c].nunique() > 1]
    out = {"residual_vs": {}, "notes": []}
    for c in covs:
        rho = spearmanr(df[c], r).statistic
        out["residual_vs"][c] = round(float(rho), 3) if np.isfinite(rho) else None
        if np.isfinite(rho) and abs(rho) > 0.35:
            out["notes"].append(f"Residuals trend with {c} (Spearman rho={rho:+.2f}): the model misses how {c} affects {y_col}.")
    # Runs test: are residual signs clustered along the main axis (curvature the model misses)?
    order_cols = [c for c in ("I", "dose", "S", "t") if c in covs]
    if order_cols:
        z = _runs_z(np.sign(df.assign(_r=r).sort_values(order_cols)["_r"].to_numpy()))
        out["runs_test_z"] = round(z, 2) if np.isfinite(z) else None
        if np.isfinite(z) and z < -2.5:
            out["notes"].append(f"Residual signs are clustered (runs test z={z:.1f}): systematic misfit, not noise.")
    group = next((c for c in ("I", "dose") if c in covs), None)
    if group:
        g = df.assign(_r=r).groupby(group)["_r"].mean() / max(sigma, 1e-12)
        out[f"mean_residual_by_{group}_in_sigmas"] = {str(k): round(float(v), 2) for k, v in g.items()}
        worst = g.abs().max()
        if worst > 1.0:
            out["notes"].append(f"Group {group}={g.abs().idxmax()} is fit poorly on average ({worst:.1f} sigma).")
    big = int(np.sum(np.abs(r) > 3 * sigma))
    if big:
        out["notes"].append(f"{big} residual(s) beyond 3 sigma.")
    if not out["notes"]:
        out["notes"].append("No systematic structure detected in the residuals.")
    return out


def predict_nrmse(pred, truth):
    """Held-out error relative to the typical size of the true values."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if not np.all(np.isfinite(pred)):
        return float("inf")
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / np.mean(np.abs(truth)))

#!/usr/bin/env python3
"""
DRND_ST_VOID_TRANSPORT_AUDIT.py

Publication-oriented diagnostic test of DRND-ST / ATP on cosmic void lensing.

Why voids?
----------
Previous DRND void attempts without an explicit transport kernel were weak.
This script tests the current frozen DRND-ST/ATP lensing kernel on void data.

Core hypothesis
---------------
For a void, the relevant source is a mass deficit rather than a positive
baryonic mass excess. The ATP memory component is therefore signed negative:

    DeltaSigma_void_ATP(R) = - DeltaSigma_ATP(R | M_def, a0, kappa)

with

    A(kappa) = kappa * sqrt(G M_def a0) / G

The reference DRND-ST/ATP value is

    kappa = 1/4.

This script does not assume the hypothesis is true. It compares it against
simple phenomenological void controls using AIC/BIC.

Input CSV
---------
Required columns:
    R_Mpc or R_kpc or R_over_Rv + Rv_Mpc
    DeltaSigma or delta_sigma or signal
    DeltaSigma_err or err or error

Optional:
    void_bin / bin / sample
    Rv_Mpc
    z
    delta_v

Example columns:
    void_bin,R_Mpc,DeltaSigma,DeltaSigma_err,Rv_Mpc

Models tested
-------------
1. null_zero
   No signal.

2. void_powerlaw_control
   A flexible negative projected profile:
       -A / sqrt(1 + (R/Rs)^2)

3. void_compensated_control
   Negative central deficit + positive compensation shell.

4. atp_void_fixed_kappa
   Signed ATP with fixed kappa, default 1/4.

5. atp_void_fixed_kappa_plus_shell
   Signed ATP + positive compensation shell.

6. atp_void_kappa_global
   Same as ATP but fitted global kappa. Diagnostic only.

7. atp_void_tilt_global
   ATP multiplied by (R/Rpivot)^alpha. Diagnostic only.

Publication claim rule
----------------------
Only the fixed-kappa ATP models are acceptable for positive DRND claims.
Global kappa and tilt are diagnostic only.

Usage
-----
python DRND_ST_VOID_TRANSPORT_AUDIT.py ^
  --void-lensing-csv void_lensing_data.csv ^
  --outdir DRND_ST_VOID_TRANSPORT_AUDIT

Fast:
python DRND_ST_VOID_TRANSPORT_AUDIT.py ^
  --void-lensing-csv void_lensing_data.csv ^
  --maxiter 200 ^
  --n-random-starts 1 ^
  --outdir DRND_ST_VOID_FAST

Outputs
-------
summary.json
tables/model_comparison.csv
tables/bin_parameters.csv
tables/fit_attempts.csv
tables/predictions_<case>.csv
plots/model_comparison_bic.png
plots/model_comparison_aic.png
plots/profile_<bin>.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

G_SI = 6.67430e-11
MSUN_KG = 1.98847e30
PC_M = 3.0856775814913673e16
KPC_M = 1.0e3 * PC_M
MPC_M = 1.0e6 * PC_M
C_M_S = 299792458.0


def norm_col(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null", "na", "--"}:
            return default
        return float(s)
    except Exception:
        return default


def pick(row, names, default=np.nan):
    m = {norm_col(k): v for k, v in row.items()}
    for n in names:
        k = norm_col(n)
        if k in m:
            v = to_float(m[k])
            if np.isfinite(v):
                return v
    return default


def pick_str(row, names, default=""):
    m = {norm_col(k): v for k, v in row.items()}
    for n in names:
        k = norm_col(n)
        if k in m and m[k] is not None:
            return str(m[k]).strip()
    return default


def safe_json(o):
    if isinstance(o, dict):
        return {str(k): safe_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [safe_json(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def write_csv(path, rows, preferred=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set(k for r in rows for k in r.keys()))
    if preferred:
        fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def a0_from_chi(H0, chi):
    return chi * C_M_S * (H0 * 1000.0 / MPC_M)


def read_void_lensing_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lb = pick_str(row, ["void_bin", "lens_bin", "bin", "sample"], "")

            R_mpc = pick(row, ["R_Mpc", "R", "rp_mpc", "radius_mpc"])
            R_kpc = pick(row, ["R_kpc", "rp_kpc", "radius_kpc"])
            x = pick(row, ["R_over_Rv", "r_over_rv", "x"])
            Rv = pick(row, ["Rv_Mpc", "Rvoid_Mpc", "void_radius_mpc", "Rv"])

            if np.isfinite(R_kpc):
                R_kpc_final = R_kpc
                R_mpc_final = R_kpc / 1000.0
            elif np.isfinite(R_mpc):
                R_mpc_final = R_mpc
                R_kpc_final = 1000.0 * R_mpc
            elif np.isfinite(x) and np.isfinite(Rv):
                R_mpc_final = x * Rv
                R_kpc_final = 1000.0 * R_mpc_final
            else:
                continue

            ds = pick(row, ["DeltaSigma", "delta_sigma", "deltasigma", "ds", "signal"])
            err = pick(row, ["DeltaSigma_err", "delta_sigma_err", "deltasigma_err", "ds_err", "err", "error", "sigma"])
            z = pick(row, ["z", "redshift"])
            delta_v = pick(row, ["delta_v", "void_delta", "density_contrast"])

            if np.isfinite(R_kpc_final) and np.isfinite(ds) and np.isfinite(err) and R_kpc_final > 0 and err > 0:
                rows.append({
                    "void_bin_original": lb,
                    "R_Mpc": float(R_mpc_final),
                    "R_kpc": float(R_kpc_final),
                    "R_over_Rv": float(x) if np.isfinite(x) else np.nan,
                    "Rv_Mpc": float(Rv) if np.isfinite(Rv) else np.nan,
                    "DeltaSigma": float(ds),
                    "DeltaSigma_err": float(err),
                    "z": float(z) if np.isfinite(z) else np.nan,
                    "delta_v_input": float(delta_v) if np.isfinite(delta_v) else np.nan,
                })

    if not rows:
        raise SystemExit("No usable void-lensing rows. Required: R_Mpc/R_kpc or R_over_Rv+Rv_Mpc, DeltaSigma, DeltaSigma_err.")
    return rows


def assign_bins(rows, mode, n_bins):
    existing = sorted(set(str(r.get("void_bin_original", "")).strip() for r in rows if str(r.get("void_bin_original", "")).strip()))
    mode_used = mode

    if mode == "auto":
        if len(existing) > 1:
            mode_used = "existing"
        elif any(np.isfinite(r.get("Rv_Mpc", np.nan)) for r in rows):
            mode_used = "radius"
        else:
            mode_used = "signal"

    if mode_used == "existing":
        for r in rows:
            r["void_bin"] = str(r.get("void_bin_original", "")).strip() or "all"
        return rows, mode_used

    if mode_used == "radius":
        key = np.array([r["Rv_Mpc"] if np.isfinite(r.get("Rv_Mpc", np.nan)) else r["R_Mpc"] for r in rows], float)
    elif mode_used == "signal":
        key = np.array([abs(r["DeltaSigma"]) for r in rows], float)
    elif mode_used == "all":
        for r in rows:
            r["void_bin"] = "all"
        return rows, mode_used
    else:
        raise SystemExit(f"Unknown bin mode {mode}")

    edges = np.unique(np.quantile(key, np.linspace(0, 1, max(2, n_bins + 1))))
    if len(edges) <= 2:
        for r in rows:
            r["void_bin"] = "bin0"
        return rows, mode_used

    for r, v in zip(rows, key):
        idx = np.searchsorted(edges, v, side="right") - 1
        idx = max(0, min(idx, len(edges) - 2))
        r["void_bin"] = f"{mode_used}_bin{idx}"
    return rows, mode_used


def atp_ds_kappa(R_kpc, logMdef, a0, kappa):
    """
    Positive ATP magnitude for a mass scale Mdef.
    Void signal uses negative sign externally.
    Output units: Msun/pc^2.
    """
    R = np.asarray(R_kpc, dtype=float) * KPC_M
    M = (10 ** float(logMdef)) * MSUN_KG
    if M <= 0:
        return np.zeros_like(R)
    r0 = math.sqrt((G_SI * M) / a0)
    A = float(kappa) * math.sqrt(G_SI * M * a0) / G_SI
    R2 = R * R
    root = np.sqrt(R2 + r0 * r0)
    sigma_bar = (2.0 * A / np.maximum(R2, 1e-20)) * (root - r0)
    sigma = A / root
    ds = sigma_bar - sigma
    ds = np.where(R2 < 1e-16, 0.0, ds)
    return ds * (PC_M ** 2) / MSUN_KG


def shell_profile(R_mpc, logA, x0, sigma, Rv_mpc=None):
    """
    Positive compensation shell. If Rv is available, shell location is in R/Rv;
    otherwise uses Mpc with x0 interpreted directly as Mpc.
    """
    R = np.asarray(R_mpc, float)
    A = 10 ** logA
    if Rv_mpc is not None and np.isfinite(Rv_mpc) and Rv_mpc > 0:
        x = R / Rv_mpc
    else:
        x = R
    return A * np.exp(-0.5 * ((x - x0) / max(sigma, 1e-6)) ** 2)


def powerlaw_void(R_kpc, logA, logRs):
    R = np.asarray(R_kpc, float)
    A = 10 ** logA
    Rs = 10 ** logRs
    return -A / np.sqrt(1.0 + (R / Rs) ** 2)


def compensated_control(R_mpc, R_kpc, logA, logRs, logAsh, x0, sigma, Rv):
    return powerlaw_void(R_kpc, logA, logRs) + shell_profile(R_mpc, logAsh, x0, sigma, Rv)


def predict_case(case, params, bins, data, args, a0):
    idx = 0
    preds, rows, bin_rows, priors = [], [], [], []

    log_kappa = None
    alpha = None
    if case in {"atp_void_kappa_global", "atp_void_tilt_global"}:
        log_kappa = params[idx]
        idx += 1
    if case == "atp_void_tilt_global":
        alpha = params[idx]
        idx += 1

    for lb in bins:
        sub = data[lb]
        R_kpc = np.array([r["R_kpc"] for r in sub], float)
        R_mpc = np.array([r["R_Mpc"] for r in sub], float)
        Rv_vals = np.array([r["Rv_Mpc"] for r in sub], float)
        Rv_med = float(np.nanmedian(Rv_vals)) if np.any(np.isfinite(Rv_vals)) else np.nan

        if case == "null_zero":
            pred = np.zeros_like(R_kpc)
            meta = {}

        elif case == "void_powerlaw_control":
            logA, logRs = params[idx], params[idx + 1]
            idx += 2
            pred = powerlaw_void(R_kpc, logA, logRs)
            meta = {"logA_void": logA, "Rs_kpc": 10 ** logRs}

        elif case == "void_compensated_control":
            logA, logRs, logAsh, x0, sigma = params[idx], params[idx + 1], params[idx + 2], params[idx + 3], params[idx + 4]
            idx += 5
            pred = compensated_control(R_mpc, R_kpc, logA, logRs, logAsh, x0, sigma, Rv_med)
            meta = {"logA_void": logA, "Rs_kpc": 10 ** logRs, "shell_logA": logAsh, "shell_x0": x0, "shell_sigma": sigma}

        elif case == "atp_void_fixed_kappa":
            logMdef = params[idx]
            idx += 1
            pred = -atp_ds_kappa(R_kpc, logMdef, a0, args.kappa)
            meta = {"logMdef_Msun": logMdef, "Mdef_Msun": 10 ** logMdef, "kappa_atp": args.kappa}

        elif case == "atp_void_fixed_kappa_plus_shell":
            logMdef, logAsh, x0, sigma = params[idx], params[idx + 1], params[idx + 2], params[idx + 3]
            idx += 4
            pred = -atp_ds_kappa(R_kpc, logMdef, a0, args.kappa) + shell_profile(R_mpc, logAsh, x0, sigma, Rv_med)
            meta = {"logMdef_Msun": logMdef, "Mdef_Msun": 10 ** logMdef, "kappa_atp": args.kappa, "shell_logA": logAsh, "shell_x0": x0, "shell_sigma": sigma}

        elif case == "atp_void_kappa_global":
            logMdef = params[idx]
            idx += 1
            kappa = 10 ** log_kappa
            pred = -atp_ds_kappa(R_kpc, logMdef, a0, kappa)
            meta = {"logMdef_Msun": logMdef, "Mdef_Msun": 10 ** logMdef, "kappa_atp": kappa}

        elif case == "atp_void_tilt_global":
            logMdef = params[idx]
            idx += 1
            kappa = 10 ** log_kappa
            pred = -atp_ds_kappa(R_kpc, logMdef, a0, kappa) * np.power(np.maximum(R_mpc / args.Rpivot_Mpc, 1e-12), alpha)
            meta = {"logMdef_Msun": logMdef, "Mdef_Msun": 10 ** logMdef, "kappa_atp": kappa, "alpha_void": alpha}

        else:
            raise ValueError(case)

        bin_rows.append({"case": case, "void_bin": lb, **meta})
        for i, r in enumerate(sub):
            p = float(pred[i]) if np.isfinite(pred[i]) else np.nan
            preds.append(p)
            rows.append({
                "case": case,
                "void_bin": lb,
                "R_Mpc": r["R_Mpc"],
                "R_kpc": r["R_kpc"],
                "DeltaSigma_obs": r["DeltaSigma"],
                "DeltaSigma_err": r["DeltaSigma_err"],
                "DeltaSigma_pred": p,
                **meta,
            })

    return np.array(preds, float), rows, bin_rows, np.array(priors, float)


def residuals(case, params, bins, data, args, a0):
    pred, _, _, priors = predict_case(case, params, bins, data, args, a0)
    obs = np.array([r["DeltaSigma"] for lb in bins for r in data[lb]], float)
    err = np.array([r["DeltaSigma_err"] for lb in bins for r in data[lb]], float)

    bad = np.where(~np.isfinite(pred), 1e4, 0.0)
    if args.cov_space == "linear":
        res = (obs - pred) / np.maximum(err, 1e-30) + bad
    else:
        scale = np.maximum(np.abs(obs), np.maximum(err, args.min_abs_signal))
        sig = np.maximum(err / scale / math.log(10), args.min_log_sigma)
        sign_obs = np.sign(obs)
        sign_pred = np.sign(pred)
        # Signed log residual only if signs match and nonzero; otherwise use linear penalty.
        ok = (sign_obs == sign_pred) & (np.abs(obs) > 0) & (np.abs(pred) > 0)
        res = np.empty_like(obs)
        res[ok] = (np.log10(np.abs(obs[ok])) - np.log10(np.abs(pred[ok]))) / sig[ok]
        res[~ok] = (obs[~ok] - pred[~ok]) / np.maximum(err[~ok], 1e-30)
        res = res + bad

    return res if args.no_prior else np.concatenate([res, priors])


def base_initial_vector(case, bins, args):
    p0, bounds = [], []

    if case in {"atp_void_kappa_global", "atp_void_tilt_global"}:
        p0.append(math.log10(args.kappa))
        bounds.append((math.log10(args.kappa_min), math.log10(args.kappa_max)))
    if case == "atp_void_tilt_global":
        p0.append(0.0)
        bounds.append((args.alpha_min, args.alpha_max))

    for _ in bins:
        if case == "null_zero":
            pass
        elif case == "void_powerlaw_control":
            p0 += [0.0, math.log10(1000.0)]
            bounds += [(-8, 8), (math.log10(1.0), math.log10(2.0e5))]
        elif case == "void_compensated_control":
            p0 += [0.0, math.log10(1000.0), -1.0, 1.0, 0.3]
            bounds += [(-8, 8), (math.log10(1.0), math.log10(2.0e5)), (-8, 8), (0.2, 3.0), (0.05, 2.0)]
        elif case == "atp_void_fixed_kappa":
            p0 += [14.0]
            bounds += [(10.0, 18.0)]
        elif case == "atp_void_fixed_kappa_plus_shell":
            p0 += [14.0, -1.0, 1.0, 0.3]
            bounds += [(10.0, 18.0), (-8, 8), (0.2, 3.0), (0.05, 2.0)]
        elif case in {"atp_void_kappa_global", "atp_void_tilt_global"}:
            p0 += [14.0]
            bounds += [(10.0, 18.0)]
        else:
            raise ValueError(case)

    return np.array(p0, float), bounds


def generate_starts(case, bins, args):
    p0, bounds = base_initial_vector(case, bins, args)
    if len(p0) == 0:
        return [p0], bounds

    starts = [p0.copy()]

    def clip(v):
        out = np.array(v, float)
        for i, (lo, hi) in enumerate(bounds):
            out[i] = min(max(out[i], lo + 1e-6), hi - 1e-6)
        return out

    for lm in [12.0, 13.0, 14.0, 15.0, 16.0]:
        v = p0.copy()
        for i, (lo, hi) in enumerate(bounds):
            if lo <= 10.1 and hi >= 17.9:
                v[i] = lm
        starts.append(clip(v))

    if "kappa_global" in case or "tilt" in case:
        for kappa in [0.125, 0.25, 0.5, 1.0, 3.0, math.pi]:
            v = p0.copy()
            v[0] = math.log10(min(max(kappa, args.kappa_min), args.kappa_max))
            starts.append(clip(v))

    if case == "atp_void_tilt_global":
        for alpha in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            v = p0.copy()
            v[1] = alpha
            starts.append(clip(v))

    seed = int(hashlib.md5(f"{case}|{len(bins)}|{args.random_seed}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    for _ in range(args.n_random_starts):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds], float))

    seen, unique = set(), []
    for s in starts:
        key = tuple(np.round(s, 6))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique, bounds


def fit_case(case, bins, data, args, a0):
    starts, bounds = generate_starts(case, bins, args)
    attempts, best = [], None

    def obj(p):
        r = residuals(case, p, bins, data, args, a0)
        return float(np.sum(r * r))

    for si, start in enumerate(starts):
        try:
            if len(start) == 0:
                chi2 = obj(start)
                class Dummy:
                    success = True
                    message = "analytic null"
                    x = start
                    fun = chi2
                opt = Dummy()
            else:
                opt = minimize(obj, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": args.maxiter, "maxfun": args.maxfun, "ftol": 1e-8})
                chi2 = float(opt.fun)

            attempts.append({"case": case, "start_id": si, "chi2": chi2, "success": bool(opt.success), "message": str(opt.message)})
            if best is None or chi2 < best["chi2"]:
                best = {"opt": opt, "chi2": chi2, "start_id": si}
        except Exception as e:
            attempts.append({"case": case, "start_id": si, "chi2": 1e300, "success": False, "message": f"exception: {e}"})

    if best is None:
        raise RuntimeError(f"All starts failed for {case}")

    opt = best["opt"]
    pred, pred_rows, bin_rows, _ = predict_case(case, opt.x, bins, data, args, a0)
    res = residuals(case, opt.x, bins, data, args, a0)
    chi2 = float(np.sum(res * res))
    n = sum(len(data[lb]) for lb in bins)
    k = len(opt.x)
    dof = max(n - k, 1)
    invalid_frac = float(np.mean(~np.isfinite(pred)))

    return {
        "case": case,
        "chi2": chi2,
        "n_data": n,
        "k_params": k,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "chi2_per_data": chi2 / max(n, 1),
        "AIC": chi2 + 2 * k,
        "BIC": chi2 + k * math.log(max(n, 2)),
        "success": bool(opt.success),
        "message": str(opt.message),
        "best_start_id": best["start_id"],
        "n_starts": len(starts),
        "invalid_prediction_fraction": invalid_frac,
        "pred_rows": pred_rows,
        "bin_rows": bin_rows,
        "attempts": attempts,
    }


def summarize_params(bin_rows):
    kappas = [r.get("kappa_atp") for r in bin_rows if r.get("kappa_atp") is not None]
    alphas = [r.get("alpha_void") for r in bin_rows if r.get("alpha_void") is not None]
    masses = [r.get("Mdef_Msun") for r in bin_rows if r.get("Mdef_Msun") is not None]
    out = {}
    if kappas:
        arr = np.array(kappas, float)
        out.update({"kappa_median": float(np.median(arr)), "kappa_min": float(np.min(arr)), "kappa_max": float(np.max(arr))})
    if alphas:
        arr = np.array(alphas, float)
        out.update({"alpha_median": float(np.median(arr)), "alpha_min": float(np.min(arr)), "alpha_max": float(np.max(arr))})
    if masses:
        arr = np.array(masses, float)
        out.update({"Mdef_median": float(np.median(arr)), "Mdef_min": float(np.min(arr)), "Mdef_max": float(np.max(arr))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--void-lensing-csv", required=True)
    ap.add_argument("--outdir", default="DRND_ST_VOID_TRANSPORT_AUDIT")
    ap.add_argument("--bin-mode", default="auto", choices=["auto", "existing", "radius", "signal", "all"])
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--H0", type=float, default=67.4)
    ap.add_argument("--chi", type=float, default=0.18)
    ap.add_argument("--kappa", type=float, default=0.25)
    ap.add_argument("--kappa-min", type=float, default=0.01)
    ap.add_argument("--kappa-max", type=float, default=100.0)
    ap.add_argument("--Rpivot-Mpc", type=float, default=10.0)
    ap.add_argument("--alpha-min", type=float, default=-3.0)
    ap.add_argument("--alpha-max", type=float, default=3.0)
    ap.add_argument("--cov-space", default="linear", choices=["linear", "signed_log"])
    ap.add_argument("--min-log-sigma", type=float, default=0.05)
    ap.add_argument("--min-abs-signal", type=float, default=1e-6)
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--maxfun", type=int, default=1500)
    ap.add_argument("--n-random-starts", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=12345)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tabledir = outdir / "tables"
    plotdir = outdir / "plots"
    tabledir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    rows = read_void_lensing_csv(args.void_lensing_csv)
    rows, mode_used = assign_bins(rows, args.bin_mode, args.n_bins)
    write_csv(tabledir / "binned_void_lensing_data.csv", rows, preferred=["void_bin", "R_Mpc", "R_kpc", "DeltaSigma", "DeltaSigma_err", "Rv_Mpc", "R_over_Rv"])

    data = {}
    for r in rows:
        data.setdefault(r["void_bin"], []).append(r)
    bins = sorted(data.keys())
    a0 = a0_from_chi(args.H0, args.chi)

    cases = [
        "null_zero",
        "void_powerlaw_control",
        "void_compensated_control",
        "atp_void_fixed_kappa",
        "atp_void_fixed_kappa_plus_shell",
        "atp_void_kappa_global",
        "atp_void_tilt_global",
    ]

    results, all_bin_rows, all_attempts, param_summary = [], [], [], {}

    for case in cases:
        print("[fit]", case, flush=True)
        fit = fit_case(case, bins, data, args, a0)
        write_csv(tabledir / f"predictions_{case}.csv", fit["pred_rows"])
        write_csv(tabledir / f"bin_parameters_{case}.csv", fit["bin_rows"])
        all_bin_rows += fit["bin_rows"]
        all_attempts += fit["attempts"]
        param_summary[case] = summarize_params(fit["bin_rows"])
        results.append({k: v for k, v in fit.items() if k not in {"pred_rows", "bin_rows", "attempts"}})

    write_csv(tabledir / "bin_parameters.csv", all_bin_rows)
    write_csv(tabledir / "fit_attempts.csv", all_attempts)

    best_aic = min(results, key=lambda r: r["AIC"])
    best_bic = min(results, key=lambda r: r["BIC"])

    comp = []
    for r in sorted(results, key=lambda x: x["BIC"]):
        rr = dict(r)
        rr["delta_AIC_vs_best"] = rr["AIC"] - best_aic["AIC"]
        rr["delta_BIC_vs_best"] = rr["BIC"] - best_bic["BIC"]
        rr.update({f"diag_{k}": v for k, v in param_summary.get(r["case"], {}).items()})
        comp.append(rr)

    write_csv(tabledir / "model_comparison.csv", comp, preferred=[
        "case", "chi2", "k_params", "dof", "chi2_per_dof", "AIC", "BIC",
        "delta_AIC_vs_best", "delta_BIC_vs_best",
        "diag_kappa_median", "diag_alpha_median", "diag_Mdef_median",
        "success", "best_start_id", "n_starts", "invalid_prediction_fraction"
    ])

    for metric in ["BIC", "AIC"]:
        ordered = sorted(comp, key=lambda r: r[metric])
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(ordered)), [r[metric] for r in ordered])
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels([r["case"] for r in ordered], rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"Void ATP transport audit by {metric}")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(plotdir / f"model_comparison_{metric.lower()}.png", dpi=180)
        plt.close(fig)

    # Per-bin profile plots.
    pred_by_case = {}
    for case in cases:
        pfile = tabledir / f"predictions_{case}.csv"
        if not pfile.exists():
            continue
        with open(pfile, "r", encoding="utf-8-sig") as f:
            pred_by_case[case] = list(csv.DictReader(f))

    for lb in bins:
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = [r for r in rows if r["void_bin"] == lb]
        R = np.array([r["R_Mpc"] for r in sub], float)
        y = np.array([r["DeltaSigma"] for r in sub], float)
        e = np.array([r["DeltaSigma_err"] for r in sub], float)
        ax.errorbar(R, y, yerr=e, fmt="o", label="data")
        for case in ["atp_void_fixed_kappa", "atp_void_fixed_kappa_plus_shell", "void_compensated_control"]:
            pr = [r for r in pred_by_case.get(case, []) if r.get("void_bin") == lb]
            if not pr:
                continue
            pr_sorted = sorted(pr, key=lambda r: float(r["R_Mpc"]))
            ax.plot([float(r["R_Mpc"]) for r in pr_sorted], [float(r["DeltaSigma_pred"]) for r in pr_sorted], label=case)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("R [Mpc]")
        ax.set_ylabel("DeltaSigma")
        ax.set_title(f"Void bin {lb}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plotdir / f"profile_{lb}.png", dpi=180)
        plt.close(fig)

    by = {r["case"]: r for r in comp}
    atp = by.get("atp_void_fixed_kappa")
    atps = by.get("atp_void_fixed_kappa_plus_shell")
    ctrl = by.get("void_compensated_control")
    null = by.get("null_zero")
    kg = by.get("atp_void_kappa_global")
    tilt = by.get("atp_void_tilt_global")

    def bd(a, b):
        return a["BIC"] - b["BIC"] if a and b else None

    decision = {
        "best_model_by_BIC": best_bic["case"],
        "best_model_by_AIC": best_aic["case"],
        "ATP_fixed_kappa_BIC": atp["BIC"] if atp else None,
        "ATP_fixed_kappa_plus_shell_BIC": atps["BIC"] if atps else None,
        "void_compensated_control_BIC": ctrl["BIC"] if ctrl else None,
        "null_zero_BIC": null["BIC"] if null else None,
        "delta_BIC_ATPfixed_minus_compensatedControl": bd(atp, ctrl),
        "delta_BIC_ATPshell_minus_compensatedControl": bd(atps, ctrl),
        "delta_BIC_ATPfixed_minus_null": bd(atp, null),
        "delta_BIC_ATPshell_minus_null": bd(atps, null),
        "kappa_global_summary": param_summary.get("atp_void_kappa_global", {}),
        "tilt_summary": param_summary.get("atp_void_tilt_global", {}),
        "fixed_kappa": args.kappa,
    }

    kg_med = decision["kappa_global_summary"].get("kappa_median")
    alpha_med = decision["tilt_summary"].get("alpha_median")
    decision["diagnostic_flags"] = {
        "fixed_ATP_beats_null_BIC": bool(atp and null and atp["BIC"] < null["BIC"]),
        "fixed_ATP_plus_shell_beats_null_BIC": bool(atps and null and atps["BIC"] < null["BIC"]),
        "fixed_ATP_competes_with_compensated_control_BIC": bool(atp and ctrl and atp["BIC"] <= ctrl["BIC"] + 6),
        "fixed_ATP_plus_shell_competes_with_compensated_control_BIC": bool(atps and ctrl and atps["BIC"] <= ctrl["BIC"] + 6),
        "global_kappa_near_lensing_value_0p25": bool(kg_med is not None and 0.125 <= kg_med <= 0.5),
        "tilt_near_zero": bool(alpha_med is not None and abs(alpha_med) <= 0.3),
    }

    summary = {
        "model": "DRND-ST void transport audit",
        "input": vars(args) | {"n_points": len(rows), "bins": bins, "bin_mode_used": mode_used, "a0_fixed": a0},
        "model_comparison": comp,
        "parameter_summary": param_summary,
        "decision": decision,
        "publication_claim_rule": {
            "allowed_positive_claim": "fixed-kappa ATP void model only if it beats null and competes with compensated control",
            "diagnostic_only": ["global kappa", "tilt", "compensation shell"],
        },
        "cautions": [
            "Void lensing is sign-sensitive; check units and sign convention in the input data.",
            "Compensation shells are diagnostic because real void stacks can include ridge/edge compensation.",
            "A survey-level void test requires covariance, void selection function, redshift-space distortion treatment, and independent catalog validation.",
        ],
    }

    (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
    (outdir / "methods_void_transport.md").write_text(
        "We tested the DRND-ST/ATP transport kernel on void lensing by treating voids as signed mass-deficit sources. "
        "The publication-claim model used fixed kappa=1/4 and fixed a0=chi c H0. "
        "Models with fitted global kappa, radial tilt, or compensation shell were used only diagnostically.\n",
        encoding="utf-8"
    )
    print(json.dumps(safe_json({
        "outdir": str(outdir),
        "decision": decision,
        "files": [
            "summary.json",
            "methods_void_transport.md",
            "tables/model_comparison.csv",
            "tables/bin_parameters.csv",
            "tables/fit_attempts.csv",
            "plots/model_comparison_bic.png",
            "plots/model_comparison_aic.png",
            "plots/profile_<bin>.png",
        ],
    }), indent=2))


if __name__ == "__main__":
    main()

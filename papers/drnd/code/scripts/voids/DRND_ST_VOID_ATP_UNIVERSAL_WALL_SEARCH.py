#!/usr/bin/env python3
"""
DRND_ST_VOID_ATP_UNIVERSAL_WALL_SEARCH.py

Goal
----
Search for the correct boundary compensation for voids under the DRND-ST/ATP
assumption that all void walls share a universal shape in x = R/Rv.

This is designed after the negative result of the direct density-prior shell:
a local galaxy-density shell at x~2.88, sigma~0.095 should not be inserted
directly into DeltaSigma. Lensing is a projected observable:

    DeltaSigma(X) = Sigma_bar(<X) - Sigma(X)

so a narrow 3D compensation wall can appear shifted/broadened in the projected
DeltaSigma profile.

Core tested model
-----------------
    DeltaSigma_void(X) =
        - DeltaSigma_ATP(X | Mdef, a0, kappa=1/4)
        + A_wall * DeltaSigma_wall_projected(X | x_wall, sigma_wall)

where:
    X = R/Rv
    x_wall and sigma_wall are global/shared across all void bins.
    A_wall may differ per bin, but the wall geometry is universal.

The wall is modeled in 3D density contrast space as a Gaussian shell:

    rho_wall(x) = exp[-0.5 ((x - x_wall)/sigma_wall)^2]

and is projected through the Abel operator before comparison with lensing.

Publication rule
----------------
Positive void claim only if fixed-kappa ATP + universal projected wall:
  - beats null by BIC, and
  - competes with or beats a phenomenological projected-wall control.

This is still a diagnostic. Full survey covariance and physical units are
required before strong claims.

Input
-----
Void-lensing CSV with columns:
  R_over_Rv or R_Mpc with Rv_Mpc=1 convention
  DeltaSigma
  DeltaSigma_err
Optional:
  void_bin, Rv_Mpc

Usage
-----
python DRND_ST_VOID_ATP_UNIVERSAL_WALL_SEARCH.py ^
  --void-lensing-csv REAL_SDSS_VOID_LENSING/real_sdss_void_lensing_rebinned.csv ^
  --outdir DRND_VOID_ATP_UNIVERSAL_WALL_SEARCH

Fast:
python DRND_ST_VOID_ATP_UNIVERSAL_WALL_SEARCH.py ^
  --void-lensing-csv REAL_SDSS_VOID_LENSING/real_sdss_void_lensing_rebinned.csv ^
  --grid-fast ^
  --outdir DRND_VOID_ATP_WALL_FAST

Outputs
-------
summary.json
tables/model_comparison.csv
tables/wall_grid_scan.csv
tables/predictions_<case>.csv
plots/model_comparison_bic.png
plots/wall_grid_bic.png
plots/profile_<bin>.png
"""

from __future__ import annotations

import argparse, csv, json, math, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, lsq_linear

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
            lb = pick_str(row, ["void_bin", "lens_bin", "bin", "sample"], "") or "all"

            x = pick(row, ["R_over_Rv", "r_over_rv", "x"])
            Rv = pick(row, ["Rv_Mpc", "Rvoid_Mpc", "void_radius_mpc", "Rv"])
            R_mpc = pick(row, ["R_Mpc", "R", "rp_mpc", "radius_mpc"])
            R_kpc = pick(row, ["R_kpc", "rp_kpc", "radius_kpc"])

            if np.isfinite(x):
                x_final = float(x)
                Rv_final = float(Rv) if np.isfinite(Rv) and Rv > 0 else 1.0
                R_mpc_final = x_final * Rv_final
            elif np.isfinite(R_mpc):
                R_mpc_final = float(R_mpc)
                Rv_final = float(Rv) if np.isfinite(Rv) and Rv > 0 else 1.0
                x_final = R_mpc_final / Rv_final
            elif np.isfinite(R_kpc):
                R_mpc_final = float(R_kpc) / 1000.0
                Rv_final = float(Rv) if np.isfinite(Rv) and Rv > 0 else 1.0
                x_final = R_mpc_final / Rv_final
            else:
                continue

            ds = pick(row, ["DeltaSigma", "delta_sigma", "deltasigma", "ds", "signal"])
            err = pick(row, ["DeltaSigma_err", "delta_sigma_err", "deltasigma_err", "ds_err", "err", "error", "sigma"])

            if np.isfinite(x_final) and x_final > 0 and np.isfinite(ds) and np.isfinite(err) and err > 0:
                rows.append({
                    "void_bin": lb,
                    "X": x_final,
                    "R_over_Rv": x_final,
                    "R_Mpc": R_mpc_final,
                    "Rv_Mpc": Rv_final,
                    "DeltaSigma": float(ds),
                    "DeltaSigma_err": float(err),
                })

    if not rows:
        raise SystemExit("No usable void-lensing rows.")
    return rows


def atp_ds_kappa_physical(R_mpc, logMdef, a0, kappa):
    """
    Positive ATP magnitude in Msun/pc^2 when R_mpc is physical.
    In R/Rv convention with Rv=1, this becomes a shape/amplitude proxy.
    Void sign is applied externally.
    """
    R = np.asarray(R_mpc, dtype=float) * MPC_M
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


def projected_wall_delta_sigma(X_eval, x_wall, sigma_wall, xmax=8.0, nr=2400):
    """
    Project a universal 3D Gaussian wall in dimensionless x=R/Rv.

    rho_wall(x) = exp[-0.5((x-x_wall)/sigma)^2]

    Sigma(X) = 2 int_X^xmax rho(x) x dx / sqrt(x^2-X^2)
    DeltaSigma = mean(<X) - Sigma(X)

    Returned profile has arbitrary amplitude; a fitted A_wall multiplies it.
    """
    X_eval = np.asarray(X_eval, float)
    x_min = max(1e-4, np.min(X_eval) / 50.0)
    grid = np.geomspace(x_min, xmax, nr)
    rho = np.exp(-0.5 * ((grid - x_wall) / max(sigma_wall, 1e-5)) ** 2)

    def sigma_at(X):
        mask = grid > X * (1 + 1e-8)
        if np.sum(mask) < 10:
            return np.nan
        r = grid[mask]
        rh = rho[mask]
        integ = rh * r / np.sqrt(np.maximum(r*r - X*X, 1e-300))
        return 2.0 * np.trapezoid(integ, r)

    Sigma_eval = np.array([sigma_at(x) for x in X_eval], float)

    Xd = np.geomspace(x_min, max(np.max(X_eval) * 1.02, x_min * 2), 500)
    Sd = np.array([sigma_at(x) for x in Xd], float)
    ok = np.isfinite(Sd)
    Xd = Xd[ok]
    Sd = Sd[ok]
    if len(Xd) < 20:
        return np.full_like(X_eval, np.nan)

    integ = Sd * Xd
    cum = np.zeros_like(Xd)
    for i in range(1, len(Xd)):
        cum[i] = cum[i-1] + 0.5 * (integ[i] + integ[i-1]) * (Xd[i] - Xd[i-1])
    Sbar = 2.0 * cum / np.maximum(Xd*Xd, 1e-300)
    Dd = Sbar - Sd
    return np.interp(X_eval, Xd, Dd)


def direct_gaussian_delta_sigma_proxy(X, logA, x0, sigma):
    """
    Old direct-in-DeltaSigma shell proxy, kept only as diagnostic control.
    """
    return (10 ** logA) * np.exp(-0.5 * ((np.asarray(X, float) - x0) / max(sigma, 1e-6)) ** 2)


def fit_linear_amplitudes(y, err, components, bounds):
    """
    Fit linear amplitudes for fixed component shapes.
    y = sum a_i components_i
    bounds are arrays/lists of (lo, hi).
    """
    A = np.vstack([np.asarray(c, float) / err for c in components]).T
    b = y / err
    lo = np.array([v[0] for v in bounds], float)
    hi = np.array([v[1] for v in bounds], float)
    res = lsq_linear(A, b, bounds=(lo, hi), method="trf", lsmr_tol="auto")
    pred = np.sum([res.x[i] * components[i] for i in range(len(components))], axis=0)
    chi2 = float(np.sum(((y - pred) / err) ** 2))
    return res.x, pred, chi2


def aic_bic(chi2, n, k):
    return {
        "chi2": float(chi2),
        "n_data": int(n),
        "k_params": int(k),
        "dof": max(int(n) - int(k), 1),
        "chi2_per_dof": float(chi2) / max(int(n) - int(k), 1),
        "AIC": float(chi2) + 2 * int(k),
        "BIC": float(chi2) + int(k) * math.log(max(int(n), 2)),
    }


def scan_wall_grid(rows, args, a0):
    X = np.array([r["X"] for r in rows], float)
    R_mpc = np.array([r["R_Mpc"] for r in rows], float)
    y = np.array([r["DeltaSigma"] for r in rows], float)
    err = np.array([r["DeltaSigma_err"] for r in rows], float)
    n = len(y)

    x_grid = np.linspace(args.x_wall_min, args.x_wall_max, args.n_x_wall)
    s_grid = np.linspace(args.sigma_wall_min, args.sigma_wall_max, args.n_sigma_wall)

    # ATP shape is nonlinear in Mdef, so use a grid in logMdef as a controlled search.
    m_grid = np.linspace(args.logMdef_min, args.logMdef_max, args.n_logMdef)

    scan_rows = []
    best = None

    for xw in x_grid:
        for sw in s_grid:
            wall = projected_wall_delta_sigma(X, xw, sw, xmax=args.x_project_max, nr=args.n_project_grid)
            if not np.all(np.isfinite(wall)):
                continue

            # Wall-only projected control.
            amps, pred, chi2 = fit_linear_amplitudes(y, err, [wall], [(args.wall_amp_min, args.wall_amp_max)])
            rec = {
                "case": "projected_wall_only_control",
                "x_wall": float(xw),
                "sigma_wall": float(sw),
                "logMdef": None,
                "A_wall": float(amps[0]),
                **aic_bic(chi2, n, 3),  # A_wall + x_wall + sigma_wall
            }
            scan_rows.append(rec)
            if best is None or rec["BIC"] < best["BIC"]:
                best = rec | {"prediction": pred}

            # ATP + universal projected wall.
            for lm in m_grid:
                atp = -atp_ds_kappa_physical(R_mpc, lm, a0, args.kappa)
                amps, pred, chi2 = fit_linear_amplitudes(y, err, [wall], [(args.wall_amp_min, args.wall_amp_max)])
                # ATP is fixed for lm, wall amplitude fitted.
                pred2 = atp + amps[0] * wall
                chi22 = float(np.sum(((y - pred2) / err) ** 2))
                rec2 = {
                    "case": "atp_fixed_kappa_plus_universal_projected_wall",
                    "x_wall": float(xw),
                    "sigma_wall": float(sw),
                    "logMdef": float(lm),
                    "A_wall": float(amps[0]),
                    **aic_bic(chi22, n, 4),  # logMdef + A_wall + x_wall + sigma_wall
                }
                scan_rows.append(rec2)
                if rec2["BIC"] < (best["BIC"] if best is not None else float("inf")):
                    best = rec2 | {"prediction": pred2}

    return scan_rows, best


def continuous_refine(rows, args, a0, start):
    """
    Optional continuous refinement around grid best for ATP + projected wall.
    """
    X = np.array([r["X"] for r in rows], float)
    R_mpc = np.array([r["R_Mpc"] for r in rows], float)
    y = np.array([r["DeltaSigma"] for r in rows], float)
    err = np.array([r["DeltaSigma_err"] for r in rows], float)
    n = len(y)

    def obj(p):
        logM, xw, logsig, A_wall = p
        sw = math.exp(logsig)
        if not (args.logMdef_min <= logM <= args.logMdef_max and args.x_wall_min <= xw <= args.x_wall_max and args.sigma_wall_min <= sw <= args.sigma_wall_max):
            return 1e100
        wall = projected_wall_delta_sigma(X, xw, sw, xmax=args.x_project_max, nr=args.n_project_grid)
        if not np.all(np.isfinite(wall)):
            return 1e100
        atp = -atp_ds_kappa_physical(R_mpc, logM, a0, args.kappa)
        pred = atp + A_wall * wall
        return float(np.sum(((y - pred) / err) ** 2))

    p0 = np.array([
        float(start.get("logMdef") if start.get("logMdef") is not None else 14.0),
        float(start["x_wall"]),
        math.log(float(start["sigma_wall"])),
        float(start["A_wall"]),
    ])
    bounds = [
        (args.logMdef_min, args.logMdef_max),
        (args.x_wall_min, args.x_wall_max),
        (math.log(args.sigma_wall_min), math.log(args.sigma_wall_max)),
        (args.wall_amp_min, args.wall_amp_max),
    ]
    opt = minimize(obj, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": args.maxiter, "maxfun": args.maxfun, "ftol": 1e-8})
    logM, xw, logsig, A_wall = opt.x
    sw = math.exp(logsig)
    wall = projected_wall_delta_sigma(X, xw, sw, xmax=args.x_project_max, nr=args.n_project_grid)
    atp = -atp_ds_kappa_physical(R_mpc, logM, a0, args.kappa)
    pred = atp + A_wall * wall
    chi2 = float(np.sum(((y - pred) / err) ** 2))
    rec = {
        "case": "atp_fixed_kappa_plus_universal_projected_wall_refined",
        "x_wall": float(xw),
        "sigma_wall": float(sw),
        "logMdef": float(logM),
        "A_wall": float(A_wall),
        "success": bool(opt.success),
        "message": str(opt.message),
        **aic_bic(chi2, n, 4),
    }
    return rec, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--void-lensing-csv", required=True)
    ap.add_argument("--outdir", default="DRND_VOID_ATP_UNIVERSAL_WALL_SEARCH")
    ap.add_argument("--H0", type=float, default=67.4)
    ap.add_argument("--chi", type=float, default=0.18)
    ap.add_argument("--kappa", type=float, default=0.25)
    ap.add_argument("--grid-fast", action="store_true")
    ap.add_argument("--x-wall-min", type=float, default=1.2)
    ap.add_argument("--x-wall-max", type=float, default=3.4)
    ap.add_argument("--sigma-wall-min", type=float, default=0.05)
    ap.add_argument("--sigma-wall-max", type=float, default=1.2)
    ap.add_argument("--n-x-wall", type=int, default=45)
    ap.add_argument("--n-sigma-wall", type=int, default=40)
    ap.add_argument("--logMdef-min", type=float, default=10.0)
    ap.add_argument("--logMdef-max", type=float, default=18.0)
    ap.add_argument("--n-logMdef", type=int, default=50)
    ap.add_argument("--wall-amp-min", type=float, default=-1e6)
    ap.add_argument("--wall-amp-max", type=float, default=1e6)
    ap.add_argument("--x-project-max", type=float, default=10.0)
    ap.add_argument("--n-project-grid", type=int, default=1600)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--maxfun", type=int, default=800)
    args = ap.parse_args()

    if args.grid_fast:
        args.n_x_wall = min(args.n_x_wall, 25)
        args.n_sigma_wall = min(args.n_sigma_wall, 22)
        args.n_logMdef = min(args.n_logMdef, 25)
        args.n_project_grid = min(args.n_project_grid, 900)

    outdir = Path(args.outdir)
    tabledir = outdir / "tables"
    plotdir = outdir / "plots"
    tabledir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    rows = read_void_lensing_csv(args.void_lensing_csv)
    write_csv(tabledir / "input_void_lensing.csv", rows, preferred=["void_bin", "X", "R_Mpc", "Rv_Mpc", "DeltaSigma", "DeltaSigma_err"])

    a0 = a0_from_chi(args.H0, args.chi)
    n = len(rows)
    X = np.array([r["X"] for r in rows], float)
    y = np.array([r["DeltaSigma"] for r in rows], float)
    err = np.array([r["DeltaSigma_err"] for r in rows], float)

    model_rows = []

    # Null.
    chi2_null = float(np.sum((y / err) ** 2))
    model_rows.append({"case": "null_zero", **aic_bic(chi2_null, n, 0)})

    # Direct Gaussian in DeltaSigma control, optimized continuously.
    def direct_obj(p):
        logA, x0, logsig = p
        pred = direct_gaussian_delta_sigma_proxy(X, logA, x0, math.exp(logsig))
        return float(np.sum(((y - pred) / err) ** 2))
    direct_best = None
    for p0 in [[-2, 2.0, math.log(0.3)], [-1, 2.5, math.log(0.6)], [0, 1.5, math.log(0.2)]]:
        opt = minimize(direct_obj, p0, method="L-BFGS-B", bounds=[(-8, 8), (0.2, 4.0), (math.log(0.03), math.log(2.0))])
        if direct_best is None or opt.fun < direct_best.fun:
            direct_best = opt
    logA, x0, logsig = direct_best.x
    pred_direct = direct_gaussian_delta_sigma_proxy(X, logA, x0, math.exp(logsig))
    model_rows.append({
        "case": "direct_deltaSigma_gaussian_control",
        "logA": float(logA),
        "x_wall": float(x0),
        "sigma_wall": float(math.exp(logsig)),
        "success": bool(direct_best.success),
        **aic_bic(float(direct_best.fun), n, 3),
    })

    # Pure fixed ATP: grid over logM only.
    best_atp = None
    for lm in np.linspace(args.logMdef_min, args.logMdef_max, args.n_logMdef):
        pred = -atp_ds_kappa_physical(np.array([r["R_Mpc"] for r in rows], float), lm, a0, args.kappa)
        chi2 = float(np.sum(((y - pred) / err) ** 2))
        rec = {"case": "atp_fixed_kappa_only", "logMdef": float(lm), **aic_bic(chi2, n, 1)}
        if best_atp is None or rec["BIC"] < best_atp["BIC"]:
            best_atp = rec | {"prediction": pred}
    model_rows.append({k: v for k, v in best_atp.items() if k != "prediction"})

    # Projected wall grid and ATP+wall grid.
    scan_rows, best_scan = scan_wall_grid(rows, args, a0)
    write_csv(tabledir / "wall_grid_scan.csv", scan_rows, preferred=[
        "case", "x_wall", "sigma_wall", "logMdef", "A_wall", "chi2", "k_params", "dof", "chi2_per_dof", "AIC", "BIC"
    ])

    best_projected_wall = min([r for r in scan_rows if r["case"] == "projected_wall_only_control"], key=lambda r: r["BIC"])
    best_atp_wall = min([r for r in scan_rows if r["case"] == "atp_fixed_kappa_plus_universal_projected_wall"], key=lambda r: r["BIC"])

    model_rows.append(best_projected_wall)
    model_rows.append(best_atp_wall)

    refined = None
    pred_refined = None
    try:
        refined, pred_refined = continuous_refine(rows, args, a0, best_atp_wall)
        model_rows.append(refined)
    except Exception as exc:
        model_rows.append({"case": "atp_fixed_kappa_plus_universal_projected_wall_refined", "error": str(exc)})

    # Sort and deltas.
    finite_rows = [r for r in model_rows if "BIC" in r]
    best = min(finite_rows, key=lambda r: r["BIC"])
    for r in finite_rows:
        r["delta_BIC_vs_best"] = r["BIC"] - best["BIC"]
        r["delta_AIC_vs_best"] = r["AIC"] - best["AIC"]
    model_rows_sorted = sorted(model_rows, key=lambda r: r.get("BIC", float("inf")))
    write_csv(tabledir / "model_comparison.csv", model_rows_sorted, preferred=[
        "case", "chi2", "k_params", "dof", "chi2_per_dof", "AIC", "BIC",
        "delta_AIC_vs_best", "delta_BIC_vs_best", "logMdef", "A_wall", "x_wall", "sigma_wall", "success", "message", "error"
    ])

    # Build predictions for key models.
    pred_rows = []
    Xorder = np.argsort(X)
    key_preds = {
        "null_zero": np.zeros_like(y),
        "direct_deltaSigma_gaussian_control": pred_direct,
        "atp_fixed_kappa_only": best_atp["prediction"],
    }

    # Reconstruct best projected wall and best ATP+wall predictions.
    def reconstruct(rec):
        wall = projected_wall_delta_sigma(X, rec["x_wall"], rec["sigma_wall"], xmax=args.x_project_max, nr=args.n_project_grid)
        if rec["case"].startswith("projected_wall_only"):
            return rec["A_wall"] * wall
        atp = -atp_ds_kappa_physical(np.array([r["R_Mpc"] for r in rows], float), rec["logMdef"], a0, args.kappa)
        return atp + rec["A_wall"] * wall

    key_preds["projected_wall_only_control"] = reconstruct(best_projected_wall)
    key_preds["atp_fixed_kappa_plus_universal_projected_wall"] = reconstruct(best_atp_wall)
    if refined is not None and pred_refined is not None:
        key_preds["atp_fixed_kappa_plus_universal_projected_wall_refined"] = pred_refined

    for case, pred in key_preds.items():
        for i, r in enumerate(rows):
            pred_rows.append({
                "case": case,
                "void_bin": r["void_bin"],
                "X": r["X"],
                "R_Mpc": r["R_Mpc"],
                "DeltaSigma_obs": r["DeltaSigma"],
                "DeltaSigma_err": r["DeltaSigma_err"],
                "DeltaSigma_pred": float(pred[i]),
            })
    write_csv(tabledir / "predictions.csv", pred_rows, preferred=[
        "case", "void_bin", "X", "R_Mpc", "DeltaSigma_obs", "DeltaSigma_err", "DeltaSigma_pred"
    ])

    # Plots.
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = sorted(finite_rows, key=lambda r: r["BIC"])
    ax.bar(range(len(ordered)), [r["BIC"] for r in ordered])
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([r["case"] for r in ordered], rotation=35, ha="right")
    ax.set_ylabel("BIC")
    ax.set_title("Void ATP universal wall search")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(plotdir / "model_comparison_bic.png", dpi=180)
    plt.close(fig)

    # Grid heatmap for ATP+wall.
    atp_grid = [r for r in scan_rows if r["case"] == "atp_fixed_kappa_plus_universal_projected_wall"]
    if atp_grid:
        fig, ax = plt.subplots(figsize=(7, 5))
        xs = np.array([r["x_wall"] for r in atp_grid])
        ss = np.array([r["sigma_wall"] for r in atp_grid])
        bs = np.array([r["BIC"] for r in atp_grid])
        sc = ax.scatter(xs, ss, c=bs, s=14)
        fig.colorbar(sc, ax=ax, label="BIC")
        ax.set_xlabel("x_wall")
        ax.set_ylabel("sigma_wall")
        ax.set_title("ATP + universal projected wall grid")
        fig.tight_layout()
        fig.savefig(plotdir / "wall_grid_bic.png", dpi=180)
        plt.close(fig)

    # Profile plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(X, y, yerr=err, fmt="o", label="data")
    for case in [
        "null_zero",
        "atp_fixed_kappa_only",
        "projected_wall_only_control",
        "atp_fixed_kappa_plus_universal_projected_wall",
        "atp_fixed_kappa_plus_universal_projected_wall_refined",
    ]:
        if case not in key_preds:
            continue
        pred = key_preds[case]
        ax.plot(X[Xorder], pred[Xorder], label=case)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("R / Rv")
    ax.set_ylabel("DeltaSigma convention of input file")
    ax.set_title("Void profile: ATP + universal projected wall")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plotdir / "profile_all.png", dpi=180)
    plt.close(fig)

    by = {r["case"]: r for r in finite_rows}
    null = by.get("null_zero")
    atp_only = by.get("atp_fixed_kappa_only")
    wall_only = by.get("projected_wall_only_control")
    atp_wall = by.get("atp_fixed_kappa_plus_universal_projected_wall")
    atp_wall_ref = by.get("atp_fixed_kappa_plus_universal_projected_wall_refined")

    preferred = atp_wall_ref or atp_wall

    def bd(a, b):
        return a["BIC"] - b["BIC"] if a and b else None

    decision = {
        "best_model_by_BIC": best["case"],
        "null_BIC": null["BIC"] if null else None,
        "ATP_fixed_only_BIC": atp_only["BIC"] if atp_only else None,
        "projected_wall_only_control_BIC": wall_only["BIC"] if wall_only else None,
        "ATP_fixed_plus_universal_projected_wall_BIC": atp_wall["BIC"] if atp_wall else None,
        "ATP_fixed_plus_universal_projected_wall_refined_BIC": atp_wall_ref["BIC"] if atp_wall_ref else None,
        "preferred_ATP_wall_case": preferred["case"] if preferred else None,
        "preferred_ATP_wall_x_wall": preferred.get("x_wall") if preferred else None,
        "preferred_ATP_wall_sigma_wall": preferred.get("sigma_wall") if preferred else None,
        "preferred_ATP_wall_logMdef": preferred.get("logMdef") if preferred else None,
        "preferred_ATP_wall_A_wall": preferred.get("A_wall") if preferred else None,
        "delta_BIC_ATPwall_minus_null": bd(preferred, null),
        "delta_BIC_ATPwall_minus_wallOnlyControl": bd(preferred, wall_only),
        "delta_BIC_ATPwall_minus_ATPonly": bd(preferred, atp_only),
        "fixed_kappa": args.kappa,
        "diagnostic_flags": {
            "ATPwall_beats_null_BIC": bool(preferred and null and preferred["BIC"] < null["BIC"]),
            "ATPwall_competes_with_wallOnlyControl_BIC": bool(preferred and wall_only and preferred["BIC"] <= wall_only["BIC"] + 6),
            "ATPwall_beats_wallOnlyControl_BIC": bool(preferred and wall_only and preferred["BIC"] < wall_only["BIC"]),
            "ATPwall_improves_over_ATPonly_BIC": bool(preferred and atp_only and preferred["BIC"] < atp_only["BIC"]),
            "wall_shape_universal_candidate_found": bool(preferred and preferred.get("sigma_wall") is not None),
        }
    }

    summary = {
        "model": "DRND-ST void ATP universal projected wall search",
        "input": vars(args) | {"n_points": n, "a0_fixed": a0},
        "model_comparison": model_rows_sorted,
        "decision": decision,
        "publication_claim_rule": {
            "allowed_positive_claim": "fixed-kappa ATP plus universal projected wall only if it beats null and competes with projected-wall control",
            "diagnostic_only": ["wall-only control", "direct DeltaSigma Gaussian control", "continuous refinement"],
        },
        "cautions": [
            "This searches a universal projected boundary compensation; it is not yet a full survey likelihood.",
            "The SDSS void-lensing file may use R/Rv and DeltaSigma/Rv conventions, so this is primarily a shape/sign test unless physical unit conversion is supplied.",
            "A universal wall is a hypothesis. If wall-only control wins, ATP is not needed by the void data.",
        ],
        "outputs": {
            "model_comparison": "tables/model_comparison.csv",
            "wall_grid_scan": "tables/wall_grid_scan.csv",
            "predictions": "tables/predictions.csv",
            "profile_plot": "plots/profile_all.png",
            "bic_plot": "plots/model_comparison_bic.png",
            "wall_grid_plot": "plots/wall_grid_bic.png",
        },
    }
    (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
    (outdir / "methods_universal_wall.md").write_text(
        "We modeled the void compensation boundary as a universal 3D Gaussian wall in x=R/Rv and projected it through the Abel operator before comparing with DeltaSigma. "
        "The DRND-ST/ATP contribution was kept at fixed kappa=1/4; only the void deficit scale and universal wall parameters were searched diagnostically.\n",
        encoding="utf-8"
    )
    print(json.dumps(safe_json({
        "outdir": str(outdir),
        "decision": decision,
        "files": [
            "summary.json",
            "methods_universal_wall.md",
            "tables/model_comparison.csv",
            "tables/wall_grid_scan.csv",
            "tables/predictions.csv",
            "plots/model_comparison_bic.png",
            "plots/wall_grid_bic.png",
            "plots/profile_all.png",
        ],
    }), indent=2))


if __name__ == "__main__":
    main()

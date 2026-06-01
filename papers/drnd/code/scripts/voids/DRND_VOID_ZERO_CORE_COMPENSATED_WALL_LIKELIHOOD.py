#!/usr/bin/env python3
"""
DRND_VOID_ZERO_CORE_COMPENSATED_WALL_LIKELIHOOD.py

Purpose
-------
Void-sector likelihood with the central ATP mass source removed.

This is the correct diagnostic after the previous failure:
  - no central baryonic/ATP mass parameter,
  - no logMdef fitted as a pseudo-galaxy mass,
  - void lensing is generated only by a projected compensated boundary
    transport profile.

Core DRND void hypothesis
-------------------------
A void is not modeled as a negative galaxy. It is modeled as:

  1. an empty / transport-deficient interior,
  2. a universal transition wall in x = R/Rv,
  3. a compensation ridge carrying the displaced relational density.

The 3D effective transport contrast is:

    q(x) = -S_in(x; x_edge, w_edge)
           + C_comp * G_wall(x; x_wall, sigma_wall)

where:

    S_in(x) = 1 / (1 + exp((x - x_edge)/w_edge))

and:

    G_wall(x) = exp[-0.5 ((x - x_wall)/sigma_wall)^2].

The coefficient C_comp is not fitted freely. It is fixed by compensation:

    integral x^2 q(x) dx = 0.

The observable is then produced by Abel projection:

    Sigma(X) = 2 integral_X^inf q(x) x dx / sqrt(x^2 - X^2)

    DeltaSigma(X) = Sigma_bar(<X) - Sigma(X).

Only one amplitude A_transport is fitted to the projected profile.

Models compared
---------------
1. null_zero
2. projected_gaussian_wall_only_control
3. zero_core_uncompensated_wall
4. zero_core_compensated_wall_DRND

Positive void claim rule
------------------------
The zero-core compensated wall model must:
  - beat null_zero by BIC, and
  - beat or compete with projected_gaussian_wall_only_control.

This is still a diagnostic, not a final survey likelihood.

Data source
-----------
By default the script downloads the public SDSS void-lensing FITS from:
  https://github.com/pmelchior/void-lensing

It stacks per-void E-mode profiles and estimates a full radial covariance by
bootstrap over voids.

Usage
-----
Fast:
python DRND_VOID_ZERO_CORE_COMPENSATED_WALL_LIKELIHOOD.py ^
  --outdir DRND_VOID_ZERO_CORE_FAST ^
  --grid-fast ^
  --nboot 2000

Stronger:
python DRND_VOID_ZERO_CORE_COMPENSATED_WALL_LIKELIHOOD.py ^
  --outdir DRND_VOID_ZERO_CORE_FULL ^
  --nboot 10000

Outputs
-------
summary.json
tables/profile_data.csv
tables/covariance.csv
tables/correlation.csv
tables/model_comparison.csv
tables/grid_scan.csv
tables/predictions.csv
plots/profile_fit.png
plots/correlation_matrix.png
plots/model_comparison_bic.png
plots/grid_bic.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, lsq_linear

DATA_URLS = {
    "collated-sv03s07.fits.gz": "https://github.com/pmelchior/void-lensing/raw/master/data/collated-sv03s07.fits.gz",
    "README.md": "https://raw.githubusercontent.com/pmelchior/void-lensing/master/README.md",
}


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


def write_csv(path: Path, rows, preferred=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set(k for r in rows for k in r.keys()))
    if preferred:
        fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_matrix_csv(path: Path, mat: np.ndarray, prefix="c"):
    rows = []
    for i in range(mat.shape[0]):
        row = {"row": i}
        for j in range(mat.shape[1]):
            row[f"{prefix}{j}"] = float(mat[i, j])
        rows.append(row)
    write_csv(path, rows)


def download(url, path: Path, force=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    print(f"[download] {url}", flush=True)
    urllib.request.urlretrieve(url, path)


def ensure_data(rawdir: Path, force=False):
    for name, url in DATA_URLS.items():
        download(url, rawdir / name, force=force)


def get_native_arrays(data):
    required = ["radius", "dsum", "osum", "rsum", "wsum", "npair"]
    missing = [k for k in required if k not in data.names]
    if missing:
        raise SystemExit(f"FITS table missing columns: {missing}. Available: {data.names}")
    return (
        np.asarray(data["radius"], dtype=float),
        np.asarray(data["dsum"], dtype=float),
        np.asarray(data["osum"], dtype=float),
        np.asarray(data["rsum"], dtype=float),
        np.asarray(data["wsum"], dtype=float),
        np.asarray(data["npair"], dtype=float),
    )


def per_void_profiles(data, observable="xshape"):
    radius, dsum, osum, rsum, wsum, npair = get_native_arrays(data)
    profiles = []
    for i in range(radius.size):
        valid = (npair[i] > 0) & (wsum[i] > 0)
        if not np.any(valid):
            continue
        rmean = rsum[i][valid] / npair[i][valid]
        e = dsum[i][valid] / wsum[i][valid]
        b = osum[i][valid] / wsum[i][valid]
        if observable == "xshape":
            x = rmean / radius[i]
            e = e / radius[i]
            b = b / radius[i]
        elif observable == "physical_proxy":
            x = rmean
        else:
            raise ValueError(observable)
        ok = np.isfinite(x) & np.isfinite(e) & np.isfinite(b)
        if np.sum(ok) >= 2:
            profiles.append({
                "void_index": i,
                "R_void": float(radius[i]),
                "x": np.asarray(x[ok], float),
                "e": np.asarray(e[ok], float),
                "b": np.asarray(b[ok], float),
                "w": np.asarray(npair[i][valid][ok], float),
            })
    return profiles


def bin_stack_profiles(profiles, bin_edges):
    nb = len(bin_edges) - 1
    e_mat = np.full((len(profiles), nb), np.nan)
    b_mat = np.full((len(profiles), nb), np.nan)

    for i, p in enumerate(profiles):
        x, e, b, w = p["x"], p["e"], p["b"], np.maximum(p["w"], 1.0)
        for j in range(nb):
            m = (x >= bin_edges[j]) & (x < bin_edges[j + 1])
            if np.any(m):
                e_mat[i, j] = np.average(e[m], weights=w[m])
                b_mat[i, j] = np.average(b[m], weights=w[m])

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    e_mean = np.nanmean(e_mat, axis=0)
    b_mean = np.nanmean(b_mat, axis=0)
    counts = np.sum(np.isfinite(e_mat), axis=0)
    return centers, e_mean, b_mean, e_mat, b_mat, counts


def bootstrap_covariance(e_mat, nboot=2000, seed=12345):
    rng = np.random.default_rng(seed)
    nvoid, nb = e_mat.shape
    col_mean = np.nanmean(e_mat, axis=0)
    filled = np.where(np.isfinite(e_mat), e_mat, col_mean[None, :])
    boot = np.empty((nboot, nb), dtype=float)
    for k in range(nboot):
        idx = rng.integers(0, nvoid, size=nvoid)
        boot[k] = np.mean(filled[idx, :], axis=0)
    cov = np.cov(boot, rowvar=False, ddof=1)
    err = np.sqrt(np.diag(cov))
    return cov, err, boot


def invert_covariance(cov, nboot, ndata, rcond=1e-8):
    inv = np.linalg.pinv(cov, rcond=rcond)
    hartlap = 1.0
    if nboot > ndata + 2:
        hartlap = (nboot - ndata - 2) / (nboot - 1)
        inv *= hartlap
    return inv, hartlap


def chi2_cov(y, pred, invcov):
    d = np.asarray(y, float) - np.asarray(pred, float)
    return float(d @ invcov @ d)


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


def q_inner(x, x_edge, w_edge):
    return 1.0 / (1.0 + np.exp((x - x_edge) / max(w_edge, 1e-6)))


def q_wall(x, x_wall, sigma_wall):
    return np.exp(-0.5 * ((x - x_wall) / max(sigma_wall, 1e-6)) ** 2)


def compensation_coefficient(x_edge, w_edge, x_wall, sigma_wall, xmax=10.0):
    grid = np.geomspace(1e-4, xmax, 4000)
    inner = q_inner(grid, x_edge, w_edge)
    wall = q_wall(grid, x_wall, sigma_wall)
    i_inner = np.trapezoid(inner * grid * grid, grid)
    i_wall = np.trapezoid(wall * grid * grid, grid)
    if i_wall <= 0 or not np.isfinite(i_wall):
        return np.nan
    return i_inner / i_wall


def projected_delta_sigma_from_q(X_eval, q_func, xmax=10.0, nr=1800):
    X_eval = np.asarray(X_eval, float)
    x_min = max(1e-4, np.min(X_eval) / 50.0)
    grid = np.geomspace(x_min, xmax, nr)
    rho = q_func(grid)

    def sigma_at(X):
        mask = grid > X * (1 + 1e-8)
        if np.sum(mask) < 10:
            return np.nan
        r = grid[mask]
        rh = rho[mask]
        integ = rh * r / np.sqrt(np.maximum(r * r - X * X, 1e-300))
        return 2.0 * np.trapezoid(integ, r)

    sigma_eval = np.array([sigma_at(x) for x in X_eval], float)

    xd = np.geomspace(x_min, max(np.max(X_eval) * 1.02, x_min * 2), 600)
    sd = np.array([sigma_at(x) for x in xd], float)
    ok = np.isfinite(sd)
    xd = xd[ok]
    sd = sd[ok]
    if len(xd) < 20:
        return np.full_like(X_eval, np.nan)

    integ = sd * xd
    cum = np.zeros_like(xd)
    for i in range(1, len(xd)):
        cum[i] = cum[i - 1] + 0.5 * (integ[i] + integ[i - 1]) * (xd[i] - xd[i - 1])
    sbar = 2.0 * cum / np.maximum(xd * xd, 1e-300)
    dd = sbar - sd
    return np.interp(X_eval, xd, dd)


def projected_gaussian_wall(X_eval, x_wall, sigma_wall, xmax=10.0, nr=1800):
    return projected_delta_sigma_from_q(
        X_eval,
        lambda x: q_wall(x, x_wall, sigma_wall),
        xmax=xmax,
        nr=nr,
    )


def projected_zero_core_compensated_wall(X_eval, x_edge, w_edge, x_wall, sigma_wall, xmax=10.0, nr=1800):
    c = compensation_coefficient(x_edge, w_edge, x_wall, sigma_wall, xmax=xmax)
    if not np.isfinite(c):
        return np.full_like(np.asarray(X_eval, float), np.nan), c

    def q(x):
        return -q_inner(x, x_edge, w_edge) + c * q_wall(x, x_wall, sigma_wall)

    return projected_delta_sigma_from_q(X_eval, q, xmax=xmax, nr=nr), c


def fit_amplitude_cov(y, invcov, shape, amp_bounds=(-1e6, 1e6)):
    evals, evecs = np.linalg.eigh(invcov)
    evals = np.maximum(evals, 0.0)
    W = np.diag(np.sqrt(evals)) @ evecs.T
    A = np.asarray(shape, float)[:, None]
    Aw = W @ A
    yw = W @ y
    res = lsq_linear(Aw, yw, bounds=([amp_bounds[0]], [amp_bounds[1]]), method="trf")
    pred = res.x[0] * shape
    return float(res.x[0]), pred


def scan_models(axis, y, invcov, args):
    n = len(y)
    model_rows = []
    grid_rows = []
    predictions = {}

    # Null.
    pred = np.zeros_like(y)
    chi2 = chi2_cov(y, pred, invcov)
    model_rows.append({"case": "null_zero", **aic_bic(chi2, n, 0)})
    predictions["null_zero"] = pred

    # Gaussian wall only control.
    best_gauss = None
    for xw in np.linspace(args.x_wall_min, args.x_wall_max, args.n_x_wall):
        for sw in np.linspace(args.sigma_wall_min, args.sigma_wall_max, args.n_sigma_wall):
            shape = projected_gaussian_wall(axis, xw, sw, xmax=args.x_project_max, nr=args.n_project_grid)
            if not np.all(np.isfinite(shape)):
                continue
            amp, pred = fit_amplitude_cov(y, invcov, shape, (args.amp_min, args.amp_max))
            chi2 = chi2_cov(y, pred, invcov)
            rec = {
                "case": "projected_gaussian_wall_only_control",
                "x_wall": float(xw),
                "sigma_wall": float(sw),
                "A": amp,
                **aic_bic(chi2, n, 3),
            }
            grid_rows.append(rec)
            if best_gauss is None or rec["BIC"] < best_gauss["BIC"]:
                best_gauss = rec | {"prediction": pred}
    model_rows.append({k: v for k, v in best_gauss.items() if k != "prediction"})
    predictions["projected_gaussian_wall_only_control"] = best_gauss["prediction"]

    # Zero-core uncompensated inner deficit only.
    best_inner = None
    for xe in np.linspace(args.x_edge_min, args.x_edge_max, args.n_x_edge):
        for we in np.linspace(args.w_edge_min, args.w_edge_max, args.n_w_edge):
            shape = projected_delta_sigma_from_q(
                axis,
                lambda x, xe=xe, we=we: -q_inner(x, xe, we),
                xmax=args.x_project_max,
                nr=args.n_project_grid,
            )
            if not np.all(np.isfinite(shape)):
                continue
            amp, pred = fit_amplitude_cov(y, invcov, shape, (args.amp_min, args.amp_max))
            chi2 = chi2_cov(y, pred, invcov)
            rec = {
                "case": "zero_core_uncompensated_deficit",
                "x_edge": float(xe),
                "w_edge": float(we),
                "A": amp,
                **aic_bic(chi2, n, 3),
            }
            grid_rows.append(rec)
            if best_inner is None or rec["BIC"] < best_inner["BIC"]:
                best_inner = rec | {"prediction": pred}
    model_rows.append({k: v for k, v in best_inner.items() if k != "prediction"})
    predictions["zero_core_uncompensated_deficit"] = best_inner["prediction"]

    # DRND zero-core compensated wall.
    best_drnd = None
    for xe in np.linspace(args.x_edge_min, args.x_edge_max, args.n_x_edge):
        for we in np.linspace(args.w_edge_min, args.w_edge_max, args.n_w_edge):
            for xw in np.linspace(args.x_wall_min, args.x_wall_max, args.n_x_wall):
                for sw in np.linspace(args.sigma_wall_min, args.sigma_wall_max, args.n_sigma_wall):
                    shape, ccomp = projected_zero_core_compensated_wall(
                        axis, xe, we, xw, sw,
                        xmax=args.x_project_max,
                        nr=args.n_project_grid,
                    )
                    if not np.all(np.isfinite(shape)):
                        continue
                    amp, pred = fit_amplitude_cov(y, invcov, shape, (args.amp_min, args.amp_max))
                    chi2 = chi2_cov(y, pred, invcov)
                    rec = {
                        "case": "zero_core_compensated_wall_DRND",
                        "x_edge": float(xe),
                        "w_edge": float(we),
                        "x_wall": float(xw),
                        "sigma_wall": float(sw),
                        "C_comp": float(ccomp),
                        "A": amp,
                        **aic_bic(chi2, n, 5),  # A + four geometry params
                    }
                    grid_rows.append(rec)
                    if best_drnd is None or rec["BIC"] < best_drnd["BIC"]:
                        best_drnd = rec | {"prediction": pred}

    model_rows.append({k: v for k, v in best_drnd.items() if k != "prediction"})
    predictions["zero_core_compensated_wall_DRND"] = best_drnd["prediction"]

    # Optional continuous refinement around grid best.
    def obj(p):
        xe, logwe, xw, logsw, amp = p
        we = math.exp(logwe)
        sw = math.exp(logsw)
        if not (args.x_edge_min <= xe <= args.x_edge_max and args.w_edge_min <= we <= args.w_edge_max and args.x_wall_min <= xw <= args.x_wall_max and args.sigma_wall_min <= sw <= args.sigma_wall_max):
            return 1e100
        shape, ccomp = projected_zero_core_compensated_wall(axis, xe, we, xw, sw, xmax=args.x_project_max, nr=args.n_project_grid)
        if not np.all(np.isfinite(shape)):
            return 1e100
        return chi2_cov(y, amp * shape, invcov)

    start = best_drnd
    p0 = np.array([
        start["x_edge"],
        math.log(start["w_edge"]),
        start["x_wall"],
        math.log(start["sigma_wall"]),
        start["A"],
    ], float)
    bounds = [
        (args.x_edge_min, args.x_edge_max),
        (math.log(args.w_edge_min), math.log(args.w_edge_max)),
        (args.x_wall_min, args.x_wall_max),
        (math.log(args.sigma_wall_min), math.log(args.sigma_wall_max)),
        (args.amp_min, args.amp_max),
    ]
    opt = minimize(obj, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": args.maxiter, "maxfun": args.maxfun})
    xe, logwe, xw, logsw, amp = opt.x
    we = math.exp(logwe)
    sw = math.exp(logsw)
    shape, ccomp = projected_zero_core_compensated_wall(axis, xe, we, xw, sw, xmax=args.x_project_max, nr=args.n_project_grid)
    pred = amp * shape
    chi2 = chi2_cov(y, pred, invcov)
    rec = {
        "case": "zero_core_compensated_wall_DRND_refined",
        "x_edge": float(xe),
        "w_edge": float(we),
        "x_wall": float(xw),
        "sigma_wall": float(sw),
        "C_comp": float(ccomp),
        "A": float(amp),
        "success": bool(opt.success),
        "message": str(opt.message),
        **aic_bic(chi2, n, 5),
    }
    model_rows.append(rec)
    predictions["zero_core_compensated_wall_DRND_refined"] = pred

    best = min(model_rows, key=lambda r: r["BIC"])
    for r in model_rows:
        r["delta_BIC_vs_best"] = r["BIC"] - best["BIC"]
        r["delta_AIC_vs_best"] = r["AIC"] - best["AIC"]
    return sorted(model_rows, key=lambda r: r["BIC"]), grid_rows, predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="DRND_VOID_ZERO_CORE_COMPENSATED_WALL")
    ap.add_argument("--rawdir", default=None)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--observable", choices=["xshape", "physical_proxy"], default="xshape")
    ap.add_argument("--rbins", default="0.25,0.4888888889,0.7277777778,0.9666666667,1.2055555556,1.4444444444,1.6833333333,1.9222222222,2.1611111111,2.4")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--cov-rcond", type=float, default=1e-8)

    ap.add_argument("--grid-fast", action="store_true")
    ap.add_argument("--x-edge-min", type=float, default=1.0)
    ap.add_argument("--x-edge-max", type=float, default=2.4)
    ap.add_argument("--w-edge-min", type=float, default=0.05)
    ap.add_argument("--w-edge-max", type=float, default=0.8)
    ap.add_argument("--x-wall-min", type=float, default=1.2)
    ap.add_argument("--x-wall-max", type=float, default=3.6)
    ap.add_argument("--sigma-wall-min", type=float, default=0.05)
    ap.add_argument("--sigma-wall-max", type=float, default=1.2)

    ap.add_argument("--n-x-edge", type=int, default=18)
    ap.add_argument("--n-w-edge", type=int, default=14)
    ap.add_argument("--n-x-wall", type=int, default=28)
    ap.add_argument("--n-sigma-wall", type=int, default=22)
    ap.add_argument("--x-project-max", type=float, default=10.0)
    ap.add_argument("--n-project-grid", type=int, default=1400)
    ap.add_argument("--amp-min", type=float, default=-1e6)
    ap.add_argument("--amp-max", type=float, default=1e6)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--maxfun", type=int, default=1000)
    args = ap.parse_args()

    if args.grid_fast:
        args.n_x_edge = min(args.n_x_edge, 10)
        args.n_w_edge = min(args.n_w_edge, 8)
        args.n_x_wall = min(args.n_x_wall, 15)
        args.n_sigma_wall = min(args.n_sigma_wall, 12)
        args.n_project_grid = min(args.n_project_grid, 800)
        args.nboot = min(args.nboot, 2000)

    outdir = Path(args.outdir)
    tabledir = outdir / "tables"
    plotdir = outdir / "plots"
    rawdir = Path(args.rawdir) if args.rawdir else outdir / "raw"
    tabledir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)
    rawdir.mkdir(parents=True, exist_ok=True)

    ensure_data(rawdir, force=args.force_download)

    try:
        from astropy.io import fits
    except Exception:
        raise SystemExit("Missing dependency astropy. Install: pip install astropy numpy scipy matplotlib")

    hdu = fits.open(rawdir / "collated-sv03s07.fits.gz")
    data = hdu[1].data
    profiles = per_void_profiles(data, observable=args.observable)

    rbins = np.array([float(x) for x in args.rbins.split(",")], dtype=float)
    axis, e_mean, b_mean, e_mat, b_mat, counts = bin_stack_profiles(profiles, rbins)

    cov, err, boot = bootstrap_covariance(e_mat, nboot=args.nboot, seed=args.seed)
    invcov, hartlap = invert_covariance(cov, args.nboot, len(axis), args.cov_rcond)
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))

    profile_rows = []
    for i in range(len(axis)):
        profile_rows.append({
            "bin": i,
            "axis": float(axis[i]),
            "E_mode": float(e_mean[i]),
            "E_err_bootstrap": float(err[i]),
            "B_mode": float(b_mean[i]),
            "n_voids_contributing": int(counts[i]),
            "observable": args.observable,
        })
    write_csv(tabledir / "profile_data.csv", profile_rows)
    write_matrix_csv(tabledir / "covariance.csv", cov)
    write_matrix_csv(tabledir / "correlation.csv", corr)

    model_rows, grid_rows, predictions = scan_models(axis, e_mean, invcov, args)
    write_csv(tabledir / "model_comparison.csv", model_rows, preferred=[
        "case", "chi2", "k_params", "dof", "chi2_per_dof", "AIC", "BIC",
        "delta_AIC_vs_best", "delta_BIC_vs_best", "A", "x_edge", "w_edge",
        "x_wall", "sigma_wall", "C_comp", "success", "message"
    ])
    write_csv(tabledir / "grid_scan.csv", grid_rows, preferred=[
        "case", "chi2", "k_params", "dof", "chi2_per_dof", "AIC", "BIC",
        "A", "x_edge", "w_edge", "x_wall", "sigma_wall", "C_comp"
    ])

    pred_rows = []
    for case, pred in predictions.items():
        for i, p in enumerate(pred):
            pred_rows.append({
                "case": case,
                "axis": float(axis[i]),
                "E_mode_obs": float(e_mean[i]),
                "E_err_bootstrap": float(err[i]),
                "E_mode_pred": float(p),
            })
    write_csv(tabledir / "predictions.csv", pred_rows)

    # Plots.
    order = np.argsort(axis)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(axis, e_mean, yerr=err, fmt="o", label="E-mode data")
    for case in ["null_zero", "projected_gaussian_wall_only_control", "zero_core_uncompensated_deficit", "zero_core_compensated_wall_DRND_refined"]:
        if case in predictions:
            ax.plot(axis[order], predictions[case][order], label=case)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("R/Rv" if args.observable == "xshape" else "native radius")
    ax.set_ylabel("E-mode observable")
    ax.set_title("Zero-core compensated void wall likelihood")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plotdir / "profile_fit.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="correlation")
    ax.set_title("Bootstrap covariance correlation")
    fig.tight_layout()
    fig.savefig(plotdir / "correlation_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = sorted(model_rows, key=lambda r: r["BIC"])
    ax.bar(range(len(ordered)), [r["BIC"] for r in ordered])
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([r["case"] for r in ordered], rotation=35, ha="right")
    ax.set_ylabel("BIC")
    ax.set_title("Zero-core void model comparison")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(plotdir / "model_comparison_bic.png", dpi=180)
    plt.close(fig)

    drnd_grid = [r for r in grid_rows if r["case"] == "zero_core_compensated_wall_DRND"]
    if drnd_grid:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(
            [r["x_wall"] for r in drnd_grid],
            [r["sigma_wall"] for r in drnd_grid],
            c=[r["BIC"] for r in drnd_grid],
            s=12,
        )
        fig.colorbar(sc, ax=ax, label="BIC")
        ax.set_xlabel("x_wall")
        ax.set_ylabel("sigma_wall")
        ax.set_title("Zero-core compensated wall grid")
        fig.tight_layout()
        fig.savefig(plotdir / "grid_bic.png", dpi=180)
        plt.close(fig)

    by = {r["case"]: r for r in model_rows}
    null = by.get("null_zero")
    gauss = by.get("projected_gaussian_wall_only_control")
    inner = by.get("zero_core_uncompensated_deficit")
    drnd = by.get("zero_core_compensated_wall_DRND_refined") or by.get("zero_core_compensated_wall_DRND")
    best = min(model_rows, key=lambda r: r["BIC"])

    def bd(a, b):
        return a["BIC"] - b["BIC"] if a and b else None

    decision = {
        "best_model_by_BIC": best["case"],
        "null_BIC": null["BIC"] if null else None,
        "gaussian_wall_only_control_BIC": gauss["BIC"] if gauss else None,
        "zero_core_uncompensated_deficit_BIC": inner["BIC"] if inner else None,
        "zero_core_compensated_wall_DRND_BIC": drnd["BIC"] if drnd else None,
        "delta_BIC_DRND_minus_null": bd(drnd, null),
        "delta_BIC_DRND_minus_gaussianWallOnly": bd(drnd, gauss),
        "delta_BIC_DRND_minus_uncompensatedDeficit": bd(drnd, inner),
        "preferred_DRND_geometry": {
            "x_edge": drnd.get("x_edge") if drnd else None,
            "w_edge": drnd.get("w_edge") if drnd else None,
            "x_wall": drnd.get("x_wall") if drnd else None,
            "sigma_wall": drnd.get("sigma_wall") if drnd else None,
            "C_comp": drnd.get("C_comp") if drnd else None,
            "A": drnd.get("A") if drnd else None,
        },
        "hartlap_factor": hartlap,
        "n_void_profiles": len(profiles),
        "nboot": args.nboot,
        "observable": args.observable,
        "diagnostic_flags": {
            "DRND_beats_null_BIC": bool(drnd and null and drnd["BIC"] < null["BIC"]),
            "DRND_competes_with_gaussianWallOnly_BIC": bool(drnd and gauss and drnd["BIC"] <= gauss["BIC"] + 6),
            "DRND_beats_gaussianWallOnly_BIC": bool(drnd and gauss and drnd["BIC"] < gauss["BIC"]),
            "DRND_improves_over_uncompensatedDeficit_BIC": bool(drnd and inner and drnd["BIC"] < inner["BIC"]),
        },
    }

    summary = {
        "model": "DRND void zero-core compensated wall likelihood",
        "data_source": {
            "repository": "pmelchior/void-lensing",
            "urls": DATA_URLS,
            "rawdir": str(rawdir),
        },
        "input": vars(args) | {"n_profile_bins": len(axis), "n_void_profiles": len(profiles), "hartlap_factor": hartlap},
        "profile_data": profile_rows,
        "model_comparison": model_rows,
        "decision": decision,
        "publication_claim_rule": {
            "positive_void_claim_requires": [
                "central ATP mass fixed to zero",
                "zero-core compensated wall beats null by BIC",
                "zero-core compensated wall beats or competes with wall-only control",
                "physical-unit amplitude validation before strong claim",
            ],
        },
        "cautions": [
            "This is a zero-core boundary-transport test, not a galaxy ATP test.",
            "xshape mode is primarily a shape/sign test; physical amplitude requires a unit-audited profile.",
            "The compensation coefficient is fixed by integral compensation in dimensionless x, not fitted freely.",
        ],
        "outputs": {
            "profile_data": "tables/profile_data.csv",
            "covariance": "tables/covariance.csv",
            "correlation": "tables/correlation.csv",
            "model_comparison": "tables/model_comparison.csv",
            "grid_scan": "tables/grid_scan.csv",
            "predictions": "tables/predictions.csv",
            "profile_plot": "plots/profile_fit.png",
            "correlation_plot": "plots/correlation_matrix.png",
            "bic_plot": "plots/model_comparison_bic.png",
            "grid_plot": "plots/grid_bic.png",
        },
    }

    (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
    (outdir / "methods_zero_core_compensated_wall.md").write_text(
        "We removed the central ATP mass source from the void-sector model and tested a zero-core compensated boundary transport profile. "
        "The 3D effective contrast q(x) consists of an empty interior plus a compensation wall, with the wall coefficient fixed by integral compensation. "
        "The profile is Abel-projected before comparison with the stacked SDSS void-lensing E-mode profile using a bootstrap covariance over voids.\n",
        encoding="utf-8",
    )
    print(json.dumps(safe_json({
        "outdir": str(outdir),
        "decision": decision,
        "files": list(summary["outputs"].values()) + ["summary.json", "methods_zero_core_compensated_wall.md"],
    }), indent=2))


if __name__ == "__main__":
    main()

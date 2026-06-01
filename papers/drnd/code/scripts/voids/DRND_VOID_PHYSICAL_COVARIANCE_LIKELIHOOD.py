#!/usr/bin/env python3
"""
DRND_VOID_PHYSICAL_COVARIANCE_LIKELIHOOD.py

Complete publication-oriented void test for DRND-ST/ATP using the best
immediately reproducible public data source: the SDSS void-lensing release
from pmelchior/void-lensing.

Why this script exists
----------------------
Previous DRND void tests used a rebinned profile with diagonal bootstrap
errors only. That is insufficient for a positive claim. This script builds a
properer likelihood layer:

  1. downloads the public SDSS void-lensing FITS data,
  2. builds a stacked E-mode profile,
  3. estimates a full radial covariance by bootstrapping voids,
  4. compares null, projected-wall-only, ATP-only, and ATP+universal-wall
     models using chi2 = (d-m)^T C^{-1} (d-m),
  5. reports AIC/BIC and a publication decision table.

Data source
-----------
Public repository:
  https://github.com/pmelchior/void-lensing

The repository contains the data/code for the SDSS cosmic-void lensing
measurement. The SDSS paper reports a stacked weak-lensing signal for 901
voids and releases data/code publicly.

Important unit note
-------------------
The original repository works primarily in r/Rv rebinned coordinates and
DeltaSigma/Rv conventions. This script supports two likelihood modes:

  --observable xshape
      uses X = R/Rv and the source's rebinned E-mode convention.
      This is the safest fully reproducible public test of shape/sign.

  --observable physical_proxy
      uses the native non-rebinned profile dsum/wsum and rsum/npair.
      This is closer to physical R and DeltaSigma, but the exact unit
      convention of the old SDSS files must be checked before a strong
      amplitude claim.

Recommended initial run
-----------------------
python DRND_VOID_PHYSICAL_COVARIANCE_LIKELIHOOD.py ^
  --outdir DRND_VOID_LIKELIHOOD_SDSS ^
  --observable xshape ^
  --nboot 2000 ^
  --grid-fast

Stronger run
------------
python DRND_VOID_PHYSICAL_COVARIANCE_LIKELIHOOD.py ^
  --outdir DRND_VOID_LIKELIHOOD_SDSS_FULL ^
  --observable xshape ^
  --nboot 10000

Outputs
-------
summary.json
tables/profile_data.csv
tables/covariance.csv
tables/correlation.csv
tables/model_comparison.csv
tables/wall_grid_scan.csv
tables/bootstrap_manifest.csv
plots/profile_fit.png
plots/correlation_matrix.png
plots/model_comparison_bic.png
plots/wall_grid_bic.png

Publication rule
----------------
A positive DRND void claim requires:

  ATP_fixed_kappa_plus_universal_projected_wall beats null by BIC
  AND
  ATP_fixed_kappa_plus_universal_projected_wall beats or competes with
  projected_wall_only_control.

If projected_wall_only_control wins, the void data support a boundary wall but
do not require ATP.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, lsq_linear

G_SI = 6.67430e-11
MSUN_KG = 1.98847e30
PC_M = 3.0856775814913673e16
MPC_M = 1.0e6 * PC_M
C_M_S = 299792458.0

DATA_URLS = {
    "collated-sv03s07.fits.gz": "https://github.com/pmelchior/void-lensing/raw/master/data/collated-sv03s07.fits.gz",
    "sky_positions_central.txt": "https://raw.githubusercontent.com/pmelchior/void-lensing/master/data/sky_positions_central.txt",
    "DSigma_LW12_PM.dat": "https://raw.githubusercontent.com/pmelchior/void-lensing/master/data/DSigma_LW12_PM.dat",
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


def ensure_sdss_data(rawdir: Path, force=False):
    for name, url in DATA_URLS.items():
        download(url, rawdir / name, force=force)


def a0_from_chi(H0, chi):
    return chi * C_M_S * (H0 * 1000.0 / MPC_M)


def get_native_arrays(data):
    required = ["radius", "dsum", "osum", "rsum", "wsum", "npair"]
    missing = [k for k in required if k not in data.names]
    if missing:
        raise SystemExit(f"FITS table missing columns: {missing}. Available: {data.names}")

    radius = np.asarray(data["radius"], dtype=float)
    dsum = np.asarray(data["dsum"], dtype=float)
    osum = np.asarray(data["osum"], dtype=float)
    rsum = np.asarray(data["rsum"], dtype=float)
    wsum = np.asarray(data["wsum"], dtype=float)
    npair = np.asarray(data["npair"], dtype=float)
    return radius, dsum, osum, rsum, wsum, npair


def per_void_profiles(data, observable="xshape"):
    """
    Return list of per-void profiles as dicts with arrays:
      x, e, b, weight

    observable='xshape':
        X = rmean / Rv
        E = dsum/wsum/Rv
        follows the repository's rebinned convention.

    observable='physical_proxy':
        R = rmean
        E = dsum/wsum
        closer to native physical profile but unit convention remains source-defined.
    """
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
    """
    Stack profiles into fixed radial bins. Each void contributes a weighted
    mean per bin if it has points in the bin. Then stack over voids.

    Returns:
      mean E, mean B, per-void E matrix [nvoid, nbins] with NaN where missing.
    """
    nb = len(bin_edges) - 1
    e_mat = np.full((len(profiles), nb), np.nan)
    b_mat = np.full((len(profiles), nb), np.nan)

    for i, p in enumerate(profiles):
        x = p["x"]
        e = p["e"]
        b = p["b"]
        w = np.maximum(p["w"], 1.0)
        for j in range(nb):
            m = (x >= bin_edges[j]) & (x < bin_edges[j + 1])
            if np.any(m):
                e_mat[i, j] = np.average(e[m], weights=w[m])
                b_mat[i, j] = np.average(b[m], weights=w[m])

    e_mean = np.nanmean(e_mat, axis=0)
    b_mean = np.nanmean(b_mat, axis=0)
    counts = np.sum(np.isfinite(e_mat), axis=0)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, e_mean, b_mean, e_mat, b_mat, counts


def bootstrap_covariance(e_mat, nboot=2000, seed=12345, min_valid_fraction=0.5):
    rng = np.random.default_rng(seed)
    nvoid, nb = e_mat.shape

    # Replace missing values by column means for bootstrap resampling, but keep
    # track of validity. This is pragmatic for the sparse public FITS stacks.
    col_mean = np.nanmean(e_mat, axis=0)
    filled = np.where(np.isfinite(e_mat), e_mat, col_mean[None, :])

    boot = np.empty((nboot, nb), dtype=float)
    for k in range(nboot):
        idx = rng.integers(0, nvoid, size=nvoid)
        boot[k] = np.mean(filled[idx, :], axis=0)

    cov = np.cov(boot, rowvar=False, ddof=1)
    err = np.sqrt(np.diag(cov))
    return cov, err, boot


def invert_covariance(cov, nboot=None, ndata=None, rcond=1e-8):
    """
    Pseudo-inverse with optional Hartlap correction if nboot is sufficiently
    large relative to ndata.
    """
    cov = np.asarray(cov, float)
    inv = np.linalg.pinv(cov, rcond=rcond)
    hartlap = 1.0
    if nboot is not None and ndata is not None and nboot > ndata + 2:
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


def atp_ds(R_axis, logMdef, a0, kappa, observable):
    """
    Positive ATP magnitude. Void sign is applied externally.

    In xshape mode, R_axis is dimensionless X=R/Rv. We map it to Mpc with
    Rv=1 as a shape/amplitude proxy. In physical_proxy mode, R_axis is the
    native radius from the source file.
    """
    R_mpc = np.asarray(R_axis, float)
    R = R_mpc * MPC_M
    M = (10 ** float(logMdef)) * MSUN_KG
    if M <= 0:
        return np.zeros_like(R)
    r0 = math.sqrt(G_SI * M / a0)
    A = kappa * math.sqrt(G_SI * M * a0) / G_SI
    R2 = R * R
    root = np.sqrt(R2 + r0 * r0)
    sigma_bar = (2.0 * A / np.maximum(R2, 1e-20)) * (root - r0)
    sigma = A / root
    ds = sigma_bar - sigma
    ds = np.where(R2 < 1e-16, 0.0, ds)
    return ds * (PC_M ** 2) / MSUN_KG


def projected_wall_delta_sigma(X_eval, x_wall, sigma_wall, xmax=10.0, nr=1600):
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
        integ = rh * r / np.sqrt(np.maximum(r * r - X * X, 1e-300))
        return 2.0 * np.trapezoid(integ, r)

    sigma_eval = np.array([sigma_at(x) for x in X_eval], float)

    xd = np.geomspace(x_min, max(np.max(X_eval) * 1.02, x_min * 2), 500)
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


def fit_linear_cov(y, invcov, components, bounds):
    """
    Approximate covariance-weighted linear fit by whitening via Cholesky/eigh.
    """
    Cinv = np.asarray(invcov, float)
    evals, evecs = np.linalg.eigh(Cinv)
    evals = np.maximum(evals, 0.0)
    W = np.diag(np.sqrt(evals)) @ evecs.T

    A = np.vstack(components).T
    Aw = W @ A
    yw = W @ y

    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    res = lsq_linear(Aw, yw, bounds=(lo, hi), method="trf", lsmr_tol="auto")
    pred = A @ res.x
    return res.x, pred


def scan_models(axis, y, invcov, args, a0):
    n = len(y)
    rows = []
    predictions = {}

    # Null
    pred_null = np.zeros_like(y)
    chi2 = chi2_cov(y, pred_null, invcov)
    rec = {"case": "null_zero", **aic_bic(chi2, n, 0)}
    rows.append(rec)
    predictions["null_zero"] = pred_null

    # Pure ATP: grid in logMdef
    best = None
    for lm in np.linspace(args.logMdef_min, args.logMdef_max, args.n_logMdef):
        pred = -atp_ds(axis, lm, a0, args.kappa, args.observable)
        chi2 = chi2_cov(y, pred, invcov)
        rec = {"case": "atp_fixed_kappa_only", "logMdef": float(lm), **aic_bic(chi2, n, 1)}
        if best is None or rec["BIC"] < best["BIC"]:
            best = rec | {"prediction": pred}
    rows.append({k: v for k, v in best.items() if k != "prediction"})
    predictions["atp_fixed_kappa_only"] = best["prediction"]

    # Universal projected wall grid
    x_grid = np.linspace(args.x_wall_min, args.x_wall_max, args.n_x_wall)
    s_grid = np.linspace(args.sigma_wall_min, args.sigma_wall_max, args.n_sigma_wall)
    m_grid = np.linspace(args.logMdef_min, args.logMdef_max, args.n_logMdef)

    grid_rows = []
    best_wall = None
    best_atp_wall = None

    for xw in x_grid:
        for sw in s_grid:
            wall = projected_wall_delta_sigma(axis, xw, sw, xmax=args.x_project_max, nr=args.n_project_grid)
            if not np.all(np.isfinite(wall)):
                continue

            amp, pred = fit_linear_cov(y, invcov, [wall], [(args.wall_amp_min, args.wall_amp_max)])
            chi2 = chi2_cov(y, pred, invcov)
            rec = {
                "case": "projected_wall_only_control",
                "x_wall": float(xw),
                "sigma_wall": float(sw),
                "A_wall": float(amp[0]),
                **aic_bic(chi2, n, 3),
            }
            grid_rows.append(rec)
            if best_wall is None or rec["BIC"] < best_wall["BIC"]:
                best_wall = rec | {"prediction": pred}

            for lm in m_grid:
                atp = -atp_ds(axis, lm, a0, args.kappa, args.observable)
                amp, wall_pred = fit_linear_cov(y - atp, invcov, [wall], [(args.wall_amp_min, args.wall_amp_max)])
                pred2 = atp + amp[0] * wall
                chi22 = chi2_cov(y, pred2, invcov)
                rec2 = {
                    "case": "atp_fixed_kappa_plus_universal_projected_wall",
                    "x_wall": float(xw),
                    "sigma_wall": float(sw),
                    "logMdef": float(lm),
                    "A_wall": float(amp[0]),
                    **aic_bic(chi22, n, 4),
                }
                grid_rows.append(rec2)
                if best_atp_wall is None or rec2["BIC"] < best_atp_wall["BIC"]:
                    best_atp_wall = rec2 | {"prediction": pred2}

    rows.append({k: v for k, v in best_wall.items() if k != "prediction"})
    rows.append({k: v for k, v in best_atp_wall.items() if k != "prediction"})
    predictions["projected_wall_only_control"] = best_wall["prediction"]
    predictions["atp_fixed_kappa_plus_universal_projected_wall"] = best_atp_wall["prediction"]

    # Optional continuous refinement of ATP+wall
    def obj(p):
        logM, xw, logs, A = p
        sw = math.exp(logs)
        if not (args.logMdef_min <= logM <= args.logMdef_max and args.x_wall_min <= xw <= args.x_wall_max and args.sigma_wall_min <= sw <= args.sigma_wall_max):
            return 1e100
        wall = projected_wall_delta_sigma(axis, xw, sw, xmax=args.x_project_max, nr=args.n_project_grid)
        if not np.all(np.isfinite(wall)):
            return 1e100
        atp = -atp_ds(axis, logM, a0, args.kappa, args.observable)
        pred = atp + A * wall
        return chi2_cov(y, pred, invcov)

    start = best_atp_wall
    p0 = np.array([
        float(start.get("logMdef", 14.0)),
        float(start["x_wall"]),
        math.log(float(start["sigma_wall"])),
        float(start["A_wall"]),
    ])
    opt = minimize(
        obj, p0, method="L-BFGS-B",
        bounds=[
            (args.logMdef_min, args.logMdef_max),
            (args.x_wall_min, args.x_wall_max),
            (math.log(args.sigma_wall_min), math.log(args.sigma_wall_max)),
            (args.wall_amp_min, args.wall_amp_max),
        ],
        options={"maxiter": args.maxiter, "maxfun": args.maxfun, "ftol": 1e-8},
    )
    logM, xw, logs, A = opt.x
    sw = math.exp(logs)
    wall = projected_wall_delta_sigma(axis, xw, sw, xmax=args.x_project_max, nr=args.n_project_grid)
    atp = -atp_ds(axis, logM, a0, args.kappa, args.observable)
    pred = atp + A * wall
    chi2 = chi2_cov(y, pred, invcov)
    rec = {
        "case": "atp_fixed_kappa_plus_universal_projected_wall_refined",
        "logMdef": float(logM),
        "x_wall": float(xw),
        "sigma_wall": float(sw),
        "A_wall": float(A),
        "success": bool(opt.success),
        "message": str(opt.message),
        **aic_bic(chi2, n, 4),
    }
    rows.append(rec)
    predictions[rec["case"]] = pred

    # Sort and add deltas
    best_row = min(rows, key=lambda r: r["BIC"])
    for r in rows:
        r["delta_BIC_vs_best"] = r["BIC"] - best_row["BIC"]
        r["delta_AIC_vs_best"] = r["AIC"] - best_row["AIC"]
    rows = sorted(rows, key=lambda r: r["BIC"])
    return rows, grid_rows, predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="DRND_VOID_PHYSICAL_COVARIANCE_LIKELIHOOD")
    ap.add_argument("--rawdir", default=None)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--observable", choices=["xshape", "physical_proxy"], default="xshape")
    ap.add_argument("--rbins", default="0.25,0.4888888889,0.7277777778,0.9666666667,1.2055555556,1.4444444444,1.6833333333,1.9222222222,2.1611111111,2.4")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--cov-rcond", type=float, default=1e-8)

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
        args.nboot = min(args.nboot, 2000)

    outdir = Path(args.outdir)
    tabledir = outdir / "tables"
    plotdir = outdir / "plots"
    rawdir = Path(args.rawdir) if args.rawdir else outdir / "raw"
    tabledir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)
    rawdir.mkdir(parents=True, exist_ok=True)

    ensure_sdss_data(rawdir, force=args.force_download)

    try:
        from astropy.io import fits
    except Exception:
        raise SystemExit("Missing dependency astropy. Install: pip install astropy numpy scipy matplotlib")

    hdu = fits.open(rawdir / "collated-sv03s07.fits.gz")
    data = hdu[1].data
    profiles = per_void_profiles(data, observable=args.observable)
    if len(profiles) < 10:
        raise SystemExit(f"Too few usable void profiles: {len(profiles)}")

    rbins = np.array([float(x) for x in args.rbins.split(",")], dtype=float)
    axis, e_mean, b_mean, e_mat, b_mat, counts = bin_stack_profiles(profiles, rbins)

    cov, err, boot = bootstrap_covariance(e_mat, nboot=args.nboot, seed=args.seed)
    invcov, hartlap = invert_covariance(cov, nboot=args.nboot, ndata=len(axis), rcond=args.cov_rcond)
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))

    profile_rows = []
    for i in range(len(axis)):
        profile_rows.append({
            "bin": i,
            "axis": float(axis[i]),
            "E_mode": float(e_mean[i]),
            "B_mode": float(b_mean[i]),
            "E_err_bootstrap": float(err[i]),
            "n_voids_contributing": int(counts[i]),
            "observable": args.observable,
        })
    write_csv(tabledir / "profile_data.csv", profile_rows, preferred=["bin", "axis", "E_mode", "E_err_bootstrap", "B_mode", "n_voids_contributing", "observable"])
    write_matrix_csv(tabledir / "covariance.csv", cov)
    write_matrix_csv(tabledir / "correlation.csv", corr)

    a0 = a0_from_chi(args.H0, args.chi)
    model_rows, grid_rows, predictions = scan_models(axis, e_mean, invcov, args, a0)

    write_csv(tabledir / "model_comparison.csv", model_rows, preferred=[
        "case", "chi2", "k_params", "dof", "chi2_per_dof", "AIC", "BIC",
        "delta_AIC_vs_best", "delta_BIC_vs_best", "logMdef", "A_wall", "x_wall",
        "sigma_wall", "success", "message"
    ])
    write_csv(tabledir / "wall_grid_scan.csv", grid_rows, preferred=[
        "case", "x_wall", "sigma_wall", "logMdef", "A_wall", "chi2", "k_params", "dof", "chi2_per_dof", "AIC", "BIC"
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
    write_csv(tabledir / "predictions.csv", pred_rows, preferred=["case", "axis", "E_mode_obs", "E_err_bootstrap", "E_mode_pred"])

    # plots
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(axis, e_mean, yerr=err, fmt="o", label="E-mode data")
    order = np.argsort(axis)
    for case in ["null_zero", "projected_wall_only_control", "atp_fixed_kappa_only", "atp_fixed_kappa_plus_universal_projected_wall_refined"]:
        if case in predictions:
            ax.plot(axis[order], predictions[case][order], label=case)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("R/Rv" if args.observable == "xshape" else "native radius")
    ax.set_ylabel("E-mode observable")
    ax.set_title(f"SDSS void lensing covariance likelihood ({args.observable})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
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
    ax.set_title("DRND void likelihood model comparison")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(plotdir / "model_comparison_bic.png", dpi=180)
    plt.close(fig)

    atp_grid = [r for r in grid_rows if r["case"] == "atp_fixed_kappa_plus_universal_projected_wall"]
    if atp_grid:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(
            [r["x_wall"] for r in atp_grid],
            [r["sigma_wall"] for r in atp_grid],
            c=[r["BIC"] for r in atp_grid],
            s=12,
        )
        fig.colorbar(sc, ax=ax, label="BIC")
        ax.set_xlabel("x_wall")
        ax.set_ylabel("sigma_wall")
        ax.set_title("ATP + universal wall grid")
        fig.tight_layout()
        fig.savefig(plotdir / "wall_grid_bic.png", dpi=180)
        plt.close(fig)

    by = {r["case"]: r for r in model_rows}
    null = by.get("null_zero")
    wall = by.get("projected_wall_only_control")
    atp = by.get("atp_fixed_kappa_only")
    atpw = by.get("atp_fixed_kappa_plus_universal_projected_wall")
    atpwr = by.get("atp_fixed_kappa_plus_universal_projected_wall_refined") or atpw
    best = min(model_rows, key=lambda r: r["BIC"])

    def bd(a, b):
        return a["BIC"] - b["BIC"] if a and b else None

    decision = {
        "best_model_by_BIC": best["case"],
        "null_BIC": null["BIC"] if null else None,
        "projected_wall_only_control_BIC": wall["BIC"] if wall else None,
        "ATP_fixed_kappa_only_BIC": atp["BIC"] if atp else None,
        "ATP_fixed_kappa_plus_universal_projected_wall_BIC": atpw["BIC"] if atpw else None,
        "ATP_fixed_kappa_plus_universal_projected_wall_refined_BIC": atpwr["BIC"] if atpwr else None,
        "delta_BIC_ATPwall_minus_null": bd(atpwr, null),
        "delta_BIC_ATPwall_minus_wallOnly": bd(atpwr, wall),
        "delta_BIC_ATPwall_minus_ATPonly": bd(atpwr, atp),
        "preferred_ATPwall_x_wall": atpwr.get("x_wall") if atpwr else None,
        "preferred_ATPwall_sigma_wall": atpwr.get("sigma_wall") if atpwr else None,
        "preferred_ATPwall_logMdef": atpwr.get("logMdef") if atpwr else None,
        "hartlap_factor": hartlap,
        "n_void_profiles": len(profiles),
        "nboot": args.nboot,
        "observable": args.observable,
        "diagnostic_flags": {
            "ATPwall_beats_null_BIC": bool(atpwr and null and atpwr["BIC"] < null["BIC"]),
            "ATPwall_competes_with_wallOnly_BIC": bool(atpwr and wall and atpwr["BIC"] <= wall["BIC"] + 6),
            "ATPwall_beats_wallOnly_BIC": bool(atpwr and wall and atpwr["BIC"] < wall["BIC"]),
            "ATPwall_improves_over_ATPonly_BIC": bool(atpwr and atp and atpwr["BIC"] < atp["BIC"]),
        },
    }

    summary = {
        "model": "DRND-ST void physical/covariance likelihood",
        "data_source": {
            "repository": "pmelchior/void-lensing",
            "urls": DATA_URLS,
            "rawdir": str(rawdir),
        },
        "input": vars(args) | {"a0_fixed": a0, "n_profile_bins": len(axis), "n_void_profiles": len(profiles), "hartlap_factor": hartlap},
        "profile_data": profile_rows,
        "model_comparison": model_rows,
        "decision": decision,
        "publication_claim_rule": {
            "positive_void_claim_requires": [
                "fixed kappa=1/4",
                "ATP+universal wall beats null by BIC",
                "ATP+universal wall beats or competes with wall-only control",
                "physical-unit amplitude validation before strong claim",
            ],
        },
        "cautions": [
            "xshape mode is the safest public reproducibility test but is primarily a shape/sign test.",
            "physical_proxy mode is closer to native units but the exact old-source unit convention must be audited before amplitude claims.",
            "This script estimates covariance by void bootstrap; a final survey likelihood should use the survey team's covariance/random tests if available.",
        ],
        "outputs": {
            "profile_data": "tables/profile_data.csv",
            "covariance": "tables/covariance.csv",
            "correlation": "tables/correlation.csv",
            "model_comparison": "tables/model_comparison.csv",
            "wall_grid_scan": "tables/wall_grid_scan.csv",
            "predictions": "tables/predictions.csv",
            "profile_plot": "plots/profile_fit.png",
            "correlation_plot": "plots/correlation_matrix.png",
            "bic_plot": "plots/model_comparison_bic.png",
            "wall_grid_plot": "plots/wall_grid_bic.png",
        },
    }
    (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
    (outdir / "methods_void_likelihood.md").write_text(
        "This likelihood downloads the public SDSS void-lensing data, stacks per-void E-mode profiles, estimates a full radial covariance by bootstrapping void profiles, and compares fixed-kappa DRND-ST/ATP models against null and universal-wall controls using chi2=(d-m)^T C^{-1}(d-m).\n",
        encoding="utf-8",
    )

    print(json.dumps(safe_json({
        "outdir": str(outdir),
        "decision": decision,
        "files": list(summary["outputs"].values()) + ["summary.json", "methods_void_likelihood.md"],
    }), indent=2))


if __name__ == "__main__":
    main()

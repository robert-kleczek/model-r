#!/usr/bin/env python3
"""
DRND_VOID_LOCKED_PHYSICAL_AMPLITUDE_TEST.py

Locked physical-amplitude validation for the DRND zero-core compensated void wall.

Purpose
-------
The previous xshape likelihood allowed a free dimensionless amplitude A and found
a strong shape-level result. This script removes that freedom.

It:
  1. reads/downloads the public SDSS void-lensing data,
  2. extracts the mean/median void radius Rv and redshift z when available,
  3. builds the stacked E-mode profile and bootstrap covariance,
  4. reads the fixed DRND wall geometry from a previous summary.json,
  5. converts the dimensionless compensated-wall projection to physical/source
     units using

        DeltaSigma_physical(R) = rho_bar(z) * Rv * DeltaSigma_norm(R/Rv)

     and, for the SDSS source convention,

        DeltaSigma/Rv = rho_bar(z) * DeltaSigma_norm.

No amplitude refit is used for the primary physical tests.

Key output
----------
tables/model_comparison_locked_physical.csv

It compares:
  - null_zero
  - locked_total_matter_comoving
  - locked_total_matter_physical_z
  - locked_baryon_comoving
  - locked_baryon_physical_z
  - diagnostic_best_free_amplitude  [diagnostic only]

Interpretation
--------------
If a locked physical model beats null and is close to the diagnostic best-free
amplitude, the DRND void amplitude is physically viable.

If only diagnostic_best_free_amplitude works, the shape is supported but the
absolute physical normalization is not yet derived correctly.

Usage
-----
python DRND_VOID_LOCKED_PHYSICAL_AMPLITUDE_TEST.py ^
  --geometry-summary DRND_VOID_ZERO_CORE_FAST/summary.json ^
  --outdir DRND_VOID_LOCKED_PHYSICAL_TEST ^
  --nboot 10000

Fast:
python DRND_VOID_LOCKED_PHYSICAL_AMPLITUDE_TEST.py ^
  --geometry-summary DRND_VOID_ZERO_CORE_FAST/summary.json ^
  --outdir DRND_VOID_LOCKED_PHYSICAL_TEST_FAST ^
  --nboot 2000

Optional:
  --Omega-b 0.049
  --Omega-m 0.315
  --h 0.674
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

DATA_URLS = {
    "collated-sv03s07.fits.gz": "https://github.com/pmelchior/void-lensing/raw/master/data/collated-sv03s07.fits.gz",
    "sky_positions_central.txt": "https://raw.githubusercontent.com/pmelchior/void-lensing/master/data/sky_positions_central.txt",
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


def norm_col(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def find_col(names, candidates):
    norm = {norm_col(n): n for n in names}
    for c in candidates:
        if norm_col(c) in norm:
            return norm[norm_col(c)]
    # relaxed contains
    for n in names:
        nn = norm_col(n)
        for c in candidates:
            cc = norm_col(c)
            if cc and cc in nn:
                return n
    return None


def extract_catalog_stats(rawdir: Path, fits_data=None):
    """
    Try to extract mean/median Rv and z from sky_positions_central.txt.
    Fall back to FITS radius for Rv if needed.
    """
    stats = {
        "source": None,
        "n_catalog": None,
        "Rv_mean_source": None,
        "Rv_median_source": None,
        "Rv_min_source": None,
        "Rv_max_source": None,
        "z_mean": None,
        "z_median": None,
        "z_min": None,
        "z_max": None,
        "warnings": [],
    }

    cat_path = rawdir / "sky_positions_central.txt"
    if cat_path.exists():
        try:
            arr = np.genfromtxt(cat_path, names=True, dtype=None, encoding=None)
            names = arr.dtype.names or []
            zcol = find_col(names, ["z", "redshift", "z_void"])
            rcol = find_col(names, ["radius", "rvoid", "r_void", "rv", "radius_mpc"])

            stats["source"] = str(cat_path)
            stats["n_catalog"] = int(len(arr))

            if zcol:
                z = np.asarray(arr[zcol], dtype=float)
                z = z[np.isfinite(z)]
                stats.update({
                    "z_mean": float(np.mean(z)),
                    "z_median": float(np.median(z)),
                    "z_min": float(np.min(z)),
                    "z_max": float(np.max(z)),
                    "z_column": zcol,
                })
            else:
                stats["warnings"].append(f"No redshift column detected in {cat_path}; columns={names}")

            if rcol:
                rv = np.asarray(arr[rcol], dtype=float)
                rv = rv[np.isfinite(rv)]
                stats.update({
                    "Rv_mean_source": float(np.mean(rv)),
                    "Rv_median_source": float(np.median(rv)),
                    "Rv_min_source": float(np.min(rv)),
                    "Rv_max_source": float(np.max(rv)),
                    "Rv_column": rcol,
                })
            else:
                stats["warnings"].append(f"No radius column detected in {cat_path}; columns={names}")
        except Exception as exc:
            stats["warnings"].append(f"Could not parse {cat_path}: {exc}")

    if stats["Rv_mean_source"] is None and fits_data is not None and "radius" in fits_data.names:
        rv = np.asarray(fits_data["radius"], dtype=float)
        rv = rv[np.isfinite(rv)]
        stats.update({
            "source": (stats["source"] or "") + " + FITS radius fallback",
            "n_catalog": int(len(rv)),
            "Rv_mean_source": float(np.mean(rv)),
            "Rv_median_source": float(np.median(rv)),
            "Rv_min_source": float(np.min(rv)),
            "Rv_max_source": float(np.max(rv)),
            "Rv_column": "FITS:radius",
        })

    return stats


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
    grid = np.geomspace(1e-4, xmax, 5000)
    inner = q_inner(grid, x_edge, w_edge)
    wall = q_wall(grid, x_wall, sigma_wall)
    i_inner = np.trapezoid(inner * grid * grid, grid)
    i_wall = np.trapezoid(wall * grid * grid, grid)
    return float(i_inner / i_wall)


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

    xd = np.geomspace(x_min, max(np.max(X_eval) * 1.02, x_min * 2), 700)
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


def zero_core_compensated_shape(X, x_edge, w_edge, x_wall, sigma_wall, xmax=10.0, nr=1800):
    c = compensation_coefficient(x_edge, w_edge, x_wall, sigma_wall, xmax=xmax)
    def q(x):
        return -q_inner(x, x_edge, w_edge) + c * q_wall(x, x_wall, sigma_wall)
    shape = projected_delta_sigma_from_q(X, q, xmax=xmax, nr=nr)
    return shape, c


def fit_amplitude_cov(y, invcov, shape):
    evals, evecs = np.linalg.eigh(invcov)
    evals = np.maximum(evals, 0.0)
    W = np.diag(np.sqrt(evals)) @ evecs.T
    A = np.asarray(shape, float)[:, None]
    Aw = W @ A
    yw = W @ y
    res = lsq_linear(Aw, yw, bounds=([-1e9], [1e9]), method="trf")
    pred = res.x[0] * shape
    return float(res.x[0]), pred


def load_geometry(summary_path: Path):
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    geom = obj.get("decision", {}).get("preferred_DRND_geometry", {})
    if not geom:
        raise SystemExit(f"No decision.preferred_DRND_geometry found in {summary_path}")
    required = ["x_edge", "w_edge", "x_wall", "sigma_wall"]
    missing = [k for k in required if geom.get(k) is None]
    if missing:
        raise SystemExit(f"Geometry missing fields: {missing}")
    return {
        "x_edge": float(geom["x_edge"]),
        "w_edge": float(geom["w_edge"]),
        "x_wall": float(geom["x_wall"]),
        "sigma_wall": float(geom["sigma_wall"]),
        "A_fit_previous": float(geom["A"]) if geom.get("A") is not None else None,
        "C_comp_previous": float(geom["C_comp"]) if geom.get("C_comp") is not None else None,
        "source_summary": str(summary_path),
    }


def rho_unit_amplitudes(Omega_m, Omega_b, h, z):
    """
    Source convention for SDSS xshape profile:
      observed E is DeltaSigma/Rv in units 1e12 h^2 Msun Mpc^-3.
    Since DeltaSigma/Rv = rho_bar * DeltaSigma_norm,
      A_source = rho_bar / (1e12 h^2 Msun Mpc^-3).

    rho_crit0 = 2.77536627e11 h^2 Msun Mpc^-3.
    """
    rho_crit_unit = 2.77536627e11  # h^2 Msun / Mpc^3
    zp = (1.0 + z) ** 3
    return {
        "total_matter_comoving": Omega_m * rho_crit_unit / 1e12,
        "total_matter_physical_z": Omega_m * rho_crit_unit * zp / 1e12,
        "baryon_comoving": Omega_b * rho_crit_unit / 1e12,
        "baryon_physical_z": Omega_b * rho_crit_unit * zp / 1e12,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry-summary", required=True)
    ap.add_argument("--outdir", default="DRND_VOID_LOCKED_PHYSICAL_AMPLITUDE_TEST")
    ap.add_argument("--rawdir", default=None)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--observable", choices=["xshape"], default="xshape")
    ap.add_argument("--rbins", default="0.25,0.4888888889,0.7277777778,0.9666666667,1.2055555556,1.4444444444,1.6833333333,1.9222222222,2.1611111111,2.4")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--cov-rcond", type=float, default=1e-8)
    ap.add_argument("--Omega-m", dest="Omega_m", type=float, default=0.315)
    ap.add_argument("--Omega-b", dest="Omega_b", type=float, default=0.049)
    ap.add_argument("--h", type=float, default=0.674)
    ap.add_argument("--topological-gain", type=float, default=9.5, help="DRND graph-wall gain; default 19/2 = 9.5.")
    ap.add_argument("--topological-gain-label", default="19_over_2", help="Label for the locked topological-gain model.")
    ap.add_argument("--z-override", type=float, default=None)
    ap.add_argument("--x-project-max", type=float, default=10.0)
    ap.add_argument("--n-project-grid", type=int, default=1800)
    args = ap.parse_args()

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

    catalog_stats = extract_catalog_stats(rawdir, fits_data=data)
    if args.z_override is not None:
        z_use = float(args.z_override)
        catalog_stats["z_used"] = z_use
        catalog_stats["z_used_source"] = "CLI override"
    elif catalog_stats.get("z_mean") is not None:
        z_use = float(catalog_stats["z_mean"])
        catalog_stats["z_used"] = z_use
        catalog_stats["z_used_source"] = "catalog mean"
    else:
        z_use = 0.1
        catalog_stats["z_used"] = z_use
        catalog_stats["z_used_source"] = "fallback 0.1"
        catalog_stats["warnings"].append("No z detected; using fallback z=0.1")

    geom = load_geometry(Path(args.geometry_summary))

    profiles = per_void_profiles(data, observable=args.observable)
    rbins = np.array([float(x) for x in args.rbins.split(",")], dtype=float)
    axis, e_mean, b_mean, e_mat, b_mat, counts = bin_stack_profiles(profiles, rbins)
    cov, err, boot = bootstrap_covariance(e_mat, nboot=args.nboot, seed=args.seed)
    invcov, hartlap = invert_covariance(cov, args.nboot, len(axis), args.cov_rcond)
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))

    shape, ccomp = zero_core_compensated_shape(
        axis,
        geom["x_edge"], geom["w_edge"], geom["x_wall"], geom["sigma_wall"],
        xmax=args.x_project_max,
        nr=args.n_project_grid,
    )

    amps = rho_unit_amplitudes(args.Omega_m, args.Omega_b, args.h, z_use)
    # Parameter-free DRND wall amplitude from graph invariant V_star/N_boundary = 19/2.
    # This locks the baryonic physical-z density scale to the topological gain.
    amps[f"DRND_baryon_topological_gain_{args.topological_gain_label}"] = args.topological_gain * amps["baryon_physical_z"]

    rows = []

    # Null
    pred_null = np.zeros_like(e_mean)
    rows.append({"case": "null_zero", "A_locked": 0.0, **aic_bic(chi2_cov(e_mean, pred_null, invcov), len(axis), 0)})

    # Locked physical predictions
    predictions = {"null_zero": pred_null}
    for name, amp in amps.items():
        pred = amp * shape
        predictions[f"locked_{name}"] = pred
        rows.append({
            "case": f"locked_{name}",
            "A_locked": float(amp),
            "amplitude_source": name,
            **aic_bic(chi2_cov(e_mean, pred, invcov), len(axis), 0),
        })

    # Diagnostic best-free amplitude for same locked geometry.
    A_req, pred_req = fit_amplitude_cov(e_mean, invcov, shape)
    predictions["diagnostic_best_free_amplitude_same_geometry"] = pred_req
    rows.append({
        "case": "diagnostic_best_free_amplitude_same_geometry",
        "A_locked": float(A_req),
        "amplitude_source": "fit diagnostic only",
        **aic_bic(chi2_cov(e_mean, pred_req, invcov), len(axis), 1),
    })

    best = min(rows, key=lambda r: r["BIC"])
    for r in rows:
        r["delta_BIC_vs_best"] = r["BIC"] - best["BIC"]
        r["delta_AIC_vs_best"] = r["AIC"] - best["AIC"]
        if A_req != 0:
            r["A_over_A_required"] = r["A_locked"] / A_req
        else:
            r["A_over_A_required"] = None
    rows = sorted(rows, key=lambda r: r["BIC"])

    profile_rows = []
    for i in range(len(axis)):
        profile_rows.append({
            "bin": i,
            "axis_R_over_Rv": float(axis[i]),
            "E_mode_obs": float(e_mean[i]),
            "E_err_bootstrap": float(err[i]),
            "B_mode": float(b_mean[i]),
            "n_voids_contributing": int(counts[i]),
            "shape_norm": float(shape[i]),
        })

    pred_rows = []
    for case, pred in predictions.items():
        for i in range(len(axis)):
            pred_rows.append({
                "case": case,
                "axis_R_over_Rv": float(axis[i]),
                "E_mode_obs": float(e_mean[i]),
                "E_err_bootstrap": float(err[i]),
                "E_mode_pred": float(pred[i]),
            })

    write_csv(tabledir / "profile_data.csv", profile_rows)
    write_csv(tabledir / "model_comparison_locked_physical.csv", rows, preferred=[
        "case", "A_locked", "A_over_A_required", "amplitude_source",
        "chi2", "k_params", "dof", "chi2_per_dof", "AIC", "BIC",
        "delta_AIC_vs_best", "delta_BIC_vs_best"
    ])
    write_csv(tabledir / "predictions_locked_physical.csv", pred_rows)
    write_matrix_csv(tabledir / "covariance.csv", cov)
    write_matrix_csv(tabledir / "correlation.csv", corr)

    # Plots
    order = np.argsort(axis)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(axis, e_mean, yerr=err, fmt="o", label="SDSS E-mode")
    for case in [
        "null_zero",
        "locked_total_matter_comoving",
        "locked_total_matter_physical_z",
        "locked_baryon_physical_z",
        f"locked_DRND_baryon_topological_gain_{args.topological_gain_label}",
        "diagnostic_best_free_amplitude_same_geometry",
    ]:
        if case in predictions:
            ax.plot(axis[order], predictions[case][order], label=case)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("R/Rv")
    ax.set_ylabel("DeltaSigma/Rv source convention")
    ax.set_title("Locked physical amplitude test")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plotdir / "locked_physical_profile.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="correlation")
    ax.set_title("Bootstrap covariance correlation")
    fig.tight_layout()
    fig.savefig(plotdir / "correlation_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(rows)), [r["BIC"] for r in rows])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["case"] for r in rows], rotation=35, ha="right")
    ax.set_ylabel("BIC")
    ax.set_title("Locked physical amplitude model comparison")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(plotdir / "model_comparison_bic.png", dpi=180)
    plt.close(fig)

    by = {r["case"]: r for r in rows}
    null = by.get("null_zero")
    diag = by.get("diagnostic_best_free_amplitude_same_geometry")

    decision = {
        "best_model_by_BIC": best["case"],
        "catalog_means": catalog_stats,
        "geometry_used": geom | {"C_comp_recomputed": ccomp},
        "amplitude_scales_source_units": amps,
        "diagnostic_required_amplitude_A": A_req,
        "previous_free_amplitude_A": geom.get("A_fit_previous"),
        "topological_gain_used": args.topological_gain,
        "topological_gain_label": args.topological_gain_label,
        "A_DRND_baryon_topological_gain": amps[f"DRND_baryon_topological_gain_{args.topological_gain_label}"],
        "A_DRND_over_A_required": (amps[f"DRND_baryon_topological_gain_{args.topological_gain_label}"] / A_req) if A_req != 0 else None,
        "hartlap_factor": hartlap,
        "n_void_profiles": len(profiles),
        "nboot": args.nboot,
        "delta_BIC_requiredAmplitude_minus_null": (diag["BIC"] - null["BIC"]) if diag and null else None,
        "locked_results": rows,
        "diagnostic_flags": {
            "total_matter_physical_beats_null": bool(by.get("locked_total_matter_physical_z") and null and by["locked_total_matter_physical_z"]["BIC"] < null["BIC"]),
            "baryon_physical_beats_null": bool(by.get("locked_baryon_physical_z") and null and by["locked_baryon_physical_z"]["BIC"] < null["BIC"]),
            "DRND_topological_gain_beats_null": bool(by.get(f"locked_DRND_baryon_topological_gain_{args.topological_gain_label}") and null and by[f"locked_DRND_baryon_topological_gain_{args.topological_gain_label}"]["BIC"] < null["BIC"]),
            "DRND_topological_gain_competes_with_required_amplitude_BIC": bool(by.get(f"locked_DRND_baryon_topological_gain_{args.topological_gain_label}") and diag and by[f"locked_DRND_baryon_topological_gain_{args.topological_gain_label}"]["BIC"] <= diag["BIC"] + 2),
            "DRND_topological_gain_within_5_percent_required_A": bool(A_req != 0 and abs(amps[f"DRND_baryon_topological_gain_{args.topological_gain_label}"] / A_req - 1.0) <= 0.05),
            "required_amplitude_close_to_total_matter_physical_factor2": bool(A_req != 0 and abs(amps["total_matter_physical_z"] / A_req - 1.0) <= 1.0),
            "required_amplitude_close_to_baryon_physical_factor2": bool(A_req != 0 and abs(amps["baryon_physical_z"] / A_req - 1.0) <= 1.0),
        },
    }

    summary = {
        "model": "DRND locked physical amplitude validation",
        "data_source": {
            "repository": "pmelchior/void-lensing",
            "urls": DATA_URLS,
            "rawdir": str(rawdir),
        },
        "input": vars(args),
        "decision": decision,
        "profile_data": profile_rows,
        "model_comparison": rows,
        "cautions": [
            "This locks amplitude using the SDSS source convention DeltaSigma/Rv in 1e12 h^2 Msun Mpc^-3.",
            "The xshape observable tests physical normalization only in the source convention, not yet a full independent shear-catalog likelihood.",
            "diagnostic_best_free_amplitude_same_geometry is included only to measure the amplitude gap; it is not a DRND prediction.",
        ],
        "outputs": {
            "profile_data": "tables/profile_data.csv",
            "model_comparison": "tables/model_comparison_locked_physical.csv",
            "predictions": "tables/predictions_locked_physical.csv",
            "covariance": "tables/covariance.csv",
            "correlation": "tables/correlation.csv",
            "profile_plot": "plots/locked_physical_profile.png",
            "correlation_plot": "plots/correlation_matrix.png",
            "bic_plot": "plots/model_comparison_bic.png",
        },
    }

    (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
    (outdir / "methods_locked_physical_amplitude.md").write_text(
        "We fixed the zero-core compensated-wall geometry from the previous covariance likelihood and removed the fitted amplitude. "
        "The dimensionless projected profile was converted to the SDSS source convention using DeltaSigma/Rv = rho_bar(z) DeltaSigma_norm, "
        "with rho_bar expressed in units of 1e12 h^2 Msun Mpc^-3. Total-matter, baryon-only, and the locked DRND graph-gain amplitude A=(19/2)A_baryon,z were tested without refitting.\n",
        encoding="utf-8",
    )

    print(json.dumps(safe_json({
        "outdir": str(outdir),
        "catalog_means": catalog_stats,
        "amplitude_scales_source_units": amps,
        "diagnostic_required_amplitude_A": A_req,
        "best_model_by_BIC": best["case"],
        "files": list(summary["outputs"].values()) + ["summary.json", "methods_locked_physical_amplitude.md"],
    }), indent=2))


if __name__ == "__main__":
    main()

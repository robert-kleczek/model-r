#!/usr/bin/env python3
"""
DRND_VOID_DENSITY_PROFILE_AUDIT.py

Audit a stacked galaxy-density profile around voids from a file containing
one column of normalized distances r_norm = R / R_void.

This script is intended for the output of build_void_profile.py:
    Void_Profile_Data_100.csv

It is not a lensing test. It is a density-profile / compensation-shell audit.
Use it to constrain the shell/ridge priors for the DRND void-lensing model.

Outputs
-------
summary.json
tables/void_density_profile_binned.csv
tables/model_comparison.csv
tables/shell_prior_for_lensing.json
plots/void_density_profile_fit.png

Usage
-----
python DRND_VOID_DENSITY_PROFILE_AUDIT.py ^
  --rnorm-csv Void_Profile_Data_100.csv ^
  --outdir DRND_VOID_DENSITY_PROFILE_AUDIT

Optional:
python DRND_VOID_DENSITY_PROFILE_AUDIT.py ^
  --rnorm-csv Void_Profile_Data_100.csv ^
  --max-r-norm 3.0 ^
  --bins 30 ^
  --outer-bins 5 ^
  --outdir DRND_VOID_DENSITY_PROFILE_AUDIT
"""

from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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


def read_rnorm(path):
    # Flexible: accepts header r_norm, first column, or plain numeric text.
    vals = []
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline()
        f.seek(0)
        if "," in first or any(c.isalpha() for c in first):
            reader = csv.DictReader(f)
            names = reader.fieldnames or []
            key = "r_norm" if "r_norm" in names else names[0]
            for row in reader:
                try:
                    vals.append(float(row[key]))
                except Exception:
                    pass
        else:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    vals.append(float(s.split()[0]))
                except Exception:
                    pass
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise SystemExit("No usable r_norm values found.")
    return arr


def logistic_void(x, delta_in, x0, width):
    width = np.maximum(width, 1e-6)
    return delta_in + (0.0 - delta_in) / (1.0 + np.exp(-(x - x0) / width))


def logistic_plus_shell(x, delta_in, x0, width, shell_amp, shell_x0, shell_sigma):
    return logistic_void(x, delta_in, x0, width) + shell_amp * np.exp(-0.5 * ((x - shell_x0) / np.maximum(shell_sigma, 1e-6)) ** 2)


def chi2_aic_bic(y, yerr, ypred, k):
    res = (y - ypred) / np.maximum(yerr, 1e-12)
    chi2 = float(np.sum(res * res))
    n = int(len(y))
    return {
        "chi2": chi2,
        "n_data": n,
        "k_params": int(k),
        "dof": max(n - int(k), 1),
        "chi2_per_dof": chi2 / max(n - int(k), 1),
        "AIC": chi2 + 2 * int(k),
        "BIC": chi2 + int(k) * math.log(max(n, 2)),
    }


def crossing_x(x, y, level):
    # Linear interpolation of first crossing upward through level.
    for i in range(len(x) - 1):
        if (y[i] - level) == 0:
            return float(x[i])
        if (y[i] - level) * (y[i+1] - level) < 0:
            t = (level - y[i]) / (y[i+1] - y[i])
            return float(x[i] + t * (x[i+1] - x[i]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rnorm-csv", required=True)
    ap.add_argument("--outdir", default="DRND_VOID_DENSITY_PROFILE_AUDIT")
    ap.add_argument("--max-r-norm", type=float, default=3.0)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--outer-bins", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tabledir = outdir / "tables"
    plotdir = outdir / "plots"
    tabledir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    vals_all = read_rnorm(args.rnorm_csv)
    vals = vals_all[(vals_all >= 0) & (vals_all <= args.max_r_norm)]

    edges = np.linspace(0, args.max_r_norm, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(vals, bins=edges)
    volumes = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    density = counts / volumes

    outer = density[-args.outer_bins:]
    rho_ref = float(np.mean(outer[outer > 0])) if np.any(outer > 0) else float(np.mean(density[density > 0]))
    contrast = (density - rho_ref) / rho_ref

    # Poisson error propagation for density contrast.
    counts_err = np.sqrt(np.maximum(counts, 1.0))
    density_err = counts_err / volumes
    contrast_err = density_err / max(rho_ref, 1e-30)

    good = (counts >= args.min_count) & np.isfinite(contrast) & np.isfinite(contrast_err) & (contrast_err > 0)
    x = centers[good]
    y = contrast[good]
    e = contrast_err[good]

    rows = []
    for i in range(len(centers)):
        rows.append({
            "bin": i,
            "r_norm_center": centers[i],
            "r_norm_lo": edges[i],
            "r_norm_hi": edges[i+1],
            "count": int(counts[i]),
            "shell_volume_norm": volumes[i],
            "density_arbitrary": density[i],
            "density_reference_outer": rho_ref,
            "contrast": contrast[i],
            "contrast_err_poisson": contrast_err[i],
            "used_in_fit": bool(good[i]),
        })
    write_csv(tabledir / "void_density_profile_binned.csv", rows, preferred=[
        "bin", "r_norm_center", "r_norm_lo", "r_norm_hi", "count",
        "density_arbitrary", "contrast", "contrast_err_poisson", "used_in_fit"
    ])

    comparison = []

    # Null model: contrast = 0.
    pred_null = np.zeros_like(y)
    r = chi2_aic_bic(y, e, pred_null, 0)
    r["case"] = "null_zero_contrast"
    comparison.append(r)

    # Constant inner deficit, no compensation.
    const_val = np.average(y, weights=1 / e**2)
    pred_const = np.full_like(y, const_val)
    r = chi2_aic_bic(y, e, pred_const, 1)
    r["case"] = "constant_contrast"
    r["delta_const"] = float(const_val)
    comparison.append(r)

    # Logistic void edge.
    try:
        p0 = [-0.95, 1.6, 0.2]
        bounds = ([-1.5, 0.2, 0.02], [0.2, 3.0, 1.0])
        popt, pcov = curve_fit(logistic_void, x, y, sigma=e, p0=p0, bounds=bounds, maxfev=20000)
        pred = logistic_void(x, *popt)
        r = chi2_aic_bic(y, e, pred, 3)
        r.update({"case": "logistic_void_edge", "delta_in": popt[0], "edge_x0": popt[1], "edge_width": popt[2]})
        comparison.append(r)
        logistic_params = popt
    except Exception as exc:
        logistic_params = None
        comparison.append({"case": "logistic_void_edge", "error": str(exc)})

    # Logistic + shell.
    try:
        p0 = [-0.95, 1.55, 0.2, 0.08, 2.7, 0.18]
        bounds = ([-1.5, 0.2, 0.02, -0.5, 0.8, 0.03], [0.2, 3.0, 1.0, 0.8, 3.2, 0.8])
        popt2, pcov2 = curve_fit(logistic_plus_shell, x, y, sigma=e, p0=p0, bounds=bounds, maxfev=50000)
        pred2 = logistic_plus_shell(x, *popt2)
        r = chi2_aic_bic(y, e, pred2, 6)
        r.update({
            "case": "logistic_plus_compensation_shell",
            "delta_in": popt2[0],
            "edge_x0": popt2[1],
            "edge_width": popt2[2],
            "shell_amp": popt2[3],
            "shell_x0": popt2[4],
            "shell_sigma": popt2[5],
        })
        comparison.append(r)
        shell_params = popt2
    except Exception as exc:
        shell_params = None
        comparison.append({"case": "logistic_plus_compensation_shell", "error": str(exc)})

    # Sort comparison with robust key.
    comparison_sorted = sorted(comparison, key=lambda r: r.get("BIC", float("inf")))
    best = comparison_sorted[0] if comparison_sorted else {}

    # Derived shell/edge priors.
    x_half = crossing_x(centers, contrast, -0.5)
    x_zero = crossing_x(centers, contrast, 0.0)
    shell_peak_idx = int(np.argmax(contrast))
    shell_peak_x = float(centers[shell_peak_idx])
    shell_peak_contrast = float(contrast[shell_peak_idx])

    shell_prior = {
        "source": str(args.rnorm_csv),
        "r_norm_profile": True,
        "edge_crossing_contrast_minus_0p5": x_half,
        "zero_crossing": x_zero,
        "empirical_shell_peak_x": shell_peak_x,
        "empirical_shell_peak_contrast": shell_peak_contrast,
        "recommended_shell_x0_range": [max(0.8, (x_zero or 2.0) - 0.4), min(3.2, (x_zero or 2.0) + 0.4)],
        "recommended_shell_sigma_range": [0.08, 0.6],
        "recommended_void_lensing_interpretation": "Use this only as a prior for compensation ridge location/width, not as lensing evidence.",
    }
    if shell_params is not None:
        shell_prior.update({
            "fit_shell_x0": float(shell_params[4]),
            "fit_shell_sigma": float(shell_params[5]),
            "fit_shell_amp": float(shell_params[3]),
        })
    if logistic_params is not None:
        shell_prior.update({
            "fit_edge_x0": float(logistic_params[1]),
            "fit_edge_width": float(logistic_params[2]),
            "fit_delta_in": float(logistic_params[0]),
        })

    write_csv(tabledir / "model_comparison.csv", comparison_sorted, preferred=[
        "case", "chi2", "k_params", "dof", "chi2_per_dof", "AIC", "BIC",
        "delta_in", "edge_x0", "edge_width", "shell_amp", "shell_x0", "shell_sigma"
    ])
    (tabledir / "shell_prior_for_lensing.json").write_text(json.dumps(safe_json(shell_prior), indent=2), encoding="utf-8")

    # Plot.
    xx = np.linspace(0.01, args.max_r_norm, 500)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(centers, contrast, yerr=contrast_err, fmt="o", ms=4, alpha=0.8, label="binned density contrast")
    ax.axvline(1.0, color="black", linestyle="--", label="R = R_void")
    ax.axhline(0.0, color="gray", linewidth=1)

    if logistic_params is not None:
        ax.plot(xx, logistic_void(xx, *logistic_params), label="logistic edge fit")
    if shell_params is not None:
        ax.plot(xx, logistic_plus_shell(xx, *shell_params), label="edge + shell fit")

    ax.set_xlabel("R / R_void")
    ax.set_ylabel("Density contrast")
    ax.set_title("Void density profile audit")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotdir / "void_density_profile_fit.png", dpi=180)
    plt.close(fig)

    summary = {
        "model": "void density profile audit",
        "input": vars(args) | {
            "n_raw_distances": int(vals_all.size),
            "n_used_distances_le_max": int(vals.size),
            "fraction_used": float(vals.size / max(vals_all.size, 1)),
            "rho_reference_outer": rho_ref,
        },
        "best_model_by_BIC": best.get("case"),
        "model_comparison": comparison_sorted,
        "shell_prior_for_lensing": shell_prior,
        "cautions": [
            "This is a galaxy number-density profile, not a lensing profile.",
            "The profile depends on SDSS spectroscopic selection, redshift window, angular footprint and edge effects.",
            "Use the derived shell location only as a void compensation prior, not as direct evidence for ATP lensing.",
        ],
        "outputs": {
            "binned_profile": "tables/void_density_profile_binned.csv",
            "model_comparison": "tables/model_comparison.csv",
            "shell_prior": "tables/shell_prior_for_lensing.json",
            "plot": "plots/void_density_profile_fit.png",
        },
    }
    (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")

    print(json.dumps(safe_json({
        "outdir": str(outdir),
        "best_model_by_BIC": best.get("case"),
        "shell_prior_for_lensing": shell_prior,
        "files": [
            "summary.json",
            "tables/void_density_profile_binned.csv",
            "tables/model_comparison.csv",
            "tables/shell_prior_for_lensing.json",
            "plots/void_density_profile_fit.png",
        ],
    }), indent=2))


if __name__ == "__main__":
    main()

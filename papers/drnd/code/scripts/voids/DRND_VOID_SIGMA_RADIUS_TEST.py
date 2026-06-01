#!/usr/bin/env python3
"""
DRND_VOID_SIGMA_RADIUS_TEST.py

Purpose
-------
Test whether the void-wall / compensation-shell width is independent of
void radius.

This directly addresses the claim:

    universal a0 scenario:
        shell/edge width sigma is independent of R_void

    geometric-size scenario:
        shell/edge width sigma grows with R_void

Important
---------
The existing file Void_Profile_Data_100.csv contains only stacked r_norm values.
It does NOT preserve void_id or R_void per point, so it cannot directly test
sigma(R_void).

This script therefore either:
  A) reads a previously saved per-void distance file with columns:
        void_id,R_void_Mpc,r_norm
     OR
  B) rebuilds that file by querying SDSS from Void_Catalog_3D.csv.

Recommended workflow
--------------------
1. First run with SDSS query and save per-void distances:
   python DRND_VOID_SIGMA_RADIUS_TEST.py ^
     --catalog Void_Catalog_3D.csv ^
     --n-voids 100 ^
     --outdir DRND_VOID_SIGMA_RADIUS_TEST

2. Later rerun without querying:
   python DRND_VOID_SIGMA_RADIUS_TEST.py ^
     --pervoid-csv DRND_VOID_SIGMA_RADIUS_TEST/tables/per_void_rnorm_distances.csv ^
     --outdir DRND_VOID_SIGMA_RADIUS_TEST_REUSE

Outputs
-------
summary.json
tables/per_void_rnorm_distances.csv
tables/radius_group_profiles.csv
tables/radius_group_fit_results.csv
tables/sigma_radius_correlation.csv
plots/sigma_vs_Rvoid.png
plots/radius_group_profiles.png

Interpretation
--------------
This is a diagnostic galaxy-density test, not a lensing test.
It can justify priors for void compensation shells, but it is not direct
evidence for ATP lensing.
"""

from __future__ import annotations

import argparse, csv, json, math, random, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr

try:
    from astropy.cosmology import FlatLambdaCDM
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except Exception:
    FlatLambdaCDM = None
    SkyCoord = None
    u = None

try:
    from astroquery.sdss import SDSS
except Exception:
    SDSS = None


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


def logistic_void(x, delta_in, x0, width):
    width = np.maximum(width, 1e-6)
    return delta_in + (0.0 - delta_in) / (1.0 + np.exp(-(x - x0) / width))


def logistic_plus_shell(x, delta_in, x0, width, shell_amp, shell_x0, shell_sigma):
    return logistic_void(x, delta_in, x0, width) + shell_amp * np.exp(-0.5 * ((x - shell_x0) / np.maximum(shell_sigma, 1e-6)) ** 2)


def build_profile(rnorm_values, max_r_norm=3.0, bins=30, outer_bins=5, min_count=1):
    vals = np.asarray(rnorm_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= 0) & (vals <= max_r_norm)]
    edges = np.linspace(0, max_r_norm, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(vals, bins=edges)
    volumes = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    density = counts / volumes
    outer = density[-outer_bins:]
    rho_ref = float(np.mean(outer[outer > 0])) if np.any(outer > 0) else float(np.mean(density[density > 0]))
    contrast = (density - rho_ref) / rho_ref
    counts_err = np.sqrt(np.maximum(counts, 1.0))
    density_err = counts_err / volumes
    contrast_err = density_err / max(rho_ref, 1e-30)
    good = (counts >= min_count) & np.isfinite(contrast) & np.isfinite(contrast_err) & (contrast_err > 0)
    return {
        "edges": edges,
        "centers": centers,
        "counts": counts,
        "volumes": volumes,
        "density": density,
        "contrast": contrast,
        "contrast_err": contrast_err,
        "good": good,
        "rho_ref": rho_ref,
        "n_used": int(vals.size),
    }


def fit_profile(centers, contrast, contrast_err, good):
    x = centers[good]
    y = contrast[good]
    e = contrast_err[good]
    if len(x) < 8:
        return None

    results = {}

    try:
        p0 = [-0.95, 1.6, 0.2]
        bounds = ([-1.5, 0.2, 0.02], [0.2, 3.0, 1.0])
        popt, pcov = curve_fit(logistic_void, x, y, sigma=e, p0=p0, bounds=bounds, maxfev=30000)
        pred = logistic_void(x, *popt)
        chi2 = float(np.sum(((y - pred) / e) ** 2))
        results["edge"] = {
            "delta_in": float(popt[0]),
            "edge_x0": float(popt[1]),
            "edge_width": float(popt[2]),
            "chi2": chi2,
            "dof": max(len(x) - 3, 1),
            "chi2_per_dof": chi2 / max(len(x) - 3, 1),
        }
    except Exception as exc:
        results["edge_error"] = str(exc)

    try:
        p0 = [-0.95, 1.6, 0.2, 0.04, 2.7, 0.15]
        bounds = ([-1.5, 0.2, 0.02, -0.5, 0.8, 0.03], [0.2, 3.0, 1.0, 0.8, 3.2, 0.8])
        popt, pcov = curve_fit(logistic_plus_shell, x, y, sigma=e, p0=p0, bounds=bounds, maxfev=50000)
        pred = logistic_plus_shell(x, *popt)
        chi2 = float(np.sum(((y - pred) / e) ** 2))
        results["shell"] = {
            "delta_in": float(popt[0]),
            "edge_x0": float(popt[1]),
            "edge_width": float(popt[2]),
            "shell_amp": float(popt[3]),
            "shell_x0": float(popt[4]),
            "shell_sigma": float(popt[5]),
            "chi2": chi2,
            "dof": max(len(x) - 6, 1),
            "chi2_per_dof": chi2 / max(len(x) - 6, 1),
        }
    except Exception as exc:
        results["shell_error"] = str(exc)

    return results


def query_sdss_for_catalog(catalog_path, out_csv, n_voids, max_r_norm, H0, Om0, z_window, sleep_base, sleep_jitter):
    if FlatLambdaCDM is None or SkyCoord is None or SDSS is None:
        raise SystemExit("Missing dependencies. Install: pip install astropy astroquery scipy pandas matplotlib")

    df = pd.read_csv(catalog_path)
    df = df.head(min(n_voids, len(df))).copy()
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        ra = float(row["RA"])
        dec = float(row["DEC"])
        z_void = float(row["Z_VOID"])
        r_void = float(row["RADIUS_MPC"])

        max_phys = max_r_norm * r_void
        mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z_void).value / 1000.0
        deg_rad = max_phys / (mpc_per_arcmin * 60.0)

        query = f"""
            SELECT p.ra, p.dec, s.z
            FROM PhotoObj AS p
            JOIN SpecObj AS s ON s.bestobjid = p.objid
            WHERE s.class = 'GALAXY'
            AND s.z BETWEEN {z_void-z_window} AND {z_void+z_window}
            AND dbo.fDistanceArcMinEq({ra}, {dec}, p.ra, p.dec) < {deg_rad * 60.0}
        """

        try:
            res = SDSS.query_sql(query)
            if res is not None and "z" in res.colnames and len(res) > 0:
                c_void = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                c_gal = SkyCoord(ra=np.asarray(res["ra"], dtype=float) * u.deg,
                                 dec=np.asarray(res["dec"], dtype=float) * u.deg)
                theta = c_void.separation(c_gal).radian

                d_void = cosmo.comoving_distance(z_void).value
                d_gal = cosmo.comoving_distance(np.asarray(res["z"], dtype=float)).value
                r_phys = np.sqrt((d_gal - d_void) ** 2 + (d_void * theta) ** 2)
                r_norm = r_phys / r_void

                for j, rn in enumerate(r_norm):
                    if np.isfinite(rn) and rn >= 0 and rn <= max_r_norm:
                        rows.append({
                            "void_id": i,
                            "RA_void": ra,
                            "DEC_void": dec,
                            "z_void": z_void,
                            "R_void_Mpc": r_void,
                            "gal_z": float(res["z"][j]),
                            "r_norm": float(rn),
                        })

            time.sleep(sleep_base + random.uniform(0.0, sleep_jitter))
            if (i + 1) % 10 == 0:
                print(f"[query] {i+1}/{len(df)} voids, saved distances: {len(rows)}", flush=True)

        except Exception as exc:
            print(f"[warn] void {i} failed: {exc}", flush=True)
            continue

    write_csv(out_csv, rows, preferred=["void_id", "R_void_Mpc", "r_norm", "z_void", "gal_z", "RA_void", "DEC_void"])
    return pd.DataFrame(rows)


def read_pervoid_csv(path):
    df = pd.read_csv(path)
    required = {"void_id", "R_void_Mpc", "r_norm"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Per-void CSV missing columns: {sorted(missing)}")
    df = df[np.isfinite(df["r_norm"]) & np.isfinite(df["R_void_Mpc"])]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=None, help="Void_Catalog_3D.csv; used only if --pervoid-csv is not supplied.")
    ap.add_argument("--pervoid-csv", default=None, help="Existing per-void distance file with void_id,R_void_Mpc,r_norm.")
    ap.add_argument("--outdir", default="DRND_VOID_SIGMA_RADIUS_TEST")
    ap.add_argument("--n-voids", type=int, default=100)
    ap.add_argument("--max-r-norm", type=float, default=3.0)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--outer-bins", type=int, default=5)
    ap.add_argument("--radius-groups", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--H0", type=float, default=67.4)
    ap.add_argument("--Om0", type=float, default=0.315)
    ap.add_argument("--z-window", type=float, default=0.05)
    ap.add_argument("--sleep-base", type=float, default=2.0)
    ap.add_argument("--sleep-jitter", type=float, default=1.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tabledir = outdir / "tables"
    plotdir = outdir / "plots"
    tabledir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    pervoid_out = tabledir / "per_void_rnorm_distances.csv"

    if args.pervoid_csv:
        df = read_pervoid_csv(args.pervoid_csv)
        # copy normalized data to output for provenance
        df.to_csv(pervoid_out, index=False)
    else:
        if not args.catalog:
            raise SystemExit("Provide either --pervoid-csv or --catalog")
        df = query_sdss_for_catalog(
            args.catalog, pervoid_out, args.n_voids, args.max_r_norm,
            args.H0, args.Om0, args.z_window, args.sleep_base, args.sleep_jitter
        )

    if df.empty:
        raise SystemExit("No per-void distances available.")

    # Build radius groups by void radius quantiles.
    void_table = df[["void_id", "R_void_Mpc"]].drop_duplicates().copy()
    void_table = void_table.sort_values("R_void_Mpc")
    void_table["radius_group"] = pd.qcut(void_table["R_void_Mpc"], q=min(args.radius_groups, len(void_table)), labels=False, duplicates="drop")
    df = df.merge(void_table[["void_id", "radius_group"]], on="void_id", how="left")

    # Overall and grouped fits.
    fit_rows = []
    profile_rows = []

    groups = [("all", df)]
    for g in sorted(df["radius_group"].dropna().unique()):
        groups.append((f"Rbin_{int(g)}", df[df["radius_group"] == g]))

    for label, sub in groups:
        prof = build_profile(sub["r_norm"].values, args.max_r_norm, args.bins, args.outer_bins, args.min_count)
        fits = fit_profile(prof["centers"], prof["contrast"], prof["contrast_err"], prof["good"])

        Rv_med = float(np.median(sub[["void_id", "R_void_Mpc"]].drop_duplicates()["R_void_Mpc"]))
        Rv_min = float(np.min(sub[["void_id", "R_void_Mpc"]].drop_duplicates()["R_void_Mpc"]))
        Rv_max = float(np.max(sub[["void_id", "R_void_Mpc"]].drop_duplicates()["R_void_Mpc"]))
        n_voids = int(sub["void_id"].nunique())

        for i, x in enumerate(prof["centers"]):
            profile_rows.append({
                "group": label,
                "R_void_Mpc_median": Rv_med,
                "R_void_Mpc_min": Rv_min,
                "R_void_Mpc_max": Rv_max,
                "n_voids": n_voids,
                "r_norm_center": float(x),
                "count": int(prof["counts"][i]),
                "density_arbitrary": float(prof["density"][i]),
                "contrast": float(prof["contrast"][i]),
                "contrast_err": float(prof["contrast_err"][i]),
                "used_in_fit": bool(prof["good"][i]),
            })

        base = {
            "group": label,
            "R_void_Mpc_median": Rv_med,
            "R_void_Mpc_min": Rv_min,
            "R_void_Mpc_max": Rv_max,
            "n_voids": n_voids,
            "n_distances": int(len(sub)),
        }

        if fits is not None:
            if "edge" in fits:
                fit_rows.append({**base, "fit_case": "logistic_edge", **fits["edge"]})
            if "shell" in fits:
                fit_rows.append({**base, "fit_case": "logistic_plus_shell", **fits["shell"]})
            if "edge_error" in fits:
                fit_rows.append({**base, "fit_case": "logistic_edge", "error": fits["edge_error"]})
            if "shell_error" in fits:
                fit_rows.append({**base, "fit_case": "logistic_plus_shell", "error": fits["shell_error"]})
        else:
            fit_rows.append({**base, "fit_case": "none", "error": "too few points"})

    write_csv(tabledir / "radius_group_profiles.csv", profile_rows, preferred=[
        "group", "R_void_Mpc_median", "R_void_Mpc_min", "R_void_Mpc_max", "n_voids",
        "r_norm_center", "count", "contrast", "contrast_err", "used_in_fit"
    ])
    write_csv(tabledir / "radius_group_fit_results.csv", fit_rows, preferred=[
        "group", "fit_case", "R_void_Mpc_median", "R_void_Mpc_min", "R_void_Mpc_max",
        "n_voids", "n_distances", "delta_in", "edge_x0", "edge_width",
        "shell_amp", "shell_x0", "shell_sigma", "chi2", "dof", "chi2_per_dof", "error"
    ])

    # Correlation on group-level fitted sigmas.
    corr_rows = []
    for fit_case, sigma_col in [("logistic_edge", "edge_width"), ("logistic_plus_shell", "shell_sigma")]:
        rows = [r for r in fit_rows if r.get("fit_case") == fit_case and r.get("group") != "all" and r.get(sigma_col) is not None]
        xs = np.array([r["R_void_Mpc_median"] for r in rows], float)
        ys = np.array([r[sigma_col] for r in rows], float)
        if len(xs) >= 3:
            pr = pearsonr(xs, ys)
            sr = spearmanr(xs, ys)
            corr_rows.append({
                "fit_case": fit_case,
                "sigma_quantity": sigma_col,
                "n_groups": int(len(xs)),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "spearman_rho": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
            })
        else:
            corr_rows.append({
                "fit_case": fit_case,
                "sigma_quantity": sigma_col,
                "n_groups": int(len(xs)),
                "error": "Need at least 3 radius groups for correlation.",
            })
    write_csv(tabledir / "sigma_radius_correlation.csv", corr_rows, preferred=[
        "fit_case", "sigma_quantity", "n_groups", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "error"
    ])

    # Plots.
    fig, ax = plt.subplots(figsize=(8, 5))
    shell_rows = [r for r in fit_rows if r.get("fit_case") == "logistic_plus_shell" and r.get("group") != "all" and r.get("shell_sigma") is not None]
    if shell_rows:
        xs = np.array([r["R_void_Mpc_median"] for r in shell_rows], float)
        ys = np.array([r["shell_sigma"] for r in shell_rows], float)
        ax.plot(xs, ys, "o-", label="shell sigma")
    edge_rows = [r for r in fit_rows if r.get("fit_case") == "logistic_edge" and r.get("group") != "all" and r.get("edge_width") is not None]
    if edge_rows:
        xs2 = np.array([r["R_void_Mpc_median"] for r in edge_rows], float)
        ys2 = np.array([r["edge_width"] for r in edge_rows], float)
        ax.plot(xs2, ys2, "s-", label="edge width")
    ax.set_xlabel("Median R_void [Mpc]")
    ax.set_ylabel("Fitted width in R/R_void units")
    ax.set_title("Void wall width vs void radius")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotdir / "sigma_vs_Rvoid.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, sub in pd.DataFrame(profile_rows).groupby("group"):
        if label == "all":
            continue
        ax.plot(sub["r_norm_center"], sub["contrast"], marker="o", linewidth=1, label=label)
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(1, color="black", linestyle="--")
    ax.set_xlabel("R / R_void")
    ax.set_ylabel("Density contrast")
    ax.set_title("Void density profiles split by R_void")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plotdir / "radius_group_profiles.png", dpi=180)
    plt.close(fig)

    # Decision logic.
    corr_shell = next((r for r in corr_rows if r.get("fit_case") == "logistic_plus_shell"), {})
    rho = corr_shell.get("spearman_rho")
    pval = corr_shell.get("spearman_p")

    if isinstance(rho, (int, float)) and isinstance(pval, (int, float)):
        if abs(rho) < 0.3 or pval > 0.1:
            interpretation = "No statistically resolved monotonic correlation in this grouped diagnostic."
        elif rho > 0:
            interpretation = "Positive width-radius trend indicated in this grouped diagnostic."
        else:
            interpretation = "Negative width-radius trend indicated in this grouped diagnostic."
    else:
        interpretation = "Correlation inconclusive."

    summary = {
        "model": "void wall sigma vs R_void diagnostic",
        "input": vars(args) | {
            "n_voids": int(df["void_id"].nunique()),
            "n_distances": int(len(df)),
            "R_void_Mpc_min": float(df[["void_id", "R_void_Mpc"]].drop_duplicates()["R_void_Mpc"].min()),
            "R_void_Mpc_max": float(df[["void_id", "R_void_Mpc"]].drop_duplicates()["R_void_Mpc"].max()),
        },
        "correlation": corr_rows,
        "interpretation": interpretation,
        "publication_status": {
            "can_claim_definitive_proof": False,
            "safe_claim": "This diagnostic tests whether shell/edge width is stable across void-radius bins. A non-detection of correlation is supportive of a universal-width hypothesis only if errors and selection effects are controlled.",
        },
        "cautions": [
            "This is a galaxy number-density diagnostic, not direct void lensing.",
            "A stacked r_norm-only file cannot test sigma vs R_void; per-void distances are required.",
            "SDSS spectroscopic selection, footprint/mask, redshift window, and edge effects can bias the fitted widths.",
            "With only a few radius groups, correlation p-values are low-power diagnostics.",
        ],
        "outputs": {
            "per_void_distances": "tables/per_void_rnorm_distances.csv",
            "radius_group_profiles": "tables/radius_group_profiles.csv",
            "radius_group_fit_results": "tables/radius_group_fit_results.csv",
            "sigma_radius_correlation": "tables/sigma_radius_correlation.csv",
            "sigma_vs_Rvoid_plot": "plots/sigma_vs_Rvoid.png",
            "radius_group_profiles_plot": "plots/radius_group_profiles.png",
        },
    }
    (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")

    print(json.dumps(safe_json({
        "outdir": str(outdir),
        "interpretation": interpretation,
        "correlation": corr_rows,
        "files": list(summary["outputs"].values()) + ["summary.json"],
    }), indent=2))


if __name__ == "__main__":
    main()

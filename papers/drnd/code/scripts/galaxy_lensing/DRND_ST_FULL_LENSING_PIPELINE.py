#!/usr/bin/env python3
"""
DRND_ST_FULL_LENSING_PIPELINE.py

Fuller DRND-ST galaxy-galaxy lensing test with data downloader.

What this does
--------------
1. Downloads public galaxy-galaxy lensing measurement files from:
   https://github.com/aamon/Galaxy-galaxy-lensing-KiDS-DES-HSC

2. Recursively scans the downloaded repository for usable numeric lensing
   tables: CSV/TXT/DAT and NumPy .npy/.npz.

3. Attempts to identify columns corresponding to:
      R / theta / projected radius
      DeltaSigma / gamma_t / signal
      error / uncertainty
      lens bin / survey
   The repository format may vary, so this script is intentionally defensive.

4. Runs a DRND-ST lensing comparison if the data can be put into physical
   DeltaSigma units and baryonic lens-bin metadata are supplied.

5. If baryonic metadata are missing, it writes a template:
      lens_bins_template.csv
   You must fill Mbar_Msun and Re_kpc for each lens_bin to run the physical
   DRND-ST test.

Why metadata are required
-------------------------
A full DRND-ST lensing prediction needs a baryonic profile for each lens bin:

    Mbar_Msun, Re_kpc, profile_type

Then the model computes:

    M_b(<r)
    g_b(r) = G M_b(<r) / r^2
    g_ST(r) = g_b(r) * nu(g_b/a0)
    M_eff(<r) = g_ST(r) r^2 / G
    rho_eff(r) = (1 / 4pi r^2) dM_eff/dr
    Sigma(R) = 2 int_R^inf rho_eff(r) r dr / sqrt(r^2-R^2)
    DeltaSigma(R) = Sigma_bar(<R) - Sigma(R)

This replaces the earlier point-mass proxy.

Usage
-----
Download only:

    python DRND_ST_FULL_LENSING_PIPELINE.py ^
      --download-only ^
      --outdir DRND_ST_FULL_LENSING

Scan downloaded data:

    python DRND_ST_FULL_LENSING_PIPELINE.py ^
      --data-root DRND_ST_FULL_LENSING/data/Galaxy-galaxy-lensing-KiDS-DES-HSC-main ^
      --scan-only ^
      --outdir DRND_ST_FULL_LENSING

Run with a prepared physical lensing CSV:

    python DRND_ST_FULL_LENSING_PIPELINE.py ^
      --physical-csv my_lensing_physical.csv ^
      --lens-bins-csv lens_bins.csv ^
      --outdir DRND_ST_FULL_LENSING_RUN

Physical CSV columns
--------------------
Required:
    lens_bin,R_kpc,DeltaSigma,DeltaSigma_err

Lens-bin metadata CSV columns
-----------------------------
Required:
    lens_bin,Mbar_Msun,Re_kpc,profile_type

profile_type supported:
    point
    hernquist
    exponential_sphere

Outputs
-------
    summary.json
    discovered_files.csv
    parsed_data_vectors.csv
    lens_bins_template.csv
    fit_table.csv
    prediction_table.csv
    plots/lensing_fit.png
    plots/lensing_residuals.png

Cautions
--------
- The downloader gives observational data. It does not guarantee that those
  files include baryonic masses for the lens samples.
- Without Mbar/Re per lens bin, a physical DRND-ST lensing prediction is not
  defined. The script will stop after producing a metadata template.
- This is a spherical projected model. It is much better than M(<R)/(pi R^2),
  but still not a full survey likelihood with covariance, selection functions,
  baryonic stellar-population priors, miscentering, satellite terms, or boost
  factors.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar


G_SI = 6.67430e-11
MSUN_KG = 1.98847e30
PC_M = 3.0856775814913673e16
KPC_M = 1e3 * PC_M
MPC_M = 1e6 * PC_M
C_M_S = 299792458.0

DEFAULT_REPO_ZIP = "https://github.com/aamon/Galaxy-galaxy-lensing-KiDS-DES-HSC/archive/refs/heads/main.zip"


def norm_col(s: str) -> str:
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


def safe_json(obj):
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def write_csv(path: Path, rows: List[Dict], preferred: Optional[List[str]] = None):
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


def download_file(url: str, out_path: Path, timeout: int = 120):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 DRND-ST lensing downloader"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    out_path.write_bytes(data)
    return out_path


def download_repo(outdir: Path, repo_zip_url: str = DEFAULT_REPO_ZIP):
    data_dir = outdir / "data"
    zip_path = data_dir / "Galaxy-galaxy-lensing-KiDS-DES-HSC-main.zip"
    extract_dir = data_dir
    if not zip_path.exists():
        print(f"[download] {repo_zip_url}")
        download_file(repo_zip_url, zip_path)
    else:
        print(f"[download] using existing {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Return root folder.
    candidates = list(extract_dir.glob("Galaxy-galaxy-lensing-KiDS-DES-HSC-*"))
    if not candidates:
        raise SystemExit("Downloaded ZIP extracted, but repository folder was not found.")
    return candidates[0]


def list_data_files(root: Path):
    exts = {".csv", ".txt", ".dat", ".tsv", ".npy", ".npz"}
    rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            rows.append({
                "path": str(p),
                "relpath": str(p.relative_to(root)),
                "suffix": p.suffix.lower(),
                "size_bytes": p.stat().st_size,
            })
    return rows


def sniff_delimiter(text: str):
    sample = text[:4096]
    for delim in [",", "\t", " ", ";"]:
        if delim == " ":
            # whitespace fallback
            continue
        if sample.count(delim) > 5:
            return delim
    return None


def try_read_text_table(path: Path):
    try:
        raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return []

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return []

    delim = sniff_delimiter("\n".join(lines[:20]))

    # Case 1: delimited with header
    if delim:
        try:
            reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delim)
            rows = list(reader)
            if rows and len(rows[0]) >= 2:
                return rows
        except Exception:
            pass

    # Case 2: whitespace table, maybe header
    split_lines = [re.split(r"\s+", ln) for ln in lines]
    if len(split_lines) < 2:
        return []

    # Determine whether first line is header.
    first = split_lines[0]
    first_numeric = sum(np.isfinite(to_float(x)) for x in first)
    has_header = first_numeric < max(1, len(first)//2)

    if has_header:
        header = [norm_col(x) or f"col{i}" for i, x in enumerate(first)]
        data_lines = split_lines[1:]
    else:
        ncol = max(len(x) for x in split_lines)
        header = [f"col{i}" for i in range(ncol)]
        data_lines = split_lines

    rows = []
    for parts in data_lines:
        if len(parts) < 2:
            continue
        row = {}
        for i, h in enumerate(header):
            row[h] = parts[i] if i < len(parts) else ""
        rows.append(row)
    return rows


def try_read_numpy_file(path: Path):
    rows = []
    try:
        obj = np.load(path, allow_pickle=True)
    except Exception:
        return []

    def array_to_rows(arr, prefix="arr"):
        out = []
        arr = np.asarray(arr)
        if arr.dtype.names:
            for rec in arr:
                out.append({name: rec[name].item() if hasattr(rec[name], "item") else rec[name] for name in arr.dtype.names})
            return out
        if arr.ndim == 1:
            for i, v in enumerate(arr):
                out.append({"index": i, f"{prefix}_value": float(v) if np.isscalar(v) else str(v)})
        elif arr.ndim == 2:
            for i in range(arr.shape[0]):
                out.append({f"col{j}": arr[i, j].item() if hasattr(arr[i,j], "item") else arr[i,j] for j in range(arr.shape[1])})
        return out

    if isinstance(obj, np.lib.npyio.NpzFile):
        for key in obj.files:
            arr_rows = array_to_rows(obj[key], prefix=key)
            for r in arr_rows:
                r["_npz_key"] = key
            rows.extend(arr_rows)
    else:
        rows = array_to_rows(obj)
    return rows


def guess_column(columns, candidates):
    nc = {norm_col(c): c for c in columns}
    for cand in candidates:
        cn = norm_col(cand)
        if cn in nc:
            return nc[cn]
    # fuzzy contains
    for c in columns:
        cn = norm_col(c)
        for cand in candidates:
            if norm_col(cand) in cn:
                return c
    return None


def parse_discovered_vectors(files: List[Dict]):
    parsed = []
    for fr in files:
        p = Path(fr["path"])
        if p.suffix.lower() in {".npy", ".npz"}:
            rows = try_read_numpy_file(p)
        else:
            rows = try_read_text_table(p)

        if not rows:
            continue

        cols = list(rows[0].keys())

        rcol = guess_column(cols, ["R", "rp", "r_p", "theta", "radius", "R_kpc", "R_mpc", "col0"])
        dscol = guess_column(cols, ["DeltaSigma", "delta_sigma", "ds", "delsig", "gamma_t", "gammat", "signal", "col1"])
        errcol = guess_column(cols, ["DeltaSigma_err", "ds_err", "err", "error", "sigma", "uncertainty", "col2"])
        bincol = guess_column(cols, ["lens_bin", "bin", "sample", "survey", "col3"])

        numeric_count = 0
        for row in rows[:50]:
            vals = [to_float(v) for v in row.values()]
            numeric_count += sum(np.isfinite(vals))

        parsed.append({
            "path": str(p),
            "relpath": fr.get("relpath", str(p.name)),
            "n_rows": len(rows),
            "columns": ",".join(cols),
            "r_col_guess": rcol,
            "signal_col_guess": dscol,
            "err_col_guess": errcol,
            "bin_col_guess": bincol,
            "numeric_score_first50": numeric_count,
            "usable_basic": bool(rcol and dscol),
        })
    return parsed


def make_unified_vector(files: List[Dict], max_files: int = 50):
    """
    Build a broad parsed vector table from guessed files.

    This does not assert physical units. It is meant to reveal the data layout.
    """
    vectors = []
    parsed = parse_discovered_vectors(files)
    usable = [p for p in parsed if p["usable_basic"]]
    for pr in usable[:max_files]:
        p = Path(pr["path"])
        rows = try_read_numpy_file(p) if p.suffix.lower() in {".npy", ".npz"} else try_read_text_table(p)
        if not rows:
            continue
        rcol, scol, ecol, bcol = pr["r_col_guess"], pr["signal_col_guess"], pr["err_col_guess"], pr["bin_col_guess"]
        for i, row in enumerate(rows):
            R = to_float(row.get(rcol))
            sig = to_float(row.get(scol))
            err = to_float(row.get(ecol)) if ecol else np.nan
            if not np.isfinite(R) or not np.isfinite(sig):
                continue
            lens_bin = str(row.get(bcol, "")) if bcol else ""
            if not lens_bin:
                lens_bin = f"{Path(pr['relpath']).stem}"
            vectors.append({
                "source_file": pr["relpath"],
                "row_index": i,
                "lens_bin": lens_bin,
                "R_raw": R,
                "signal_raw": sig,
                "error_raw": err,
                "R_col": rcol,
                "signal_col": scol,
                "err_col": ecol,
            })
    return vectors, parsed


# ============================================================
# DRND-ST full projection model
# ============================================================

def nu_simple(x):
    x = np.maximum(np.asarray(x, dtype=float), 1e-300)
    return 0.5 + np.sqrt(0.25 + 1.0 / x)


def a0_from_chi(H0_km_s_Mpc, chi):
    H0_s = H0_km_s_Mpc * 1000.0 / MPC_M
    return chi * C_M_S * H0_s


def mbar_enclosed(r_kpc, Mbar_Msun, Re_kpc, profile_type):
    r = np.asarray(r_kpc, dtype=float)
    M = float(Mbar_Msun)
    Re = max(float(Re_kpc), 1e-9)
    p = str(profile_type).lower()

    if p == "point":
        return np.full_like(r, M, dtype=float)

    if p == "hernquist":
        # Hernquist projected half-light Re ≈ 1.8153 a
        a = Re / 1.8153
        return M * (r**2 / (r + a)**2)

    if p in {"exponential_sphere", "exponential"}:
        # Spherical approximation to cumulative exponential mass.
        x = r / Re
        return M * (1.0 - np.exp(-x) * (1.0 + x + 0.5 * x**2))

    raise ValueError(f"Unsupported profile_type: {profile_type}")


def build_effective_profile(Mbar_Msun, Re_kpc, profile_type, a0, r_grid_kpc):
    r = np.asarray(r_grid_kpc, dtype=float)
    Menc = mbar_enclosed(r, Mbar_Msun, Re_kpc, profile_type)
    gbar = G_SI * (Menc * MSUN_KG) / np.maximum((r * KPC_M)**2, 1e-300)
    gst = gbar * nu_simple(gbar / a0)
    Meff_kg = gst * (r * KPC_M)**2 / G_SI
    Meff_msun = Meff_kg / MSUN_KG

    # enforce monotonicity to avoid numerical derivative negatives from noise
    Meff_msun = np.maximum.accumulate(Meff_msun)

    dM_dr = np.gradient(Meff_msun * MSUN_KG, r * KPC_M)
    rho = dM_dr / (4.0 * math.pi * np.maximum((r * KPC_M)**2, 1e-300))  # kg/m^3
    rho = np.maximum(rho, 0.0)
    return {"r_kpc": r, "Mbar_enc": Menc, "Meff_enc": Meff_msun, "rho_eff": rho, "gbar": gbar, "gst": gst}


def sigma_projected(R_kpc, r_grid_kpc, rho_kg_m3):
    R = float(R_kpc) * KPC_M
    r = np.asarray(r_grid_kpc, dtype=float) * KPC_M
    rho = np.asarray(rho_kg_m3, dtype=float)

    mask = r > R * (1.0 + 1e-8)
    if np.sum(mask) < 5:
        return np.nan

    rr = r[mask]
    rh = rho[mask]
    integrand = rh * rr / np.sqrt(np.maximum(rr**2 - R**2, 1e-300))
    Sigma_kg_m2 = 2.0 * np.trapz(integrand, rr) if hasattr(np, "trapz") else 2.0 * np.trapezoid(integrand, rr)
    return Sigma_kg_m2 * (PC_M**2) / MSUN_KG


def delta_sigma_profile(R_eval_kpc, Mbar_Msun, Re_kpc, profile_type, a0):
    R_eval = np.asarray(R_eval_kpc, dtype=float)
    rmin = max(min(np.min(R_eval) / 100.0, 1e-3), 1e-5)
    rmax = max(np.max(R_eval) * 100.0, 1e4)
    r_grid = np.geomspace(rmin, rmax, 1400)

    prof = build_effective_profile(Mbar_Msun, Re_kpc, profile_type, a0, r_grid)
    Sigma = np.array([sigma_projected(R, prof["r_kpc"], prof["rho_eff"]) for R in R_eval])

    # Compute Sigma_bar via cumulative integral on a dense R grid.
    R_dense = np.geomspace(max(np.min(R_eval)/20.0, 1e-4), np.max(R_eval)*1.05, 400)
    Sigma_dense = np.array([sigma_projected(R, prof["r_kpc"], prof["rho_eff"]) for R in R_dense])
    valid = np.isfinite(Sigma_dense)
    if np.sum(valid) < 10:
        return np.full_like(R_eval, np.nan), Sigma

    Rd = R_dense[valid]
    Sd = Sigma_dense[valid]
    integrand = Sd * Rd
    # cumulative trapezoid manually to avoid version issues
    cum = np.zeros_like(Rd)
    for i in range(1, len(Rd)):
        cum[i] = cum[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * (Rd[i] - Rd[i-1])
    Sigma_bar_dense = 2.0 * cum / np.maximum(Rd**2, 1e-300)

    Sigma_bar = np.interp(R_eval, Rd, Sigma_bar_dense)
    DeltaSigma = Sigma_bar - Sigma
    return DeltaSigma, Sigma


def read_lens_bins(path: Optional[str]):
    if not path or not Path(path).exists():
        return {}
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lb = str(row.get("lens_bin", "")).strip()
            if not lb:
                continue
            M = to_float(row.get("Mbar_Msun"))
            Re = to_float(row.get("Re_kpc"))
            prof = row.get("profile_type", "hernquist")
            if np.isfinite(M) and np.isfinite(Re):
                rows.append((lb, {"Mbar_Msun": M, "Re_kpc": Re, "profile_type": prof}))
    return dict(rows)


def read_physical_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lb = str(row.get("lens_bin", "")).strip()
            R = to_float(row.get("R_kpc"))
            ds = to_float(row.get("DeltaSigma"))
            err = to_float(row.get("DeltaSigma_err"))
            if lb and np.isfinite(R) and np.isfinite(ds) and np.isfinite(err) and err > 0:
                rows.append({"lens_bin": lb, "R_kpc": R, "DeltaSigma": ds, "DeltaSigma_err": err})
    return rows


def evaluate_physical_lensing(data_rows, lens_bins, a0):
    preds = []
    chi2 = 0.0
    n = 0
    for lb in sorted(set(r["lens_bin"] for r in data_rows)):
        meta = lens_bins.get(lb)
        if meta is None:
            continue
        sub = [r for r in data_rows if r["lens_bin"] == lb]
        R = np.array([r["R_kpc"] for r in sub], dtype=float)
        obs = np.array([r["DeltaSigma"] for r in sub], dtype=float)
        err = np.array([r["DeltaSigma_err"] for r in sub], dtype=float)
        pred, Sigma = delta_sigma_profile(R, meta["Mbar_Msun"], meta["Re_kpc"], meta["profile_type"], a0)

        for i, r in enumerate(sub):
            if not np.isfinite(pred[i]):
                continue
            sigma_log = max(err[i] / max(abs(obs[i]), 1e-30) / math.log(10), 0.03)
            res = (math.log10(max(obs[i], 1e-30)) - math.log10(max(pred[i], 1e-30))) / sigma_log
            chi2 += res**2
            n += 1
            preds.append({
                "lens_bin": lb,
                "R_kpc": float(R[i]),
                "DeltaSigma_obs": float(obs[i]),
                "DeltaSigma_err": float(err[i]),
                "DeltaSigma_DRND_ST": float(pred[i]),
                "Sigma_DRND_ST": float(Sigma[i]) if np.isfinite(Sigma[i]) else np.nan,
                "Mbar_Msun": meta["Mbar_Msun"],
                "Re_kpc": meta["Re_kpc"],
                "profile_type": meta["profile_type"],
            })
    return {"chi2": float(chi2), "n": int(n), "chi2_per_point": float(chi2/max(n, 1)), "pred_rows": preds}


def fit_a0_physical(data_rows, lens_bins, fixed_a0):
    fixed = evaluate_physical_lensing(data_rows, lens_bins, fixed_a0)

    def obj(loga0):
        return evaluate_physical_lensing(data_rows, lens_bins, 10**loga0)["chi2"]

    res = minimize_scalar(obj, bounds=(-11.5, -8.8), method="bounded")
    best_a0 = 10**float(res.x)
    best = evaluate_physical_lensing(data_rows, lens_bins, best_a0)
    best["a0"] = best_a0
    best["log10_a0"] = float(res.x)
    best["fit_success"] = bool(res.success)
    fixed["a0"] = fixed_a0
    fixed["log10_a0"] = math.log10(fixed_a0)
    return fixed, best


def plot_physical_fit(outdir: Path, pred_rows: List[Dict]):
    if not pred_rows:
        return
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for lb in sorted(set(r["lens_bin"] for r in pred_rows)):
        sub = [r for r in pred_rows if r["lens_bin"] == lb]
        R = np.array([r["R_kpc"] for r in sub])
        obs = np.array([r["DeltaSigma_obs"] for r in sub])
        err = np.array([r["DeltaSigma_err"] for r in sub])
        pred = np.array([r["DeltaSigma_DRND_ST"] for r in sub])
        idx = np.argsort(R)
        ax.errorbar(R[idx], obs[idx], yerr=err[idx], fmt="o", ms=3, label=f"{lb} obs")
        ax.plot(R[idx], pred[idx], label=f"{lb} DRND-ST")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("DeltaSigma [Msun/pc^2]")
    ax.set_title("Full projected DRND-ST lensing model")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plotdir / "full_projected_lensing_fit.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="DRND_ST_FULL_LENSING")
    ap.add_argument("--download-url", default=DEFAULT_REPO_ZIP)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--data-root", default=None)

    ap.add_argument("--physical-csv", default=None)
    ap.add_argument("--lens-bins-csv", default=None)

    ap.add_argument("--H0", type=float, default=67.4)
    ap.add_argument("--chi", type=float, default=0.18)
    ap.add_argument("--fit-a0", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tabledir = outdir / "tables"
    tabledir.mkdir(parents=True, exist_ok=True)

    repo_root = None
    download_status = None
    if args.data_root:
        repo_root = Path(args.data_root)
    elif not args.physical_csv:
        try:
            repo_root = download_repo(outdir, args.download_url)
            download_status = f"downloaded_or_reused:{repo_root}"
        except Exception as e:
            download_status = f"failed:{e}"
            print(f"[WARN] download failed: {e}")
            repo_root = None

    if args.download_only:
        summary = {"download_status": download_status, "repo_root": str(repo_root) if repo_root else None}
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    discovered = []
    parsed = []
    vectors = []
    if repo_root and repo_root.exists():
        discovered = list_data_files(repo_root)
        write_csv(tabledir / "discovered_files.csv", discovered)
        vectors, parsed = make_unified_vector(discovered)
        write_csv(tabledir / "parsed_file_summary.csv", parsed)
        write_csv(tabledir / "parsed_data_vectors.csv", vectors)

        # Generate lens-bin template.
        lens_bins = sorted(set(v["lens_bin"] for v in vectors))
        template = [{
            "lens_bin": lb,
            "Mbar_Msun": "",
            "Re_kpc": "",
            "profile_type": "hernquist",
            "note": "Fill Mbar_Msun and Re_kpc to run physical DRND-ST projection."
        } for lb in lens_bins]
        write_csv(outdir / "lens_bins_template.csv", template,
                  preferred=["lens_bin", "Mbar_Msun", "Re_kpc", "profile_type", "note"])

    if args.scan_only or not args.physical_csv:
        summary = {
            "mode": "download_scan",
            "download_status": download_status,
            "repo_root": str(repo_root) if repo_root else None,
            "n_discovered_files": len(discovered),
            "n_parsed_file_summaries": len(parsed),
            "n_unified_vector_rows": len(vectors),
            "physical_run": False,
            "next_step": "Inspect tables/parsed_file_summary.csv and lens_bins_template.csv. Fill Mbar_Msun/Re_kpc or provide --physical-csv and --lens-bins-csv.",
        }
        (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
        print(json.dumps(safe_json(summary), indent=2))
        return

    # Physical projected model run.
    data_rows = read_physical_csv(args.physical_csv)
    lens_bins = read_lens_bins(args.lens_bins_csv)
    if not data_rows:
        raise SystemExit("No physical lensing rows found. Need lens_bin,R_kpc,DeltaSigma,DeltaSigma_err.")
    if not lens_bins:
        raise SystemExit("No lens-bin metadata found. Need lens_bin,Mbar_Msun,Re_kpc,profile_type.")

    fixed_a0 = a0_from_chi(args.H0, args.chi)
    fixed, best = fit_a0_physical(data_rows, lens_bins, fixed_a0)

    write_csv(tabledir / "prediction_table_fixed_a0.csv", fixed["pred_rows"])
    fit_rows = [
        {"case": "fixed_a0", **{k: v for k, v in fixed.items() if k != "pred_rows"}},
        {"case": "best_fit_a0", **{k: v for k, v in best.items() if k != "pred_rows"}},
    ]
    write_csv(tabledir / "fit_table.csv", fit_rows)
    plot_physical_fit(outdir, fixed["pred_rows"])

    summary = {
        "mode": "physical_full_projection",
        "physical_csv": args.physical_csv,
        "lens_bins_csv": args.lens_bins_csv,
        "H0": args.H0,
        "chi": args.chi,
        "a0_fixed": fixed_a0,
        "fixed_a0_metrics": {k: v for k, v in fixed.items() if k != "pred_rows"},
        "best_fit_a0_metrics": {k: v for k, v in best.items() if k != "pred_rows"},
        "decision": {
            "best_a0_near_rotation_scale_factor_2": 0.5 <= best["a0"] / fixed_a0 <= 2.0,
            "note": "Full projection test is meaningful only if lens-bin baryonic masses/profiles are credible."
        }
    }
    (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
    print(json.dumps(safe_json(summary), indent=2))


if __name__ == "__main__":
    main()

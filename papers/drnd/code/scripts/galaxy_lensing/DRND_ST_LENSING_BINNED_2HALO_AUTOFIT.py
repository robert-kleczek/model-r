#!/usr/bin/env python3
"""
DRND_ST_LENSING_BINNED_2HALO_AUTOFIT.py

Purpose
-------
Binned DRND-ST lensing autofit with an explicit environmental / two-halo term.

Previous result:
    - Fixed a0 = chi*c*H0 strongly beat baryon-only.
    - Free a0 still ran high by ~9.5x even after mass binning.
Interpretation:
    - Missing large-scale environment/two-halo contribution may be forcing a0 upward.

This script tests that.

Model
-----
For each bin:

    DeltaSigma_total(R)
      = DeltaSigma_DRND_ST(R; Mbar_bin, Re_bin, a0)
        + A_2h_bin * (R / R0)^(-p)

Where:
    - Mbar_bin and Re_bin are fitted per bin.
    - A_2h can be:
        * one global amplitude shared by bins
        * one amplitude per bin
    - p can be:
        * fixed
        * fitted globally
    - a0 can be:
        * fixed to chi*c*H0
        * fitted globally

Models compared:
    1. baryon_only
    2. drnd_fixed_a0
    3. drnd_fixed_a0_plus_2halo
    4. drnd_free_a0_plus_2halo

Main question
-------------
Does adding a plausible two-halo/environment term bring free a0 back toward
the rotation/RAR scale?

Good sign:
    free_a0_over_fixed drops from ~9.5 to ~1-2
    fixed_a0_plus_2halo fits well
    A_2h values are positive and not absurdly dominant at small R

Usage
-----
    python DRND_ST_LENSING_BINNED_2HALO_AUTOFIT.py ^
      --lensing-csv lensing_data.csv ^
      --bin-mode auto ^
      --n-bins 4 ^
      --fit-a0 ^
      --twohalo-mode per-bin ^
      --fit-twohalo-slope ^
      --outdir DRND_ST_LENSING_2HALO_RESULTS

Conservative first run:

    python DRND_ST_LENSING_BINNED_2HALO_AUTOFIT.py ^
      --lensing-csv lensing_data.csv ^
      --bin-mode auto ^
      --n-bins 4 ^
      --fit-a0 ^
      --twohalo-mode global ^
      --twohalo-slope 0.8 ^
      --outdir DRND_ST_LENSING_2HALO_GLOBAL

Outputs
-------
    summary.json
    tables/binned_lensing_data.csv
    tables/fit_table.csv
    tables/bin_parameters.csv
    tables/twohalo_parameters.csv
    tables/predictions_*.csv
    plots/binned_2halo_fit.png
    plots/a0_comparison.png

Caution
-------
This is still a diagnostic:
    - diagonal errors only
    - no real survey covariance
    - no halo occupation model
    - no miscentering/satellite model
    - two-halo term is phenomenological
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize


G_SI = 6.67430e-11
MSUN_KG = 1.98847e30
PC_M = 3.0856775814913673e16
KPC_M = 1e3 * PC_M
MPC_M = 1e6 * PC_M
C_M_S = 299792458.0


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


def pick(row: Dict[str, str], names: List[str], default=np.nan):
    m = {norm_col(k): v for k, v in row.items()}
    for n in names:
        k = norm_col(n)
        if k in m:
            val = to_float(m[k])
            if np.isfinite(val):
                return val
    return default


def pick_str(row: Dict[str, str], names: List[str], default=""):
    m = {norm_col(k): v for k, v in row.items()}
    for n in names:
        k = norm_col(n)
        if k in m and m[k] is not None:
            return str(m[k]).strip()
    return default


def write_csv(path: Path, rows: List[Dict], preferred: List[str] | None = None):
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


def read_lensing_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lb = pick_str(row, ["lens_bin", "bin", "sample"], default="")
            R = pick(row, ["R_kpc", "R", "rp", "radius"])
            ds = pick(row, ["DeltaSigma", "delta_sigma", "deltasigma", "ds", "signal"])
            err = pick(row, ["DeltaSigma_err", "delta_sigma_err", "deltasigma_err", "ds_err", "err", "error", "sigma"])
            Mbar = pick(row, ["Mbar_Msun", "Mbar", "mbar", "baryonic_mass"])
            logMbar = pick(row, ["logMbar", "log_mbar", "logM", "log_mass"])
            if not np.isfinite(logMbar) and np.isfinite(Mbar) and Mbar > 0:
                logMbar = math.log10(Mbar)

            if np.isfinite(R) and np.isfinite(ds) and np.isfinite(err) and R > 0 and err > 0 and ds > 0:
                rows.append({
                    "lens_bin_original": lb,
                    "R_kpc": float(R),
                    "DeltaSigma": float(ds),
                    "DeltaSigma_err": float(err),
                    "logMbar_input": float(logMbar) if np.isfinite(logMbar) else np.nan,
                })
    if not rows:
        raise SystemExit("No usable rows. Need R_kpc, DeltaSigma, DeltaSigma_err.")
    return rows


def assign_bins(rows: List[Dict], mode: str, n_bins: int):
    mode_used = mode
    existing = [str(r.get("lens_bin_original", "")).strip() for r in rows]
    existing_unique = sorted(set(x for x in existing if x))

    if mode == "auto":
        if len(existing_unique) > 1:
            mode_used = "existing"
        elif any(np.isfinite(r.get("logMbar_input", np.nan)) for r in rows):
            mode_used = "mass"
        else:
            mode_used = "signal"

    if mode_used == "existing":
        if not existing_unique:
            for r in rows:
                r["lens_bin"] = "all"
        else:
            for r in rows:
                r["lens_bin"] = str(r.get("lens_bin_original", "")).strip() or "unlabeled"
        return rows, mode_used

    if mode_used == "mass":
        vals = np.array([r.get("logMbar_input", np.nan) for r in rows], dtype=float)
        if not np.all(np.isfinite(vals)):
            raise SystemExit("bin-mode mass requires Mbar/logMbar for all rows.")
        key = vals
    elif mode_used == "radius":
        key = np.array([r["R_kpc"] for r in rows], dtype=float)
    elif mode_used == "signal":
        key = np.array([r["DeltaSigma"] for r in rows], dtype=float)
    else:
        raise SystemExit(f"Unknown bin mode: {mode}")

    n_bins = max(1, min(int(n_bins), len(rows)))
    edges = np.unique(np.quantile(key, np.linspace(0, 1, n_bins + 1)))
    if len(edges) <= 2:
        for r in rows:
            r["lens_bin"] = "bin0"
        return rows, mode_used

    for r, v in zip(rows, key):
        idx = np.searchsorted(edges, v, side="right") - 1
        idx = max(0, min(idx, len(edges) - 2))
        r["lens_bin"] = f"{mode_used}_bin{idx}"
    return rows, mode_used


def a0_from_chi(H0_km_s_Mpc, chi):
    H0_s = H0_km_s_Mpc * 1000.0 / MPC_M
    return chi * C_M_S * H0_s


def nu_simple(x):
    x = np.maximum(np.asarray(x, dtype=float), 1e-300)
    return 0.5 + np.sqrt(0.25 + 1.0/x)


def mbar_enclosed(r_kpc, Mbar_Msun, Re_kpc, profile_type="hernquist"):
    r = np.asarray(r_kpc, dtype=float)
    M = float(Mbar_Msun)
    Re = max(float(Re_kpc), 1e-5)
    p = str(profile_type).lower()
    if p == "point":
        return np.full_like(r, M)
    if p == "hernquist":
        a = Re / 1.8153
        return M * r**2 / np.maximum((r+a)**2, 1e-300)
    if p in {"exponential", "exponential_sphere"}:
        x = r/Re
        return M * (1.0 - np.exp(-x)*(1.0+x+0.5*x*x))
    raise ValueError(profile_type)


def build_profile(logM, logRe, a0, model="drnd", profile_type="hernquist", r_grid_kpc=None):
    if r_grid_kpc is None:
        r_grid_kpc = np.geomspace(1e-3, 1e5, 1400)
    Mbar = 10**logM
    Re = 10**logRe
    Menc = mbar_enclosed(r_grid_kpc, Mbar, Re, profile_type)
    r_m = r_grid_kpc*KPC_M
    gbar = G_SI*(Menc*MSUN_KG)/np.maximum(r_m*r_m, 1e-300)
    if model == "baryon":
        g = gbar
    elif model == "drnd":
        g = gbar*nu_simple(gbar/a0)
    else:
        raise ValueError(model)
    Meff = g*r_m*r_m/G_SI/MSUN_KG
    Meff = np.maximum.accumulate(Meff)
    dM_dr = np.gradient(Meff*MSUN_KG, r_m)
    rho = dM_dr/(4*math.pi*np.maximum(r_m*r_m, 1e-300))
    return r_grid_kpc, np.maximum(rho, 0), Meff


def sigma_projected(R_kpc, r_grid_kpc, rho):
    Rm = float(R_kpc)*KPC_M
    rm = np.asarray(r_grid_kpc)*KPC_M
    mask = rm > Rm*(1+1e-8)
    if np.sum(mask) < 10:
        return np.nan
    rr = rm[mask]
    rh = rho[mask]
    integrand = rh*rr/np.sqrt(np.maximum(rr*rr - Rm*Rm, 1e-300))
    if hasattr(np, "trapezoid"):
        sig = 2*np.trapezoid(integrand, rr)
    else:
        sig = 2*np.trapz(integrand, rr)
    return sig * PC_M**2 / MSUN_KG


def delta_sigma_model(R_eval, logM, logRe, a0, model="drnd", profile_type="hernquist"):
    R_eval = np.asarray(R_eval, dtype=float)
    rmin = max(np.min(R_eval)/200, 1e-4)
    rmax = max(np.max(R_eval)*200, 1e4)
    rgrid = np.geomspace(rmin, rmax, 1400)
    rgrid, rho, _ = build_profile(logM, logRe, a0, model=model, profile_type=profile_type, r_grid_kpc=rgrid)

    Sigma = np.array([sigma_projected(R, rgrid, rho) for R in R_eval])
    Rdense = np.geomspace(max(np.min(R_eval)/50, 1e-4), np.max(R_eval)*1.05, 400)
    Sdense = np.array([sigma_projected(R, rgrid, rho) for R in Rdense])
    valid = np.isfinite(Sdense)
    Rd, Sd = Rdense[valid], Sdense[valid]
    if len(Rd) < 20:
        return np.full_like(R_eval, np.nan)

    integ = Sd*Rd
    cum = np.zeros_like(Rd)
    for i in range(1, len(Rd)):
        cum[i] = cum[i-1] + 0.5*(integ[i]+integ[i-1])*(Rd[i]-Rd[i-1])
    Sbar = 2*cum/np.maximum(Rd*Rd, 1e-300)
    return np.interp(R_eval, Rd, Sbar) - Sigma


def twohalo_term(R_kpc, logA, p, R0_kpc):
    A = 10**logA
    return A * np.power(np.maximum(np.asarray(R_kpc, dtype=float)/R0_kpc, 1e-12), -p)


def mass_size_prior(logM, logRe, Re0=5.0, alpha=0.25, sigma_dex=0.35):
    logRe_pred = math.log10(Re0) + alpha*(logM-10.0)
    return ((logRe-logRe_pred)/sigma_dex)**2


def plausibility(logM, logRe):
    Re = 10**logRe
    return {
        "Mbar_plausible_1e8_1e12p5": bool(8.0 <= logM <= 12.5),
        "Re_plausible_0p2_80_kpc": bool(0.2 <= Re <= 80.0),
        "mass_size_prior_chi2": mass_size_prior(logM, logRe),
    }


class ParamLayout:
    def __init__(self, bins, include_2h=False, twohalo_mode="global", fit_p=False, fit_a0=False, a0_fixed=1e-10, p_fixed=0.8):
        self.bins = bins
        self.include_2h = include_2h
        self.twohalo_mode = twohalo_mode
        self.fit_p = fit_p
        self.fit_a0 = fit_a0
        self.a0_fixed = a0_fixed
        self.p_fixed = p_fixed
        self.slices = {}
        idx = 0
        self.slices["profile"] = (idx, idx+2*len(bins)); idx += 2*len(bins)

        if include_2h:
            if twohalo_mode == "global":
                self.slices["logA"] = (idx, idx+1); idx += 1
            elif twohalo_mode == "per-bin":
                self.slices["logA"] = (idx, idx+len(bins)); idx += len(bins)
            else:
                raise ValueError(twohalo_mode)
            if fit_p:
                self.slices["p"] = (idx, idx+1); idx += 1

        if fit_a0:
            self.slices["loga0"] = (idx, idx+1); idx += 1

        self.n = idx

    def unpack(self, p):
        prof = p[self.slices["profile"][0]:self.slices["profile"][1]]
        logM = prof[0::2]
        logRe = prof[1::2]

        if self.include_2h:
            a, b = self.slices["logA"]
            raw = p[a:b]
            if self.twohalo_mode == "global":
                logA = {lb: raw[0] for lb in self.bins}
            else:
                logA = {lb: raw[i] for i, lb in enumerate(self.bins)}
            if self.fit_p:
                aa, bb = self.slices["p"]
                slope = p[aa]
            else:
                slope = self.p_fixed
        else:
            logA = {lb: -99.0 for lb in self.bins}
            slope = self.p_fixed

        if self.fit_a0:
            a, b = self.slices["loga0"]
            a0 = 10**p[a]
        else:
            a0 = self.a0_fixed

        return logM, logRe, logA, slope, a0


def initial_params_and_bounds(bins, layout, data_by_bin):
    p0 = np.zeros(layout.n)
    bounds = [(None, None)]*layout.n

    # Profiles
    a, b = layout.slices["profile"]
    vals = []
    bds = []
    for lb in bins:
        logM = 11.0
        logRe = math.log10(5.0*(10**(logM-10.0))**0.25)
        vals.extend([logM, logRe])
        bds.extend([(8.0, 12.8), (math.log10(0.2), math.log10(80.0))])
    p0[a:b] = vals
    bounds[a:b] = bds

    if layout.include_2h:
        a, b = layout.slices["logA"]
        logA_vals = []
        for i in range(b-a):
            # Initialize around 10% median signal.
            if layout.twohalo_mode == "global":
                all_ds = [r["DeltaSigma"] for rows in data_by_bin.values() for r in rows]
                med = np.median(all_ds)
            else:
                lb = bins[i]
                med = np.median([r["DeltaSigma"] for r in data_by_bin[lb]])
            logA_vals.append(math.log10(max(0.05*med, 1e-12)))
        p0[a:b] = logA_vals
        # broad positive amplitude in Msun/pc^2
        bounds[a:b] = [(-8.0, 6.0)]*(b-a)

        if layout.fit_p:
            a, b = layout.slices["p"]
            p0[a] = layout.p_fixed
            bounds[a] = (0.0, 3.0)

    if layout.fit_a0:
        a, b = layout.slices["loga0"]
        p0[a] = math.log10(layout.a0_fixed)
        bounds[a] = (-11.5, -8.8)

    return p0, bounds


def evaluate(data_by_bin, bins, params, layout, model, use_prior, prior_weight, profile_type, R0_kpc, twohalo_prior_weight):
    logM, logRe, logA, slope, a0 = layout.unpack(params)
    chi2 = 0.0
    n = 0
    preds = []
    bin_rows = []

    for bi, lb in enumerate(bins):
        sub = data_by_bin[lb]
        R = np.array([r["R_kpc"] for r in sub])
        obs = np.array([r["DeltaSigma"] for r in sub])
        err = np.array([r["DeltaSigma_err"] for r in sub])

        pred_base = delta_sigma_model(R, logM[bi], logRe[bi], a0, model=model, profile_type=profile_type)
        pred_2h = twohalo_term(R, logA[lb], slope, R0_kpc) if layout.include_2h else np.zeros_like(R)
        pred = pred_base + pred_2h

        chi2_bin = 0.0
        if use_prior:
            pr = prior_weight*mass_size_prior(logM[bi], logRe[bi])
            chi2 += pr
            chi2_bin += pr

        # Penalize strong two-halo dominance at the smallest radii: phenomenological but stabilizing.
        if layout.include_2h and twohalo_prior_weight > 0:
            small = R <= np.quantile(R, 0.35)
            if np.any(small):
                ratio = np.median(pred_2h[small]/np.maximum(pred_base[small], 1e-30))
                # softly prefer two-halo not dominating inner stack by >100%
                if ratio > 1.0:
                    penalty = twohalo_prior_weight*((math.log10(ratio)) / 0.5)**2
                    chi2 += penalty
                    chi2_bin += penalty

        for i in range(len(R)):
            if not np.isfinite(pred[i]) or pred[i] <= 0:
                val = 1e8
            else:
                sigma_log = max(err[i]/max(obs[i], 1e-30)/math.log(10), 0.03)
                val = ((math.log10(max(obs[i],1e-30))-math.log10(max(pred[i],1e-30)))/sigma_log)**2
                preds.append({
                    "model": model + ("_2halo" if layout.include_2h else ""),
                    "lens_bin": lb,
                    "R_kpc": float(R[i]),
                    "DeltaSigma_obs": float(obs[i]),
                    "DeltaSigma_err": float(err[i]),
                    "DeltaSigma_base": float(pred_base[i]),
                    "DeltaSigma_2halo": float(pred_2h[i]),
                    "DeltaSigma_pred": float(pred[i]),
                    "twohalo_fraction": float(pred_2h[i]/max(pred[i], 1e-30)),
                    "logMbar": float(logM[bi]),
                    "Mbar_Msun": float(10**logM[bi]),
                    "Re_kpc": float(10**logRe[bi]),
                    "a0": float(a0),
                    "twohalo_logA": float(logA[lb]),
                    "twohalo_slope": float(slope),
                })
            chi2 += val
            chi2_bin += val
            n += 1

        bin_rows.append({
            "model": model + ("_2halo" if layout.include_2h else ""),
            "lens_bin": lb,
            "n_points": len(R),
            "chi2_bin": float(chi2_bin),
            "chi2_per_point_bin": float(chi2_bin/max(len(R),1)),
            "logMbar": float(logM[bi]),
            "Mbar_Msun": float(10**logM[bi]),
            "logRe": float(logRe[bi]),
            "Re_kpc": float(10**logRe[bi]),
            "twohalo_logA": float(logA[lb]),
            "twohalo_A_at_R0": float(10**logA[lb]) if layout.include_2h else 0.0,
            "twohalo_slope": float(slope),
            **plausibility(float(logM[bi]), float(logRe[bi])),
        })

    return chi2, n, preds, bin_rows, a0, slope


def fit_case(data_by_bin, bins, case_name, model, include_2h, twohalo_mode, fit_p, fit_a0, a0_fixed, p_fixed,
             use_prior, prior_weight, profile_type, R0_kpc, twohalo_prior_weight):
    layout = ParamLayout(bins, include_2h=include_2h, twohalo_mode=twohalo_mode, fit_p=fit_p, fit_a0=fit_a0,
                         a0_fixed=a0_fixed, p_fixed=p_fixed)
    p0, bounds = initial_params_and_bounds(bins, layout, data_by_bin)

    def obj(p):
        return evaluate(data_by_bin, bins, p, layout, model, use_prior, prior_weight, profile_type, R0_kpc, twohalo_prior_weight)[0]

    res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1e-8})
    chi2, n, preds, bin_rows, a0, slope = evaluate(data_by_bin, bins, res.x, layout, model, use_prior, prior_weight,
                                                    profile_type, R0_kpc, twohalo_prior_weight)

    twohalo_inner_fracs = []
    twohalo_outer_fracs = []
    if include_2h and preds:
        for lb in bins:
            sub = [r for r in preds if r["lens_bin"] == lb]
            R = np.array([r["R_kpc"] for r in sub])
            frac = np.array([r["twohalo_fraction"] for r in sub])
            if len(R):
                twohalo_inner_fracs.append(float(np.median(frac[R <= np.quantile(R, 0.35)])))
                twohalo_outer_fracs.append(float(np.median(frac[R >= np.quantile(R, 0.65)])))

    return {
        "case": case_name,
        "model": model,
        "include_2halo": include_2h,
        "twohalo_mode": twohalo_mode if include_2h else "none",
        "fit_twohalo_slope": fit_p if include_2h else False,
        "fit_a0": fit_a0,
        "a0": float(a0),
        "log10_a0": float(math.log10(a0)),
        "a0_over_fixed": float(a0/a0_fixed),
        "twohalo_slope": float(slope),
        "chi2": float(chi2),
        "n": int(n),
        "chi2_per_point": float(chi2/max(n,1)),
        "success": bool(res.success),
        "message": str(res.message),
        "median_twohalo_fraction_inner": float(np.median(twohalo_inner_fracs)) if twohalo_inner_fracs else 0.0,
        "median_twohalo_fraction_outer": float(np.median(twohalo_outer_fracs)) if twohalo_outer_fracs else 0.0,
        "params": res.x.tolist(),
        "bin_rows": bin_rows,
        "pred_rows": preds,
    }


def plot_results(outdir: Path, results):
    plotdir = outdir/"plots"
    plotdir.mkdir(parents=True, exist_ok=True)

    # Best 2halo or fixed
    prefer_order = ["drnd_free_a0_2halo", "drnd_fixed_a0_2halo", "drnd_fixed_a0"]
    chosen = None
    for name in prefer_order:
        chosen = next((r for r in results if r["case"] == name), None)
        if chosen:
            break
    if chosen:
        rows = chosen["pred_rows"]
        fig, ax = plt.subplots(figsize=(9,6))
        for lb in sorted(set(r["lens_bin"] for r in rows)):
            sub = [r for r in rows if r["lens_bin"]==lb]
            R = np.array([r["R_kpc"] for r in sub])
            obs = np.array([r["DeltaSigma_obs"] for r in sub])
            err = np.array([r["DeltaSigma_err"] for r in sub])
            base = np.array([r["DeltaSigma_base"] for r in sub])
            th = np.array([r["DeltaSigma_2halo"] for r in sub])
            pred = np.array([r["DeltaSigma_pred"] for r in sub])
            idx = np.argsort(R)
            ax.errorbar(R[idx], obs[idx], yerr=err[idx], fmt="o", ms=3, alpha=0.75, label=f"{lb} obs")
            ax.plot(R[idx], pred[idx], label=f"{lb} total")
            if np.max(th) > 0:
                ax.plot(R[idx], base[idx], ls="--", alpha=0.7)
                ax.plot(R[idx], th[idx], ls=":", alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("R [kpc]"); ax.set_ylabel("DeltaSigma [Msun/pc^2]")
        ax.set_title(f"Binned DRND-ST lensing with environment: {chosen['case']}")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plotdir/"binned_2halo_fit.png", dpi=180)
        plt.close(fig)

    # a0 comparison
    fig, ax = plt.subplots(figsize=(8,5))
    labels = [r["case"] for r in results]
    vals = [r["a0_over_fixed"] for r in results]
    ax.bar(range(len(vals)), vals)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.axhline(2.0, color="red", ls="--", lw=1)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("a0 / fixed a0")
    ax.set_title("Does two-halo term reduce a0 runaway?")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(plotdir/"a0_comparison.png", dpi=180)
    plt.close(fig)


def safe_json(obj):
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return None
        return obj
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lensing-csv", required=True)
    ap.add_argument("--bin-mode", default="auto", choices=["auto","existing","radius","signal","mass"])
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--H0", type=float, default=67.4)
    ap.add_argument("--chi", type=float, default=0.18)

    ap.add_argument("--fit-a0", action="store_true")
    ap.add_argument("--twohalo-mode", default="per-bin", choices=["global","per-bin"])
    ap.add_argument("--twohalo-slope", type=float, default=0.8)
    ap.add_argument("--fit-twohalo-slope", action="store_true")
    ap.add_argument("--R0-kpc", type=float, default=300.0)
    ap.add_argument("--twohalo-prior-weight", type=float, default=0.2)

    ap.add_argument("--profile-type", default="hernquist", choices=["hernquist","exponential_sphere","point"])
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--outdir", default="DRND_ST_LENSING_2HALO_RESULTS")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tabledir = outdir/"tables"
    tabledir.mkdir(parents=True, exist_ok=True)

    rows = read_lensing_csv(args.lensing_csv)
    rows, mode_used = assign_bins(rows, args.bin_mode, args.n_bins)
    write_csv(tabledir/"binned_lensing_data.csv", rows,
              preferred=["lens_bin","lens_bin_original","R_kpc","DeltaSigma","DeltaSigma_err","logMbar_input"])

    data_by_bin = {}
    for r in rows:
        data_by_bin.setdefault(r["lens_bin"], []).append(r)
    bins = sorted(data_by_bin.keys())
    a0_fixed = a0_from_chi(args.H0, args.chi)
    use_prior = not args.no_prior

    results = []

    # Baseline cases.
    results.append(fit_case(data_by_bin, bins, "baryon_only", "baryon",
                            include_2h=False, twohalo_mode=args.twohalo_mode, fit_p=False, fit_a0=False,
                            a0_fixed=a0_fixed, p_fixed=args.twohalo_slope,
                            use_prior=use_prior, prior_weight=args.prior_weight, profile_type=args.profile_type,
                            R0_kpc=args.R0_kpc, twohalo_prior_weight=args.twohalo_prior_weight))

    results.append(fit_case(data_by_bin, bins, "drnd_fixed_a0", "drnd",
                            include_2h=False, twohalo_mode=args.twohalo_mode, fit_p=False, fit_a0=False,
                            a0_fixed=a0_fixed, p_fixed=args.twohalo_slope,
                            use_prior=use_prior, prior_weight=args.prior_weight, profile_type=args.profile_type,
                            R0_kpc=args.R0_kpc, twohalo_prior_weight=args.twohalo_prior_weight))

    # 2-halo fixed a0.
    results.append(fit_case(data_by_bin, bins, "drnd_fixed_a0_2halo", "drnd",
                            include_2h=True, twohalo_mode=args.twohalo_mode, fit_p=args.fit_twohalo_slope, fit_a0=False,
                            a0_fixed=a0_fixed, p_fixed=args.twohalo_slope,
                            use_prior=use_prior, prior_weight=args.prior_weight, profile_type=args.profile_type,
                            R0_kpc=args.R0_kpc, twohalo_prior_weight=args.twohalo_prior_weight))

    if args.fit_a0:
        # Free a0 without 2halo for reference.
        results.append(fit_case(data_by_bin, bins, "drnd_free_a0_no2halo", "drnd",
                                include_2h=False, twohalo_mode=args.twohalo_mode, fit_p=False, fit_a0=True,
                                a0_fixed=a0_fixed, p_fixed=args.twohalo_slope,
                                use_prior=use_prior, prior_weight=args.prior_weight, profile_type=args.profile_type,
                                R0_kpc=args.R0_kpc, twohalo_prior_weight=args.twohalo_prior_weight))

        # Free a0 with 2halo.
        results.append(fit_case(data_by_bin, bins, "drnd_free_a0_2halo", "drnd",
                                include_2h=True, twohalo_mode=args.twohalo_mode, fit_p=args.fit_twohalo_slope, fit_a0=True,
                                a0_fixed=a0_fixed, p_fixed=args.twohalo_slope,
                                use_prior=use_prior, prior_weight=args.prior_weight, profile_type=args.profile_type,
                                R0_kpc=args.R0_kpc, twohalo_prior_weight=args.twohalo_prior_weight))

    fit_rows = []
    bin_rows = []
    twohalo_rows = []
    for res in results:
        fit_rows.append({k:v for k,v in res.items() if k not in {"params","bin_rows","pred_rows"}})
        bin_rows.extend([{**br, "case": res["case"]} for br in res["bin_rows"]])
        for br in res["bin_rows"]:
            if res["include_2halo"]:
                twohalo_rows.append({
                    "case": res["case"],
                    "lens_bin": br["lens_bin"],
                    "twohalo_A_at_R0": br["twohalo_A_at_R0"],
                    "twohalo_logA": br["twohalo_logA"],
                    "twohalo_slope": br["twohalo_slope"],
                })
        write_csv(tabledir/f"predictions_{res['case']}.csv", res["pred_rows"])

    write_csv(tabledir/"fit_table.csv", fit_rows)
    write_csv(tabledir/"bin_parameters.csv", bin_rows)
    write_csv(tabledir/"twohalo_parameters.csv", twohalo_rows)
    plot_results(outdir, results)

    bary = next(r for r in results if r["case"]=="baryon_only")
    fixed = next(r for r in results if r["case"]=="drnd_fixed_a0")
    fixed2h = next(r for r in results if r["case"]=="drnd_fixed_a0_2halo")
    free_no2h = next((r for r in results if r["case"]=="drnd_free_a0_no2halo"), None)
    free2h = next((r for r in results if r["case"]=="drnd_free_a0_2halo"), None)

    plausible = [br for br in fixed2h["bin_rows"] if br["Mbar_plausible_1e8_1e12p5"] and br["Re_plausible_0p2_80_kpc"]]

    summary = {
        "model": "DRND-ST binned lensing + two-halo autofit",
        "input": {
            "lensing_csv": args.lensing_csv,
            "n_points": len(rows),
            "n_bins": len(bins),
            "bins": bins,
            "bin_mode_requested": args.bin_mode,
            "bin_mode_used": mode_used,
            "H0": args.H0,
            "chi": args.chi,
            "a0_fixed": a0_fixed,
            "profile_type": args.profile_type,
            "use_prior": use_prior,
            "prior_weight": args.prior_weight,
            "twohalo_mode": args.twohalo_mode,
            "twohalo_slope_fixed": args.twohalo_slope,
            "fit_twohalo_slope": args.fit_twohalo_slope,
            "R0_kpc": args.R0_kpc,
            "twohalo_prior_weight": args.twohalo_prior_weight,
        },
        "fit_table": fit_rows,
        "decision": {
            "fixed_drnd_beats_baryon": fixed["chi2"] < bary["chi2"],
            "fixed_2halo_beats_fixed_no2halo": fixed2h["chi2"] < fixed["chi2"],
            "fixed_2halo_chi2_per_point": fixed2h["chi2_per_point"],
            "fixed_no2halo_chi2_per_point": fixed["chi2_per_point"],
            "baryon_chi2_per_point": bary["chi2_per_point"],
            "fixed_2halo_all_bins_plausible": len(plausible) == len(fixed2h["bin_rows"]),
            "free_no2halo_a0_over_fixed": free_no2h["a0_over_fixed"] if free_no2h else None,
            "free_2halo_a0_over_fixed": free2h["a0_over_fixed"] if free2h else None,
            "free_2halo_near_fixed_factor_2": (0.5 <= free2h["a0_over_fixed"] <= 2.0) if free2h else None,
            "a0_runaway_reduced_by_2halo": (free2h["a0_over_fixed"] < free_no2h["a0_over_fixed"]) if free2h and free_no2h else None,
            "median_twohalo_fraction_inner_fixed2h": fixed2h["median_twohalo_fraction_inner"],
            "median_twohalo_fraction_outer_fixed2h": fixed2h["median_twohalo_fraction_outer"],
        },
        "cautions": [
            "Two-halo term is phenomenological and can absorb missing survey physics.",
            "If two-halo dominates inner radii, the interpretation is weak.",
            "Still diagonal errors only; no covariance, HOD, miscentering, boost factors, or selection model.",
            "This test asks whether environment removes a0 runaway, not whether final lensing likelihood is solved."
        ]
    }

    (outdir/"summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")

    print(json.dumps(safe_json({
        "outdir": str(outdir),
        "decision": summary["decision"],
        "files": [
            "summary.json",
            "tables/binned_lensing_data.csv",
            "tables/fit_table.csv",
            "tables/bin_parameters.csv",
            "tables/twohalo_parameters.csv",
            "tables/predictions_baryon_only.csv",
            "tables/predictions_drnd_fixed_a0.csv",
            "tables/predictions_drnd_fixed_a0_2halo.csv",
            "tables/predictions_drnd_free_a0_no2halo.csv",
            "tables/predictions_drnd_free_a0_2halo.csv",
            "plots/binned_2halo_fit.png",
            "plots/a0_comparison.png",
        ]
    }), indent=2))


if __name__ == "__main__":
    main()

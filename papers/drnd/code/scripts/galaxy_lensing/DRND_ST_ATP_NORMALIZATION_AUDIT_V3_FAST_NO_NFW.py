#!/usr/bin/env python3
"""
DRND_ST_ATP_NORMALIZATION_AUDIT.py

Goal
----
Audit the ATP amplitude normalization, with an explicit 4*pi spherical
projection candidate.

Hypothesis tested explicitly
----------------------------
The previous ATP eta result was eta ~= 12.26. Since old ATP had kappa=1/4
and eta = 4*kappa, eta ~= 4*pi corresponds to:

    kappa = pi

Therefore the candidate lensing amplitude is:

    A_lens = pi * sqrt(G M_b a0) / G

This is tested as a fixed, theory-motivated prefactor, not fitted to win.

The previous ATP diagnostic found:
  - ATP radial shape is competitive.
  - The fitted global eta is not near 1.
  - Therefore the urgent question is the prefactor A.

Old ATP amplitude:
    A_old = (1/4) * sqrt(G M_b a0) / G

This script replaces 1/4 by a dimensionless prefactor kappa:

    A(kappa) = kappa * sqrt(G M_b a0) / G

Relation to previous eta:
    eta = kappa / (1/4) = 4 kappa
    kappa = eta / 4

If earlier eta_global ~ 12.26, the implied kappa_eff ~ 3.07.

This script tests:
  1. fixed kappa grid values
  2. fitted global kappa
  3. fitted global kappa + screened environment
  4. fitted global kappa + radial tilt
  5. NFW+baryon global c(M) control

This is diagnostic. kappa is not declared a new physical constant unless it
emerges stably across data choices and bins.

Usage
-----
python DRND_ST_ATP_NORMALIZATION_AUDIT.py ^
  --lensing-csv lensing_data.csv ^
  --bin-mode auto ^
  --n-bins 4 ^
  --outdir DRND_ST_ATP_NORMALIZATION_AUDIT

Fast:
python DRND_ST_ATP_NORMALIZATION_AUDIT.py ^
  --lensing-csv lensing_data.csv ^
  --maxiter 200 ^
  --n-random-starts 1 ^
  --outdir DRND_ST_ATP_NORM_FAST

Custom fixed kappa grid:
python DRND_ST_ATP_NORMALIZATION_AUDIT.py ^
  --lensing-csv lensing_data.csv ^
  --kappa-grid "0.25,0.5,1,2,3,4,6,8,12.566" ^
  --outdir DRND_ST_ATP_NORM_GRID

Outputs
-------
summary.json
tables/model_comparison.csv
tables/bin_parameters.csv
tables/fit_attempts.csv
tables/predictions_<case>.csv
plots/model_comparison_bic.png
plots/model_comparison_aic.png
"""

from __future__ import annotations
import argparse, csv, json, math, re, hashlib
from pathlib import Path
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


def read_lensing_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lb = pick_str(row, ["lens_bin", "bin", "sample"], "")
            R = pick(row, ["R_kpc", "R", "rp", "radius"])
            ds = pick(row, ["DeltaSigma", "delta_sigma", "deltasigma", "ds", "signal"])
            err = pick(row, ["DeltaSigma_err", "delta_sigma_err", "deltasigma_err", "ds_err", "err", "error", "sigma"])
            Mbar = pick(row, ["Mbar_Msun", "Mbar", "mbar", "baryonic_mass"])
            logM = pick(row, ["logMbar", "log_mbar", "logM", "log_mass"])
            if not np.isfinite(logM) and np.isfinite(Mbar) and Mbar > 0:
                logM = math.log10(Mbar)
            if np.isfinite(R) and np.isfinite(ds) and np.isfinite(err) and R > 0 and ds > 0 and err > 0:
                rows.append({
                    "lens_bin_original": lb,
                    "R_kpc": float(R),
                    "DeltaSigma": float(ds),
                    "DeltaSigma_err": float(err),
                    "logMbar_input": float(logM) if np.isfinite(logM) else np.nan,
                })
    if not rows:
        raise SystemExit("No usable rows")
    return rows


def assign_bins(rows, mode, n_bins):
    mode_used = mode
    existing = sorted(set(str(r.get("lens_bin_original", "")).strip() for r in rows if str(r.get("lens_bin_original", "")).strip()))
    if mode == "auto":
        if len(existing) > 1:
            mode_used = "existing"
        elif any(np.isfinite(r.get("logMbar_input", np.nan)) for r in rows):
            mode_used = "mass"
        else:
            mode_used = "signal"

    if mode_used == "existing":
        for r in rows:
            r["lens_bin"] = str(r.get("lens_bin_original", "")).strip() or "all"
        return rows, mode_used

    if mode_used == "mass":
        key = np.array([r.get("logMbar_input", np.nan) for r in rows], float)
        if not np.all(np.isfinite(key)):
            raise SystemExit("mass binning requires logMbar/Mbar")
    elif mode_used == "radius":
        key = np.array([r["R_kpc"] for r in rows], float)
    elif mode_used == "signal":
        key = np.array([r["DeltaSigma"] for r in rows], float)
    else:
        raise SystemExit(f"Unknown bin mode {mode}")

    edges = np.unique(np.quantile(key, np.linspace(0, 1, max(2, n_bins + 1))))
    if len(edges) <= 2:
        for r in rows:
            r["lens_bin"] = "bin0"
        return rows, mode_used

    for r, v in zip(rows, key):
        idx = np.searchsorted(edges, v, side="right") - 1
        idx = max(0, min(idx, len(edges) - 2))
        r["lens_bin"] = f"{mode_used}_bin{idx}"
    return rows, mode_used


def a0_from_chi(H0, chi):
    return chi * C_M_S * (H0 * 1000 / MPC_M)


def rhocrit_msun_kpc3(H0):
    H = H0 * 1000 / MPC_M
    rho_kg_m3 = 3 * H * H / (8 * math.pi * G_SI)
    return rho_kg_m3 * (KPC_M ** 3) / MSUN_KG


def mbar_enclosed(r, Mbar, Re, profile):
    r = np.asarray(r, float)
    M = float(Mbar)
    Re = max(float(Re), 1e-5)
    if profile == "point":
        return np.full_like(r, M)
    if profile == "hernquist":
        a = Re / 1.8153
        return M * r * r / np.maximum((r + a) ** 2, 1e-300)
    x = r / Re
    return M * (1 - np.exp(-x) * (1 + x + 0.5 * x * x))


def density_from_enclosed(r_kpc, M_msun):
    rm = r_kpc * KPC_M
    Mkg = M_msun * MSUN_KG
    dMdr = np.gradient(Mkg, rm)
    return np.maximum(dMdr / (4 * math.pi * np.maximum(rm * rm, 1e-300)), 0)


def sigma_projected(R, rgrid, rho):
    Rm = R * KPC_M
    rm = rgrid * KPC_M
    mask = rm > Rm * (1 + 1e-8)
    if np.sum(mask) < 10:
        return np.nan
    rr = rm[mask]
    rh = rho[mask]
    integ = rh * rr / np.sqrt(np.maximum(rr * rr - Rm * Rm, 1e-300))
    val = 2 * (np.trapezoid(integ, rr) if hasattr(np, "trapezoid") else np.trapz(integ, rr))
    return val * PC_M ** 2 / MSUN_KG


def delta_sigma_from_density(R_eval, rgrid, rho):
    R_eval = np.asarray(R_eval, float)
    Sigma = np.array([sigma_projected(R, rgrid, rho) for R in R_eval])
    Rd = np.geomspace(max(np.min(R_eval) / 50, 1e-4), np.max(R_eval) * 1.05, 220)
    Sd = np.array([sigma_projected(R, rgrid, rho) for R in Rd])
    ok = np.isfinite(Sd)
    Rd = Rd[ok]
    Sd = Sd[ok]
    if len(Rd) < 20:
        return np.full_like(R_eval, np.nan)
    integ = Sd * Rd
    cum = np.zeros_like(Rd)
    for i in range(1, len(Rd)):
        cum[i] = cum[i - 1] + 0.5 * (integ[i] + integ[i - 1]) * (Rd[i] - Rd[i - 1])
    Sbar = 2 * cum / np.maximum(Rd * Rd, 1e-300)
    return np.interp(R_eval, Rd, Sbar) - Sigma


def baryon_ds(R, logM, logRe, profile):
    rgrid = np.geomspace(max(np.min(R) / 200, 1e-4), max(np.max(R) * 200, 1e4), 600)
    rho = density_from_enclosed(rgrid, mbar_enclosed(rgrid, 10 ** logM, 10 ** logRe, profile))
    return delta_sigma_from_density(R, rgrid, rho)


def atp_ds_kappa(R_kpc, logMbar, a0, kappa):
    """
    ATP memory component with free/fixed prefactor kappa.

    A(kappa) = kappa * sqrt(G M_b a0) / G

    Old formula had kappa = 1/4.
    """
    R = np.asarray(R_kpc, dtype=float) * KPC_M
    M_b = (10 ** float(logMbar)) * MSUN_KG
    if M_b <= 0:
        return np.zeros_like(R)

    r0 = math.sqrt((G_SI * M_b) / a0)
    A = float(kappa) * math.sqrt(G_SI * M_b * a0) / G_SI

    R2 = R * R
    root = np.sqrt(R2 + r0 * r0)
    R2_safe = np.maximum(R2, 1e-20)
    sigma_bar = (2.0 * A / R2_safe) * (root - r0)
    sigma = A / root
    ds_kg_m2 = sigma_bar - sigma
    ds_kg_m2 = np.where(R2 < 1e-16, 0.0, ds_kg_m2)
    return ds_kg_m2 * (PC_M ** 2) / MSUN_KG


def screen(R, Rt, q):
    R = np.asarray(R, float)
    return 1 / (1 + np.power(np.maximum(Rt / np.maximum(R, 1e-12), 1e-12), q))


def twohalo(R, logA, p, R0, Rt, q):
    return 10 ** logA * np.power(np.maximum(np.asarray(R, float) / R0, 1e-12), -p) * screen(R, Rt, q)


def c_relation_log10(logM200):
    return math.log10(8.0) - 0.1 * (logM200 - 12.0)


def nfw_delta_sigma_analytic(R_kpc, logM200, logc, H0):
    R = np.asarray(R_kpc, float)
    c = 10 ** logc
    M200 = 10 ** logM200
    rho_c = rhocrit_msun_kpc3(H0)
    r200 = (3 * M200 / (4 * math.pi * 200 * rho_c)) ** (1 / 3)
    rs = r200 / c
    fc = math.log(1 + c) - c / (1 + c)
    rho_s = M200 / (4 * math.pi * rs ** 3 * fc)
    x = np.maximum(R / rs, 1e-8)

    f = np.zeros_like(x)
    g = np.zeros_like(x)
    lt = x < 1 - 1e-6
    gt = x > 1 + 1e-6
    eq = ~(lt | gt)

    if np.any(lt):
        xl = x[lt]
        y = np.sqrt((1 - xl) / (1 + xl))
        ay = np.arctanh(np.clip(y, 0, 1 - 1e-14))
        f[lt] = (1 - 2 / np.sqrt(1 - xl * xl) * ay) / (xl * xl - 1)
        g[lt] = np.log(xl / 2) + 2 / np.sqrt(1 - xl * xl) * ay

    if np.any(gt):
        xg = x[gt]
        y = np.sqrt((xg - 1) / (1 + xg))
        ay = np.arctan(y)
        f[gt] = (1 - 2 / np.sqrt(xg * xg - 1) * ay) / (xg * xg - 1)
        g[gt] = np.log(xg / 2) + 2 / np.sqrt(xg * xg - 1) * ay

    if np.any(eq):
        f[eq] = 1 / 3
        g[eq] = 1 + math.log(0.5)

    sigma = 2 * rho_s * rs * f
    sigma_bar = 4 * rho_s * rs * g / np.maximum(x * x, 1e-30)
    return (sigma_bar - sigma) / 1e6


def mass_size_prior(logM, logRe):
    pred = math.log10(5.0) + 0.25 * (logM - 10)
    return ((logRe - pred) / 0.35) ** 2


def parse_kappa_grid(s):
    vals = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def read_nfw_baseline_json(path):
    """
    Read a previous summary.json and extract an NFW-control row if present.
    This lets new ATP/kappa scans avoid refitting NFW repeatedly.

    Accepted sources:
      - decision.NFW_control_BIC
      - model_comparison row with case == nfw_plus_baryon_global_relation
    """
    if not path:
        return None
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    rows = obj.get("model_comparison", [])
    for r in rows:
        if r.get("case") == "nfw_plus_baryon_global_relation":
            return dict(r)
    dec = obj.get("decision", {})
    bic = dec.get("NFW_control_BIC") or dec.get("NFW_plus_baryon_global_relation_BIC")
    if bic is not None:
        return {
            "case": "nfw_plus_baryon_global_relation",
            "chi2": dec.get("NFW_control_chi2"),
            "n_data": obj.get("input", {}).get("n_points"),
            "k_params": dec.get("NFW_control_k_params"),
            "dof": dec.get("NFW_control_dof"),
            "chi2_per_dof": dec.get("NFW_control_chi2_per_dof"),
            "chi2_per_data": None,
            "AIC": dec.get("NFW_control_AIC"),
            "BIC": bic,
            "success": True,
            "message": f"loaded from baseline json: {p}",
            "best_start_id": "",
            "n_starts": 0,
            "invalid_prediction_fraction": 0.0,
            "loaded_baseline": True,
        }
    raise SystemExit(f"Could not find NFW baseline in {p}")


def predict_case(case, params, bins, data, args, a0, fixed_kappa=None):
    idx = 0
    preds, rows, bin_rows, priors = [], [], [], []
    rpiv = args.Rpivot_kpc

    log_kappa = None
    alpha = None
    if case in {"atp_kappa_global", "atp_kappa_global_screen2h", "atp_kappa_tilt_global"}:
        log_kappa = params[idx]
        idx += 1
    if case == "atp_kappa_tilt_global":
        alpha = params[idx]
        idx += 1

    for lb in bins:
        sub = data[lb]
        R = np.array([r["R_kpc"] for r in sub], float)

        if case == "baryon_only":
            logM, logRe = params[idx], params[idx + 1]; idx += 2
            pred = baryon_ds(R, logM, logRe, args.profile_type)
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe}

        elif case == "atp_kappa_fixed_plus_baryon":
            logM, logRe = params[idx], params[idx + 1]; idx += 2
            kappa = fixed_kappa
            pred = baryon_ds(R, logM, logRe, args.profile_type) + atp_ds_kappa(R, logM, a0, kappa)
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe, "kappa_atp": kappa, "eta_equiv": 4 * kappa}

        elif case == "atp_kappa_global":
            logM, logRe = params[idx], params[idx + 1]; idx += 2
            kappa = 10 ** log_kappa
            pred = baryon_ds(R, logM, logRe, args.profile_type) + atp_ds_kappa(R, logM, a0, kappa)
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe, "kappa_atp": kappa, "eta_equiv": 4 * kappa}

        elif case == "atp_kappa_global_screen2h":
            logM, logRe, logA = params[idx], params[idx + 1], params[idx + 2]; idx += 3
            kappa = 10 ** log_kappa
            pred = baryon_ds(R, logM, logRe, args.profile_type) + atp_ds_kappa(R, logM, a0, kappa) + twohalo(R, logA, args.twohalo_slope, args.R0_kpc, args.Rt_kpc, args.screen_q)
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe, "kappa_atp": kappa, "eta_equiv": 4 * kappa, "twohalo_logA": logA}

        elif case == "atp_kappa_tilt_global":
            logM, logRe = params[idx], params[idx + 1]; idx += 2
            kappa = 10 ** log_kappa
            pred = (baryon_ds(R, logM, logRe, args.profile_type) +
                    atp_ds_kappa(R, logM, a0, kappa) * np.power(np.maximum(R / rpiv, 1e-12), alpha))
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe, "kappa_atp": kappa, "eta_equiv": 4 * kappa, "alpha_lens": alpha}

        elif case == "nfw_plus_baryon_global_relation":
            logM, logRe, logM200 = params[idx], params[idx + 1], params[idx + 2]; idx += 3
            logc = c_relation_log10(logM200)
            pred = baryon_ds(R, logM, logRe, args.profile_type) + nfw_delta_sigma_analytic(R, logM200, logc, args.H0)
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe, "logM200": logM200, "M200_Msun": 10 ** logM200, "c200_relation": 10 ** logc}

        else:
            raise ValueError(case)

        bin_rows.append({"case": case, "lens_bin": lb, **meta})
        for i, r in enumerate(sub):
            p = float(pred[i]) if np.isfinite(pred[i]) else np.nan
            preds.append(p)
            rows.append({
                "case": case, "lens_bin": lb,
                "R_kpc": r["R_kpc"], "DeltaSigma_obs": r["DeltaSigma"],
                "DeltaSigma_err": r["DeltaSigma_err"], "DeltaSigma_pred": p,
                **meta
            })

    return np.array(preds, float), rows, bin_rows, np.array(priors, float)


def residuals(case, params, bins, data, args, a0, fixed_kappa=None):
    pred, _, _, priors = predict_case(case, params, bins, data, args, a0, fixed_kappa=fixed_kappa)
    obs = np.array([r["DeltaSigma"] for lb in bins for r in data[lb]], float)
    err = np.array([r["DeltaSigma_err"] for lb in bins for r in data[lb]], float)
    bad = np.where((~np.isfinite(pred)) | (pred <= 0), 1e4, 0.0)
    if args.cov_space == "linear":
        res = (obs - pred) / np.maximum(err, 1e-30) + bad
    else:
        sig = np.maximum(err / np.maximum(obs, 1e-30) / math.log(10), args.min_log_sigma)
        res = (np.log10(np.maximum(obs, 1e-30)) - np.log10(np.maximum(pred, 1e-30))) / sig + bad
    return res if args.no_prior else np.concatenate([res, priors])


def base_initial_vector(case, bins, args):
    p0, bounds = [], []
    if case in {"atp_kappa_global", "atp_kappa_global_screen2h", "atp_kappa_tilt_global"}:
        p0.append(math.log10(3.0))
        bounds.append((math.log10(args.kappa_min), math.log10(args.kappa_max)))
    if case == "atp_kappa_tilt_global":
        p0.append(0.0)
        bounds.append((args.alpha_min, args.alpha_max))

    for _ in bins:
        lm = 11.0
        lr = math.log10(5 * (10 ** (lm - 10)) ** 0.25)
        if case in {"baryon_only", "atp_kappa_fixed_plus_baryon", "atp_kappa_global", "atp_kappa_tilt_global"}:
            p0 += [lm, lr]
            bounds += [(8, 12.8), (math.log10(0.2), math.log10(80))]
        elif case == "atp_kappa_global_screen2h":
            p0 += [lm, lr, 0.0]
            bounds += [(8, 12.8), (math.log10(0.2), math.log10(80)), (-8, 6)]
        elif case == "nfw_plus_baryon_global_relation":
            p0 += [lm, lr, 12.0]
            bounds += [(8, 12.8), (math.log10(0.2), math.log10(80)), (9, 15)]
        else:
            raise ValueError(case)
    return np.array(p0, float), bounds


def generate_starts(case, bins, args):
    p0, bounds = base_initial_vector(case, bins, args)
    starts = [p0.copy()]

    def clip(v):
        out = np.array(v, float)
        for i, (lo, hi) in enumerate(bounds):
            out[i] = min(max(out[i], lo + 1e-6), hi - 1e-6)
        return out

    if "kappa_global" in case:
        for kappa in [0.25, 0.5, 1.0, 2.0, 3.0, math.pi, 4.0, 2 * math.pi, 4 * math.pi]:
            v = p0.copy()
            v[0] = math.log10(min(max(kappa, args.kappa_min), args.kappa_max))
            starts.append(clip(v))

    if case == "atp_kappa_tilt_global":
        for a in [-0.75, -0.25, 0.0, 0.25, 0.75]:
            v = p0.copy()
            v[1] = a
            starts.append(clip(v))

    for lm in [9.5, 10.5, 11.2, 12.0]:
        v = p0.copy()
        for i, (lo, hi) in enumerate(bounds):
            if lo <= 8.1 and hi >= 12.7:
                v[i] = lm
            elif lo <= math.log10(0.21) and hi >= math.log10(79):
                v[i] = math.log10(5 * (10 ** (lm - 10)) ** 0.25)
        starts.append(clip(v))

    if case == "nfw_plus_baryon_global_relation":
        for hm in [11.0, 12.0, 13.0, 14.0]:
            v = p0.copy()
            for i, (lo, hi) in enumerate(bounds):
                if lo <= 9.1 and hi >= 14.9:
                    v[i] = hm
            starts.append(clip(v))

    seed = int(hashlib.md5(f"{case}|{len(bins)}|{args.random_seed}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    for _ in range(args.n_random_starts):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds], float))

    seen, unique = set(), []
    for s in starts:
        key = tuple(np.round(s, 6))
        if key not in seen:
            seen.add(key); unique.append(s)
    return unique, bounds


def fit_case(case, bins, data, args, a0, fixed_kappa=None):
    starts, bounds = generate_starts(case, bins, args)
    attempts, best = [], None

    def obj(p):
        r = residuals(case, p, bins, data, args, a0, fixed_kappa=fixed_kappa)
        return float(np.sum(r * r))

    for si, start in enumerate(starts):
        try:
            opt = minimize(obj, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": args.maxiter, "maxfun": args.maxfun, "ftol": 1e-8})
            chi2 = float(opt.fun)
            attempts.append({"case": case if fixed_kappa is None else f"{case}_kappa_{fixed_kappa:g}", "start_id": si, "chi2": chi2, "success": bool(opt.success), "message": str(opt.message)})
            if best is None or chi2 < best["chi2"]:
                best = {"opt": opt, "chi2": chi2, "start_id": si}
        except Exception as e:
            attempts.append({"case": case if fixed_kappa is None else f"{case}_kappa_{fixed_kappa:g}", "start_id": si, "chi2": 1e300, "success": False, "message": f"exception: {e}"})

    if best is None:
        raise RuntimeError(f"All starts failed for {case}")

    opt = best["opt"]
    pred, pred_rows, bin_rows, _ = predict_case(case, opt.x, bins, data, args, a0, fixed_kappa=fixed_kappa)
    res = residuals(case, opt.x, bins, data, args, a0, fixed_kappa=fixed_kappa)
    chi2 = float(np.sum(res * res))
    n = sum(len(data[lb]) for lb in bins)
    k = len(opt.x)
    dof = max(n - k, 1)
    invalid_frac = float(np.mean((~np.isfinite(pred)) | (pred <= 0)))

    case_name = case if fixed_kappa is None else f"atp_kappa_fixed_{fixed_kappa:g}_plus_baryon"
    for row in pred_rows:
        row["case"] = case_name
    for row in bin_rows:
        row["case"] = case_name

    return {
        "case": case_name, "chi2": chi2, "n_data": n, "k_params": k, "dof": dof,
        "chi2_per_dof": chi2 / dof, "chi2_per_data": chi2 / max(n, 1),
        "AIC": chi2 + 2 * k, "BIC": chi2 + k * math.log(max(n, 2)),
        "success": bool(opt.success), "message": str(opt.message),
        "best_start_id": best["start_id"], "n_starts": len(starts),
        "invalid_prediction_fraction": invalid_frac,
        "pred_rows": pred_rows, "bin_rows": bin_rows, "attempts": attempts,
    }


def summarize_params(bin_rows):
    kappas = [r.get("kappa_atp") for r in bin_rows if r.get("kappa_atp") is not None]
    etas = [r.get("eta_equiv") for r in bin_rows if r.get("eta_equiv") is not None]
    alphas = [r.get("alpha_lens") for r in bin_rows if r.get("alpha_lens") is not None]
    out = {}
    if kappas:
        arr = np.array(kappas, float)
        out.update({"kappa_median": float(np.median(arr)), "kappa_min": float(np.min(arr)), "kappa_max": float(np.max(arr))})
    if etas:
        arr = np.array(etas, float)
        out.update({"eta_equiv_median": float(np.median(arr)), "eta_equiv_min": float(np.min(arr)), "eta_equiv_max": float(np.max(arr))})
    if alphas:
        arr = np.array(alphas, float)
        out.update({"alpha_median": float(np.median(arr)), "alpha_min": float(np.min(arr)), "alpha_max": float(np.max(arr))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lensing-csv", required=True)
    ap.add_argument("--cov-space", default="log", choices=["log", "linear"])
    ap.add_argument("--bin-mode", default="auto", choices=["auto", "existing", "radius", "signal", "mass"])
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--H0", type=float, default=67.4)
    ap.add_argument("--chi", type=float, default=0.18)
    ap.add_argument("--profile-type", default="hernquist", choices=["hernquist", "exponential_sphere", "point"])
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--twohalo-slope", type=float, default=0.8)
    ap.add_argument("--R0-kpc", type=float, default=300.0)
    ap.add_argument("--Rt-kpc", type=float, default=200.0)
    ap.add_argument("--screen-q", type=float, default=3.0)
    ap.add_argument("--Rpivot-kpc", type=float, default=100.0)
    ap.add_argument("--kappa-grid", default="0.25,0.5,1,2,2.5,3,3.141592653589793,3.25,3.5,4,6.283185307179586,12.566370614359172")
    ap.add_argument("--kappa-min", type=float, default=0.01)
    ap.add_argument("--kappa-max", type=float, default=100.0)
    ap.add_argument("--alpha-min", type=float, default=-3.0)
    ap.add_argument("--alpha-max", type=float, default=3.0)
    ap.add_argument("--min-log-sigma", type=float, default=0.03)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--maxfun", type=int, default=800)
    ap.add_argument("--n-random-starts", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=12345)
    ap.add_argument("--skip-nfw", action="store_true", help="Skip refitting NFW control for fast repeated ATP/kappa scans.")
    ap.add_argument("--nfw-baseline-json", default=None, help="Optional previous summary.json containing NFW baseline row/BIC.")
    ap.add_argument("--outdir", default="DRND_ST_ATP_NORMALIZATION_AUDIT_V3_FAST_NO_NFW")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tabledir = outdir / "tables"; plotdir = outdir / "plots"
    tabledir.mkdir(parents=True, exist_ok=True); plotdir.mkdir(parents=True, exist_ok=True)

    rows = read_lensing_csv(args.lensing_csv)
    rows, mode_used = assign_bins(rows, args.bin_mode, args.n_bins)
    write_csv(tabledir / "binned_lensing_data.csv", rows, preferred=["lens_bin", "R_kpc", "DeltaSigma", "DeltaSigma_err", "logMbar_input"])
    data = {}
    for r in rows:
        data.setdefault(r["lens_bin"], []).append(r)
    bins = sorted(data.keys())
    a0 = a0_from_chi(args.H0, args.chi)

    jobs = [("baryon_only", None)]
    for kappa in parse_kappa_grid(args.kappa_grid):
        jobs.append(("atp_kappa_fixed_plus_baryon", kappa))
    jobs += [
        ("atp_kappa_global", None),
        ("atp_kappa_global_screen2h", None),
        ("atp_kappa_tilt_global", None),
    ]

    loaded_nfw_baseline = read_nfw_baseline_json(args.nfw_baseline_json) if args.nfw_baseline_json else None
    if not args.skip_nfw and loaded_nfw_baseline is None:
        jobs.append(("nfw_plus_baryon_global_relation", None))

    results, all_bin_rows, all_attempts, param_summary = [], [], [], {}
    for case, fixed_kappa in jobs:
        label = case if fixed_kappa is None else f"{case}_kappa_{fixed_kappa:g}"
        print("[fit]", label, flush=True)
        fit = fit_case(case, bins, data, args, a0, fixed_kappa=fixed_kappa)
        write_csv(tabledir / f"predictions_{fit['case']}.csv", fit["pred_rows"])
        write_csv(tabledir / f"bin_parameters_{fit['case']}.csv", fit["bin_rows"])
        all_bin_rows += fit["bin_rows"]
        all_attempts += fit["attempts"]
        param_summary[fit["case"]] = summarize_params(fit["bin_rows"])
        results.append({k: v for k, v in fit.items() if k not in {"pred_rows", "bin_rows", "attempts"}})

    if loaded_nfw_baseline is not None:
        results.append(loaded_nfw_baseline)

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
        "diag_kappa_median", "diag_kappa_min", "diag_kappa_max",
        "diag_eta_equiv_median", "diag_eta_equiv_min", "diag_eta_equiv_max",
        "diag_alpha_median", "diag_alpha_min", "diag_alpha_max",
        "success", "best_start_id", "n_starts", "invalid_prediction_fraction"
    ])

    for metric in ["BIC", "AIC"]:
        ordered = sorted(comp, key=lambda r: r[metric])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(ordered)), [r[metric] for r in ordered])
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels([r["case"] for r in ordered], rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"ATP normalization audit by {metric}")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(plotdir / f"model_comparison_{metric.lower()}.png", dpi=180)
        plt.close(fig)

    by = {r["case"]: r for r in comp}
    nfw = by.get("nfw_plus_baryon_global_relation")
    kg = by.get("atp_kappa_global")
    kgs = by.get("atp_kappa_global_screen2h")
    kt = by.get("atp_kappa_tilt_global")
    old = by.get("atp_kappa_fixed_0.25_plus_baryon")
    kappa_pi = by.get("atp_kappa_fixed_3.14159_plus_baryon") or by.get("atp_kappa_fixed_3.141592653589793_plus_baryon") or by.get("atp_kappa_fixed_3.1415926535_plus_baryon")
    kappa_3 = by.get("atp_kappa_fixed_3_plus_baryon")
    kappa_325 = by.get("atp_kappa_fixed_3.25_plus_baryon")
    kappa_4 = by.get("atp_kappa_fixed_4_plus_baryon")

    fixed_rows = [r for r in comp if r["case"].startswith("atp_kappa_fixed_")]
    best_fixed = min(fixed_rows, key=lambda r: r["BIC"]) if fixed_rows else None

    def bd(a, b):
        return a["BIC"] - b["BIC"] if a and b else None

    decision = {
        "best_model_by_BIC": best_bic["case"],
        "best_model_by_AIC": best_aic["case"],
        "NFW_control_BIC": nfw["BIC"] if nfw else None,
        "old_kappa_0p25_BIC": old["BIC"] if old else None,
        "kappa_pi_4pi_projection_BIC": kappa_pi["BIC"] if kappa_pi else None,
        "kappa_3_BIC": kappa_3["BIC"] if kappa_3 else None,
        "kappa_3p25_BIC": kappa_325["BIC"] if kappa_325 else None,
        "kappa_4_BIC": kappa_4["BIC"] if kappa_4 else None,
        "best_fixed_kappa_model": best_fixed["case"] if best_fixed else None,
        "best_fixed_kappa_BIC": best_fixed["BIC"] if best_fixed else None,
        "kappa_global_BIC": kg["BIC"] if kg else None,
        "kappa_global_screen2h_BIC": kgs["BIC"] if kgs else None,
        "kappa_tilt_global_BIC": kt["BIC"] if kt else None,
        "delta_BIC_kappaPi4piProjection_minus_NFW": bd(kappa_pi, nfw),
        "delta_BIC_kappaPi4piProjection_minus_bestFixed": bd(kappa_pi, best_fixed),
        "delta_BIC_bestFixedKappa_minus_NFW": bd(best_fixed, nfw),
        "delta_BIC_kappaGlobal_minus_NFW": bd(kg, nfw),
        "delta_BIC_kappaTilt_minus_NFW": bd(kt, nfw),
        "kappa_global_summary": param_summary.get("atp_kappa_global", {}),
        "kappa_screened_summary": param_summary.get("atp_kappa_global_screen2h", {}),
        "kappa_tilt_summary": param_summary.get("atp_kappa_tilt_global", {}),
        "old_kappa_relation": "old ATP corresponds to kappa=0.25 and eta_equiv=1",
        "nfw_mode": "loaded_from_json" if loaded_nfw_baseline is not None else ("skipped" if args.skip_nfw else "refit"),
    }

    kg_med = decision["kappa_global_summary"].get("kappa_median")
    alpha_med = decision["kappa_tilt_summary"].get("alpha_median")
    decision["diagnostic_flags"] = {
        "old_kappa_0p25_competes_with_NFW": bool(old and nfw and old["BIC"] <= nfw["BIC"] + 6),
        "kappa_pi_4pi_projection_competes_with_NFW": bool(kappa_pi and nfw and kappa_pi["BIC"] <= nfw["BIC"] + 6),
        "kappa_pi_4pi_projection_near_best_fixed": bool(kappa_pi and best_fixed and abs(kappa_pi["BIC"] - best_fixed["BIC"]) <= 2),
        "best_fixed_kappa_competes_with_NFW": bool(best_fixed and nfw and best_fixed["BIC"] <= nfw["BIC"] + 6),
        "global_kappa_competes_with_NFW": bool(kg and nfw and kg["BIC"] <= nfw["BIC"] + 6),
        "global_kappa_near_old_0p25": bool(kg_med is not None and 0.125 <= kg_med <= 0.5),
        "global_kappa_order_unity_or_few": bool(kg_med is not None and 0.5 <= kg_med <= 10.0),
        "tilt_near_zero": bool(alpha_med is not None and abs(alpha_med) <= 0.3),
    }

    summary = {
        "model": "DRND-ST ATP normalization/prefactor audit V3 fast/no-NFW option",
        "input": vars(args) | {"n_points": len(rows), "bins": bins, "bin_mode_used": mode_used, "a0_fixed": a0},
        "model_comparison": comp,
        "parameter_summary": param_summary,
        "decision": decision,
        "cautions": [
            "kappa=pi is the explicit fixed candidate corresponding to eta=4*pi relative to the old kappa=1/4 ATP normalization.",
            "kappa is diagnostic until derived from DRND graph/topological geometry.",
            "A lower BIC for fitted kappa indicates the old 1/4 prefactor is incomplete, not that kappa is yet fundamental.",
            "Use --skip-nfw for fast repeated scans; use --nfw-baseline-json when you want BIC deltas without refitting NFW.",
            "No full covariance/HOD/miscentering/satellite model is included.",
        ],
    }

    (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
    print(json.dumps(safe_json({
        "outdir": str(outdir),
        "decision": decision,
        "files": [
            "summary.json", "tables/model_comparison.csv", "tables/fit_attempts.csv",
            "tables/bin_parameters.csv", "plots/model_comparison_bic.png", "plots/model_comparison_aic.png"
        ],
    }), indent=2))


if __name__ == "__main__":
    main()

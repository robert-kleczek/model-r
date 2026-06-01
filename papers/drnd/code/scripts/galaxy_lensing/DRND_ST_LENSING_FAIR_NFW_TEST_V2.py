#!/usr/bin/env python3
"""
DRND_ST_LENSING_FAIR_NFW_TEST_V2.py

Objective correction of the previous fair NFW test.

What was wrong before
---------------------
The previous fair script could let DRND fall into a pathological numerical
solution, producing chi2 ~ 1e8 even though the earlier baseline script found
DRND fixed-a0 chi2 ~ O(20) on the same data. That is not a physical model
comparison; it is an optimizer / initialization failure.

What this version changes
-------------------------
1. Uses the same objective style for all models.
2. Uses multi-start optimization for ALL models, not only DRND.
3. Tracks invalid predictions and optimizer failures.
4. Reports sanity flags instead of hiding failures.
5. Does not force DRND to win: NFW variants get the same multi-start treatment.
6. Keeps the same model comparison metrics: chi2, AIC, BIC.

Models
------
baryon_only
drnd_fixed_a0
drnd_fixed_a0_screen2h
nfw_only_perbin
nfw_plus_baryon_perbin
nfw_global_c_only
nfw_global_relation
nfw_plus_baryon_global_c
nfw_plus_baryon_global_relation

Usage
-----
python DRND_ST_LENSING_FAIR_NFW_TEST_V2.py ^
  --lensing-csv lensing_data.csv ^
  --bin-mode auto ^
  --n-bins 4 ^
  --outdir DRND_ST_FAIR_NFW_TEST_V2

More robust:
python DRND_ST_LENSING_FAIR_NFW_TEST_V2.py ^
  --lensing-csv lensing_data.csv ^
  --bin-mode auto ^
  --n-bins 4 ^
  --n-random-starts 24 ^
  --outdir DRND_ST_FAIR_NFW_TEST_V2_ROBUST

Outputs
-------
summary.json
tables/model_comparison.csv
tables/bin_parameters.csv
tables/fit_attempts.csv
tables/predictions_<case>.csv
plots/model_comparison_bic.png
plots/model_comparison_aic.png

Caveat
------
Still diagnostic: no full covariance, no HOD, no survey selection, no
miscentering, no satellites. This is a fair numerical control test, not a final
publication likelihood.
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


def rhocrit(H0):
    H = H0 * 1000 / MPC_M
    return 3 * H * H / (8 * math.pi * G_SI)


def nu_simple(x):
    x = np.maximum(np.asarray(x, float), 1e-300)
    return 0.5 + np.sqrt(0.25 + 1 / x)


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
    Rd = np.geomspace(max(np.min(R_eval) / 50, 1e-4), np.max(R_eval) * 1.05, 320)
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
    rgrid = np.geomspace(max(np.min(R) / 200, 1e-4), max(np.max(R) * 200, 1e4), 900)
    rho = density_from_enclosed(rgrid, mbar_enclosed(rgrid, 10 ** logM, 10 ** logRe, profile))
    return delta_sigma_from_density(R, rgrid, rho)


def drnd_ds(R, logM, logRe, a0, profile):
    rgrid = np.geomspace(max(np.min(R) / 200, 1e-4), max(np.max(R) * 200, 1e4), 900)
    Menc = mbar_enclosed(rgrid, 10 ** logM, 10 ** logRe, profile)
    rm = rgrid * KPC_M
    gbar = G_SI * (Menc * MSUN_KG) / np.maximum(rm * rm, 1e-300)
    gst = gbar * nu_simple(gbar / a0)
    Meff = np.maximum.accumulate(gst * rm * rm / G_SI / MSUN_KG)
    rho = density_from_enclosed(rgrid, Meff)
    return delta_sigma_from_density(R, rgrid, rho)


def screen(R, Rt, q):
    R = np.asarray(R, float)
    return 1 / (1 + np.power(np.maximum(Rt / np.maximum(R, 1e-12), 1e-12), q))


def twohalo(R, logA, p, R0, Rt, q):
    return 10 ** logA * np.power(np.maximum(np.asarray(R, float) / R0, 1e-12), -p) * screen(R, Rt, q)


def nfw_r200(logM200, H0):
    Mkg = 10 ** logM200 * MSUN_KG
    r = (3 * Mkg / (4 * math.pi * 200 * rhocrit(H0))) ** (1 / 3)
    return r / KPC_M


def c_relation_log10(logM200):
    return math.log10(8.0) - 0.1 * (logM200 - 12.0)


def nfw_ds(R, logM200, logc, H0):
    c = 10 ** logc
    r200 = nfw_r200(logM200, H0)
    rs = r200 / c
    M = 10 ** logM200
    fc = math.log(1 + c) - c / (1 + c)
    rho_s = M / (4 * math.pi * rs ** 3 * fc)
    rgrid = np.geomspace(max(np.min(R) / 300, 1e-4), max(r200 * 50, np.max(R) * 300, 1e5), 950)
    x = rgrid / rs
    rho_msun_kpc3 = rho_s / (np.maximum(x, 1e-300) * (1 + x) ** 2)
    rho = rho_msun_kpc3 * MSUN_KG / (KPC_M ** 3)
    return delta_sigma_from_density(R, rgrid, rho)


def mass_size_prior(logM, logRe):
    pred = math.log10(5.0) + 0.25 * (logM - 10)
    return ((logRe - pred) / 0.35) ** 2


def c_prior(logc, logM200):
    pred = c_relation_log10(logM200)
    return ((logc - pred) / 0.25) ** 2


def predict_case(case, params, bins, data, args, a0):
    idx = 0
    preds, rows, bin_rows, priors = [], [], [], []
    global_logc = None
    if case in {"nfw_global_c_only", "nfw_plus_baryon_global_c"}:
        global_logc = params[idx]
        idx += 1

    for lb in bins:
        sub = data[lb]
        R = np.array([r["R_kpc"] for r in sub], float)

        if case == "baryon_only":
            logM, logRe = params[idx], params[idx + 1]
            idx += 2
            pred = baryon_ds(R, logM, logRe, args.profile_type)
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe}

        elif case == "drnd_fixed_a0":
            logM, logRe = params[idx], params[idx + 1]
            idx += 2
            pred = drnd_ds(R, logM, logRe, a0, args.profile_type)
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe}

        elif case == "drnd_fixed_a0_screen2h":
            logM, logRe, logA = params[idx], params[idx + 1], params[idx + 2]
            idx += 3
            base = drnd_ds(R, logM, logRe, a0, args.profile_type)
            th = twohalo(R, logA, args.twohalo_slope, args.R0_kpc, args.Rt_kpc, args.screen_q)
            pred = base + th
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            frac = th / np.maximum(pred, 1e-30)
            inner = frac[R <= np.quantile(R, args.inner_quantile)]
            med = float(np.median(inner)) if len(inner) else 0.0
            if med > args.f_inner_max:
                priors.append(math.sqrt(args.inner_penalty_weight) * (med - args.f_inner_max) / 0.1)
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe, "twohalo_logA": logA, "inner_twohalo_fraction": med}

        elif case == "nfw_only_perbin":
            logM200, logc = params[idx], params[idx + 1]
            idx += 2
            pred = nfw_ds(R, logM200, logc, args.H0)
            priors.append(math.sqrt(args.c_prior_weight * c_prior(logc, logM200)))
            meta = {"logM200": logM200, "M200_Msun": 10 ** logM200, "c200": 10 ** logc}

        elif case == "nfw_plus_baryon_perbin":
            logM, logRe, logM200, logc = params[idx], params[idx + 1], params[idx + 2], params[idx + 3]
            idx += 4
            pred = baryon_ds(R, logM, logRe, args.profile_type) + nfw_ds(R, logM200, logc, args.H0)
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            priors.append(math.sqrt(args.c_prior_weight * c_prior(logc, logM200)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe, "logM200": logM200, "M200_Msun": 10 ** logM200, "c200": 10 ** logc}

        elif case == "nfw_global_c_only":
            logM200 = params[idx]
            idx += 1
            logc = global_logc
            pred = nfw_ds(R, logM200, logc, args.H0)
            priors.append(math.sqrt(args.c_prior_weight * c_prior(logc, logM200)))
            meta = {"logM200": logM200, "M200_Msun": 10 ** logM200, "c200": 10 ** logc}

        elif case == "nfw_global_relation":
            logM200 = params[idx]
            idx += 1
            logc = c_relation_log10(logM200)
            pred = nfw_ds(R, logM200, logc, args.H0)
            meta = {"logM200": logM200, "M200_Msun": 10 ** logM200, "c200_relation": 10 ** logc}

        elif case == "nfw_plus_baryon_global_c":
            logM, logRe, logM200 = params[idx], params[idx + 1], params[idx + 2]
            idx += 3
            logc = global_logc
            pred = baryon_ds(R, logM, logRe, args.profile_type) + nfw_ds(R, logM200, logc, args.H0)
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            priors.append(math.sqrt(args.c_prior_weight * c_prior(logc, logM200)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe, "logM200": logM200, "M200_Msun": 10 ** logM200, "c200": 10 ** logc}

        elif case == "nfw_plus_baryon_global_relation":
            logM, logRe, logM200 = params[idx], params[idx + 1], params[idx + 2]
            idx += 3
            logc = c_relation_log10(logM200)
            pred = baryon_ds(R, logM, logRe, args.profile_type) + nfw_ds(R, logM200, logc, args.H0)
            priors.append(math.sqrt(args.prior_weight * mass_size_prior(logM, logRe)))
            meta = {"logMbar": logM, "Mbar_Msun": 10 ** logM, "Re_kpc": 10 ** logRe, "logM200": logM200, "M200_Msun": 10 ** logM200, "c200_relation": 10 ** logc}

        else:
            raise ValueError(case)

        bin_rows.append({"case": case, "lens_bin": lb, **meta})
        for i, r in enumerate(sub):
            p = float(pred[i]) if np.isfinite(pred[i]) else np.nan
            preds.append(p)
            rows.append({"case": case, "lens_bin": lb, "R_kpc": r["R_kpc"], "DeltaSigma_obs": r["DeltaSigma"], "DeltaSigma_err": r["DeltaSigma_err"], "DeltaSigma_pred": p, **meta})

    return np.array(preds, float), rows, bin_rows, np.array(priors, float)


def residuals(case, params, bins, data, args, a0):
    pred, _, _, priors = predict_case(case, params, bins, data, args, a0)
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
    if case in {"nfw_global_c_only", "nfw_plus_baryon_global_c"}:
        p0.append(math.log10(8.0))
        bounds.append((math.log10(1), math.log10(40)))

    for _ in bins:
        lm = 11.0
        lr = math.log10(5 * (10 ** (lm - 10)) ** 0.25)
        if case in {"baryon_only", "drnd_fixed_a0"}:
            p0 += [lm, lr]
            bounds += [(8, 12.8), (math.log10(0.2), math.log10(80))]
        elif case == "drnd_fixed_a0_screen2h":
            p0 += [lm, lr, 0.0]
            bounds += [(8, 12.8), (math.log10(0.2), math.log10(80)), (-8, 6)]
        elif case == "nfw_only_perbin":
            p0 += [12.0, math.log10(8.0)]
            bounds += [(9, 15), (math.log10(1), math.log10(40))]
        elif case == "nfw_plus_baryon_perbin":
            p0 += [lm, lr, 12.0, math.log10(8.0)]
            bounds += [(8, 12.8), (math.log10(0.2), math.log10(80)), (9, 15), (math.log10(1), math.log10(40))]
        elif case == "nfw_global_c_only":
            p0 += [12.0]
            bounds += [(9, 15)]
        elif case == "nfw_global_relation":
            p0 += [12.0]
            bounds += [(9, 15)]
        elif case == "nfw_plus_baryon_global_c":
            p0 += [lm, lr, 12.0]
            bounds += [(8, 12.8), (math.log10(0.2), math.log10(80)), (9, 15)]
        elif case == "nfw_plus_baryon_global_relation":
            p0 += [lm, lr, 12.0]
            bounds += [(8, 12.8), (math.log10(0.2), math.log10(80)), (9, 15)]
        else:
            raise ValueError(case)
    return np.array(p0, float), bounds


def generate_starts(case, bins, args):
    p0, bounds = base_initial_vector(case, bins, args)
    starts = [p0.copy()]

    # Deterministic baryonic mass/size starts.
    bary_logM = [9.5, 10.5, 11.2, 12.0]
    halo_logM = [11.0, 12.0, 13.0, 14.0]
    logc_vals = [math.log10(4.0), math.log10(8.0), math.log10(16.0)]
    logA_vals = [-3.0, -1.0, 0.0, 1.0]

    def clip_vec(v):
        out = np.array(v, float)
        for i, (lo, hi) in enumerate(bounds):
            out[i] = min(max(out[i], lo + 1e-6), hi - 1e-6)
        return out

    for lm in bary_logM:
        v = p0.copy()
        # modify all baryonic slots heuristically
        for i in range(len(v)):
            lo, hi = bounds[i]
            if lo <= 8.1 and hi >= 12.7:
                v[i] = lm
            if lo <= math.log10(0.21) and hi >= math.log10(79):
                v[i] = math.log10(5 * (10 ** (lm - 10)) ** 0.25)
        starts.append(clip_vec(v))

    for hm in halo_logM:
        for lc in logc_vals:
            v = p0.copy()
            for i, (lo, hi) in enumerate(bounds):
                if lo <= 9.1 and hi >= 14.9:
                    v[i] = hm
                elif lo <= math.log10(1.01) and hi >= math.log10(39):
                    v[i] = lc
            starts.append(clip_vec(v))

    if "screen2h" in case:
        for la in logA_vals:
            v = p0.copy()
            for i, (lo, hi) in enumerate(bounds):
                if lo <= -7.9 and hi >= 5.9:
                    v[i] = la
            starts.append(clip_vec(v))

    # Random starts, same mechanism for all models.
    seed_raw = f"{case}|{len(bins)}|{args.random_seed}".encode("utf-8")
    seed = int(hashlib.md5(seed_raw).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    for _ in range(args.n_random_starts):
        v = np.array([rng.uniform(lo, hi) for lo, hi in bounds], float)
        starts.append(v)

    # Deduplicate starts by rounded values.
    seen = set()
    unique = []
    for s in starts:
        key = tuple(np.round(s, 6))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique, bounds


def fit_case(case, bins, data, args, a0):
    starts, bounds = generate_starts(case, bins, args)
    attempts = []
    best = None

    def obj(p):
        r = residuals(case, p, bins, data, args, a0)
        return float(np.sum(r * r))

    for si, start in enumerate(starts):
        try:
            opt = minimize(obj, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": args.maxiter, "ftol": 1e-8})
            chi2 = float(opt.fun)
            attempts.append({"case": case, "start_id": si, "chi2": chi2, "success": bool(opt.success), "message": str(opt.message)})
            if best is None or chi2 < best["chi2"]:
                best = {"opt": opt, "chi2": chi2, "start_id": si}
        except Exception as e:
            attempts.append({"case": case, "start_id": si, "chi2": 1e300, "success": False, "message": f"exception: {e}"})

    if best is None:
        raise RuntimeError(f"All starts failed for {case}")

    opt = best["opt"]
    res = residuals(case, opt.x, bins, data, args, a0)
    chi2 = float(np.sum(res * res))
    pred, pred_rows, bin_rows, _ = predict_case(case, opt.x, bins, data, args, a0)
    n = sum(len(data[lb]) for lb in bins)
    k = len(opt.x)
    dof = max(n - k, 1)
    invalid_frac = float(np.mean((~np.isfinite(pred)) | (pred <= 0)))

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
    ap.add_argument("--c-prior-weight", type=float, default=1.0)
    ap.add_argument("--twohalo-slope", type=float, default=0.8)
    ap.add_argument("--R0-kpc", type=float, default=300.0)
    ap.add_argument("--Rt-kpc", type=float, default=200.0)
    ap.add_argument("--screen-q", type=float, default=3.0)
    ap.add_argument("--f-inner-max", type=float, default=0.3)
    ap.add_argument("--inner-quantile", type=float, default=0.35)
    ap.add_argument("--inner-penalty-weight", type=float, default=25.0)
    ap.add_argument("--min-log-sigma", type=float, default=0.03)
    ap.add_argument("--maxiter", type=int, default=700)
    ap.add_argument("--n-random-starts", type=int, default=12)
    ap.add_argument("--random-seed", type=int, default=12345)
    ap.add_argument("--outdir", default="DRND_ST_FAIR_NFW_TEST_V2")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tabledir = outdir / "tables"
    plotdir = outdir / "plots"
    tabledir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    rows = read_lensing_csv(args.lensing_csv)
    rows, mode_used = assign_bins(rows, args.bin_mode, args.n_bins)
    write_csv(tabledir / "binned_lensing_data.csv", rows, preferred=["lens_bin", "R_kpc", "DeltaSigma", "DeltaSigma_err", "logMbar_input"])

    data = {}
    for r in rows:
        data.setdefault(r["lens_bin"], []).append(r)
    bins = sorted(data.keys())
    a0 = a0_from_chi(args.H0, args.chi)

    cases = [
        "baryon_only",
        "drnd_fixed_a0",
        "drnd_fixed_a0_screen2h",
        "nfw_only_perbin",
        "nfw_plus_baryon_perbin",
        "nfw_global_c_only",
        "nfw_global_relation",
        "nfw_plus_baryon_global_c",
        "nfw_plus_baryon_global_relation",
    ]

    results = []
    all_bin_rows = []
    all_attempts = []

    for case in cases:
        print("[fit]", case)
        fit = fit_case(case, bins, data, args, a0)
        write_csv(tabledir / f"predictions_{case}.csv", fit["pred_rows"])
        all_bin_rows += fit["bin_rows"]
        all_attempts += fit["attempts"]
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
        comp.append(rr)

    write_csv(tabledir / "model_comparison.csv", comp, preferred=[
        "case", "chi2", "k_params", "dof", "chi2_per_dof", "AIC", "BIC",
        "delta_AIC_vs_best", "delta_BIC_vs_best", "success", "best_start_id",
        "n_starts", "invalid_prediction_fraction"
    ])

    for metric in ["BIC", "AIC"]:
        ordered = sorted(comp, key=lambda r: r[metric])
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(ordered)), [r[metric] for r in ordered])
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels([r["case"] for r in ordered], rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"DRND-ST fair NFW comparison V2 by {metric}")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(plotdir / f"model_comparison_{metric.lower()}.png", dpi=180)
        plt.close(fig)

    by = {r["case"]: r for r in comp}
    drnd = by.get("drnd_fixed_a0")
    drnd2 = by.get("drnd_fixed_a0_screen2h")
    nfw_rel = by.get("nfw_plus_baryon_global_relation")
    nfw_gc = by.get("nfw_plus_baryon_global_c")
    nfw_per = by.get("nfw_plus_baryon_perbin")

    decision = {
        "best_model_by_BIC": best_bic["case"],
        "best_model_by_AIC": best_aic["case"],
        "DRND_fixed_BIC": drnd["BIC"] if drnd else None,
        "DRND_screened_BIC": drnd2["BIC"] if drnd2 else None,
        "NFW_plus_baryon_perbin_BIC": nfw_per["BIC"] if nfw_per else None,
        "NFW_plus_baryon_global_c_BIC": nfw_gc["BIC"] if nfw_gc else None,
        "NFW_plus_baryon_global_relation_BIC": nfw_rel["BIC"] if nfw_rel else None,
        "DRND_fixed_beats_NFW_global_relation_BIC": drnd["BIC"] < nfw_rel["BIC"] if drnd and nfw_rel else None,
        "DRND_fixed_beats_NFW_global_c_BIC": drnd["BIC"] < nfw_gc["BIC"] if drnd and nfw_gc else None,
        "DRND_screened_beats_NFW_global_relation_BIC": drnd2["BIC"] < nfw_rel["BIC"] if drnd2 and nfw_rel else None,
        "delta_BIC_DRNDfixed_minus_NFWglobalRelation": drnd["BIC"] - nfw_rel["BIC"] if drnd and nfw_rel else None,
        "delta_BIC_DRNDfixed_minus_NFWglobalC": drnd["BIC"] - nfw_gc["BIC"] if drnd and nfw_gc else None,
        "DRND_fixed_sanity_ok_chi2_lt_1e4": drnd["chi2"] < 1e4 if drnd else None,
        "DRND_screened_sanity_ok_chi2_lt_1e4": drnd2["chi2"] < 1e4 if drnd2 else None,
    }

    summary = {
        "model": "DRND-ST fair NFW parameter-control test V2",
        "input": vars(args) | {"n_points": len(rows), "bins": bins, "bin_mode_used": mode_used, "a0_fixed": a0},
        "model_comparison": comp,
        "decision": decision,
        "cautions": [
            "Diagnostic spherical projection only.",
            "Multi-start is applied to all models equally to avoid optimizer-driven conclusions.",
            "The concentration-mass relation is a simple diagnostic relation, not a precision cosmology calibration.",
            "No full HOD, CDM two-halo, miscentering, satellites, selection model or covariance.",
        ],
    }

    (outdir / "summary.json").write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
    print(json.dumps(safe_json({
        "outdir": str(outdir),
        "decision": decision,
        "files": [
            "summary.json",
            "tables/model_comparison.csv",
            "tables/bin_parameters.csv",
            "tables/fit_attempts.csv",
            "plots/model_comparison_bic.png",
            "plots/model_comparison_aic.png",
        ],
    }), indent=2))


if __name__ == "__main__":
    main()

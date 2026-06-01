#!/usr/bin/env python3
"""
DRND v7.1 Joint Multi-Lensing + Growth Test

Combines:
    - multiple CMB lensing NPZ datasets, e.g. Planck + ACT + SPT,
    - fσ8 / fs8 growth measurements,
    - E_G measurements,
    - optional S8 weak-lensing / clustering summary points.

The purpose is to test the surviving DRND signature:

    growth suppressed:          fσ8 down, S8 down
    lensing less suppressed:    Sigma / mu >= 1
    CMB lensing:                no large standalone boost required

Input formats
-------------

1) Lensing NPZ files:
    required arrays:
        L, cl_obs, cl_lcdm, cov

    Example:
        planck_lensing_inputs_from_uploaded.npz
        act_lensing_inputs.npz
        spt_lensing_inputs.npz

2) fσ8 CSV:
    required columns:
        z,fs8_obs,fs8_err,fs8_lcdm

    Optional columns:
        k
    If k is absent, --fs8-k-eff is used.

3) E_G CSV:
    required columns:
        z,k,eg_obs,eg_err,eg_lcdm

4) S8 CSV:
    required columns:
        s8_obs,s8_err,s8_lcdm

    Optional columns:
        z,k,label

    If z or k is absent, --s8-z-eff and --s8-k-eff are used.

Run example
-----------

    python DRND_v71_joint_multi_lensing_growth.py ^
      --lensing-npz planck_lensing_inputs_from_uploaded.npz act_lensing_inputs.npz spt_lensing_inputs.npz ^
      --labels Planck ACT SPT ^
      --fs8-csv fsigma8_data.csv ^
      --eg-csv eg_data.csv ^
      --s8-csv s8_data.csv ^
      --outdir results_v71 ^
      --run-all-modes ^
      --nsteps 4000 --burn 1000 --thin 10

Fast smoke test
---------------

    python DRND_v71_joint_multi_lensing_growth.py ^
      --lensing-npz planck_lensing_inputs_from_uploaded.npz act_lensing_inputs.npz ^
      --labels Planck ACT ^
      --make-example-growth-csv

Then:

    python DRND_v71_joint_multi_lensing_growth.py ^
      --lensing-npz planck_lensing_inputs_from_uploaded.npz act_lensing_inputs.npz ^
      --labels Planck ACT ^
      --fs8-csv example_fs8.csv ^
      --eg-csv example_eg.csv ^
      --s8-csv example_s8.csv ^
      --outdir results_v71_example ^
      --run-all-modes ^
      --nsteps 1000 --burn 250 --thin 5

Important limitation
--------------------
This is an audit-level semi-analytic likelihood. The fσ8 and S8 response models
are first-order compressed approximations. Publication-grade validation requires
a modified CLASS/CAMB implementation for mu(k,a), Sigma(k,a), and growth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

try:
    import emcee
except ImportError as exc:
    raise SystemExit("Missing dependency: emcee. Install with: pip install emcee") from exc

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False


# ============================================================
# 1. Cosmology and DRND kernels
# ============================================================

class Cosmology:
    def __init__(self, Om0: float = 0.315, Ol0: float = 0.685, h: float = 0.674, c_kms: float = 299792.458):
        self.Om0 = float(Om0)
        self.Ol0 = float(Ol0)
        self.h = float(h)
        self.c_kms = float(c_kms)


class FixedDRND:
    def __init__(self, alpha_inv: float = 137.035999177):
        self.alpha_inv = float(alpha_inv)

    @property
    def alpha(self) -> float:
        return 1.0 / self.alpha_inv

    @property
    def q(self) -> float:
        return 1.0 - 2.0 / np.pi

    @property
    def mu0_core(self) -> float:
        return -self.q * (1.0 + 1.0 / 9.0)

    @property
    def cD_kms(self) -> float:
        return (2.0 / 3.0) * self.alpha**1.5 * 299792.458


def E_of_a(a: np.ndarray, cosmo: Cosmology) -> np.ndarray:
    return np.sqrt(cosmo.Om0 / a**3 + cosmo.Ol0)


def Omega_L_of_a(a: np.ndarray, cosmo: Cosmology) -> np.ndarray:
    return cosmo.Ol0 / E_of_a(a, cosmo) ** 2


def L_D(a: np.ndarray, cosmo: Cosmology) -> np.ndarray:
    return Omega_L_of_a(a, cosmo) / cosmo.Ol0


def k_star_hmpc(a: np.ndarray, cosmo: Cosmology, fixed: FixedDRND) -> np.ndarray:
    return (5.0 / 19.0) * (100.0 * a * E_of_a(a, cosmo)) / fixed.cD_kms


def W_D(k_hmpc: np.ndarray, a: np.ndarray, cosmo: Cosmology, fixed: FixedDRND) -> np.ndarray:
    x = k_hmpc / k_star_hmpc(a, cosmo, fixed)
    return 4.0 * x**4 / (1.0 + x**4) ** 2


def T_kernel(k_hmpc: np.ndarray, z: np.ndarray, cosmo: Cosmology, fixed: FixedDRND) -> np.ndarray:
    a = 1.0 / (1.0 + z)
    return L_D(a, cosmo) * W_D(k_hmpc, a, cosmo, fixed)


def fractal_boost(epsilon: float) -> float:
    if epsilon >= 3.0:
        return np.inf
    return (3.0 / (3.0 - epsilon)) ** 2 - 1.0


def mu_eff(k_hmpc: np.ndarray, z: np.ndarray, epsilon: float, mu0: float, cosmo: Cosmology, fixed: FixedDRND) -> np.ndarray:
    T = T_kernel(k_hmpc, z, cosmo, fixed)
    return 1.0 + mu0 * T


def Sigma_eff(k_hmpc: np.ndarray, z: np.ndarray, epsilon: float, mu0: float, cosmo: Cosmology, fixed: FixedDRND) -> np.ndarray:
    T = T_kernel(k_hmpc, z, cosmo, fixed)
    return 1.0 + (mu0 + fractal_boost(epsilon)) * T


def comoving_chi_mpc(z_grid: np.ndarray, cosmo: Cosmology) -> np.ndarray:
    c_over_H0 = cosmo.c_kms / (100.0 * cosmo.h)
    a = 1.0 / (1.0 + z_grid)
    Ez = np.sqrt(cosmo.Om0 / a**3 + cosmo.Ol0)
    return c_over_H0 * cumulative_trapezoid(1.0 / Ez, z_grid, initial=0.0)


# ============================================================
# 2. Generic loading utilities
# ============================================================

def stable_inv(cov: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("[WARN] Covariance inversion failed; using pseudo-inverse.")
        return np.linalg.pinv(cov)


def read_csv(path: str) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if arr.ndim == 0:
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def require_columns(arr: np.ndarray, required: List[str], path: str) -> None:
    names = arr.dtype.names
    if names is None:
        raise ValueError(f"{path} must have a header row.")
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"{path} missing columns {missing}; found {names}")


def col_float(arr: np.ndarray, name: str, default: Optional[float] = None) -> np.ndarray:
    names = arr.dtype.names or ()
    if name in names:
        return np.asarray(arr[name], dtype=float)
    if default is None:
        raise KeyError(name)
    return np.full(len(arr), float(default), dtype=float)


def col_str(arr: np.ndarray, name: str, default_prefix: str) -> List[str]:
    names = arr.dtype.names or ()
    if name in names:
        return [str(x) for x in arr[name]]
    return [f"{default_prefix}_{i}" for i in range(len(arr))]


# ============================================================
# 3. Lensing loading and model
# ============================================================

def load_lensing_npz(path: str, label: str) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    required = ["L", "cl_obs", "cl_lcdm", "cov"]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise ValueError(f"{path} missing arrays: {missing}")

    L = np.asarray(d["L"], dtype=float)
    obs = np.asarray(d["cl_obs"], dtype=float)
    lcdm = np.asarray(d["cl_lcdm"], dtype=float)
    cov = np.asarray(d["cov"], dtype=float)

    if L.ndim != 1:
        raise ValueError(f"{path}: L must be 1D.")
    if obs.shape != L.shape:
        raise ValueError(f"{path}: cl_obs shape {obs.shape} != L shape {L.shape}.")
    if lcdm.shape != L.shape:
        raise ValueError(f"{path}: cl_lcdm shape {lcdm.shape} != L shape {L.shape}.")
    if cov.shape != (len(L), len(L)):
        raise ValueError(f"{path}: cov shape {cov.shape} != ({len(L)}, {len(L)}).")
    if np.any(lcdm == 0):
        raise ValueError(f"{path}: cl_lcdm contains zeros.")

    metadata = ""
    if "metadata" in d.files:
        try:
            metadata = str(d["metadata"])
        except Exception:
            metadata = ""

    return {
        "path": path,
        "label": label,
        "L": L,
        "cl_obs": obs,
        "cl_lcdm": lcdm,
        "cov": cov,
        "cov_inv": stable_inv(cov),
        "metadata": metadata,
    }


def precompute_los_terms(L_values: np.ndarray, cosmo: Cosmology, fixed: FixedDRND, zmax: float, nz: int) -> Dict[str, np.ndarray]:
    z = np.linspace(1e-4, zmax, nz)
    chi = comoving_chi_mpc(z, cosmo)
    chi_star = chi[-1]
    a = 1.0 / (1.0 + z)
    dchi = np.gradient(chi)

    geom_base = ((chi_star - chi) / (chi_star * np.maximum(chi, 1e-6))) ** 2
    geom_base[chi <= 1e-3] = 0.0

    weights_all = []
    T_all = []

    for L in L_values:
        k_hmpc = ((float(L) + 0.5) / np.maximum(chi, 1e-6)) / cosmo.h
        T = L_D(a, cosmo) * W_D(k_hmpc, a, cosmo, fixed)
        weights = geom_base * dchi
        den = np.sum(weights)
        if den <= 0 or not np.isfinite(den):
            raise ValueError(f"Invalid LoS denominator for L={L}")
        weights_all.append(weights / den)
        T_all.append(T)

    return {
        "z": z,
        "weights": np.asarray(weights_all, dtype=float),
        "T": np.asarray(T_all, dtype=float),
    }


def lensing_ratio(theta: np.ndarray, pre: Dict[str, np.ndarray], growth_response: float) -> np.ndarray:
    epsilon, mu0 = theta
    T = pre["T"]
    weights = pre["weights"]
    fb = fractal_boost(epsilon)

    mu = 1.0 + mu0 * T
    Sigma = 1.0 + (mu0 + fb) * T
    Dmu = 1.0 + growth_response * (mu - 1.0)

    if np.any(~np.isfinite(Sigma)) or np.any(~np.isfinite(Dmu)):
        return np.full(T.shape[0], np.nan)
    if np.any(Sigma <= 0) or np.any(Dmu <= 0):
        return np.full(T.shape[0], np.nan)

    return np.sum(weights * Sigma**2 * Dmu**2, axis=1)


def model_lensing_cl(theta: np.ndarray, dataset: Dict[str, Any], pre: Dict[str, np.ndarray], growth_response: float) -> np.ndarray:
    return dataset["cl_lcdm"] * lensing_ratio(theta, pre, growth_response)


def chi2_lensing(theta: np.ndarray, dataset: Dict[str, Any], pre: Dict[str, np.ndarray], growth_response: float) -> float:
    th = model_lensing_cl(theta, dataset, pre, growth_response)
    if np.any(~np.isfinite(th)):
        return np.inf
    r = dataset["cl_obs"] - th
    return float(r @ dataset["cov_inv"] @ r)


def chi2_lensing_lcdm(dataset: Dict[str, Any]) -> float:
    r = dataset["cl_obs"] - dataset["cl_lcdm"]
    return float(r @ dataset["cov_inv"] @ r)


# ============================================================
# 4. Growth datasets: fσ8, E_G, S8
# ============================================================

def load_fs8(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    arr = read_csv(path)
    require_columns(arr, ["z", "fs8_obs", "fs8_err", "fs8_lcdm"], path)
    return {
        "path": path,
        "z": col_float(arr, "z"),
        "k": col_float(arr, "k", np.nan),
        "obs": col_float(arr, "fs8_obs"),
        "err": col_float(arr, "fs8_err"),
        "lcdm": col_float(arr, "fs8_lcdm"),
        "label": col_str(arr, "label", "fs8"),
    }


def load_eg(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    arr = read_csv(path)
    require_columns(arr, ["z", "k", "eg_obs", "eg_err", "eg_lcdm"], path)
    return {
        "path": path,
        "z": col_float(arr, "z"),
        "k": col_float(arr, "k"),
        "obs": col_float(arr, "eg_obs"),
        "err": col_float(arr, "eg_err"),
        "lcdm": col_float(arr, "eg_lcdm"),
        "label": col_str(arr, "label", "eg"),
    }


def load_s8(path: Optional[str], default_z: float, default_k: float) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    arr = read_csv(path)
    require_columns(arr, ["s8_obs", "s8_err", "s8_lcdm"], path)
    return {
        "path": path,
        "z": col_float(arr, "z", default_z),
        "k": col_float(arr, "k", default_k),
        "obs": col_float(arr, "s8_obs"),
        "err": col_float(arr, "s8_err"),
        "lcdm": col_float(arr, "s8_lcdm"),
        "label": col_str(arr, "label", "s8"),
    }


def fs8_ratio(theta: np.ndarray, fs8: Dict[str, Any], cosmo: Cosmology, fixed: FixedDRND, k_eff: float, response: float) -> np.ndarray:
    epsilon, mu0 = theta
    z = fs8["z"]
    k = np.asarray(fs8["k"], dtype=float)
    k = np.where(np.isfinite(k), k, float(k_eff))
    T = T_kernel(k, z, cosmo, fixed)
    return np.clip(1.0 + response * mu0 * T, 0.02, 5.0)


def model_fs8(theta: np.ndarray, fs8: Dict[str, Any], cosmo: Cosmology, fixed: FixedDRND, k_eff: float, response: float) -> np.ndarray:
    return fs8["lcdm"] * fs8_ratio(theta, fs8, cosmo, fixed, k_eff, response)


def eg_ratio(theta: np.ndarray, eg: Dict[str, Any], cosmo: Cosmology, fixed: FixedDRND) -> np.ndarray:
    epsilon, mu0 = theta
    z = eg["z"]
    k = eg["k"]
    mu = mu_eff(k, z, epsilon, mu0, cosmo, fixed)
    Sigma = Sigma_eff(k, z, epsilon, mu0, cosmo, fixed)
    if np.any(mu <= 0) or np.any(Sigma <= 0):
        return np.full_like(z, np.nan, dtype=float)
    return Sigma / mu


def model_eg(theta: np.ndarray, eg: Dict[str, Any], cosmo: Cosmology, fixed: FixedDRND) -> np.ndarray:
    return eg["lcdm"] * eg_ratio(theta, eg, cosmo, fixed)


def s8_ratio(theta: np.ndarray, s8: Dict[str, Any], cosmo: Cosmology, fixed: FixedDRND, response: float) -> np.ndarray:
    epsilon, mu0 = theta
    z = s8["z"]
    k = s8["k"]
    T = T_kernel(k, z, cosmo, fixed)
    return np.clip(1.0 + response * mu0 * T, 0.02, 5.0)


def model_s8(theta: np.ndarray, s8: Dict[str, Any], cosmo: Cosmology, fixed: FixedDRND, response: float) -> np.ndarray:
    return s8["lcdm"] * s8_ratio(theta, s8, cosmo, fixed, response)


def chi2_diag(obs: np.ndarray, th: np.ndarray, err: np.ndarray) -> float:
    if np.any(~np.isfinite(th)):
        return np.inf
    r = (obs - th) / err
    return float(np.sum(r**2))


# ============================================================
# 5. Priors and modes
# ============================================================

def box_prior(theta: np.ndarray, box: str) -> float:
    epsilon, mu0 = theta

    if box == "wide":
        return 0.0 if (0.0 < epsilon < 0.8 and -1.5 < mu0 < 0.8) else -np.inf

    if box == "growth_negative":
        return 0.0 if (0.0 < epsilon < 0.8 and -0.8 < mu0 < 0.1) else -np.inf

    if box == "conservative":
        return 0.0 if (0.0 < epsilon < 0.3 and -0.8 < mu0 < 0.3) else -np.inf

    raise ValueError(f"Unknown box prior: {box}")


def gaussian_prior(x: float, mean: Optional[float], sigma: Optional[float]) -> float:
    if mean is None or sigma is None:
        return 0.0
    if sigma <= 0:
        raise ValueError("Gaussian prior sigma must be positive.")
    return -0.5 * ((x - mean) / sigma) ** 2


def mode_config(mode: str, fixed: FixedDRND, args: argparse.Namespace) -> Dict[str, Any]:
    if mode == "free":
        return {"mode": mode, "box": "wide", "mu0_mean": None, "mu0_sigma": None}

    if mode == "mild":
        return {"mode": mode, "box": "wide", "mu0_mean": args.mild_mu0_mean, "mu0_sigma": args.mild_mu0_sigma}

    if mode == "drnd_core":
        return {"mode": mode, "box": "growth_negative", "mu0_mean": fixed.mu0_core, "mu0_sigma": args.core_mu0_sigma}

    if mode == "drnd_core_loose":
        return {"mode": mode, "box": "growth_negative", "mu0_mean": fixed.mu0_core, "mu0_sigma": args.core_mu0_sigma_loose}

    raise ValueError(f"Unknown mode: {mode}")


def log_prior(theta: np.ndarray, cfg: Dict[str, Any]) -> float:
    lp = box_prior(theta, cfg["box"])
    if not np.isfinite(lp):
        return -np.inf
    return lp + gaussian_prior(theta[1], cfg["mu0_mean"], cfg["mu0_sigma"])


# ============================================================
# 6. Total likelihood / chi2
# ============================================================

def chi2_components(
    theta: np.ndarray,
    lensing: List[Dict[str, Any]],
    pre: List[Dict[str, np.ndarray]],
    fs8: Optional[Dict[str, Any]],
    eg: Optional[Dict[str, Any]],
    s8: Optional[Dict[str, Any]],
    cosmo: Cosmology,
    fixed: FixedDRND,
    args: argparse.Namespace,
) -> Dict[str, float]:
    out = {}
    total = 0.0

    for ds, pr in zip(lensing, pre):
        c = chi2_lensing(theta, ds, pr, args.growth_response)
        key = f"lensing_{ds['label']}"
        out[key] = float(c)
        total += c

    if fs8 is not None:
        th = model_fs8(theta, fs8, cosmo, fixed, args.fs8_k_eff, args.fs8_response)
        c = chi2_diag(fs8["obs"], th, fs8["err"])
        out["fs8"] = float(c)
        total += c

    if eg is not None:
        th = model_eg(theta, eg, cosmo, fixed)
        c = chi2_diag(eg["obs"], th, eg["err"])
        out["eg"] = float(c)
        total += c

    if s8 is not None:
        th = model_s8(theta, s8, cosmo, fixed, args.s8_response)
        c = chi2_diag(s8["obs"], th, s8["err"])
        out["s8"] = float(c)
        total += c

    out["total"] = float(total)
    return out


def chi2_lcdm_components(
    lensing: List[Dict[str, Any]],
    fs8: Optional[Dict[str, Any]],
    eg: Optional[Dict[str, Any]],
    s8: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    out = {}
    total = 0.0

    for ds in lensing:
        c = chi2_lensing_lcdm(ds)
        key = f"lensing_{ds['label']}"
        out[key] = float(c)
        total += c

    if fs8 is not None:
        c = chi2_diag(fs8["obs"], fs8["lcdm"], fs8["err"])
        out["fs8"] = float(c)
        total += c

    if eg is not None:
        c = chi2_diag(eg["obs"], eg["lcdm"], eg["err"])
        out["eg"] = float(c)
        total += c

    if s8 is not None:
        c = chi2_diag(s8["obs"], s8["lcdm"], s8["err"])
        out["s8"] = float(c)
        total += c

    out["total"] = float(total)
    return out


def log_likelihood(
    theta: np.ndarray,
    lensing: List[Dict[str, Any]],
    pre: List[Dict[str, np.ndarray]],
    fs8: Optional[Dict[str, Any]],
    eg: Optional[Dict[str, Any]],
    s8: Optional[Dict[str, Any]],
    cosmo: Cosmology,
    fixed: FixedDRND,
    args: argparse.Namespace,
) -> float:
    comps = chi2_components(theta, lensing, pre, fs8, eg, s8, cosmo, fixed, args)
    c = comps["total"]
    if not np.isfinite(c):
        return -np.inf
    return -0.5 * c


def log_probability(
    theta: np.ndarray,
    cfg: Dict[str, Any],
    lensing: List[Dict[str, Any]],
    pre: List[Dict[str, np.ndarray]],
    fs8: Optional[Dict[str, Any]],
    eg: Optional[Dict[str, Any]],
    s8: Optional[Dict[str, Any]],
    cosmo: Cosmology,
    fixed: FixedDRND,
    args: argparse.Namespace,
) -> float:
    lp = log_prior(theta, cfg)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, lensing, pre, fs8, eg, s8, cosmo, fixed, args)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ============================================================
# 7. MCMC helpers
# ============================================================

def init_walkers(cfg: Dict[str, Any], args: argparse.Namespace, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps0 = args.start_epsilon if args.start_epsilon is not None else 0.10

    if args.start_mu0 is not None:
        mu0 = args.start_mu0
    elif cfg["mu0_mean"] is not None:
        mu0 = cfg["mu0_mean"]
    else:
        mu0 = -0.05

    pos = np.column_stack([
        eps0 + args.start_spread * rng.normal(size=args.nwalkers),
        mu0 + args.start_spread * rng.normal(size=args.nwalkers),
    ])

    for i in range(args.nwalkers):
        tries = 0
        while not np.isfinite(log_prior(pos[i], cfg)):
            pos[i] = np.array([
                eps0 + args.start_spread * rng.normal(),
                mu0 + args.start_spread * rng.normal(),
            ])
            tries += 1
            if tries > 10000:
                raise RuntimeError("Could not initialize walkers.")
    return pos


def percentile_summary(samples: np.ndarray) -> Dict[str, Dict[str, float]]:
    eps = np.percentile(samples[:, 0], [16, 50, 84])
    mu = np.percentile(samples[:, 1], [16, 50, 84])
    df = 3.0 - samples[:, 0]
    fb = np.array([fractal_boost(e) for e in samples[:, 0]])

    return {
        "epsilon": {
            "p16": float(eps[0]), "p50": float(eps[1]), "p84": float(eps[2]),
            "minus": float(eps[1] - eps[0]), "plus": float(eps[2] - eps[1]),
        },
        "mu0": {
            "p16": float(mu[0]), "p50": float(mu[1]), "p84": float(mu[2]),
            "minus": float(mu[1] - mu[0]), "plus": float(mu[2] - mu[1]),
        },
        "df": {
            "p16": float(np.percentile(df, 16)),
            "p50": float(np.percentile(df, 50)),
            "p84": float(np.percentile(df, 84)),
        },
        "fractal_boost": {
            "p16": float(np.percentile(fb, 16)),
            "p50": float(np.percentile(fb, 50)),
            "p84": float(np.percentile(fb, 84)),
        },
    }


def best_fit(samples: np.ndarray, cfg: Dict[str, Any], lensing, pre, fs8, eg, s8, cosmo, fixed, args) -> Tuple[np.ndarray, float]:
    max_eval = min(len(samples), 100000)
    if len(samples) > max_eval:
        rng = np.random.default_rng(123)
        eval_samples = samples[rng.choice(len(samples), max_eval, replace=False)]
    else:
        eval_samples = samples

    vals = np.array([
        log_probability(s, cfg, lensing, pre, fs8, eg, s8, cosmo, fixed, args)
        for s in eval_samples
    ])
    idx = int(np.nanargmax(vals))
    return eval_samples[idx], float(vals[idx])


def n_data_points(lensing, fs8, eg, s8) -> int:
    n = sum(len(ds["L"]) for ds in lensing)
    if fs8 is not None:
        n += len(fs8["z"])
    if eg is not None:
        n += len(eg["z"])
    if s8 is not None:
        n += len(s8["obs"])
    return int(n)


def information_criteria(chi2_lcdm: float, chi2_drnd: float, n_data: int, k_lcdm: int = 0, k_drnd: int = 2) -> Dict[str, float]:
    aic_l = chi2_lcdm + 2 * k_lcdm
    aic_d = chi2_drnd + 2 * k_drnd
    bic_l = chi2_lcdm + k_lcdm * np.log(n_data)
    bic_d = chi2_drnd + k_drnd * np.log(n_data)
    return {
        "aic_lcdm": float(aic_l),
        "aic_drnd": float(aic_d),
        "delta_aic_drnd_minus_lcdm": float(aic_d - aic_l),
        "bic_lcdm": float(bic_l),
        "bic_drnd": float(bic_d),
        "delta_bic_drnd_minus_lcdm": float(bic_d - bic_l),
    }


# ============================================================
# 8. Plotting
# ============================================================

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def plot_trace(chain: np.ndarray, outpath: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(chain[:, :, 0], alpha=0.35, lw=0.5)
    axes[0].set_ylabel("epsilon")
    axes[0].grid(alpha=0.25)
    axes[1].plot(chain[:, :, 1], alpha=0.35, lw=0.5)
    axes[1].set_ylabel("mu0")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_corner(samples: np.ndarray, outpath: Path) -> None:
    if HAS_CORNER:
        fig = corner.corner(samples, labels=[r"$\epsilon$", r"$\mu_0$"], quantiles=[0.16, 0.5, 0.84], show_titles=True)
        fig.savefig(outpath, dpi=180)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.25)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("mu0")
        ax.set_title("Posterior samples; install corner for full corner plot")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outpath, dpi=180)
        plt.close(fig)


def plot_lensing_fit(ds, pre, theta, args, mode, outdir):
    L = ds["L"]
    obs = ds["cl_obs"]
    lcdm = ds["cl_lcdm"]
    cov = ds["cov"]
    th = model_lensing_cl(theta, ds, pre, args.growth_response)
    err = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="green", ls="--", label="LCDM")
    ax.errorbar(L, obs/lcdm, yerr=err/np.abs(lcdm), fmt="o", color="red", capsize=2, label=f"{ds['label']} / LCDM")
    ax.plot(L, th/lcdm, color="blue", lw=2.5, label="DRND / LCDM")
    ax.set_xlabel("L")
    ax.set_ylabel("lensing power / LCDM")
    ax.set_title(f"{ds['label']} lensing, mode={mode}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / safe_filename(f"fit_lensing_{ds['label']}_{mode}.png"), dpi=180)
    plt.close(fig)


def plot_growth_fit(name, data, obs_name, model_values, mode, outdir):
    z = data["z"]
    obs = data["obs"]
    err = data["err"]
    lcdm = data["lcdm"]
    order = np.argsort(z)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="green", ls="--", label="LCDM")
    ax.errorbar(z, obs/lcdm, yerr=err/np.abs(lcdm), fmt="o", color="red", capsize=2, label=f"{obs_name} obs/LCDM")
    ax.plot(z[order], (model_values/lcdm)[order], color="blue", lw=2.5, label="DRND/LCDM")
    ax.set_xlabel("z")
    ax.set_ylabel(f"{obs_name} / LCDM")
    ax.set_title(f"{obs_name}, mode={mode}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / safe_filename(f"fit_{name}_{mode}.png"), dpi=180)
    plt.close(fig)


def plot_mode_comparison(results: Dict[str, Any], outdir: Path) -> None:
    modes = list(results.keys())
    x = np.arange(len(modes))
    eps_med, eps_lo, eps_hi = [], [], []
    mu_med, mu_lo, mu_hi = [], [], []
    dchi2, daic, dbic = [], [], []

    for mode in modes:
        r = results[mode]
        ep = r["posterior"]["epsilon"]
        mp = r["posterior"]["mu0"]
        eps_med.append(ep["p50"]); eps_lo.append(ep["minus"]); eps_hi.append(ep["plus"])
        mu_med.append(mp["p50"]); mu_lo.append(mp["minus"]); mu_hi.append(mp["plus"])
        dchi2.append(r["delta_chi2_lcdm_minus_drnd"])
        daic.append(r["information_criteria"]["delta_aic_drnd_minus_lcdm"])
        dbic.append(r["information_criteria"]["delta_bic_drnd_minus_lcdm"])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    axes[0].errorbar(x, eps_med, yerr=[eps_lo, eps_hi], fmt="o", capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(modes, rotation=25, ha="right")
    axes[0].set_ylabel("epsilon")
    axes[0].set_title("Posterior epsilon")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(x, mu_med, yerr=[mu_lo, mu_hi], fmt="o", capsize=4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(modes, rotation=25, ha="right")
    axes[1].set_ylabel("mu0")
    axes[1].set_title("Posterior mu0")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, dchi2, "o-")
    axes[2].axhline(0, color="k", ls="--")
    axes[2].set_xticks(x); axes[2].set_xticklabels(modes, rotation=25, ha="right")
    axes[2].set_ylabel("positive favors DRND")
    axes[2].set_title("Delta chi2")
    axes[2].grid(alpha=0.25)

    axes[3].plot(x, daic, "o-", label="Delta AIC")
    axes[3].plot(x, dbic, "s-", label="Delta BIC")
    axes[3].axhline(0, color="k", ls="--")
    axes[3].set_xticks(x); axes[3].set_xticklabels(modes, rotation=25, ha="right")
    axes[3].set_ylabel("positive favors LCDM")
    axes[3].set_title("Information criteria")
    axes[3].grid(alpha=0.25)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(outdir / "mode_comparison_v71.png", dpi=180)
    plt.close(fig)


# ============================================================
# 9. Run one mode
# ============================================================

def run_mode(mode, lensing, pre, fs8, eg, s8, cosmo, fixed, args, outdir, seed_offset=0):
    cfg = mode_config(mode, fixed, args)
    pos = init_walkers(cfg, args, args.seed + seed_offset)

    sampler = emcee.EnsembleSampler(
        args.nwalkers, 2, log_probability,
        args=(cfg, lensing, pre, fs8, eg, s8, cosmo, fixed, args)
    )

    print(f"[INFO] Running mode={mode}: box={cfg['box']}, mu0_prior={cfg['mu0_mean']} +/- {cfg['mu0_sigma']}")
    sampler.run_mcmc(pos, args.nsteps, progress=args.progress)

    chain = sampler.get_chain()
    samples = sampler.get_chain(discard=args.burn, thin=args.thin, flat=True)
    if len(samples) == 0:
        raise ValueError("No samples after burn/thin.")

    theta_best, logpost_best = best_fit(samples, cfg, lensing, pre, fs8, eg, s8, cosmo, fixed, args)
    chi2_l = chi2_lcdm_components(lensing, fs8, eg, s8)
    chi2_d = chi2_components(theta_best, lensing, pre, fs8, eg, s8, cosmo, fixed, args)
    n = n_data_points(lensing, fs8, eg, s8)
    ic = information_criteria(chi2_l["total"], chi2_d["total"], n)

    tag = safe_filename(f"mode_{mode}")
    np.savez(outdir / f"samples_{tag}.npz", samples=samples, chain=chain, best_theta=theta_best)
    plot_trace(chain, outdir / f"trace_{tag}.png", f"DRND v7.1 joint trace: {mode}")
    plot_corner(samples, outdir / f"corner_{tag}.png")

    for ds, pr in zip(lensing, pre):
        plot_lensing_fit(ds, pr, theta_best, args, mode, outdir)

    if fs8 is not None:
        plot_growth_fit("fs8", fs8, "f_sigma8", model_fs8(theta_best, fs8, cosmo, fixed, args.fs8_k_eff, args.fs8_response), mode, outdir)
    if eg is not None:
        plot_growth_fit("eg", eg, "E_G", model_eg(theta_best, eg, cosmo, fixed), mode, outdir)
    if s8 is not None:
        plot_growth_fit("s8", s8, "S8", model_s8(theta_best, s8, cosmo, fixed, args.s8_response), mode, outdir)

    return {
        "mode": mode,
        "config": cfg,
        "best_fit": {
            "epsilon": float(theta_best[0]),
            "mu0": float(theta_best[1]),
            "df": float(3.0 - theta_best[0]),
            "fractal_boost": float(fractal_boost(theta_best[0])),
            "log_posterior": float(logpost_best),
        },
        "posterior": percentile_summary(samples),
        "chi2_lcdm": chi2_l,
        "chi2_drnd_best": chi2_d,
        "delta_chi2_lcdm_minus_drnd": float(chi2_l["total"] - chi2_d["total"]),
        "information_criteria": ic,
    }


# ============================================================
# 10. Example CSV generation
# ============================================================

def make_example_growth_csv() -> None:
    z = np.array([0.15, 0.25, 0.38, 0.51, 0.61, 0.76, 1.0])
    fs8_lcdm = np.array([0.48, 0.47, 0.46, 0.45, 0.44, 0.42, 0.39])
    fs8_obs = fs8_lcdm * np.array([0.98, 0.96, 0.97, 0.95, 0.96, 0.99, 1.02])
    fs8_err = np.array([0.05, 0.04, 0.035, 0.035, 0.04, 0.05, 0.07])
    np.savetxt("example_fs8.csv", np.column_stack([z, fs8_obs, fs8_err, fs8_lcdm]), delimiter=",",
               header="z,fs8_obs,fs8_err,fs8_lcdm", comments="")

    z2 = np.array([0.32, 0.55, 0.70, 0.85])
    k2 = np.array([0.18, 0.18, 0.20, 0.20])
    eg_lcdm = np.array([0.45, 0.40, 0.37, 0.34])
    eg_obs = eg_lcdm * np.array([0.97, 0.94, 1.02, 1.00])
    eg_err = np.array([0.10, 0.06, 0.08, 0.10])
    np.savetxt("example_eg.csv", np.column_stack([z2, k2, eg_obs, eg_err, eg_lcdm]), delimiter=",",
               header="z,k,eg_obs,eg_err,eg_lcdm", comments="")

    labels = ["DES_like", "KiDS_like", "HSC_like"]
    s8_obs = np.array([0.776, 0.766, 0.780])
    s8_err = np.array([0.017, 0.020, 0.030])
    s8_lcdm = np.array([0.832, 0.832, 0.832])
    z_eff = np.array([0.5, 0.5, 0.6])
    k_eff = np.array([0.2, 0.2, 0.2])

    with open("example_s8.csv", "w", encoding="utf-8") as f:
        f.write("label,z,k,s8_obs,s8_err,s8_lcdm\n")
        for row in zip(labels, z_eff, k_eff, s8_obs, s8_err, s8_lcdm):
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")

    print("[OK] wrote example_fs8.csv, example_eg.csv, example_s8.csv")
    print("[WARN] These are synthetic smoke-test files, not scientific data.")


# ============================================================
# 11. Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--lensing-npz", nargs="*", default=[], help="Zero or more lensing NPZ files.")
    ap.add_argument("--labels", nargs="*", help="Labels for lensing NPZ files.")

    ap.add_argument("--fs8-csv")
    ap.add_argument("--eg-csv")
    ap.add_argument("--s8-csv")

    ap.add_argument("--make-example-growth-csv", action="store_true")

    ap.add_argument("--outdir", default="results_v71_joint_multi_growth")

    ap.add_argument("--run-all-modes", action="store_true")
    ap.add_argument("--mode", choices=["free", "mild", "drnd_core", "drnd_core_loose"], default="free")

    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--nsteps", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=800)
    ap.add_argument("--thin", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--progress", action="store_true")

    ap.add_argument("--start-epsilon", type=float, default=None)
    ap.add_argument("--start-mu0", type=float, default=None)
    ap.add_argument("--start-spread", type=float, default=1e-3)

    ap.add_argument("--growth-response", type=float, default=0.5)
    ap.add_argument("--fs8-k-eff", type=float, default=0.15)
    ap.add_argument("--fs8-response", type=float, default=0.5)
    ap.add_argument("--s8-z-eff", type=float, default=0.5)
    ap.add_argument("--s8-k-eff", type=float, default=0.2)
    ap.add_argument("--s8-response", type=float, default=0.5)

    ap.add_argument("--mild-mu0-mean", type=float, default=-0.05)
    ap.add_argument("--mild-mu0-sigma", type=float, default=0.02)
    ap.add_argument("--core-mu0-sigma", type=float, default=0.03)
    ap.add_argument("--core-mu0-sigma-loose", type=float, default=0.10)

    ap.add_argument("--zmax", type=float, default=8.0)
    ap.add_argument("--nz", type=int, default=1000)

    ap.add_argument("--Om0", type=float, default=0.315)
    ap.add_argument("--Ol0", type=float, default=0.685)
    ap.add_argument("--h", type=float, default=0.674)
    ap.add_argument("--alpha-inv", type=float, default=137.035999177)

    args = ap.parse_args()

    if args.make_example_growth_csv:
        make_example_growth_csv()
        return

    if args.burn >= args.nsteps:
        raise SystemExit("burn must be smaller than nsteps.")
    if args.nwalkers < 4:
        raise SystemExit("nwalkers must be at least 4.")

    if not args.lensing_npz and args.fs8_csv is None and args.eg_csv is None and args.s8_csv is None:
        raise SystemExit("Provide at least one data input: --lensing-npz, --fs8-csv, --eg-csv, or --s8-csv.")

    if args.labels:
        if len(args.labels) != len(args.lensing_npz):
            raise SystemExit("--labels length must match --lensing-npz length.")
        labels = args.labels
    else:
        labels = [Path(p).stem for p in args.lensing_npz]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cosmo = Cosmology(Om0=args.Om0, Ol0=args.Ol0, h=args.h)
    fixed = FixedDRND(alpha_inv=args.alpha_inv)

    lensing = [load_lensing_npz(p, lab) for p, lab in zip(args.lensing_npz, labels)]

    pre = []
    for ds in lensing:
        print(f"[INFO] Precomputing LoS for {ds['label']} ({len(ds['L'])} bins)")
        pre.append(precompute_los_terms(ds["L"], cosmo, fixed, args.zmax, args.nz))

    fs8 = load_fs8(args.fs8_csv)
    eg = load_eg(args.eg_csv)
    s8 = load_s8(args.s8_csv, args.s8_z_eff, args.s8_k_eff)

    modes = ["free", "mild", "drnd_core", "drnd_core_loose"] if args.run_all_modes else [args.mode]

    results = {
        "inputs": {
            "lensing": [
                {"path": ds["path"], "label": ds["label"], "n_bins": int(len(ds["L"])), "chi2_lcdm": float(chi2_lensing_lcdm(ds)), "metadata": str(ds.get("metadata", ""))[:2000]}
                for ds in lensing
            ],
            "fs8_csv": args.fs8_csv,
            "eg_csv": args.eg_csv,
            "s8_csv": args.s8_csv,
        },
        "n_data_total": n_data_points(lensing, fs8, eg, s8),
        "settings": {
            "growth_response": args.growth_response,
            "fs8_k_eff": args.fs8_k_eff,
            "fs8_response": args.fs8_response,
            "s8_z_eff": args.s8_z_eff,
            "s8_k_eff": args.s8_k_eff,
            "s8_response": args.s8_response,
            "mild_mu0_mean": args.mild_mu0_mean,
            "mild_mu0_sigma": args.mild_mu0_sigma,
            "core_mu0_mean": fixed.mu0_core,
            "core_mu0_sigma": args.core_mu0_sigma,
            "core_mu0_sigma_loose": args.core_mu0_sigma_loose,
        },
        "lcdm_chi2": chi2_lcdm_components(lensing, fs8, eg, s8),
        "modes": {},
        "warning": "Audit-level compressed joint test. Final validation requires CAMB/CLASS.",
    }

    for i, mode in enumerate(modes):
        res = run_mode(mode, lensing, pre, fs8, eg, s8, cosmo, fixed, args, outdir, seed_offset=1000*i)
        results["modes"][mode] = res

    with open(outdir / "joint_multi_growth_summary_v71.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_mode_comparison(results["modes"], outdir)

    print(json.dumps(results, indent=2))
    print(f"[OK] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()

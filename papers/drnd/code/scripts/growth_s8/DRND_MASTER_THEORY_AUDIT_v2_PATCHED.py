#!/usr/bin/env python3
"""
DRND_MASTER_THEORY_AUDIT_v2.py

Scalone sprawdzanie DRND-U v5.0: Phase Time & Graph Memory.

Jeden skrypt zamiast kolejnych pojedynczych testow. To jest audyt diagnostyczny,
nie finalny likelihood CMB/BBN/SH0ES.

Sprawdza naraz:
1. LCDM / DRND / DRND causal / DRND phase-time background na H(z), SN, BAO.
2. AIC/BIC i delta chi2 vs najlepszy LCDM.
3. Kalibratorowy offset beta_cal i mapowanie H0_bg -> H0_SH0ES.
4. Zgodnosc phase-time: gamma_t, epsilon_bg, epsilon_cal_eff, BBN safety.
5. CMB proxy: theta_star, R_shift, z_eq dla H0 grid.
6. Sound horizon requirement dla H0 target.
7. Growth toy z retardacja czasu fazowego.
8. Status ciemnej materii: reinterpretacja Am, nie eliminacja.

Uruchomienie:
  python DRND_MASTER_THEORY_AUDIT_v2.py --outdir DRND_MASTER_V2_RESULTS

Z danymi hostow:
  python DRND_MASTER_THEORY_AUDIT_v2.py --sn-csv sn_data_with_hosts.csv --outdir DRND_MASTER_V2_RESULTS

Szybki tryb:
  python DRND_MASTER_THEORY_AUDIT_v2.py --quick --outdir DRND_MASTER_V2_RESULTS
"""
from __future__ import annotations

import argparse, csv, json, math, re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, solve_ivp

def trapz_compat(y, x):
    """NumPy 1.x/2.x compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return trapz_compat(y, x)


C_KM_S = 299792.458
T_CMB = 2.7255
NEFF_STD = 3.046


def norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null", "na"}:
            return default
        return float(s)
    except Exception:
        return default


def boolish(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "calibrator", "anchor", "cepheid", "sh0es"}:
        return True
    if s in {"0", "false", "no", "n", "hubble", "flow", "hf"}:
        return False
    v = to_float(s)
    if np.isfinite(v):
        return bool(v)
    return None


def read_csv_rows(path: Optional[str]):
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def write_csv(path: Path, rows: List[Dict], preferred_fields: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_fields = sorted(set(k for r in rows for k in r.keys()))
    if preferred_fields:
        fields = [f for f in preferred_fields if f in all_fields] + [f for f in all_fields if f not in preferred_fields]
    else:
        fields = all_fields
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


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
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def load_hz(path):
    rows = read_csv_rows(path)
    out = []
    for row in rows:
        z = pick(row, ["z", "redshift"])
        H = pick(row, ["H", "Hz", "H_z", "Hobs", "H_obs"])
        err = pick(row, ["H_err", "Hz_err", "sigma_H", "err", "error", "sigma"])
        if np.isfinite(z) and np.isfinite(H) and np.isfinite(err) and err > 0:
            out.append((z, H, err))
    a = np.array(out, dtype=float) if out else np.empty((0, 3))
    return {"z": a[:, 0] if len(a) else np.array([]),
            "H": a[:, 1] if len(a) else np.array([]),
            "err": a[:, 2] if len(a) else np.array([])}


def load_sn(path):
    rows = read_csv_rows(path)
    out, raw = [], []
    for row in rows:
        z = pick(row, ["z", "zcmb", "zhel", "redshift"])
        mu = pick(row, ["mu", "distance_modulus", "mub", "mu_obs"])
        err = pick(row, ["mu_err", "sigma_mu", "mu_error", "err", "error", "sigma"])
        if np.isfinite(z) and np.isfinite(mu) and np.isfinite(err) and err > 0 and z > 0:
            out.append((z, mu, err)); raw.append(row)
    if not out:
        return {"z": np.array([]), "mu": np.array([]), "err": np.array([]), "rows": [], "columns": []}
    a = np.array(out, dtype=float); idx = np.argsort(a[:, 0]); a = a[idx]; raw = [raw[i] for i in idx]
    return {"z": a[:, 0], "mu": a[:, 1], "err": a[:, 2], "rows": raw, "columns": list(raw[0].keys())}


def canonical_bao_obs(s: str):
    aliases = {
        "dv_over_rd": "DV_over_rd", "dv_rd": "DV_over_rd", "dvrd": "DV_over_rd",
        "dm_over_rd": "DM_over_rd", "dm_rd": "DM_over_rd", "dmrd": "DM_over_rd",
        "dh_over_rd": "DH_over_rd", "dh_rd": "DH_over_rd", "dhrd": "DH_over_rd",
        "h_rd": "H_rd", "hrd": "H_rd",
        "rd_over_dv": "rd_over_DV", "rd_dv": "rd_over_DV", "rddv": "rd_over_DV",
    }
    return aliases.get(norm_col(s))


def load_bao(path):
    rows = read_csv_rows(path); out = []
    for row in rows:
        z = pick(row, ["z", "redshift"])
        obs = canonical_bao_obs(pick_str(row, ["observable", "obs", "type", "measurement"]))
        if not np.isfinite(z) or obs is None:
            continue
        val = pick(row, ["value", "val", "measurement_value", obs])
        err = pick(row, ["error", "err", "sigma", "uncertainty", f"{obs}_err", f"{obs}_error"])
        if np.isfinite(val) and np.isfinite(err) and err > 0:
            out.append({"z": float(z), "obs": obs, "value": float(val), "err": float(err)})
    return out


def get_calibrator_flags(sn):
    if not sn["rows"]:
        return None
    cols = ["calibrator", "is_calibrator", "anchor", "cepheid_calibrator", "cepheid", "sh0es"]
    flags, found = [], False
    for row in sn["rows"]:
        b = None
        for c in cols:
            val = pick_str(row, [c], default="")
            if val != "":
                bb = boolish(val)
                if bb is not None:
                    b = bb; found = True; break
        flags.append(False if b is None else b)
    return np.array(flags, dtype=bool) if found else None


def get_host_mass(sn):
    if not sn["rows"]:
        return None
    vals = np.array([pick(r, ["host_mass", "logmass", "mass", "log_mstar", "mstar"]) for r in sn["rows"]], dtype=float)
    return vals if np.sum(np.isfinite(vals)) >= 5 else None


def omega_gamma_h2(Tcmb=T_CMB):
    return 2.469e-5 * (Tcmb / 2.7255) ** 4


def omega_gamma_from_h(H0, Tcmb=T_CMB):
    h = H0 / 100.0
    return omega_gamma_h2(Tcmb) / (h * h)


def omega_r_from_h(H0, Neff=NEFF_STD, Tcmb=T_CMB):
    h = H0 / 100.0
    return omega_gamma_h2(Tcmb) * (1.0 + 0.2271 * Neff) / (h * h)


def epsilon_bg_a(a, epsilon_today, a_c):
    a = np.asarray(a, dtype=float)
    return epsilon_today * (1.0 + a_c) * a / (a + a_c)


class Background:
    def __init__(self, model, H0, Omega_m_lcdm, Omega_b, Am, epsilon_today, a_c, rd,
                 causal_power=0.0, gamma_t=0.0, phase_time_mode="none", zmax=3.0, ngrid=5000):
        self.model = model; self.H0 = float(H0); self.Omega_b = float(Omega_b)
        self.Omega_m_lcdm = float(Omega_m_lcdm) if np.isfinite(Omega_m_lcdm) else np.nan
        self.Am = float(Am) if np.isfinite(Am) else np.nan
        self.epsilon_today = float(epsilon_today); self.a_c = float(a_c) if np.isfinite(a_c) else 0.4
        self.rd = float(rd); self.causal_power = float(causal_power); self.gamma_t = float(gamma_t)
        self.phase_time_mode = phase_time_mode; self.Or = omega_r_from_h(self.H0)
        amin = min(1e-4, 1.0 / (1.0 + max(zmax, 0.1) * 1.1))
        self.a_grid = np.geomspace(amin, 1.0, int(ngrid)); self.loga = np.log(self.a_grid)
        self.E_grid = self._make_E()
        self.z_grid = np.linspace(0, max(zmax * 1.05, 0.01), int(ngrid))
        self.Ez_grid = self.E(self.z_grid)
        self.chi = (C_KM_S / self.H0) * cumulative_trapezoid(1.0 / self.Ez_grid, self.z_grid, initial=0.0)

    def _drnd_Rm(self):
        eps = epsilon_bg_a(self.a_grid, self.epsilon_today, self.a_c)
        p_m = 3.0 - eps
        integ = cumulative_trapezoid(p_m, self.loga, initial=0.0)
        return np.exp(integ[-1] - integ)

    def _make_E(self):
        a = self.a_grid
        if self.model == "lcdm":
            Om = self.Omega_m_lcdm; Ode = 1.0 - self.Or - Om
            return np.sqrt(self.Or * a**-4 + Om * a**-3 + Ode)
        if self.model in {"drnd", "drnd_causal", "drnd_phase"}:
            Rm = self._drnd_Rm(); Ax = 1.0 - self.Or - self.Omega_b - self.Am
            E = np.sqrt(self.Or * a**-4 + self.Omega_b * a**-3 + self.Am * Rm + Ax)
            if self.model == "drnd_causal":
                E = E * np.power(1.0 + self.epsilon_today * (1.0 - a), self.causal_power)
            if self.model == "drnd_phase":
                eps = epsilon_bg_a(a, self.epsilon_today, self.a_c)
                dt_ratio = 1.0 - self.gamma_t * eps
                if self.phase_time_mode == "multiply_H":
                    E = E * dt_ratio
                elif self.phase_time_mode == "divide_H":
                    E = E / np.maximum(dt_ratio, 1e-8)
                elif self.phase_time_mode == "none":
                    pass
                else:
                    raise ValueError(self.phase_time_mode)
            return E
        raise ValueError(self.model)

    def E(self, z):
        z = np.asarray(z, dtype=float); a = 1.0 / (1.0 + z)
        return np.interp(np.log(a), self.loga, self.E_grid)
    def H(self, z): return self.H0 * self.E(z)
    def DM(self, z): return np.interp(np.asarray(z, dtype=float), self.z_grid, self.chi)
    def DH(self, z): return C_KM_S / self.H(z)
    def DL(self, z): z = np.asarray(z, dtype=float); return (1.0 + z) * self.DM(z)
    def DV(self, z): z = np.asarray(z, dtype=float); return (z * self.DM(z)**2 * self.DH(z))**(1/3)
    def mu(self, z): return 5.0 * np.log10(np.maximum(self.DL(z), 1e-12)) + 25.0
    def bao_pred(self, z, obs):
        if obs == "DV_over_rd": return self.DV(z) / self.rd
        if obs == "DM_over_rd": return self.DM(z) / self.rd
        if obs == "DH_over_rd": return self.DH(z) / self.rd
        if obs == "H_rd": return self.H(z) * self.rd
        if obs == "rd_over_DV": return self.rd / self.DV(z)
        raise ValueError(obs)


def fit_offset(obs, pred, err):
    w = 1.0 / np.maximum(err, 1e-12)**2
    off = np.sum(w * (obs - pred)) / np.sum(w)
    chi2 = np.sum(w * (obs - pred - off)**2)
    return float(off), float(chi2)


def eval_background(params, hz, sn, bao, zmax, ngrid):
    bg = Background(zmax=zmax, ngrid=ngrid, **params)
    chi_hz = float(np.sum(((hz["H"] - bg.H(hz["z"])) / hz["err"])**2)) if len(hz["z"]) else 0.0
    sn_off, chi_sn = fit_offset(sn["mu"], bg.mu(sn["z"]), sn["err"]) if len(sn["z"]) else (0.0, 0.0)
    chi_bao = 0.0
    for r in bao:
        pred = float(bg.bao_pred(r["z"], r["obs"]))
        chi_bao += ((r["value"] - pred) / r["err"])**2
    total = chi_hz + chi_sn + chi_bao
    n_eff = len(hz["z"]) + max(0, len(sn["z"]) - 1) + len(bao)
    k = 3 if params["model"] == "lcdm" else 5
    if params["model"] == "drnd_causal" and abs(params.get("causal_power", 0.0)) > 1e-12: k += 1
    if params["model"] == "drnd_phase" and params.get("phase_time_mode") != "none": k += 1
    return {**{k0: v for k0, v in params.items() if k0 != "Omega_b"}, "Omega_b": params["Omega_b"],
            "chi2_total": float(total), "chi2_hz": float(chi_hz), "chi2_sn": float(chi_sn),
            "chi2_bao": float(chi_bao), "sn_offset": float(sn_off), "n_eff": int(n_eff), "k_eff": int(k),
            "aic": float(total + 2 * k), "bic": float(total + k * math.log(max(n_eff, 2))), "_bg": bg}


def required_DeltaM_SH0ES(H0_bg, H0_local): return 5.0 * math.log10(H0_local / H0_bg)
def required_DeltaMu_distance(H0_bg, H0_local): return -5.0 * math.log10(H0_local / H0_bg)
def H0_from_DeltaM(H0_bg, DeltaM): return H0_bg * 10.0 ** (DeltaM / 5.0)


def weighted_fit(y, X, err):
    w = 1.0 / np.maximum(err, 1e-12)**2
    WX = X * np.sqrt(w)[:, None]; Wy = y * np.sqrt(w)
    cov = np.linalg.pinv(WX.T @ WX); coeff = cov @ (WX.T @ Wy)
    pred = X @ coeff; resid = y - pred
    chi2 = float(np.sum((resid / err)**2))
    return coeff, cov, resid, chi2


def calibrator_fit(sn, H0_bg, Omega_m):
    if not len(sn["z"]): return [], None
    z, mu, err = sn["z"], sn["mu"], sn["err"]
    bg = Background("lcdm", H0_bg, Omega_m, 0.049, np.nan, 0.0, 0.4, 147.09, zmax=max(float(np.max(z)), 0.5), ngrid=5000)
    y = mu - bg.mu(z); flags = get_calibrator_flags(sn); host = get_host_mass(sn); rows = []
    X0 = np.ones((len(z), 1)); c0, cov0, resid0, chi0 = weighted_fit(y, X0, err)
    rows.append({"model": "offset_only", "H0_bg": H0_bg, "chi2": chi0, "delta_chi2": 0.0,
                 "offset": float(c0[0]), "beta_cal": None, "beta_cal_err": None, "beta_mass": None, "beta_mass_err": None})
    if flags is not None:
        X = np.column_stack([np.ones(len(z)), flags.astype(float)])
        c, cov, resid, chi = weighted_fit(y, X, err)
        rows.append({"model": "calibrator_only", "H0_bg": H0_bg, "chi2": chi, "delta_chi2": chi - chi0,
                     "offset": float(c[0]), "beta_cal": float(c[1]), "beta_cal_err": float(math.sqrt(max(cov[1, 1], 0.0))),
                     "beta_mass": None, "beta_mass_err": None, "n_cal": int(np.sum(flags)), "n_flow": int(np.sum(~flags))})
    xstd = None
    if host is not None:
        mask = np.isfinite(host)
        if np.sum(mask) > 5 and np.std(host[mask]) > 0:
            xstd = np.where(mask, (host - np.mean(host[mask])) / np.std(host[mask]), 0.0)
            X = np.column_stack([np.ones(len(z)), xstd])
            c, cov, resid, chi = weighted_fit(y, X, err)
            rows.append({"model": "host_mass_only", "H0_bg": H0_bg, "chi2": chi, "delta_chi2": chi - chi0,
                         "offset": float(c[0]), "beta_cal": None, "beta_cal_err": None,
                         "beta_mass": float(c[1]), "beta_mass_err": float(math.sqrt(max(cov[1, 1], 0.0)))})
    if flags is not None and xstd is not None:
        X = np.column_stack([np.ones(len(z)), flags.astype(float), xstd])
        c, cov, resid, chi = weighted_fit(y, X, err)
        rows.append({"model": "calibrator_plus_host_mass", "H0_bg": H0_bg, "chi2": chi, "delta_chi2": chi - chi0,
                     "offset": float(c[0]), "beta_cal": float(c[1]), "beta_cal_err": float(math.sqrt(max(cov[1, 1], 0.0))),
                     "beta_mass": float(c[2]), "beta_mass_err": float(math.sqrt(max(cov[2, 2], 0.0))),
                     "n_cal": int(np.sum(flags)), "n_flow": int(np.sum(~flags))})
    return rows, {"flags": flags, "host_mass": host, "residual": y - c0[0]}


def phase_time_check_rows(epsilon_grid, gamma_grid, H0_bg_grid, H0_target, a_c=0.3, a_bbn=1e-9):
    rows = []
    for eps in epsilon_grid:
        eps_bbn = float(epsilon_bg_a(a_bbn, eps, a_c))
        for gamma in gamma_grid:
            dt_today_global = 1.0 - gamma * eps
            dt_bbn = 1.0 - gamma * eps_bbn
            DeltaM_global = 5.0 * math.log10(max(dt_today_global, 1e-30))
            for H0_bg in H0_bg_grid:
                H0_app = H0_bg * dt_today_global
                req_dt = H0_target / H0_bg
                req_DeltaM = required_DeltaM_SH0ES(H0_bg, H0_target)
                eps_cal_eff_needed = (1.0 - req_dt) / gamma if abs(gamma) > 1e-12 else np.nan
                rows.append({"epsilon_today_bg": eps, "gamma_t": gamma, "a_c": a_c, "epsilon_bbn": eps_bbn,
                             "dt_ratio_bbn": dt_bbn, "BBN_safe_abs_dt_minus_1": abs(dt_bbn - 1.0),
                             "dt_ratio_today_if_global": dt_today_global, "DeltaM_if_global": DeltaM_global,
                             "H0_bg": H0_bg, "H0_target": H0_target, "H0_apparent_if_global": H0_app,
                             "H0_error_if_global": H0_app - H0_target, "required_dt_ratio": req_dt,
                             "required_DeltaM_SH0ES": req_DeltaM,
                             "epsilon_cal_eff_needed_for_target": eps_cal_eff_needed,
                             "global_epsilon_overshoots_local_need": bool(np.isfinite(eps_cal_eff_needed) and abs(eps) > abs(eps_cal_eff_needed) * 1.5)})
    return rows


def sound_speed(z, H0, Omega_b):
    a = 1.0 / (1.0 + np.asarray(z, dtype=float)); Og = omega_gamma_from_h(H0)
    R = 3.0 * Omega_b * a / (4.0 * Og)
    return C_KM_S / np.sqrt(3.0 * (1.0 + R))


def E_lcdm_z(z, H0, Om, DeltaNeff=0.0):
    z = np.asarray(z, dtype=float); a = 1.0 / (1.0 + z)
    Or = omega_r_from_h(H0, NEFF_STD + DeltaNeff); Ode = 1.0 - Or - Om
    return np.sqrt(Or * a**-4 + Om * a**-3 + Ode)


def sound_horizon(H0, Om, Ob, z_start=1059.0, zmax=1e7, nz=10000, DeltaNeff=0.0):
    y = np.linspace(math.log(1 + z_start), math.log(1 + zmax), int(nz)); z = np.exp(y) - 1; dzdy = np.exp(y)
    H = H0 * E_lcdm_z(z, H0, Om, DeltaNeff); cs = sound_speed(z, H0, Ob)
    return float(trapz_compat(cs / H * dzdy, y))


def comoving_distance(H0, Om, zmax, nz=10000, DeltaNeff=0.0):
    y = np.linspace(0, math.log(1 + zmax), int(nz)); z = np.exp(y) - 1; dzdy = np.exp(y)
    E = E_lcdm_z(z, H0, Om, DeltaNeff)
    return float((C_KM_S / H0) * trapz_compat(dzdy / E, y))


def cmb_proxy(H0, Om, Ob, zstar=1089.92, zdrag=1059.0, nz=10000):
    rs = sound_horizon(H0, Om, Ob, z_start=zstar, nz=nz); rd = sound_horizon(H0, Om, Ob, z_start=zdrag, nz=nz)
    DM = comoving_distance(H0, Om, zstar, nz=nz); theta = rs / DM; Rshift = math.sqrt(Om) * H0 * DM / C_KM_S
    zeq = Om / omega_r_from_h(H0) - 1.0
    return {"H0": H0, "Omega_m": Om, "Omega_b": Ob, "rs_star": rs, "rd_drag": rd, "DM_star": DM, "theta_star": theta, "R_shift": Rshift, "z_eq": zeq}


def growth_toy_row(H0=70.5, Om=0.315, epsilon_today=0.135, a_c=0.4, gamma_t=-0.25, lambda_tau=0.0, lambda_mu=0.0, target_damping=0.96):
    def E2(a):
        Or = omega_r_from_h(H0); Ode = 1.0 - Or - Om
        return Or * a**-4 + Om * a**-3 + Ode
    def dlnH_dlna(a):
        Or = omega_r_from_h(H0); e2 = E2(a); de2 = -4 * Or * a**-4 - 3 * Om * a**-3
        return 0.5 * de2 / e2
    def rhs(x, y):
        a = math.exp(x); eps_s = float(epsilon_bg_a(a, epsilon_today, a_c)); eps_t = -gamma_t * eps_s
        Om_a = Om * a**-3 / E2(a); A = 2.0 + dlnH_dlna(a) + lambda_tau * eps_t
        mu = max(0.0, 1.0 - lambda_mu * eps_s); B = 1.5 * mu * Om_a; D, Dp = y
        return [Dp, B * D - A * Dp]
    def rhs0(x, y):
        a = math.exp(x); Om_a = Om * a**-3 / E2(a); A = 2.0 + dlnH_dlna(a); B = 1.5 * Om_a; D, Dp = y
        return [Dp, B * D - A * Dp]
    x0, x1 = math.log(1e-3), 0.0
    sol = solve_ivp(rhs, (x0, x1), [math.exp(x0), math.exp(x0)], rtol=1e-6, atol=1e-9)
    sol0 = solve_ivp(rhs0, (x0, x1), [math.exp(x0), math.exp(x0)], rtol=1e-6, atol=1e-9)
    D = float(sol.y[0, -1]); D0 = float(sol0.y[0, -1]); damping = D / D0 if D0 != 0 else np.nan
    return {"H0": H0, "Omega_m": Om, "epsilon_today": epsilon_today, "a_c": a_c, "gamma_t": gamma_t,
            "lambda_tau": lambda_tau, "lambda_mu": lambda_mu, "D_today": D, "D_baseline": D0,
            "growth_damping_ratio": damping, "target_damping_ratio": target_damping,
            "abs_error_to_target": abs(damping - target_damping) if np.isfinite(damping) else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz-csv", default="hz_data.csv")
    ap.add_argument("--sn-csv", default=None)
    ap.add_argument("--bao-csv", default="bao_data.csv")
    ap.add_argument("--outdir", default="DRND_MASTER_V2_RESULTS")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--H0-bg-grid", type=float, nargs="+", default=[67.4, 69.0, 70.5, 71.0, 72.0])
    ap.add_argument("--H0-local-target", type=float, default=73.0)
    ap.add_argument("--Om-lcdm-grid", type=float, nargs="+", default=[0.30, 0.315, 0.33])
    ap.add_argument("--Omega-b", type=float, default=0.049)
    ap.add_argument("--Am-grid", type=float, nargs="+", default=[0.26, 0.28, 0.30])
    ap.add_argument("--epsilon-grid", type=float, nargs="+", default=[0.05, 0.135, 0.15])
    ap.add_argument("--a-c-grid", type=float, nargs="+", default=[0.3, 0.4])
    ap.add_argument("--rd-grid", type=float, nargs="+", default=[147.09, 144.0, 142.0])
    ap.add_argument("--causal-power-grid", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--gamma-t-grid", type=float, nargs="+", default=[-0.7, -0.5, -0.25, 0.0, 0.25])
    ap.add_argument("--phase-time-background-modes", nargs="+", default=["none"], choices=["none", "multiply_H", "divide_H"])
    ap.add_argument("--ngrid", type=int, default=3500)
    ap.add_argument("--skip-growth", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.Om_lcdm_grid = [0.315]; args.Am_grid = [0.26, 0.30]; args.epsilon_grid = [0.05, 0.15]
        args.a_c_grid = [0.3]; args.rd_grid = [147.09]; args.causal_power_grid = [0.0, 0.5]; args.gamma_t_grid = [-0.7, 0.0]
    outdir = Path(args.outdir); tabledir = outdir / "tables"; plotdir = outdir / "plots"
    tabledir.mkdir(parents=True, exist_ok=True); plotdir.mkdir(parents=True, exist_ok=True)
    sn_csv = args.sn_csv or ("sn_data_with_hosts.csv" if Path("sn_data_with_hosts.csv").exists() else "sn_data.csv")
    hz, sn, bao = load_hz(args.hz_csv), load_sn(sn_csv), load_bao(args.bao_csv)
    zmax = 3.0
    if len(hz["z"]): zmax = max(zmax, float(np.max(hz["z"])))
    if len(sn["z"]): zmax = max(zmax, float(np.max(sn["z"])))
    if bao: zmax = max(zmax, max(r["z"] for r in bao))
    bg_rows = []
    for H0 in args.H0_bg_grid:
        for Om in args.Om_lcdm_grid:
            for rd in args.rd_grid:
                p = {"model": "lcdm", "H0": H0, "Omega_m_lcdm": Om, "Omega_b": args.Omega_b, "Am": np.nan,
                     "epsilon_today": 0.0, "a_c": np.nan, "rd": rd, "causal_power": 0.0, "gamma_t": 0.0, "phase_time_mode": "none"}
                try: bg_rows.append(eval_background(p, hz, sn, bao, zmax, args.ngrid))
                except Exception: pass
    for H0 in args.H0_bg_grid:
        for Am in args.Am_grid:
            for eps in args.epsilon_grid:
                for ac in args.a_c_grid:
                    for rd in args.rd_grid:
                        base = {"H0": H0, "Omega_m_lcdm": np.nan, "Omega_b": args.Omega_b, "Am": Am, "epsilon_today": eps, "a_c": ac, "rd": rd}
                        for model, cp, gamma, mode in [("drnd", 0.0, 0.0, "none")]:
                            try: bg_rows.append(eval_background({"model": model, **base, "causal_power": cp, "gamma_t": gamma, "phase_time_mode": mode}, hz, sn, bao, zmax, args.ngrid))
                            except Exception: pass
                        for cp in args.causal_power_grid:
                            if abs(cp) < 1e-12: continue
                            try: bg_rows.append(eval_background({"model": "drnd_causal", **base, "causal_power": cp, "gamma_t": 0.0, "phase_time_mode": "none"}, hz, sn, bao, zmax, args.ngrid))
                            except Exception: pass
                        for gamma in args.gamma_t_grid:
                            for mode in args.phase_time_background_modes:
                                try: bg_rows.append(eval_background({"model": "drnd_phase", **base, "causal_power": 0.0, "gamma_t": gamma, "phase_time_mode": mode}, hz, sn, bao, zmax, args.ngrid))
                                except Exception: pass
    bg_rows.sort(key=lambda r: r["chi2_total"])
    lcdm_best = next((r for r in bg_rows if r["model"] == "lcdm"), None)
    for r in bg_rows:
        if lcdm_best:
            r["delta_chi2_vs_best_lcdm"] = r["chi2_total"] - lcdm_best["chi2_total"]
            r["delta_aic_vs_best_lcdm"] = r["aic"] - lcdm_best["aic"]
            r["delta_bic_vs_best_lcdm"] = r["bic"] - lcdm_best["bic"]
    write_csv(tabledir / "background_fit_table.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in bg_rows])
    best_bg = bg_rows[0] if bg_rows else None
    bias_rows = [{"H0_bg": H0, "H0_local_target": args.H0_local_target,
                  "required_DeltaM_SH0ES": required_DeltaM_SH0ES(H0, args.H0_local_target),
                  "required_DeltaMu_distance": required_DeltaMu_distance(H0, args.H0_local_target)} for H0 in args.H0_bg_grid]
    write_csv(tabledir / "required_calibrator_bias.csv", bias_rows)
    cal_rows = []; cal_aux = None
    for H0 in args.H0_bg_grid:
        rows, aux = calibrator_fit(sn, H0, 0.315); cal_rows.extend(rows); cal_aux = cal_aux or aux
    write_csv(tabledir / "calibrator_fit_table.csv", cal_rows)
    beta_source = None
    for pref in ["calibrator_plus_host_mass", "calibrator_only"]:
        beta_source = next((r for r in cal_rows if r["model"] == pref and abs(r["H0_bg"] - 70.5) < 1e-9 and r.get("beta_cal") is not None), None)
        if beta_source: break
    sign_rows = []
    if beta_source:
        beta, beta_err = beta_source["beta_cal"], beta_source["beta_cal_err"]
        for H0 in args.H0_bg_grid:
            req = required_DeltaM_SH0ES(H0, args.H0_local_target)
            for mapping, dm in [("DeltaM=+beta", beta), ("DeltaM=-beta", -beta)]:
                H0_from = H0_from_DeltaM(H0, dm)
                sign_rows.append({"H0_bg": H0, "H0_target": args.H0_local_target, "required_DeltaM": req, "mapping": mapping,
                                  "DeltaM_used": dm, "beta": beta, "beta_err": beta_err, "H0_from_beta": H0_from,
                                  "H0_error": H0_from - args.H0_local_target,
                                  "compatible_1sigma": abs(dm - req) <= beta_err, "compatible_2sigma": abs(dm - req) <= 2 * beta_err})
    write_csv(tabledir / "SH0ES_sign_table.csv", sign_rows)
    phase_rows = phase_time_check_rows(args.epsilon_grid, args.gamma_t_grid, args.H0_bg_grid, args.H0_local_target, a_c=0.3, a_bbn=1e-9)
    write_csv(tabledir / "phase_time_consistency_table.csv", phase_rows)
    cmb_rows = []; baseline_cmb = cmb_proxy(67.4, 0.315, args.Omega_b, nz=6000 if args.quick else 10000)
    for H0 in args.H0_bg_grid:
        row = cmb_proxy(H0, 0.315, args.Omega_b, nz=6000 if args.quick else 10000)
        row["theta_frac_diff_vs_67p4_percent"] = 100 * (row["theta_star"] / baseline_cmb["theta_star"] - 1)
        row["R_frac_diff_vs_67p4_percent"] = 100 * (row["R_shift"] / baseline_cmb["R_shift"] - 1)
        row["CMB_theta_proxy_safe_0p15pct"] = abs(row["theta_frac_diff_vs_67p4_percent"]) <= 0.15
        cmb_rows.append(row)
    write_csv(tabledir / "cmb_proxy_table.csv", cmb_rows)
    rd_ref = baseline_cmb["rd_drag"]
    sound_rows = []
    for H0 in args.H0_bg_grid:
        rd_req = rd_ref * 67.4 / args.H0_local_target
        sound_rows.append({"H0_bg": H0, "H0_target": args.H0_local_target, "rd_reference_67p4": rd_ref,
                           "rd_required_for_target_by_H0rd": rd_req,
                           "required_rd_reduction_percent": 100 * (1 - rd_req / rd_ref),
                           "comment": "aggressive if >5%" if 100 * (1 - rd_req / rd_ref) > 5 else "mild"})
    write_csv(tabledir / "sound_horizon_requirement.csv", sound_rows)
    growth_rows = []
    if not args.skip_growth:
        for gamma in [-0.7, -0.5, -0.25]:
            for lt in [0.0, 0.5, 1.0, 1.5, 2.0]:
                for lm in [0.0, 0.25, 0.5]:
                    growth_rows.append(growth_toy_row(H0=70.5, Om=0.315, epsilon_today=0.15, a_c=0.3, gamma_t=gamma, lambda_tau=lt, lambda_mu=lm, target_damping=0.96))
        growth_rows.sort(key=lambda r: r["abs_error_to_target"])
    write_csv(tabledir / "growth_toy_table.csv", growth_rows)
    try:
        if bg_rows:
            fig, ax = plt.subplots(figsize=(10, 6)); top = bg_rows[:min(25, len(bg_rows))]
            ax.bar(range(len(top)), [r["chi2_total"] for r in top]); ax.set_xticks(range(len(top)))
            ax.set_xticklabels([f"{r['model']}\nH0={r['H0']}\neps={r.get('epsilon_today')}" for r in top], rotation=90)
            ax.set_ylabel("chi2 total"); ax.set_title("Top background fits"); ax.grid(alpha=0.25, axis="y"); fig.tight_layout(); fig.savefig(plotdir / "top_background_chi2.png", dpi=180); plt.close(fig)
        if sign_rows:
            fig, ax = plt.subplots(figsize=(10, 5)); plus = [r for r in sign_rows if r["mapping"] == "DeltaM=+beta"]
            ax.plot([r["H0_bg"] for r in plus], [r["H0_from_beta"] for r in plus], marker="o", label="DeltaM=+beta")
            ax.axhline(args.H0_local_target, color="black", ls="--", lw=1, label="target")
            ax.set_xlabel("H0 background"); ax.set_ylabel("H0 local from beta"); ax.set_title("SH0ES local H0 from calibrator beta")
            ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(plotdir / "H0_from_calibrator_beta.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5))
        for gamma in sorted(set(r["gamma_t"] for r in phase_rows)):
            sub = [r for r in phase_rows if r["gamma_t"] == gamma and abs(r["H0_bg"] - 70.5) < 1e-9]
            if sub: ax.plot([r["epsilon_today_bg"] for r in sub], [r["H0_apparent_if_global"] for r in sub], marker="o", label=f"gamma={gamma}")
        ax.axhline(args.H0_local_target, color="black", ls="--", lw=1); ax.set_xlabel("epsilon_today if used globally"); ax.set_ylabel("H0 apparent from dt_ratio")
        ax.set_title("Phase-time global overshoot check at H0_bg=70.5"); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(plotdir / "phase_time_global_overshoot.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5)); ax.plot([r["H0"] for r in cmb_rows], [r["theta_frac_diff_vs_67p4_percent"] for r in cmb_rows], marker="o")
        ax.axhline(0.15, color="red", ls="--", lw=1); ax.axhline(-0.15, color="red", ls="--", lw=1)
        ax.set_xlabel("global H0"); ax.set_ylabel("theta_star proxy shift [%]"); ax.set_title("CMB theta proxy warning"); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(plotdir / "cmb_theta_proxy_shift.png", dpi=180); plt.close(fig)
    except Exception as e:
        print("[WARN] plotting failed:", e)
    verdict = []
    if best_bg:
        verdict.append(f"Best background in this grid: {best_bg['model']} with H0={best_bg['H0']}, chi2={best_bg['chi2_total']:.3f}.")
        if lcdm_best:
            verdict.append(f"Delta chi2 vs best LCDM: {best_bg['chi2_total'] - lcdm_best['chi2_total']:.3f}.")
            verdict.append(f"Delta AIC vs best LCDM: {best_bg['aic'] - lcdm_best['aic']:.3f}; Delta BIC: {best_bg['bic'] - lcdm_best['bic']:.3f}.")
    if beta_source:
        verdict.append(f"Calibrator beta source: {beta_source['model']}, beta_cal={beta_source['beta_cal']:.5f} +/- {beta_source['beta_cal_err']:.5f}.")
        near = [r for r in sign_rows if r["mapping"] == "DeltaM=+beta" and abs(r["H0_error"]) <= 1.0]
        if near:
            n0 = sorted(near, key=lambda r: abs(r["H0_error"]))[0]
            verdict.append(f"If beta maps to SH0ES DeltaM, H0_bg={n0['H0_bg']} maps to H0_local={n0['H0_from_beta']:.2f}.")
        else:
            verdict.append("Calibrator beta does not map any scanned H0_bg to local target within 1 km/s/Mpc.")
    else:
        verdict.append("No calibrator beta test possible: missing calibrator flag or usable SN calibration data.")
    phase70 = [r for r in phase_rows if abs(r["H0_bg"] - 70.5) < 1e-9 and abs(r["gamma_t"] + 0.7) < 1e-9]
    if phase70:
        eps05 = next((r for r in phase70 if abs(r["epsilon_today_bg"] - 0.05) < 1e-9), None)
        eps15 = next((r for r in phase70 if abs(r["epsilon_today_bg"] - 0.15) < 1e-9), None)
        if eps05: verdict.append(f"Phase-time with gamma=-0.7 and epsilon=0.05 maps H0_bg=70.5 to H0={eps05['H0_apparent_if_global']:.2f}.")
        if eps15: verdict.append(f"Phase-time with gamma=-0.7 and epsilon=0.15 maps H0_bg=70.5 to H0={eps15['H0_apparent_if_global']:.2f}; this overshoots if used globally.")
    unsafe = [r for r in cmb_rows if abs(r["H0"] - 70.5) < 1e-9 and not r["CMB_theta_proxy_safe_0p15pct"]]
    if unsafe: verdict.append(f"CMB proxy warning: global H0=70.5 shifts theta_star by {unsafe[0]['theta_frac_diff_vs_67p4_percent']:.2f}%, not CMB-safe in this proxy.")
    dm_pos = "DRND still contains an effective matter sector Am. It can be interpreted as relational mass/connectivity or graph memory, but it is not eliminated. Halo rotation, lensing and growth tests remain decisive."
    recommended = "Do not claim v5.0 solves BBN+DESI+H0 yet. The clean live hypothesis is narrower: phase-time can naturally explain a local calibrator DeltaM ~0.08 mag, while BBN is protected because epsilon(a_BBN) is approximately zero. The unresolved issue is obtaining a CMB-safe global background near H0~70.5, or accepting global H0~67.4 with only partial H0 relief."
    master = {"model": "DRND master theory audit v2", "inputs": {"hz_csv": args.hz_csv, "sn_csv": sn_csv, "bao_csv": args.bao_csv,
              "n_hz": int(len(hz["z"])), "n_sn": int(len(sn["z"])), "n_bao": int(len(bao)),
              "has_calibrator_flag": get_calibrator_flags(sn) is not None, "has_host_mass": get_host_mass(sn) is not None},
              "best_background": {k: v for k, v in best_bg.items() if not k.startswith("_")} if best_bg else None,
              "best_lcdm": {k: v for k, v in lcdm_best.items() if not k.startswith("_")} if lcdm_best else None,
              "calibrator_beta_source": beta_source,
              "best_SH0ES_mapping_rows": sorted(sign_rows, key=lambda r: abs(r["H0_error"]))[:10] if sign_rows else [],
              "phase_time_key_rows": [r for r in phase_rows if abs(r["H0_bg"] - 70.5) < 1e-9 and r["gamma_t"] in (-0.7, -0.5, -0.25)][:50],
              "cmb_proxy_rows": cmb_rows, "sound_horizon_requirement": sound_rows, "best_growth_toy_rows": growth_rows[:20],
              "dark_matter_position": dm_pos, "verdict": verdict, "recommended_next_step": recommended,
              "cautions": ["Consolidated diagnostic, not final likelihood.", "CMB proxy is not CLASS/CAMB.", "BBN test is only epsilon(a_BBN)->0, not a BBN code.", "Growth toy is not a weak-lensing/RSD likelihood.", "Calibrator flag can encode sample construction effects.", "Host/calibrator interpretation requires real SH0ES-like hierarchy."]}
    (outdir / "MASTER_v2_summary.json").write_text(json.dumps(safe_json(master), indent=2), encoding="utf-8")
    report = ["# DRND Master Audit v2", "", "## Verdict"]
    report += [f"- {v}" for v in verdict]
    report.append(f"- {dm_pos}")
    report += ["", "## Interpretation", recommended, "", "## Key files", "- tables/background_fit_table.csv", "- tables/SH0ES_sign_table.csv", "- tables/phase_time_consistency_table.csv", "- tables/cmb_proxy_table.csv", "- tables/sound_horizon_requirement.csv", "- tables/growth_toy_table.csv", "", "## Bottom line", "Phase-time is plausible as a local calibrator/growth/memory mechanism. It is not yet a global background solution unless it passes the CMB-safe H0~70.5 problem."]
    (outdir / "MASTER_v2_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(safe_json({"outdir": str(outdir), "best_background": master["best_background"], "calibrator_beta_source": beta_source, "verdict": verdict, "files": ["MASTER_v2_report.md", "MASTER_v2_summary.json", "tables/background_fit_table.csv", "tables/calibrator_fit_table.csv", "tables/SH0ES_sign_table.csv", "tables/phase_time_consistency_table.csv", "tables/cmb_proxy_table.csv", "tables/sound_horizon_requirement.csv", "tables/growth_toy_table.csv", "plots/top_background_chi2.png", "plots/H0_from_calibrator_beta.png", "plots/phase_time_global_overshoot.png", "plots/cmb_theta_proxy_shift.png"]}), indent=2))
    print(f"[OK] wrote consolidated audit v2 to {outdir}")


if __name__ == "__main__":
    main()

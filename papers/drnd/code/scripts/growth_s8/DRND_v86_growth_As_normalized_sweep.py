#!/usr/bin/env python3
"""
DRND_v86_growth_As_normalized_sweep.py

Purpose
-------
Next rigorous growth test after v8.5.

v8.5 showed that dynamic DRND has the right qualitative history:
    early growth close to LCDM, late-time suppression.

But v8.5 treated sigma8_0 as an external input, so S8 was identical across
lcdm/static/dynamic if Omega_cl0 was identical.

v8.6 fixes that by normalizing all models to the SAME primordial amplitude,
implemented as:

    sigma8_LCDM(today) = sigma8_lcdm_ref

Then:

    sigma8_model(today)
      = sigma8_lcdm_ref * growth_model_from_aini_to_today / growth_lcdm_from_aini_to_today

and:

    S8_model = sigma8_model(today) * sqrt(Omega_cl0 / 0.3)

This is not a Boltzmann-code replacement, but it tests whether the integrated
growth history can lower S8 from a common early amplitude.

Dynamic DRND:
    epsilon(a) = epsilon_inf * a/(a+a_c)
    epsilon_today = epsilon_inf/(1+a_c)
    d_f(a) = 3 - epsilon(a)
    mu(a) = -lambda_mu * epsilon(a)
    G_eff(a,k) = 1 + mu(a) T(a,k)

Graph mass:
    p_m(a)=3-epsilon(a)
    R_m(a)=exp[-int_{ln1}^{lna} p_m(a') dln a']

Background:
    E^2(a)=Omega_r a^-4 + Omega_b a^-3 + A_m R_m(a) + A_x
    A_x=1-Omega_r-Omega_b-A_m unless --A-x supplied.

Modes
-----
1. single:
   Evaluate one parameter point.

2. sweep:
   Grid over Q_m, lambda_mu, a_c and optionally epsilon_today.
   Rank points by closeness to target S8 and f_sigma8(z=0).

Outputs
-------
    v86_single_summary.json
    v86_sweep_results.csv
    v86_best_points.json
    v86_sweep_S8_hist.png
    v86_sweep_Qm_lambda.png
    v86_best_growth_curves.png

Usage
-----
Single Golden Point:

    python DRND_v86_growth_As_normalized_sweep.py ^
      --mode single ^
      --epsilon-today 0.135 ^
      --a-c 0.5 ^
      --lambda-mu 2.5 ^
      --Q-m 0.75 ^
      --sigma8-lcdm-ref 0.81 ^
      --target-S8 0.78 ^
      --outdir results_v86_single

Sweep:

    python DRND_v86_growth_As_normalized_sweep.py ^
      --mode sweep ^
      --epsilon-grid 0.08 0.11 0.135 0.16 0.20 ^
      --a-c-grid 0.2 0.5 1.0 2.0 ^
      --lambda-grid 1.0 1.5 2.0 2.5 3.0 3.5 ^
      --Qm-grid 0.45 0.55 0.65 0.75 0.85 1.0 ^
      --sigma8-lcdm-ref 0.81 ^
      --target-S8 0.78 ^
      --target-fs8-today 0.40 ^
      --outdir results_v86_sweep

Important
---------
Compressed audit only:
    - no transfer functions,
    - no scale-dependent sigma8 window,
    - no CMB normalization As through CLASS/CAMB,
    - no survey covariance.

Still useful because it tests the main question:
    Can dynamic DRND lower S8 from a common primordial amplitude?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, solve_ivp


def omega_r_from_h(H0: float, omega_gamma_h2: float = 2.469e-5, n_eff: float = 3.046) -> float:
    h = H0 / 100.0
    return omega_gamma_h2 * (1.0 + 0.2271 * n_eff) / (h*h)


def epsilon_dynamic(a, epsilon_today: float, a_c: float):
    a = np.asarray(a, dtype=float)
    eps_inf = epsilon_today * (1.0 + a_c)
    return eps_inf * a / (a + a_c)


def epsilon_zero(a, epsilon_today: float, a_c: float):
    return np.zeros_like(np.asarray(a, dtype=float))


def make_params(args, model: str, epsilon_today=None, a_c=None, lambda_mu=None, Q_m=None) -> Dict[str, float]:
    eps = args.epsilon_today if epsilon_today is None else float(epsilon_today)
    ac = args.a_c if a_c is None else float(a_c)
    lam = args.lambda_mu if lambda_mu is None else float(lambda_mu)
    qm = args.Q_m if Q_m is None else float(Q_m)

    p = {
        "model": model,
        "H0": float(args.H0),
        "Omega_b": float(args.Omega_b),
        "A_m": float(args.A_m),
        "Q_m": qm,
        "lambda_mu": lam,
        "epsilon_today": eps,
        "a_c": ac,
        "k_eff": float(args.k_eff),
        "Omega_r": omega_r_from_h(float(args.H0)),
    }
    p["A_x"] = float(args.A_x) if args.A_x is not None else 1.0 - p["Omega_r"] - p["Omega_b"] - p["A_m"]
    return p


def epsilon_a(a, p):
    if p["model"] == "lcdm":
        return epsilon_zero(a, 0.0, p["a_c"])
    return epsilon_dynamic(a, p["epsilon_today"], p["a_c"])


def Rm_grid(p, a_grid):
    a_grid = np.asarray(a_grid, dtype=float)
    x = np.log(a_grid)
    pm = 3.0 - epsilon_a(a_grid, p)
    integ = cumulative_trapezoid(pm, x, initial=0.0)
    lnR = integ[-1] - integ
    return np.exp(lnR)


def Rm_a(a, p):
    a = np.asarray(a, dtype=float)
    amin = max(min(float(np.min(a)), 1e-4), 1e-6)
    grid = np.geomspace(amin, 1.0, 1200)
    return np.interp(a, grid, Rm_grid(p, grid))


def E2_a(a, p):
    a = np.asarray(a, dtype=float)
    return (
        p["Omega_r"] * a**-4
        + p["Omega_b"] * a**-3
        + p["A_m"] * Rm_a(a, p)
        + p["A_x"]
    )


def dlnH_dlna(a, p):
    x = np.log(float(a))
    h = 1e-4
    e1 = E2_a(np.exp(x - h), p)
    e2 = E2_a(np.exp(x + h), p)
    return float(0.5 * (np.log(e2) - np.log(e1)) / (2*h))


def omega_cl_a(a, p):
    a = np.asarray(a, dtype=float)
    src = p["Omega_b"] * a**-3 + p["Q_m"] * p["A_m"] * Rm_a(a, p)
    return src / E2_a(a, p)


def T_kernel(a, k_eff):
    a = np.asarray(a, dtype=float)
    a_t = 0.5
    s = 4.0
    k0 = 0.12
    late = a**s / (a**s + a_t**s)
    x = k_eff / (k0 / np.maximum(a, 1e-8))
    return late * 4.0 * x**4 / (1.0 + x**4)**2


def mu_a(a, p):
    if p["model"] == "lcdm":
        return np.zeros_like(np.asarray(a, dtype=float))
    return -p["lambda_mu"] * epsilon_a(a, p)


def Geff_a(a, p):
    return 1.0 + mu_a(a, p) * T_kernel(a, p["k_eff"])


def solve_growth(p, z_eval, a_ini=1e-3):
    z_eval = np.asarray(z_eval, dtype=float)
    a_eval = 1.0 / (1.0 + z_eval)
    a_ini = min(a_ini, float(np.min(a_eval)) * 0.5)
    a_ini = max(a_ini, 1e-5)
    x_ini = np.log(a_ini)

    def rhs(x, y):
        a = float(np.exp(x))
        D, V = y
        friction = 2.0 + dlnH_dlna(a, p)
        source = 1.5 * float(Geff_a(a, p)) * float(omega_cl_a(a, p))
        return [V, -friction * V + source * D]

    sol = solve_ivp(
        rhs,
        (x_ini, 0.0),
        [a_ini, a_ini],
        dense_output=True,
        rtol=5e-7,
        atol=1e-9,
        max_step=0.02,
    )
    if not sol.success:
        raise RuntimeError(f"growth solve failed: {sol.message}")

    D0, V0 = sol.sol(0.0)
    D_eval, V_eval = sol.sol(np.log(a_eval))
    D_norm = D_eval / D0
    V_norm = V_eval / D0
    f = V_norm / D_norm
    return {
        "z": z_eval,
        "a": a_eval,
        "D_norm": D_norm,
        "f": f,
        "D0_raw": float(D0),
        "growth_from_ini_to_today": float(D0 / a_ini),
    }


def evaluate_point(args, epsilon_today, a_c, lambda_mu, Q_m, z_eval):
    p_lcdm = make_params(args, "lcdm", epsilon_today=0.0, a_c=a_c, lambda_mu=0.0, Q_m=Q_m)
    p_dyn = make_params(args, "dynamic", epsilon_today=epsilon_today, a_c=a_c, lambda_mu=lambda_mu, Q_m=Q_m)

    g_lcdm = solve_growth(p_lcdm, z_eval)
    g_dyn = solve_growth(p_dyn, z_eval)

    growth_ratio = g_dyn["growth_from_ini_to_today"] / g_lcdm["growth_from_ini_to_today"]
    sigma8_dyn = args.sigma8_lcdm_ref * growth_ratio
    Omega_cl0 = p_dyn["Omega_b"] + p_dyn["Q_m"] * p_dyn["A_m"]
    S8_dyn = sigma8_dyn * np.sqrt(Omega_cl0 / 0.3)
    fs8_dyn = g_dyn["f"] * g_dyn["D_norm"] * sigma8_dyn

    sigma8_lcdm = args.sigma8_lcdm_ref
    Omega_cl0_lcdm = p_lcdm["Omega_b"] + p_lcdm["Q_m"] * p_lcdm["A_m"]
    S8_lcdm = sigma8_lcdm * np.sqrt(Omega_cl0_lcdm / 0.3)
    fs8_lcdm = g_lcdm["f"] * g_lcdm["D_norm"] * sigma8_lcdm

    score = ((S8_dyn - args.target_S8) / args.target_S8_sigma)**2
    if args.target_fs8_today is not None:
        score += ((fs8_dyn[0] - args.target_fs8_today) / args.target_fs8_sigma)**2

    return {
        "params_dynamic": p_dyn,
        "params_lcdm": p_lcdm,
        "growth_dynamic": g_dyn,
        "growth_lcdm": g_lcdm,
        "growth_ratio": float(growth_ratio),
        "sigma8_dynamic_As_norm": float(sigma8_dyn),
        "sigma8_lcdm_ref": float(sigma8_lcdm),
        "Omega_cl0_dynamic": float(Omega_cl0),
        "Omega_cl0_lcdm": float(Omega_cl0_lcdm),
        "S8_dynamic": float(S8_dyn),
        "S8_lcdm": float(S8_lcdm),
        "fs8_dynamic": fs8_dyn,
        "fs8_lcdm": fs8_lcdm,
        "score": float(score),
        "f_today_dynamic": float(g_dyn["f"][0]),
        "f_today_lcdm": float(g_lcdm["f"][0]),
        "fs8_today_dynamic": float(fs8_dyn[0]),
        "fs8_today_lcdm": float(fs8_lcdm[0]),
    }


def write_sweep_csv(path: Path, rows: List[Dict]):
    fields = [
        "score",
        "epsilon_today",
        "df_today",
        "a_c",
        "lambda_mu",
        "mu_today",
        "Q_m",
        "Omega_cl0",
        "growth_ratio",
        "sigma8_dynamic_As_norm",
        "S8_dynamic",
        "S8_lcdm",
        "fs8_today_dynamic",
        "fs8_today_lcdm",
        "f_today_dynamic",
        "Geff_today",
        "A_x",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def plot_sweep(outdir: Path, rows: List[Dict], target_S8: float):
    S8 = np.array([r["S8_dynamic"] for r in rows])
    score = np.array([r["score"] for r in rows])
    Qm = np.array([r["Q_m"] for r in rows])
    lam = np.array([r["lambda_mu"] for r in rows])
    eps = np.array([r["epsilon_today"] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(S8, bins=40)
    ax.axvline(target_S8, ls="--", lw=2, label=f"target S8={target_S8:g}")
    ax.set_xlabel("S8 dynamic, As-normalized")
    ax.set_ylabel("count")
    ax.set_title("DRND v8.6 sweep: S8 distribution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "v86_sweep_S8_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(Qm, lam, c=S8, s=45, alpha=0.8)
    ax.set_xlabel("Q_m")
    ax.set_ylabel("lambda_mu")
    ax.set_title("DRND v8.6 sweep: S8 over Q_m-lambda plane")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("S8 dynamic")
    fig.tight_layout()
    fig.savefig(outdir / "v86_sweep_Qm_lambda.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(eps, S8, c=score, s=45, alpha=0.8)
    ax.axhline(target_S8, ls="--", lw=2)
    ax.set_xlabel("epsilon_today")
    ax.set_ylabel("S8 dynamic")
    ax.set_title("DRND v8.6 sweep: epsilon vs S8")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("score")
    fig.tight_layout()
    fig.savefig(outdir / "v86_sweep_epsilon_S8.png", dpi=180)
    plt.close(fig)


def plot_best_curves(outdir: Path, best_eval: Dict):
    z = best_eval["growth_dynamic"]["z"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(z, best_eval["fs8_lcdm"], lw=2.3, label="LCDM As-normalized")
    ax.plot(z, best_eval["fs8_dynamic"], lw=2.3, label="Dynamic DRND As-normalized")
    ax.set_xlabel("z")
    ax.set_ylabel("f sigma8(z)")
    ax.set_title("DRND v8.6 best point: f sigma8 history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "v86_best_growth_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(z, best_eval["growth_lcdm"]["D_norm"], lw=2.3, label="LCDM")
    ax.plot(z, best_eval["growth_dynamic"]["D_norm"], lw=2.3, label="Dynamic DRND")
    ax.set_xlabel("z")
    ax.set_ylabel("D(z), normalized to D(0)=1")
    ax.set_title("DRND v8.6 best point: normalized D(z)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "v86_best_D_curves.png", dpi=180)
    plt.close(fig)


def run_single(args, outdir, z_eval):
    ev = evaluate_point(args, args.epsilon_today, args.a_c, args.lambda_mu, args.Q_m, z_eval)
    p = ev["params_dynamic"]
    summary = {
        "model": "DRND v8.6 As-normalized growth single test",
        "input": {
            "epsilon_today": args.epsilon_today,
            "df_today": 3.0 - args.epsilon_today,
            "a_c": args.a_c,
            "lambda_mu": args.lambda_mu,
            "mu_today": -args.lambda_mu * args.epsilon_today,
            "Q_m": args.Q_m,
            "H0": args.H0,
            "Omega_b": args.Omega_b,
            "A_m": args.A_m,
            "A_x": p["A_x"],
            "sigma8_lcdm_ref": args.sigma8_lcdm_ref,
            "target_S8": args.target_S8,
        },
        "result": {
            "growth_ratio_dynamic_over_lcdm": ev["growth_ratio"],
            "sigma8_dynamic_As_norm": ev["sigma8_dynamic_As_norm"],
            "S8_dynamic": ev["S8_dynamic"],
            "S8_lcdm": ev["S8_lcdm"],
            "Omega_cl0_dynamic": ev["Omega_cl0_dynamic"],
            "f_today_dynamic": ev["f_today_dynamic"],
            "fs8_today_dynamic": ev["fs8_today_dynamic"],
            "f_today_lcdm": ev["f_today_lcdm"],
            "fs8_today_lcdm": ev["fs8_today_lcdm"],
            "Geff_today": float(Geff_a(1.0, p)),
            "score": ev["score"],
        },
        "warning": "Compressed As-normalized growth audit; not CLASS/CAMB.",
    }
    with open(outdir / "v86_single_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_best_curves(outdir, ev)
    print(json.dumps(summary, indent=2))


def run_sweep(args, outdir, z_eval):
    rows = []
    best_eval = None
    best_tuple = None

    total = len(args.epsilon_grid) * len(args.a_c_grid) * len(args.lambda_grid) * len(args.Qm_grid)
    count = 0
    print(f"[INFO] Sweep points: {total}")

    for eps in args.epsilon_grid:
        for ac in args.a_c_grid:
            for lam in args.lambda_grid:
                for qm in args.Qm_grid:
                    count += 1
                    try:
                        ev = evaluate_point(args, eps, ac, lam, qm, z_eval)
                        p = ev["params_dynamic"]
                        row = {
                            "score": ev["score"],
                            "epsilon_today": float(eps),
                            "df_today": float(3.0 - eps),
                            "a_c": float(ac),
                            "lambda_mu": float(lam),
                            "mu_today": float(-lam * eps),
                            "Q_m": float(qm),
                            "Omega_cl0": ev["Omega_cl0_dynamic"],
                            "growth_ratio": ev["growth_ratio"],
                            "sigma8_dynamic_As_norm": ev["sigma8_dynamic_As_norm"],
                            "S8_dynamic": ev["S8_dynamic"],
                            "S8_lcdm": ev["S8_lcdm"],
                            "fs8_today_dynamic": ev["fs8_today_dynamic"],
                            "fs8_today_lcdm": ev["fs8_today_lcdm"],
                            "f_today_dynamic": ev["f_today_dynamic"],
                            "Geff_today": float(Geff_a(1.0, p)),
                            "A_x": float(p["A_x"]),
                        }
                        rows.append(row)
                        if best_tuple is None or row["score"] < best_tuple["score"]:
                            best_tuple = row
                            best_eval = ev
                    except Exception as exc:
                        print(f"[WARN] failed eps={eps} ac={ac} lam={lam} Qm={qm}: {exc}")

    rows.sort(key=lambda r: r["score"])
    write_sweep_csv(outdir / "v86_sweep_results.csv", rows)
    plot_sweep(outdir, rows, args.target_S8)

    if best_eval is not None:
        plot_best_curves(outdir, best_eval)

    best_points = {
        "model": "DRND v8.6 As-normalized growth sweep",
        "n_success": len(rows),
        "n_total": total,
        "top_20": rows[:20],
        "warning": "Compressed As-normalized growth audit; not CLASS/CAMB.",
    }
    with open(outdir / "v86_best_points.json", "w", encoding="utf-8") as f:
        json.dump(best_points, f, indent=2)

    print(json.dumps(best_points, indent=2))
    print(f"[OK] best score: {rows[0]['score'] if rows else 'NA'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "sweep"], default="single")

    # Cosmology / DRND defaults
    ap.add_argument("--epsilon-today", type=float, default=0.135)
    ap.add_argument("--a-c", type=float, default=0.5)
    ap.add_argument("--lambda-mu", type=float, default=2.5)
    ap.add_argument("--Q-m", type=float, default=0.75)
    ap.add_argument("--H0", type=float, default=72.2)
    ap.add_argument("--Omega-b", type=float, default=0.049)
    ap.add_argument("--A-m", type=float, default=0.40)
    ap.add_argument("--A-x", type=float, default=None)
    ap.add_argument("--k-eff", type=float, default=0.15)

    # As-normalized amplitude
    ap.add_argument("--sigma8-lcdm-ref", type=float, default=0.81)
    ap.add_argument("--target-S8", type=float, default=0.78)
    ap.add_argument("--target-S8-sigma", type=float, default=0.02)
    ap.add_argument("--target-fs8-today", type=float, default=None)
    ap.add_argument("--target-fs8-sigma", type=float, default=0.04)

    # Sweep grids
    ap.add_argument("--epsilon-grid", type=float, nargs="+", default=[0.08, 0.11, 0.135, 0.16, 0.20])
    ap.add_argument("--a-c-grid", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0])
    ap.add_argument("--lambda-grid", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--Qm-grid", type=float, nargs="+", default=[0.45, 0.55, 0.65, 0.75, 0.85, 1.0])

    # Output
    ap.add_argument("--z-max", type=float, default=10.0)
    ap.add_argument("--n-z", type=int, default=350)
    ap.add_argument("--outdir", default="results_v86_growth_As_norm")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z_low = np.linspace(0.0, 2.0, int(args.n_z * 0.65))
    z_high = np.geomspace(2.001, args.z_max, max(5, args.n_z - len(z_low)))
    z_eval = np.unique(np.concatenate([z_low, z_high]))

    if args.mode == "single":
        run_single(args, outdir, z_eval)
    else:
        run_sweep(args, outdir, z_eval)

    print(f"[OK] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()

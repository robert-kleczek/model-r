#!/usr/bin/env python3
"""
DRND_v86_growth_As_normalized_sweep_FAST.py

Fast optimized version of DRND_v86_growth_As_normalized_sweep.py.

Main optimizations versus v86:
1. Cached background/growth interpolation grids.
   The old script recomputed R_m(a) grids many times inside the ODE RHS.
   This version precomputes epsilon(a), R_m(a), E2(a), dlnH/dlna on a log-a grid.

2. LCDM growth is cached per Q_m.
   In the sweep, LCDM does not depend on epsilon_today, a_c or lambda_mu.
   It only depends on Q_m through the clustering source term. The old script
   solved the LCDM ODE for every grid point. This version solves it once per Q_m.

3. Progress counter.
   Prints [i/total], percent, elapsed time, ETA and current parameter point.

4. Optional coarse-to-fine mode.
   You can run a quick scan first with fewer redshift points and then refine.

Usage
-----
Fast sweep equivalent to previous one:

    python DRND_v86_growth_As_normalized_sweep_FAST.py ^
      --mode sweep ^
      --epsilon-grid 0.08 0.11 0.135 0.16 0.20 ^
      --a-c-grid 0.2 0.5 1.0 2.0 ^
      --lambda-grid 1.0 1.5 2.0 2.5 3.0 3.5 ^
      --Qm-grid 0.45 0.55 0.65 0.75 0.85 1.0 ^
      --sigma8-lcdm-ref 0.81 ^
      --target-S8 0.78 ^
      --target-fs8-today 0.40 ^
      --outdir results_v86_sweep_fast ^
      --n-z 180 ^
      --progress-every 10

Even faster exploratory scan:

    python DRND_v86_growth_As_normalized_sweep_FAST.py ^
      --mode sweep ^
      --epsilon-grid 0.10 0.135 0.16 ^
      --a-c-grid 0.3 0.7 1.5 ^
      --lambda-grid 2.0 2.5 3.0 ^
      --Qm-grid 0.45 0.55 0.65 ^
      --n-z 120 ^
      --cache-n 900 ^
      --outdir results_v86_sweep_quick

Outputs
-------
    v86_fast_sweep_results.csv
    v86_fast_best_points.json
    v86_fast_sweep_S8_hist.png
    v86_fast_sweep_Qm_lambda.png
    v86_fast_sweep_epsilon_S8.png
    v86_fast_best_growth_curves.png
    v86_fast_best_D_curves.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, solve_ivp


def omega_r_from_h(H0: float, omega_gamma_h2: float = 2.469e-5, n_eff: float = 3.046) -> float:
    h = H0 / 100.0
    return omega_gamma_h2 * (1.0 + 0.2271 * n_eff) / (h * h)


@dataclass(frozen=True)
class BaseArgs:
    H0: float
    Omega_b: float
    A_m: float
    A_x: Optional[float]
    k_eff: float
    cache_n: int


@dataclass
class GrowthCache:
    model: str
    epsilon_today: float
    a_c: float
    lambda_mu: float
    Q_m: float
    H0: float
    Omega_b: float
    A_m: float
    A_x: float
    Omega_r: float
    k_eff: float
    loga_grid: np.ndarray
    a_grid: np.ndarray
    eps_grid: np.ndarray
    Rm_grid: np.ndarray
    E2_grid: np.ndarray
    dlnH_grid: np.ndarray
    Omega_cl_grid: np.ndarray
    Geff_grid: np.ndarray

    def interp(self, arr: np.ndarray, a: float) -> float:
        la = math.log(max(float(a), float(self.a_grid[0])))
        return float(np.interp(la, self.loga_grid, arr))

    def E2(self, a: float) -> float:
        return self.interp(self.E2_grid, a)

    def dlnH(self, a: float) -> float:
        return self.interp(self.dlnH_grid, a)

    def omega_cl(self, a: float) -> float:
        return self.interp(self.Omega_cl_grid, a)

    def Geff(self, a: float) -> float:
        return self.interp(self.Geff_grid, a)


def epsilon_grid(model: str, a_grid: np.ndarray, epsilon_today: float, a_c: float) -> np.ndarray:
    if model == "lcdm":
        return np.zeros_like(a_grid)
    eps_inf = epsilon_today * (1.0 + a_c)
    return eps_inf * a_grid / (a_grid + a_c)


def T_kernel_grid(a: np.ndarray, k_eff: float) -> np.ndarray:
    a_t = 0.5
    s = 4.0
    k0 = 0.12
    late = a**s / (a**s + a_t**s)
    x = k_eff / (k0 / np.maximum(a, 1e-8))
    return late * 4.0 * x**4 / (1.0 + x**4)**2


def make_cache(
    model: str,
    epsilon_today: float,
    a_c: float,
    lambda_mu: float,
    Q_m: float,
    args: BaseArgs,
    a_min: float = 1e-5,
) -> GrowthCache:
    Omega_r = omega_r_from_h(args.H0)
    A_x = args.A_x if args.A_x is not None else 1.0 - Omega_r - args.Omega_b - args.A_m

    loga = np.linspace(math.log(a_min), 0.0, args.cache_n)
    a = np.exp(loga)
    eps = epsilon_grid(model, a, epsilon_today, a_c)

    pm = 3.0 - eps
    integ = cumulative_trapezoid(pm, loga, initial=0.0)
    lnR = integ[-1] - integ
    Rm = np.exp(lnR)

    E2 = Omega_r * a**-4 + args.Omega_b * a**-3 + args.A_m * Rm + A_x
    if np.any(E2 <= 0) or np.any(~np.isfinite(E2)):
        raise ValueError("non-positive E2 in cache")

    # dlnH/dlna = 0.5 dlnE2/dlna
    dlnE2 = np.gradient(np.log(E2), loga, edge_order=2)
    dlnH = 0.5 * dlnE2

    Omega_cl = (args.Omega_b * a**-3 + Q_m * args.A_m * Rm) / E2

    if model == "lcdm":
        mu = np.zeros_like(a)
    else:
        mu = -lambda_mu * eps
    Geff = 1.0 + mu * T_kernel_grid(a, args.k_eff)

    return GrowthCache(
        model=model,
        epsilon_today=epsilon_today,
        a_c=a_c,
        lambda_mu=lambda_mu,
        Q_m=Q_m,
        H0=args.H0,
        Omega_b=args.Omega_b,
        A_m=args.A_m,
        A_x=A_x,
        Omega_r=Omega_r,
        k_eff=args.k_eff,
        loga_grid=loga,
        a_grid=a,
        eps_grid=eps,
        Rm_grid=Rm,
        E2_grid=E2,
        dlnH_grid=dlnH,
        Omega_cl_grid=Omega_cl,
        Geff_grid=Geff,
    )


def solve_growth_cached(cache: GrowthCache, z_eval: np.ndarray, a_ini: float = 1e-3) -> Dict:
    z_eval = np.asarray(z_eval, dtype=float)
    a_eval = 1.0 / (1.0 + z_eval)
    a_ini = min(a_ini, float(np.min(a_eval)) * 0.5)
    a_ini = max(a_ini, float(cache.a_grid[0]))
    x_ini = math.log(a_ini)

    def rhs(x, y):
        a = math.exp(x)
        D, V = y
        friction = 2.0 + cache.dlnH(a)
        source = 1.5 * cache.Geff(a) * cache.omega_cl(a)
        return [V, -friction * V + source * D]

    sol = solve_ivp(
        rhs,
        (x_ini, 0.0),
        [a_ini, a_ini],
        dense_output=True,
        rtol=5e-7,
        atol=1e-9,
        max_step=0.025,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

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


def evaluate_dynamic_against_lcdm(
    args,
    base_args: BaseArgs,
    z_eval: np.ndarray,
    lcdm_by_qm: Dict[float, Tuple[GrowthCache, Dict]],
    epsilon_today: float,
    a_c: float,
    lambda_mu: float,
    Q_m: float,
) -> Dict:
    if Q_m not in lcdm_by_qm:
        lcdm_cache = make_cache("lcdm", 0.0, a_c, 0.0, Q_m, base_args)
        lcdm_growth = solve_growth_cached(lcdm_cache, z_eval)
        lcdm_by_qm[Q_m] = (lcdm_cache, lcdm_growth)
    else:
        lcdm_cache, lcdm_growth = lcdm_by_qm[Q_m]

    dyn_cache = make_cache("dynamic", epsilon_today, a_c, lambda_mu, Q_m, base_args)
    dyn_growth = solve_growth_cached(dyn_cache, z_eval)

    growth_ratio = dyn_growth["growth_from_ini_to_today"] / lcdm_growth["growth_from_ini_to_today"]
    sigma8_dyn = args.sigma8_lcdm_ref * growth_ratio

    Omega_cl0_dyn = dyn_cache.Omega_b + dyn_cache.Q_m * dyn_cache.A_m
    Omega_cl0_lcdm = lcdm_cache.Omega_b + lcdm_cache.Q_m * lcdm_cache.A_m

    S8_dyn = sigma8_dyn * math.sqrt(Omega_cl0_dyn / 0.3)
    S8_lcdm = args.sigma8_lcdm_ref * math.sqrt(Omega_cl0_lcdm / 0.3)

    fs8_dyn = dyn_growth["f"] * dyn_growth["D_norm"] * sigma8_dyn
    fs8_lcdm = lcdm_growth["f"] * lcdm_growth["D_norm"] * args.sigma8_lcdm_ref

    score = ((S8_dyn - args.target_S8) / args.target_S8_sigma) ** 2
    if args.target_fs8_today is not None:
        score += ((fs8_dyn[0] - args.target_fs8_today) / args.target_fs8_sigma) ** 2

    return {
        "score": float(score),
        "epsilon_today": float(epsilon_today),
        "df_today": float(3.0 - epsilon_today),
        "a_c": float(a_c),
        "lambda_mu": float(lambda_mu),
        "mu_today": float(-lambda_mu * epsilon_today),
        "Q_m": float(Q_m),
        "Omega_cl0": float(Omega_cl0_dyn),
        "growth_ratio": float(growth_ratio),
        "sigma8_dynamic_As_norm": float(sigma8_dyn),
        "S8_dynamic": float(S8_dyn),
        "S8_lcdm": float(S8_lcdm),
        "fs8_today_dynamic": float(fs8_dyn[0]),
        "fs8_today_lcdm": float(fs8_lcdm[0]),
        "f_today_dynamic": float(dyn_growth["f"][0]),
        "f_today_lcdm": float(lcdm_growth["f"][0]),
        "Geff_today": float(dyn_cache.Geff(1.0)),
        "A_x": float(dyn_cache.A_x),
        "_dyn_cache": dyn_cache,
        "_dyn_growth": dyn_growth,
        "_lcdm_cache": lcdm_cache,
        "_lcdm_growth": lcdm_growth,
        "_fs8_dyn": fs8_dyn,
        "_fs8_lcdm": fs8_lcdm,
    }


def clean_row(row: Dict) -> Dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def make_z_eval(n_z: int, z_max: float) -> np.ndarray:
    z_low = np.linspace(0.0, min(2.0, z_max), int(n_z * 0.65))
    if z_max > 2.0:
        z_high = np.geomspace(2.001, z_max, max(5, n_z - len(z_low)))
        z = np.unique(np.concatenate([z_low, z_high]))
    else:
        z = np.unique(z_low)
    return z


def format_seconds(s: float) -> str:
    if not np.isfinite(s):
        return "NA"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


def write_sweep_csv(path: Path, rows: List[Dict]) -> None:
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
        "f_today_lcdm",
        "Geff_today",
        "A_x",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = clean_row(r)
            w.writerow({k: rr[k] for k in fields})


def plot_best_curves(outdir: Path, best: Dict) -> None:
    z = best["_dyn_growth"]["z"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(z, best["_lcdm_growth"]["D_norm"], lw=2.3, label="LCDM")
    ax.plot(z, best["_dyn_growth"]["D_norm"], lw=2.3, label="Dynamic DRND")
    ax.set_xlabel("z")
    ax.set_ylabel("D(z), normalized to D(0)=1")
    ax.set_title("DRND v8.6 FAST best point: normalized D(z)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "v86_fast_best_D_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(z, best["_fs8_lcdm"], lw=2.3, label="LCDM As-normalized")
    ax.plot(z, best["_fs8_dyn"], lw=2.3, label="Dynamic DRND As-normalized")
    ax.set_xlabel("z")
    ax.set_ylabel("f sigma8(z)")
    ax.set_title("DRND v8.6 FAST best point: f sigma8 history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "v86_fast_best_growth_curves.png", dpi=180)
    plt.close(fig)


def plot_sweep(outdir: Path, rows: List[Dict], target_S8: float) -> None:
    if not rows:
        return
    S8 = np.array([r["S8_dynamic"] for r in rows])
    Qm = np.array([r["Q_m"] for r in rows])
    lam = np.array([r["lambda_mu"] for r in rows])
    eps = np.array([r["epsilon_today"] for r in rows])
    score = np.array([r["score"] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(S8, bins=40)
    ax.axvline(target_S8, ls="--", lw=2, label=f"target S8={target_S8:g}")
    ax.set_xlabel("S8 dynamic, As-normalized")
    ax.set_ylabel("count")
    ax.set_title("DRND v8.6 FAST sweep: S8 distribution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "v86_fast_sweep_S8_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(Qm, lam, c=S8, s=45, alpha=0.8)
    ax.set_xlabel("Q_m")
    ax.set_ylabel("lambda_mu")
    ax.set_title("DRND v8.6 FAST sweep: S8 over Q_m-lambda plane")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("S8 dynamic")
    fig.tight_layout()
    fig.savefig(outdir / "v86_fast_sweep_Qm_lambda.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(eps, S8, c=score, s=45, alpha=0.8)
    ax.axhline(target_S8, ls="--", lw=2)
    ax.set_xlabel("epsilon_today")
    ax.set_ylabel("S8 dynamic")
    ax.set_title("DRND v8.6 FAST sweep: epsilon vs S8")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("score")
    fig.tight_layout()
    fig.savefig(outdir / "v86_fast_sweep_epsilon_S8.png", dpi=180)
    plt.close(fig)


def run_single(args, base_args: BaseArgs, outdir: Path, z_eval: np.ndarray) -> None:
    lcdm_by_qm = {}
    row = evaluate_dynamic_against_lcdm(
        args, base_args, z_eval, lcdm_by_qm,
        args.epsilon_today, args.a_c, args.lambda_mu, args.Q_m
    )
    summary = {
        "model": "DRND v8.6 FAST As-normalized growth single test",
        "result": clean_row(row),
        "input": {
            "epsilon_today": args.epsilon_today,
            "df_today": 3.0 - args.epsilon_today,
            "a_c": args.a_c,
            "lambda_mu": args.lambda_mu,
            "Q_m": args.Q_m,
            "sigma8_lcdm_ref": args.sigma8_lcdm_ref,
            "target_S8": args.target_S8,
            "target_fs8_today": args.target_fs8_today,
            "n_z": args.n_z,
            "cache_n": args.cache_n,
        },
        "warning": "Compressed As-normalized growth audit; not CLASS/CAMB.",
    }
    with open(outdir / "v86_fast_single_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_best_curves(outdir, row)
    print(json.dumps(summary, indent=2))


def run_sweep(args, base_args: BaseArgs, outdir: Path, z_eval: np.ndarray) -> None:
    combos = [
        (eps, ac, lam, qm)
        for eps in args.epsilon_grid
        for ac in args.a_c_grid
        for lam in args.lambda_grid
        for qm in args.Qm_grid
    ]
    total = len(combos)
    print(f"[INFO] Sweep points: {total}")
    print(f"[INFO] z points: {len(z_eval)}, cache_n: {args.cache_n}")
    print(f"[INFO] LCDM cache will solve at most {len(set(args.Qm_grid))} LCDM growth curves.")

    rows: List[Dict] = []
    lcdm_by_qm: Dict[float, Tuple[GrowthCache, Dict]] = {}
    best: Optional[Dict] = None
    t0 = time.time()

    for i, (eps, ac, lam, qm) in enumerate(combos, start=1):
        try:
            row = evaluate_dynamic_against_lcdm(args, base_args, z_eval, lcdm_by_qm, eps, ac, lam, qm)
            rows.append(row)
            if best is None or row["score"] < best["score"]:
                best = row
                print(
                    f"[BEST] i={i}/{total} score={row['score']:.4g} "
                    f"S8={row['S8_dynamic']:.4f} fs8_0={row['fs8_today_dynamic']:.4f} "
                    f"eps={eps:g} ac={ac:g} lam={lam:g} Qm={qm:g}"
                )
        except Exception as exc:
            print(f"[WARN] failed i={i}/{total} eps={eps} ac={ac} lam={lam} Qm={qm}: {exc}")

        if i == 1 or i % args.progress_every == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else float("nan")
            eta = (total - i) / rate if rate > 0 else float("nan")
            pct = 100.0 * i / total
            print(
                f"[PROGRESS] {i}/{total} ({pct:.1f}%) "
                f"elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)} "
                f"current eps={eps:g} ac={ac:g} lam={lam:g} Qm={qm:g}"
            )

    rows.sort(key=lambda r: r["score"])
    write_sweep_csv(outdir / "v86_fast_sweep_results.csv", rows)
    plot_sweep(outdir, rows, args.target_S8)

    if rows:
        best = rows[0]
        plot_best_curves(outdir, best)

    best_points = {
        "model": "DRND v8.6 FAST As-normalized growth sweep",
        "n_success": len(rows),
        "n_total": total,
        "top_20": [clean_row(r) for r in rows[:20]],
        "runtime_seconds": time.time() - t0,
        "lcdm_cached_Qm_values": sorted([float(x) for x in lcdm_by_qm.keys()]),
        "input": {
            "epsilon_grid": args.epsilon_grid,
            "a_c_grid": args.a_c_grid,
            "lambda_grid": args.lambda_grid,
            "Qm_grid": args.Qm_grid,
            "sigma8_lcdm_ref": args.sigma8_lcdm_ref,
            "target_S8": args.target_S8,
            "target_fs8_today": args.target_fs8_today,
            "n_z": args.n_z,
            "cache_n": args.cache_n,
        },
        "warning": "Compressed As-normalized growth audit; not CLASS/CAMB.",
    }
    with open(outdir / "v86_fast_best_points.json", "w", encoding="utf-8") as f:
        json.dump(best_points, f, indent=2)

    print(json.dumps(best_points, indent=2))
    print(f"[OK] best score: {rows[0]['score'] if rows else 'NA'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "sweep"], default="single")

    ap.add_argument("--epsilon-today", type=float, default=0.135)
    ap.add_argument("--a-c", type=float, default=0.5)
    ap.add_argument("--lambda-mu", type=float, default=2.5)
    ap.add_argument("--Q-m", type=float, default=0.75)

    ap.add_argument("--H0", type=float, default=72.2)
    ap.add_argument("--Omega-b", type=float, default=0.049)
    ap.add_argument("--A-m", type=float, default=0.40)
    ap.add_argument("--A-x", type=float, default=None)
    ap.add_argument("--k-eff", type=float, default=0.15)

    ap.add_argument("--sigma8-lcdm-ref", type=float, default=0.81)
    ap.add_argument("--target-S8", type=float, default=0.78)
    ap.add_argument("--target-S8-sigma", type=float, default=0.02)
    ap.add_argument("--target-fs8-today", type=float, default=None)
    ap.add_argument("--target-fs8-sigma", type=float, default=0.04)

    ap.add_argument("--epsilon-grid", type=float, nargs="+", default=[0.08, 0.11, 0.135, 0.16, 0.20])
    ap.add_argument("--a-c-grid", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0])
    ap.add_argument("--lambda-grid", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--Qm-grid", type=float, nargs="+", default=[0.45, 0.55, 0.65, 0.75, 0.85, 1.0])

    ap.add_argument("--z-max", type=float, default=10.0)
    ap.add_argument("--n-z", type=int, default=180)
    ap.add_argument("--cache-n", type=int, default=1200)
    ap.add_argument("--progress-every", type=int, default=10)

    ap.add_argument("--outdir", default="results_v86_growth_As_norm_fast")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z_eval = make_z_eval(args.n_z, args.z_max)
    base_args = BaseArgs(
        H0=args.H0,
        Omega_b=args.Omega_b,
        A_m=args.A_m,
        A_x=args.A_x,
        k_eff=args.k_eff,
        cache_n=args.cache_n,
    )

    if args.mode == "single":
        run_single(args, base_args, outdir, z_eval)
    else:
        run_sweep(args, base_args, outdir, z_eval)

    print(f"[OK] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DRND_ST_LENSING_FAIR_NFW_TEST_V2_FASTPATCH.py

Fast patch for DRND_ST_LENSING_FAIR_NFW_TEST_V2.py.

It does NOT change the statistical objective and does NOT force any model to win.
It only replaces the very slow numerical NFW projection with the standard
analytic projected NFW DeltaSigma formula.

Why this helps
--------------
Your run is hanging at nfw_plus_baryon_perbin because V2 computes NFW lensing by
numerical projection inside every optimizer evaluation. This patch makes NFW
projection analytic, which is typically orders of magnitude faster.

Usage
-----
Put this file in the same folder as DRND_ST_LENSING_FAIR_NFW_TEST_V2.py.

Quick fair run:

python DRND_ST_LENSING_FAIR_NFW_TEST_V2_FASTPATCH.py ^
  --v2-script DRND_ST_LENSING_FAIR_NFW_TEST_V2.py ^
  --lensing-csv lensing_data.csv ^
  --bin-mode auto ^
  --n-bins 4 ^
  --maxiter 250 ^
  --n-random-starts 2 ^
  --outdir DRND_ST_FAIR_NFW_TEST_V2_FASTPATCH

More robust run:

python DRND_ST_LENSING_FAIR_NFW_TEST_V2_FASTPATCH.py ^
  --v2-script DRND_ST_LENSING_FAIR_NFW_TEST_V2.py ^
  --lensing-csv lensing_data.csv ^
  --bin-mode auto ^
  --n-bins 4 ^
  --maxiter 500 ^
  --n-random-starts 6 ^
  --outdir DRND_ST_FAIR_NFW_TEST_V2_FASTPATCH_ROBUST
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


def load_module(path: str):
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("fair_nfw_v2_module", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nfw_delta_sigma_analytic_factory(mod):
    """
    Returns an analytic replacement for mod.nfw_ds(R, logM200, logc, H0).

    Output units: Msun/pc^2, matching the diagnostic data convention used by
    the previous scripts.
    """
    G_SI = mod.G_SI
    MSUN_KG = mod.MSUN_KG
    KPC_M = mod.KPC_M
    MPC_M = mod.MPC_M

    def rhocrit_msun_kpc3(H0):
        H = H0 * 1000.0 / MPC_M
        rho_kg_m3 = 3.0 * H * H / (8.0 * math.pi * G_SI)
        return rho_kg_m3 * (KPC_M ** 3) / MSUN_KG

    def patched_nfw_ds(R, logM200, logc, H0):
        R = np.asarray(R, dtype=float)
        c = 10.0 ** float(logc)
        M200 = 10.0 ** float(logM200)

        rho_c = rhocrit_msun_kpc3(H0)
        r200 = (3.0 * M200 / (4.0 * math.pi * 200.0 * rho_c)) ** (1.0 / 3.0)
        rs = r200 / c
        fc = math.log(1.0 + c) - c / (1.0 + c)
        rho_s = M200 / (4.0 * math.pi * rs ** 3 * fc)

        x = np.maximum(R / rs, 1e-8)
        f = np.zeros_like(x)
        g = np.zeros_like(x)

        lt = x < 1.0 - 1e-6
        gt = x > 1.0 + 1e-6
        eq = ~(lt | gt)

        if np.any(lt):
            xl = x[lt]
            y = np.sqrt((1.0 - xl) / (1.0 + xl))
            atanh_y = np.arctanh(np.clip(y, 0.0, 1.0 - 1e-14))
            f[lt] = (1.0 - 2.0 / np.sqrt(1.0 - xl * xl) * atanh_y) / (xl * xl - 1.0)
            g[lt] = np.log(xl / 2.0) + 2.0 / np.sqrt(1.0 - xl * xl) * atanh_y

        if np.any(gt):
            xg = x[gt]
            y = np.sqrt((xg - 1.0) / (1.0 + xg))
            atan_y = np.arctan(y)
            f[gt] = (1.0 - 2.0 / np.sqrt(xg * xg - 1.0) * atan_y) / (xg * xg - 1.0)
            g[gt] = np.log(xg / 2.0) + 2.0 / np.sqrt(xg * xg - 1.0) * atan_y

        if np.any(eq):
            f[eq] = 1.0 / 3.0
            g[eq] = 1.0 + math.log(0.5)

        sigma = 2.0 * rho_s * rs * f
        sigma_bar = 4.0 * rho_s * rs * g / np.maximum(x * x, 1e-30)
        delta_sigma_msun_kpc2 = sigma_bar - sigma
        return delta_sigma_msun_kpc2 / 1e6

    return patched_nfw_ds


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v2-script", default="DRND_ST_LENSING_FAIR_NFW_TEST_V2.py")
    known, rest = parser.parse_known_args()

    mod = load_module(known.v2_script)
    mod.nfw_ds = nfw_delta_sigma_analytic_factory(mod)

    # Keep runs finite unless user explicitly overrides these.
    if "--maxiter" not in rest:
        rest += ["--maxiter", "250"]
    if "--n-random-starts" not in rest:
        rest += ["--n-random-starts", "2"]

    sys.argv = [known.v2_script] + rest
    mod.main()


if __name__ == "__main__":
    main()

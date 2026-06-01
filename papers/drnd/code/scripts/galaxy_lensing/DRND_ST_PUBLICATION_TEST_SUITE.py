#!/usr/bin/env python3
"""
DRND_ST_PUBLICATION_TEST_SUITE.py

Publication-oriented closing tests for DRND-ST / ATP lensing.

Runs four closing tests:
1. Final reproducibility run with full NFW refit.
2. Binning sensitivity over n_bins=3,4,5 and bin modes mass/signal.
3. Strict no-eta/no-tilt claim subset extraction.
4. Prefactor audit including lower kappa values below 0.25.

Required:
  DRND_ST_ATP_NORMALIZATION_AUDIT_V3_FAST_NO_NFW.py

Recommended:
  python DRND_ST_PUBLICATION_TEST_SUITE.py ^
    --lensing-csv lensing_data.csv ^
    --audit-script DRND_ST_ATP_NORMALIZATION_AUDIT_V3_FAST_NO_NFW.py ^
    --outdir DRND_ST_PUBLICATION_TESTS ^
    --maxiter 500 ^
    --n-random-starts 4

Fast:
  python DRND_ST_PUBLICATION_TEST_SUITE.py ^
    --lensing-csv lensing_data.csv ^
    --audit-script DRND_ST_ATP_NORMALIZATION_AUDIT_V3_FAST_NO_NFW.py ^
    --outdir DRND_ST_PUBLICATION_TESTS_FAST ^
    --maxiter 200 ^
    --n-random-starts 1 ^
    --fast
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_KAPPA_GRID = ",".join([
    "0.03125", "0.0625", "0.125", "0.1875", "0.25",
    "0.3026720203", "0.375", "0.5", "0.75", "1",
    "1.5", "2", "2.5", "3", str(math.pi), "3.25",
    "3.5", "4", str(2 * math.pi), str(4 * math.pi),
])

STRICT_KAPPA_GRID = ",".join([
    "0.03125", "0.0625", "0.125", "0.1875", "0.25",
    "0.3026720203", "0.375", "0.5", "0.75", "1",
    "2", "3", str(math.pi),
])


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], preferred: Optional[List[str]] = None) -> None:
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


def model_row(summary: Dict[str, Any], case: str) -> Optional[Dict[str, Any]]:
    for r in summary.get("model_comparison", []):
        if r.get("case") == case:
            return r
    return None


def best_case(summary: Dict[str, Any], metric: str = "BIC") -> Optional[str]:
    rows = summary.get("model_comparison", [])
    if not rows:
        return None
    return min(rows, key=lambda r: r.get(metric, float("inf"))).get("case")


def preferred_case_from_kappa(kappa_text: str) -> str:
    s = str(kappa_text).strip()
    if s in {"0.250000", "0.25"}:
        return "atp_kappa_fixed_0.25_plus_baryon"
    return f"atp_kappa_fixed_{s}_plus_baryon"


def run_cmd(cmd: List[str], log_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rec = {"cmd": cmd, "log_path": str(log_path), "start_time_epoch": start, "dry_run": dry_run}
    if dry_run:
        log_path.write_text("DRY RUN:\n" + " ".join(cmd) + "\n", encoding="utf-8")
        rec.update({"returncode": 0, "elapsed_sec": 0.0})
        return rec
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    rec.update({"returncode": proc.returncode, "elapsed_sec": time.time() - start})
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}. See {log_path}")
    return rec


def audit_command(
    python_exe: str,
    audit_script: Path,
    lensing_csv: Path,
    outdir: Path,
    bin_mode: str,
    n_bins: int,
    kappa_grid: str,
    maxiter: int,
    maxfun: int,
    n_random_starts: int,
    H0: float,
    chi: float,
    cov_space: str,
    profile_type: str,
    skip_nfw: bool = False,
) -> List[str]:
    cmd = [
        python_exe, str(audit_script),
        "--lensing-csv", str(lensing_csv),
        "--bin-mode", bin_mode,
        "--n-bins", str(n_bins),
        "--kappa-grid", kappa_grid,
        "--maxiter", str(maxiter),
        "--maxfun", str(maxfun),
        "--n-random-starts", str(n_random_starts),
        "--H0", str(H0),
        "--chi", str(chi),
        "--cov-space", cov_space,
        "--profile-type", profile_type,
        "--outdir", str(outdir),
    ]
    if skip_nfw:
        cmd.append("--skip-nfw")
    return cmd


def strict_extract(summary: Dict[str, Any], preferred_kappa_case: str) -> Dict[str, Any]:
    nfw = model_row(summary, "nfw_plus_baryon_global_relation")
    baryon = model_row(summary, "baryon_only")
    atp = model_row(summary, preferred_kappa_case)
    dec = summary.get("decision", {})
    return {
        "preferred_kappa_case": preferred_kappa_case,
        "best_model_by_BIC": dec.get("best_model_by_BIC") or best_case(summary, "BIC"),
        "best_model_by_AIC": dec.get("best_model_by_AIC") or best_case(summary, "AIC"),
        "ATP_BIC": atp.get("BIC") if atp else None,
        "ATP_AIC": atp.get("AIC") if atp else None,
        "ATP_chi2": atp.get("chi2") if atp else None,
        "ATP_chi2_per_dof": atp.get("chi2_per_dof") if atp else None,
        "NFW_BIC": nfw.get("BIC") if nfw else None,
        "NFW_AIC": nfw.get("AIC") if nfw else None,
        "NFW_chi2": nfw.get("chi2") if nfw else None,
        "NFW_chi2_per_dof": nfw.get("chi2_per_dof") if nfw else None,
        "Baryon_BIC": baryon.get("BIC") if baryon else None,
        "delta_BIC_ATP_minus_NFW": (atp.get("BIC") - nfw.get("BIC")) if atp and nfw else None,
        "delta_AIC_ATP_minus_NFW": (atp.get("AIC") - nfw.get("AIC")) if atp and nfw else None,
        "n_points": summary.get("input", {}).get("n_points"),
        "bin_mode_used": summary.get("input", {}).get("bin_mode_used"),
        "n_bins": summary.get("input", {}).get("n_bins"),
    }


def find_best_fixed_kappa(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows = [
        r for r in summary.get("model_comparison", [])
        if str(r.get("case", "")).startswith("atp_kappa_fixed_")
        and str(r.get("case", "")).endswith("_plus_baryon")
    ]
    return min(rows, key=lambda r: r.get("BIC", float("inf"))) if rows else None


def methods_text() -> str:
    return r"""# Methods text for publication draft

We performed a closing diagnostic audit of the DRND-ST analytic topological projection (ATP) lensing kernel. The tested publication-claim model was

\[
\Delta\Sigma_{\rm model}(R)=\Delta\Sigma_{\rm baryon}(R)+\Delta\Sigma_{\rm ATP}(R),
\]

with

\[
A(\kappa)=\kappa\frac{\sqrt{G M_b a_0}}{G}.
\]

The reference DRND-ST/ATP value is \(\kappa=1/4\). The acceleration scale was fixed throughout,

\[
a_0=\chi cH_0,\qquad \chi=0.18,
\]

and was not refitted. A constrained NFW+baryon model with a concentration--mass relation was used as the diagnostic control. Model comparison used \(\chi^2\), AIC, and BIC. Positive claims are restricted to fixed-\(\kappa\) ATP+baryon models; global \(\kappa\), screened environment, \(\eta\), and radial tilt are diagnostic only.
"""


def results_text(summary: Dict[str, Any]) -> str:
    lines = ["# Results text for publication draft", ""]
    primary = summary.get("primary_result", {})
    sens = summary.get("sensitivity", {})
    pref = summary.get("prefactor_audit", {})

    lines.append("## Primary reproducibility result")
    if primary:
        lines.append(
            f"The primary run selected `{primary.get('best_model_by_BIC')}` by BIC. "
            f"For `{primary.get('preferred_kappa_case')}`, BIC={primary.get('ATP_BIC')}; "
            f"the constrained NFW+baryon control had BIC={primary.get('NFW_BIC')}. "
            f"Thus ΔBIC(ATP−NFW)={primary.get('delta_BIC_ATP_minus_NFW')}."
        )
    else:
        lines.append("Primary result unavailable.")

    lines.append("\n## Binning sensitivity")
    if sens:
        lines.append(
            f"The preferred fixed ATP model beat the NFW control in "
            f"{sens.get('n_successes_ATP_beats_NFW_BIC')} / {sens.get('n_runs_with_NFW')} "
            f"binning runs by BIC."
        )
    else:
        lines.append("Sensitivity result unavailable.")

    lines.append("\n## Prefactor audit")
    if pref:
        lines.append(
            f"The best fixed-kappa model was `{pref.get('best_fixed_kappa_model')}` "
            f"with BIC={pref.get('best_fixed_kappa_BIC')}. "
            f"The lower-kappa scan explicitly tested values below 1/4."
        )
        lines.append(
            f"The kappa=pi candidate had BIC={pref.get('kappa_pi_BIC')}; "
            f"ΔBIC(kappa=pi − best fixed kappa)="
            f"{pref.get('delta_BIC_kappaPi_minus_bestFixed')}."
        )
    else:
        lines.append("Prefactor audit unavailable.")

    lines.append("\nThese results remain diagnostic until tested with full covariance, HOD/miscentering, satellite terms, and independent lensing data.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lensing-csv", required=True)
    ap.add_argument("--audit-script", default="DRND_ST_ATP_NORMALIZATION_AUDIT_V3_FAST_NO_NFW.py")
    ap.add_argument("--outdir", default="DRND_ST_PUBLICATION_TESTS")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--H0", type=float, default=67.4)
    ap.add_argument("--chi", type=float, default=0.18)
    ap.add_argument("--cov-space", default="log", choices=["log", "linear"])
    ap.add_argument("--profile-type", default="hernquist")
    ap.add_argument("--primary-bin-mode", default="auto", choices=["auto", "existing", "mass", "signal", "radius"])
    ap.add_argument("--primary-n-bins", type=int, default=4)
    ap.add_argument("--kappa-grid", default=DEFAULT_KAPPA_GRID)
    ap.add_argument("--strict-kappa-grid", default=STRICT_KAPPA_GRID)
    ap.add_argument("--preferred-kappa", default="0.25")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--maxfun", type=int, default=1500)
    ap.add_argument("--n-random-starts", type=int, default=4)
    ap.add_argument("--fast", action="store_true", help="Use fewer sensitivity runs.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-sensitivity-nfw", action="store_true", help="Not publication-safe; quick only.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audit_script = Path(args.audit_script)
    lensing_csv = Path(args.lensing_csv)
    preferred_case = preferred_case_from_kappa(args.preferred_kappa)

    if not audit_script.exists():
        raise SystemExit(f"Audit script not found: {audit_script}")
    if not lensing_csv.exists():
        raise SystemExit(f"Lensing CSV not found: {lensing_csv}")

    manifest: Dict[str, Any] = {
        "suite": "DRND-ST publication closing tests",
        "lensing_csv": str(lensing_csv),
        "audit_script": str(audit_script),
        "created_epoch": time.time(),
        "runs": [],
    }
    decision_rows: List[Dict[str, Any]] = []

    # TEST 1
    t1_dir = outdir / "T1_final_reproducibility_full_nfw"
    cmd = audit_command(
        args.python, audit_script, lensing_csv, t1_dir,
        args.primary_bin_mode, args.primary_n_bins, args.kappa_grid,
        args.maxiter, args.maxfun, args.n_random_starts,
        args.H0, args.chi, args.cov_space, args.profile_type,
        skip_nfw=False,
    )
    manifest["runs"].append(run_cmd(cmd, outdir / "logs" / "T1_final_reproducibility.log", args.dry_run))
    t1_summary = read_json(t1_dir / "summary.json") if (t1_dir / "summary.json").exists() else {}
    primary = strict_extract(t1_summary, preferred_case) if t1_summary else {}
    primary.update({"test": "T1_final_reproducibility_full_nfw", "outdir": str(t1_dir)})
    decision_rows.append(primary)

    # TEST 2
    sensitivity_specs = [("mass", 3), ("mass", 4), ("mass", 5)] if args.fast else [
        ("mass", 3), ("mass", 4), ("mass", 5),
        ("signal", 3), ("signal", 4), ("signal", 5),
    ]
    sens_results = []
    for bin_mode, n_bins in sensitivity_specs:
        label = f"T2_sensitivity_{bin_mode}_{n_bins}bins"
        run_dir = outdir / label
        cmd = audit_command(
            args.python, audit_script, lensing_csv, run_dir,
            bin_mode, n_bins, args.strict_kappa_grid,
            args.maxiter, args.maxfun, args.n_random_starts,
            args.H0, args.chi, args.cov_space, args.profile_type,
            skip_nfw=args.skip_sensitivity_nfw,
        )
        manifest["runs"].append(run_cmd(cmd, outdir / "logs" / f"{label}.log", args.dry_run))
        sp = run_dir / "summary.json"
        if sp.exists():
            s = read_json(sp)
            row = strict_extract(s, preferred_case)
            row.update({"test": label, "outdir": str(run_dir), "bin_mode": bin_mode, "n_bins_requested": n_bins})
            sens_results.append(row)
            decision_rows.append(row)

    # TEST 3
    t3_dir = outdir / "T3_strict_no_eta_no_tilt_claim_subset"
    cmd = audit_command(
        args.python, audit_script, lensing_csv, t3_dir,
        args.primary_bin_mode, args.primary_n_bins, args.strict_kappa_grid,
        args.maxiter, args.maxfun, args.n_random_starts,
        args.H0, args.chi, args.cov_space, args.profile_type,
        skip_nfw=False,
    )
    manifest["runs"].append(run_cmd(cmd, outdir / "logs" / "T3_strict_claim_subset.log", args.dry_run))
    t3_summary = read_json(t3_dir / "summary.json") if (t3_dir / "summary.json").exists() else {}
    strict = strict_extract(t3_summary, preferred_case) if t3_summary else {}
    strict.update({"test": "T3_strict_no_eta_no_tilt_claim_subset", "outdir": str(t3_dir)})
    decision_rows.append(strict)

    # TEST 4
    t4_dir = outdir / "T4_prefactor_lower_kappa_audit"
    cmd = audit_command(
        args.python, audit_script, lensing_csv, t4_dir,
        args.primary_bin_mode, args.primary_n_bins, args.kappa_grid,
        args.maxiter, args.maxfun, args.n_random_starts,
        args.H0, args.chi, args.cov_space, args.profile_type,
        skip_nfw=False,
    )
    manifest["runs"].append(run_cmd(cmd, outdir / "logs" / "T4_prefactor_lower_kappa.log", args.dry_run))
    t4_summary = read_json(t4_dir / "summary.json") if (t4_dir / "summary.json").exists() else {}
    best_fixed = find_best_fixed_kappa(t4_summary) if t4_summary else None
    kappa_pi = None
    if t4_summary:
        for case in ["atp_kappa_fixed_3.14159_plus_baryon", f"atp_kappa_fixed_{math.pi}_plus_baryon"]:
            kappa_pi = model_row(t4_summary, case)
            if kappa_pi:
                break
    pref = {
        "test": "T4_prefactor_lower_kappa_audit",
        "outdir": str(t4_dir),
        "best_fixed_kappa_model": best_fixed.get("case") if best_fixed else None,
        "best_fixed_kappa_BIC": best_fixed.get("BIC") if best_fixed else None,
        "best_fixed_kappa_AIC": best_fixed.get("AIC") if best_fixed else None,
        "best_fixed_kappa_chi2": best_fixed.get("chi2") if best_fixed else None,
        "kappa_pi_BIC": kappa_pi.get("BIC") if kappa_pi else None,
        "delta_BIC_kappaPi_minus_bestFixed": (kappa_pi.get("BIC") - best_fixed.get("BIC")) if kappa_pi and best_fixed else None,
    }
    decision_rows.append(pref)

    n_with_nfw = sum(1 for r in sens_results if r.get("NFW_BIC") is not None)
    n_atp_beats = sum(
        1 for r in sens_results
        if r.get("delta_BIC_ATP_minus_NFW") is not None and r["delta_BIC_ATP_minus_NFW"] < 0
    )

    publication_summary = {
        "suite": "DRND-ST publication closing tests",
        "primary_result": primary,
        "strict_claim_subset": strict,
        "sensitivity": {
            "n_runs": len(sens_results),
            "n_runs_with_NFW": n_with_nfw,
            "n_successes_ATP_beats_NFW_BIC": n_atp_beats,
            "rows": sens_results,
        },
        "prefactor_audit": pref,
        "publication_claim_rule": {
            "allowed_positive_claim": "fixed-kappa ATP+baryon only",
            "diagnostic_only": ["global kappa", "screened two-halo", "tilt", "eta"],
            "preferred_kappa_case": preferred_case,
        },
        "lower_kappa_included": True,
        "kappa_grid": args.kappa_grid,
        "strict_kappa_grid": args.strict_kappa_grid,
        "cautions": [
            "Publication claims should not rely on eta, tilt, screened two-halo, or free global kappa.",
            "Full survey covariance, HOD, miscentering, satellites, and independent data remain required.",
            "NFW is refit when binning changes unless --skip-sensitivity-nfw is used.",
        ],
    }

    write_json(outdir / "run_manifest.json", manifest)
    write_json(outdir / "publication_summary.json", publication_summary)
    write_csv(outdir / "publication_decision_table.csv", decision_rows, preferred=[
        "test", "outdir", "bin_mode", "n_bins_requested", "preferred_kappa_case",
        "best_model_by_BIC", "best_model_by_AIC",
        "ATP_BIC", "NFW_BIC", "delta_BIC_ATP_minus_NFW",
        "ATP_AIC", "NFW_AIC", "delta_AIC_ATP_minus_NFW",
        "ATP_chi2", "ATP_chi2_per_dof", "NFW_chi2", "NFW_chi2_per_dof",
        "best_fixed_kappa_model", "best_fixed_kappa_BIC", "kappa_pi_BIC",
        "delta_BIC_kappaPi_minus_bestFixed",
    ])
    (outdir / "publication_methods_text.md").write_text(methods_text(), encoding="utf-8")
    (outdir / "publication_results_text.md").write_text(results_text(publication_summary), encoding="utf-8")

    print(json.dumps({
        "outdir": str(outdir),
        "primary_result": primary,
        "sensitivity_summary": publication_summary["sensitivity"],
        "prefactor_audit": pref,
        "files": [
            "publication_summary.json",
            "publication_decision_table.csv",
            "publication_methods_text.md",
            "publication_results_text.md",
            "run_manifest.json",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

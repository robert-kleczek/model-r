#!/usr/bin/env python3
"""
prepare_growth_inputs.py

Validate and clean growth CSV files for:

    DRND_v71_joint_multi_lensing_growth.py

Accepted inputs:
    fsigma8_data.csv or fsigma8_data_TEMPLATE.csv
    eg_data.csv      or eg_data_TEMPLATE.csv
    s8_data.csv      or s8_data_TEMPLATE.csv

This script does NOT invent literature values. It only:
    - checks required columns,
    - removes rows with empty required numeric fields if --drop-incomplete is used,
    - writes cleaned CSVs to an output directory,
    - prints the command for DRND_v71.

Usage:
    python prepare_growth_inputs.py --indir growth_sources --outdir .
    python prepare_growth_inputs.py --indir growth_sources --outdir . --drop-incomplete

If you already created the real CSVs in C:\\Programming\\DRND:
    python prepare_growth_inputs.py --indir . --outdir . --drop-incomplete
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Tuple


SPECS = {
    "fsigma8": {
        "input_names": ["fsigma8_data.csv", "fsigma8_data_TEMPLATE.csv", "example_fs8.csv"],
        "output": "fsigma8_data.csv",
        "required": ["z", "fs8_obs", "fs8_err", "fs8_lcdm"],
        "optional": ["label", "k", "source", "notes"],
    },
    "eg": {
        "input_names": ["eg_data.csv", "eg_data_TEMPLATE.csv", "example_eg.csv"],
        "output": "eg_data.csv",
        "required": ["z", "k", "eg_obs", "eg_err", "eg_lcdm"],
        "optional": ["label", "source", "notes"],
    },
    "s8": {
        "input_names": ["s8_data.csv", "s8_data_TEMPLATE.csv", "example_s8.csv"],
        "output": "s8_data.csv",
        "required": ["s8_obs", "s8_err", "s8_lcdm"],
        "optional": ["label", "z", "k", "source", "notes"],
    },
}


def find_input(indir: Path, names: List[str]) -> Path | None:
    for name in names:
        p = indir / name
        if p.exists():
            return p
    return None


def is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def read_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        rows = [{k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_one(name: str, spec: Dict, indir: Path, outdir: Path, drop_incomplete: bool) -> Dict:
    inp = find_input(indir, spec["input_names"])
    result = {
        "name": name,
        "input": None if inp is None else str(inp),
        "output": str(outdir / spec["output"]),
        "found": inp is not None,
        "n_rows_in": 0,
        "n_rows_out": 0,
        "warnings": [],
        "ok": False,
    }

    if inp is None:
        result["warnings"].append(f"No input file found for {name}.")
        return result

    fieldnames, rows = read_rows(inp)
    result["n_rows_in"] = len(rows)

    missing_cols = [c for c in spec["required"] if c not in fieldnames]
    if missing_cols:
        result["warnings"].append(f"Missing required columns: {missing_cols}")
        return result

    complete_rows = []
    incomplete_rows = []

    for i, row in enumerate(rows, start=2):
        bad = []
        for c in spec["required"]:
            val = row.get(c, "")
            if val is None or str(val).strip() == "":
                bad.append(c)
            elif not is_number(str(val)):
                bad.append(c)
        if bad:
            incomplete_rows.append((i, bad, row))
        else:
            complete_rows.append(row)

    if incomplete_rows:
        msg = f"{len(incomplete_rows)} incomplete/non-numeric rows found."
        if drop_incomplete:
            result["warnings"].append(msg + " Dropping them.")
        else:
            result["warnings"].append(msg + " Use --drop-incomplete after filling or to drop placeholders.")
            # Do not write an output that silently contains blanks.
            return result

    output_rows = complete_rows if drop_incomplete else rows
    if len(output_rows) == 0:
        result["warnings"].append("No complete rows remain.")
        return result

    # Preserve all columns, but put required columns first.
    ordered = []
    for c in spec["required"] + spec["optional"]:
        if c in fieldnames and c not in ordered:
            ordered.append(c)
    for c in fieldnames:
        if c not in ordered:
            ordered.append(c)

    outdir.mkdir(parents=True, exist_ok=True)
    write_rows(outdir / spec["output"], ordered, output_rows)

    result["n_rows_out"] = len(output_rows)
    result["ok"] = True
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="growth_sources")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--drop-incomplete", action="store_true")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    print(f"[INFO] input directory:  {indir.resolve()}")
    print(f"[INFO] output directory: {outdir.resolve()}")

    results = []
    for name, spec in SPECS.items():
        res = validate_one(name, spec, indir, outdir, args.drop_incomplete)
        results.append(res)

    print("\nSummary:")
    any_ok = False
    for res in results:
        status = "OK" if res["ok"] else "NOT READY"
        print(f"  {res['name']:8s} {status:9s} rows {res['n_rows_in']} -> {res['n_rows_out']}")
        if res["input"]:
            print(f"    input:  {res['input']}")
        print(f"    output: {res['output']}")
        for w in res["warnings"]:
            print(f"    warning: {w}")
        any_ok = any_ok or res["ok"]

    if any_ok:
        print("\nNext command skeleton:")
        parts = [
            "python DRND_v71_joint_multi_lensing_growth.py ^",
            "  --lensing-npz planck_lensing_inputs_from_uploaded.npz act_lensing_inputs.npz spt_lensing_inputs.npz ^",
            "  --labels Planck ACT SPT ^",
        ]
        if (outdir / "fsigma8_data.csv").exists():
            parts.append("  --fs8-csv fsigma8_data.csv ^")
        if (outdir / "eg_data.csv").exists():
            parts.append("  --eg-csv eg_data.csv ^")
        if (outdir / "s8_data.csv").exists():
            parts.append("  --s8-csv s8_data.csv ^")
        parts += [
            "  --outdir results_v71_planck_act_spt_growth ^",
            "  --run-all-modes ^",
            "  --nsteps 4000 ^",
            "  --burn 1000 ^",
            "  --thin 10",
        ]
        print("\n".join(parts))
    else:
        print("\nNo ready growth CSV was produced. Fill the TEMPLATE files first, then rerun with --drop-incomplete if needed.")


if __name__ == "__main__":
    main()

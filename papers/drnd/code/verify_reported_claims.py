#!/usr/bin/env python3
"""Verify machine-readable values reported in the DRND II v1.1.0-alpha manuscript.

Run from papers/drnd:
    python code/verify_reported_claims.py
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]  # papers/drnd
pub = json.loads((root / "data/galaxy_lensing/publication_summary.json").read_text(encoding="utf-8"))
rep = json.loads((root / "results_reported/reported_claim_values.json").read_text(encoding="utf-8"))

print("Galaxy/lensing ATP vs NFW")
primary = pub["primary_result"]
print("  BIC_ATP_baryon =", primary["ATP_BIC"])
print("  BIC_NFW_baryon =", primary["NFW_BIC"])
print("  Delta_BIC_ATP_NFW =", primary["delta_BIC_ATP_minus_NFW"])

expected = rep["galaxy_lensing_ATP_vs_NFW"]
assert abs(primary["ATP_BIC"] - expected["BIC_ATP_baryon"]) < 1e-9
assert abs(primary["NFW_BIC"] - expected["BIC_NFW_baryon"]) < 1e-9
assert abs(primary["delta_BIC_ATP_minus_NFW"] - expected["Delta_BIC_ATP_NFW"]) < 1e-9

print("Void topological gain 19/2")
v = rep["void_topological_gain_19_over_2"]
for k in [
    "A_DRND_19_over_2", "A_required", "ratio", "relative_mismatch",
    "BIC_19_over_2", "BIC_A_free", "BIC_null",
    "Delta_BIC_19_over_2_null", "Delta_BIC_19_over_2_A_free",
    "chi2_19_over_2", "chi2_A_free",
]:
    print(f"  {k} = {v[k]}")

assert abs(v["ratio"] - 0.99497) < 1e-5
assert abs(v["relative_mismatch"] - 0.00503) < 1e-5
print("PASS: bundled reported values are internally consistent.")

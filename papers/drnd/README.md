# DRND II v2.0.0-alpha - paper and reproduction package

This directory contains the manuscript PDF, LaTeX source files, machine-readable audit data, and reproduction scripts for:

Discrete Relational Network Dynamics II: Spectral-Topological Transport, ATP Lensing, and Void-Wall Diagnostics

Version: v2.0.0-alpha
Date: May 2026

Contents

Discrete_Relational_Network_Dynamics_II_v2_0_0_alpha.pdf
  Manuscript PDF.

source/
  LaTeX source files used for the manuscript build, including main.tex and the local IOP class/style files.

code/requirements.txt
  Python requirements for the reproduction scripts.

code/verify_reported_claims.py
  Quick consistency check for the numerical values reported in the paper.

code/scripts/galaxy_lensing/
  Galaxy/lensing ATP vs NFW audit scripts.

code/scripts/voids/
  Void morphology and topological-gain amplitude scripts.

code/scripts/growth_s8/
  Supporting S8/growth scripts. These are included as supporting material, not as the central DRND II v2.0.0-alpha claim.

data/galaxy_lensing/
  Machine-readable galaxy/lensing audit outputs.

data/voids/
  Void geometry summaries and diagnostic inputs.

data/growth_s8/
  Supporting growth inputs.

results_reported/reported_claim_values.json
  Exact scalar values quoted in the manuscript.

code/run/*.sh and code/run/*.ps1
  Run wrappers for selected modules.

Quick verification

Linux or macOS:

python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
python code/verify_reported_claims.py

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r code\requirements.txt
python code\verify_reported_claims.py

Reported manuscript values

Frozen acceleration scale:

a0 = chi * c * H0
chi = 0.18
H0 = 67.4 km s^-1 Mpc^-1
c = 299792458 m s^-1

a0 = 1.1786980345e-10 m s^-2
reported as 1.1787e-10 m s^-2

In DRND II v2.0.0-alpha, this is a frozen present-normalized transport scale. It is not replaced by chi*c*H(z) in the validation sector.

Galaxy/lensing ATP vs NFW

The bundled data/galaxy_lensing/publication_summary.json and data/galaxy_lensing/run_manifest.json reproduce the reported BIC values:

BIC_ATP_baryon = 31.70
BIC_NFW_baryon = 44.26
Delta_BIC_ATP_NFW = -12.56

The convention is:

Delta_BIC_A_minus_B = BIC_A - BIC_B

Negative Delta_BIC_ATP_NFW favours the fixed DRND ATP+baryon model over the NFW+baryon control.

The original raw lensing_data.csv is not bundled. To rerun the audit end-to-end, place it at:

data/galaxy_lensing/lensing_data.csv

then run:

bash code/run/RUN_GALAXY_LENSING.sh

or in Windows PowerShell:

code\run\RUN_GALAXY_LENSING.ps1

Void morphology

Run:

bash code/run/RUN_VOID_MORPHOLOGY_WIDE.sh

This script downloads the public SDSS void-lensing FITS data if absent.

The wide SDSS R/Rv x-shape morphology diagnostic reports:

Best model:
zero_core_compensated_wall_DRND_refined

BIC_DRND_refined = 26.7753
chi2_DRND_refined = 15.7892
k_params = 5
n_data = 9

BIC_null = 112.6509
BIC_wall_only = 75.1571
BIC_uncompensated = 77.6183

Delta_BIC_DRND_null = -85.8756
Delta_BIC_DRND_wall_only = -48.3818
Delta_BIC_DRND_uncompensated = -50.8430

This supports the morphology claim that the zero-core compensated-wall model is preferred over the null, wall-only, and uncompensated templates in the diagnostic x-shape audit.

This is a shape and morphology diagnostic, not a full physical-unit shear likelihood.

Void amplitude with topological gain 19/2

Run:

bash code/run/RUN_VOID_TOPO_GAIN_19_OVER_2.sh

This reproduces the locked-amplitude model:

A_DRND_19_over_2 = 0.2029359
A_required = 0.2039621
ratio = 0.99497
relative_mismatch = 0.00503

BIC_19_over_2 = 37.70
BIC_A_free = 39.90
BIC_null = 112.65

Delta_BIC_19_over_2_A_free = -2.20
Delta_BIC_19_over_2_null = -74.95

chi2_19_over_2 = 37.7027
chi2_A_free = 37.7008

The fitted-amplitude model has a slightly lower chi-square, but the locked DRND 19/2 model is preferred by BIC because it uses no fitted amplitude parameter.

S8/growth supporting smoke run

S8/growth code is included as supporting material, not as the main DRND II v2.0.0-alpha claim.

Run:

bash code/run/RUN_GROWTH_S8_SMOKE.sh

Scope

These scripts reproduce the tables and scalar diagnostics reported in the paper. They are not a complete end-to-end survey pipeline.

Full independent validation requires external shear catalogues, void catalogues, covariance matrices, mock catalogues, void-finder systematics, and physically normalized DeltaSigma profiles.

What this archive does not contain

This archive does not contain a complete cosmological likelihood, full CMB/BAO/SNe/BBN constraints, complete graph microdynamics, or a completed Standard Model derivation.

These remain future-work requirements for subsequent DRND releases.

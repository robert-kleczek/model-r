# DRND II v2.0.0-alpha - Zenodo reproduction archive

This repository-style archive accompanies the manuscript:

Discrete Relational Network Dynamics II: Spectral-Topological Transport, ATP Lensing, and Void-Wall Diagnostics

Version: v2.0.0-alpha
Date: May 2026
Author: Robert Kleczek

This archive is intended for Zenodo deposition and independent inspection of the numerical diagnostics reported in the manuscript.

It contains the manuscript PDF, LaTeX source files, machine-readable reported values, reproduction scripts, run wrappers, and audit summaries required for the paper's diagnostic claims.

This archive does not constitute a complete end-to-end survey pipeline. It reproduces the reported scalar audit values used in the manuscript: the frozen acceleration scale, the BIC differences, the void morphology BIC audit, and the locked void-wall amplitude ratio.

Directory structure

model-r-v2.0.0-alpha/
  README.md
  CITATION.cff
  LICENSE
  ZENODO_METADATA_DRAFT.json
  MANIFEST_SHA256.txt
  papers/drnd/
    README.md
    Discrete_Relational_Network_Dynamics_II_v2_0_0_alpha.pdf
    source/
      main.tex
      iopart.cls
      iopart10.clo
      iopart12.clo
      iopams.sty
      setstack.sty
      orcid.png
    code/
      requirements.txt
      verify_reported_claims.py
      run/
      scripts/
    data/
      galaxy_lensing/
      voids/
      growth_s8/
    results_reported/
      reported_claim_values.json
    figures/

Main reproduction directory

Go to:

cd papers/drnd

Then read:

papers/drnd/README.md

Quick verification

Linux or macOS:

cd papers/drnd
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
python code/verify_reported_claims.py

Windows PowerShell:

cd papers\drnd
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r code\requirements.txt
python code\verify_reported_claims.py

Scope

This archive reproduces the paper's machine-readable tables and key diagnostics. It is not a full end-to-end survey pipeline.

The galaxy/lensing audit can be rerun end-to-end only if the original lensing_data.csv is placed in:

papers/drnd/data/galaxy_lensing/lensing_data.csv

The void scripts download public SDSS void-lensing data automatically when run.

Version status

This release is marked v2.0.0-alpha because the manuscript defines a new frozen DRND II validation sector relative to DRND v1.0.3.

The suffix alpha denotes a frozen theoretical core with diagnostic validation, but not yet a fully independent end-to-end cosmological likelihood implementation.

Related prior version

The previous DRND baseline is:

Robert Kleczek, Discrete Relational Network Dynamics (DRND): A Fully Emergent Framework for Spacetime, Matter, and the Evolution of Fundamental Constants, Zenodo v1.0.3, DOI: 10.5281/zenodo.18615542

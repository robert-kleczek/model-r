# Run from papers/drnd in PowerShell.
# Requires data/galaxy_lensing/lensing_data.csv.
python code/scripts/galaxy_lensing/DRND_ST_PUBLICATION_TEST_SUITE.py `
  --lensing-csv data/galaxy_lensing/lensing_data.csv `
  --audit-script code/scripts/galaxy_lensing/DRND_ST_ATP_NORMALIZATION_AUDIT_V3_FAST_NO_NFW.py `
  --outdir outputs/galaxy_lensing/DRND_ST_PUBLICATION_TESTS_FAST `
  --maxiter 200 `
  --n-random-starts 1 `
  --fast

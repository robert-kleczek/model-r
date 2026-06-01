# Galaxy/lensing data status

The publication-level ATP vs NFW values are preserved in `publication_summary.json` and the exact run commands are preserved in `run_manifest.json`.

The full audit rerun expects the original input file:

```text
lensing_data.csv
```

This raw CSV was referenced by the completed audit manifest, but it was not present among the files available when this Zenodo package was assembled. Therefore:

- `publication_summary.json` is the machine-readable audit output for the paper tables.
- `run_manifest.json` records the exact commands that produced it.
- `synthetic_lensing_sample_SMOKE_TEST_ONLY.csv` is included only for parser/smoke checks and must not be used to reproduce the reported BIC claim.

To rerun the galaxy/lensing audit end-to-end, place the original `lensing_data.csv` in this directory and run the commands in `run/RUN_GALAXY_LENSING.ps1` or `run/RUN_GALAXY_LENSING.sh`.

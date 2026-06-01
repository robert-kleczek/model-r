# Run from papers/drnd in PowerShell. Downloads public SDSS void-lensing data if absent.
python code/scripts/voids/DRND_VOID_LOCKED_PHYSICAL_AMPLITUDE_TEST_V2_TOPO_GAIN.py `
  --geometry-summary data/voids/geometry_summary_published_fast.json `
  --outdir outputs/voids/DRND_VOID_LOCKED_TOPO_GAIN_TEST_FULL `
  --nboot 10000 `
  --topological-gain 9.5 `
  --topological-gain-label 19_over_2

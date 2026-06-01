#!/usr/bin/env bash
set -euo pipefail
# Run from papers/drnd. Downloads public SDSS void-lensing data if absent.
python code/scripts/voids/DRND_VOID_ZERO_CORE_COMPENSATED_WALL_LIKELIHOOD.py   --outdir outputs/voids/DRND_VOID_ZERO_CORE_WIDE   --nboot 10000   --x-edge-max 3.2   --w-edge-min 0.02   --sigma-wall-max 1.8

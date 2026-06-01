#!/usr/bin/env python3
"""
download_growth_data_sources.py

Downloads public source material useful for building fsigma8_data.csv.

This script does not pretend to create a definitive fσ8 compilation automatically:
different papers use different fiducial cosmologies, correlations, Alcock-Paczynski
rescalings, and covariance treatments. It downloads public source packages/pages and
creates an editable starter CSV template for DRND_v71.

Outputs:
    growth_sources/
    fsigma8_data_TEMPLATE.csv
    eg_data_TEMPLATE.csv
    s8_data_TEMPLATE.csv

Usage:
    python download_growth_data_sources.py --outdir growth_sources
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


URLS = {
    "DESI_QA_download_fsigma8_data.html": "https://help.desi.lbl.gov/index.php?qa=123&qa_1=download-fsigma8-data",
    "DESI_DR1_releases.html": "https://data.desi.lbl.gov/doc/releases/dr1/",
    "SDSS_eBOSS_DR16_BAO_plus_README.txt": "https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_0/likelihoods/BAO-plus/README.txt",
}


def fetch(url: str, dest: Path, timeout: int = 60) -> bool:
    print(f"[INFO] downloading {url}")
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 DRND downloader"})
        with urlopen(req, timeout=timeout) as r:
            dest.write_bytes(r.read())
        print(f"[OK] wrote {dest}")
        return True
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"[WARN] failed {url}: {exc}")
        return False


def write_templates(root: Path) -> None:
    (root / "fsigma8_data_TEMPLATE.csv").write_text(
        "label,z,k,fs8_obs,fs8_err,fs8_lcdm,source,notes\n"
        "BOSS_LOWZ_EDIT,0.38,0.15,,,,'fill from chosen RSD paper','do not use until filled'\n"
        "BOSS_CMASS_EDIT,0.51,0.15,,,,'fill from chosen RSD paper','do not use until filled'\n"
        "eBOSS_LRG_EDIT,0.70,0.15,,,,'fill from eBOSS DR16 likelihood/table','do not use until filled'\n"
        "eBOSS_QSO_EDIT,1.48,0.15,,,,'fill from eBOSS DR16 likelihood/table','do not use until filled'\n",
        encoding="utf-8"
    )

    (root / "eg_data_TEMPLATE.csv").write_text(
        "label,z,k,eg_obs,eg_err,eg_lcdm,source,notes\n"
        "EG_EDIT,0.32,0.18,,,,'fill from selected E_G measurement','do not use until filled'\n",
        encoding="utf-8"
    )

    (root / "s8_data_TEMPLATE.csv").write_text(
        "label,z,k,s8_obs,s8_err,s8_lcdm,source,notes\n"
        "DES_Y3_EDIT,0.5,0.2,,,,'fill from DES Y3/selected analysis','do not use until filled'\n"
        "KiDS_EDIT,0.5,0.2,,,,'fill from KiDS/selected analysis','do not use until filled'\n"
        "HSC_EDIT,0.6,0.2,,,,'fill from HSC/selected analysis','do not use until filled'\n",
        encoding="utf-8"
    )

    print("[OK] wrote CSV templates")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="growth_sources")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for fname, url in URLS.items():
        fetch(url, outdir / fname)

    write_templates(outdir)

    print("\nNext:")
    print("  1. Fill fsigma8_data_TEMPLATE.csv with selected literature values.")
    print("  2. Rename to fsigma8_data.csv.")
    print("  3. Run DRND_v71_joint_multi_lensing_growth.py with --fs8-csv fsigma8_data.csv")


if __name__ == "__main__":
    main()

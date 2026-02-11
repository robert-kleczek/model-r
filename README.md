# model-r (v1.0) — DRND preprint sources

This repository contains materials related to the Model R project.
Release v1.0 contains the DRND preprint sources (IOP LaTeX), bibliography, figures, and code used for the toy-network plot.

## Paper
Location: `papers/drnd/`

Included:
- `main.tex`
- `refs.bib` (if used)
- IOP class/style files (`iopart.cls`, `iopams.sty`, `iopart*.clo`, etc.)
- figures in `papers/drnd/figures/`
- (optional) compiled PDF in `papers/drnd/` (if you keep it in-repo)

## Zenodo (archived releases)
- **v1.0.1 DOI:** https://doi.org/10.5281/zenodo.18615078
- **v1.0.2 DOI:** https://doi.org/10.5281/zenodo.18615446

## How to compile the paper (LaTeX)
From `papers/drnd/`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex


### Reproducing the figure
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r papers/drnd/code/requirements.txt
python papers/drnd/code/generate_toy_figure.py

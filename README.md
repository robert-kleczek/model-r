# model_r (v1.0)

This repository contains materials related to the Model R project.
Release v1.0 contains the DRND preprint sources (IOP LaTeX) and bibliography.

## Paper
Location: `papers/drnd/`

- `main.tex`
- `refs.bib`
- IOP class files (`iopart.cls`, `iopams.sty`, `iopart*.clo`)
- figures in `papers/drnd/figures/`

## Compile the paper
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

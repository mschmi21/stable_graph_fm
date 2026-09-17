# Stable Filters for Generative Modeling of Graph Signals

This is the code for the paper "Stable Filters for Generative Modeling of Graph Signals" by Martin Schmidt and Gonzalo Mateos.

Paper is available on [arXiv](https://arxiv.org/abs/2609.18759)

The configuration file is located here: `configs/graph_fm.yaml`.

## How to Run

**To train:**
`python -m scripts.train`

**To evaluate:**
You have two ways to evaluate the model across graph perturbations and generate the plots from the paper:
- Run `python -m scripts.evaluate_1` for (i) the Wasserstein-1 ($W_1$) distance to quantify generative performance.
- Run `python -m scripts.evaluate_2` for (ii) the variation $\|\Phi_1(\mathbf{x}_0; \tilde{\mathbf{L}}) - \Phi_1(\mathbf{x}_0; \mathbf{L})\|$ in generated outputs induced by graph perturbations.

## Details
The details for the graph-aware paths and the training routine can be found in `src/paths/README.md` and `src/training/README.md`.

## fMRI Data Requirement

The fMRI experiments require:

`data/fmri/RS_TCs.mat`

The file must contain a MATLAB variable named `RS_TCs`.

`RS_TCs.mat` is not included in this repository. Place it in `data/fmri/` before running the fMRI experiments.

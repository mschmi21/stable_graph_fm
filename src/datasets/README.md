
## Datasets

### fMRI

The `FMRIData` class loads the fMRI signals from:

```text
data/fmri/RS_TCs.mat
```

and creates train/validation/test splits. It also constructs a functional-connectivity graph from the training signals and its normalized graph Laplacian.

```python
from src.datasets.fmri import FMRIData

fmri = FMRIData("data/fmri")
fmri.prepare_data()
```

Perturbed fMRI graphs can be generated with:

```python
L_perturbed = fmri.perturb_laplacian(epsilon=0.5, seed=0)
```

where `epsilon` controls the fraction of training samples used to estimate the perturbed functional-connectivity graph.

### SBM

The `SBMData` class generates the synthetic stochastic block model graph and Gaussian graph signals used in the experiments.

```python
from src.datasets.sbm import SBMData

sbm = SBMData()
sbm.prepare_data()
```

Relative perturbations of the graph Laplacian can be generated with:

```python
L_perturbed = sbm.perturb_laplacian(epsilon=0.1, seed=0)
```

where `epsilon` controls the spectral norm of the relative perturbation.
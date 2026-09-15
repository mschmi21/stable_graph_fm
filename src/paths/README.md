# Paths

This folder implements the conditional paths used to construct training samples and target vector fields.

Two paths are available:

- `LinearPath`: standard linear interpolation between an initial sample $x_0$ and target sample $z$.
- `TopologicalPath`: graph-aware interpolation induced by a reference drift $H(L)x$.

## Linear Path

The linear path is

$$
x_t = (1-t)x_0 + tz,
$$

with target vector field

$$
u_t = z-x_0.
$$

## Topological Path

Let the graph Laplacian have eigendecomposition

$$
L = V\Lambda V^\top,
$$

with eigenvalues $\lambda_1,\ldots,\lambda_N$. The graph Fourier transform of a signal $x$ is

$$
\hat{x}=V^\top x.
$$

The reference drift is defined by a polynomial graph filter

$$
H(L)=\sum_{k=0}^{K} h_k L^k,
$$

with frequency response

$$
h(\lambda)=\sum_{k=0}^{K} h_k\lambda^k.
$$

For each graph frequency, define

$$
h_i = h(\lambda_i).
$$

If

$$
y = H(L)x,
$$

then the filter acts independently on each graph frequency as

$$
\hat{y}_i = h_i \hat{x}_i.
$$

The implementation generalizes the heat-equation spectral topological path introduced in *Topological Flow Matching* by Kacper Wyrwal et al., corresponding to

$$
h_i = -\kappa \lambda_i,
$$

to an arbitrary polynomial graph filter.

For each graph frequency $i$, the deterministic conditional path is

$$
\hat{x}_{t,i}
=
e^{h_i t}\hat{x}_{0,i}
+
c_{t,i}
\left(
\hat{z}_i-e^{h_i}\hat{x}_{0,i}
\right),
$$

where

$$
c_{t,i}
=
\begin{cases}
e^{h_i(1-t)}
\dfrac{e^{2h_i t}-1}{e^{2h_i}-1},
& h_i\neq 0,\\
t,
& h_i=0.
\end{cases}
$$

The corresponding corrective target vector field is

$$
\hat{u}_{t,i}
=
a_{t,i}
\left(
\hat{z}_i-e^{h_i}\hat{x}_{0,i}
\right),
$$

where

$$
a_{t,i}
=
\begin{cases}
e^{h_i(1-t)}
\dfrac{2h_i}{e^{2h_i}-1},
& h_i\neq 0,\\
1,
& h_i=0.
\end{cases}
$$

The resulting path and target vector field are transformed back to the vertex domain using the inverse graph Fourier transform.

## Usage

```python
from src.paths.linear_path import LinearPath
from src.paths.topological_path import TopologicalPath

linear_path = LinearPath()

topological_path = TopologicalPath(
    L=L,
    coeffs=coeffs,
)
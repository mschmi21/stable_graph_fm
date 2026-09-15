
# Stable Graph Filters

This folder contains the implementation of the stable graph-filter design proposed in the paper.

The filter is parameterized as

$$
h(\lambda)=\sum_{k=1}^{K} h_k \lambda^k,
$$

with $h(0)=0$. The coefficients are chosen to minimize the integral-Lipschitz stability term while enforcing non-amplification and a prescribed smoothness bound.

The optimization problem is

$$
\begin{aligned}
    & \min_{\mathbf{h}, \rho} \rho \\
    \text{s.t.} \quad
    & -\rho \leq \lambda h'(\lambda) \leq \rho,
    && \forall \lambda \in (0,\lambda_{\max}],\\
    & h(\lambda) \leq \frac{1}{2}
    \ln\left(\frac{\eta}{\lambda}\right),
    && \forall \lambda \in (0,\lambda_{\max}],\\
    & h(\lambda) \leq 0,
    && \forall \lambda \in [0,\lambda_{\max}].
\end{aligned}
$$

The spectral interval is discretized over a dense grid and the resulting finite-dimensional linear program is solved using `scipy.optimize.linprog`.

## Usage

```python
from src.filters.stable_filter import solve_stable_filter

result = solve_stable_filter(
    K=4,
    alpha=0.1,
    lambda_max=2.0,
)
```

Note that `alpha` in the implementation corresponds to the smoothness parameter $\eta$ in the paper.
# src/filters/stable_filter.py

import numpy as np
from scipy.optimize import linprog


def solve_stable_filter(
    K: int,
    alpha: float,
    lambda_max: float = 2.0,
    num_grid_points: int = 2000,
):
    """
    Solve

        min_theta max_lambda |lambda h'(lambda)|

    subject to
        h(lambda) <= 0
        h(lambda) <= 0.5 log(alpha / lambda)
        h(0) = 0

    for a polynomial
        h(lambda) = sum_{k=1}^K theta_k lambda^k.

    Returns coefficients including theta_0 = 0.
    """

    lam_grid = np.linspace(0.0, lambda_max, num_grid_points)
    lam_positive = lam_grid[lam_grid > 0]

    # Variables:
    # [theta_1, ..., theta_K, t]
    c = np.zeros(K + 1)
    c[-1] = 1.0

    A_ub = []
    b_ub = []

    # |lambda h'(lambda)| <= t
    for lam in lam_grid:
        psi = np.array([
            k * lam**k
            for k in range(1, K + 1)
        ])

        A_ub.append(np.r_[psi, -1.0])
        b_ub.append(0.0)

        A_ub.append(np.r_[-psi, -1.0])
        b_ub.append(0.0)

    # h(lambda) <= 0
    for lam in lam_grid:
        phi = np.array([
            lam**k
            for k in range(1, K + 1)
        ])

        A_ub.append(np.r_[phi, 0.0])
        b_ub.append(0.0)

    # h(lambda) <= 1/2 log(alpha / lambda)
    for lam in lam_positive:
        phi = np.array([
            lam**k
            for k in range(1, K + 1)
        ])

        A_ub.append(np.r_[phi, 0.0])
        b_ub.append(0.5 * np.log(alpha / lam))

    bounds = [(None, None)] * K + [(0.0, None)]

    result = linprog(
        c,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            f"Stable-filter optimization failed: {result.message}"
        )

    theta = result.x[:-1]
    t = result.x[-1]

    # Include theta_0 = 0
    coeffs = [0.0] + theta.tolist()

    return {
        "K": K,
        "alpha": alpha,
        "lambda_max": lambda_max,
        "coeffs": coeffs,
        "stability_constant": float(t),
    }
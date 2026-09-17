import torch 

def compute_integral_lipschitz(L: torch.Tensor, coeffs: list) -> float:
    """
        C = max {
            max_i |lambda_i h'(lambda_i)|,

            max_{i != j}
            (lambda_i + lambda_j)/2
            * |h(lambda_i) - h(lambda_j)|
              / |lambda_i - lambda_j|
        }.
    """

    # Eigenvalues of L
    eigenvalues, _ = torch.linalg.eigh(L.cpu())
    eigenvalues = eigenvalues.to(L.device)

    # Evaluate h(lambda_i)
    h = torch.zeros_like(eigenvalues)

    for k, theta_k in enumerate(coeffs):
        h += theta_k * eigenvalues**k

    # -----------------------------------------------------
    # Diagonal term:
    # max_i |lambda_i h'(lambda_i)|
    # -----------------------------------------------------

    lambda_hprime = torch.zeros_like(eigenvalues)

    for k, theta_k in enumerate(coeffs[1:], start=1):
        lambda_hprime += k * theta_k * eigenvalues**k

    C_diag = torch.max(
        torch.abs(lambda_hprime)
    )

    # -----------------------------------------------------
    # Off-diagonal term:
    #
    # max_{i != j}
    # (lambda_i + lambda_j)/2
    # * |h(lambda_i) - h(lambda_j)|
    # / |lambda_i - lambda_j|
    # -----------------------------------------------------

    lambda_i = eigenvalues[:, None]
    lambda_j = eigenvalues[None, :]

    h_i = h[:, None]
    h_j = h[None, :]

    delta_lambda = torch.abs(lambda_i - lambda_j)

    numerator = ((lambda_i + lambda_j) / 2.0 * torch.abs(h_i - h_j))

    # Ignore diagonal and repeated eigenvalues
    mask = delta_lambda > 1e-12

    quotient = torch.zeros_like(delta_lambda)

    quotient[mask] = (numerator[mask] / delta_lambda[mask])

    C_pair = torch.max(quotient)

    # Final constant 
    C = torch.maximum(C_diag, C_pair)

    return C.item()
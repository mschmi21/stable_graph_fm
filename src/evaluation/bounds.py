import torch 

def compute_integral_lipschitz(L: torch.Tensor, coeffs: list) -> float:
    """
    Computes

        C = max_i |lambda_i h'(lambda_i)|

    over the eigenvalues of L.
    """

    eigenvalues, _ = torch.linalg.eigh(L.cpu())
    eigenvalues = eigenvalues.to(L.device)

    lambda_hprime = torch.zeros_like(eigenvalues)

    for k, theta_k in enumerate(coeffs[1:], start=1):
        lambda_hprime += k * theta_k * eigenvalues**k

    C = torch.max(torch.abs(lambda_hprime))

    return C.item()
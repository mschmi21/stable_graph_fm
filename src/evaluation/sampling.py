import torch


def polynomial_drift(x: torch.Tensor, L: torch.Tensor, coeffs: list) -> torch.Tensor:

    x_Lk = x
    drift = coeffs[0] * x

    for theta_k in coeffs[1:]:
        x_Lk = x_Lk @ L
        drift = drift + theta_k * x_Lk

    return drift

@torch.no_grad()
def euler_sampler(vector_field, x_init, L, L_tilde, coeffs, num_steps):

    vector_field.eval()

    x = x_init.clone()
    B = x.shape[0]

    dt = 1.0 / num_steps

    # initial norm
    current_norms = torch.linalg.vector_norm(x, ord=2, dim=1)

    max_norm = torch.sqrt(torch.mean(current_norms ** 2)).item()

    for step in range(num_steps):

        t = torch.full(
            (B,),
            step * dt,
            device=x.device,
            dtype=x.dtype,
        )

        # GNN uses nominal graph
        u_t = vector_field(x, t, L)

        # polynomial drift uses perturbed graph
        drift = polynomial_drift(x=x, L=L_tilde, coeffs=coeffs)

        x = x + dt * (drift + u_t)

        # RMS norm across samples
        current_norms = torch.linalg.vector_norm(x, ord=2, dim=1)

        current_rms = torch.sqrt(torch.mean(current_norms ** 2)).item()

        max_norm = max(max_norm, current_rms)

    return x, max_norm
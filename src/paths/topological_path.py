import torch

class TopologicalPath:
    def __init__(self, L: torch.Tensor, coeffs: list, eps: float=1e-8):
        """
        L:
            Symmetric graph Laplacian / GSO, shape (N, N)

        coeffs:
            List [theta_0, theta_1, ..., theta_K] defining

                H(L) = sum_k theta_k L^k

        Example:
            Heat drift H(L) = -kappa L:

                coeffs = [0.0, -kappa]
        """
        self.L = L
        self.coeffs = coeffs
        self.eps = eps

        # L = V diag(lambda) V^T
        self.eigenvalues, self.V = torch.linalg.eigh(L.cpu())
        self.eigenvalues, self.V = self.eigenvalues.to(L.device), self.V.to(L.device)

        # Evaluate the polynomial frequency response
        #
        # h(lambda_i) = sum_k theta_k lambda_i^k
        self.h = torch.zeros_like(self.eigenvalues)

        for k, theta_k in enumerate(coeffs):
            self.h += theta_k * self.eigenvalues ** k


    def GFT(self, x: torch.Tensor) -> torch.Tensor:
        """
        x_hat = V^T x

        Assuming x has shape (..., N).
        """
        return x @ self.V


    def IGFT(self, x_hat: torch.Tensor) -> torch.Tensor:
        """
        x = V x_hat
        """
        return x_hat @ self.V.T


    def sample_xt(self, x_0: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample the deterministic zero-noise bridge x_t.
        """
        x0_hat = self.GFT(x_0)
        z_hat = self.GFT(z)

        h = self.h

        # Make t broadcast with the spectral dimension
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        # Phi_{0,t} = exp(h t)
        phi_0t = torch.exp(h * t)

        # Phi_{0,1} = exp(h)
        phi_01 = torch.exp(h)

        # Bridge coefficient:
        #
        # Sigma_{t,1} / Sigma_{1,1}
        #
        # For h != 0:
        #
        # exp(h(1-t)) (exp(2 h t)-1)/(exp(2h)-1)
        #
        # For h = 0:
        #
        # t
        numerator = torch.expm1(2.0 * h * t)
        denominator = torch.expm1(2.0 * h)

        bridge_coeff = torch.where(
            torch.abs(h) > self.eps,
            torch.exp(h * (1.0 - t))
            * numerator
            / denominator,
            t,
        )

        xt_hat = (phi_0t * x0_hat + bridge_coeff * (z_hat - phi_01 * x0_hat))

        return self.IGFT(xt_hat)


    def target_vector_field(self, x_0: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TFM conditional control u_t.
        """
        x0_hat = self.GFT(x_0)
        z_hat = self.GFT(z)

        h = self.h

        if t.ndim == 1:
            t = t.unsqueeze(-1)

        phi_01 = torch.exp(h)

        # For h != 0:
        #
        # exp(h(1-t)) * 2h/(exp(2h)-1)
        #
        # For h = 0:
        #
        # 1
        denominator = torch.expm1(2.0 * h)

        control_coeff = torch.where(
            torch.abs(h) > self.eps,
            torch.exp(h * (1.0 - t))
            * (2.0 * h)
            / denominator,
            torch.ones_like(h),
        )

        u_hat = (control_coeff * (z_hat - phi_01 * x0_hat))

        return self.IGFT(u_hat)


    def reference_drift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute H(L)x.

        In spectral coordinates this is just

            h(lambda_i) * x_hat_i.
        """
        x_hat = self.GFT(x)

        drift_hat = self.h * x_hat

        return self.IGFT(drift_hat)
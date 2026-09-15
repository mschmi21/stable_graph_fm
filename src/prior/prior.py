import torch

class GaussianPrior:
    def __init__(self, dim: int):
        self.dim = dim

    def sample(self, batch_size: int, device=None, dtype=None,) -> torch.Tensor:
        return torch.randn(batch_size, self.dim, device=device, dtype=dtype)
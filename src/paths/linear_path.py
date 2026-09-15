import torch

class LinearPath: 
    def __init__(self):
        pass

    def sample_xt(self, x_0: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ x_t = (1-t)x_0 tz"""
        return (1-t) * x_0 + t * z
    
    def target_vector_field(self, x_0: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ u_t = z - x_0 """
        return z - x_0

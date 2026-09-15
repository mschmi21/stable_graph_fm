import torch
import torch.nn as nn
from torch.optim import Optimizer

class FMTrainer:
    def __init__(self, model: nn.Module, prior, path, optimizer: Optimizer, L: torch.Tensor):
        self.model = model
        self.prior = prior
        self.path = path
        self.loss_fn = nn.MSELoss()
        self.optimizer = optimizer
        self.L = L

    def train_step(self, z: torch.Tensor) -> float:
        self.model.train()
        batch_size = z.shape[0]
        
        # 1. Sample time t ~ U(0,1)
        t = torch.rand(batch_size, 1, device=z.device)
        
        # 2. Sample prior x_0
        x_0 = self.prior.sample(batch_size, device=z.device, dtype=z.dtype)
        
        # 3. Compute interpolant x_t and target vector field u_t
        x_t = self.path.sample_xt(x_0, z, t)
        u_target = self.path.target_vector_field(x_0, z, t)
        
        # 4. Model prediction
        u_pred = self.model(x_t, t, self.L)
        
        # 5. Calculate loss (mse)
        loss = self.loss_fn(u_pred, u_target)
        
        # 6. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    @torch.no_grad()
    def validation_step(self, z: torch.Tensor) -> float:
        self.model.eval()

        batch_size = z.shape[0]

        # 1. Sample time t ~ U(0,1)
        t = torch.rand(batch_size, 1, device=z.device)

        # 2. Sample prior x_0
        x_0 = self.prior.sample(batch_size, device=z.device, dtype=z.dtype)

        # 3. Compute interpolant x_t and target vector field u_t
        x_t = self.path.sample_xt(x_0, z, t)

        u_target = self.path.target_vector_field(x_0, z, t)

        # 4. Model prediction
        u_pred = self.model(x_t, t, self.L)

        # 5. Calculate loss
        loss = self.loss_fn(u_pred, u_target)

        return loss.item()
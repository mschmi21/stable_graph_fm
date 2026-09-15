import torch 
import torch.nn as nn
import torch.nn.functional as F

from src.models.time import TimePreprocessing


class MLPLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_final_layer: bool = False):
        super().__init__()

        self.is_final_layer = is_final_layer 
        
        self.linear = nn.Linear(in_features, out_features) 
        
    def forward(self, X: torch.Tensor) -> torch.Tensor: 
        
        out = self.linear(X) 
        
        if not self.is_final_layer: 
            out = F.silu(out) 
            
        return out


class MLP(nn.Module):
    """
    Input:
        x: (B, N) or (B, N, 1)
        t: (B, 1)
        L: (N, N)   <-- accepted only to keep the same interface,
                       but completely ignored.

    Output:
        y: (B, N)

    """

    def __init__(self, N: int, F_hidden: list):
        super().__init__()

        self.preprocess = TimePreprocessing()

        self.layers = nn.ModuleList()

        # After preprocessing: (B, N, 1 + time_dim) 
        # Flatten: (B, N * (1 + time_dim))

        in_features = N * self.preprocess.out_dim

        # Hidden layers 
        for out_features in F_hidden: 
            self.layers.append(MLPLayer(in_features, out_features)) 
            in_features = out_features

        # Output layer
        self.out = MLPLayer(in_features, N, is_final_layer=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor, L: torch.Tensor = None) -> torch.Tensor:

        x = self.preprocess(x, t)
        x = x.flatten(start_dim=1)  # (B, N * (1 + time_dim))

        # Apply the MLP 
        for layer in self.layers:
            x = layer(x)

        return self.out(x)
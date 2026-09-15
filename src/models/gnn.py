import torch 
import torch.nn as nn
import torch.nn.functional as F

from src.models.time import TimePreprocessing


class GraphConvLayer(nn.Module):
    """
    computes sum_k L^k X W_k

    where: 
        L: normalized graph laplacian
        W_k: learnable matrices

    input: 
        X: (B, N, F_in)
        L: (N, N)
    
    output: 
        X_out: (B, N, F_out)
    """
    def __init__(self, in_features: int, out_features: int, K: int, is_final_layer: bool = False):
        super().__init__()

        self.K = K
        self.is_final_layer = is_final_layer

        # learnable matrices W_0,...,W_{k-1}; where W_k: (F_in, F_out)
        self.weights = nn.ParameterList([
            nn.Parameter(
                torch.randn(in_features, out_features) * 0.1
            )
            for _ in range(K)
        ])

    def forward(self, X: torch.Tensor, Ls: list[torch.Tensor]) -> torch.Tensor:
        out = 0.0

        for k in range(self.K):
            out = out + Ls[k] @ X @ self.weights[k] # (B, N, F_out)
        
        if not self.is_final_layer:
            out = F.silu(out)

        return out
    

class GNN(nn.Module):
    """
    graph neural network vector field
    u_theta(x_t, t, L)

    input: 
        x: (B, N) or (B, N, 1)
        t: (B, 1)
        L: (N, N)

    output: 
        u: (B, N)
    """

    def __init__(self, L: torch.tensor, K: int, F_hidden: int, num_layers: int):
        super().__init__()
        
        # compute Ls powers once
        self.K = K

        self.preprocess = TimePreprocessing()
        
        self.layers = nn.ModuleList()

        in_channels = self.preprocess.out_dim # 1 + time_embd

        # first graph convolution
        self.layers.append(GraphConvLayer(in_channels, F_hidden, K))

        # hidden graph convolutions
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvLayer(F_hidden, F_hidden, K))

        # output layer
        self.out = GraphConvLayer(F_hidden, 1, K, is_final_layer=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # compute L^0, ..., L^(K-1)
        Ls = []

        L_power = torch.eye(
            L.shape[0],
            device=L.device,
            dtype=L.dtype
        )

        for k in range(self.K):

            Ls.append(L_power)

            if k < self.K - 1:
                L_power = L_power @ L
        
        x = self.preprocess(x, t) # (B, N, 1 + time_emb)

        for layer in self.layers:
            x = layer(x, Ls)

        # remove feature dim
        return self.out(x, Ls).squeeze(-1)

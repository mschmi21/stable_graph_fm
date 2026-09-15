import torch 
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    """
    computes [cos(w_1t),...,cos(w_dt),...,sin(w_dt)]

    input: 
        t: (B,)
    
    output:
        t_emb: (B, time_dim)  
    """
    def __init__(self, dim: int = 128, max_period: float = 10000.0):
        super().__init__()
        
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:

        # t: (B,)
        half = self.dim // 2

        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device)) 
            * torch.arange(0, half, device=t.device) / half
        )

        # args: (B, half)
        args = t[:, None] * freqs[None, :] * 2.0 * torch.pi

        # emb: (B, time_dim)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:  
            emb = F.pad(emb, (0,1))

        return emb

class TimePreprocessing(nn.Module):
    """ 
    concat(x, emb(t)) 

    input: 
        x: (B,N) or (B,N,1)
        t: (B,1)
    """
    def __init__(self, time_dim: int = 128):
        super().__init__()

        self.time_embed = TimeEmbedding(dim=time_dim)
        self.out_dim = 1 + time_dim
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2: 
            # x: (B,N) -> add feature dimension (B, N, 1)
            x = x.unsqueeze(-1)

        B,N,_ = x.shape

        # time embedding
        t_flat = t.squeeze(-1) # t:(B,1) -> (B,)
        emb = self.time_embed(t_flat) # emb: (B, time_dim)
        emb = emb.unsqueeze(1) # emb: (B, 1, time_dim)
        emb = emb.expand(-1,N,-1) # emb: (B, N, time_dim)

        return torch.cat([x, emb], dim=-1) #(B, N, 1 + time_dim)
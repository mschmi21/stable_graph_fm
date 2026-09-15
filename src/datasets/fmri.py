import scipy.io as sio
import torch
import os

class FMRIData:
    def __init__(self, data_path: str, train_prop: float = 0.7, val_prop: float = 0.1, split_seed: int = 0):
        self.data_path = data_path
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.split_seed = split_seed
    
    def prepare_data(self) -> None:
        """
        Loads fMRI signals and structural connectivity (SC). 
        The fMRI signals are split across the time dimension into train/validation/test sets. 
        The functional connectivity graph is computed using only the training fMRI signals. 

        Stores: 
            self.train_signals 
            self.val_signals 
            self.test_signals 
            self.A_fc : functional connectivity adjacency 
            self.L - normalized Laplacian of FC 
        """
        # load signals
        rs_path = os.path.join(self.data_path, "RS_TCs.mat")
        data = sio.loadmat(rs_path)
        signals = torch.tensor(data["RS_TCs"], dtype=torch.float32).T  # (T,N)

        # permute the signals 
        generator = torch.Generator().manual_seed(self.split_seed)
        permutation = torch.randperm(signals.shape[0], generator=generator)
        signals = signals[permutation, :]

        # split data across the time dimension 
        num_timepoints = signals.shape[0]

        train_end = int(self.train_prop * num_timepoints)
        val_end = train_end + int(self.val_prop * num_timepoints)

        self.train_signals = signals[:train_end, :]
        self.val_signals = signals[train_end:val_end, :]
        self.test_signals = signals[val_end:, :]

        self.A_fc = self.build_fc_adjacency(self.train_signals)
        self.L = self.compute_normalized_laplacian(self.A_fc)

    def build_fc_adjacency(self, signals: torch.Tensor) -> torch.Tensor:
        corr = torch.corrcoef(signals.T) 
        corr.fill_diagonal_(0)
        A = torch.clamp(corr, min=0)

        return A
    
    def compute_normalized_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        """L = I - D^{-1/2} A D^{-1/2}"""
        degree = torch.sum(A, dim=1)
        d_inv_sqrt = torch.zeros_like(degree)
        mask = degree > 0 
        d_inv_sqrt[mask] = degree[mask].rsqrt()

        D_inv_sqrt = torch.diag(d_inv_sqrt)

        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        L = I - D_inv_sqrt @ A @ D_inv_sqrt

        return L
    
    def perturb_laplacian(self, epsilon: float, seed: int,) -> torch.Tensor:
        """
        Construct an FC Laplacian using only a fraction p
        of the training samples.

        epsilon = 1.0:
            use 100% of training samples

        epsilon = 0.1:
            use 10% of training samples

        etc.
        """

        num_train = self.train_signals.shape[0]

        num_samples = max(2, int(epsilon * num_train))

        generator = torch.Generator().manual_seed(seed)

        permutation = torch.randperm(num_train,generator=generator)

        indices = permutation[:num_samples]

        subset = self.train_signals[indices]

        A_p = self.build_fc_adjacency(subset)

        L_p = self.compute_normalized_laplacian(A_p)

        return L_p
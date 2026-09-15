import torch

class SBMData:
    def __init__(
        self,
        sizes: tuple[int] = (10, 10),
        p_in: float = 0.8,
        p_out: float = 0.05,
        num_samples: int = 1000,
        train_prop: float = 0.7,
        val_prop: float = 0.1,
        split_seed: int = 0,
    ):
        self.sizes = sizes
        self.p_in = p_in
        self.p_out = p_out
        self.num_samples = num_samples
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.split_seed = split_seed

    def prepare_data(self) -> None:
        """
        Generates a fixed SBM graph and fixed Gaussian signal dataset,
        then creates a train/val/test split according to split_seed.
        """

        # generate fixed SBM adjacency and community labels
        self.A, self.labels = self.generate_sbm_adjacency()

        # normalized Laplacian
        self.L = self.compute_normalized_laplacian(self.A)

        # generate fixed signals
        signals = self.generate_signals()

        # Shuffle samples according to split_seed
        split_generator = torch.Generator().manual_seed(self.split_seed)

        permutation = torch.randperm(
            signals.shape[0],
            generator=split_generator,
        )

        signals = signals[permutation]

        # Train / val / test split
        num_samples = signals.shape[0]

        train_end = int(self.train_prop * num_samples)
        val_end = train_end + int(self.val_prop * num_samples)

        self.train_signals = signals[:train_end]
        self.val_signals = signals[train_end:val_end]
        self.test_signals = signals[val_end:]

    def generate_sbm_adjacency(self):
        """Generate a fixed undirected two-community SBM adjacency matrix."""

        n1, n2 = self.sizes

        # fixed graph across all runs
        generator = torch.Generator().manual_seed(42)

        # Community 1
        A11 = (torch.rand((n1, n1), generator=generator) < self.p_in).float()

        A11.fill_diagonal_(0)
        A11 = torch.triu(A11, diagonal=1)
        A11 = A11 + A11.T

        # Community 2
        A22 = (torch.rand((n2, n2), generator=generator) < self.p_in).float()

        A22.fill_diagonal_(0)
        A22 = torch.triu(A22, diagonal=1)
        A22 = A22 + A22.T

        # Between communities
        A12 = (torch.rand((n1, n2), generator=generator) < self.p_out).float()

        A21 = A12.T

        # Full adjacency matrix
        A = torch.cat(
            [
                torch.cat([A11, A12], dim=1),
                torch.cat([A21, A22], dim=1),
            ],
            dim=0,
        )

        # Community labels
        labels = torch.cat(
            [
                torch.zeros(n1, dtype=torch.long),
                torch.ones(n2, dtype=torch.long),
            ]
        )

        return A, labels

    def generate_signals(self):
        """
        Generate a fixed collection of Gaussian node signals.

        Community 0: N(1, ...)
        Community 1: N(-1, ...)
        """

        num_nodes = sum(self.sizes)

        # fixed signals across all runs
        generator = torch.Generator().manual_seed(42)

        signals = torch.randn(
            self.num_samples,
            num_nodes,
            generator=generator,
        ) * 0.475

        means = torch.where(
            self.labels == 0,
            torch.tensor(1.0),
            torch.tensor(-1.0),
        )

        signals = signals + means

        return signals

    def compute_normalized_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        """L = I - D^{-1/2} A D^{-1/2}"""

        degree = torch.sum(A, dim=1)

        d_inv_sqrt = torch.zeros_like(degree)

        mask = degree > 0
        d_inv_sqrt[mask] = degree[mask].rsqrt()

        D_inv_sqrt = torch.diag(d_inv_sqrt)

        I = torch.eye(
            A.shape[0],
            device=A.device,
            dtype=A.dtype,
        )

        L = I - D_inv_sqrt @ A @ D_inv_sqrt

        return L

    def compute_laplacian(self, A: torch.Tensor) -> torch.Tensor:

        degree = torch.sum(A, dim=1)
        D = torch.diag(degree)

        return D - A
    
    def perturb_laplacian(self, epsilon: float, seed: int) -> torch.Tensor:

        num_nodes = self.L.shape[0]

        generator = torch.Generator().manual_seed(seed)

        # Random dense matrix
        E_random = torch.randn(
            num_nodes,
            num_nodes,
            generator=generator,
            dtype=self.L.dtype,
        )

        # Symmetrize
        E_sym = 0.5 * (E_random + E_random.T)

        # Spectral norm
        current_norm = torch.linalg.matrix_norm(E_sym, ord=2)

        # Scale so ||E||_2 = epsilon
        E = E_sym * (epsilon / current_norm)

        # Relative perturbation model
        L_tilde = self.L + 0.5 * (E @ self.L+ self.L @ E)

        return L_tilde
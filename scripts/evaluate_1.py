import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import get_device, load_config
from src.datasets import get_dataset
from src.models import get_model
from src.prior.prior import GaussianPrior
from src.evaluation.sampling import euler_sampler
from src.evaluation.metrics import compute_wasserstein


@torch.no_grad()
def evaluate_checkpoint(
    config: dict,
    checkpoint_path: str,
    epsilon: float,
) -> float:
    """
    Evaluate one trained checkpoint at one perturbation level epsilon.

    Returns:
        W1 distance for this checkpoint and perturbation level.
    """

    device = get_device()

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    split_seed = checkpoint["split_seed"]
    coeffs = checkpoint["coeffs"]

    # -----------------------------------------------------
    # Rebuild the exact same dataset split
    # -----------------------------------------------------

    dataset = get_dataset(
        config,
        split_seed=split_seed,
    )

    test_signals = dataset.test_signals.to(device)

    # -----------------------------------------------------
    # Perturb the nominal graph
    #
    # We use split_seed as the perturbation seed so that
    # each training run gets a different perturbation.
    # -----------------------------------------------------

    L = dataset.L.to(device)

    L_tilde = dataset.perturb_laplacian(
        epsilon=epsilon,
        seed=split_seed,
    ).to(device)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = get_model(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # -----------------------------------------------------
    # Prior
    # -----------------------------------------------------

    prior = GaussianPrior(
        L_tilde.shape[0]
    )

    num_samples = test_signals.shape[0]

    # -----------------------------------------------------
    # Sample initial condition
    # -----------------------------------------------------

    x0 = prior.sample(
        num_samples,
        device=device,
        dtype=test_signals.dtype,
    )

    # -----------------------------------------------------
    # Generate samples using perturbed graph for the drift
    # -----------------------------------------------------

    generated_samples, _ = euler_sampler(
        vector_field=model,
        x_init=x0,
        L=L,
        L_tilde=L_tilde,
        coeffs=coeffs,
        num_steps=config["evaluation"]["num_steps"],
    )

    # -----------------------------------------------------
    # Wasserstein-1 distance
    # -----------------------------------------------------

    W1 = compute_wasserstein(
        real_samples=test_signals,
        generated_samples=generated_samples,
        p=1,
        blur=config["evaluation"]["blur"],
    )

    return W1


def evaluate_K(
    config: dict,
    K: int,
    epsilon: float,
    num_runs: int,
):
    """
    Evaluate all trained runs for a fixed K and epsilon.

    Returns:
        median W1
        25th percentile W1
        75th percentile W1
        individual W1 values
    """

    W1_runs = []

    dataset_name = config["dataset"]["name"]

    for run in range(num_runs):

        checkpoint_path = (
            f"checkpoints/{dataset_name}/K_{K}/run_{run}.pt"
        )

        W1 = evaluate_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            epsilon=epsilon,
        )

        W1_runs.append(W1)

    W1_runs = np.asarray(W1_runs)

    median = np.median(W1_runs)
    q25 = np.percentile(W1_runs, 25)
    q75 = np.percentile(W1_runs, 75)

    return median, q25, q75, W1_runs


def evaluate_curve(
    config: dict,
    K: int,
    epsilons: np.ndarray,
    num_runs: int,
):
    """
    Evaluate one polynomial degree K over the epsilon grid.

    Returns:
        medians
        25th percentiles
        75th percentiles
    """

    medians = []
    q25s = []
    q75s = []

    print("\n===================================")
    print(f"Evaluating K = {K}")
    print("===================================")

    for epsilon in epsilons:

        median, q25, q75, _ = evaluate_K(
            config=config,
            K=K,
            epsilon=epsilon,
            num_runs=num_runs,
        )

        medians.append(median)
        q25s.append(q25)
        q75s.append(q75)

        print(
            f"epsilon = {epsilon:.4e} | "
            f"W1 median = {median:.6f} | "
            f"25-75% = [{q25:.6f}, {q75:.6f}]"
        )

    return (
        np.asarray(medians),
        np.asarray(q25s),
        np.asarray(q75s),
    )


def main(config_path: str):

    # -----------------------------------------------------
    # Load config
    # -----------------------------------------------------

    config = load_config(config_path)
    dataset_name = config["dataset"]["name"]

    # Polynomial degrees to evaluate
    Ks = [1, 2, 4]

    # Number of independently trained models per K
    num_runs = config["training"]["num_runs"]

    # -----------------------------------------------------
    # Perturbation grid
    #
    # 10 logarithmically spaced points:
    #
    # 10^-3, ..., 10^0
    # -----------------------------------------------------

    if dataset_name == "fmri":
        epsilons = np.logspace(-2.62, -0.05, 10)
        # epsilons = np.arange(0.1, 1.0, 0.1)
    else:
        epsilons = np.logspace(-3, 0, 10)

    # Store results
    results = {}

    # -----------------------------------------------------
    # Evaluate all K
    # -----------------------------------------------------

    for K in Ks:

        medians, q25s, q75s = evaluate_curve(
            config=config,
            K=K,
            epsilons=epsilons,
            num_runs=num_runs,
        )

        results[K] = {
            "median": medians,
            "q25": q25s,
            "q75": q75s,
        }

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    dataset_name = config["dataset"]["name"]

    plt.figure(figsize=(10, 6))

    for K in Ks:

        medians = results[K]["median"]
        q25s = results[K]["q25"]
        q75s = results[K]["q75"]

        plt.plot(
            epsilons,
            medians,
            linewidth=2,
            label=f"K={K}",
        )

        plt.fill_between(
            epsilons,
            q25s,
            q75s,
            alpha=0.2,
        )

    # if dataset_name == "fmri":
    #     plt.xscale("linear")
    # else:
    #     plt.xscale("log")
        
    plt.xscale("log")

    if dataset_name == "fmri":
        plt.xlabel("Training set ratio", fontsize=18)
    else:
        plt.xlabel(r"$\epsilon$", fontsize=18)

    plt.ylabel(r"$W_1$", fontsize=18)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(
        alpha=0.3,
    )

    plt.legend(fontsize=18)

    plt.tight_layout()

    save_path = f"checkpoints/{dataset_name}/evaluate_1.pdf"
    plt.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate structural stability of trained models."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/graph_fm.yaml",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
    )
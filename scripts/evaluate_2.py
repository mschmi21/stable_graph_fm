import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import get_device, load_config
from src.datasets import get_dataset
from src.models import get_model
from src.prior.prior import GaussianPrior
from src.evaluation.sampling import euler_sampler
from src.evaluation.bounds import compute_integral_lipschitz


@torch.no_grad()
def evaluate_checkpoint(
    config: dict,
    checkpoint_path: str,
    epsilon: float,
) -> float:
    """
    Evaluate one trained checkpoint at one perturbation level epsilon.

    Returns:
        Mean L2 norm between the generated outputs using the
        unperturbed and perturbed graph filters.
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
    # Nominal and perturbed graphs
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
        L.shape[0]
    )

    num_samples = test_signals.shape[0]

    # -----------------------------------------------------
    # Sample SAME initial condition for both dynamics
    # -----------------------------------------------------

    x0 = prior.sample(
        num_samples,
        device=device,
        dtype=test_signals.dtype,
    )

    # -----------------------------------------------------
    # Generate with unperturbed graph
    #
    # GNN uses L
    # polynomial drift also uses L
    # -----------------------------------------------------

    generated_nominal, C_H = euler_sampler(
        vector_field=model,
        x_init=x0,
        L=L,
        L_tilde=L,
        coeffs=coeffs,
        num_steps=config["evaluation"]["num_steps"],
    )

    # -----------------------------------------------------
    # Generate with perturbed graph
    #
    # GNN still uses L
    # polynomial drift uses L_tilde
    # -----------------------------------------------------

    generated_perturbed, _ = euler_sampler(
        vector_field=model,
        x_init=x0,
        L=L,
        L_tilde=L_tilde,
        coeffs=coeffs,
        num_steps=config["evaluation"]["num_steps"],
    )

    # -----------------------------------------------------
    # Output difference
    # -----------------------------------------------------

    differences = generated_perturbed - generated_nominal

    norms = torch.linalg.vector_norm(
        differences,
        ord=2,
        dim=1,
    )

    output_norm = norms.mean().item()

    # -----------------------------------------------------
    # Theoretical bound
    # -----------------------------------------------------

    C = compute_integral_lipschitz(L=L, coeffs=coeffs)

    N = L.shape[0]

    gamma = C * (1.0 + 8.0 * np.sqrt(N))

    dataset_name = config["dataset"]["name"]

    if dataset_name == "fmri":
        epsilon_bound = torch.linalg.norm((L - L_tilde).cpu(),ord=2).item()
    else:
        epsilon_bound = epsilon

    theoretical_bound = gamma * C_H * epsilon_bound

    return output_norm, theoretical_bound


def evaluate_K(
    config: dict,
    K: int,
    epsilon: float,
    num_runs: int,
):
    """
    Evaluate all trained runs for a fixed K and epsilon.

    Returns:
        median output norm
        25th percentile
        75th percentile
        individual output norms
    """

    norm_runs = []
    bound_runs = []

    dataset_name = config["dataset"]["name"]

    for run in range(num_runs):

        checkpoint_path = (f"checkpoints/{dataset_name}/K_{K}/run_{run}.pt")

        output_norm, theoretical_bound = evaluate_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            epsilon=epsilon,
        )

        norm_runs.append(output_norm)
        bound_runs.append(theoretical_bound)

    norm_runs = np.asarray(norm_runs)
    bound_runs = np.asarray(bound_runs)

    median = np.median(norm_runs)
    q25 = np.percentile(norm_runs, 25)
    q75 = np.percentile(norm_runs, 75)

    bound_median = np.median(bound_runs)
    bound_q25 = np.percentile(bound_runs, 25)
    bound_q75 = np.percentile(bound_runs, 75)

    return median, q25, q75, bound_median, bound_q25, bound_q75, norm_runs


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
    bound_medians = []
    bound_q25s = []
    bound_q75s = []

    print("\n===================================")
    print(f"Evaluating K = {K}")
    print("===================================")

    for epsilon in epsilons:

        median, q25, q75, bound_median,bound_q25, bound_q75, _ = evaluate_K(
            config=config,
            K=K,
            epsilon=epsilon,
            num_runs=num_runs,
        )

        medians.append(median)
        q25s.append(q25)
        q75s.append(q75)
        bound_medians.append(bound_median)
        bound_q25s.append(bound_q25)
        bound_q75s.append(bound_q75)

        print(
            f"epsilon = {epsilon:.4e} | "
            f"norm median = {median:.6f} | "
            f"25-75% = [{q25:.6f}, {q75:.6f}] | "
            f"bound median = {bound_median:.6f} | "
            f"bound 25-75% = [{bound_q25:.6f}, {bound_q75:.6f}]"
        )

    return (
        np.asarray(medians),
        np.asarray(q25s),
        np.asarray(q75s),
        np.asarray(bound_medians),
        np.asarray(bound_q25s),
        np.asarray(bound_q75s),
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

        medians, q25s, q75s, bounds, bound_q25s, bound_q75s = evaluate_curve(
            config=config,
            K=K,
            epsilons=epsilons,
            num_runs=num_runs,
        )

        results[K] = {
            "median": medians,
            "q25": q25s,
            "q75": q75s,
            "bound": bounds,
            "bound_q25": bound_q25s,
            "bound_q75": bound_q75s,
        }

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    dataset_name = config["dataset"]["name"]

    plt.figure(figsize=(10, 6))

    empirical_lines = []
    bound_lines = []

    for K in Ks:

        medians = results[K]["median"]
        q25s = results[K]["q25"]
        q75s = results[K]["q75"]
        bounds = results[K]["bound"]
        bound_q25s = results[K]["bound_q25"]
        bound_q75s = results[K]["bound_q75"]

        # Empirical curve
        line, = plt.plot(
            epsilons,
            medians,
            linewidth=2,
            label=f"K={K}",
        )

        empirical_lines.append(line)

        color = line.get_color()

        # Empirical 25-75 percentile
        plt.fill_between(
            epsilons,
            q25s,
            q75s,
            alpha=0.2,
            color=color,
        )

        # Theoretical bound - same color, dotted
        bound_line, = plt.plot(
            epsilons,
            bounds,
            linestyle=":",
            linewidth=2.5,
            color=color,
            label=f"K={K}",
        )

        bound_lines.append(bound_line)

        plt.fill_between(
            epsilons,
            bound_q25s,
            bound_q75s,
            alpha=0.1,
            color=color,
        )

    plt.xscale("log")
    plt.yscale("log")

    if dataset_name == "fmri":
        plt.xlabel("Training set ratio", fontsize=18)
    else:
        plt.xlabel(r"$\epsilon$", fontsize=18)

    plt.ylabel(
        r"$\| \Phi_1(\mathbf{x}_0; \tilde{\mathbf{L}}) - \Phi_1(\mathbf{x}_0; \mathbf{L})\|$",
        fontsize=18,
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(
        alpha=0.3,
    )

    ax = plt.gca()

    if dataset_name == "fmri":
        bound_loc = "upper right"
        empirical_loc = "lower left"
    else:
        bound_loc = "upper left"
        empirical_loc = "lower right"

    legend_bound = ax.legend(
        handles=bound_lines,
        labels=[f"K={K}" for K in Ks],
        fontsize=18,
        loc=bound_loc,
        title="Bound",
        title_fontsize=18,
    )

    ax.add_artist(legend_bound)

    ax.legend(
        handles=empirical_lines,
        labels=[f"K={K}" for K in Ks],
        fontsize=18,
        loc=empirical_loc,
        title="Empirical",
        title_fontsize=18,
    )

    plt.tight_layout()

    save_path = f"checkpoints/{dataset_name}/evaluate_2.pdf"

    plt.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate output sensitivity of trained models."
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
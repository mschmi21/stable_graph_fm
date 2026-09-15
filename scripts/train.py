import argparse
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from src.utils import get_device, load_config
from src.datasets import get_dataset
from src.paths import get_path
from src.models import get_model
from src.training.trainer import FMTrainer
from src.prior.prior import GaussianPrior
from src.filters.stable_filter import solve_stable_filter
import os
    
def main(config_path: str):
    # config
    config = load_config(config_path)
    train_cfg = config['training']
    path_cfg = config["path"]
    dataset_name = config["dataset"]["name"]
    DEVICE = get_device()

    # stable polynomial filter 
    stable_filter_cfg = path_cfg["stable_filter"]
    Ks = stable_filter_cfg["Ks"]
    alpha = stable_filter_cfg["alpha"]

    num_runs = train_cfg["num_runs"]

    for K in Ks:

        filter_info = solve_stable_filter(K=K, alpha=alpha, lambda_max= 2, num_grid_points= 2000)
        coeffs = filter_info["coeffs"]

        print(f"Stable filter:")
        print(f"  K = {K}")
        print(f"  alpha = {alpha}")
        print(f"  coeffs = {coeffs}")
        print(
            f"  max |lambda h'(lambda)| = "
            f"{filter_info['stability_constant']:.6f}"
        )

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")

            # dataset
            dataset = get_dataset(config)
            train_signals = dataset.train_signals.to(DEVICE) 
            val_signals = dataset.val_signals.to(DEVICE)

            L = dataset.L.to(DEVICE) # normalized laplacian

            train_dataloader = DataLoader(TensorDataset(train_signals), batch_size=train_cfg['batch_size'], shuffle=True)
            val_dataloader = DataLoader(TensorDataset(val_signals), batch_size=train_cfg['batch_size'], shuffle=False)

            # model, path, prior, loss
            model = get_model(config).to(DEVICE)
            path = get_path(config, L, coeffs = coeffs)
            prior = GaussianPrior(L.shape[0])

            # trainer setup
            optimizer = Adam(model.parameters(), lr=train_cfg['learning_rate'])
            trainer = FMTrainer(model, prior, path, optimizer, L)

            # early stopping 
            EPOCHS = train_cfg["epochs"]

            early_stopping_cfg = train_cfg["early_stopping"]
            early_stopping_enabled = early_stopping_cfg["enabled"]
            patience = early_stopping_cfg["patience"]

            best_val_loss = float("inf")
            epochs_without_improvement = 0

            # store the parameters of the best model 
            best_state = None

            # training loop
            print("Starting training...")

            for epoch in range(EPOCHS):
                # training 
                epoch_loss = 0.0

                for batch in train_dataloader:
                    z = batch[0].to(DEVICE)

                    loss = trainer.train_step(z)

                    epoch_loss += loss

                avg_train_loss = epoch_loss / len(train_dataloader)

                # validation
                epoch_val_loss = 0.0

                for batch in val_dataloader:

                    z = batch[0].to(DEVICE)

                    loss = trainer.validation_step(z)

                    epoch_val_loss += loss

                avg_val_loss = (epoch_val_loss / len(val_dataloader))

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Train Loss: {avg_train_loss:.4f} "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

                # early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    epochs_without_improvement = 0

                else:
                    epochs_without_improvement += 1

                    if early_stopping_enabled and epochs_without_improvement >= patience:
                        print("Early stopping triggered.")
                        break
            
            # save model
            checkpoint_dir = f"checkpoints/{dataset_name}/K_{K}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = f"{checkpoint_dir}/run_{run}.pt"

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "split_seed": run,
                "K": K,
                "alpha": alpha,
                "coeffs": coeffs,
            }

            torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/graph_fm.yaml")
    args = parser.parse_args()

    main(args.config)
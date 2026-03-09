import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
import os
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


from kan_model import KAN, get_kan_hidden_dim, count_parameters
from cfnet import CFNet_Standard, HybridRationalNet, EnsembleResCoFrNet, MoE_Ensemble

# ==========================================
# 1. Environment Configuration
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Data Preparation
# ==========================================
def generate_complex_data(n_samples=10000, epsilon=1e-5):
    """f(X) = (x1 * x2) / (x3 + epsilon)"""
    X_12 = np.random.uniform(-2, 2, (n_samples, 2))
    X_3 = np.random.uniform(1, 3, (n_samples, 1))
    X = np.hstack([X_12, X_3]).astype(np.float32)

    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]

    y = ((x1 * x2) / (x3 + epsilon)).reshape(-1, 1)
    y += 0.02 * np.random.normal(size=y.shape).astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(30/35), random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ==========================================
# 3. CFNet Parameter Calculator
# ==========================================
def get_final_cfnet_params(model_type, input_dim, output_dim, d, p):
    """Calculate CFNet parameter count for matching."""
    if model_type == "standard":
        m = CFNet_Standard(input_dim, output_dim, depth=d, poly_degree=p)
    elif model_type == "hybrid":
        m = HybridRationalNet(input_dim, output_dim, unit_degree=p, num_units=d)
    elif model_type == "boost":
        m = EnsembleResCoFrNet(input_dim, output_dim, shallow_depth=4, poly_degree=p, learning_rate=0.1)
        for _ in range(d): m.add_model()
    elif model_type == "moe":
        hparams = {'input_dim': input_dim, 'output_dim': output_dim,
                   'shallow_depth_per_cofrnet': 4, 'polynomial_degree': p}
        m = MoE_Ensemble(hparams)
        for _ in range(d):
            m.add_expert()
            m.gating.add_expert_gate(initial_center=np.zeros(input_dim))

    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ==========================================
# 4. Training Function
# ==========================================
def train_model(model, train_loader, val_loader, epochs, model_type="kan"):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_rmse": []}

    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_mse = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                val_mse += nn.functional.mse_loss(model(vx), vy, reduction='sum').item()

        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_rmse"].append(np.sqrt(val_mse / len(val_loader.dataset)))

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {history['train_loss'][-1]:.6f}")

    return history

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    epochs_per_stage = 100
    input_dim, output_dim = 3, 1
    grid_size, spline_order = 5, 3

    # Simplified: only best config for maximum fitting accuracy
    params_pair = [{"d": 25, "p": 25}]

    for model_type in ["standard"]:
        for item in params_pair:
            d = item['d']
            p = item['p']

            result_path = f"kan"
            os.makedirs(result_path, exist_ok=True)
            after_pix = f"_{model_type}_d{d}_p{p}"

            # Data preparation
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = generate_complex_data()
            train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                                      batch_size=128, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                                    batch_size=128)

            # Fixed hidden_dim=2 to match CFNet-Hybrid (d=5, p=5) parameter count (~104)
            kan_h = 2

            # Initialize KAN model
            kan_model = KAN(input_dim, output_dim, hidden_dims=[kan_h, kan_h],
                           num_knots=8)

            actual_params = count_parameters(kan_model)
            print(f"[{model_type.upper()}] KAN Hidden: {kan_h} | Actual Params: {actual_params} (target: ~104 for CFNet-Hybrid d=5,p=5)")

            # Train KAN
            total_epochs = d * epochs_per_stage
            history_kan = train_model(kan_model, train_loader, val_loader, total_epochs, "kan")

            # Get test predictions
            kan_model.eval()
            with torch.no_grad():
                tx = torch.from_numpy(X_test).to(DEVICE)
                preds_kan = kan_model(tx).cpu().numpy()

            # Save results
            final_output = {
                "config": {"model_type": model_type, "d": d, "p": p,
                          "kan_h": kan_h, "actual_params": actual_params},
                "history": {"kan": history_kan},
                "test_data": {
                    "y_true": y_test.flatten().tolist(),
                    "y_kan": preds_kan.flatten().tolist(),
                }
            }
            os.makedirs(f"{result_path}/result", exist_ok=True)
            with open(f"{result_path}/result/comparison{after_pix}.json", "w") as f:
                json.dump(final_output, f, indent=4)

            # Visualization
            plt.style.use('seaborn-v0_8-muted')
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            epochs_range = range(1, total_epochs + 1)
            label_name = f"KAN-{model_type.capitalize()}"

            # Training Loss
            axes[0, 0].plot(epochs_range, history_kan["train_loss"], label=label_name, color='C1')
            axes[0, 0].set_title(f"Training Loss: {label_name}", fontweight='bold')
            axes[0, 0].set_yscale('log')
            axes[0, 0].legend()

            # Validation RMSE
            axes[0, 1].plot(epochs_range, history_kan["val_rmse"], label=label_name, color='C1')
            axes[0, 1].set_title(f"Validation RMSE: {label_name}", fontweight='bold')
            axes[0, 1].legend()

            # Scatter
            axes[1, 0].scatter(y_test, preds_kan, alpha=0.5, s=15, label=label_name, color='C1')
            lims = [y_test.min(), y_test.max()]
            axes[1, 0].plot(lims, lims, 'r--', label='Ideal')
            axes[1, 0].set_title("True vs Predicted", fontweight='bold')
            axes[1, 0].legend()

            # Function Fitting
            sort_idx = np.argsort(X_test[:, 0])
            axes[1, 1].scatter(X_test[sort_idx, 0], y_test[sort_idx], c='black', s=10, alpha=0.15, label='GT')
            axes[1, 1].scatter(X_test[sort_idx, 0], preds_kan[sort_idx], s=15, alpha=0.6, label='Pred', color='C1')
            axes[1, 1].set_title(f"Fitting over Dim 0 ({model_type})", fontweight='bold')
            axes[1, 1].legend()

            plt.tight_layout()
            os.makedirs(f'{result_path}/plot', exist_ok=True)
            plt.savefig(f"{result_path}/plot/kan_comparison{after_pix}.png", dpi=300)
            plt.close()

            print(f"Completed: {model_type} d={d}, p={p}\n")
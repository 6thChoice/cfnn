import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import copy
import logging
import os # --- FIX: Import the os module ---

import sys
# Make sure the path to your codebase is correct
sys.path.append("/home/zxc/CodeBase/cofrnet") 

# --- MODIFICATION: Import the iterative MoE regressor ---
from cfnet_regressor_moe_iterative import CoFrNetRegressor_MoE

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")


# --- 1. Dataset Generation (No changes) ---
def create_continual_learning_dataset(num_phases=5, points_per_phase=500, noise_level=0.01):
    """Generates a dataset for continual learning with distinct phases."""
    logging.info("Generating continual learning dataset...")
    datasets = []
    means = np.linspace(-0.8, 0.8, num_phases)
    std_dev = 0.08

    x_full_range = np.linspace(-1, 1, 1000).reshape(-1, 1)
    y_full_range = np.zeros_like(x_full_range)

    for i in range(num_phases):
        x_phase = np.random.normal(loc=means[i], scale=std_dev, size=(points_per_phase, 1))
        x_phase = np.clip(x_phase, -1, 1)
        
        y_true_on_phase = np.zeros_like(x_phase)
        for j in range(num_phases):
             y_true_on_phase += np.exp(-((x_phase - means[j])**2) / (2 * std_dev**2))
        
        y_phase = y_true_on_phase + np.random.normal(0, noise_level, x_phase.shape)
        datasets.append(TensorDataset(torch.from_numpy(x_phase).float(), torch.from_numpy(y_phase).float()))
        
        y_full_range += np.exp(-((x_full_range - means[i])**2) / (2 * std_dev**2))

    return datasets, (x_full_range, y_full_range)

# --- 2. MLP Baseline Model Definition (No changes) ---
class MLP(nn.Module):
    """A simple Multi-Layer Perceptron baseline."""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x): return self.network(x)

# --- 3. MLP Training Function (No changes) ---
def train_mlp_on_phase(model, dataset, epochs=150, lr=1e-3):
    """Trains the MLP model on a single phase of data."""
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

# --- 4. Main Experiment Flow (Core modifications) ---
if __name__ == "__main__":
    # --- FIX: Create directories for saving models and logs ---
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    # --- END FIX ---

    phase_datasets, (x_vis, y_vis) = create_continual_learning_dataset()

    # --- MODIFICATION: Define hyperparameters for the MoE model ---
    moe_hparams = {
        'task_name': 'Continual_MoE_CFNet',
        'input_dim': 1,
        'output_dim': 1,
        'shallow_depth_per_cofrnet': 4,
        'polynomial_degree': 4,
        'learning_rate_adam': 1e-2,
        'weight_decay': 1e-5,
        'batch_size': 128,
        'epochs_per_model': 150, # Epochs for training each candidate MoE model
        'early_stopping_patience': 15,
        'max_experts': 10, # Max experts to try adding in each phase
    }

    # Initialize MLP model
    mlp_model = MLP().to(DEVICE)

    moe_predictions_over_time = []
    mlp_predictions_over_time = []

    # --- MODIFICATION: Cumulative data storage for MoE model ---
    cumulative_X_tensors = []
    cumulative_Y_tensors = []

    # --- MODIFICATION: New phase-based training loop ---
    for phase_idx, dataset in enumerate(phase_datasets):
        logging.info(f"--- Phase {phase_idx + 1}/{len(phase_datasets)} ---")

        # --- MoE Training: Train on all data seen so far (cumulative) ---
        logging.info("Updating cumulative dataset for MoE-CFNet...")
        X_phase, y_phase = dataset.tensors
        cumulative_X_tensors.append(X_phase)
        cumulative_Y_tensors.append(y_phase)
        
        # Concatenate tensors from all phases up to the current one
        X_train_cumulative = torch.cat(cumulative_X_tensors, dim=0).numpy()
        y_train_cumulative = torch.cat(cumulative_Y_tensors, dim=0).numpy()
        
        logging.info(f"Training MoE-CFNet on {X_train_cumulative.shape[0]} cumulative data points...")
        
        # Instantiate a new MoE regressor for this cumulative dataset
        moe_regressor = CoFrNetRegressor_MoE(moe_hparams)
        
        # The train_iterative method handles the building of the MoE model
        moe_regressor.train_iterative(
            train_data=(X_train_cumulative, y_train_cumulative),
            val_data=(X_train_cumulative, y_train_cumulative), # Use train set for validation as in original script
            log_filepath=f"logs/moe_phase_{phase_idx+1}_log.json",
            model_save_path=f"models/moe_phase_{phase_idx+1}.pth"
        )
        
        # --- MLP Training (Unchanged): Train only on the current phase's data ---
        logging.info("Training MLP for current phase...")
        train_mlp_on_phase(mlp_model, dataset)
        
        # --- Record prediction results for visualization ---
        with torch.no_grad():
            x_vis_tensor = torch.from_numpy(x_vis).float().to(DEVICE)
            
            # Get predictions from the trained MoE model
            moe_pred = moe_regressor.predict(x_vis)
            
            # Get predictions from the MLP model
            mlp_pred = mlp_model(x_vis_tensor).cpu().numpy()
            
            moe_predictions_over_time.append(moe_pred)
            mlp_predictions_over_time.append(mlp_pred)
            
    logging.info("All phases completed.")

    # --- 5. Results Visualization (Adjusted for MoE model) ---
    logging.info("Generating result plots...")
    fig, axes = plt.subplots(3, 5, figsize=(25, 12), sharex=True, sharey=True)
    fig.suptitle("Continual Learning: Cumulative MoE-CFNet vs. Naive MLP", fontsize=24, weight='bold')

    for phase_idx in range(5):
        # Plot 1: Data Distribution
        ax = axes[0, phase_idx]
        ax.set_title(f"Phase {phase_idx + 1}", fontsize=18)
        ax.plot(x_vis, y_vis, color='lightgray', linestyle='--', label='True Function')
        # Plot data from all previous phases in a lighter color
        for i in range(phase_idx):
             x_prev, y_prev = phase_datasets[i].tensors
             ax.scatter(x_prev.cpu(), y_prev.cpu(), color='silver', s=5, alpha=0.5)
        # Plot current phase data
        x_train, y_train = phase_datasets[phase_idx].tensors
        ax.scatter(x_train.cpu(), y_train.cpu(), color='black', s=10, label=f'Phase {phase_idx+1} Data')
        if phase_idx == 0: ax.set_ylabel("Data", fontsize=16)

        # Plot 2: MoE-CFNet Predictions
        ax = axes[1, phase_idx]
        ax.plot(x_vis, y_vis, color='lightgray', linestyle='--')
        ax.plot(x_vis, moe_predictions_over_time[phase_idx], color='black')
        if phase_idx == 0: ax.set_ylabel("MoE-CFNet (Cumulative)", fontsize=16)
        
        # Plot 3: MLP Predictions
        ax = axes[2, phase_idx]
        ax.plot(x_vis, y_vis, color='lightgray', linestyle='--')
        ax.plot(x_vis, mlp_predictions_over_time[phase_idx], color='black')
        if phase_idx == 0: ax.set_ylabel("MLP (Naive)", fontsize=16)

    for ax in axes.flat:
        ax.set_ylim(-0.2, 1.5)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("continual_learning_comparison_moe.png")
    logging.info("Comparison plot saved to 'continual_learning_comparison_moe.png'")
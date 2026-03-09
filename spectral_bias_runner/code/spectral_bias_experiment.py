from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

"""
谱偏差实验 - 参数匹配与扫描
阶段一: 所有模型参数量匹配 CFNN
阶段二: Baseline 参数量扫描 (上限 10000)
"""
import os
import sys
sys.path.insert(0, str(CODE_DIR))
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy import fft


from baselines import SIREN, RFFMLP, ChebyshevKAN, count_parameters
from cfnet import CFNet_Standard, HybridRationalNet, EnsembleResCoFrNet, MoE_Ensemble

# ==========================================
# 配置
# ==========================================
ROOT_PATH = str(BASE_DIR / "results")
OUTPUT_DIR = os.path.join(ROOT_PATH, "spectral_bias_experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500  # 增加训练轮数
BATCH_SIZE = 128
LR = 0.001
INPUT_DIM = 3
OUTPUT_DIM = 1

# CFNN 配置
CFNN_CONFIGS = [
    {'type': 'standard', 'd': 5, 'p': 5},
    {'type': 'hybrid', 'd': 5, 'p': 5},
    {'type': 'moe', 'd': 5, 'p': 5},
]

# Baseline 参数量扫描点
BASELINE_PARAM_SWEEP = [50, 100, 200, 400, 800, 1600, 3200, 6400, 10000]

SEEDS = [42, 123, 456]


# ==========================================
# 工具函数
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_data(n_samples=10000, epsilon=1e-5, seed=42):
    np.random.seed(seed)
    X_12 = np.random.uniform(-2, 2, (n_samples, 2))
    X_3 = np.random.uniform(1, 3, (n_samples, 1))
    X = np.hstack([X_12, X_3]).astype(np.float32)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    y = ((x1 * x2) / (x3 + epsilon)).reshape(-1, 1)
    y += 0.02 * np.random.normal(size=y.shape).astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=30/35, random_state=seed)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def compute_power_spectrum(residual):
    """计算功率谱密度"""
    n = len(residual)
    yf = fft.fft(residual)
    power = np.abs(yf[:n//2]) ** 2
    freq = fft.fftfreq(n, 1.0)[:n//2]
    return freq, power


def compute_high_freq_power(power, ratio=0.25):
    """计算高频段平均功率"""
    start = len(power) // 4
    return np.mean(power[start:])


# ==========================================
# 模型创建
# ==========================================
def create_cfnn_model(model_type, d, p):
    if model_type == 'standard':
        return CFNet_Standard(INPUT_DIM, OUTPUT_DIM, depth=d, poly_degree=p)
    elif model_type == 'hybrid':
        return HybridRationalNet(INPUT_DIM, OUTPUT_DIM, unit_degree=p, num_units=d)
    elif model_type == 'moe':
        hparams = {'input_dim': INPUT_DIM, 'output_dim': OUTPUT_DIM,
                   'shallow_depth_per_cofrnet': 4, 'polynomial_degree': p}
        return MoE_Ensemble(hparams)
    elif model_type == 'boost':
        return EnsembleResCoFrNet(INPUT_DIM, OUTPUT_DIM, shallow_depth=4, poly_degree=p, learning_rate=0.1)


def create_baseline_model(model_type, hidden_dim):
    """创建 baseline 模型"""
    if model_type == 'mlp':
        return nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, OUTPUT_DIM)
        )
    elif model_type == 'siren':
        return SIREN(INPUT_DIM, OUTPUT_DIM, hidden_dim=hidden_dim, hidden_layers=2, omega_0=30)
    elif model_type == 'rff_mlp':
        return RFFMLP(INPUT_DIM, OUTPUT_DIM, hidden_dim=hidden_dim, rff_features=hidden_dim*2, sigma=1.0)
    elif model_type == 'chebyshev_kan':
        return ChebyshevKAN(INPUT_DIM, OUTPUT_DIM, hidden_dim=hidden_dim, degree=5, num_layers=2)


def get_hidden_dim_for_params(model_type, target_params):
    """根据目标参数量计算 hidden_dim"""
    if model_type == 'chebyshev_kan':
        # ChebyshevKAN: params ≈ h * (2*3 + h + 1) * 6 ≈ h * (7 + h) * 6
        degree = 5
        d_plus_1 = degree + 1
        for h in range(2, 512):
            params = h * (2*INPUT_DIM + h + OUTPUT_DIM) * d_plus_1
            if params >= target_params * 0.9:
                return h
        return 512
    else:
        # MLP/SIREN/RFF-MLP: params ≈ h*(3+1+2) + h² + h*(1+1) = h² + 7h
        # params = h² + 7h + 1
        a = 1
        b = 7
        c = 1 - target_params
        delta = b**2 - 4*a*c
        if delta < 0:
            return 4
        h = (-b + math.sqrt(delta)) / (2*a)
        return max(4, int(h))


# ==========================================
# 训练函数
# ==========================================
def train_model(model, train_loader, val_loader, epochs=EPOCHS, verbose=False):
    """通用训练函数"""
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, min_lr=1e-6)

    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0
    max_patience = 100

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_mse = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                val_mse += nn.functional.mse_loss(model(vx), vy, reduction='sum').item()

        val_rmse = np.sqrt(val_mse / len(val_loader.dataset))
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return best_val_rmse


def train_cfnn_adaptive(model, model_type, train_loader, val_loader, d, epochs_per_stage=100):
    """训练 Boost/MoE 模型"""
    model.to(DEVICE)
    criterion = nn.MSELoss()

    for stage in range(d):
        if model_type == 'boost':
            model.add_model()
            model.freeze_all_but_latest()
        else:  # moe
            model.add_expert()
            initial_center = np.random.uniform(-2, 2, INPUT_DIM)
            model.gating.add_expert_gate(initial_center=initial_center)

        model.to(DEVICE)
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

        for epoch in range(epochs_per_stage):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

    # 最终评估
    model.eval()
    val_mse = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(DEVICE), vy.to(DEVICE)
            val_mse += nn.functional.mse_loss(model(vx), vy, reduction='sum').item()

    return np.sqrt(val_mse / len(val_loader.dataset))


# ==========================================
# 实验流程
# ==========================================
def evaluate_spectral_bias(model, X_test, y_test):
    """评估模型的谱偏差指标"""
    model.eval()
    with torch.no_grad():
        tx = torch.from_numpy(X_test).to(DEVICE)
        predictions = model(tx).cpu().numpy().flatten()

    residual = y_test.flatten() - predictions
    sort_idx = np.argsort(X_test[:, 0])
    residual_sorted = residual[sort_idx]

    freq, power = compute_power_spectrum(residual_sorted)
    high_freq_power = compute_high_freq_power(power)

    rmse = np.sqrt(np.mean((y_test.flatten() - predictions)**2))

    return {
        'predictions': predictions.tolist(),
        'rmse': float(rmse),
        'high_freq_power': float(high_freq_power),
        'power_spectrum': power.tolist(),
        'freq': freq.tolist(),
    }


def run_phase1_matched_params(data, cfnn_results, seed=42):
    """
    阶段一: 所有 baseline 参数量匹配 CFNN
    """
    print("\n" + "="*60)
    print("Phase 1: Parameter-Matched Comparison")
    print("="*60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=BATCH_SIZE)

    baseline_types = ['mlp', 'siren', 'rff_mlp', 'chebyshev_kan']
    results = {}

    # 获取 CFNN 的参数量作为目标
    target_params_list = [cfnn_results[cf['type']]['params'] for cf in CFNN_CONFIGS]
    target_params = min(target_params_list)  # 使用最小的作为基准
    print(f"Target params (from CFNN): {target_params}")

    for model_type in baseline_types:
        print(f"\n[{model_type.upper()}] target_params={target_params}")

        hidden_dim = get_hidden_dim_for_params(model_type, target_params)
        model = create_baseline_model(model_type, hidden_dim)
        actual_params = count_parameters(model)

        print(f"  hidden_dim={hidden_dim}, actual_params={actual_params}")

        val_rmse = train_model(model, train_loader, val_loader)
        spectral_metrics = evaluate_spectral_bias(model, X_test, y_test)

        results[model_type] = {
            'hidden_dim': hidden_dim,
            'params': actual_params,
            'target_params': target_params,
            'val_rmse': float(val_rmse),
            **spectral_metrics
        }
        print(f"  RMSE: {spectral_metrics['rmse']:.6f}, High-Freq Power: {spectral_metrics['high_freq_power']:.2f}")

    return results


def run_phase2_param_sweep(data, cfnn_results, seed=42):
    """
    阶段二: Baseline 参数量扫描
    """
    print("\n" + "="*60)
    print("Phase 2: Parameter Sweep (Baseline up to 10000)")
    print("="*60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=BATCH_SIZE)

    baseline_types = ['mlp', 'siren', 'rff_mlp', 'chebyshev_kan']
    results = {bt: {} for bt in baseline_types}

    # 获取 CFNN 的最佳高频功率作为目标
    cfnn_best_high_freq = min(r['high_freq_power'] for r in cfnn_results.values())
    print(f"CFNN best high-freq power: {cfnn_best_high_freq:.2f}")

    for target_params in BASELINE_PARAM_SWEEP:
        print(f"\n--- Target Params: {target_params} ---")

        for model_type in baseline_types:
            hidden_dim = get_hidden_dim_for_params(model_type, target_params)
            model = create_baseline_model(model_type, hidden_dim)
            actual_params = count_parameters(model)

            # 跳过参数量过大的情况
            if actual_params > 12000:
                continue

            val_rmse = train_model(model, train_loader, val_loader)
            spectral_metrics = evaluate_spectral_bias(model, X_test, y_test)

            results[model_type][target_params] = {
                'hidden_dim': hidden_dim,
                'params': actual_params,
                'val_rmse': float(val_rmse),
                **spectral_metrics
            }

            print(f"  {model_type}: params={actual_params}, RMSE={spectral_metrics['rmse']:.4f}, "
                  f"HighFreq={spectral_metrics['high_freq_power']:.2f}")

    return results


def run_cfnn_experiments(data, seed=42):
    """训练所有 CFNN 模型"""
    print("\n" + "="*60)
    print("Training CFNN Models")
    print("="*60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=BATCH_SIZE)

    results = {}

    for config in CFNN_CONFIGS:
        model_type = config['type']
        d, p = config['d'], config['p']

        print(f"\n[{model_type.upper()}] d={d}, p={p}")

        model = create_cfnn_model(model_type, d, p)
        params = count_parameters(model)
        print(f"  Params: {params}")

        if model_type in ['boost', 'moe']:
            val_rmse = train_cfnn_adaptive(model, model_type, train_loader, val_loader, d, epochs_per_stage=EPOCHS//d)
        else:
            val_rmse = train_model(model, train_loader, val_loader)

        spectral_metrics = evaluate_spectral_bias(model, X_test, y_test)

        results[model_type] = {
            'd': d,
            'p': p,
            'params': params,
            'val_rmse': float(val_rmse),
            **spectral_metrics
        }
        print(f"  RMSE: {spectral_metrics['rmse']:.6f}, High-Freq Power: {spectral_metrics['high_freq_power']:.2f}")

    return results


def main():
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")

    all_results = {'seeds': {}}

    for seed in SEEDS:
        print(f"\n{'#'*70}")
        print(f"# SEED: {seed}")
        print(f"{'#'*70}")

        set_seed(seed)
        data = generate_data(seed=seed)

        # 训练 CFNN
        cfnn_results = run_cfnn_experiments(data, seed)

        # 阶段一: 参数匹配
        phase1_results = run_phase1_matched_params(data, cfnn_results, seed)

        # 阶段二: 参数扫描
        phase2_results = run_phase2_param_sweep(data, cfnn_results, seed)

        all_results['seeds'][seed] = {
            'cfnn': cfnn_results,
            'phase1_matched': phase1_results,
            'phase2_sweep': phase2_results,
        }

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUTPUT_DIR, f"spectral_bias_experiment_{timestamp}.json")

    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved: {filepath}")

    # 打印汇总
    print_summary(all_results)

    return all_results


def print_summary(results):
    """打印实验汇总"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    for seed, seed_data in results['seeds'].items():
        print(f"\n--- Seed {seed} ---")

        print("\n[CFNN Models]")
        print(f"{'Model':<15} {'Params':>8} {'RMSE':>10} {'High-Freq':>12}")
        print("-"*50)
        for model_type, data in seed_data['cfnn'].items():
            print(f"{model_type:<15} {data['params']:>8} {data['rmse']:>10.6f} {data['high_freq_power']:>12.2f}")

        print("\n[Phase 1: Parameter-Matched Baselines]")
        print(f"{'Model':<15} {'Params':>8} {'RMSE':>10} {'High-Freq':>12}")
        print("-"*50)
        for model_type, data in seed_data['phase1_matched'].items():
            print(f"{model_type:<15} {data['params']:>8} {data['rmse']:>10.6f} {data['high_freq_power']:>12.2f}")

        # 找到每个 baseline 达到 CFNN 最佳高频功率所需的参数量
        cfnn_best = min(d['high_freq_power'] for d in seed_data['cfnn'].values())
        print(f"\n[Phase 2: Params needed to match CFNN best (high-freq={cfnn_best:.2f})]")
        print(f"{'Model':<15} {'Min Params':>12} {'At Params':>12}")
        print("-"*45)

        for model_type in ['mlp', 'siren', 'rff_mlp', 'chebyshev_kan']:
            sweep_data = seed_data['phase2_sweep'].get(model_type, {})
            min_params = None
            for params in sorted(sweep_data.keys()):
                if sweep_data[params]['high_freq_power'] <= cfnn_best * 1.1:  # 10% 容差
                    min_params = params
                    break
            if min_params:
                print(f"{model_type:<15} {min_params:>12} {sweep_data[min_params]['high_freq_power']:>12.2f}")
            else:
                print(f"{model_type:<15} {'Not achieved':>12}")


if __name__ == "__main__":
    main()

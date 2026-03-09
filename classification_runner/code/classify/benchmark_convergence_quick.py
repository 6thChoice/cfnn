"""
快速收敛实验 - 仅运行Waveform数据集，2次重复
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import json
import logging
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))




from cfnet_complicate import CFNet
from hybrid import HybridRationalNet
from kan_classification_base import KAN
from benchmark_convergence import (
    load_waveform, train_with_tracking, ConvergenceTracker,
    evaluate_model, set_seed, DEVICE
)
from benchmark_utils import save_benchmark_results, count_parameters


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_quick_experiment(model_type, model_config, num_runs=2):
    """快速实验，仅2次运行"""

    hparams = {
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 0.005,
        'weight_decay': 1e-4,
        'early_stopping_patience': 20
    }

    logging.info(f"\n{'='*60}")
    logging.info(f"Model: {model_type}, Runs: {num_runs}")
    logging.info(f"{'='*60}")

    all_results = []

    for run_id in range(num_runs):
        run_seed = 42 + run_id
        set_seed(run_seed)

        train_ds, val_ds, test_ds, (n_features, n_classes) = load_waveform(run_seed)

        train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=hparams['batch_size'], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=hparams['batch_size'], shuffle=False)

        # 创建模型
        if model_type == 'CFNet':
            model = CFNet(n_features, n_classes, **model_config)
        elif model_type == 'Hybrid':
            model = HybridRationalNet(n_features, n_classes, **model_config)
        elif model_type == 'KAN':
            model = KAN(n_features, n_classes, **model_config)
        elif model_type == 'MLP':
            from benchmark_utils import MLP
            model = MLP(n_features, model_config['hidden_dim'], n_classes, model_config.get('num_layers', 2))
        else:
            raise ValueError(f"Unknown model: {model_type}")

        params = count_parameters(model)
        logging.info(f"Run {run_id+1}/{num_runs}: Params={params:,}")

        tracker = ConvergenceTracker()
        model, best_val_acc = train_with_tracking(model, train_loader, val_loader, hparams, tracker)
        test_acc = evaluate_model(model, test_loader)

        # 计算收敛指标
        epochs_to_80 = tracker.get_convergence_epoch(0.80)
        epochs_to_84 = tracker.get_convergence_epoch(0.84)

        logging.info(f"  Test Acc: {test_acc:.4f}, Val Acc: {best_val_acc:.4f}")
        logging.info(f"  Epochs to 80%: {epochs_to_80}, Epochs to 84%: {epochs_to_84}")

        all_results.append({
            'run_id': run_id,
            'params': params,
            'test_acc': test_acc,
            'best_val_acc': best_val_acc,
            'epochs_to_80': epochs_to_80,
            'epochs_to_84': epochs_to_84,
            'final_epoch': len(tracker.history['epoch']),
            'convergence_history': tracker.history
        })

    summary = {
        'model_type': model_type,
        'model_config': model_config,
        'mean_test_acc': np.mean([r['test_acc'] for r in all_results]),
        'std_test_acc': np.std([r['test_acc'] for r in all_results]),
        'mean_epochs_to_80': np.mean([r['epochs_to_80'] for r in all_results if r['epochs_to_80'] is not None]),
        'mean_epochs_to_84': np.mean([r['epochs_to_84'] for r in all_results if r['epochs_to_84'] is not None]),
        'all_results': all_results
    }

    return summary


def main():
    setup_logging()

    model_configs = {
        'CFNet': {'depth': 5, 'poly_degree': 3},
        'Hybrid': {'num_units': 10, 'unit_degree': 10},
        'KAN': {'hidden_dim': 32, 'grid_size': 10, 'spline_order': 3},
    }

    all_summaries = []

    for model_type, config in model_configs.items():
        try:
            summary = run_quick_experiment(model_type, config, num_runs=2)
            all_summaries.append(summary)
        except Exception as e:
            logging.error(f"Error with {model_type}: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    save_benchmark_results(
        {'experiment': 'convergence_quick', 'summaries': all_summaries},
        'benchmark_results/convergence_quick_results.json'
    )

    # 打印汇总
    logging.info("\n" + "="*80)
    logging.info("Convergence Summary (Waveform, 2 runs)")
    logging.info("="*80)
    for s in all_summaries:
        logging.info(f"\n{s['model_type']}:")
        logging.info(f"  Test Acc: {s['mean_test_acc']:.4f} ± {s['std_test_acc']:.4f}")
        if s['mean_epochs_to_80']:
            logging.info(f"  Epochs to 80%: {s['mean_epochs_to_80']:.1f}")
        if s['mean_epochs_to_84']:
            logging.info(f"  Epochs to 84%: {s['mean_epochs_to_84']:.1f}")


if __name__ == "__main__":
    main()
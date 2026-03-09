"""
实验三：参数效率极限测试
在严格参数量限制下对比各模型能达到的最佳性能
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


import time

from benchmark_utils import (
    create_model_with_param_limit, save_benchmark_results, count_parameters
)
from benchmark_convergence import (
    train_with_tracking, ConvergenceTracker, evaluate_model, set_seed,
    load_waveform, load_magic, load_credit_card
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%%Y-%m-%d %H:%M:%S'
    )


def run_param_efficiency_experiment(dataset_name, load_data_fn, model_type,
                                    param_limits, hparams, num_runs=5):
    """
    运行参数效率实验

    Args:
        param_limits: 参数量限制列表，如 [1000, 5000, 10000, 50000]
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Dataset: {dataset_name}, Model: {model_type}")
    logging.info(f"{'='*60}")

    results_by_limit = {}

    for param_limit in param_limits:
        logging.info(f"\n参数量限制: {param_limit:,}")

        limit_results = []

        for run_id in range(num_runs):
            run_seed = 42 + run_id
            set_seed(run_seed)

            # 加载数据
            train_ds, val_ds, test_ds, (n_features, n_classes) = load_data_fn(run_seed)

            train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=hparams['batch_size'], shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=hparams['batch_size'], shuffle=False)

            # 创建符合参数量限制的模型
            model, model_config = create_model_with_param_limit(
                model_type, param_limit, n_features, n_classes
            )

            if model is None:
                logging.warning(f"无法创建参数量<{param_limit}的{model_type}模型")
                continue

            actual_params = count_parameters(model)
            logging.info(f"  Run {run_id+1}: Actual params={actual_params:,}, "
                        f"Config={model_config}")

            # 训练
            tracker = ConvergenceTracker()
            try:
                model, best_val_acc = train_with_tracking(
                    model, train_loader, val_loader, hparams, tracker
                )
                test_acc = evaluate_model(model, test_loader)

                limit_results.append({
                    'run_id': run_id,
                    'actual_params': actual_params,
                    'model_config': model_config,
                    'test_acc': test_acc,
                    'best_val_acc': best_val_acc,
                    'epochs_trained': len(tracker.history['epoch'])
                })
            except Exception as e:
                logging.error(f"训练失败: {e}")

        if limit_results:
            results_by_limit[param_limit] = {
                'param_limit': param_limit,
                'mean_test_acc': np.mean([r['test_acc'] for r in limit_results]),
                'std_test_acc': np.std([r['test_acc'] for r in limit_results]),
                'mean_params': np.mean([r['actual_params'] for r in limit_results]),
                'all_runs': limit_results
            }

    return {
        'dataset': dataset_name,
        'model_type': model_type,
        'results_by_limit': results_by_limit
    }


def main():
    setup_logging()

    # 数据集
    datasets = {
        'Waveform': load_waveform,
        'MAGIC': load_magic,
    }

    # 参数量限制
    param_limits = [1000, 5000, 10000, 20000, 50000]

    hparams = {
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.005,
        'weight_decay': 1e-4,
        'early_stopping_patience': 30
    }

    all_results = []

    for dataset_name, load_fn in datasets.items():
        for model_type in ['CFNet', 'Hybrid', 'KAN', 'MLP']:
            try:
                result = run_param_efficiency_experiment(
                    dataset_name=dataset_name,
                    load_data_fn=load_fn,
                    model_type=model_type,
                    param_limits=param_limits,
                    hparams=hparams,
                    num_runs=5
                )
                all_results.append(result)
            except Exception as e:
                logging.error(f"Error: {e}")
                import traceback
                traceback.print_exc()

    # 保存结果
    save_benchmark_results(
        {'experiment': 'param_efficiency', 'results': all_results},
        'benchmark_results/param_efficiency_results.json'
    )

    # 打印Pareto前沿对比表
    logging.info("\n" + "="*80)
    logging.info("参数效率对比 (Pareto前沿)")
    logging.info("="*80)

    for dataset_name in datasets.keys():
        logging.info(f"\n{dataset_name}:")
        dataset_results = [r for r in all_results if r['dataset'] == dataset_name]

        for model_result in dataset_results:
            model_type = model_result['model_type']
            logging.info(f"\n  {model_type}:")
            for limit, stats in model_result['results_by_limit'].items():
                logging.info(f"    {limit:6,} params: "
                           f"Test Acc={stats['mean_test_acc']:.4f}±{stats['std_test_acc']:.4f}, "
                           f"Actual={stats['mean_params']:.0f}")

    logging.info("\n结果已保存至 benchmark_results/param_efficiency_results.json")


if __name__ == "__main__":
    main()
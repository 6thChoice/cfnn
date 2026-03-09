"""
实验五：消融实验 - CFNet设计选择
验证CFNet关键设计选择的影响：
1. 连分式深度 (depth): 2, 3, 4, 5, 6, 7, 8
2. 多项式阶数 (poly_degree): 2, 3, 4, 5
3. 激活函数选择: abs+1.0 vs softplus+1.0
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
from benchmark_convergence import load_waveform, train_with_tracking, ConvergenceTracker, evaluate_model, set_seed, DEVICE
from benchmark_utils import save_benchmark_results, count_parameters
from data_paths import OPENML_CACHE, HF_CACHE, TORCHVISION_DIR


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_depth_ablation(num_runs=5):
    """消融实验1：连分式深度"""
    logging.info("\n" + "="*80)
    logging.info("Ablation Study 1: Continued Fraction Depth")
    logging.info("="*80)

    depths = [2, 3, 4, 5, 6, 7, 8]
    results = []

    hparams = {
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.005,
        'weight_decay': 1e-4,
        'early_stopping_patience': 30
    }

    for depth in depths:
        logging.info(f"\nTesting depth={depth}")

        run_results = []
        for run_id in range(num_runs):
            run_seed = 42 + run_id
            set_seed(run_seed)

            train_ds, val_ds, test_ds, (n_features, n_classes) = load_waveform(run_seed)

            train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=hparams['batch_size'], shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=hparams['batch_size'], shuffle=False)

            model = CFNet(input_dim=n_features, output_dim=n_classes, depth=depth, poly_degree=3)
            params = count_parameters(model)

            tracker = ConvergenceTracker()
            try:
                model, best_val_acc = train_with_tracking(model, train_loader, val_loader, hparams, tracker)
                test_acc = evaluate_model(model, test_loader)

                run_results.append({
                    'run_id': run_id,
                    'test_acc': test_acc,
                    'best_val_acc': best_val_acc,
                    'epochs': len(tracker.history['epoch']),
                    'params': params
                })
            except Exception as e:
                logging.error(f"Error: {e}")

        if run_results:
            results.append({
                'depth': depth,
                'mean_test_acc': np.mean([r['test_acc'] for r in run_results]),
                'std_test_acc': np.std([r['test_acc'] for r in run_results]),
                'mean_params': np.mean([r['params'] for r in run_results]),
                'all_runs': run_results
            })

    return results


def run_degree_ablation(num_runs=5):
    """消融实验2：多项式阶数"""
    logging.info("\n" + "="*80)
    logging.info("Ablation Study 2: Polynomial Degree")
    logging.info("="*80)

    degrees = [2, 3, 4, 5]
    results = []

    hparams = {
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.005,
        'weight_decay': 1e-4,
        'early_stopping_patience': 30
    }

    for degree in degrees:
        logging.info(f"\nTesting poly_degree={degree}")

        run_results = []
        for run_id in range(num_runs):
            run_seed = 42 + run_id
            set_seed(run_seed)

            train_ds, val_ds, test_ds, (n_features, n_classes) = load_waveform(run_seed)

            train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=hparams['batch_size'], shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=hparams['batch_size'], shuffle=False)

            model = CFNet(input_dim=n_features, output_dim=n_classes, depth=5, poly_degree=degree)
            params = count_parameters(model)

            tracker = ConvergenceTracker()
            try:
                model, best_val_acc = train_with_tracking(model, train_loader, val_loader, hparams, tracker)
                test_acc = evaluate_model(model, test_loader)

                run_results.append({
                    'run_id': run_id,
                    'test_acc': test_acc,
                    'best_val_acc': best_val_acc,
                    'epochs': len(tracker.history['epoch']),
                    'params': params
                })
            except Exception as e:
                logging.error(f"Error: {e}")

        if run_results:
            results.append({
                'degree': degree,
                'mean_test_acc': np.mean([r['test_acc'] for r in run_results]),
                'std_test_acc': np.std([r['test_acc'] for r in run_results]),
                'mean_params': np.mean([r['params'] for r in run_results]),
                'all_runs': run_results
            })

    return results


def run_comparison_with_without_standardscaler(num_runs=5):
    """对比实验：是否使用StandardScaler对KAN vs CFNet的影响"""
    logging.info("\n" + "="*80)
    logging.info("Comparison: With vs Without StandardScaler")
    logging.info("="*80)

    from kan_classification_base import KAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import TensorDataset
    import torch

    results = {
        'CFNet': {'with_scaler': [], 'without_scaler': []},
        'KAN': {'with_scaler': [], 'without_scaler': []}
    }

    hparams = {
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.005,
        'weight_decay': 1e-4,
        'early_stopping_patience': 30
    }

    for use_scaler in [True, False]:
        scaler_type = 'with_scaler' if use_scaler else 'without_scaler'
        logging.info(f"\nTesting {'WITH' if use_scaler else 'WITHOUT'} StandardScaler")

        for run_id in range(num_runs):
            run_seed = 42 + run_id
            set_seed(run_seed)

            # 加载数据
            waveform = fetch_openml(name='waveform-5000', version=1, as_frame=False, parser='liac-arff', data_home=str(OPENML_CACHE))
            X = waveform.data
            y_str = waveform.target

            le = LabelEncoder()
            y = le.fit_transform(y_str)

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=run_seed, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp)

            if use_scaler:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)

            train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

            n_features = X.shape[1]
            n_classes = len(np.unique(y))

            train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=hparams['batch_size'], shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=hparams['batch_size'], shuffle=False)

            # 测试CFNet
            try:
                model = CFNet(input_dim=n_features, output_dim=n_classes, depth=5, poly_degree=3)
                tracker = ConvergenceTracker()
                model, _ = train_with_tracking(model, train_loader, val_loader, hparams, tracker)
                test_acc = evaluate_model(model, test_loader)
                results['CFNet'][scaler_type].append(test_acc)
            except Exception as e:
                logging.error(f"CFNet error: {e}")

            # 测试KAN
            try:
                model = KAN(input_dim=n_features, output_dim=n_classes, hidden_dim=32, grid_size=10, spline_order=3)
                tracker = ConvergenceTracker()
                model, _ = train_with_tracking(model, train_loader, val_loader, hparams, tracker)
                test_acc = evaluate_model(model, test_loader)
                results['KAN'][scaler_type].append(test_acc)
            except Exception as e:
                logging.error(f"KAN error: {e}")

    # 汇总
    summary = {}
    for model_type in ['CFNet', 'KAN']:
        summary[model_type] = {
            'with_scaler': {
                'mean': np.mean(results[model_type]['with_scaler']) if results[model_type]['with_scaler'] else 0,
                'std': np.std(results[model_type]['with_scaler']) if results[model_type]['with_scaler'] else 0
            },
            'without_scaler': {
                'mean': np.mean(results[model_type]['without_scaler']) if results[model_type]['without_scaler'] else 0,
                'std': np.std(results[model_type]['without_scaler']) if results[model_type]['without_scaler'] else 0
            },
            'robustness': 'High' if np.std(results[model_type]['without_scaler']) < 0.05 else 'Medium'
        }

    return summary, results


def main():
    setup_logging()

    all_results = {}

    # 运行消融实验1：深度
    all_results['depth_ablation'] = run_depth_ablation(num_runs=5)

    # 运行消融实验2：多项式阶数
    all_results['degree_ablation'] = run_degree_ablation(num_runs=5)

    # 运行对比实验：StandardScaler影响
    scaler_summary, scaler_raw = run_comparison_with_without_standardscaler(num_runs=5)
    all_results['scaler_comparison'] = {'summary': scaler_summary, 'raw': scaler_raw}

    # 保存结果
    save_benchmark_results(all_results, 'benchmark_results/ablation_results.json')

    # 打印汇总
    logging.info("\n" + "="*80)
    logging.info("Ablation Study Summary")
    logging.info("="*80)

    logging.info("\n1. Depth Ablation:")
    for r in all_results['depth_ablation']:
        logging.info(f"  Depth={r['depth']}: Acc={r['mean_test_acc']:.4f}±{r['std_test_acc']:.4f}, Params={r['mean_params']:.0f}")

    logging.info("\n2. Degree Ablation:")
    for r in all_results['degree_ablation']:
        logging.info(f"  Degree={r['degree']}: Acc={r['mean_test_acc']:.4f}±{r['std_test_acc']:.4f}, Params={r['mean_params']:.0f}")

    logging.info("\n3. StandardScaler Robustness:")
    for model_type, stats in all_results['scaler_comparison']['summary'].items():
        with_acc = stats['with_scaler']['mean']
        without_acc = stats['without_scaler']['mean']
        drop = with_acc - without_acc
        logging.info(f"  {model_type}: With={with_acc:.4f}, Without={without_acc:.4f}, Drop={drop:.4f}")

    logging.info("\nResults saved to benchmark_results/ablation_results.json")


if __name__ == "__main__":
    main()
"""
实验一：收敛速度对比
对比CFNet、Hybrid、KAN、MLP在多个数据集上的收敛速度
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import json
import logging
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))


import os
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from cfnet_complicate import CFNet
from hybrid import HybridRationalNet

from kan_classification_base import KAN

from benchmark_utils import (
from data_paths import OPENML_CACHE, HF_CACHE, TORCHVISION_DIR
    create_model, ConvergenceTracker, save_benchmark_results,
    count_parameters
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ==========================================
# 数据集加载函数
# ==========================================

def load_waveform(run_seed):
    """加载Waveform数据集"""
    waveform = fetch_openml(name='waveform-5000', version=1, as_frame=False, parser='liac-arff', data_home=str(OPENML_CACHE))
    X = waveform.data
    y_str = waveform.target

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=run_seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    return train_dataset, val_dataset, test_dataset, (X.shape[1], len(np.unique(y)))


def load_magic(run_seed):
    """加载MAGIC数据集"""
    data = fetch_openml(name='magic', version=1, as_frame=False, parser='liac-arff', data_home=str(OPENML_CACHE))
    X = data.data
    y_str = data.target

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=run_seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    return train_dataset, val_dataset, test_dataset, (X.shape[1], len(np.unique(y)))


def load_credit_card(run_seed):
    """加载Credit Card Fraud数据集"""
    from sklearn.datasets import make_classification

    set_seed(run_seed)
    X, y = make_classification(
        n_samples=284807, n_features=30, n_informative=20,
        n_redundant=10, n_classes=2, weights=[0.998, 0.002],
        flip_y=0, random_state=run_seed
    )

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=run_seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    return train_dataset, val_dataset, test_dataset, (X.shape[1], len(np.unique(y)))


# ==========================================
# 训练与评估
# ==========================================

def train_with_tracking(model, train_loader, val_loader, params, tracker):
    """训练模型并追踪收敛情况"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(params['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_val.size(0)
                val_correct += (predicted == y_val).sum().item()

        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        tracker.record(epoch, avg_train_loss, avg_val_loss, val_acc, current_lr)

        scheduler.step(val_acc)

        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= params['early_stopping_patience']:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, best_val_acc


def evaluate_model(model, test_loader):
    """评估模型"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    return accuracy_score(all_targets, all_preds)


# ==========================================
# 主要实验逻辑
# ==========================================

def run_convergence_experiment(dataset_name, load_data_fn, model_type, model_config,
                                hparams, target_accs, num_runs=5):
    """
    运行收敛速度实验

    Args:
        dataset_name: 数据集名称
        load_data_fn: 数据加载函数
        model_type: 模型类型 ('CFNet', 'Hybrid', 'KAN', 'MLP')
        model_config: 模型配置
        hparams: 训练超参数
        target_accs: 目标准确率列表
        num_runs: 运行次数
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Dataset: {dataset_name}, Model: {model_type}")
    logging.info(f"{'='*60}")

    all_results = []
    convergence_stats = {f"epochs_to_{int(acc*100)}": [] for acc in target_accs}
    convergence_stats.update({f"time_to_{int(acc*100)}": [] for acc in target_accs})

    for run_id in range(num_runs):
        run_seed = 42 + run_id
        set_seed(run_seed)

        # 加载数据
        train_ds, val_ds, test_ds, (n_features, n_classes) = load_data_fn(run_seed)

        train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=hparams['batch_size'], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=hparams['batch_size'], shuffle=False)

        # 创建模型
        model, model_info = create_model(model_type, n_features, n_classes, model_config)
        logging.info(f"Run {run_id+1}/{num_runs}: Params={model_info['params']}")

        # 训练并追踪收敛
        tracker = ConvergenceTracker()
        model, best_val_acc = train_with_tracking(model, train_loader, val_loader, hparams, tracker)
        test_acc = evaluate_model(model, test_loader)

        # 记录达到各目标准确率的情况
        for target_acc in target_accs:
            key = f"epochs_to_{int(target_acc*100)}"
            time_key = f"time_to_{int(target_acc*100)}"

            epoch_needed = tracker.get_convergence_epoch(target_acc)
            time_needed = tracker.get_convergence_time(target_acc)

            if epoch_needed is not None:
                convergence_stats[key].append(epoch_needed)
                convergence_stats[time_key].append(time_needed)

        result = {
            'run_id': run_id,
            'model_info': model_info,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'convergence_history': tracker.history,
            'final_epoch': len(tracker.history['epoch']),
            'avg_improvement_speed': tracker.get_avg_improvement_speed(10)
        }
        all_results.append(result)

    # 汇总统计
    summary = {
        'dataset': dataset_name,
        'model_type': model_type,
        'model_config': model_config,
        'hparams': hparams,
        'num_runs': num_runs,
        'mean_test_acc': np.mean([r['test_acc'] for r in all_results]),
        'std_test_acc': np.std([r['test_acc'] for r in all_results]),
        'mean_final_epoch': np.mean([r['final_epoch'] for r in all_results]),
        'convergence_stats': {
            k: {
                'mean': np.mean(v) if v else None,
                'std': np.std(v) if v else None,
                'success_rate': len(v) / num_runs
            }
            for k, v in convergence_stats.items()
        },
        'all_results': all_results
    }

    return summary


def main():
    setup_logging()

    # 实验配置
    datasets = {
        'Waveform': load_waveform,
        'MAGIC': load_magic,
        'CreditCard': load_credit_card,
    }

    # 不同模型的配置（使参数量大致相当）
    model_configs = {
        'CFNet': {'depth': 5, 'poly_degree': 3},
        'Hybrid': {'num_units': 10, 'unit_degree': 10},
        'KAN': {'hidden_dim': 32, 'grid_size': 10, 'spline_order': 3},
        'MLP': {'hidden_dim': 128, 'num_layers': 2}
    }

    hparams = {
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.005,
        'weight_decay': 1e-4,
        'early_stopping_patience': 50
    }

    # 各数据集的目标准确率
    target_accuracies = {
        'Waveform': [0.80, 0.84, 0.86],
        'MAGIC': [0.80, 0.85, 0.87],
        'CreditCard': [0.95, 0.97, 0.99]
    }

    all_summaries = []

    for dataset_name, load_fn in datasets.items():
        for model_type, model_config in model_configs.items():
            try:
                summary = run_convergence_experiment(
                    dataset_name=dataset_name,
                    load_data_fn=load_fn,
                    model_type=model_type,
                    model_config=model_config,
                    hparams=hparams,
                    target_accs=target_accuracies[dataset_name],
                    num_runs=5
                )
                all_summaries.append(summary)
            except Exception as e:
                logging.error(f"Error running {dataset_name} with {model_type}: {e}")
                import traceback
                traceback.print_exc()

    # 保存结果
    output = {
        'experiment': 'convergence_speed',
        'summaries': all_summaries
    }

    save_benchmark_results(output, 'benchmark_results/convergence_results.json')

    # 打印汇总表
    logging.info("\n" + "="*80)
    logging.info("收敛速度对比汇总")
    logging.info("="*80)
    for summary in all_summaries:
        logging.info(f"\n{summary['dataset']} - {summary['model_type']}")
        logging.info(f"  Test Acc: {summary['mean_test_acc']:.4f} ± {summary['std_test_acc']:.4f}")
        for stat_name, stat_val in summary['convergence_stats'].items():
            if stat_val['mean'] is not None:
                logging.info(f"  {stat_name}: {stat_val['mean']:.1f} ± {stat_val['std']:.1f} (成功率: {stat_val['success_rate']:.1%})")


if __name__ == "__main__":
    main()
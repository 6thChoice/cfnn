"""
实验二：内存占用对比
对比CFNet、Hybrid、KAN、MLP的内存使用情况
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))


import pandas as pd

from benchmark_utils import (
    create_model, measure_model_memory, measure_inference_memory,
    save_benchmark_results, count_parameters
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_memory_benchmark(model_type, model_config, input_dim, output_dim, batch_sizes):
    """
    运行内存基准测试

    Returns:
        dict: 内存使用统计
    """
    model, model_info = create_model(model_type, input_dim, output_dim, model_config)

    results = {
        'model_type': model_type,
        'model_config': model_config,
        'model_info': model_info,
        'param_count': count_parameters(model),
        'param_size_mb': count_parameters(model) * 4 / 1024**2,  # 假设float32
        'batch_results': {}
    }

    for batch_size in batch_sizes:
        try:
            # 训练模式内存（含前向和反向）
            train_mem = measure_model_memory(model, batch_size, input_dim, device=str(DEVICE))

            # 推理模式内存
            infer_mem = measure_inference_memory(model, batch_size, input_dim, device=str(DEVICE))

            results['batch_results'][batch_size] = {
                'train': train_mem,
                'inference': infer_mem
            }
        except Exception as e:
            logging.error(f"Error measuring memory for {model_type} with batch={batch_size}: {e}")
            results['batch_results'][batch_size] = {'error': str(e)}

    return results


def main():
    setup_logging()
    logging.info(f"Running memory benchmark on device: {DEVICE}")

    # 测试配置：模拟不同规模的数据集
    dataset_configs = [
        {'name': 'Small (MAGIC-like)', 'input_dim': 10, 'output_dim': 2, 'batch_sizes': [32, 128, 512]},
        {'name': 'Medium (Waveform-like)', 'input_dim': 40, 'output_dim': 3, 'batch_sizes': [32, 128, 512]},
        {'name': 'Large (IMDB-like)', 'input_dim': 5000, 'output_dim': 2, 'batch_sizes': [32, 64, 128]},
    ]

    # 不同模型的配置（对齐参数量）
    model_configs_aligned = {
        'CFNet': {'depth': 5, 'poly_degree': 3},
        'Hybrid': {'num_units': 10, 'unit_degree': 10},
        'KAN': {'hidden_dim': 32, 'grid_size': 10, 'spline_order': 3},
        'MLP': {'hidden_dim': 128, 'num_layers': 2}
    }

    # 小参数配置（用于对比不同规模）
    model_configs_small = {
        'CFNet': {'depth': 3, 'poly_degree': 3},
        'Hybrid': {'num_units': 4, 'unit_degree': 6},
        'KAN': {'hidden_dim': 16, 'grid_size': 10, 'spline_order': 3},
        'MLP': {'hidden_dim': 64, 'num_layers': 2}
    }

    all_results = []

    # 运行标准配置测试
    logging.info("\n" + "="*80)
    logging.info("标准参数量配置内存测试")
    logging.info("="*80)

    for ds_config in dataset_configs:
        logging.info(f"\n数据集规模: {ds_config['name']}")
        logging.info(f"输入维度: {ds_config['input_dim']}, 输出维度: {ds_config['output_dim']}")

        for model_type, model_config in model_configs_aligned.items():
            try:
                result = run_memory_benchmark(
                    model_type, model_config,
                    ds_config['input_dim'], ds_config['output_dim'],
                    ds_config['batch_sizes']
                )
                result['dataset_config'] = ds_config
                all_results.append(result)

                logging.info(f"\n  {model_type}:")
                logging.info(f"    参数量: {result['param_count']:,}")
                logging.info(f"    参数大小: {result['param_size_mb']:.2f} MB")

                for bs, mem in result['batch_results'].items():
                    if 'train' in mem:
                        logging.info(f"    Batch={bs}: "
                                   f"Forward={mem['train']['forward_peak_mb']:.2f}MB, "
                                   f"Backward={mem['train']['backward_peak_mb']:.2f}MB, "
                                   f"Total={mem['train']['total_peak_mb']:.2f}MB")

            except Exception as e:
                logging.error(f"Error with {model_type} on {ds_config['name']}: {e}")

    # 计算内存效率指标
    logging.info("\n" + "="*80)
    logging.info("内存效率对比分析")
    logging.info("="*80)

    analysis = []
    for result in all_results:
        for bs, mem in result['batch_results'].items():
            if 'train' in mem and 'forward_peak_mb' in mem['train']:
                analysis.append({
                    'dataset': result['dataset_config']['name'],
                    'model': result['model_type'],
                    'batch_size': bs,
                    'params': result['param_count'],
                    'param_mb': result['param_size_mb'],
                    'forward_mb': mem['train']['forward_peak_mb'],
                    'backward_mb': mem['train']['backward_peak_mb'],
                    'total_mb': mem['train']['total_peak_mb'],
                    'inference_mb': mem['inference']['inference_peak_mb'],
                    'memory_per_param': mem['train']['total_peak_mb'] / result['param_count'] * 1e6 if result['param_count'] > 0 else 0
                })

    df = pd.DataFrame(analysis)

    # 按数据集分组对比
    for dataset in df['dataset'].unique():
        logging.info(f"\n数据集: {dataset}")
        ds_df = df[df['dataset'] == dataset]

        for bs in ds_df['batch_size'].unique():
            logging.info(f"\n  Batch Size: {bs}")
            bs_df = ds_df[ds_df['batch_size'] == bs].sort_values('total_mb')

            for _, row in bs_df.iterrows():
                logging.info(f"    {row['model']:10s}: Params={row['params']:6,}, "
                           f"Total={row['total_mb']:8.2f}MB, "
                           f"Inference={row['inference_mb']:8.2f}MB")

    # 保存结果
    output = {
        'experiment': 'memory_usage',
        'device': str(DEVICE),
        'detailed_results': all_results,
        'analysis': analysis
    }

    save_benchmark_results(output, 'benchmark_results/memory_results.json')

    logging.info("\n结果已保存至 benchmark_results/memory_results.json")


if __name__ == "__main__":
    main()
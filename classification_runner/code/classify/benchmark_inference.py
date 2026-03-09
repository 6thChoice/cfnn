"""
实验四：推理速度对比
对比CFNet、Hybrid、KAN、MLP的推理速度
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
    create_model, benchmark_inference, benchmark_inference_cpu_vs_gpu,
    save_benchmark_results, count_parameters
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_inference_benchmark(model_type, model_config, input_dim, output_dim, batch_sizes, num_iters=100):
    """
    运行推理速度基准测试

    Returns:
        dict: 推理速度统计
    """
    model, model_info = create_model(model_type, input_dim, output_dim, model_config)

    # GPU测试
    gpu_results = benchmark_inference(
        model, input_dim, batch_sizes=batch_sizes,
        device='cuda', num_warmup=10, num_iters=num_iters
    )

    # CPU测试（仅选几个batch size）
    cpu_batch_sizes = [1, 32] if 1 in batch_sizes else batch_sizes[:2]
    cpu_results = {}
    for bs in cpu_batch_sizes:
        if bs in batch_sizes:
            result = benchmark_inference(
                model, input_dim, batch_sizes=[bs],
                device='cpu', num_warmup=5, num_iters=num_iters//2
            )
            cpu_results[bs] = result[bs]

    return {
        'model_type': model_type,
        'model_config': model_config,
        'model_info': model_info,
        'param_count': count_parameters(model),
        'gpu_results': gpu_results,
        'cpu_results': cpu_results
    }


def main():
    setup_logging()
    logging.info(f"Running inference benchmark on device: {DEVICE}")

    # 测试配置
    dataset_configs = [
        {'name': 'Small (MAGIC-like)', 'input_dim': 10, 'output_dim': 2,
         'batch_sizes': [1, 16, 32, 64, 128, 256]},
        {'name': 'Medium (Waveform-like)', 'input_dim': 40, 'output_dim': 3,
         'batch_sizes': [1, 16, 32, 64, 128, 256]},
        {'name': 'Large (IMDB-like)', 'input_dim': 5000, 'output_dim': 2,
         'batch_sizes': [1, 16, 32, 64, 128]},
    ]

    # 对齐参数量
    model_configs = {
        'CFNet': {'depth': 5, 'poly_degree': 3},
        'Hybrid': {'num_units': 10, 'unit_degree': 10},
        'KAN': {'hidden_dim': 32, 'grid_size': 10, 'spline_order': 3},
        'MLP': {'hidden_dim': 128, 'num_layers': 2}
    }

    all_results = []

    # 运行测试
    logging.info("\n" + "="*80)
    logging.info("推理速度对比测试")
    logging.info("="*80)

    for ds_config in dataset_configs:
        logging.info(f"\n数据集规模: {ds_config['name']}")
        logging.info(f"输入维度: {ds_config['input_dim']}, 输出维度: {ds_config['output_dim']}")

        for model_type, model_config in model_configs.items():
            try:
                result = run_inference_benchmark(
                    model_type, model_config,
                    ds_config['input_dim'], ds_config['output_dim'],
                    ds_config['batch_sizes'],
                    num_iters=100
                )
                result['dataset_config'] = ds_config
                all_results.append(result)

                logging.info(f"\n  {model_type}:")
                logging.info(f"    参数量: {result['param_count']:,}")

                # GPU结果
                logging.info("    GPU推理速度:")
                for bs, perf in result['gpu_results'].items():
                    logging.info(f"      Batch={bs:3d}: {perf['samples_per_sec']:8.1f} samples/s, "
                               f"{perf['ms_per_batch']:6.2f} ms/batch")

                # CPU结果
                if result['cpu_results']:
                    logging.info("    CPU推理速度:")
                    for bs, perf in result['cpu_results'].items():
                        logging.info(f"      Batch={bs:3d}: {perf['samples_per_sec']:8.1f} samples/s, "
                                   f"{perf['ms_per_batch']:6.2f} ms/batch")

            except Exception as e:
                logging.error(f"Error with {model_type} on {ds_config['name']}: {e}")
                import traceback
                traceback.print_exc()

    # 生成对比分析
    logging.info("\n" + "="*80)
    logging.info("推理速度对比分析")
    logging.info("="*80)

    # 构建DataFrame进行分析
    analysis_rows = []
    for result in all_results:
        for bs, perf in result['gpu_results'].items():
            analysis_rows.append({
                'dataset': result['dataset_config']['name'],
                'input_dim': result['dataset_config']['input_dim'],
                'model': result['model_type'],
                'batch_size': bs,
                'params': result['param_count'],
                'samples_per_sec': perf['samples_per_sec'],
                'ms_per_batch': perf['ms_per_batch'],
                'ms_per_sample': perf['ms_per_sample']
            })

    df = pd.DataFrame(analysis_rows)

    # 按数据集分组对比
    for dataset in df['dataset'].unique():
        logging.info(f"\n数据集: {dataset}")
        ds_df = df[df['dataset'] == dataset]

        # 单样本延迟对比 (batch_size=1)
        single_sample = ds_df[ds_df['batch_size'] == 1].sort_values('ms_per_sample')
        if not single_sample.empty:
            logging.info("\n  单样本延迟 (ms/sample):")
            for _, row in single_sample.iterrows():
                logging.info(f"    {row['model']:10s}: {row['ms_per_sample']*1000:.3f} ms")

        # 大批量吞吐量对比 (最大batch size)
        max_batch_results = []
        for model in ds_df['model'].unique():
            model_df = ds_df[ds_df['model'] == model]
            max_row = model_df.loc[model_df['batch_size'].idxmax()]
            max_batch_results.append(max_row)

        max_batch_df = pd.DataFrame(max_batch_results).sort_values('samples_per_sec', ascending=False)
        logging.info("\n  最大批次吞吐量 (samples/sec):")
        for _, row in max_batch_df.iterrows():
            logging.info(f"    {row['model']:10s} (batch={row['batch_size']:.0f}): "
                       f"{row['samples_per_sec']:.1f} samples/s")

    # 计算速度比率（以KAN为基准）
    logging.info("\n" + "="*80)
    logging.info("相对推理速度（以KAN为基准 = 1.0x）")
    logging.info("="*80)

    for dataset in df['dataset'].unique():
        logging.info(f"\n{dataset}:")
        ds_df = df[df['dataset'] == dataset]

        for bs in [1, 32, 128]:
            if bs in ds_df['batch_size'].values:
                bs_df = ds_df[ds_df['batch_size'] == bs]
                kan_speed = bs_df[bs_df['model'] == 'KAN']['samples_per_sec'].values

                if len(kan_speed) > 0:
                    kan_speed = kan_speed[0]
                    logging.info(f"\n  Batch Size: {bs}")
                    for _, row in bs_df.sort_values('samples_per_sec', ascending=False).iterrows():
                        ratio = row['samples_per_sec'] / kan_speed if kan_speed > 0 else 0
                        logging.info(f"    {row['model']:10s}: {ratio:.2f}x ({row['samples_per_sec']:.1f} samples/s)")

    # 保存结果
    output = {
        'experiment': 'inference_speed',
        'device': str(DEVICE),
        'detailed_results': all_results,
        'analysis': analysis_rows
    }

    save_benchmark_results(output, 'benchmark_results/inference_results.json')

    logging.info("\n结果已保存至 benchmark_results/inference_results.json")


if __name__ == "__main__":
    main()
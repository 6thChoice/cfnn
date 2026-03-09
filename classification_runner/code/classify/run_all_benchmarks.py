"""
CFNet 优势凸显实验 - 主控脚本
运行所有benchmark实验并生成可视化报告

使用方法:
    python run_all_benchmarks.py [--experiment EXP] [--quick]

参数:
    --experiment: 选择运行的实验 (convergence, memory, param_efficiency, inference, all)
    --quick: 快速模式（减少运行次数）
"""

import argparse
import logging
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))


import os
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加路径


def run_experiment(name, module_name, quick_mode=False):
    """运行单个实验"""
    logging.info(f"\n{'='*80}")
    logging.info(f"Running: {name}")
    logging.info(f"{'='*80}")

    start_time = time.time()

    try:
        if module_name == 'benchmark_convergence':
            from benchmark_convergence import main as run_main
            if quick_mode:
                # 修改全局变量减少运行次数
                import benchmark_convergence
                benchmark_convergence.datasets = {
                    'Waveform': benchmark_convergence.load_waveform
                }
        elif module_name == 'benchmark_memory':
            from benchmark_memory import main as run_main
        elif module_name == 'benchmark_param_efficiency':
            from benchmark_param_efficiency import main as run_main
            if quick_mode:
                import benchmark_param_efficiency
                benchmark_param_efficiency.param_limits = [5000, 20000]
        elif module_name == 'benchmark_inference':
            from benchmark_inference import main as run_main
        else:
            logging.error(f"Unknown module: {module_name}")
            return False

        run_main()

        elapsed = time.time() - start_time
        logging.info(f"\n{name} completed in {elapsed:.1f} seconds")
        return True

    except Exception as e:
        logging.error(f"Error running {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_visualization():
    """运行可视化"""
    logging.info(f"\n{'='*80}")
    logging.info("Generating visualizations")
    logging.info(f"{'='*80}")

    try:
        from visualize_benchmark_results import main as viz_main
        viz_main()
        return True
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Run CFNet benchmark experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'convergence', 'memory', 'param_efficiency', 'inference'],
                       help='Which experiment to run')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced runs')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization')

    args = parser.parse_args()

    # 创建结果目录
    os.makedirs('benchmark_results', exist_ok=True)
    os.makedirs('benchmark_results/figures', exist_ok=True)

    experiments = []
    if args.experiment == 'all':
        experiments = [
            ('Convergence Speed', 'benchmark_convergence'),
            ('Memory Usage', 'benchmark_memory'),
            ('Parameter Efficiency', 'benchmark_param_efficiency'),
            ('Inference Speed', 'benchmark_inference'),
        ]
    else:
        exp_map = {
            'convergence': ('Convergence Speed', 'benchmark_convergence'),
            'memory': ('Memory Usage', 'benchmark_memory'),
            'param_efficiency': ('Parameter Efficiency', 'benchmark_param_efficiency'),
            'inference': ('Inference Speed', 'benchmark_inference'),
        }
        experiments = [exp_map[args.experiment]]

    # 运行实验
    results = []
    for name, module in experiments:
        success = run_experiment(name, module, args.quick)
        results.append((name, success))

    # 运行可视化
    if not args.no_viz:
        viz_success = run_visualization()
        results.append(('Visualization', viz_success))

    # 打印总结
    logging.info(f"\n{'='*80}")
    logging.info("Experiment Summary")
    logging.info(f"{'='*80}")
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logging.info(f"  {name:30s}: {status}")

    # 输出结果位置
    logging.info(f"\n{'='*80}")
    logging.info("Results Location")
    logging.info(f"{'='*80}")
    logging.info("  JSON results: benchmark_results/")
    logging.info("  Figures:      benchmark_results/figures/")
    logging.info("  Summary:      benchmark_results/summary_table.md")


if __name__ == "__main__":
    main()
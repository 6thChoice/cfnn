"""
梯度稳定性实验

目的：证明 CFNN 使用 torch.abs() 导致的梯度不稳定问题
对比四种模型：CFNN, CFNN-Boost, CFNN-MoE, CFNN-Hybrid
"""

import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RUNNER_BASE = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import numpy as np
import json
from copy import deepcopy

# 导入模型定义
from cfnet import (
    CFNet_Standard,
    EnsembleResCoFrNet,
    MoE_Ensemble,
    HybridRationalNet
)

# 导入工具函数
from utils import (
    TestFunctions,
    generate_1d_data,
    train_standard,
    train_boost_iterative,
    train_moe_iterative,
    GradientMonitor,
    get_device,
    set_seed,
    save_results,
    count_parameters,
    MODEL_NAME_MAP
)

# 实验配置
EXPERIMENT_CONFIG = {
    # 测试函数
    "test_functions": ["f4_composite", "f7_high_freq", "f9_runge"],
    "function_names": {
        "f4_composite": "Composite (exp * sin)",
        "f7_high_freq": "High Frequency",
        "f9_runge": "Runge Function"
    },

    # 数据参数
    "n_samples": 5000,
    "x_range": (-2, 2),
    "batch_size": 128,

    # 训练参数
    "epochs": 2000,
    "lr": 0.001,
    "grad_clip": 1.0,

    # 深度级别
    "depths": [4, 6, 8],
    "poly_degree": 3,

    # 重复实验
    "seeds": [42, 123, 456],

    # 输出目录
    "output_dir": str(RUNNER_BASE / "results" / "gradient_stability"),
}


def create_model(model_type: str, input_dim: int, output_dim: int,
                 depth: int, poly_degree: int, device: torch.device):
    """创建指定类型的模型"""

    if model_type == "CFNet_Standard":
        model = CFNet_Standard(
            input_dim=input_dim,
            output_dim=output_dim,
            depth=depth,
            poly_degree=poly_degree
        )
    elif model_type == "EnsembleResCoFrNet":
        model = EnsembleResCoFrNet(
            input_dim=input_dim,
            output_dim=output_dim,
            shallow_depth=2,  # Boost使用浅层
            poly_degree=poly_degree,
            learning_rate=0.1
        )
    elif model_type == "MoE_Ensemble":
        hparams = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'shallow_depth_per_cofrnet': 2,
            'polynomial_degree': poly_degree
        }
        model = MoE_Ensemble(hparams)
    elif model_type == "HybridRationalNet":
        model = HybridRationalNet(
            input_dim=input_dim,
            output_dim=output_dim,
            unit_degree=poly_degree,
            num_units=depth
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def run_single_experiment(model_type: str, func_name: str, depth: int,
                          seed: int, config: dict, device: torch.device):
    """运行单次实验"""

    # 生成数据
    func = TestFunctions.get_function(func_name)
    X, Y = generate_1d_data(
        func=func,
        n_samples=config["n_samples"],
        x_range=config["x_range"],
        seed=seed
    )
    X, Y = X.to(device), Y.to(device)

    # 创建模型
    model = create_model(
        model_type=model_type,
        input_dim=1,
        output_dim=1,
        depth=depth,
        poly_degree=config["poly_degree"],
        device=device
    )

    param_count = count_parameters(model)

    # 创建梯度监测器
    grad_monitor = GradientMonitor(model)

    # 根据模型类型选择训练函数
    if model_type == "CFNet_Standard":
        result = train_standard(
            model=model,
            X=X, Y=Y,
            epochs=config["epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            grad_clip=config["grad_clip"],
            device=device,
            grad_monitor=grad_monitor,
            verbose=False
        )
    elif model_type == "EnsembleResCoFrNet":
        result = train_boost_iterative(
            model=model,
            X=X, Y=Y,
            total_units=depth,
            total_epochs=config["epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            grad_clip=config["grad_clip"],
            device=device,
            grad_monitor=grad_monitor,
            verbose=False
        )
    elif model_type == "MoE_Ensemble":
        result = train_moe_iterative(
            model=model,
            X=X, Y=Y,
            total_experts=depth,
            total_epochs=config["epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            grad_clip=config["grad_clip"],
            device=device,
            grad_monitor=grad_monitor,
            verbose=False
        )
    elif model_type == "HybridRationalNet":
        result = train_standard(
            model=model,
            X=X, Y=Y,
            epochs=config["epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            grad_clip=config["grad_clip"],
            device=device,
            grad_monitor=grad_monitor,
            verbose=False
        )

    # 清理
    grad_monitor.remove_hooks()

    # 添加额外信息
    result["model_type"] = MODEL_NAME_MAP.get(model_type, model_type)
    result["model_type_raw"] = model_type
    result["func_name"] = func_name
    result["depth"] = depth
    result["seed"] = seed
    result["n_parameters"] = param_count

    return result


def main():
    """主实验流程"""
    config = EXPERIMENT_CONFIG
    device = get_device()

    print("=" * 60)
    print("梯度稳定性实验")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"测试函数: {config['test_functions']}")
    print(f"深度级别: {config['depths']}")
    print(f"随机种子: {config['seeds']}")
    print("=" * 60)

    # 模型类型
    model_types = [
        "CFNet_Standard",
        "EnsembleResCoFrNet",
        "MoE_Ensemble",
        "HybridRationalNet"
    ]

    # 存储所有结果
    all_results = {
        "config": config,
        "experiments": []
    }

    # 遍历所有配置
    total_experiments = (
        len(config["test_functions"]) *
        len(config["depths"]) *
        len(config["seeds"]) *
        len(model_types)
    )

    current = 0

    for func_name in config["test_functions"]:
        print(f"\n{'=' * 60}")
        print(f"测试函数: {func_name} ({config['function_names'].get(func_name, func_name)})")
        print(f"{'=' * 60}")

        for depth in config["depths"]:
            print(f"\n深度 = {depth}")

            for seed in config["seeds"]:
                set_seed(seed)

                for model_type in model_types:
                    current += 1
                    model_name = MODEL_NAME_MAP.get(model_type, model_type)
                    print(f"  [{current}/{total_experiments}] {model_name} | depth={depth} | seed={seed}...", end=" ")

                    try:
                        result = run_single_experiment(
                            model_type=model_type,
                            func_name=func_name,
                            depth=depth,
                            seed=seed,
                            config=config,
                            device=device
                        )
                        all_results["experiments"].append(result)

                        # 输出简要结果
                        grad_stats = result.get("grad_stats", {})
                        print(f"Loss={result['final_loss']:.6f}, "
                              f"GradStd={grad_stats.get('std_grad_norm', 0):.4f}, "
                              f"NaN={grad_stats.get('nan_count', 0)}, "
                              f"Inf={grad_stats.get('inf_count', 0)}")

                    except Exception as e:
                        print(f"ERROR: {e}")
                        import traceback
                        traceback.print_exc()

    # 保存结果
    os.makedirs(config["output_dir"], exist_ok=True)
    save_results(all_results, os.path.join(config["output_dir"], "gradient_stability_results.json"))

    # 生成汇总统计
    summary = generate_summary(all_results)
    save_results(summary, os.path.join(config["output_dir"], "gradient_stability_summary.json"))

    print("\n" + "=" * 60)
    print("梯度稳定性实验完成")
    print(f"结果已保存至: {config['output_dir']}")
    print("=" * 60)


def generate_summary(results: dict) -> dict:
    """生成实验汇总统计"""

    experiments = results["experiments"]

    summary = {
        "by_function": {},
        "by_depth": {},
        "by_model": {},
        "overall": {}
    }

    # 按函数汇总
    for func_name in results["config"]["test_functions"]:
        func_exps = [e for e in experiments if e["func_name"] == func_name]
        summary["by_function"][func_name] = summarize_experiments(func_exps)

    # 按深度汇总
    for depth in results["config"]["depths"]:
        depth_exps = [e for e in experiments if e["depth"] == depth]
        summary["by_depth"][str(depth)] = summarize_experiments(depth_exps)

    # 按模型汇总
    model_types = set(e["model_type"] for e in experiments)
    for model_type in model_types:
        model_exps = [e for e in experiments if e["model_type"] == model_type]
        summary["by_model"][model_type] = summarize_experiments(model_exps)

    # 整体汇总
    summary["overall"] = summarize_experiments(experiments)

    return summary


def summarize_experiments(experiments: list) -> dict:
    """汇总实验列表"""

    if not experiments:
        return {}

    # 按模型分组
    by_model = {}
    for exp in experiments:
        model = exp["model_type"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(exp)

    summary = {}

    for model, exps in by_model.items():
        grad_stats_list = [e.get("grad_stats", {}) for e in exps if e.get("grad_stats")]

        if grad_stats_list:
            summary[model] = {
                "final_loss_mean": float(np.mean([e["final_loss"] for e in exps])),
                "final_loss_std": float(np.std([e["final_loss"] for e in exps])),
                "grad_std_mean": float(np.mean([s.get("std_grad_norm", 0) for s in grad_stats_list])),
                "grad_std_std": float(np.std([s.get("std_grad_norm", 0) for s in grad_stats_list])),
                "grad_max_mean": float(np.mean([s.get("max_grad_norm", 0) for s in grad_stats_list])),
                "nan_count_total": int(sum(s.get("nan_count", 0) for s in grad_stats_list)),
                "inf_count_total": int(sum(s.get("inf_count", 0) for s in grad_stats_list)),
                "n_experiments": len(exps),
            }

    return summary


if __name__ == "__main__":
    main()

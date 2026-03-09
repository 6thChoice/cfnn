"""
参数扩展性实验

目的：通过逐步增加模型深度/参数量，展示 CFNN 的扩展性瓶颈及改进模型的优势
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
    generate_3d_data,
    split_data,
    train_standard,
    train_boost_iterative,
    train_moe_iterative,
    get_device,
    set_seed,
    save_results,
    count_parameters,
    MODEL_NAME_MAP
)

# 实验配置
EXPERIMENT_CONFIG = {
    # 测试函数
    "test_functions": ["f3d_exp_sum", "f3d_rational"],
    "function_names": {
        "f3d_exp_sum": "3D Exponential Sum",
        "f3d_rational": "3D Rational Function"
    },

    # 数据参数
    "n_samples": 10000,
    "x_range": (-2, 2),
    "train_ratio": 0.65,
    "val_ratio": 0.05,
    "test_ratio": 0.30,
    "batch_size": 128,

    # 训练参数
    "base_epochs": 300,  # 每个单位深度的训练轮数
    "lr": 0.001,
    "grad_clip": 1.0,

    # 深度扩展级别
    "depth_levels": [2, 4, 6, 8, 10, 12, 15, 20],
    "poly_degree": 3,

    # 重复实验
    "seeds": [42, 123, 456],

    # 输出目录
    "output_dir": str(RUNNER_BASE / "results" / "scalability"),
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
    X, Y = generate_3d_data(
        func=func,
        n_samples=config["n_samples"],
        x_range=config["x_range"],
        seed=seed
    )

    # 划分数据集
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(
        X, Y,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        seed=seed
    )

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    # 创建模型
    model = create_model(
        model_type=model_type,
        input_dim=3,
        output_dim=1,
        depth=depth,
        poly_degree=config["poly_degree"],
        device=device
    )

    param_count = count_parameters(model)

    # 计算训练轮数（与深度成正比，保持总计算量相近）
    epochs = config["base_epochs"] * max(1, depth // 2)

    # 根据模型类型选择训练函数
    if model_type == "CFNet_Standard":
        result = train_standard(
            model=model,
            X=X_train, Y=Y_train,
            epochs=epochs,
            lr=config["lr"],
            batch_size=config["batch_size"],
            grad_clip=config["grad_clip"],
            device=device,
            grad_monitor=None,
            verbose=False
        )
    elif model_type == "EnsembleResCoFrNet":
        result = train_boost_iterative(
            model=model,
            X=X_train, Y=Y_train,
            total_units=depth,
            total_epochs=epochs,
            lr=config["lr"],
            batch_size=config["batch_size"],
            grad_clip=config["grad_clip"],
            device=device,
            grad_monitor=None,
            verbose=False
        )
    elif model_type == "MoE_Ensemble":
        result = train_moe_iterative(
            model=model,
            X=X_train, Y=Y_train,
            total_experts=depth,
            total_epochs=epochs,
            lr=config["lr"],
            batch_size=config["batch_size"],
            grad_clip=config["grad_clip"],
            device=device,
            grad_monitor=None,
            verbose=False
        )
    elif model_type == "HybridRationalNet":
        result = train_standard(
            model=model,
            X=X_train, Y=Y_train,
            epochs=epochs,
            lr=config["lr"],
            batch_size=config["batch_size"],
            grad_clip=config["grad_clip"],
            device=device,
            grad_monitor=None,
            verbose=False
        )

    # 计算测试集损失
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = torch.nn.functional.mse_loss(test_pred, Y_test).item()

    # 计算最后100个epoch的loss方差（训练稳定性）
    loss_history = result["loss_history"]
    last_100_variance = float(np.var(loss_history[-100:])) if len(loss_history) >= 100 else 0.0

    # 计算收敛速度（达到最终loss 90% 的epoch数）
    target_loss = result["final_loss"] * 1.1
    convergence_epoch = len(loss_history)
    for i, loss in enumerate(loss_history):
        if loss <= target_loss:
            convergence_epoch = i
            break

    # 添加额外信息
    result["model_type"] = MODEL_NAME_MAP.get(model_type, model_type)
    result["model_type_raw"] = model_type
    result["func_name"] = func_name
    result["depth"] = depth
    result["seed"] = seed
    result["n_parameters"] = param_count
    result["test_loss"] = test_loss
    result["loss_variance_last_100"] = last_100_variance
    result["convergence_epoch"] = convergence_epoch
    result["epochs_trained"] = epochs
    result["training_successful"] = not np.isnan(result["final_loss"]) and not np.isinf(result["final_loss"])

    return result


def main():
    """主实验流程"""
    config = EXPERIMENT_CONFIG
    device = get_device()

    print("=" * 60)
    print("参数扩展性实验")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"测试函数: {config['test_functions']}")
    print(f"深度级别: {config['depth_levels']}")
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
        len(config["depth_levels"]) *
        len(config["seeds"]) *
        len(model_types)
    )

    current = 0

    for func_name in config["test_functions"]:
        print(f"\n{'=' * 60}")
        print(f"测试函数: {func_name} ({config['function_names'].get(func_name, func_name)})")
        print(f"{'=' * 60}")

        for depth in config["depth_levels"]:
            print(f"\n深度/单元数 = {depth}")

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
                        status = "OK" if result["training_successful"] else "FAILED"
                        print(f"TestLoss={result['test_loss']:.6f}, "
                              f"Params={result['n_parameters']}, "
                              f"Status={status}")

                    except Exception as e:
                        print(f"ERROR: {e}")
                        import traceback
                        traceback.print_exc()

    # 保存结果
    os.makedirs(config["output_dir"], exist_ok=True)
    save_results(all_results, os.path.join(config["output_dir"], "scalability_results.json"))

    # 生成汇总统计
    summary = generate_summary(all_results)
    save_results(summary, os.path.join(config["output_dir"], "scalability_summary.json"))

    print("\n" + "=" * 60)
    print("参数扩展性实验完成")
    print(f"结果已保存至: {config['output_dir']}")
    print("=" * 60)


def generate_summary(results: dict) -> dict:
    """生成实验汇总统计"""

    experiments = results["experiments"]

    summary = {
        "by_function": {},
        "by_depth": {},
        "by_model": {},
        "scalability_curves": {}
    }

    # 按函数汇总
    for func_name in results["config"]["test_functions"]:
        func_exps = [e for e in experiments if e["func_name"] == func_name]
        summary["by_function"][func_name] = summarize_experiments(func_exps)

        # 生成扩展性曲线数据
        summary["scalability_curves"][func_name] = generate_scalability_curves(func_exps)

    # 按深度汇总
    for depth in results["config"]["depth_levels"]:
        depth_exps = [e for e in experiments if e["depth"] == depth]
        summary["by_depth"][str(depth)] = summarize_experiments(depth_exps)

    # 按模型汇总
    model_types = set(e["model_type"] for e in experiments)
    for model_type in model_types:
        model_exps = [e for e in experiments if e["model_type"] == model_type]
        summary["by_model"][model_type] = summarize_experiments(model_exps)

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
        successful_exps = [e for e in exps if e.get("training_successful", False)]

        if successful_exps:
            summary[model] = {
                "test_loss_mean": float(np.mean([e["test_loss"] for e in successful_exps])),
                "test_loss_std": float(np.std([e["test_loss"] for e in successful_exps])),
                "train_loss_mean": float(np.mean([e["final_loss"] for e in successful_exps])),
                "params_mean": float(np.mean([e["n_parameters"] for e in exps])),
                "loss_variance_mean": float(np.mean([e.get("loss_variance_last_100", 0) for e in successful_exps])),
                "convergence_epoch_mean": float(np.mean([e.get("convergence_epoch", len(e["loss_history"])) for e in successful_exps])),
                "success_rate": len(successful_exps) / len(exps),
                "n_experiments": len(exps),
            }

    return summary


def generate_scalability_curves(experiments: list) -> dict:
    """生成扩展性曲线数据"""

    # 按模型和深度分组
    by_model_depth = {}
    for exp in experiments:
        model = exp["model_type"]
        depth = exp["depth"]
        key = f"{model}_{depth}"
        if key not in by_model_depth:
            by_model_depth[key] = []
        by_model_depth[key].append(exp)

    # 按模型组织
    curves = {}
    for exp in experiments:
        model = exp["model_type"]
        if model not in curves:
            curves[model] = {
                "depths": [],
                "test_loss_mean": [],
                "test_loss_std": [],
                "parameters": [],
            }

    # 聚合数据
    for key, exps in by_model_depth.items():
        model = exps[0]["model_type"]
        depth = exps[0]["depth"]
        successful = [e for e in exps if e.get("training_successful", False)]

        if successful:
            curves[model]["depths"].append(depth)
            curves[model]["test_loss_mean"].append(float(np.mean([e["test_loss"] for e in successful])))
            curves[model]["test_loss_std"].append(float(np.std([e["test_loss"] for e in successful])))
            curves[model]["parameters"].append(int(exps[0]["n_parameters"]))

    # 排序
    for model in curves:
        if curves[model]["depths"]:
            combined = sorted(zip(
                curves[model]["depths"],
                curves[model]["test_loss_mean"],
                curves[model]["test_loss_std"],
                curves[model]["parameters"]
            ))
            curves[model]["depths"] = [x[0] for x in combined]
            curves[model]["test_loss_mean"] = [x[1] for x in combined]
            curves[model]["test_loss_std"] = [x[2] for x in combined]
            curves[model]["parameters"] = [x[3] for x in combined]

    return curves


if __name__ == "__main__":
    main()

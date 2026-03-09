"""
方案1帕累托前沿实验配置：参数量-性能权衡分析

在不同噪声比例下，测试各模型在不同参数量级别的性能表现，
绘制帕累托前沿曲线（X轴：参数量，Y轴：测试MSE）
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, List, Optional
from .base_config import BASE_CONFIG

# ==========================================
# 1. 实验参数设置
# ==========================================

# 帕累托参数量级别（目标参数量）
PARETO_LEVELS = [50, 100, 200, 400, 800, 1200]

# 噪声比例水平
NOISE_RATIOS = [0.2, 0.4, 0.6, 0.8, 0.9]

# 输入/输出维度（根据实验数据）
INPUT_DIM = 11  # 4 true + 4 noise + 2 redundant + 1 deceptive (for 50% noise ratio baseline)
OUTPUT_DIM = 1

# ==========================================
# 2. 参数量计算函数
# ==========================================

def count_params_standard(input_dim: int, output_dim: int, depth: int, poly_degree: int) -> int:
    """
    计算CFNet-Standard的参数量

    PolynomialTerm: (input_dim * output_dim + output_dim) + (output_dim * (poly_degree + 1))
    betas: (depth - 1)
    """
    # Linear projection params: weight + bias
    linear_params = input_dim * output_dim + output_dim
    # Polynomial coeffs: output_dim * (degree + 1)
    poly_params = output_dim * (poly_degree + 1)
    # Total per term
    term_params = linear_params + poly_params
    # betas
    beta_params = max(0, depth - 1)
    return depth * term_params + beta_params


def count_params_hybrid(input_dim: int, output_dim: int, num_units: int, unit_degree: int) -> int:
    """
    计算CFNet-Hybrid的参数量

    Linear skip: input_dim * output_dim + output_dim
    RationalUnit: 2 * PolynomialTerm
    """
    # Linear skip
    linear_params = input_dim * output_dim + output_dim
    # Each RationalUnit has 2 PolynomialTerms
    unit_linear_params = input_dim * output_dim + output_dim
    unit_poly_params = output_dim * (unit_degree + 1)
    unit_params = 2 * (unit_linear_params + unit_poly_params)
    return linear_params + num_units * unit_params


def count_params_boost_per_stage(input_dim: int, output_dim: int, shallow_depth: int,
                                  poly_degree: int) -> int:
    """
    计算CFNet-Boost每个stage的参数量（基于CFNet_Core）

    CFNet_Core使用per-output beta，所以beta_params = (depth - 1) * output_dim
    """
    # Same as Standard but with per-output beta
    linear_params = input_dim * output_dim + output_dim
    poly_params = output_dim * (poly_degree + 1)
    term_params = linear_params + poly_params
    # Per-output betas
    beta_params = max(0, shallow_depth - 1) * output_dim
    return shallow_depth * term_params + beta_params


def count_params_moe_per_expert(input_dim: int, output_dim: int, shallow_depth: int,
                                 poly_degree: int) -> int:
    """
    计算CFNet-MoE每个expert的参数量
    """
    return count_params_boost_per_stage(input_dim, output_dim, shallow_depth, poly_degree)


def count_params_moe_gating(input_dim: int, num_experts: int) -> int:
    """
    计算MoE门控网络的参数量

    RBFGatingNetwork: centers + widths
    centers: num_experts * input_dim
    widths: num_experts
    """
    return input_dim * num_experts + num_experts


def count_params_kan(input_dim: int, output_dim: int, hidden_dim: int,
                     num_layers: int = 2, grid_size: int = 10, spline_order: int = 3) -> int:
    """
    计算KAN的参数量

    每层: in_features * out_features * (grid_size + spline_order) + out_features (residual)
    LayerNorm: 2 * hidden_dim (除了最后一层)
    """
    n_basis = grid_size + spline_order
    total = 0
    dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

    for i in range(len(dims) - 1):
        # coeffs + residual_weight
        layer_params = dims[i] * dims[i + 1] * n_basis + dims[i + 1]
        # LayerNorm参数 (除了最后一层)
        if i < len(dims) - 2:
            layer_params += 2 * dims[i + 1]  # LayerNorm有weight和bias
        total += layer_params

    return total


def count_params_mlp(input_dim: int, output_dim: int, hidden_dim: int, num_layers: int = 2) -> int:
    """
    计算MLP的参数量

    与实际BaselineMLP模型一致:
    - num_layers个隐藏层 (input->hidden, hidden->hidden, ...)
    - 1个输出层 (hidden->output)

    总参数量 = 第1层 + (num_layers-1)个隐藏层 + 输出层
    """
    # 第1层: input_dim -> hidden_dim
    first_layer = input_dim * hidden_dim + hidden_dim

    # 中间隐藏层: hidden_dim -> hidden_dim (共num_layers-1个)
    hidden_layers = (num_layers - 1) * (hidden_dim * hidden_dim + hidden_dim) if num_layers > 1 else 0

    # 输出层: hidden_dim -> output_dim
    output_layer = hidden_dim * output_dim + output_dim

    return first_layer + hidden_layers + output_layer


# ==========================================
# 3. 参数对齐搜索函数
# ==========================================

def get_aligned_standard(input_dim: int, output_dim: int, target_params: int,
                         tolerance: float = 0.20) -> Dict[str, Any]:
    """
    搜索与目标参数量最接近的Standard配置

    搜索空间: depth [2-50], poly_degree [2-12]
    """
    best_config = None
    best_diff = float('inf')

    # 动态调整搜索范围 - 扩大以支持高参数量
    max_depth = min(60, max(10, target_params // 15))
    max_degree = min(15, max(6, target_params // 40))

    for depth in range(2, max_depth):
        for poly_degree in range(2, max_degree):
            params = count_params_standard(input_dim, output_dim, depth, poly_degree)
            diff = abs(params - target_params) / target_params

            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'depth': depth,
                    'poly_degree': poly_degree,
                    'actual_params': params
                }

            if diff < tolerance:
                return best_config

    return best_config


def get_aligned_hybrid(input_dim: int, output_dim: int, target_params: int,
                       tolerance: float = 0.15) -> Dict[str, Any]:
    """
    搜索与目标参数量最接近的Hybrid配置

    搜索空间: unit_degree [2-6], num_units [1-50]
    """
    best_config = None
    best_diff = float('inf')

    for unit_degree in range(2, 7):
        for num_units in range(1, 51):
            params = count_params_hybrid(input_dim, output_dim, num_units, unit_degree)
            diff = abs(params - target_params) / target_params

            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'unit_degree': unit_degree,
                    'num_units': num_units,
                    'actual_params': params
                }

            if diff < tolerance:
                return best_config

    return best_config


def get_aligned_boost(input_dim: int, output_dim: int, target_params: int,
                      num_stages: int = None, tolerance: float = 0.20) -> Dict[str, Any]:
    """
    搜索与目标参数量最接近的Boost配置

    策略: 动态选择num_stages，调整shallow_depth和poly_degree
    搜索空间:
    - num_stages: 根据target_params动态选择 [3, 5, 10, 20]
    - shallow_depth [2-8]
    - poly_degree [2-6]

    注意：Boost总参数量 = num_stages * per_stage_params
    """
    best_config = None
    best_diff = float('inf')

    # 根据目标参数量动态选择可能的num_stages
    if num_stages is None:
        if target_params <= 100:
            stages_options = [3, 5]
        elif target_params <= 400:
            stages_options = [5, 10]
        else:
            stages_options = [10, 20]
    else:
        stages_options = [num_stages]

    for stages in stages_options:
        target_per_stage = target_params / stages

        for shallow_depth in range(2, 9):
            for poly_degree in range(2, 7):
                per_stage = count_params_boost_per_stage(input_dim, output_dim, shallow_depth, poly_degree)
                total_params = per_stage * stages
                diff = abs(total_params - target_params) / target_params

                if diff < best_diff:
                    best_diff = diff
                    best_config = {
                        'num_stages': stages,
                        'shallow_depth': shallow_depth,
                        'poly_degree': poly_degree,
                        'actual_params': total_params,
                        'per_stage_params': per_stage
                    }

                if diff < tolerance:
                    return best_config

    return best_config


def get_aligned_moe(input_dim: int, output_dim: int, target_params: int,
                    num_experts: int = None, tolerance: float = 0.20) -> Dict[str, Any]:
    """
    搜索与目标参数量最接近的MoE配置

    策略: 动态选择num_experts，调整shallow_depth和poly_degree
    搜索空间:
    - num_experts: [3, 5, 8] (根据target_params动态选择)
    - shallow_depth [2-8]
    - poly_degree [2-6]

    注意：MoE总参数量 = num_experts * per_expert_params + gating_params
    """
    best_config = None
    best_diff = float('inf')

    # 根据目标参数量动态选择可能的num_experts
    if num_experts is None:
        if target_params <= 100:
            experts_options = [3]
        elif target_params <= 400:
            experts_options = [3, 5]
        else:
            experts_options = [5, 8]
    else:
        experts_options = [num_experts]

    for experts in experts_options:
        # 门控网络参数量
        gating_params = count_params_moe_gating(input_dim, experts)

        # 目标所有experts的参数量
        target_experts_params = target_params - gating_params

        # 如果门控网络已经占用了太多参数，跳过这个配置
        if target_experts_params <= 0:
            continue

        target_per_expert = target_experts_params / experts

        for shallow_depth in range(2, 9):
            for poly_degree in range(2, 7):
                per_expert = count_params_moe_per_expert(input_dim, output_dim, shallow_depth, poly_degree)
                total_experts_params = per_expert * experts
                total_params = total_experts_params + gating_params
                diff = abs(total_params - target_params) / target_params

                if diff < best_diff:
                    best_diff = diff
                    best_config = {
                        'num_experts': experts,
                        'shallow_depth': shallow_depth,
                        'poly_degree': poly_degree,
                        'actual_params': total_params,
                        'per_expert_params': per_expert,
                        'gating_params': gating_params
                    }

                if diff < tolerance:
                    return best_config

    # 如果没有找到有效配置，使用最小配置
    if best_config is None:
        # 使用最小可能的配置
        experts = 2
        gating_params = count_params_moe_gating(input_dim, experts)
        best_config = {
            'num_experts': experts,
            'shallow_depth': 2,
            'poly_degree': 2,
            'actual_params': gating_params + 2 * count_params_moe_per_expert(input_dim, output_dim, 2, 2),
            'per_expert_params': count_params_moe_per_expert(input_dim, output_dim, 2, 2),
            'gating_params': gating_params
        }

    return best_config


def get_aligned_kan(input_dim: int, output_dim: int, target_params: int,
                    tolerance: float = 0.20) -> Dict[str, Any]:
    """
    搜索与目标参数量最接近的KAN配置

    策略: 调整num_layers, grid_size和hidden_dim
    搜索空间:
    - num_layers: 1-3 (1层适用于小参数量)
    - grid_size [1, 2, 3, 5, 10, 15] (1为最小值)
    - hidden_dim [1-100]
    """
    best_config = None
    best_diff = float('inf')

    # 对于小参数量使用更小的grid_size和更少的层数
    if target_params < 100:
        grid_sizes = [1, 2, 3]
        num_layers_options = [1, 2]
    elif target_params < 400:
        grid_sizes = [2, 3, 5, 10]
        num_layers_options = [2]
    else:
        grid_sizes = [3, 5, 10, 15]
        num_layers_options = [2, 3]

    spline_order = 3

    for num_layers in num_layers_options:
        for grid_size in grid_sizes:
            # 对于单层网络，不需要hidden_dim变化
            if num_layers == 1:
                hidden_dims = range(1, min(20, target_params // 10) + 1)
            else:
                hidden_dims = range(2, 101)

            for hidden_dim in hidden_dims:
                params = count_params_kan(input_dim, output_dim, hidden_dim, num_layers, grid_size, spline_order)
                diff = abs(params - target_params) / target_params

                if diff < best_diff:
                    best_diff = diff
                    best_config = {
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'grid_size': grid_size,
                        'spline_order': spline_order,
                        'actual_params': params
                    }

                if diff < tolerance:
                    return best_config

                # 提前终止条件
                if params > target_params * 1.5 and best_config is not None:
                    break

    return best_config


def get_aligned_mlp(input_dim: int, output_dim: int, target_params: int,
                    num_layers: int = 2, tolerance: float = 0.15) -> Dict[str, Any]:
    """
    搜索与目标参数量最接近的MLP配置

    策略: 调整hidden_dim
    搜索空间: hidden_dim [4-200]
    """
    best_config = None
    best_diff = float('inf')

    for hidden_dim in range(4, 201):
        params = count_params_mlp(input_dim, output_dim, hidden_dim, num_layers)
        diff = abs(params - target_params) / target_params

        if diff < best_diff:
            best_diff = diff
            best_config = {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'actual_params': params
            }

        if diff < tolerance:
            return best_config

        # 提前终止
        if params > target_params * 1.5 and best_config is not None:
            break

    return best_config


# ==========================================
# 4. 配置生成主函数
# ==========================================

def get_pareto_model_config(model_name: str, target_params: int,
                            input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM,
                            **kwargs) -> Dict[str, Any]:
    """
    获取指定模型在目标参数量下的配置

    Args:
        model_name: 模型名称 ('MLP', 'KAN', 'Standard', 'Hybrid', 'Boost', 'MoE')
        target_params: 目标参数量
        input_dim: 输入维度
        output_dim: 输出维度
        **kwargs: 额外参数（如num_stages, num_experts等）

    Returns:
        config: 包含模型配置和实际参数量的字典
    """
    model_name = model_name.lower()

    if model_name == 'mlp':
        config = get_aligned_mlp(input_dim, output_dim, target_params)
        return {
            'name': 'MLP',
            'type': 'mlp',
            'target_params': target_params,
            **config
        }

    elif model_name == 'kan':
        config = get_aligned_kan(input_dim, output_dim, target_params)
        return {
            'name': 'KAN',
            'type': 'kan',
            'target_params': target_params,
            **config
        }

    elif model_name == 'standard':
        config = get_aligned_standard(input_dim, output_dim, target_params)
        return {
            'name': 'Standard',
            'type': 'standard',
            'target_params': target_params,
            **config
        }

    elif model_name == 'hybrid':
        config = get_aligned_hybrid(input_dim, output_dim, target_params)
        return {
            'name': 'Hybrid',
            'type': 'hybrid',
            'target_params': target_params,
            **config
        }

    elif model_name == 'boost':
        # num_stages现在由get_aligned_boost根据target_params动态选择
        config = get_aligned_boost(input_dim, output_dim, target_params)
        return {
            'name': 'Boost',
            'type': 'boost',
            'target_params': target_params,
            **config
        }

    elif model_name == 'moe':
        # num_experts现在由get_aligned_moe根据target_params动态选择
        config = get_aligned_moe(input_dim, output_dim, target_params)
        return {
            'name': 'MoE',
            'type': 'moe',
            'target_params': target_params,
            **config
        }

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_pareto_experiment_config(noise_ratio: float, target_params: int,
                                  n_total_features: int = 20) -> Dict[str, Any]:
    """
    获取帕累托实验的完整配置（包含数据和模型配置）

    Args:
        noise_ratio: 噪声比例 (0.0-0.9)
        target_params: 目标参数量
        n_total_features: 总特征数

    Returns:
        config: 完整实验配置
    """
    # 基础配置
    config = BASE_CONFIG.copy()
    config['models'] = BASE_CONFIG['models'].copy()

    # 数据配置（基于噪声比例）
    n_true = 4
    n_noise = int(n_total_features * noise_ratio) - 3
    n_noise = max(0, n_noise)
    n_redundant = 2
    n_deceptive = 1
    actual_total = n_true + n_noise + n_redundant + n_deceptive

    config['data'] = {
        'n_samples': config['n_samples'],
        'noise_std': config['noise_std'],
        'test_split': config['test_split'],
        'val_split': config['val_split'],
        'random_seed': config['random_seed'],
        'batch_size': config['batch_size'],
        'n_true_features': n_true,
        'n_noise_features': n_noise,
        'n_redundant_features': n_redundant,
        'n_deceptive_features': n_deceptive,
        'redundant_noise_std': 0.05,
        'deceptive_corr': 0.8,
        'noise_ratio': noise_ratio,
        'n_total_features': actual_total
    }

    # 训练配置
    config['epochs'] = 50
    config['models']['boost']['epochs_per_stage'] = 20

    # 帕累托实验特定配置
    config['pareto'] = {
        'target_params': target_params,
        'noise_ratio': noise_ratio
    }

    return config


def get_all_pareto_configs(noise_ratios: List[float] = None,
                           pareto_levels: List[int] = None,
                           models: List[str] = None) -> List[Dict[str, Any]]:
    """
    生成所有帕累托实验配置组合

    Args:
        noise_ratios: 噪声比例列表，默认使用NOISE_RATIOS
        pareto_levels: 参数量级别列表，默认使用PARETO_LEVELS
        models: 模型列表，默认所有6个模型

    Returns:
        configs: 所有配置组合的列表，每个元素包含:
            - noise_ratio: 噪声比例
            - target_params: 目标参数量
            - model_name: 模型名称
            - experiment_config: 完整实验配置
            - model_config: 模型特定配置
    """
    if noise_ratios is None:
        noise_ratios = NOISE_RATIOS
    if pareto_levels is None:
        pareto_levels = PARETO_LEVELS
    if models is None:
        models = ['MLP', 'KAN', 'Standard', 'Hybrid', 'Boost', 'MoE']

    configs = []

    for noise_ratio in noise_ratios:
        for target_params in pareto_levels:
            for model_name in models:
                # 获取实验配置
                experiment_config = get_pareto_experiment_config(noise_ratio, target_params)

                # 获取模型配置
                n_features = experiment_config['data']['n_total_features']

                # Boost和MoE的num_stages/num_experts现在由get_aligned_*函数动态选择
                model_config = get_pareto_model_config(
                    model_name, target_params, n_features, OUTPUT_DIM
                )

                configs.append({
                    'noise_ratio': noise_ratio,
                    'target_params': target_params,
                    'model_name': model_name,
                    'experiment_config': experiment_config,
                    'model_config': model_config
                })

    return configs


def get_quick_test_configs() -> List[Dict[str, Any]]:
    """
    获取快速测试配置（用于验证代码正确性）

    使用更小的参数集快速测试:
    - 噪声比例: 0.4, 0.8 (2个)
    - 参数量级别: 100, 400 (2个)
    - 模型: MLP, Standard, Boost (3个)
    """
    return get_all_pareto_configs(
        noise_ratios=[0.4, 0.8],
        pareto_levels=[100, 400],
        models=['MLP', 'Standard', 'Boost']
    )


# ==========================================
# 5. 预计算配置表（可选优化）
# ==========================================

def precompute_pareto_table(input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM,
                           save_path: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    预计算所有帕累托级别的配置表

    可用于验证配置正确性或加速实验启动
    """
    table = {}

    for model_name in ['MLP', 'KAN', 'Standard', 'Hybrid', 'Boost', 'MoE']:
        table[model_name] = []
        for target in PARETO_LEVELS:
            try:
                config = get_pareto_model_config(model_name, target, input_dim, output_dim)
                table[model_name].append({
                    'target': target,
                    'actual': config['actual_params'],
                    'diff_pct': abs(config['actual_params'] - target) / target * 100,
                    'config': config
                })
            except Exception as e:
                table[model_name].append({
                    'target': target,
                    'error': str(e)
                })

    return table


# ==========================================
# 6. 配置验证和打印
# ==========================================

def print_pareto_config_table(input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM):
    """
    打印帕累托配置表（用于验证）
    """
    print("="*80)
    print("帕累托前沿实验配置表")
    print("="*80)
    print(f"输入维度: {input_dim}, 输出维度: {output_dim}")
    print(f"目标参数量级别: {PARETO_LEVELS}")
    print(f"噪声比例: {NOISE_RATIOS}")
    print()

    table = precompute_pareto_table(input_dim, output_dim)

    for model_name, configs in table.items():
        print(f"\n【{model_name}】")
        print("-"*60)
        for cfg in configs:
            if 'error' in cfg:
                print(f"  Target {cfg['target']:4d}: ERROR - {cfg['error']}")
            else:
                diff_str = f"{cfg['diff_pct']:5.1f}%"
                config_details = str(cfg['config'])
                # 简化输出
                if model_name == 'MLP':
                    detail = f"hidden_dim={cfg['config'].get('hidden_dim', 'N/A')}"
                elif model_name == 'KAN':
                    detail = f"hidden_dim={cfg['config'].get('hidden_dim', 'N/A')}, grid={cfg['config'].get('grid_size', 'N/A')}"
                elif model_name == 'Standard':
                    detail = f"depth={cfg['config'].get('depth', 'N/A')}, degree={cfg['config'].get('poly_degree', 'N/A')}"
                elif model_name == 'Hybrid':
                    detail = f"num_units={cfg['config'].get('num_units', 'N/A')}, degree={cfg['config'].get('unit_degree', 'N/A')}"
                elif model_name == 'Boost':
                    detail = f"stages={cfg['config'].get('num_stages', 'N/A')}, depth={cfg['config'].get('shallow_depth', 'N/A')}"
                elif model_name == 'MoE':
                    detail = f"experts={cfg['config'].get('num_experts', 'N/A')}, depth={cfg['config'].get('shallow_depth', 'N/A')}"
                else:
                    detail = ""

                print(f"  Target {cfg['target']:4d} -> Actual {cfg['actual']:5d} ({diff_str}) [{detail}]")

    print("\n" + "="*80)


if __name__ == '__main__':
    # 验证配置生成
    print_pareto_config_table()

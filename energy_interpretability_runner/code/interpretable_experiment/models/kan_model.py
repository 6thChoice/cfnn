"""
KAN (Kolmogorov-Arnold Network) 模型实现
用于与 CFNet 系列模型的公平对比
"""

import torch
import torch.nn as nn
import numpy as np


class SplineBasis(nn.Module):
    """
    B-样条基函数实现
    使用简单的网格插值实现
    """
    def __init__(self, grid_size=10, spline_order=3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

    def forward(self, x):
        """
        计算B-样条基函数值
        x: [batch_size, features]
        返回: [batch_size, features, grid_size + spline_order]
        """
        batch_size, n_features = x.shape

        # 创建网格点 (在[-1, 1]范围内)
        # n_basis = grid_size + spline_order
        n_basis = self.grid_size + self.spline_order
        grid = torch.linspace(-1, 1, n_basis, device=x.device)

        # 扩展维度用于广播
        x_expanded = x.unsqueeze(-1)  # [batch, features, 1]
        grid_expanded = grid.view(1, 1, -1)  # [1, 1, n_basis]

        # 计算基函数值 (使用简单的RBF近似)
        distances = torch.abs(x_expanded - grid_expanded)  # [batch, features, n_basis]

        # 使用高斯基函数
        width = (grid[1] - grid[0]) * (self.spline_order + 1)
        basis = torch.exp(-0.5 * (distances / width) ** 2)

        return basis


class KANLayer(nn.Module):
    """
    KAN层: 每个输入-输出对有一个样条函数
    简化为: y = sum_j(phi_ij(x_j)) 对每个输出i
    """
    def __init__(self, in_features, out_features, grid_size=10, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order

        # 样条基函数
        self.spline_basis = SplineBasis(grid_size, spline_order)

        # 可学习系数: [in_features, out_features, n_basis]
        # 表示每个输入-输出对的样条系数
        self.coeffs = nn.Parameter(
            torch.randn(in_features, out_features, self.n_basis) * 0.1
        )

        # 残差连接权重 (类似原始KAN的残差)
        self.residual_weight = nn.Parameter(torch.ones(out_features) * 0.1)

    def forward(self, x):
        """
        x: [batch_size, in_features]
        返回: [batch_size, out_features]
        """
        batch_size = x.shape[0]

        # 计算样条基函数 [batch, in_features, n_basis]
        basis = self.spline_basis(x)

        # 应用可学习系数
        # basis: [batch, in_features, n_basis]
        # coeffs: [in_features, out_features, n_basis]
        # 结果: [batch, out_features]

        # 对每个输出特征，计算所有输入的贡献
        output = torch.zeros(batch_size, self.out_features, device=x.device)

        for i in range(self.out_features):
            # 第i个输出的贡献
            # basis: [batch, in_features, n_basis]
            # coeffs[:, i, :]: [in_features, n_basis]
            contribution = (basis * self.coeffs[:, i, :].unsqueeze(0)).sum(dim=[1, 2])
            output[:, i] = contribution

        # 添加偏置项（替代有问题的残差连接）
        output = output + self.residual_weight.unsqueeze(0)

        return output


class KAN(nn.Module):
    """
    完整KAN网络
    多层KANLayer堆叠
    """
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=2,
                 grid_size=10, spline_order=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i+1], grid_size, spline_order))
            if i < len(dims) - 2:  # 非最后一层添加归一化和激活
                layers.append(nn.LayerNorm(dims[i+1]))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [batch_size, input_dim] 或 [batch_size]
        返回: [batch_size, output_dim] 或 [batch_size]
        """
        # 处理输入维度
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        output = self.network(x)

        # 如果输出维度是1，压缩最后一维
        if self.output_dim == 1 and output.shape[-1] == 1:
            output = output.squeeze(-1)

        return output


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_aligned_kan(input_dim, output_dim, target_params, num_layers=2):
    """
    获取与目标参数量对齐的KAN模型
    通过调整 hidden_dim 来匹配参数量

    KAN参数量估算:
    - 每层: in_features * out_features * (grid_size + spline_order)
    - 加上残差权重: out_features
    """
    grid_size = 10
    spline_order = 3
    n_basis = grid_size + spline_order

    def calc_params(hidden_dim):
        total = 0
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            # coeffs + residual_weight
            layer_params = dims[i] * dims[i+1] * n_basis + dims[i+1]
            # LayerNorm参数 (除了最后一层)
            if i < len(dims) - 2:
                layer_params += 2 * dims[i+1]  # LayerNorm有weight和bias
            total += layer_params
        return total

    # 搜索最优hidden_dim
    best_h = 8
    min_diff = float('inf')

    for h in range(4, 200):
        p = calc_params(h)
        diff = abs(p - target_params)
        if diff < min_diff:
            min_diff = diff
            best_h = h
        if p > target_params * 1.5:
            break

    return KAN(input_dim, output_dim, best_h, num_layers, grid_size, spline_order), best_h


if __name__ == "__main__":
    # 测试KAN模型
    print("Testing KAN Model...")

    # 测试基本功能
    model = KAN(input_dim=4, output_dim=1, hidden_dim=16, num_layers=2)
    x = torch.randn(32, 4)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {count_parameters(model)}")

    # 测试参数对齐
    print("\nTesting parameter alignment...")
    target = 1000
    kan_model, h = get_aligned_kan(4, 1, target)
    actual = count_parameters(kan_model)
    print(f"Target params: {target}")
    print(f"Aligned KAN (h={h}) params: {actual}")
    print(f"Difference: {abs(actual - target)} ({abs(actual - target)/target*100:.1f}%)")

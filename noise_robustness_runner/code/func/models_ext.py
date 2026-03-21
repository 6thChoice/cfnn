import torch
import torch.nn as nn
import sys
import os

# 将父目录添加到路径以导入 cfnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cfnet import CFNet_Standard, HybridRationalNet, EnsembleResCoFrNet, MoE_Ensemble

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.Tanh()) # 使用 Tanh 与 CFNet 保持一致
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    """计算模型可学习参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_aligned_mlp(input_dim, output_dim, target_params, num_layers=2):
    """
    通过调整 hidden_dim 寻找与目标参数量最接近的 MLP
    """
    def calc_params(h):
        # 参数量估计: (in*h + h) + (layers-1)*(h*h + h) + (h*out + out)
        # 简化: P = in*h + h + (n-1)*h^2 + (n-1)*h + h*out + out
        p = (input_dim * h + h) + (num_layers - 1) * (h * h + h) + (h * output_dim + output_dim)
        return p

    # 简单的二分查找或线性搜索
    best_h = 1
    min_diff = float('inf')

    for h in range(1, 1000):
        p = calc_params(h)
        diff = abs(p - target_params)
        if diff < min_diff:
            min_diff = diff
            best_h = h
        if p > target_params * 1.5: # 提前终止
            break

    return BaselineMLP(input_dim, output_dim, best_h, num_layers), best_h

if __name__ == "__main__":
    # 测试参数对齐
    input_dim = 4
    output_dim = 1

    # 以一个标准的 CFNet 为基准
    base_model = CFNet_Standard(input_dim, output_dim, depth=4, poly_degree=3)
    target = count_parameters(base_model)
    print(f"Target Parameters (CFNet): {target}")

    mlp, h = get_aligned_mlp(input_dim, output_dim, target)
    print(f"Aligned MLP (h={h}) Parameters: {count_parameters(mlp)}")

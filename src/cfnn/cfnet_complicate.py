import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ==========================================
# 1. 模型定义 (保持不变)
# ==========================================
class PolynomialTerm(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.degree = degree
        self.projection = nn.Linear(input_dim, output_dim)
        self.coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.05) 

    def forward(self, x):
        z = torch.tanh(self.projection(x))
        powers = [torch.ones_like(z)]
        for d in range(1, self.degree + 1):
            powers.append(powers[-1] * z)
        z_powered = torch.stack(powers, dim=-1)
        return torch.sum(z_powered * self.coeffs, dim=2)

class CFNet(nn.Module):
    def __init__(self, input_dim, output_dim, depth=4, poly_degree=3):
        super().__init__()
        self.terms = nn.ModuleList([PolynomialTerm(input_dim, output_dim, poly_degree) for _ in range(depth)])
        self.raw_betas = nn.ParameterList([nn.Parameter(torch.tensor([1.0])) for _ in range(max(0, depth - 1))])

    def forward(self, x):
        if not self.terms: return torch.zeros(x.shape[0], self.terms[0].projection.out_features).to(x.device)
        output = self.terms[-1](x)
        output = torch.abs(output) + 1.0 
        for i in range(len(self.terms) - 2, -1, -1):
            beta = F.softplus(self.raw_betas[i])
            output = self.terms[i](x) + beta / (output + 1e-8)
        return output
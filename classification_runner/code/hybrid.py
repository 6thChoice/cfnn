import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.special as sp
import logging
import json
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 模型定义 (保持一致)
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

class RationalUnit(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.P = PolynomialTerm(input_dim, output_dim, degree)
        self.Q = PolynomialTerm(input_dim, output_dim, degree)
    def forward(self, x):
        denom = self.Q(x) ** 2 + 1.0
        return self.P(x) / denom

class HybridRationalNet(nn.Module):
    def __init__(self, input_dim, output_dim, unit_degree=3, num_units=4):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim)
        self.units = nn.ModuleList([
            RationalUnit(input_dim, output_dim, unit_degree)
            for _ in range(num_units)
        ])
    def forward(self, x):
        # total_output = 0
        total_output = self.linear_skip(x)
        for unit in self.units:
            total_output = total_output + unit(x)
        return total_output
    
# ==========================================
# 2. 辅助工具
# ==========================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_data(func, domain, n_points, is_complex):
    x1_range = np.linspace(domain['x'][0], domain['x'][1], n_points)
    x2_range = np.linspace(domain['y'][0], domain['y'][1], n_points)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    with np.errstate(all='ignore'):
        Z_full = func(X1, X2)
    
    X_input_full = np.vstack([X1.ravel(), X2.ravel()]).T
    
    if is_complex:
        Z_flat = Z_full.ravel()
        Y_output_full = np.vstack([np.real(Z_flat), np.imag(Z_flat)]).T
        valid_indices = np.isfinite(Y_output_full).all(axis=1)
        X_train = X_input_full[valid_indices]
        Y_train = Y_output_full[valid_indices]
    else:
        Z_flat = Z_full.ravel()
        valid_indices = np.isfinite(Z_flat)
        X_train = X_input_full[valid_indices]
        Y_train = Z_flat[valid_indices].reshape(-1, 1)
        
    return X_train, Y_train

def train_model(model, x, y, epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*0.7), gamma=0.5) 
    
    model.train()
    final_loss = 0.0
    
    for i in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        scheduler.step()
        final_loss = loss.item()
        
    return final_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
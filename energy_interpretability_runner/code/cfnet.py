import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# ==========================================
# 1. 基础组件 (Shared Components)
# ==========================================

class PolynomialTerm(nn.Module):
    """
    通用多项式项：输入 -> 线性投影 -> Tanh -> 多项式展开 -> 加权求和
    被所有模型变体共用。
    """
    def __init__(self, input_dim, output_dim, degree, init_std=0.1):
        super().__init__()
        self.degree = degree
        self.projection = nn.Linear(input_dim, output_dim)
        # 统一初始化标准差，默认为 0.1 (Boost/MoE/Standard)，Hybrid 原版为 0.05
        self.coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * init_std)

    def forward(self, x):
        z = torch.tanh(self.projection(x))
        powers = [torch.ones_like(z)]
        for d in range(1, self.degree + 1):
            powers.append(powers[-1] * z)
        z_powered = torch.stack(powers, dim=-1)
        # Sum over the degree dimension
        return torch.sum(z_powered * self.coeffs, dim=2)


# ==========================================
# 2. Hybrid Rational Net (from hybrid.py)
# ==========================================

class RationalUnit(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.P = PolynomialTerm(input_dim, output_dim, degree, init_std=0.05)
        self.Q = PolynomialTerm(input_dim, output_dim, degree, init_std=0.05)

    def forward(self, x):
        # Q(x)^2 + 1 保证分母非零且正
        denom = self.Q(x) ** 2 + 1.0
        return self.P(x) / denom

class HybridRationalNet(nn.Module):
    """
    混合有理网络：线性跳跃连接 + 多个并行的 RationalUnit
    """
    def __init__(self, input_dim, output_dim, unit_degree=3, num_units=4):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim)
        self.units = nn.ModuleList([
            RationalUnit(input_dim, output_dim, unit_degree)
            for _ in range(num_units)
        ])

    def forward(self, x):
        total_output = self.linear_skip(x)
        for unit in self.units:
            total_output = total_output + unit(x)
        return total_output


# ==========================================
# 3. Standard CFNet (from cfnet_complicate.py)
# ==========================================

class CFNet_Standard(nn.Module):
    """
    标准版连分式网络。
    特点：使用 torch.abs() + 1.0 处理分母，逻辑源自 cfnet_complicate.py
    """
    def __init__(self, input_dim, output_dim, depth=4, poly_degree=3):
        super().__init__()
        self.terms = nn.ModuleList([
            PolynomialTerm(input_dim, output_dim, poly_degree, init_std=0.05) 
            for _ in range(depth)
        ])
        self.raw_betas = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0])) for _ in range(max(0, depth - 1))
        ])

    def forward(self, x):
        if not self.terms: 
            return torch.zeros(x.shape[0], self.terms[0].projection.out_features).to(x.device)
        
        output = self.terms[-1](x)
        # 这里的处理方式是 abs + 1.0
        output = torch.abs(output) + 1.0 
        
        for i in range(len(self.terms) - 2, -1, -1):
            beta = F.softplus(self.raw_betas[i])
            output = self.terms[i](x) + beta / (output + 1e-8)
        return output


# ==========================================
# 4. Core Components for Boost & MoE
# ==========================================

class CFNet_Core(nn.Module):
    """
    核心连分式单元。
    特点：使用 F.softplus() + 1.0 处理分母，数值更稳定，用于集成模型。
    源自 cfnn_boost.py 和 cfnet_regressor_moe_iterative.py
    """
    def __init__(self, input_dim, output_dim, depth, poly_degree):
        super().__init__()
        self.output_dim = output_dim
        self.terms = nn.ModuleList([
            PolynomialTerm(input_dim, output_dim, poly_degree, init_std=0.1) 
            for _ in range(depth)
        ])
        self.raw_betas = nn.ParameterList([
            nn.Parameter(torch.full((output_dim,), 0.1)) for _ in range(depth - 1)
        ])

    def forward(self, x):
        if not self.terms:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)

        # 最后一项：softplus + 1.0 保证正定
        output = F.softplus(self.terms[-1](x)) + 1.0

        # 反向递归
        for i in range(len(self.terms) - 2, -1, -1):
            beta = F.softplus(self.raw_betas[i])
            output = self.terms[i](x) + beta / (output + 1e-8)

        return output


# ==========================================
# 5. Boosting Model (from cfnn_boost.py)
# ==========================================

class EnsembleResCoFrNet(nn.Module):
    """
    集成模型：Boosting 风格
    """
    def __init__(self, input_dim, output_dim, shallow_depth, poly_degree, learning_rate):
        super().__init__()
        self.models = nn.ModuleList()
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shallow_depth = shallow_depth
        self.poly_degree = poly_degree
        # 注册一个伪参数来跟踪设备
        self._device_tracker = nn.Parameter(torch.zeros(0))

    def forward(self, x):
        if not self.models:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        total_output = torch.zeros(x.shape[0], self.output_dim, device=x.device)
        for model in self.models:
            total_output += self.learning_rate * model(x)
        return total_output

    def add_model(self):
        """添加一个新子模型 (CFNet_Core)"""
        # 获取当前设备
        device = self._device_tracker.device

        new_model = CFNet_Core(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            depth=self.shallow_depth,
            poly_degree=self.poly_degree
        ).to(device)

        self.models.append(new_model)
        return new_model

    def get_current_prediction(self, x):
        """获取当前集成模型（不含最新模型）的预测"""
        if len(self.models) <= 1:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        total_output = torch.zeros(x.shape[0], self.output_dim, device=x.device)
        for model in self.models[:-1]:
            total_output += self.learning_rate * model(x)
        return total_output

    def freeze_all_but_latest(self):
        """冻结旧模型，只训练最新模型"""
        for model in self.models[:-1]:
            for param in model.parameters():
                param.requires_grad = False
        for param in self.models[-1].parameters():
            param.requires_grad = True


# ==========================================
# 6. MoE Model (from cfnet_regressor_moe_iterative.py)
# ==========================================

class RBFGatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.empty(0, input_dim))
        self.widths = nn.Parameter(torch.empty(0, 1))

    def forward(self, x):
        if self.centers.shape[0] == 0:
            return torch.ones(x.shape[0], 1, device=x.device)
        
        x_reshaped = x.unsqueeze(1)
        centers_reshaped = self.centers.unsqueeze(0)
        positive_widths = F.softplus(self.widths.squeeze()) + 1e-8
        
        dist_sq = torch.sum((x_reshaped - centers_reshaped) ** 2, dim=-1)
        logits = -dist_sq / (2 * (positive_widths ** 2))
        return logits

    def add_expert_gate(self, initial_center, initial_width_param=1.0, device=None):
        # 确定目标设备
        if device is None:
            if self.centers.shape[0] > 0:
                target_device = self.centers.device
            else:
                # 检查父模型的设备
                target_device = torch.device('cpu')
        else:
            target_device = device

        center_tensor = torch.tensor(initial_center, dtype=torch.float32).unsqueeze(0).to(target_device)
        width_tensor = torch.tensor([[initial_width_param]], dtype=torch.float32).to(target_device)

        if self.centers.shape[0] > 0:
            self.centers = nn.Parameter(torch.cat([self.centers.data, center_tensor], dim=0))
            self.widths = nn.Parameter(torch.cat([self.widths.data, width_tensor], dim=0))
        else:
            self.centers = nn.Parameter(center_tensor)
            self.widths = nn.Parameter(width_tensor)


class MoE_Ensemble(nn.Module):
    """
    混合专家模型 (Mixture of Experts)
    依赖 hparams 字典进行初始化，以兼容原有训练逻辑。
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.experts = nn.ModuleList()
        self.gating = RBFGatingNetwork(self.hparams['input_dim'])
        # 注册一个伪参数来跟踪设备
        self._device_tracker = nn.Parameter(torch.zeros(0))

    def forward(self, x):
        if not self.experts:
            return torch.zeros(x.shape[0], self.hparams['output_dim'], device=x.device)
        
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1) # (B, Num_Experts, Out_Dim)
        
        gate_logits = self.gating(x) # (B, Num_Experts)
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1) # (B, Num_Experts, 1)
        
        return torch.sum(gate_weights * expert_outputs_stacked, dim=1)

    def add_expert(self):
        """添加一个新的 CFNet_Core 专家"""
        # 获取当前设备
        device = self._device_tracker.device

        new_expert = CFNet_Core(
            input_dim=self.hparams['input_dim'],
            output_dim=self.hparams['output_dim'],
            depth=self.hparams['shallow_depth_per_cofrnet'],
            poly_degree=self.hparams['polynomial_degree']
        ).to(device)

        self.experts.append(new_expert)
        logging.info(f"Added new expert. Total experts: {len(self.experts)}")
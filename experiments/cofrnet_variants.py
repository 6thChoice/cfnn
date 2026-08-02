"""
cofrnet_variants.py

Self-contained module aggregating verbatim class copies from:

1. `PolynomialTerm`, `CFNet_Standard`
   Source: /home/zxc/CodeBase/cofrnet/experiment_refine/CFNN-Experiments/Energy Efficiency/cfnet.py

2. `SingleLadder`, `FullLadder`, `CoFrNetDL`
   Source: /home/zxc/CodeBase/cofrnet/CoFrNet_DL.py

3. `CFNet_Core`, `EnsembleResCoFrNet`, `RBFGatingNetwork`, `MoE_Ensemble`
   Source: /home/zxc/CodeBase/cofrnet/experiment_refine/CFNN-Experiments/Energy Efficiency/cfnet.py

4. `get_polynomial_basis`, `DifferentiableLearnableFunction`,
   `ElementwiseCFN_Activation`, `MultiLayerCFN`
   Source: /home/zxc/CodeBase/cofrnet/cfnn/cfnn_poly.py

Class bodies (constructors, forward logic, helper methods such as
`safe_reciprocal`) are copied EXACTLY as they appear in the original files,
with no logic changes, so these classes can be imported and used without
depending on those files' paths. Only import statements and this module
docstring were added; no class/method signatures were renamed or altered.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


# ==========================================================================
# From: CFNN-Experiments/Energy Efficiency/cfnet.py
# ==========================================================================

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


# ==========================================================================
# From: /home/zxc/CodeBase/cofrnet/CoFrNet_DL.py
# ==========================================================================

class SingleLadder(nn.Module):

    def __init__(self, depth, epsilon=0.1):
        super(SingleLadder, self).__init__()
        self.depth = depth
        self.epsilon = epsilon
        # 使用单一参数张量
        self.weights = nn.Parameter(torch.randn(depth + 1) * 0.1)

    def safe_reciprocal(self, x):

        sign = torch.sign(x)
        abs_x = torch.abs(x)
        safe_abs = torch.clamp(abs_x, min=self.epsilon)
        return sign / safe_abs

    def forward(self, x_j):

        if self.depth == 0:
            return self.weights[0] * x_j

        # 从最深层开始
        result = self.weights[self.depth] * x_j

        for k in range(self.depth - 1, 0, -1):
            denominator = self.weights[k] * x_j + self.safe_reciprocal(result)
            result = denominator

        # 最终层: w0 * x_j + 1/result
        final_result = self.weights[0] * x_j + self.safe_reciprocal(result)
        return final_result

class FullLadder(nn.Module):

    def __init__(self, input_dim, depth, epsilon=0.1):
        super(FullLadder, self).__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.epsilon = epsilon
        # 权重矩阵: (depth+1, input_dim)
        self.weights = nn.Parameter(torch.randn(depth + 1, input_dim) * 0.1)

    def safe_reciprocal(self, x):

        sign = torch.sign(x)
        abs_x = torch.abs(x)
        safe_abs = torch.clamp(abs_x, min=self.epsilon)
        return sign / safe_abs

    def forward(self, x):

        if self.depth == 0:
            return F.linear(x, self.weights[0])

        # 从最深层开始
        result = F.linear(x, self.weights[self.depth])

        for k in range(self.depth - 1, 0, -1):
            a_k = F.linear(x, self.weights[k])
            denominator = a_k + self.safe_reciprocal(result)
            result = denominator

        # 最终层: w0^T * x + 1/result
        a_0 = F.linear(x, self.weights[0])
        final_result = a_0 + self.safe_reciprocal(result)
        return final_result

class CoFrNetDL(nn.Module):

    def __init__(self, input_dim, output_dim, num_full_ladders=25,
                 max_diag_ladder_depth=12, max_full_ladder_depth=12, epsilon=0.1):
        super(CoFrNetDL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_full_ladders = num_full_ladders
        self.max_diag_ladder_depth = max_diag_ladder_depth
        self.max_full_ladder_depth = max_full_ladder_depth

        # 1. 对角梯子部分
        # 每个输入维度对应一个单特征梯子
        self.diag_ladders = nn.ModuleList()
        self.diag_ladder_depths = []

        for i in range(input_dim):

            depth = min(max_diag_ladder_depth, max(2, 5 + (i * 2) % 10))
            self.diag_ladder_depths.append(depth)
            self.diag_ladders.append(SingleLadder(depth, epsilon))

        # 2. 全连接梯子部分
        # 深度递增
        self.full_ladders = nn.ModuleList()
        self.full_ladder_depths = []

        for i in range(num_full_ladders):
            # 使用线性插值分配深度
            if num_full_ladders > 1:
                depth = 2 + int((i / (num_full_ladders - 1)) * (max_full_ladder_depth - 2))
            else:
                depth = 2
            depth = min(max_full_ladder_depth, max(2, depth))
            self.full_ladder_depths.append(depth)
            self.full_ladders.append(FullLadder(input_dim, depth, epsilon))

        # 3. 最终线性组合层
        combined_input_dim = input_dim + num_full_ladders
        self.output_layer = nn.Linear(combined_input_dim, output_dim)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):

        batch_size = x.shape[0]

        # 1. 计算对角梯子输出
        diag_outputs = []
        for j in range(self.input_dim):
            x_j = x[:, j]
            ladder_output = self.diag_ladders[j](x_j)
            diag_outputs.append(ladder_output.unsqueeze(1))

        diag_features = torch.cat(diag_outputs, dim=1)

        # 2. 计算全连接梯子输出
        full_outputs = []
        for ladder in self.full_ladders:
            ladder_output = ladder(x)
            full_outputs.append(ladder_output.unsqueeze(1))

        full_features = torch.cat(full_outputs, dim=1)

        # 3. 拼接所有特征
        combined_features = torch.cat([diag_features, full_features], dim=1)

        # 4. 最终线性组合
        output = self.output_layer(combined_features)
        return output

    def get_feature_importance(self, x):

        device = next(self.parameters()).device

        with torch.no_grad():

            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            x = x.to(device)

            importance_dict = {}

            # 1. 对角梯子的特征重要性
            diag_importance = []
            for j in range(self.input_dim):
                x_j = x[:, j]
                ladder_output = self.diag_ladders[j](x_j)
                # 使用输出的平均绝对值作为重要性度量
                importance = torch.mean(torch.abs(ladder_output)).item()
                diag_importance.append(importance)

            importance_dict['diagonal_features'] = np.array(diag_importance)

            # 2. 全连接梯子的重要性
            full_importance = []
            for i, ladder in enumerate(self.full_ladders):
                ladder_output = ladder(x)
                importance = torch.mean(torch.abs(ladder_output)).item()
                full_importance.append(importance)

            importance_dict['interaction_features'] = np.array(full_importance)
            importance_dict['interaction_depths'] = np.array(self.full_ladder_depths)

            # 3. 总体特征重要性排序
            all_importance = np.concatenate([diag_importance, full_importance])
            importance_dict['total_importance'] = all_importance

            return importance_dict


# ==========================================================================
# From: CFNN-Experiments/Energy Efficiency/cfnet.py
# (Core Components for Boost & MoE, Boosting Model, MoE Model)
# ==========================================================================

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


# ==========================================================================
# From: /home/zxc/CodeBase/cofrnet/cfnn/cfnn_poly.py
# (CF-as-activation variant: MultiLayerCFN and its dependencies)
# ==========================================================================

def get_polynomial_basis(x: torch.Tensor, num_basis_functions: int) -> torch.Tensor:
    """
    Computes polynomial basis function values for input tensor x.
    Args:
        x: Input tensor, shape (batch_size, 1).
        num_basis_functions: Number of basis functions.

    Returns:
        Basis values tensor, shape (batch_size, num_basis_functions).
    """
    if num_basis_functions <= 0:
        return torch.empty(x.shape[0], 0, dtype=x.dtype, device=x.device)

    # Create polynomial basis: [x^0, x^1, x^2, ..., x^(num_basis_functions-1)]
    # Ensure x is 2D (batch_size, 1)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    elif x.ndim > 2 or x.shape[-1] != 1:
        raise ValueError(f"Input to get_polynomial_basis must be (batch_size, 1) or (batch_size,) but got {x.shape}")

    # Compute powers element-wise across the batch
    # Squeeze x to (batch_size,) for power calculation, then stack
    basis_values = torch.stack([x.squeeze(-1)**i for i in range(num_basis_functions)], dim=1) # Shape (batch_size, num_basis_functions)

    return basis_values

# --- Differentiable Learnable Function Module (uses coefficients for a basis) ---
# This represents a single learnable function f(scalar) -> scalar
class DifferentiableLearnableFunction(nn.Module):
    def __init__(self, num_basis_functions: int):
        """
        Represents a trainable function using learned coefficients for basis functions.
        Args:
            num_basis_functions: Number of basis functions / coefficients to learn.
        """
        super().__init__()
        self.num_basis_functions = num_basis_functions
        # These are the learnable parameters (coefficients for the basis functions)
        # Initialize with smaller values might help stability
        self.coeffs = nn.Parameter(torch.randn(num_basis_functions) * 0.01) # Using smaller initialization

        # Note: We conceptually use basis functions here (polynomials in the placeholder)
        # If using B-splines, basis parameters (knots, degree) would be fixed hyperparameters
        # and the actual B-spline basis evaluation needs a PyTorch implementation.


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the learnable function for input x.
        Args:
            x: Input tensor, shape (batch_size, 1).

        Returns:
            Output tensor, shape (batch_size, 1).
        """
        # Get basis function values for the input x
        # Using polynomial basis placeholder
        basis_values = get_polynomial_basis(x, self.num_basis_functions) # Shape (batch_size, num_basis_functions)

        # Linear combination of basis functions with learned coefficients
        # output = sum(coeffs_i * basis_i(x)) -> batch matrix multiplication
        # (batch_size, num_basis_functions) @ (num_basis_functions,) -> (batch_size,)
        output = torch.matmul(basis_values, self.coeffs)

        # Reshape output to (batch_size, 1) to maintain dimension for potential stacking
        output = output.unsqueeze(-1) # Shape (batch_size, 1)

        return output

# --- Element-wise Continued Fraction Activation Module ---
# This module applies the continued fraction calculation to each element of the input tensor.
class ElementwiseCFN_Activation(nn.Module):
    def __init__(self, depth: int, num_basis_functions_per_term: int):
        """
        Applies a trainable continued fraction function element-wise.

        Args:
            depth: Fixed recursion depth of the continued fraction.
            num_basis_functions_per_term: Number of basis functions/coefficients
                                        for each a_n(.) and b_n(.) term.
        """
        super().__init__()
        self.depth = depth
        self.num_basis_functions_per_term = num_basis_functions_per_term

        # Create ModuleLists to hold the trainable terms (a_n(scalar)->scalar, b_n(scalar)->scalar)
        # Each term uses the DifferentiableLearnableFunction
        # a_n(x) terms: a_0, a_1, ..., a_depth (depth + 1 terms)
        self.a_terms = nn.ModuleList([
            DifferentiableLearnableFunction(num_basis_functions_per_term)
            for _ in range(depth + 1)
        ])

        # b_n(x) terms: b_1, b_2, ..., b_depth (depth terms)
        self.b_terms = nn.ModuleList([
            DifferentiableLearnableFunction(num_basis_functions_per_term)
            for _ in range(depth)
        ])

        # Define clamping limits for numerical stability
        self.clamp_min = -1e0 # Minimum value for intermediate results
        self.clamp_max = 1e0  # Maximum value for intermediate results


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the value of the fixed-depth continued fraction for each element of the input tensor.
        Args:
            x: Input tensor, shape (batch_size, num_features).

        Returns:
            Output tensor, shape (batch_size, num_features).
        """
        # Store original shape to reshape back at the end
        original_shape = x.shape
        batch_size, num_features = original_shape

        # Flatten the input to apply the scalar function logic element-wise
        # Shape becomes (batch_size * num_features, 1)
        x_flat = x.view(-1, 1)

        # Evaluate the bottom term: a_depth(x_flat)
        # Each a_term/b_term takes (batch_size*num_features, 1) and returns (batch_size*num_features, 1)
        result = self.a_terms[self.depth](x_flat) # Shape (batch_size * num_features, 1)

        # Clamp the initial result
        result = torch.clamp(result, self.clamp_min, self.clamp_max)


        # Evaluate upwards recursively/iteratively
        # Iterate from a_{depth-1} down to a_0
        for i in range(self.depth -1, -1, -1):
            a_val = self.a_terms[i](x_flat) # Shape (batch_size * num_features, 1)
            # b_terms index i corresponds to b_{i+1}(x) in the standard CF notation
            b_val = self.b_terms[i](x_flat) # Shape (batch_size * num_features, 1)

            # Clamp a_val and b_val
            a_val = torch.clamp(a_val, self.clamp_min, self.clamp_max)
            b_val = torch.clamp(b_val, self.clamp_min, self.clamp_max)

            # Handle potential division by zero or very small numbers
            # Add epsilon for numerical stability during training
            epsilon = 1e-8 # Keep epsilon


            # Perform the continued fraction step: a_i(x) + b_{i+1}(x) / (a_{i+1}(x) + ...)
            # Add epsilon to the denominator to prevent division by zero
            denominator = result + epsilon * torch.sign(result) # Add epsilon in a sign-preserving way
            # Or a simpler version: denominator = result + epsilon # But can change sign near zero

            # Ensure denominator is not zero after adding epsilon if result was -epsilon
            denominator = torch.where(torch.abs(denominator) < epsilon, torch.tensor(epsilon, device=result.device, dtype=result.dtype) * torch.sign(result), denominator)
            # Handle case where result was exactly 0 -> denominator is epsilon * sign(0) which is 0. Make it epsilon
            denominator = torch.where(denominator == 0, torch.tensor(epsilon, device=result.device, dtype=result.dtype), denominator)


            term_to_add = b_val / denominator # Shape (batch_size * num_features, 1)

            # Clamp the term before adding to prevent explosion
            term_to_add = torch.clamp(term_to_add, self.clamp_min, self.clamp_max)


            result = a_val + term_to_add # Shape (batch_size * num_features, 1)

            # Clamp the result for the next iteration
            result = torch.clamp(result, self.clamp_min, self.clamp_max)


        # Reshape the result back to the original input shape
        result = result.view(original_shape) # Shape (batch_size, num_features)

        return result


# --- Multi-Layer Network using Elementwise CFN Activation ---
class MultiLayerCFN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list,
                cfn_depth: int, cfn_num_basis_functions_per_term: int):
        """
        A multi-layer network using ElementwiseCFN_Activation.

        Args:
            input_dim: Dimension of the input features.
            output_dim: Dimension of the output.
            hidden_dims: A list of integers specifying the number of features in hidden layers.
            cfn_depth: Recursion depth for the Continued Fraction Activation.
            cfn_num_basis_functions_per_term: Number of basis functions for each term
                                            within the CFN Activation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.cfn_depth = cfn_depth
        self.cfn_num_basis_functions_per_term = cfn_num_basis_functions_per_term

        layers = []
        # Input layer
        in_features = input_dim
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(in_features, hidden_dim))
            # CFN Activation layer
            for i in range(hidden_dim):
                layers.append(ElementwiseCFN_Activation(cfn_depth, cfn_num_basis_functions_per_term))
            in_features = hidden_dim

        # Output layer (no activation after the last linear layer for classification)
        layers.append(nn.Linear(in_features, output_dim))

        # Combine layers into a Sequential model
        self.net = nn.Sequential(*layers)

        # Initialize weights with smaller values might help stability
        # Note: Model weights are also initialized based on a random seed
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization suitable for ReLU-like activations (CFN is not ReLU, but often a decent starting point)
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(init_weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x: Input tensor, shape (batch_size, input_dim).

        Returns:
            Output tensor, shape (batch_size, output_dim).
        """
        # Ensure input shape is correct (batch_size, input_dim)
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Input tensor must have shape (batch_size, {self.input_dim}), but got {x.shape}")

        return self.net(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    # -----------------------------------------------------------------
    # Forward-shape verification
    # -----------------------------------------------------------------
    print("=" * 70)
    print("Forward-shape verification (input shape (B,1), expect output (B,1))")
    print("=" * 70)

    B = 8
    x = torch.randn(B, 1)

    cfnet_model = CFNet_Standard(1, 1, depth=2, poly_degree=3)
    cfnet_out = cfnet_model(x)
    print(f"CFNet_Standard(1,1,depth=2,poly_degree=3) output shape: {tuple(cfnet_out.shape)}")
    cfnet_ok = tuple(cfnet_out.shape) == (B, 1)
    assert cfnet_ok, f"Expected shape (B,1)=({B},1), got {tuple(cfnet_out.shape)}"
    print("  -> PASS: CFNet_Standard forward shape is (B,1)")

    cofrnetdl_model = CoFrNetDL(1, 1, num_full_ladders=3, max_diag_ladder_depth=3, max_full_ladder_depth=3)
    cofrnetdl_out = cofrnetdl_model(x)
    print(f"CoFrNetDL(1,1,num_full_ladders=3,max_diag_ladder_depth=3,max_full_ladder_depth=3) output shape: {tuple(cofrnetdl_out.shape)}")
    if tuple(cofrnetdl_out.shape) == (B, 1):
        print("  -> PASS: CoFrNetDL forward shape is (B,1)")
    else:
        print(f"  -> WARNING: CoFrNetDL forward returned shape {tuple(cofrnetdl_out.shape)}, "
              f"NOT (B,1)=({B},1). Not fixed here per instructions; caller must handle.")
    assert tuple(cofrnetdl_out.shape) == (B, 1), \
        f"CoFrNetDL forward shape mismatch: expected (B,1)=({B},1), got {tuple(cofrnetdl_out.shape)}"

    print()

    # -----------------------------------------------------------------
    # Parameter-count table: CFNet_Standard
    # -----------------------------------------------------------------
    print("=" * 70)
    print("Parameter counts: CFNet_Standard(1, 1, depth=d, poly_degree=pd)")
    print("=" * 70)
    print(f"{'depth':>6} {'poly_degree':>12} {'#params':>12}")
    for d in [1, 2, 3, 4, 6, 8]:
        for pd in [2, 3, 5]:
            m = CFNet_Standard(1, 1, depth=d, poly_degree=pd)
            n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"{d:>6} {pd:>12} {n_params:>12}")

    print()

    # -----------------------------------------------------------------
    # Parameter-count table: CoFrNetDL
    # -----------------------------------------------------------------
    print("=" * 70)
    print("Parameter counts: CoFrNetDL(1, 1, num_full_ladders=n, "
          "max_diag_ladder_depth=md, max_full_ladder_depth=md)")
    print("=" * 70)
    print(f"{'num_full_ladders':>18} {'max_ladder_depth':>18} {'#params':>12}")
    for n, md in [(1, 3), (2, 4), (4, 6), (8, 8), (16, 10), (24, 12), (32, 12)]:
        m = CoFrNetDL(1, 1, num_full_ladders=n, max_diag_ladder_depth=md, max_full_ladder_depth=md)
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{n:>18} {md:>18} {n_params:>12}")

    print()

    # -----------------------------------------------------------------
    # Forward-shape verification: MultiLayerCFN
    # -----------------------------------------------------------------
    print("=" * 70)
    print("Forward-shape verification: MultiLayerCFN(1,1,hidden_dims=[2],cfn_depth=2,cfn_num_basis_functions_per_term=3)")
    print("=" * 70)
    mlcfn_model = MultiLayerCFN(1, 1, hidden_dims=[2], cfn_depth=2, cfn_num_basis_functions_per_term=3)
    mlcfn_out = mlcfn_model(x)
    print(f"MultiLayerCFN output shape: {tuple(mlcfn_out.shape)}")
    assert tuple(mlcfn_out.shape) == (B, 1), \
        f"MultiLayerCFN forward shape mismatch: expected (B,1)=({B},1), got {tuple(mlcfn_out.shape)}"
    print("  -> PASS: MultiLayerCFN forward shape is (B,1)")

    print()

    # -----------------------------------------------------------------
    # Forward-shape verification: EnsembleResCoFrNet
    # -----------------------------------------------------------------
    print("=" * 70)
    print("Forward-shape verification: EnsembleResCoFrNet (add_model() x3)")
    print("=" * 70)
    boost_model = EnsembleResCoFrNet(input_dim=1, output_dim=1, shallow_depth=2,
                                      poly_degree=3, learning_rate=0.5)
    for _ in range(3):
        boost_model.add_model()
        boost_model.freeze_all_but_latest()
    boost_out = boost_model(x)
    print(f"EnsembleResCoFrNet output shape after 3x add_model(): {tuple(boost_out.shape)}")
    assert tuple(boost_out.shape) == (B, 1), \
        f"EnsembleResCoFrNet forward shape mismatch: expected (B,1)=({B},1), got {tuple(boost_out.shape)}"
    print("  -> PASS: EnsembleResCoFrNet forward shape is (B,1)")

    print()

    # -----------------------------------------------------------------
    # Forward-shape verification: MoE_Ensemble
    # -----------------------------------------------------------------
    print("=" * 70)
    print("Forward-shape verification: MoE_Ensemble (gating.add_expert_gate() + add_expert() x3)")
    print("=" * 70)
    moe_hparams = {
        'input_dim': 1,
        'output_dim': 1,
        'shallow_depth_per_cofrnet': 2,
        'polynomial_degree': 3,
    }
    moe_model = MoE_Ensemble(moe_hparams)
    for k in range(3):
        center = x[k].detach().cpu().numpy()  # pick a center from the sample batch
        moe_model.gating.add_expert_gate(center, initial_width_param=1.0, device=x.device)
        moe_model.add_expert()
    moe_out = moe_model(x)
    print(f"MoE_Ensemble output shape after 3x (add_expert_gate + add_expert): {tuple(moe_out.shape)}")
    assert tuple(moe_out.shape) == (B, 1), \
        f"MoE_Ensemble forward shape mismatch: expected (B,1)=({B},1), got {tuple(moe_out.shape)}"
    print("  -> PASS: MoE_Ensemble forward shape is (B,1)")

    print()

    # -----------------------------------------------------------------
    # Parameter-count table: MultiLayerCFN
    # -----------------------------------------------------------------
    print("=" * 70)
    print("Parameter counts: MultiLayerCFN(1, 1, hidden_dims=[w], cfn_depth=cd, "
          "cfn_num_basis_functions_per_term=nb)")
    print("=" * 70)
    print(f"{'width(w)':>10} {'cfn_depth':>10} {'num_basis(nb)':>14} {'#params':>12}")
    for w in [1, 2, 4, 8, 16]:
        for cd, nb in [(2, 3), (3, 5)]:
            m = MultiLayerCFN(1, 1, hidden_dims=[w], cfn_depth=cd,
                               cfn_num_basis_functions_per_term=nb)
            n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"{w:>10} {cd:>10} {nb:>14} {n_params:>12}")

    print()

    # -----------------------------------------------------------------
    # Parameter-count table: EnsembleResCoFrNet
    # -----------------------------------------------------------------
    print("=" * 70)
    print("Parameter counts: EnsembleResCoFrNet(1,1,shallow_depth=sd,poly_degree=3,"
          "learning_rate=0.5) after adding n stages")
    print("=" * 70)
    print(f"{'n_stages':>10} {'shallow_depth':>14} {'#params':>12}")
    for sd in [2, 4]:
        for n in [1, 2, 4, 8, 16]:
            m = EnsembleResCoFrNet(input_dim=1, output_dim=1, shallow_depth=sd,
                                    poly_degree=3, learning_rate=0.5)
            for _ in range(n):
                m.add_model()
            n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"{n:>10} {sd:>14} {n_params:>12}")

    print()

    # -----------------------------------------------------------------
    # Parameter-count table: MoE_Ensemble
    # -----------------------------------------------------------------
    print("=" * 70)
    print("Parameter counts: MoE_Ensemble(hparams with shallow_depth_per_cofrnet=sd, "
          "polynomial_degree=3) after adding n experts")
    print("=" * 70)
    print(f"{'n_experts':>10} {'shallow_depth':>14} {'#params':>12}")
    for sd in [2, 4]:
        for n in [1, 2, 4, 8, 16]:
            hparams = {
                'input_dim': 1,
                'output_dim': 1,
                'shallow_depth_per_cofrnet': sd,
                'polynomial_degree': 3,
            }
            m = MoE_Ensemble(hparams)
            for k in range(n):
                center = x[k % B].detach().cpu().numpy()
                m.gating.add_expert_gate(center, initial_width_param=1.0)
                m.add_expert()
            n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"{n:>10} {sd:>14} {n_params:>12}")

    print()
    print("All checks completed.")

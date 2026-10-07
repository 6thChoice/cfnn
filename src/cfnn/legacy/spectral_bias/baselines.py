from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

"""
谱偏差实验基线模型集合
包含: SIREN, RFF-MLP, Chebyshev-KAN, AAA
"""
import torch
import torch.nn as nn
import numpy as np
import math
from scipy.interpolate import AAA


# ==========================================
# 1. SIREN (Sinusoidal Representation Networks)
# ==========================================
class SineLayer(nn.Module):
    """SIREN 的正弦激活层"""
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    Sinusoidal Representation Network
    参考: Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, hidden_layers=2, omega_0=30):
        super().__init__()
        self.net = nn.ModuleList()

        # 第一层
        self.net.append(SineLayer(input_dim, hidden_dim, is_first=True, omega_0=omega_0))

        # 隐藏层
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))

        # 输出层 (线性)
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            self.final_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_0,
                np.sqrt(6 / hidden_dim) / omega_0
            )

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return self.final_layer(x)


# ==========================================
# 2. RFF-MLP (Random Fourier Features + MLP)
# ==========================================
class RandomFourierFeatures(nn.Module):
    """随机傅里叶特征映射层"""
    def __init__(self, input_dim, num_features, sigma=1.0):
        super().__init__()
        self.num_features = num_features
        # 固定的随机投影矩阵 B
        B = torch.randn(input_dim, num_features // 2) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        # 映射: [cos(Bx), sin(Bx)]
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class RFFMLP(nn.Module):
    """
    Random Fourier Features + MLP
    参考: Rahimi & Recht, "Random Features for Large-Scale Kernel Machines"
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, rff_features=128, sigma=1.0):
        super().__init__()
        self.rff = RandomFourierFeatures(input_dim, rff_features, sigma)
        self.mlp = nn.Sequential(
            nn.Linear(rff_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.rff(x)
        return self.mlp(x)


# ==========================================
# 3. Chebyshev-KAN (Chebyshev Kolmogorov-Arnold Network)
# ==========================================
class ChebyshevKANLayer(nn.Module):
    """切比雪夫多项式 KAN 层"""
    def __init__(self, input_dim, output_dim, degree=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # 每个输入-输出对有一组切比雪夫系数
        self.coeffs = nn.Parameter(torch.randn(input_dim, output_dim, degree + 1) * 0.1)

    def forward(self, x):
        # x: (batch, input_dim)
        # 归一化到 [-1, 1]
        x = torch.tanh(x)  # 简单归一化

        # 计算切比雪夫多项式 T_0(x), T_1(x), ..., T_degree(x)
        batch_size = x.shape[0]

        # 使用列表存储避免 inplace operation
        T_list = []
        T_0 = torch.ones(batch_size, self.input_dim, device=x.device)
        T_list.append(T_0)

        if self.degree >= 1:
            T_1 = x
            T_list.append(T_1)

        for k in range(2, self.degree + 1):
            T_k = 2 * x * T_list[k-1] - T_list[k-2]
            T_list.append(T_k)

        # 堆叠: (batch, input_dim, degree+1)
        T = torch.stack(T_list, dim=-1)

        # 加权求和: sum_i sum_k coeffs[i, j, k] * T_k(x_i)
        # T: (batch, input_dim, degree+1)
        # coeffs: (input_dim, output_dim, degree+1)
        out = torch.einsum('bid,iod->bo', T, self.coeffs)
        return out


class ChebyshevKAN(nn.Module):
    """
    Chebyshev Kolmogorov-Arnold Network
    使用切比雪夫多项式作为基函数的 KAN
    """
    def __init__(self, input_dim, output_dim, hidden_dim=16, degree=5, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(ChebyshevKANLayer(input_dim, hidden_dim, degree))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(ChebyshevKANLayer(hidden_dim, hidden_dim, degree))

        # 输出层
        self.layers.append(ChebyshevKANLayer(hidden_dim, output_dim, degree))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ==========================================
# 4. AAA 算法 (非神经网络，用于基准对比)
# ==========================================
class AAAApproximator:
    """
    AAA (Adaptive Antoulas-Anderson) 有理逼近算法
    使用 scipy 实现，仅支持 1D 输入
    """
    def __init__(self):
        self.model = None
        self.support_points = None

    def fit(self, X, y, max_terms=50):
        """
        拟合 AAA 模型
        X: (n_samples,) or (n_samples, 1) - 仅支持 1D
        y: (n_samples,) or (n_samples, 1)
        """
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()

        # scipy 的 AAA 实现 (需要 scipy >= 1.14)
        try:
            from scipy.interpolate import AAA as ScipyAAA
            self.model = ScipyAAA(X, y, max_terms=max_terms)
        except ImportError:
            # 回退到简化实现
            self._fit_simple(X, y, max_terms)

    def _fit_simple(self, X, y, max_terms):
        """简化的 AAA 实现 (基于 barycentric rational approximation)"""
        from scipy.interpolate import BarycentricInterpolator
        # 使用有理插值的近似
        n = min(max_terms, len(X))
        idx = np.linspace(0, len(X)-1, n, dtype=int)
        self.support_points = (X[idx], y[idx])
        self.model = BarycentricInterpolator(X[idx], y[idx])

    def predict(self, X):
        """预测"""
        X = np.asarray(X).flatten()
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model(X)


# ==========================================
# 5. 参数量计算工具函数
# ==========================================
def count_parameters(model):
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_hidden_dim_for_target_params(model_class, target_params, input_dim=3, output_dim=1,
                                      model_type='mlp', **kwargs):
    """
    根据目标参数量反推隐藏层维度
    """
    if model_type == 'mlp':
        # MLP: params = (input_dim * h + h) + (h * h + h) + (h * output_dim + output_dim)
        # 简化: params ≈ input_dim*h + h² + h*output_dim + 2h + output_dim
        # 对于 2 层隐藏层: params = h*(input_dim + output_dim + 2) + h² + output_dim
        a = 1
        b = input_dim + output_dim + 2
        c = output_dim - target_params
        delta = b**2 - 4*a*c
        if delta < 0:
            return 8
        h = (-b + math.sqrt(delta)) / (2*a)
        return max(4, int(h))

    elif model_type == 'siren':
        # SIREN 参数量类似 MLP
        return get_hidden_dim_for_target_params(None, target_params, input_dim, output_dim, 'mlp')

    elif model_type == 'rff_mlp':
        # RFF-MLP: 额外的 RFF 层不增加可训练参数
        return get_hidden_dim_for_target_params(None, target_params, input_dim, output_dim, 'mlp')

    elif model_type == 'chebyshev_kan':
        # Chebyshev-KAN: 每层 params = input_dim * output_dim * (degree + 1)
        degree = kwargs.get('degree', 5)
        # 假设 3 层结构: input->hidden, hidden->hidden, hidden->output
        # params ≈ 2 * input_dim * h * (d+1) + h * output_dim * (d+1)
        d_plus_1 = degree + 1
        # 简化: params ≈ h * (2*input_dim + output_dim) * (d+1)
        h = target_params / ((2*input_dim + output_dim) * d_plus_1)
        return max(4, int(h))

    return 64


# ==========================================
# 6. 模型工厂函数
# ==========================================
def create_baseline_model(model_type, input_dim, output_dim, hidden_dim, **kwargs):
    """
    创建基线模型

    Args:
        model_type: 'mlp', 'siren', 'rff_mlp', 'chebyshev_kan'
        input_dim: 输入维度
        output_dim: 输出维度
        hidden_dim: 隐藏层维度
        **kwargs: 额外参数
            - omega_0: SIREN 的频率参数
            - sigma: RFF 的高斯方差
            - degree: Chebyshev 多项式阶数
    """
    if model_type == 'mlp':
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    elif model_type == 'siren':
        omega_0 = kwargs.get('omega_0', 30)
        return SIREN(input_dim, output_dim, hidden_dim=hidden_dim, hidden_layers=2, omega_0=omega_0)

    elif model_type == 'rff_mlp':
        sigma = kwargs.get('sigma', 1.0)
        rff_features = kwargs.get('rff_features', hidden_dim * 2)
        return RFFMLP(input_dim, output_dim, hidden_dim=hidden_dim, rff_features=rff_features, sigma=sigma)

    elif model_type == 'chebyshev_kan':
        degree = kwargs.get('degree', 5)
        return ChebyshevKAN(input_dim, output_dim, hidden_dim=hidden_dim, degree=degree, num_layers=2)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ==========================================
# 7. 测试代码
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("基线模型测试")
    print("=" * 60)

    input_dim, output_dim = 3, 1
    batch_size = 32
    x = torch.randn(batch_size, input_dim)

    models = {
        'MLP': create_baseline_model('mlp', input_dim, output_dim, hidden_dim=64),
        'SIREN (ω₀=30)': create_baseline_model('siren', input_dim, output_dim, hidden_dim=64, omega_0=30),
        'RFF-MLP': create_baseline_model('rff_mlp', input_dim, output_dim, hidden_dim=64, sigma=1.0),
        'Chebyshev-KAN': create_baseline_model('chebyshev_kan', input_dim, output_dim, hidden_dim=16, degree=5),
    }

    print(f"\n输入: {x.shape}")
    print("-" * 60)

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            y = model(x)
        params = count_parameters(model)
        print(f"{name:20s}: output={y.shape}, params={params}")

    print("\n所有模型测试通过!")

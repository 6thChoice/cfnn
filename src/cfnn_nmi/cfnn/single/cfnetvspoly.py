import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ==========================================
# 1. 基础组件: PolynomialTerm (保持不变)
# ==========================================
class PolynomialTerm(nn.Module):
    """
    基础多项式项: Sum( c_k * tanh(Wx)^k )
    """
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.degree = degree
        self.projection = nn.Linear(input_dim, output_dim)
        self.coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.1)

    def forward(self, x):
        z = torch.tanh(self.projection(x))
        powers = [torch.ones_like(z)]
        for d in range(1, self.degree + 1):
            powers.append(powers[-1] * z)
        z_powered = torch.stack(powers, dim=-1)
        return torch.sum(z_powered * self.coeffs, dim=2)

# ==========================================
# 2. 原有模型 (CFNet & RationalNet) (保持不变，省略以节省空间)
# ... (您的 CFNet 和 RationalNet 代码应保留在此处) ...
# ==========================================
class CFNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, depth=4, poly_degree=3):
        super().__init__()
        self.terms = nn.ModuleList([PolynomialTerm(input_dim, output_dim, poly_degree) for _ in range(depth)])
        self.raw_betas = nn.ParameterList([nn.Parameter(torch.tensor([1.0])) for _ in range(max(0, depth - 1))])
    def forward(self, x):
        if not self.terms: return torch.zeros_like(x)
        output = self.terms[-1](x)
        output = torch.abs(output) + 1.0 
        for i in range(len(self.terms) - 2, -1, -1):
            beta = F.softplus(self.raw_betas[i])
            output = self.terms[i](x) + beta / (output + 1e-8)
        return output

class RationalNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, poly_degree=3):
        super().__init__()
        self.P = PolynomialTerm(input_dim, output_dim, poly_degree)
        self.Q = PolynomialTerm(input_dim, output_dim, poly_degree)
    def forward(self, x):
        numerator = self.P(x)
        denominator = self.Q(x)
        return numerator / (torch.abs(denominator) + 0.1)

# ==========================================
# 2. 新增: 单个有理单元 (封装方案一)
# ==========================================
class RationalUnit(nn.Module):
    """
    原子组件: P(x) / (Q(x)^2 + 1)
    """
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        # 保持与连分式一致的多项式结构
        self.P = PolynomialTerm(input_dim, output_dim, degree)
        self.Q = PolynomialTerm(input_dim, output_dim, degree)

    def forward(self, x):
        # 方案一：结构性正定分母，消除极点和梯度爆炸
        denom = self.Q(x) ** 2 + 1.0
        return self.P(x) / denom

# ==========================================
# 3. 改进: 支持级联数量的混合网络
# ==========================================
class HybridRationalNet(nn.Module):
    """
    结构: Linear(x) + Sum( RationalUnit_i(x) )
    参数:
      - unit_degree: 每个单元的多项式阶数 (建议设小一点，如 2 或 3)
      - num_units:   加法级联的数量 (即并行的有理单元个数)
    """
    def __init__(self, input_dim=1, output_dim=1, unit_degree=3, num_units=4):
        super().__init__()
        
        # 1. 线性跳跃连接：负责拟合全局趋势 (Global Trend)
        self.linear_skip = nn.Linear(input_dim, output_dim)

        # 2. 加法级联：创建 num_units 个并行的有理单元
        #    相比于一个高阶的有理式，多个低阶有理式的和更容易训练且更稳定
        self.units = nn.ModuleList([
            RationalUnit(input_dim, output_dim, unit_degree)
            for _ in range(num_units)
        ])

    def forward(self, x):
        # 初始基准
        total_output = self.linear_skip(x)
        
        # 累加所有单元的输出 (Additive Cascade)
        for unit in self.units:
            total_output = total_output + unit(x)
            
        return total_output
    
# ==========================================
# 3. 新增: MLP 基准模型
# ==========================================
class MLP(nn.Module):
    """
    标准的 3 层感知机
    使用 Tanh 激活函数，与论文中讨论的激活函数保持一致，且适合拟合光滑函数。
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # 论文重点讨论了 Tanh 的硬件实现 [cite: 23]
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. 数据生成与训练工具 (保持不变)
# ==========================================
def generate_rational_data(n=1000):
    x = torch.linspace(-2, 2, n).view(-1, 1)
    # 构造一个有明显波峰波谷且不对称的函数
    y = (2 * x - 1) / (x**2 + 0.2) + 0.3 * torch.sin(5 * x)
    return x.to(DEVICE), y.to(DEVICE)

def train(model, x, y, epochs=2000, lr=0.01, name="Model"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    
    # [优化] 添加简单的学习率衰减，进一步消除后期震荡
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)

    model.train()
    for i in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        
        # [优化] 简单的梯度裁剪，防止偶尔的梯度爆炸
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        # scheduler.step()
        losses.append(loss.item())
        
        if i % 500 == 0:
            print(f"[{name}] Epoch {i}: Loss = {loss.item():.6f}")
    
    return losses

# ==========================================
# 5. 主程序对比 (已更新)
# ==========================================
if __name__ == "__main__":
    # 参数设置
    CF_DEGREE = 4       
    CF_DEPTH = 4     
    POLY_DEPTH = 16
    HYBRID_DEGREE = 8  # 混合模型通常不需要那么高的阶数，8阶足矣
    HYBRID_UNITS = 4
    HYBRID_DEGREE = 4  
    MLP_HIDDEN = 64
    LR = 0.02
    EPOCHS = 3000
    
    x, y = generate_rational_data(1000)
    
    # 实例化模型
    cf_net = CFNet(depth=CF_DEPTH, poly_degree=CF_DEGREE).to(DEVICE)
    rational_net = RationalNet(poly_degree=POLY_DEPTH).to(DEVICE)
    
    # 新增混合模型
    hybrid_net = HybridRationalNet(unit_degree=HYBRID_DEGREE, num_units=HYBRID_UNITS).to(DEVICE)

    # 3. MLP 基准
    mlp_net = MLP(hidden_dim=MLP_HIDDEN).to(DEVICE)
    
    print("--- Training CFNet (Recursive) ---")
    loss_cf = train(cf_net, x, y, epochs=EPOCHS, lr=LR, name="CFNet")
    
    print("\n--- Training RationalNet (Naive P/Q) ---")
    loss_rat = train(rational_net, x, y, epochs=EPOCHS, lr=LR, name="Rational")

    print(f"\n--- Training HybridNet (Units={HYBRID_UNITS}, Degree={HYBRID_DEGREE}) ---")
    loss_hybrid = train(hybrid_net, x, y, epochs=EPOCHS, lr=LR, name="Hybrid")

    print(f"\n--- Training MLP (Hidden={MLP_HIDDEN}) ---")
    loss_mlp = train(mlp_net, x, y, epochs=EPOCHS, lr=LR, name="MLP")
    
    # 可视化
    cf_net.eval()
    rational_net.eval()
    hybrid_net.eval()
    mlp_net.eval()
    
    with torch.no_grad():
        pred_cf = cf_net(x).cpu().numpy()
        pred_rat = rational_net(x).cpu().numpy()
        pred_hybrid = hybrid_net(x).cpu().numpy()
        pred_mlp = mlp_net(x).cpu().numpy()
    
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
    
    plt.figure(figsize=(14, 6))
    
    # 1. 拟合曲线
    plt.subplot(1, 2, 1)
    plt.scatter(x_np, y_np, c='gray', s=2, alpha=0.5, label='Ground Truth')
    plt.plot(x_np, pred_cf, 'r-', linewidth=1.5, label='CFNet')
    plt.plot(x_np, pred_rat, 'b--', linewidth=1.5, alpha=0.6, label='RationalNet (Naive)')
    plt.plot(x_np, pred_hybrid, 'g-', linewidth=2.5, label='HybridNet (Optimized)') # 绿色粗线
    plt.plot(x_np, pred_mlp, 'k-.', linewidth=1.5, alpha=0.8, label='MLP (Baseline)')
    plt.title('Function Fitting Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Loss 曲线
    plt.subplot(1, 2, 2)
    plt.plot(loss_cf, 'r-', linewidth=1, label='CFNet')
    plt.plot(loss_rat, 'b--', linewidth=1, alpha=0.6, label='RationalNet')
    plt.plot(loss_hybrid, 'g-', linewidth=2, label='HybridNet')
    plt.plot(loss_mlp, 'k-.', label='MLP')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.title('Training Convergence')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("hyb.png")

    print(f"\n最终 Loss 对比:")
    print(f"CFNet Final: {loss_cf[-1]:.6f}")
    print(f"RationalNet Final: {loss_rat[-1]:.6f}")
    print(f"HybridNet Final: {loss_hybrid[-1]:.6f}")
    print(f"MLP (Baseline)    : {loss_mlp[-1]:.6f}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Part 1: 独立的 CFNet 定义 (支持动态生长)
# ==========================================

class PolynomialTerm(nn.Module):
    """多项式基函数项 (用于 DynamicCFNet)"""
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.degree = degree
        self.projection = nn.Linear(input_dim, output_dim)
        self.coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.01)

    def forward(self, x):
        z = torch.tanh(self.projection(x)) 
        powers = [torch.ones_like(z)]
        for d in range(1, self.degree + 1):
            powers.append(powers[-1] * z)
        z_powered = torch.stack(powers, dim=-1)
        return torch.sum(z_powered * self.coeffs, dim=2)

class DynamicCFNet(nn.Module):
    """支持垂直生长的连分式网络"""
    def __init__(self, input_dim=1, output_dim=1, initial_depth=1, poly_degree=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.poly_degree = poly_degree
        self.terms = nn.ModuleList([
            PolynomialTerm(input_dim, output_dim, poly_degree) 
            for _ in range(initial_depth)
        ])
        self.raw_betas = nn.ParameterList([
            nn.Parameter(torch.tensor([0.0])) 
            for _ in range(max(0, initial_depth - 1))
        ])

    def forward(self, x):
        if not self.terms: return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        output = F.softplus(self.terms[-1](x)) + 1.0
        for i in range(len(self.terms) - 2, -1, -1):
            beta = F.softplus(self.raw_betas[i])
            output = self.terms[i](x) + beta / (output + 1e-8)
        return output

    def grow_depth(self):
        """增加一层深度"""
        new_term = PolynomialTerm(self.input_dim, self.output_dim, self.poly_degree).to(DEVICE)
        self.terms.append(new_term)
        # 恒等初始化: beta 极小，使得新层初始影响几乎为 0
        new_beta = nn.Parameter(torch.tensor([-5.0])).to(DEVICE)
        self.raw_betas.append(new_beta)
        logging.info(f"[Progressive] 模型垂直生长: 深度 {len(self.terms)-1} -> {len(self.terms)}")

# ==========================================
# Part 2: MLP 基准模型
# ==========================================

class MLP(nn.Module):
    """标准多层感知机 (Baseline)"""
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Tanh 通常比 ReLU 更适合平滑函数逼近
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# Part 3: MoE 相关定义 (集成 V3 Freeze Logic)
# ==========================================

class MoE_PolynomialTerm(nn.Module):
    """MoE 专用的多项式项 (保持与原 MoE 代码一致)"""
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

class MoE_CFNet(nn.Module):
    """MoE 专用的固定深度 CFNet (专家)"""
    def __init__(self, input_dim, output_dim, depth, poly_degree):
        super().__init__()
        self.output_dim = output_dim
        self.terms = nn.ModuleList([MoE_PolynomialTerm(input_dim, output_dim, poly_degree) for _ in range(depth)])
        self.raw_betas = nn.ParameterList([nn.Parameter(torch.full((output_dim,), 0.1)) for _ in range(depth - 1)])
    def forward(self, x):
        if not self.terms: return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        output = F.softplus(self.terms[-1](x)) + 1.0
        for i in range(len(self.terms) - 2, -1, -1):
            output = self.terms[i](x) + F.softplus(self.raw_betas[i]) / (output + 1e-8)
        return output

class RBFGatingNetwork(nn.Module):
    """支持参数列表的 RBF 门控"""
    def __init__(self, input_dim):
        super().__init__()
        self.centers_list = nn.ParameterList()
        self.widths_list = nn.ParameterList()
    def forward(self, x):
        if len(self.centers_list) == 0: return torch.ones(x.shape[0], 1, device=x.device)
        centers = torch.cat([c for c in self.centers_list], dim=0)
        widths = torch.cat([w for w in self.widths_list], dim=0)
        x_reshaped = x.unsqueeze(1)
        centers_reshaped = centers.unsqueeze(0)
        positive_widths = F.softplus(widths.squeeze()) + 1e-8
        dist_sq = torch.sum((x_reshaped - centers_reshaped) ** 2, dim=-1)
        logits = -dist_sq / (2 * (positive_widths ** 2) + 1e-6)
        return logits
    def add_expert_gate(self, center, width_param):
        target_device = self.centers_list[0].device if self.centers_list else DEVICE
        self.centers_list.append(nn.Parameter(torch.tensor(center, dtype=torch.float32, device=target_device).unsqueeze(0)))
        self.widths_list.append(nn.Parameter(torch.tensor([[width_param]], dtype=torch.float32, device=target_device)))

class MoE_Ensemble(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.experts = nn.ModuleList()
        self.gating = RBFGatingNetwork(self.hparams['input_dim'])
    def forward(self, x):
        if not self.experts: return torch.zeros(x.shape[0], self.hparams['output_dim'], device=x.device)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)
        gate_weights = F.softmax(self.gating(x), dim=-1).unsqueeze(-1)
        return torch.sum(gate_weights * expert_outputs_stacked, dim=1)
    def add_expert(self):
        self.experts.append(MoE_CFNet(self.hparams['input_dim'], self.hparams['output_dim'], 
                                      self.hparams['shallow_depth'], self.hparams['polynomial_degree']))

class MoE_Manager:
    """MoE 训练管理器 (V3 Freeze Logic)"""
    def __init__(self, hparams):
        self.hparams = hparams
        self.model = MoE_Ensemble(hparams).to(DEVICE)
    
    def train_iterative(self, x, y):
        """
        在全量数据上进行迭代式训练:
        1. 训练 Expert 1
        2. 冻结 Expert 1, 寻找误差最大处, 添加 Expert 2 (Small Sigma)
        3. ...
        """
        loss_history = []
        criterion = nn.MSELoss()
        max_experts = self.hparams['max_experts']
        epochs_per_stage = self.hparams['epochs_per_model']
        
        # 数据集
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.hparams['batch_size'], shuffle=True)
        full_loader = torch.utils.data.DataLoader(dataset, batch_size=self.hparams['batch_size'], shuffle=False)

        for i in range(max_experts):
            logging.info(f"\n[MoE] --- Iteration: Adding Expert {i+1} ---")
            
            # 1. 寻找初始化中心
            if i == 0:
                center = x.mean(dim=0).cpu().numpy()
            else:
                self.model.eval()
                all_losses = []
                with torch.no_grad():
                    for bx, by in full_loader:
                        pred = self.model(bx)
                        loss = F.mse_loss(pred, by, reduction='none').sum(dim=1)
                        all_losses.append(loss)
                all_losses = torch.cat(all_losses).cpu().numpy()
                center_idx = np.argmax(all_losses)
                center = x[center_idx].cpu().numpy()
            
            # 2. 冻结旧参数
            frozen_count = 0
            for param in self.model.parameters():
                param.requires_grad = False
                frozen_count += 1
            
            # 3. 添加新专家 (Small Sigma)
            target_sigma = 0.08
            width_param = np.log(np.exp(target_sigma) - 1)
            self.model.add_expert()
            self.model.gating.add_expert_gate(center, width_param)
            self.model.to(DEVICE)
            
            # 4. 训练新专家
            # 只优化 requires_grad=True 的参数
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                   lr=self.hparams['lr'])
            
            self.model.train()
            for epoch in range(epochs_per_stage):
                epoch_loss = 0
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = self.model(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                loss_history.append(epoch_loss / len(loader))
        
        return loss_history

# ==========================================
# Part 4: 实验辅助函数
# ==========================================

def generate_data(n_samples=1000):
    """复杂数据: 包含高频震荡、奇点趋势和有理特征"""
    x = torch.linspace(-1.5, 1.5, n_samples).view(-1, 1)
    # y = sin(4*pi*x) [高频] + 1/(1+25x^2) [Runge现象] + 0.5*tan(x) [奇点]
    y = torch.sin(4 * np.pi * x) + 1.0 / (1 + 25 * x**2) + 0.5 * torch.tan(x)
    # 稍微裁剪一下极端值以便于训练可视化
    y = torch.clamp(y, -10, 10)
    return x.to(DEVICE), y.to(DEVICE)

def train_model(model, x, y, epochs, lr=0.01, mode="Joint"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % (epochs // 5) == 0:
            logging.info(f"[{mode}] Epoch {epoch}: Loss {loss.item():.6f}")
    return loss_history

def train_progressive(model, x, y, total_epochs, target_depth, lr=0.01):
    criterion = nn.MSELoss()
    loss_history = []
    epochs_per_stage = total_epochs // target_depth
    
    for depth in range(1, target_depth + 1):
        logging.info(f"\n[Progressive] --- Stage: Depth {depth} ---")
        if depth > 1:
            model.grow_depth()
            # 冻结旧层
            for i in range(len(model.terms) - 1):
                for param in model.terms[i].parameters(): param.requires_grad = False
            for i in range(len(model.raw_betas) - 1): model.raw_betas[i].requires_grad = False
            
            # Warmup New Layer
            optimizer_new = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            for _ in range(epochs_per_stage // 2):
                optimizer_new.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_new.step()
                loss_history.append(loss.item())
            
            # 解冻
            for param in model.parameters(): param.requires_grad = True
        
        # 整体微调
        current_lr = lr * (0.8 ** (depth - 1))
        optimizer = optim.Adam(model.parameters(), lr=current_lr)
        loops = epochs_per_stage if depth == 1 else (epochs_per_stage // 2)
        for _ in range(loops):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_history.append(loss.item())
    return loss_history

# ==========================================
# Part 5: 主程序
# ==========================================

if __name__ == "__main__":
    # --- 参数设置 ---
    TOTAL_EPOCHS = 3000
    TARGET_DEPTH = 5
    POLY_DEGREE = 3
    LR = 0.02
    
    # MoE 参数
    MOE_HPARAMS = {
        'input_dim': 1, 'output_dim': 1,
        'shallow_depth': 3,    
        'polynomial_degree': 3,
        'learning_rate_adam': 0.1,
        'batch_size': 256,
        'max_experts': 8,      
        'epochs_per_model': 300, 
        'lr': 0.02
    }
    
    # 1. 准备数据
    x, y = generate_data(1200)
    
    # 2. 联合训练 (Joint Training)
    logging.info(f"{'='*30}\n开始联合训练 (Joint Training)...\n{'='*30}")
    model_joint = DynamicCFNet(input_dim=1, output_dim=1, initial_depth=TARGET_DEPTH, poly_degree=POLY_DEGREE).to(DEVICE)
    loss_joint = train_model(model_joint, x, y, epochs=TOTAL_EPOCHS, lr=LR, mode="Joint")
    
    # 3. 渐进式训练 (Progressive Training)
    logging.info(f"\n{'='*30}\n开始渐进式训练 (Progressive Training)...\n{'='*30}")
    model_prog = DynamicCFNet(input_dim=1, output_dim=1, initial_depth=1, poly_degree=POLY_DEGREE).to(DEVICE)
    loss_prog = train_progressive(model_prog, x, y, total_epochs=TOTAL_EPOCHS, target_depth=TARGET_DEPTH, lr=LR)
    
    # 4. MoE 训练 (Horizontal Growth)
    logging.info(f"\n{'='*30}\n开始 MoE 训练 (Iterative Experts)...\n{'='*30}")
    moe_manager = MoE_Manager(MOE_HPARAMS)
    loss_moe = moe_manager.train_iterative(x, y)
    
    # 5. MLP 训练 (Deep Learning Baseline)
    logging.info(f"\n{'='*30}\n开始 MLP 训练 (Baseline)...\n{'='*30}")
    # 3层隐藏层，128个神经元，参数量明显大于 CFNet，作为强力对比
    model_mlp = MLP(input_dim=1, hidden_dim=128, output_dim=1).to(DEVICE)
    loss_mlp = train_model(model_mlp, x, y, epochs=TOTAL_EPOCHS, lr=0.01, mode="MLP")

    # 6. 可视化对比
    plt.figure(figsize=(18, 6))
    
    # 图 1: Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_joint, label='Joint (Fixed Depth 5)', alpha=0.5, color='gray')
    plt.plot(loss_prog, label='Progressive (Growing Depth 1->5)', linewidth=2, color='blue')
    
    # MoE x轴对齐
    moe_x = np.linspace(0, len(loss_joint), len(loss_moe))
    plt.plot(moe_x, loss_moe, label='MoE (Adding Experts 1->8)', linewidth=2, color='orange')
    
    # MLP Loss
    plt.plot(loss_mlp, label='MLP (3x128 Tanh)', linewidth=2, color='purple', linestyle='--')
    
    plt.yscale('log')
    plt.xlabel('Approx. Iterations')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.title('Training Dynamics Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 图 2: 拟合效果
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        pred_joint = model_joint(x).cpu().numpy()
        pred_prog = model_prog(x).cpu().numpy()
        
        # MoE 预测
        moe_manager.model.eval()
        pred_moe = moe_manager.model(x).cpu().numpy()
        
        # MLP 预测
        model_mlp.eval()
        pred_mlp = model_mlp(x).cpu().numpy()
        
        x_cpu = x.cpu().numpy()
        y_cpu = y.cpu().numpy()
        
    plt.scatter(x_cpu, y_cpu, s=1, c='black', alpha=0.2, label='True Data')
    plt.plot(x_cpu, pred_joint, 'r--', alpha=0.4, label=f'Joint')
    # plt.plot(x_cpu, pred_prog, 'b-', linewidth=1.5, label=f'Progressive')
    plt.plot(x_cpu, pred_moe, 'g-', linewidth=1.5, label=f'MoE')
    plt.plot(x_cpu, pred_mlp, 'm:', linewidth=2, label=f'MLP')
    
    plt.title('Function Approximation Result')
    plt.legend()
    plt.ylim(-12, 12) 
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cfnet_benchmark_comparison_v3_mlp.png')
    logging.info("结果已保存至 cfnet_benchmark_comparison_v3_mlp.png")
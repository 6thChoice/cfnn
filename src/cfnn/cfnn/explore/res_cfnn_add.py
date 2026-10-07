import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import sys

# --- 配置与CoFrNet定义 (与之前版本相同) ---
def setup_logging(log_file='training_final_ensemble.log'):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger(); root_logger.handlers = []; root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w'); file_handler.setFormatter(log_formatter); root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); root_logger.addHandler(console_handler)

torch.manual_seed(42); np.random.seed(42)

class TermNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.layer(x)

class CoFrNet(nn.Module):
    def __init__(self, input_dim, depth, hidden_dim_per_term=64):
        super().__init__()
        self.terms = nn.ModuleList([TermNet(input_dim, hidden_dim_per_term) for _ in range(depth)])
        self.raw_betas = nn.ParameterList([nn.Parameter(torch.tensor([0.85])) for _ in range(depth - 1)])
    def forward(self, x):
        if not self.terms: return torch.zeros_like(x)
        output = F.softplus(self.terms[-1](x)) + 1.0
        for i in range(len(self.terms) - 2, -1, -1):
            beta = F.softplus(self.raw_betas[i])
            output = self.terms[i](x) + beta / output
        return output

class EnsembleResCoFrNet(nn.Module):
    def __init__(self, input_dim, shallow_depth=3, hidden_dim=64, learning_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.shallow_depth = shallow_depth
        self.hidden_dim = hidden_dim
        self.models = nn.ModuleList([])
        # 1. 引入学习率/收缩率
        self.learning_rate = learning_rate

    def forward(self, x):
        if not self.models:
            return torch.zeros_like(x)
        # 预测是所有子模型 *加权* 后的总和
        total_prediction = sum(self.learning_rate * model(x) for model in self.models)
        return total_prediction
        
    def add_model(self):
        logging.info(f"---> 集成模型增加新的子网络 (当前共 {len(self.models) + 1} 个) ---")
        new_cofr_net = CoFrNet(self.input_dim, self.shallow_depth, self.hidden_dim)
        self.models.append(new_cofr_net)

def train_final_ensemble(ensemble_model, X_train, y_train, X_val, y_val,
                         num_models_max=30, accuracy_target=0.01, epochs_per_model=1000):
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    val_loss_history = []
    
    current_train_preds = torch.zeros_like(y_train)

    logging.info("="*30)
    logging.info("  启动 Res-CoFrNet 训练流程 (完全体)")
    logging.info("="*30)

    for i in range(num_models_max):
        ensemble_model.add_model()
        new_model = ensemble_model.models[-1]

        residual_target_train = y_train - current_train_preds
        
        for model in ensemble_model.models[:-1]:
            for param in model.parameters(): param.requires_grad = False
        for param in new_model.parameters(): param.requires_grad = True

        # 3. 引入 L2 正则化 (weight_decay)
        optimizer = optim.Adam(new_model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=80)

        logging.info(f"--- 正在训练第 {i+1} 个子网络 ---")
        for epoch in range(epochs_per_model):
            new_model.train()
            optimizer.zero_grad()
            pred_residual = new_model(X_train)
            loss = criterion(pred_residual, residual_target_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(new_model.parameters(), 1.0)
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                new_model.eval()
                with torch.no_grad():
                    # 注意：这里的验证集评估仅用于调度器，不代表最终性能
                    val_residual_preds = new_model(X_val)
                    current_val_total_preds = ensemble_model(X_val)
                    val_residual_target = y_val - (current_val_total_preds - ensemble_model.learning_rate * val_residual_preds)
                    val_residual_loss = criterion(val_residual_preds, val_residual_target)
                    scheduler.step(val_residual_loss)
                    logging.info(f"子网络 {i+1} | Epoch {epoch+1}/{epochs_per_model} | 残差损失(Val): {val_residual_loss.item():.6f}")

        ensemble_model.eval()
        with torch.no_grad():
            # 使用学习率进行加权
            current_train_preds += ensemble_model.learning_rate * new_model(X_train)
            
            # 评估整个集成模型的性能
            val_total_preds = ensemble_model(X_val)
            val_loss = criterion(val_total_preds, y_val)

        val_loss_history.append(val_loss.item())
        
        logging.info(f"第 {i+1} 个子网络训练完成。集成模型总体验证集损失: {val_loss.item():.6f}")
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            logging.info(f"*** 新的最佳损失: {best_val_loss:.6f}，在 {i+1} 个子网络时达到。 ***")
        else:
            best_iter_so_far = np.argmin(val_loss_history)
            patience = 5
            if i - best_iter_so_far > patience:
                logging.error(f"性能连续 {patience} 次未改善 (最佳在第 {best_iter_so_far + 1} 次)，提前停止！")
                break
        
        if best_val_loss < accuracy_target:
            logging.info(f"目标精度 {accuracy_target} 已达到。停止训练。")
            break

    # 裁剪模型到最佳大小
    best_size = np.argmin(val_loss_history) + 1
    ensemble_model.models = nn.ModuleList(list(ensemble_model.models)[:best_size])
    logging.info(f"训练流程结束。裁剪模型到最佳大小: {best_size} 个子网络。")
    return ensemble_model

def count_parameters(model: nn.Module):
    """
    计算并打印模型中可训练参数和总参数的数量。
    
    Args:
        model (nn.Module): 需要计算参数的PyTorch模型。
    """
    # 计算可训练参数的数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 计算总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 使用 logging 模块进行输出，与您代码风格保持一致
    logging.info("-" * 40)
    logging.info("模型参数统计:")
    # 使用 f-string 的格式化功能，让数字更易读 (例如: 1,234,567)
    logging.info(f"  - 可训练参数量: {trainable_params:,}")
    logging.info(f"  - 总参数量:     {total_params:,}")
    logging.info("-" * 40)
    return total_params

if __name__ == '__main__':
    setup_logging('training_final_ensemble.log')
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False
    X = torch.linspace(-np.pi, np.pi, 500).view(-1, 1); y = torch.sin(X) + torch.cos(2*X) + torch.randn(X.size(0), 1) * 0.1
    train_size = int(0.8*len(X)); X_train, X_val = X[:train_size], X[train_size:]; y_train, y_val = y[:train_size], y[train_size:]
    
    res_cofr_net = EnsembleResCoFrNet(input_dim=1, shallow_depth=3, hidden_dim=64, learning_rate=0.1)

    start_time = time.time()
    final_model = train_final_ensemble(
        res_cofr_net, X_train, y_train, X_val, y_val,
        num_models_max=100, accuracy_target=0.015, epochs_per_model=2000
    )
    
    logging.info(f"总训练时间: {time.time() - start_time:.2f} 秒")
    if final_model and final_model.models:
        logging.info(f"最终模型大小: {len(final_model.models)} 个子网络。")

        # =======================================================
        # 【新增功能】统计并打印最终模型的参数量
        count_parameters(final_model)
        # =======================================================
        
        final_model.eval()
        with torch.no_grad(): final_preds = final_model(X_val).numpy()
        plt.figure(figsize=(12, 8)); plt.title("残差连分式网络 (完全体)")
        plt.plot(X.numpy(), y.numpy(), 'k.', alpha=0.2, label="真实数据")
        plt.plot(X_val.numpy(), final_preds, 'r-', linewidth=2, label='最终模型预测')
        plt.legend(); plt.grid(True); plt.xlabel("x"); plt.ylabel("y")
        plt.savefig('training_final_ensemble_plot.png'); plt.show()
        logging.info("程序执行完毕，图表已保存为 training_final_ensemble_plot.png。")
    else:
        logging.info("未能成功训练模型。")
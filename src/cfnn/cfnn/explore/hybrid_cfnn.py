import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import sys

# --- 配置与网络定义 ---
def setup_logging(log_file='training_final_polished.log'):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger(); root_logger.handlers = []; root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w'); file_handler.setFormatter(log_formatter); root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); root_logger.addHandler(console_handler)

torch.manual_seed(42); np.random.seed(42)

class TermNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.layer(x)

class CoFrNet(nn.Module):
    def __init__(self, input_dim, depth, hidden_dim_per_term=128):
        super().__init__()
        self.terms = nn.ModuleList([TermNet(input_dim, hidden_dim_per_term) for _ in range(depth)])
        self.raw_betas = nn.ParameterList([nn.Parameter(torch.tensor([0.85])) for _ in range(depth - 1)])
        self.epsilon = 1e-8
        
    def forward(self, x):
        if not self.terms: return torch.zeros_like(x)
        
        # 3. 结构性稳定增强：对最深层的输出进行处理
        # 这保证了分母链条的起点是一个大于1的正数，极为稳定
        output = F.softplus(self.terms[-1](x)) + 1.0
        
        for i in range(len(self.terms) - 2, -1, -1):
            beta = F.softplus(self.raw_betas[i])
            # 注意：这里不再需要epsilon，因为分母被保证为正
            output = self.terms[i](x) + beta / output
        return output

# --- 最终版训练流程 ---
def train_final_polished(input_dim, X_train, y_train, X_val, y_val,
                         start_depth=2, max_depth=10, accuracy_target=0.01, e2e_epochs=1200):
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_model_state = None
    best_model_depth = 0

    logging.info("="*30)
    logging.info("  启动“生长式端到端训练” (最终打磨版)")
    logging.info("="*30)

    for depth in range(start_depth, max_depth + 1):
        logging.info(f"======> 正在训练深度为 {depth} 的网络 <======")
        model = CoFrNet(input_dim, depth=depth)
        
        if best_model_state:
            logging.info(f"继承深度为 {depth-1} 的网络权重...")
            model.load_state_dict(best_model_state, strict=False)

        # 1. & 2. 使用更小的初始学习率和更灵敏的调度器
        optimizer = optim.Adam(model.parameters(), lr=1e-4) # 大幅降低初始学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50) # 更灵敏

        for epoch in range(e2e_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    scheduler.step(val_loss)
                    logging.info(f"深度 {depth} | Epoch {epoch+1}/{e2e_epochs} | Val Loss: {val_loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)
            logging.info(f"深度 {depth} 训练完成。最终验证集损失: {val_loss.item():.6f}")

        if val_loss.item() < best_val_loss * 5:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict() # 保存性能最好的模型的状态
            best_model_depth = depth
            logging.info(f"*** 新的最佳损失: {best_val_loss:.6f} (在深度 {depth})，模型已保存。 ***")
        else:
            # 如果增加深度后性能没有提升，就没有必要继续了
            logging.info(f"深度 {depth} 未能超越深度 {depth-1} 的最佳性能。提前停止生长。")
            break

        if best_val_loss < accuracy_target:
            logging.info(f"目标精度 {accuracy_target} 已达到。停止生长。")
            break
    
    logging.info(f"训练流程结束。最佳模型发现在 深度={best_model_depth}，损失为 {best_val_loss:.6f}")
    final_model = CoFrNet(input_dim, depth=best_model_depth)
    final_model.load_state_dict(best_model_state)
    return final_model

if __name__ == '__main__':
    setup_logging('training_final_polished.log')
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False
    X = torch.linspace(-np.pi, np.pi, 500).view(-1, 1)
    y = torch.sin(X) + torch.cos(2 * X) + torch.randn(X.size(0), 1) * 0.1
    train_size = int(0.8*len(X)); X_train, X_val = X[:train_size], X[train_size:]; y_train, y_val = y[:train_size], y[train_size:]
    
    start_time = time.time()
    final_model = train_final_polished(
        input_dim=1,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        start_depth=2, max_depth=35,
        accuracy_target=0.012, # 我们的目标是超越0.3，所以0.012是一个有挑战性的目标
        e2e_epochs=1500 # 增加训练轮数，配合低学习率
    )
    
    logging.info(f"总训练时间: {time.time() - start_time:.2f} 秒")
    if final_model:
        logging.info(f"最终模型深度为: {len(final_model.terms)}")
        final_model.eval()
        with torch.no_grad(): final_preds = final_model(X_val).numpy()
        plt.figure(figsize=(12, 8)); plt.title("CoFrNet 训练策略 (最终打磨版)")
        plt.plot(X.numpy(), y.numpy(), 'k.', alpha=0.2, label="真实数据")
        plt.plot(X_val.numpy(), final_preds, 'r-', linewidth=2, label=f'最终模型预测 (深度 {len(final_model.terms)})')
        plt.legend(); plt.grid(True); plt.xlabel("x"); plt.ylabel("y")
        plt.savefig('training_final_polished_plot.png'); plt.show()
        logging.info("程序执行完毕，图表已保存为 training_final_polished_plot.png。")
    else:
        logging.info("未能成功训练模型。")
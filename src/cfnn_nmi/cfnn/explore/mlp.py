import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 定义MLP模型 ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化MLP模型。
        
        Args:
            input_dim (int): 输入特征的维度 (这里是1, 因为x是一个值).
            hidden_dim (int): 隐藏层的神经元数量.
            output_dim (int): 输出的维度 (这里是1, 因为y是一个值).
        """
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # 非线性激活函数
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """定义前向传播."""
        return self.model(x)

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
    print("-" * 40)
    print("模型参数统计:")
    # 使用 f-string 的格式化功能，让数字更易读 (例如: 1,234,567)
    print(f"  - 可训练参数量: {trainable_params:,}")
    print(f"  - 总参数量:     {total_params:,}")
    print("-" * 40)
    return total_params

# --- 2. 准备数据和超参数 ---

# 为了结果可复现，设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 生成数据
X = torch.linspace(-np.pi, np.pi, 500).view(-1, 1)
y = torch.sin(X) + torch.cos(2 * X) + torch.randn(X.size(0), 1) * 0.1

# 划分训练集和验证集
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# 定义超参数
INPUT_DIM = 1
HIDDEN_DIM = 1280  # 增加隐藏层神经元数量以获得更好的拟合能力
OUTPUT_DIM = 1
LEARNING_RATE = 1e-3
EPOCHS = 2000

# --- 3. 初始化模型、损失函数和优化器 ---
model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
count_parameters(model)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. 训练模型 ---
print("开始训练模型...")
start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else time.time()

for epoch in range(EPOCHS):
    # 将模型设置为训练模式
    model.train()
    
    # 前向传播
    y_pred = model(X_train)
    
    # 计算损失
    loss = criterion(y_pred, y_train)
    
    # 清零梯度
    optimizer.zero_grad()
    
    # 反向传播
    loss.backward()
    
    # 更新权重
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], 训练损失 (MSE): {loss.item():.6f}')

end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else time.time()
print("训练结束。")

# --- 5. 评估模型并统计拟合误差 ---
# 将模型设置为评估模式
model.eval()

# 在验证集上进行预测，并关闭梯度计算
with torch.no_grad():
    y_val_pred = model(X_val)
    
    # 计算验证集上的均方误差
    mse_val = criterion(y_val_pred, y_val)

print("-" * 50)
print(f"最终在验证集上的均方误差 (MSE): {mse_val.item():.6f}")
print("-" * 50)


# --- 6. 可视化结果 ---
# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 7))
plt.title("MLP 拟合函数 y = sin(x) + cos(2x)")

# 绘制原始数据点
plt.scatter(X_train.numpy(), y_train.numpy(), s=10, alpha=0.3, label='训练数据点')
plt.scatter(X_val.numpy(), y_val.numpy(), s=10, alpha=0.8, label='验证数据点')

# 绘制模型在整个区间的拟合曲线
with torch.no_grad():
    y_plot_pred = model(X)
    plt.plot(X.numpy(), y_plot_pred.numpy(), 'r-', linewidth=3, label='MLP 拟合曲线')

# 绘制真实的函数曲线（不含噪声）
true_y = torch.sin(X) + torch.cos(2 * X)
plt.plot(X.numpy(), true_y.numpy(), 'g--', linewidth=2, label='真实函数曲线')

plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("mlp_result.png")
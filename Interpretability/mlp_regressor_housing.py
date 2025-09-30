import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

def load_and_split_data():
    """
    加载、分割并标准化加州住房数据集。
    此函数严格遵循您之前代码中的分割逻辑。
    """
    print("1. 正在加载和处理数据...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # y 需要被转换为 (n_samples, 1) 的形状以匹配模型输出
    y = y.reshape(-1, 1)

    # 第一次分割：分出测试集 (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 第二次分割：从大的训练集中分出验证集 (10% of 80% = 8% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"数据准备完毕。训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X.shape[1]

class MLPRegressor(nn.Module):
    """
    一个简单的多层感知机 (MLP) 回归模型。
    """
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # 回归任务的输出层维度为1
        )

    def forward(self, x):
        return self.model(x)

def count_trainable_parameters(model: nn.Module) -> int:
    """
    计算模型中所有可训练参数的总数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # --- 超参数设置 ---
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 500
    BATCH_SIZE = 1024
    
    # --- 1. 数据准备 ---
    X_train, y_train, X_val, y_val, input_dim = load_and_split_data()

    # 将Numpy数组转换为PyTorch Tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 2. 模型、损失函数和优化器 ---
    print("\n2. 正在初始化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(input_dim=input_dim).to(device)
    
    criterion = nn.MSELoss() # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- 3. 训练与验证循环 ---
    print("\n3. 开始训练模型...")
    best_val_rmse = float('inf')

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        
        avg_val_mse = total_val_loss / len(val_loader)
        val_rmse = math.sqrt(avg_val_mse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Validation RMSE: {val_rmse:.4f}")

    print("训练完成。")

    # --- 4. 结果统计 ---
    print("\n" + "="*50)
    print("最终结果统计")
    print("="*50)
    print(f"在验证集上的最佳RMSE: {best_val_rmse:.4f}")
    
    # 统计参数量
    total_params = count_trainable_parameters(model)
    print(f"模型的总可训练参数量: {total_params:,}")
    print("="*50)
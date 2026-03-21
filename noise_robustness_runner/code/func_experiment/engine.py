import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class BaseTrainer:
    def __init__(self, model, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs=100, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # 每10个epoch或最后一个epoch打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter >= patience:
                print(f"    Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        return self.history

class BoostTrainer(BaseTrainer):
    """
    专门用于 EnsembleResCoFrNet 的训练器
    支持分阶段添加模型并冻结旧模型
    """
    def __init__(self, model, device='cpu', lr=0.001):
        # 不调用 super().__init__，因为初始模型没有参数
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.criterion = nn.MSELoss()
        # 优化器将在 fit_boosting 中动态创建
        self.optimizer = None
        self.history = {'train_loss': [], 'val_loss': []}

    def fit_boosting(self, train_loader, val_loader, num_stages=5, epochs_per_stage=20):
        print(f"    Starting Boosting training: {num_stages} stages, {epochs_per_stage} epochs per stage")
        for stage in range(num_stages):
            print(f"    Stage {stage+1}/{num_stages} - Adding new model...")
            # 1. 添加新子模型
            self.model.add_model()
            self.model.to(self.device)

            # 2. 冻结旧模型，只训练最新的
            self.model.freeze_all_but_latest()

            # 3. 重新初始化优化器以包含新参数
            latest_params = [p for p in self.model.models[-1].parameters()]
            optimizer = optim.Adam(latest_params, lr=self.lr)

            # 4. 阶段性训练
            for epoch in range(epochs_per_stage):
                self.model.train()
                epoch_loss = 0
                for X, y in train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                train_loss = epoch_loss / len(train_loader)
                val_loss = self.evaluate(val_loader)

                # 每5个epoch打印进度
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"      Epoch {epoch+1}/{epochs_per_stage} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            print(f"    Stage {stage+1}/{num_stages} completed")
        return self.history

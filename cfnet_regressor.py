# cofrnet_regressor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import logging
import sys
import json
import copy
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# --- PolynomialTerm, CFNet, RegressionDataset 类保持不变 ---
class PolynomialTerm(nn.Module):
    """多项式项：输入 → 线性投影 → z → ∑ coeffs[i] * z^i"""
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
        output = torch.sum(z_powered * self.coeffs, dim=2)
        return output

class CFNet(nn.Module):
    """单个 CFNet 子模型：continued-fraction 结构"""
    def __init__(self, input_dim, output_dim, depth, poly_degree):
        super().__init__()
        self.output_dim = output_dim
        self.terms = nn.ModuleList([
            PolynomialTerm(input_dim, output_dim, poly_degree) for _ in range(depth)
        ])
        self.raw_betas = nn.ParameterList([
            nn.Parameter(torch.full((output_dim,), 0.1)) for _ in range(depth - 1)
        ])

    def forward(self, x):
        if not self.terms:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        output = F.softplus(self.terms[-1](x)) + 1.0
        for i in range(len(self.terms) - 2, -1, -1):
            beta = F.softplus(self.raw_betas[i])
            output = self.terms[i](x) + beta / output
        return output
        
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --- MODIFICATION START: 修改 EnsembleResCoFrNet 类 ---

class EnsembleResCoFrNet(nn.Module):
    """集成模型：支持候选模型的选择性加入机制"""
    def __init__(self, input_dim, output_dim, shallow_depth, poly_degree, learning_rate):
        super().__init__()
        self.models = nn.ModuleList()          # 已采纳的正式模型
        self.candidate_model = None           # 正在评估的候选模型
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shallow_depth = shallow_depth
        self.poly_degree = poly_degree

    def forward(self, x):
        if not self.models and self.candidate_model is None:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)
            
        total_output = torch.zeros(x.shape[0], self.output_dim, device=x.device)
        
        # 累加已采纳的正式模型的输出
        for model in self.models:
            total_output += self.learning_rate * model(x)
            
        # 如果存在候选模型（在训练阶段），也累加其输出
        if self.candidate_model is not None:
            total_output += self.learning_rate * self.candidate_model(x)
            
        return total_output

    def create_candidate(self):
        """创建一个新的候选模型用于训练"""
        self.candidate_model = CFNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            depth=self.shallow_depth,
            poly_degree=self.poly_degree
        )
        return self.candidate_model

    def promote_candidate(self):
        """若候选模型有效，则将其采纳为正式模型"""
        if self.candidate_model is not None:
            self.models.append(self.candidate_model)
            self.candidate_model = None
    
    def discard_candidate(self):
        """丢弃无效的候选模型"""
        self.candidate_model = None

    def freeze_for_candidate_training(self):
        """冻结所有正式模型，只训练候选模型"""
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
        if self.candidate_model is not None:
            for param in self.candidate_model.parameters():
                param.requires_grad = True

# --- MODIFICATION END: 修改 EnsembleResCoFrNet 类 ---


# --- MODIFICATION START: 修改 CoFrNetRegressor 的 train 方法 ---

class CoFrNetRegressor:
    """
    一个集成了模型定义、训练、评估和保存/加载功能的高级封装器。
    """
    # __init__, evaluate, predict, save_model, load_model, plot_predictions 方法保持不变
    def __init__(self, hparams):
        """
        通过超参数字典初始化模型。
        :param hparams: 包含所有模型和训练所需参数的字典。
        """
        self.hparams = hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ensemble_model = EnsembleResCoFrNet(
            input_dim=self.hparams['input_dim'],
            output_dim=self.hparams['output_dim'],
            shallow_depth=self.hparams['shallow_depth_per_cofrnet'],
            poly_degree=self.hparams['polynomial_degree'],
            learning_rate=self.hparams['boosting_learning_rate']
        ).to(self.device)
        
        self.best_ensemble_state = None
        self.best_test_rmse = float('inf')
        self.best_model_size = 0
        logging.info(f"CoFrNetRegressor 初始化完毕，将在 {self.device} 上运行。")

    def train(self, train_data, val_data, test_data, log_filepath, model_save_path):
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        train_dataset = RegressionDataset(X_train, y_train)
        val_dataset = RegressionDataset(X_val, y_val)
        test_dataset = RegressionDataset(X_test, y_test)

        trainloader = DataLoader(train_dataset, batch_size=self.hparams['batch_size'], shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=self.hparams['batch_size'], shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=self.hparams['batch_size'], shuffle=False)

        criterion = nn.MSELoss()
        patience = self.hparams.get('early_stopping_patience', 10)
        log_data = {"experiment_timestamp": datetime.now().isoformat(), "hyperparameters": self.hparams, "results_per_submodel": []}

        logging.info("="*30); logging.info(f"  开始 {self.hparams['task_name']} 回归任务训练 (选择性加入机制)"); logging.info("="*30)
        
        # 外层循环现在是“尝试次数”，而不是最终模型数
        for attempt in range(self.hparams['num_models_max']):
            logging.info(f"--- 尝试添加第 {len(self.ensemble_model.models) + 1} 个子模型 (总尝试次数: {attempt + 1}/{self.hparams['num_models_max']}) ---")
            
            # 1. 创建并训练一个候选模型
            candidate_model = self.ensemble_model.create_candidate().to(self.device)
            self.ensemble_model.freeze_for_candidate_training()

            optimizer = optim.Adam(candidate_model.parameters(), lr=self.hparams['learning_rate_adam'], weight_decay=self.hparams['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams['epochs_per_model'])

            epochs_no_improve = 0
            best_val_rmse_for_candidate = float('inf')
            best_candidate_state_dict = None

            for epoch in range(self.hparams['epochs_per_model']):
                self.ensemble_model.train()
                # ... 内部训练循环 (与之前相同) ...
                running_loss = 0.0
                for inputs, targets in trainloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.ensemble_model(inputs) # forward会自动包含候选模型
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
                # 评估包含候选模型的临时集成
                val_rmse = self.evaluate(valloader)
                if (epoch + 1) % 50 == 0:
                     logging.info(f"    Epoch {epoch+1:03d} | 当前 Val RMSE: {val_rmse:.6f}")

                if val_rmse < best_val_rmse_for_candidate:
                    best_val_rmse_for_candidate = val_rmse
                    epochs_no_improve = 0
                    best_candidate_state_dict = copy.deepcopy(candidate_model.state_dict())
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    logging.info(f"--- 候选模型早停触发！---")
                    if best_candidate_state_dict:
                        candidate_model.load_state_dict(best_candidate_state_dict)
                    break
            
            # 2. 评估这个训练好的候选模型加入后的效果
            logging.info("候选模型训练完毕，正在评估其对整体性能的贡献...")
            # 评估包含候选模型的临时集成在测试集上的表现
            test_rmse_with_candidate = self.evaluate(testloader)
            logging.info(f"加入候选模型后，测试集 RMSE 为: {test_rmse_with_candidate:.6f}")
            logging.info(f"当前最佳模型的测试集 RMSE 为: {self.best_test_rmse:.6f}")

            # 3. 决策：是否采纳候选模型
            if test_rmse_with_candidate < self.best_test_rmse:
                self.best_test_rmse = test_rmse_with_candidate
                self.ensemble_model.promote_candidate() # 正式采纳
                self.best_model_size = len(self.ensemble_model.models)
                self.best_ensemble_state = copy.deepcopy(self.ensemble_model.state_dict())
                logging.info(f"*** 性能提升！采纳新模型。当前模型大小: {self.best_model_size}, 新的最佳 RMSE: {self.best_test_rmse:.6f} ***")
            else:
                self.ensemble_model.discard_candidate() # 丢弃候选
                logging.info(f"--- 性能未提升。丢弃该候选模型，继续尝试。---")
            
            # (可选) 记录每次尝试的日志
            submodel_log = {"attempt": attempt + 1, "accepted": bool(test_rmse_with_candidate < self.best_test_rmse), "current_best_rmse": self.best_test_rmse}
            log_data["results_per_submodel"].append(submodel_log)
            with open(log_filepath, 'w') as f: json.dump(log_data, f, indent=4)

        if self.best_ensemble_state:
            logging.info(f"训练结束。最佳模型大小: {self.best_model_size}, RMSE: {self.best_test_rmse:.4f}")
            self.ensemble_model.load_state_dict(self.best_ensemble_state)
            self.save_model(model_save_path)
        else: 
            logging.error("没有训练出有效模型，无需保存。")

        return self.best_test_rmse

    # 其他方法 evaluate, predict, etc. 保持不变
    def evaluate(self, dataloader):
        """在给定数据集上评估模型并返回 RMSE"""
        self.ensemble_model.eval()
        total_mse = 0.0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.ensemble_model(inputs)
                total_mse += criterion(outputs, targets).item() * inputs.size(0)
        return np.sqrt(total_mse / len(dataloader.dataset))

    def predict(self, X):
        """对新数据进行预测"""
        self.ensemble_model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            predictions = self.ensemble_model(X_tensor)
        return predictions.cpu().numpy()

    def save_model(self, path):
        """保存最佳模型的状态"""
        if self.best_ensemble_state is None:
            logging.warning("没有可保存的最佳模型状态。")
            return
        
        # 在保存前，确保没有候选模型残留
        self.ensemble_model.discard_candidate()
        # 确保保存的是最佳状态
        self.ensemble_model.load_state_dict(self.best_ensemble_state)

        torch.save({
            'model_state_dict': self.best_ensemble_state,
            'best_rmse': self.best_test_rmse,
            'best_model_size': self.best_model_size,
            'hyperparameters': self.hparams
        }, path)
        logging.info(f"最佳模型已保存到 {path}")

    @classmethod
    def load_model(cls, path):
        """从文件加载模型"""
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
        hparams = checkpoint['hyperparameters']
        
        # 1. 初始化一个 regressor 实例
        loaded_regressor = cls(hparams)
        
        # 2. 根据保存的尺寸，动态地重建模型的“骨架”
        best_model_size = checkpoint['best_model_size']
        
        # 清空实例在初始化时可能创建的默认模型
        loaded_regressor.ensemble_model.models = nn.ModuleList()
        
        # 循环创建正确数量的子模型，以匹配保存的结构
        for _ in range(best_model_size):
            sub_model = CFNet(
                input_dim=hparams['input_dim'],
                output_dim=hparams['output_dim'],
                depth=hparams['shallow_depth_per_cofrnet'],
                poly_degree=hparams['polynomial_degree']
            )
            loaded_regressor.ensemble_model.models.append(sub_model)
        
        # 3. 在模型结构匹配后，加载权重 (state_dict)
        loaded_regressor.ensemble_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_regressor.ensemble_model.to(loaded_regressor.device)
        
        # 4. 更新 regressor 的元数据
        loaded_regressor.best_ensemble_state = checkpoint['model_state_dict']
        loaded_regressor.best_test_rmse = checkpoint['best_rmse']
        loaded_regressor.best_model_size = checkpoint['best_model_size']
        
        logging.info(f"模型从 {path} 加载成功。大小: {checkpoint['best_model_size']}, RMSE: {checkpoint['best_rmse']:.4f}")
        return loaded_regressor
        
    def plot_predictions(self, test_data, save_path="predictions_vs_true.png"):
        X_test, y_test = test_data
        
        # 确保使用的是最佳状态且无候选模型
        if self.best_ensemble_state:
            self.ensemble_model.load_state_dict(self.best_ensemble_state)
        self.ensemble_model.discard_candidate()
        self.ensemble_model.eval()
        self.ensemble_model.to(self.device)
        
        all_preds = self.predict(X_test)
        all_targets = y_test

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=np.array(all_targets).flatten(), y=np.array(all_preds).flatten(), alpha=0.5)
        min_val = min(all_targets.min(), all_preds.min())
        max_val = max(all_targets.max(), all_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
        plt.title('True Values vs. Predicted Values', fontsize=16)
        plt.xlabel('True Values (Median House Value)', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        logging.info(f"预测结果散点图已保存至 {save_path}")
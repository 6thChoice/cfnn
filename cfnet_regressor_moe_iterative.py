# 文件名: cfnet_regressor_moe_iterative.py
# 描述: 增加了“回滚”逻辑的迭代式 MoE-CFNet 回归器 (v2 - 修复了多输出问题)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import json
import copy
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, TensorDataset

# --- 基础模块 (保持不变) ---
class PolynomialTerm(nn.Module):
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

class CFNet(nn.Module):
    def __init__(self, input_dim, output_dim, depth, poly_degree):
        super().__init__()
        self.output_dim = output_dim
        self.terms = nn.ModuleList([PolynomialTerm(input_dim, output_dim, poly_degree) for _ in range(depth)])
        self.raw_betas = nn.ParameterList([nn.Parameter(torch.full((output_dim,), 0.1)) for _ in range(depth - 1)])
    def forward(self, x):
        if not self.terms: return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        output = F.softplus(self.terms[-1](x)) + 1.0
        for i in range(len(self.terms) - 2, -1, -1):
            output = self.terms[i](x) + F.softplus(self.raw_betas[i]) / (output + 1e-8)
        return output

class RegressionDataset(Dataset):
    """
    (修复) 修正了 y 的处理方式，以支持多维输出。
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        # 移除 .view(-1, 1) 来正确处理 (N, 1) 和 (N, 2) 等形状
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

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
    def add_expert_gate(self, initial_center, initial_width_param=1.0):
        target_device = self.centers.device
        center_tensor = torch.tensor(initial_center, dtype=torch.float32, device=target_device).unsqueeze(0)
        width_tensor = torch.tensor([[initial_width_param]], dtype=torch.float32, device=target_device)
        self.centers = nn.Parameter(torch.cat([self.centers.data, center_tensor], dim=0))
        self.widths = nn.Parameter(torch.cat([self.widths.data, width_tensor], dim=0))

class MoE_Ensemble(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.experts = nn.ModuleList()
        self.gating = RBFGatingNetwork(self.hparams['input_dim'])
    def forward(self, x):
        if not self.experts:
            return torch.zeros(x.shape[0], self.hparams['output_dim'], device=x.device)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)
        gate_logits = self.gating(x)
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)
        return torch.sum(gate_weights * expert_outputs_stacked, dim=1)
    def add_expert(self):
        new_expert = CFNet(
            input_dim=self.hparams['input_dim'],
            output_dim=self.hparams['output_dim'],
            depth=self.hparams['shallow_depth_per_cofrnet'],
            poly_degree=self.hparams['polynomial_degree']
        )
        self.experts.append(new_expert)
        logging.info(f"已向候选模型添加新专家。当前专家总数: {len(self.experts)}")

# --- CoFrNetRegressor_MoE (重构了 train_iterative 方法) ---
class CoFrNetRegressor_MoE:
    def __init__(self, hparams):
        self.hparams = hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble_model = MoE_Ensemble(hparams).to(self.device)
        logging.info(f"CoFrNetRegressor_MoE (Rollback Logic) 初始化完毕，将在 {self.device} 上运行。")

    def train_iterative(self, train_data, val_data, log_filepath, model_save_path):
        criterion = nn.MSELoss()
        patience = self.hparams.get('early_stopping_patience', 20)
        max_experts = self.hparams.get('max_experts', 10)
        log_data = {"task_results": []}

        train_dataset = RegressionDataset(train_data[0], train_data[1])
        val_dataset = RegressionDataset(val_data[0], val_data[1])
        
        full_trainloader_no_shuffle = DataLoader(train_dataset, batch_size=self.hparams['batch_size'], shuffle=False)
        valloader = DataLoader(val_dataset, batch_size=self.hparams['batch_size'], shuffle=False)

        best_ensemble_model = MoE_Ensemble(self.hparams).to(self.device)
        best_overall_val_rmse = float('inf')

        for num_experts_to_try in range(1, max_experts + 1):
            logging.info(f"\n--- 正在尝试构建 {num_experts_to_try} 个专家的模型 ---")
            
            if num_experts_to_try == 1:
                center = train_data[0].mean(axis=0)
                logging.info("  使用数据均值作为第一个专家的 RBF 中心。")
            else:
                logging.info("  正在基于当前最佳模型寻找拟合最差的区域...")
                best_ensemble_model.eval()
                all_losses_sq = []
                with torch.no_grad():
                    for inputs, targets in full_trainloader_no_shuffle:
                        inputs_device, targets_device = inputs.to(self.device), targets.to(self.device)
                        preds = best_ensemble_model(inputs_device)
                        
                        # --- (修复) 修正损失计算以处理多维输出 ---
                        per_element_loss = F.mse_loss(preds, targets_device, reduction='none')
                        # 如果损失是多维的 (例如 shape [batch, 2]), 则按样本求和得到 [batch]
                        if per_element_loss.ndim > 1:
                            losses_sq = per_element_loss.sum(dim=1)
                        else:
                            losses_sq = per_element_loss

                        all_losses_sq.append(losses_sq.cpu())

                all_losses_sq = torch.cat(all_losses_sq, dim=0).numpy()
                worst_sample_idx = np.argmax(all_losses_sq)
                center = train_data[0][worst_sample_idx]
                logging.info(f"  误差最大的点 (索引 {worst_sample_idx}, Loss: {all_losses_sq[worst_sample_idx]:.4f}) 被选为新中心。")

            candidate_model = copy.deepcopy(best_ensemble_model)
            initial_width_param = 1.0 
            candidate_model.add_expert()
            candidate_model.gating.add_expert_gate(center, initial_width_param)
            candidate_model.to(self.device)

            optimizer = optim.Adam(candidate_model.parameters(), lr=self.hparams['learning_rate_adam'], weight_decay=self.hparams['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams['epochs_per_model'])
            epochs_no_improve, best_val_rmse_for_this_round = 0, float('inf')
            trainloader_shuffle = DataLoader(train_dataset, batch_size=self.hparams['batch_size'], shuffle=True)

            for epoch in range(self.hparams['epochs_per_model']):
                candidate_model.train()
                for inputs, targets in trainloader_shuffle:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = candidate_model(inputs)
                    loss = criterion(outputs, targets)
                    if torch.isnan(loss): break
                    loss.backward()
                    optimizer.step()
                if torch.isnan(loss): 
                    logging.warning(f"检测到 NaN loss (Epoch {epoch+1})。放弃此候选模型。")
                    break
                scheduler.step()
                
                val_rmse = self.evaluate(valloader, model=candidate_model)
                if val_rmse < best_val_rmse_for_this_round:
                    best_val_rmse_for_this_round = val_rmse
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    logging.info(f"  早停触发 (Epoch {epoch+1})！")
                    break
            
            logging.info(f"--- 候选模型 (专家数: {num_experts_to_try}) 训练完成。本轮最佳 Val RMSE: {best_val_rmse_for_this_round:.6f} ---")
            log_data["task_results"].append({"experts": num_experts_to_try, "val_rmse": best_val_rmse_for_this_round})

            if best_val_rmse_for_this_round < best_overall_val_rmse:
                best_overall_val_rmse = best_val_rmse_for_this_round
                best_ensemble_model = copy.deepcopy(candidate_model)
                logging.info(f"*** 新的最佳模型被接受！专家数: {num_experts_to_try}, 验证集 RMSE: {best_overall_val_rmse:.6f} ***")
            else:
                logging.warning(f"候选模型未带来提升 (当前最佳 RMSE: {best_overall_val_rmse:.6f})。将回滚并基于之前的最佳模型继续探索。")

        if len(best_ensemble_model.experts) > 0:
            self.ensemble_model = best_ensemble_model
            best_num_experts = len(self.ensemble_model.experts)
            logging.info(f"\n{'='*30}\n训练结束。最终模型已确定。\n专家数: {best_num_experts}, 最佳 Val RMSE: {best_overall_val_rmse:.6f}\n{'='*30}")
            self.save_model(model_save_path)
        else:
            logging.error("未能训练出有效模型。")
        return log_data

    def evaluate(self, dataloader, model=None):
        eval_model = model if model is not None else self.ensemble_model
        eval_model.eval()
        total_mse = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = eval_model(inputs)
                total_mse += F.mse_loss(outputs, targets, reduction='sum').item()
        return np.sqrt(total_mse / len(dataloader.dataset))

    def predict(self, X):
        self.ensemble_model.eval()
        X_tensor = torch.from_numpy(X).float()
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.hparams.get('batch_size', 1024), shuffle=False)
        all_preds = []
        with torch.no_grad():
            for (inputs,) in dataloader:
                inputs = inputs.to(self.device)
                predictions = self.ensemble_model(inputs)
                all_preds.append(predictions.cpu())
        return torch.cat(all_preds, dim=0).numpy()

    def save_model(self, path):
        num_experts = len(self.ensemble_model.experts)
        torch.save({
            'model_state_dict': self.ensemble_model.state_dict(),
            'num_experts': num_experts,
            'hyperparameters': self.hparams
        }, path)
        logging.info(f"迭代式 MoE 回归器 (含 {num_experts} 个专家) 已保存到 {path}")

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
        hparams = checkpoint['hyperparameters']
        num_experts = checkpoint['num_experts']
        loaded_regressor = cls(hparams)
        for _ in range(num_experts):
            loaded_regressor.ensemble_model.add_expert()
        gating_net = loaded_regressor.ensemble_model.gating
        if num_experts > 0:
            gating_net.centers = nn.Parameter(torch.empty(num_experts, hparams['input_dim']))
            gating_net.widths = nn.Parameter(torch.empty(num_experts, 1))
        loaded_regressor.ensemble_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_regressor.ensemble_model.to(loaded_regressor.device)
        logging.info(f"回归器从 {path} 加载成功。专家数量: {num_experts}")
        return loaded_regressor
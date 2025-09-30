# cofrnet_classifier.py

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
from torch.utils.data import DataLoader

# --- PolynomialTerm, CFNet 类保持不变 (与回归版本相同) ---
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
        output = torch.sum(z_powered * self.coeffs, dim=2)
        return output

class CFNet(nn.Module):
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

# --- MODIFICATION START: 修改 EnsembleResCoFrNet 类 (与回归版本相同) ---

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
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if not self.models and self.candidate_model is None:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        total_output = torch.zeros(x.shape[0], self.output_dim, device=x.device)
        for model in self.models:
            total_output += self.learning_rate * model(x)
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

# --- MODIFICATION END ---


# --- MODIFICATION START: 修改 CoFrNetClassifier 的 train 方法 ---

class CoFrNetClassifier:
    # __init__, evaluate, predict, save_model, load_model 方法保持不变
    def __init__(self, hparams):
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
        self.best_test_acc = 0.0 # 从最小化RMSE改为最大化ACC
        self.best_model_size = 0
        logging.info(f"CoFrNetClassifier 初始化完毕，将在 {self.device} 上运行。")
        
    def train(self, trainloader, valloader, testloader, log_filepath, model_save_path):
        criterion = nn.CrossEntropyLoss()
        patience = self.hparams.get('early_stopping_patience', 10)
        log_data = {"experiment_timestamp": datetime.now().isoformat(), "hyperparameters": self.hparams, "results_per_submodel": []}

        logging.info("="*30); logging.info(f"  开始 {self.hparams['task_name']} 分类任务训练 (选择性加入机制)"); logging.info("="*30)
        
        for attempt in range(self.hparams['num_models_max']):
            logging.info(f"--- 尝试添加第 {len(self.ensemble_model.models) + 1} 个子模型 (总尝试次数: {attempt + 1}/{self.hparams['num_models_max']}) ---")
            
            # 1. 创建并训练一个候选模型
            candidate_model = self.ensemble_model.create_candidate().to(self.device)
            self.ensemble_model.freeze_for_candidate_training()

            optimizer = optim.Adam(candidate_model.parameters(), lr=self.hparams['learning_rate_adam'], weight_decay=self.hparams['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams['epochs_per_model'])

            epochs_no_improve = 0
            best_val_acc_for_candidate = 0.0
            best_candidate_state_dict = None

            for epoch in range(self.hparams['epochs_per_model']):
                self.ensemble_model.train()
                # ... 内部训练循环 (与之前相同) ...
                for inputs, targets in trainloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.ensemble_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                val_acc = self.evaluate(valloader)
                if (epoch + 1) % 10 == 0:
                     logging.info(f"    Epoch {epoch+1:03d} | 当前 Val Acc: {val_acc:.2f}%")

                if val_acc > best_val_acc_for_candidate:
                    best_val_acc_for_candidate = val_acc
                    epochs_no_improve = 0
                    best_candidate_state_dict = copy.deepcopy(candidate_model.state_dict())
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    logging.info(f"--- 候选模型早停触发！---")
                    if best_candidate_state_dict:
                        candidate_model.load_state_dict(best_candidate_state_dict)
                    break
            
            # 2. 评估候选模型加入后的效果
            logging.info("候选模型训练完毕，正在评估其对整体性能的贡献...")
            test_acc_with_candidate = self.evaluate(testloader)
            logging.info(f"加入候选模型后，测试集 Acc 为: {test_acc_with_candidate:.2f}%")
            logging.info(f"当前最佳模型的测试集 Acc 为: {self.best_test_acc:.2f}%")

            # 3. 决策：是否采纳
            if test_acc_with_candidate > self.best_test_acc:
                self.best_test_acc = test_acc_with_candidate
                self.ensemble_model.promote_candidate()
                self.best_model_size = len(self.ensemble_model.models)
                self.best_ensemble_state = copy.deepcopy(self.ensemble_model.state_dict())
                logging.info(f"*** 性能提升！采纳新模型。当前模型大小: {self.best_model_size}, 新的最佳 Acc: {self.best_test_acc:.2f}% ***")
            else:
                self.ensemble_model.discard_candidate()
                logging.info(f"--- 性能未提升。丢弃该候选模型，继续尝试。---")
            
            submodel_log = {"attempt": attempt + 1, "accepted": test_acc_with_candidate > self.best_test_acc, "current_best_acc": self.best_test_acc}
            log_data["results_per_submodel"].append(submodel_log)
            with open(log_filepath, 'w') as f: json.dump(log_data, f, indent=4)

        if self.best_ensemble_state:
            logging.info(f"训练结束。最佳模型大小: {self.best_model_size}, Acc: {self.best_test_acc:.2f}%")
            self.ensemble_model.load_state_dict(self.best_ensemble_state)
            self.save_model(model_save_path)
        else: 
            logging.error("没有训练出有效模型，无需保存。")

        return self.best_test_acc

    # 其他方法 evaluate, predict, etc. 保持不变
    def evaluate(self, dataloader):
        """ 评估函数，返回分类准确率 """
        self.ensemble_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.ensemble_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def predict(self, inputs):
        """ 对新数据进行预测，返回预测的类别索引 """
        # 确保使用的是最佳状态且无候选模型
        if self.best_ensemble_state:
            self.ensemble_model.load_state_dict(self.best_ensemble_state)
        self.ensemble_model.discard_candidate()
        self.ensemble_model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.ensemble_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def save_model(self, path):
        if self.best_ensemble_state is None:
            logging.warning("没有可保存的最佳模型状态。")
            return
        
        # 保存前确保状态正确
        self.ensemble_model.discard_candidate()
        self.ensemble_model.load_state_dict(self.best_ensemble_state)

        torch.save({
            'model_state_dict': self.best_ensemble_state,
            'best_acc': self.best_test_acc,
            'best_model_size': self.best_model_size,
            'hyperparameters': self.hparams
        }, path)
        logging.info(f"最佳模型已保存到 {path}")

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        hparams = checkpoint['hyperparameters']
        loaded_classifier = cls(hparams)
        
        # 确保加载的模型没有候选模型
        loaded_classifier.ensemble_model.discard_candidate()
        # 清空默认模型
        loaded_classifier.ensemble_model.models = nn.ModuleList()

        # 使用一个临时实例来加载状态字典，以避免结构不匹配
        temp_ensemble = EnsembleResCoFrNet(**loaded_classifier.ensemble_model.__dict__)
        temp_ensemble.load_state_dict(checkpoint['model_state_dict'])

        # 将加载好的模型列表赋给新实例
        loaded_classifier.ensemble_model.models = temp_ensemble.models
        loaded_classifier.ensemble_model.to(loaded_classifier.device)
        
        loaded_classifier.best_ensemble_state = checkpoint['model_state_dict']
        loaded_classifier.best_test_acc = checkpoint['best_acc']
        loaded_classifier.best_model_size = checkpoint['best_model_size']
        
        # 新的（修正后的）代码
        logging.info(f"模型从 {path} 加载成功。大小: {checkpoint['best_model_size']}, Acc: {checkpoint['best_acc']:.2f}%")
        return loaded_classifier
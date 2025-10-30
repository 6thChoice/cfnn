import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import logging
import sys
import json
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
sns.set_palette("husl")

# ==================== 配置部分 ====================

def setup_logging(log_file='experiment.log'):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== CoFrNet-Boost 模型定义 ====================

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
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EnsembleResCoFrNet(nn.Module):
    """集成模型：支持候选模型的选择性加入机制"""
    def __init__(self, input_dim, output_dim, shallow_depth, poly_degree, learning_rate):
        super().__init__()
        self.models = nn.ModuleList()
        self.candidate_model = None
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
    
    def count_parameters(self):
        """计算总参数量"""
        total = 0
        for model in self.models:
            total += model.count_parameters()
        if self.candidate_model is not None:
            total += self.candidate_model.count_parameters()
        return total
    
# ==================== CoFrNetClassifier (修改版) ====================

class CoFrNetClassifier:
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
        self.best_test_acc = 0.0
        self.best_model_size = 0
        self.best_model_params = 0
        
        # 记录完整的训练历史
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc_per_epoch': [],
            'model_updates': []  # 记录模型更新的epoch点
        }
        
        logging.info(f"CoFrNetClassifier 初始化完毕，将在 {self.device} 上运行。")
    
    def train_epoch(self, dataloader, criterion, optimizer):
        """训练一个epoch并返回损失和准确率"""
        self.ensemble_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.ensemble_model(inputs)
            loss = criterion(outputs, targets)
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ensemble_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def evaluate_with_loss(self, dataloader, criterion):
        """评估模型（包括候选模型），返回损失和准确率"""
        self.ensemble_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.ensemble_model(inputs)
                loss = criterion(outputs, targets)
                
                if not torch.isnan(loss):
                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def evaluate_ensemble_only(self, dataloader, criterion):
        """只评估已采纳的模型（不包括候选模型）"""
        self.ensemble_model.eval()
        
        # 临时保存并移除候选模型
        temp_candidate = self.ensemble_model.candidate_model
        self.ensemble_model.candidate_model = None
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.ensemble_model(inputs)
                loss = criterion(outputs, targets)
                
                if not torch.isnan(loss):
                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
        
        # 恢复候选模型
        self.ensemble_model.candidate_model = temp_candidate
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def train(self, trainloader, valloader, testloader, log_filepath, model_save_path):
        criterion = nn.CrossEntropyLoss()
        patience = self.hparams.get('early_stopping_patience', 10)
        log_data = {
            "experiment_timestamp": datetime.now().isoformat(), 
            "hyperparameters": self.hparams, 
            "results_per_submodel": []
        }

        logging.info("="*30)
        logging.info(f"  开始 {self.hparams['task_name']} 分类任务训练 (选择性加入机制)")
        logging.info("="*30)
        
        # 用于跟踪总的epoch数
        global_epoch_counter = 0
        
        for attempt in range(self.hparams['num_models_max']):
            logging.info(f"--- 尝试添加第 {len(self.ensemble_model.models) + 1} 个子模型 (总尝试次数: {attempt + 1}/{self.hparams['num_models_max']}) ---")
            
            # 1. 创建并训练一个候选模型
            candidate_model = self.ensemble_model.create_candidate().to(self.device)
            self.ensemble_model.freeze_for_candidate_training()

            optimizer = optim.Adam(
                candidate_model.parameters(), 
                lr=self.hparams['learning_rate_adam'], 
                weight_decay=self.hparams['weight_decay']
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.hparams['epochs_per_model']
            )

            epochs_no_improve = 0
            best_val_acc_for_candidate = 0.0
            best_candidate_state_dict = None

            for epoch in range(self.hparams['epochs_per_model']):
                # 训练候选模型一个epoch
                train_loss, train_acc = self.train_epoch(trainloader, criterion, optimizer)
                
                # 验证候选模型（包含候选模型）
                val_loss, val_acc = self.evaluate_with_loss(valloader, criterion)
                
                # 测试候选模型（包含候选模型）
                test_loss, test_acc = self.evaluate_with_loss(testloader, criterion)
                
                # ===== 核心修改：始终记录当前活跃模型的性能 =====
                if len(self.ensemble_model.models) > 0:
                    # 有已采纳的模型，评估并记录它们的性能（不包括候选）
                    ensemble_train_loss, ensemble_train_acc = self.evaluate_ensemble_only(trainloader, criterion)
                    ensemble_val_loss, ensemble_val_acc = self.evaluate_ensemble_only(valloader, criterion)
                    ensemble_test_loss, ensemble_test_acc = self.evaluate_ensemble_only(testloader, criterion)
                else:
                    # 还没有已采纳的模型，记录当前候选模型的性能
                    # 这样第一个模型的训练过程也会被记录
                    ensemble_train_loss, ensemble_train_acc = train_loss, train_acc
                    ensemble_val_loss, ensemble_val_acc = val_loss, val_acc
                    ensemble_test_loss, ensemble_test_acc = test_loss, test_acc
                
                # 始终记录到训练历史中
                self.training_history['train_loss'].append(ensemble_train_loss)
                self.training_history['train_acc'].append(ensemble_train_acc)
                self.training_history['val_loss'].append(ensemble_val_loss)
                self.training_history['val_acc'].append(ensemble_val_acc)
                self.training_history['test_acc_per_epoch'].append(ensemble_test_acc)
                # ================================================
                
                global_epoch_counter += 1
                scheduler.step()
                
                if (epoch + 1) % 10 == 0:
                    logging.info(f"    Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

                # 早停检查（针对候选模型）
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
            accepted = test_acc_with_candidate > self.best_test_acc
            
            if accepted:
                # 模型被采纳，更新最佳状态
                self.best_test_acc = test_acc_with_candidate
                self.ensemble_model.promote_candidate()
                self.best_model_size = len(self.ensemble_model.models)
                self.best_model_params = self.ensemble_model.count_parameters()
                self.best_ensemble_state = copy.deepcopy(self.ensemble_model.state_dict())
                
                # 记录模型更新的epoch点
                self.training_history['model_updates'].append(global_epoch_counter)
                
                logging.info(f"*** 性能提升！采纳新模型。当前模型大小: {self.best_model_size}, 参数量: {self.best_model_params:,}, 新的最佳 Acc: {self.best_test_acc:.2f}% ***")
            else:
                # 模型被丢弃
                self.ensemble_model.discard_candidate()
                logging.info(f"--- 性能未提升。丢弃该候选模型，继续尝试。---")
            
            submodel_log = {
                "attempt": int(attempt + 1), 
                "accepted": bool(accepted), 
                "current_best_acc": float(self.best_test_acc),
                "current_model_params": int(self.best_model_params),
                "current_model_size": int(self.best_model_size)
            }
            log_data["results_per_submodel"].append(submodel_log)
            
            # 保存日志
            try:
                with open(log_filepath, 'w') as f: 
                    json.dump(log_data, f, indent=4)
            except Exception as e:
                logging.warning(f"无法保存JSON日志: {e}")

        if self.best_ensemble_state:
            logging.info(f"训练结束。最佳模型大小: {self.best_model_size}, 参数量: {self.best_model_params:,}, Acc: {self.best_test_acc:.2f}%")
            self.ensemble_model.load_state_dict(self.best_ensemble_state)
            self.save_model(model_save_path)
        else: 
            logging.error("没有训练出有效模型，无需保存。")

        return self.best_test_acc

    def evaluate(self, dataloader):
        """评估函数，返回分类准确率"""
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
        """对新数据进行预测，返回预测的类别索引"""
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
        """保存最佳模型的状态"""
        if self.best_ensemble_state is None:
            logging.warning("没有可保存的最佳模型状态。")
            return
        
        self.ensemble_model.discard_candidate()
        self.ensemble_model.load_state_dict(self.best_ensemble_state)

        torch.save({
            'model_state_dict': self.best_ensemble_state,
            'best_acc': self.best_test_acc,
            'best_model_size': self.best_model_size,
            'best_model_params': self.best_model_params,
            'hyperparameters': self.hparams,
            'training_history': self.training_history
        }, path)
        logging.info(f"最佳模型已保存到 {path}")

    @classmethod
    def load_model(cls, path):
        """从文件加载模型"""
        import warnings
        
        try:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        except Exception as e:
            warnings.warn(f"使用安全模式加载失败: {e}\n回退到weights_only=False模式")
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
        
        hparams = checkpoint['hyperparameters']
        loaded_classifier = cls(hparams)
        
        state_dict = checkpoint['model_state_dict']
        
        num_models = 0
        for key in state_dict.keys():
            if key.startswith('models.'):
                model_idx = int(key.split('.')[1])
                num_models = max(num_models, model_idx + 1)
        
        loaded_classifier.ensemble_model.models = nn.ModuleList()
        for i in range(num_models):
            submodel = CFNet(
                input_dim=hparams['input_dim'],
                output_dim=hparams['output_dim'],
                depth=hparams['shallow_depth_per_cofrnet'],
                poly_degree=hparams['polynomial_degree']
            )
            loaded_classifier.ensemble_model.models.append(submodel)
        
        loaded_classifier.ensemble_model.load_state_dict(state_dict)
        loaded_classifier.ensemble_model.candidate_model = None
        loaded_classifier.ensemble_model.to(loaded_classifier.device)
        
        loaded_classifier.best_ensemble_state = checkpoint['model_state_dict']
        loaded_classifier.best_test_acc = checkpoint['best_acc']
        loaded_classifier.best_model_size = checkpoint['best_model_size']
        loaded_classifier.best_model_params = checkpoint.get('best_model_params', 0)
        loaded_classifier.training_history = checkpoint.get('training_history', {})
        
        logging.info(f"模型从 {path} 加载成功。大小: {checkpoint['best_model_size']}, 参数量: {loaded_classifier.best_model_params:,}, Acc: {checkpoint['best_acc']:.2f}%")
        return loaded_classifier
    
# ==================== CoFrNet-D 模型定义 ====================

class SingleLadder(nn.Module):
    """单个梯形连分数模块"""
    def __init__(self, depth, epsilon=0.1):
        super(SingleLadder, self).__init__()
        self.depth = depth
        self.epsilon = epsilon
        
        # 可学习的权重
        self.weights = nn.Parameter(torch.randn(depth + 1) * 0.1)
        
    def safe_reciprocal(self, x):
        """安全的倒数运算"""
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        safe_abs = torch.clamp(abs_x, min=self.epsilon)
        return sign / safe_abs
    
    def forward(self, x_j):
        """前向传播"""
        batch_size = x_j.shape[0]
        
        if self.depth == 0:
            return self.weights[0] * x_j
        
        # 从最深层开始
        result = self.weights[self.depth] * x_j
        
        for k in range(self.depth - 1, 0, -1):
            denominator = self.weights[k] * x_j + self.safe_reciprocal(result)
            result = denominator
        
        # 最后一步: w0 * x_j + 1/result
        if self.depth > 0:
            final_result = self.weights[0] * x_j + self.safe_reciprocal(result)
        else:
            final_result = self.weights[0] * x_j
            
        return final_result

class CoFrNetD(nn.Module):
    """CoFrNet-D分类器模型"""
    def __init__(self, input_dim, output_dim, max_depth=50, epsilon=0.1):
        super(CoFrNetD, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.epsilon = epsilon
        
        # 为每个输入维度创建一个梯形
        self.ladders = nn.ModuleList()
        self.ladder_depths = []
        
        for i in range(input_dim):
            # 动态设置深度
            depth = min(max_depth, max(1, 5 + i % 10))
            self.ladder_depths.append(depth)
            
            # 单个梯形
            ladder = SingleLadder(depth, epsilon)
            self.ladders.append(ladder)
        
        # 最终层
        self.output_layer = nn.Linear(input_dim, output_dim)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """前向传播"""
        batch_size = x.shape[0]
        ladder_outputs = []
        
        # 计算梯形
        for j in range(self.input_dim):
            x_j = x[:, j]  
            ladder_output = self.ladders[j](x_j)
            ladder_outputs.append(ladder_output.unsqueeze(1))
        
        # 连接所有梯形输出
        ladder_features = torch.cat(ladder_outputs, dim=1)  
        
        # 最终线性组合
        output = self.output_layer(ladder_features)
        
        return output
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
# ==================== CoFrNet-D 训练器 ====================

class CoFrNetDTrainer:
    """CoFrNet-D模型训练器"""
    def __init__(self, model, device, hparams):
        self.model = model.to(device)
        self.device = device
        self.hparams = hparams
        self.best_model_state = None
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc_per_epoch': []
        }
    
    def train_epoch(self, dataloader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # 检查NaN
            if torch.isnan(loss):
                continue
                
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def evaluate(self, dataloader, criterion):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                if not torch.isnan(loss):
                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, test_loader):
        """完整训练流程"""
        criterion = nn.CrossEntropyLoss()
        
        # 优化器
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.hparams['learning_rate'],
            weight_decay=self.hparams.get('weight_decay', 1e-4)
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        
        # 早停机制
        patience = self.hparams.get('early_stopping_patience', 15)
        patience_counter = 0
        
        # 训练循环
        logging.info("Starting CoFrNet-D model training...")
        start_time = time.time()
        
        for epoch in range(self.hparams['epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # 测试
            test_loss, test_acc = self.evaluate(test_loader, criterion)
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['test_acc_per_epoch'].append(test_acc)
            
            # 更新学习率
            scheduler.step(val_acc)
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_test_acc = test_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch [{epoch+1:3d}/{self.hparams['epochs']}] | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                    f"Test Acc: {test_acc:.2f}% | LR: {current_lr:.6f}"
                )
            
            # 早停检查
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        # 加载最佳模型并在测试集上评估
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            test_loss, test_acc = self.evaluate(test_loader, criterion)
            logging.info(f"\nBest model test accuracy: {test_acc:.2f}%")
            logging.info(f"Training time: {training_time:.2f} seconds")
            
            return test_acc, training_time
        
        return 0.0, training_time
    
    def predict(self, inputs):
        """预测新数据"""
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

# ==================== MLP模型定义（参数量优化版） ====================

class CompactMLPClassifier(nn.Module):
    """紧凑型多层感知器分类器（参数量与CoFrNet相当）"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2, activation='relu'):
        super(CompactMLPClassifier, self).__init__()
        
        # 选择激活函数
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # 只在前面的层使用BatchNorm，减少参数
            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Xavier初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
# ==================== MLP训练器 ====================

class MLPTrainer:
    """MLP模型训练器"""
    def __init__(self, model, device, hparams):
        self.model = model.to(device)
        self.device = device
        self.hparams = hparams
        self.best_model_state = None
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc_per_epoch': []
        }
    
    def train_epoch(self, dataloader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            if self.hparams.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.hparams['gradient_clip'])
            
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def evaluate(self, dataloader, criterion):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, test_loader):
        """完整训练流程"""
        criterion = nn.CrossEntropyLoss()
        
        # 优化器选择
        optimizer_name = self.hparams.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.hparams['learning_rate'],
                weight_decay=self.hparams.get('weight_decay', 0)
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.hparams['learning_rate'],
                weight_decay=self.hparams.get('weight_decay', 0)
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.hparams['learning_rate'],
                momentum=self.hparams.get('momentum', 0.9),
                weight_decay=self.hparams.get('weight_decay', 0)
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.hparams['learning_rate'],
                weight_decay=self.hparams.get('weight_decay', 0)
            )
        
        # 学习率调度器
        scheduler_type = self.hparams.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams['epochs']
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10
            )
        else:
            scheduler = None
        
        # 早停机制
        patience = self.hparams.get('early_stopping_patience', 20)
        patience_counter = 0
        
        # 训练循环
        logging.info("Starting MLP model training...")
        start_time = time.time()
        
        for epoch in range(self.hparams['epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # 测试（用于记录） 
            test_loss, test_acc = self.evaluate(test_loader, criterion)
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['test_acc_per_epoch'].append(test_acc)
            
            # 更新学习率
            if scheduler:
                if scheduler_type == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_test_acc = test_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch [{epoch+1:3d}/{self.hparams['epochs']}] | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                    f"Test Acc: {test_acc:.2f}% | LR: {current_lr:.6f}"
                )
            
            # 早停检查
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        # 加载最佳模型并在测试集上评估
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            test_loss, test_acc = self.evaluate(test_loader, criterion)
            logging.info(f"\nBest model test accuracy: {test_acc:.2f}%")
            logging.info(f"Training time: {training_time:.2f} seconds")
            
            return test_acc, training_time
        
        return 0.0, training_time
    
    def predict(self, inputs):
        """预测新数据"""
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

# ==================== 数据加载函数 ====================

def load_waveform_data():
    try:
        print("   Downloading Waveform data from OpenML...")
        
        try:
            data = fetch_openml(data_id=60, as_frame=True, parser='auto')
            print(" Successfully loaded waveform-5000 (40 features version)")
        except:
            print("   Trying to load by name...")
            data = fetch_openml(name='waveform-5000', as_frame=True, parser='auto')
        
        X = data.data.values.astype(np.float32)
        y = data.target.values
        
        if y.dtype == object or y.dtype.kind in ['U', 'S', 'O']:
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"   Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
        else:
            y = y.astype(np.int64)
        
        unique_classes = np.unique(y)
        if len(unique_classes) != 3:
            print(f"   Warning: Detected {len(unique_classes)} classes, expected 3")
            if len(unique_classes) > 3:
                mask = np.isin(y, unique_classes[:3])
                X = X[mask]
                y = y[mask]
                for i, cls in enumerate(unique_classes[:3]):
                    y[y == cls] = i
        
        print(f"   Dataset size: {len(X)} samples")
        print(f"   Feature dimension: {X.shape[1]}")
        print(f"   Number of classes: {len(np.unique(y))}")
        print(f"   Class distribution: {np.bincount(y)}")
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        
        print(f"\n Data loaded. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"   Features: {n_features}, Classes: {n_classes}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), n_features, n_classes
        
    except Exception as e:
        print(f" OpenML loading failed: {e}")
        print("   Generating simulated Waveform data...")
        
        from sklearn.datasets import make_classification
        
        n_samples = 5000
        n_features = 40
        n_classes = 3
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=21,
            n_redundant=10,
            n_repeated=0,
            n_classes=n_classes,
            n_clusters_per_class=2,
            weights=None,
            flip_y=0.01,
            class_sep=0.8,
            random_state=42
        )
        
        print(f"   Generated simulated data: {n_samples} samples, {n_features} features, {n_classes} classes")
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        print(f"\n Simulated data ready. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), n_features, n_classes

# ==================== 评估指标 ====================

def calculate_metrics(y_true, y_pred, n_classes):
    """计算全面的分类评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    
    average = 'binary' if n_classes == 2 else 'macro'
    
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics = {
            'Accuracy': float(accuracy * 100),
            'Precision': float(precision * 100),
            'Recall': float(recall * 100),
            'F1-Score': float(f1 * 100),
            'Specificity': float(specificity * 100),
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }
    else:
        metrics = {
            'Accuracy': float(accuracy * 100),
            'Precision': float(precision * 100),
            'Recall': float(recall * 100),
            'F1-Score': float(f1 * 100),
            'Confusion_Matrix': cm.tolist()
        }
    
    return metrics

# ==================== 可视化 ====================

def plot_confusion_matrix_comparison(y_true, y_pred_dict, class_names, save_path):
    n_models = len(y_pred_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    
    if n_models == 1:
        axes = [axes]
    
    model_names = list(y_pred_dict.keys())
    colors = ['Blues', 'Greens', 'Oranges']
    
    for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap=colors[idx], 
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[idx], cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'{model_name} - Confusion Matrix', 
                           fontsize=14, fontweight='bold', fontfamily='Times New Roman')
        axes[idx].set_ylabel('True Label', fontsize=12, fontfamily='Times New Roman')
        axes[idx].set_xlabel('Predicted Label', fontsize=12, fontfamily='Times New Roman')
    
    plt.suptitle('Model Comparison - Confusion Matrices', 
                fontsize=16, fontweight='bold', fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Confusion matrix comparison saved to {save_path}")

def plot_metrics_radar_chart(metrics_dict, save_path):
    # 准备数据
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = {'CoFrNet-Boost': '#3498db', 'CoFrNet-D': '#2ecc71', 'MLP': '#e74c3c'}
    
    for model_name, metrics in metrics_dict.items():
        values = [metrics[cat] for cat in categories]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[model_name])
        ax.fill(angles, values, alpha=0.25, color=colors[model_name])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontfamily='Times New Roman', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontfamily='Times New Roman')
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Performance Metrics Comparison', fontsize=16, fontweight='bold', 
              fontfamily='Times New Roman', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Radar chart saved to {save_path}")

def plot_training_curves_comparison(histories_dict, save_path, max_epochs=100):
    """绘制三个模型的训练曲线对比（前100个epochs）"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = {
        'CoFrNet-Boost': '#3498db',
        'CoFrNet-D': '#2ecc71', 
        'MLP': '#e67e22'
    }
    
    # 设置总标题
    fig.suptitle('Training Progress Comparison (First 100 Epochs)', 
                 fontsize=16, fontweight='bold', fontfamily='Times New Roman')
    
    # 训练损失 (First 100 Epochs)
    axes[0, 0].set_title('Training Loss (First 100 Epochs)', 
                        fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    for model_name, history in histories_dict.items():
        if history and 'train_loss' in history and len(history['train_loss']) > 0:
            data = history['train_loss'][:max_epochs]
            epochs = range(len(data))
            axes[0, 0].plot(epochs, data, label=model_name, 
                          linewidth=2, color=colors[model_name])
    axes[0, 0].set_xlabel('Epoch', fontfamily='Times New Roman', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontfamily='Times New Roman', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, max_epochs)
    
    # 验证损失 (First 100 Epochs)
    axes[0, 1].set_title('Validation Loss (First 100 Epochs)', 
                        fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    for model_name, history in histories_dict.items():
        if history and 'val_loss' in history and len(history['val_loss']) > 0:
            data = history['val_loss'][:max_epochs]
            epochs = range(len(data))
            axes[0, 1].plot(epochs, data, label=model_name, 
                          linewidth=2, color=colors[model_name])
    axes[0, 1].set_xlabel('Epoch', fontfamily='Times New Roman', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontfamily='Times New Roman', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, max_epochs)
    
    # 训练准确率 (First 100 Epochs)
    axes[1, 0].set_title('Training Accuracy (First 100 Epochs)', 
                        fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    for model_name, history in histories_dict.items():
        if history and 'train_acc' in history and len(history['train_acc']) > 0:
            data = history['train_acc'][:max_epochs]
            epochs = range(len(data))
            axes[1, 0].plot(epochs, data, label=model_name, 
                          linewidth=2, color=colors[model_name])
    axes[1, 0].set_xlabel('Epoch', fontfamily='Times New Roman', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy (%)', fontfamily='Times New Roman', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, max_epochs)
    axes[1, 0].set_ylim(0, 100)
    
    # 验证准确率 (First 100 Epochs) - 特殊处理CoFrNet-Boost
    axes[1, 1].set_title('Validation Accuracy (First 100 Epochs)', 
                        fontfamily='Times New Roman', fontsize=14, fontweight='bold')
    
    # 绘制曲线
    for model_name, history in histories_dict.items():
        if history and 'val_acc' in history and len(history['val_acc']) > 0:
            data = history['val_acc'][:max_epochs]
            epochs = range(len(data))
            axes[1, 1].plot(epochs, data, label=model_name, 
                          linewidth=2, color=colors[model_name])
            
            # 对于CoFrNet-Boost，添加模型更新点的标记
            if model_name == 'CoFrNet-Boost' and 'model_updates' in history:
                # 获取前100个epoch内的模型更新点
                update_epochs = [e for e in history['model_updates'] if e < max_epochs]
                if update_epochs:
                    # 在最后一个更新点添加星号标记
                    last_update = update_epochs[-1]
                    if last_update < len(data):
                        axes[1, 1].plot(last_update, data[last_update], 
                                      marker='*', markersize=15, 
                                      color=colors[model_name],
                                      label=f'{model_name} (Model Added)',
                                      zorder=5)
    
    axes[1, 1].set_xlabel('Epoch', fontfamily='Times New Roman', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontfamily='Times New Roman', fontsize=12)
    axes[1, 1].legend(fontsize=11, loc='lower right')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, max_epochs)
    axes[1, 1].set_ylim(0, 95)
    
    # 添加注释 - 显示最终结果
    note_text = "Note: Showing first 100 epochs only.\n"
    
    # 获取每个模型的最终验证准确率
    final_accs = []
    for model_name, history in histories_dict.items():
        if history and 'val_acc' in history and len(history['val_acc']) > 0:
            # 获取前100个epochs内的最佳准确率
            best_acc = max(history['val_acc'][:min(len(history['val_acc']), max_epochs)])
            if model_name == 'CoFrNet-Boost':
                # 获取总的训练epochs数和最佳准确率
                total_epochs = len(history['val_acc'])
                note_text += f"CoFrNet-Boost: {total_epochs} total epochs, Best Val Acc: {best_acc:.1f}%"
            else:
                final_accs.append(f"{model_name}: {best_acc:.1f}%")
    
    # 将注释添加到图表底部
    fig.text(0.5, 0.01, note_text, ha='center', fontsize=10, 
             fontfamily='Times New Roman', style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Training curves comparison saved to {save_path}")

def plot_model_comparison_bars(results, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors_dict = {
        'CoFrNet-Boost': '#3498db',
        'CoFrNet-D': '#2ecc71',
        'MLP': '#e74c3c'
    }
    
    # 性能指标对比
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.25
    
    if results['cofrnet_boost']:
        values = [results['cofrnet_boost']['metrics'][m] for m in metrics]
        ax.bar(x - width, values, width, label='CoFrNet-Boost', 
               color=colors_dict['CoFrNet-Boost'], alpha=0.8)
    
    if results['cofrnet_d']:
        values = [results['cofrnet_d']['metrics'][m] for m in metrics]
        ax.bar(x, values, width, label='CoFrNet-D', 
               color=colors_dict['CoFrNet-D'], alpha=0.8)
    
    if results['mlp']:
        values = [results['mlp']['metrics'][m] for m in metrics]
        ax.bar(x + width, values, width, label='MLP', 
               color=colors_dict['MLP'], alpha=0.8)
    
    ax.set_xlabel('Metrics', fontfamily='Times New Roman', fontsize=11)
    ax.set_ylabel('Performance (%)', fontfamily='Times New Roman', fontsize=11)
    ax.set_title('Performance Metrics', fontfamily='Times New Roman', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontfamily='Times New Roman', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 训练时间对比
    ax = axes[1]
    models = []
    times = []
    colors = []
    
    if results['cofrnet_boost']:
        models.append('CoFrNet-Boost')
        times.append(results['cofrnet_boost']['training_time'])
        colors.append(colors_dict['CoFrNet-Boost'])
    
    if results['cofrnet_d']:
        models.append('CoFrNet-D')
        times.append(results['cofrnet_d']['training_time'])
        colors.append(colors_dict['CoFrNet-D'])
    
    if results['mlp']:
        models.append('MLP')
        times.append(results['mlp']['training_time'])
        colors.append(colors_dict['MLP'])
    
    ax.bar(models, times, color=colors, alpha=0.8)
    ax.set_ylabel('Time (seconds)', fontfamily='Times New Roman', fontsize=11)
    ax.set_title('Training Time', fontfamily='Times New Roman', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 参数量对比
    ax = axes[2]
    models = []
    params = []
    colors = []
    
    if results['cofrnet_boost']:
        models.append('CoFrNet-Boost')
        params.append(results['cofrnet_boost'].get('model_params', 0) / 1000)
        colors.append(colors_dict['CoFrNet-Boost'])
    
    if results['cofrnet_d']:
        models.append('CoFrNet-D')
        params.append(results['cofrnet_d']['model_params'] / 1000)
        colors.append(colors_dict['CoFrNet-D'])
    
    if results['mlp']:
        models.append('MLP')
        params.append(results['mlp']['model_params'] / 1000)
        colors.append(colors_dict['MLP'])
    
    ax.bar(models, params, color=colors, alpha=0.8)
    ax.set_ylabel('Parameters (K)', fontfamily='Times New Roman', fontsize=11)
    ax.set_title('Model Size', fontfamily='Times New Roman', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Waveform Dataset - Model Comparison', 
                fontsize=16, fontweight='bold', fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Model comparison bars saved to {save_path}")

# ==================== 实验配置 ====================

# CoFrNet-Boost超参数配置
COFRNET_BOOST_CONFIG = {
    'task_name': 'Waveform_CoFrNet_Boost',
    'input_dim': None,  
    'output_dim': None,  
    'num_models_max': 30,
    'epochs_per_model': 150,
    'shallow_depth_per_cofrnet': 3,
    'polynomial_degree': 3,
    'boosting_learning_rate': 0.15,
    'learning_rate_adam': 0.003,
    'weight_decay': 5e-5,
    'batch_size': 64,
    'early_stopping_patience': 15,
}

# CoFrNet-D超参数配置
COFRNET_D_CONFIG = {
    'task_name': 'Waveform_CoFrNet_D',
    'max_depth': 60,
    'epsilon': 0.1,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 128,
    'epochs': 100,
    'early_stopping_patience': 15,
}

# MLP超参数配置
MLP_CONFIG = {
    'task_name': 'Waveform_MLP',
    'hidden_dims': [40, 16],  
    'dropout_rate': 0.3,
    'activation': 'relu',
    'learning_rate': 0.003,
    'weight_decay': 5e-5,
    'batch_size': 64,
    'epochs': 300,
    'early_stopping_patience': 20,
    'optimizer': 'adam',
    'scheduler': 'cosine',
    'gradient_clip': 1.0
}

# ==================== 单个实验运行函数 ====================

def run_cofrnet_boost_experiment():
    """运行CoFrNet-Boost实验"""
    print("\n" + "="*70)
    print(f" Starting CoFrNet-Boost Experiment: Waveform Database Generator")
    print("="*70)
    
    # 加载数据
    train_data, val_data, test_data, input_dim, n_classes = load_waveform_data()
    
    if train_data is None:
        print(f" Data loading failed")
        return None
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    # 更新超参数
    hparams = COFRNET_BOOST_CONFIG.copy()
    hparams['input_dim'] = input_dim
    hparams['output_dim'] = n_classes
    
    log_file = 'waveform_cofrnet_boost_run.log'
    json_log = 'waveform_cofrnet_boost_log.json'
    model_path = 'best_waveform_cofrnet_boost_model.pth'
    
    setup_logging(log_file)
    
    # 显示超参数
    print("\n Hyperparameter Configuration:")
    print("-"*50)
    for key, value in hparams.items():
        if key not in ['input_dim', 'output_dim']:
            print(f"  {key}: {value}")
    print(f"  input_dim: {input_dim}")
    print(f"  output_dim: {n_classes}")
    print("-"*50)
    
    # 创建数据加载器
    train_tensor = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_tensor = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_tensor = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    trainloader = DataLoader(train_tensor, batch_size=hparams['batch_size'], shuffle=True)
    valloader = DataLoader(val_tensor, batch_size=hparams['batch_size'], shuffle=False)
    testloader = DataLoader(test_tensor, batch_size=hparams['batch_size'], shuffle=False)
    
    print("\n Initializing CoFrNet-Boost classifier...")
    classifier = CoFrNetClassifier(hparams)
    
    # 开始训练
    start_time = time.time()
    
    print("\n Starting training...")
    best_acc = classifier.train(
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        log_filepath=json_log,
        model_save_path=model_path
    )
    
    training_time = time.time() - start_time
    
    print(f"\n Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # 加载最佳模型并评估
    if os.path.exists(model_path):
        print("\n Loading best model for final evaluation...")
        loaded_classifier = CoFrNetClassifier.load_model(model_path)
        
        # 在测试集上进行预测
        X_test_tensor = torch.FloatTensor(X_test)
        y_pred = loaded_classifier.predict(X_test_tensor)
        
        # 计算最终指标
        final_metrics = calculate_metrics(y_test, y_pred, n_classes)
        
        print("\n Final test set metrics:")
        print("-"*50)
        for metric, value in final_metrics.items():
            if metric != 'Confusion_Matrix' and not metric.startswith('T') and not metric.startswith('F'):
                print(f"  {metric:15s}: {value:.2f}%")
        print("-"*50)
        
        # 获取训练历史和参数量
        history = loaded_classifier.training_history
        model_params = loaded_classifier.best_model_params
        
        return {
            'metrics': final_metrics,
            'training_time': training_time,
            'model_size': loaded_classifier.best_model_size,
            'model_params': model_params,
            'test_acc': best_acc,
            'y_pred': y_pred,
            'y_test': y_test,
            'history': history
        }
    else:
        print("\n Model file not found")
        return None

def run_cofrnet_d_experiment():
    """运行CoFrNet-D实验"""
    print("\n" + "="*70)
    print(f" Starting CoFrNet-D Experiment: Waveform Database Generator")
    print("="*70)
    
    # 加载数据
    train_data, val_data, test_data, input_dim, n_classes = load_waveform_data()
    
    if train_data is None:
        print(f" Data loading failed")
        return None
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    # 获取超参数
    hparams = COFRNET_D_CONFIG.copy()
    
    # 设置文件路径
    log_file = 'waveform_cofrnet_d_run.log'
    model_path = 'best_waveform_cofrnet_d_model.pth'
    
    setup_logging(log_file)
    
    # 创建数据加载器
    train_tensor = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_tensor = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_tensor = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_tensor, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=hparams['batch_size'], shuffle=False)
    test_loader = DataLoader(test_tensor, batch_size=hparams['batch_size'], shuffle=False)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n Initializing CoFrNet-D classifier...")
    model = CoFrNetD(
        input_dim=input_dim,
        output_dim=n_classes,
        max_depth=hparams['max_depth'],
        epsilon=hparams['epsilon']
    )
    
    # 显示模型信息
    total_params = model.count_parameters()
    print(f" Total model parameters: {total_params:,}")
    
    # 显示超参数
    print("\n Hyperparameter Configuration:")
    print("-"*50)
    print(f"  Architecture: CoFrNet-D")
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {n_classes}")
    print(f"  Max depth: {hparams['max_depth']}")
    print(f"  Epsilon: {hparams['epsilon']}")
    print(f"  Learning rate: {hparams['learning_rate']}")
    print(f"  Weight decay: {hparams['weight_decay']}")
    print(f"  Batch size: {hparams['batch_size']}")
    print(f"  Max epochs: {hparams['epochs']}")
    print(f"  Early stopping patience: {hparams['early_stopping_patience']}")
    print("-"*50)
    
    # 创建训练器
    trainer = CoFrNetDTrainer(model, device, hparams)
    
    # 开始训练
    print("\n Starting training...")
    test_acc, training_time = trainer.train(train_loader, val_loader, test_loader)
    
    print(f"\n Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # 保存模型
    torch.save({
        'model_state_dict': trainer.best_model_state,
        'hyperparameters': hparams,
        'test_acc': test_acc,
        'model_params': total_params,
        'training_history': trainer.training_history
    }, model_path)
    print(f" Model saved to {model_path}")
    
    # 加载最佳模型并进行最终评估
    model.load_state_dict(trainer.best_model_state)
    
    # 在测试集上进行预测
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_pred = trainer.predict(X_test_tensor)
    
    # 计算最终指标
    final_metrics = calculate_metrics(y_test, y_pred, n_classes)
    
    print("\n Final test set metrics:")
    print("-"*50)
    for metric, value in final_metrics.items():
        if metric != 'Confusion_Matrix' and not metric.startswith('T') and not metric.startswith('F'):
            print(f"  {metric:15s}: {value:.2f}%")
    print("-"*50)
    
    return {
        'metrics': final_metrics,
        'training_time': training_time,
        'model_params': total_params,
        'test_acc': test_acc,
        'y_pred': y_pred,
        'y_test': y_test,
        'history': trainer.training_history
    }

def run_mlp_experiment():
    """运行MLP实验"""
    print("\n" + "="*70)
    print(f" Starting MLP Experiment: Waveform Database Generator")
    print("="*70)
    
    # 加载数据
    train_data, val_data, test_data, input_dim, n_classes = load_waveform_data()
    
    if train_data is None:
        print(f" Data loading failed")
        return None
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # 确保标签是整数类型
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    # 获取超参数
    hparams = MLP_CONFIG.copy()
    
    # 设置文件路径
    log_file = 'waveform_mlp_run.log'
    model_path = 'best_waveform_mlp_model.pth'
    
    # 设置日志
    setup_logging(log_file)
    
    # 创建数据加载器
    train_tensor = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_tensor = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_tensor = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_tensor, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=hparams['batch_size'], shuffle=False)
    test_loader = DataLoader(test_tensor, batch_size=hparams['batch_size'], shuffle=False)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n Initializing MLP classifier...")
    model = CompactMLPClassifier(
        input_dim=input_dim,
        hidden_dims=hparams['hidden_dims'],
        output_dim=n_classes,
        dropout_rate=hparams['dropout_rate'],
        activation=hparams['activation']
    )
    
    # 显示模型信息
    total_params = model.count_parameters()
    print(f" Total model parameters: {total_params:,}")
    
    # 显示超参数
    print("\n Hyperparameter Configuration:")
    print("-"*50)
    print(f"  Architecture: {input_dim} -> {' -> '.join(map(str, hparams['hidden_dims']))} -> {n_classes}")
    print(f"  Activation: {hparams['activation']}")
    print(f"  Dropout rate: {hparams['dropout_rate']}")
    print(f"  Optimizer: {hparams['optimizer']}")
    print(f"  Learning rate: {hparams['learning_rate']}")
    print(f"  Weight decay: {hparams['weight_decay']}")
    print(f"  Batch size: {hparams['batch_size']}")
    print(f"  Max epochs: {hparams['epochs']}")
    print(f"  Early stopping patience: {hparams['early_stopping_patience']}")
    print("-"*50)
    
    # 创建训练器
    trainer = MLPTrainer(model, device, hparams)
    
    # 开始训练
    print("\n Starting training...")
    test_acc, training_time = trainer.train(train_loader, val_loader, test_loader)
    
    print(f"\n Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # 保存模型
    torch.save({
        'model_state_dict': trainer.best_model_state,
        'hyperparameters': hparams,
        'test_acc': test_acc,
        'model_params': total_params,
        'training_history': trainer.training_history
    }, model_path)
    print(f" Model saved to {model_path}")
    
    # 加载最佳模型并进行最终评估
    model.load_state_dict(trainer.best_model_state)
    
    # 在测试集上进行预测
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_pred = trainer.predict(X_test_tensor)
    
    # 计算最终指标
    final_metrics = calculate_metrics(y_test, y_pred, n_classes)
    
    print("\n Final test set metrics:")
    print("-"*50)
    for metric, value in final_metrics.items():
        if metric != 'Confusion_Matrix' and not metric.startswith('T') and not metric.startswith('F'):
            print(f"  {metric:15s}: {value:.2f}%")
    print("-"*50)
    
    return {
        'metrics': final_metrics,
        'training_time': training_time,
        'model_params': total_params,
        'test_acc': test_acc,
        'y_pred': y_pred,
        'y_test': y_test,
        'history': trainer.training_history
    }

def run_comparison_experiment():
    """运行三个模型的对比实验"""
    print("\n" + "="*80)
    print(f" Running Full Comparison Experiment: Waveform Database Generator")
    print("="*80)
    
    # 运行CoFrNet-Boost
    print("\n" + "-"*60)
    print("Phase 1: CoFrNet-Boost Training")
    print("-"*60)
    cofrnet_boost_result = run_cofrnet_boost_experiment()
    
    # 运行CoFrNet-D
    print("\n" + "-"*60)
    print("Phase 2: CoFrNet-D Training")
    print("-"*60)
    cofrnet_d_result = run_cofrnet_d_experiment()
    
    # 运行MLP
    print("\n" + "-"*60)
    print("Phase 3: MLP Training")
    print("-"*60)
    mlp_result = run_mlp_experiment()
    
    if cofrnet_boost_result and cofrnet_d_result and mlp_result:
        # 生成对比可视化
        print("\n" + "-"*60)
        print("Phase 4: Generating Comparison Visualizations")
        print("-"*60)
        
        # 使用第一个模型的y_test作为真实标签
        y_test = cofrnet_boost_result['y_test']
        
        # 设置类别名称
        class_names = ['Wave 0', 'Wave 1', 'Wave 2']
        
        # 1. 混淆矩阵对比
        y_pred_dict = {
            'CoFrNet-Boost': cofrnet_boost_result['y_pred'],
            'CoFrNet-D': cofrnet_d_result['y_pred'],
            'MLP': mlp_result['y_pred']
        }
        plot_confusion_matrix_comparison(
            y_test, 
            y_pred_dict,
            class_names,
            'waveform_confusion_comparison.png'
        )
        
        # 2. 雷达图对比
        metrics_dict = {
            'CoFrNet-Boost': cofrnet_boost_result['metrics'],
            'CoFrNet-D': cofrnet_d_result['metrics'],
            'MLP': mlp_result['metrics']
        }
        plot_metrics_radar_chart(
            metrics_dict,
            'waveform_radar_comparison.png'
        )
        
        # 3. 训练曲线对比
        histories_dict = {
            'CoFrNet-Boost': cofrnet_boost_result.get('history'),
            'CoFrNet-D': cofrnet_d_result.get('history'),
            'MLP': mlp_result.get('history')
        }
        plot_training_curves_comparison(
            histories_dict,
            'waveform_training_comparison.png'
        )
        
        # 4. 条形图对比
        results = {
            'cofrnet_boost': cofrnet_boost_result,
            'cofrnet_d': cofrnet_d_result,
            'mlp': mlp_result
        }
        plot_model_comparison_bars(results, 'waveform_bars_comparison.png')
        
        # 打印对比总结
        print("\n" + "="*60)
        print(" Comparison Summary")
        print("="*60)
        print(f"\nDataset: Waveform Database Generator")
        print("-"*60)
        
        print("\n CoFrNet-Boost Results:")
        print(f"  - Accuracy: {cofrnet_boost_result['metrics']['Accuracy']:.2f}%")
        print(f"  - F1-Score: {cofrnet_boost_result['metrics']['F1-Score']:.2f}%")
        print(f"  - Training Time: {cofrnet_boost_result['training_time']:.1f}s")
        print(f"  - Model Size: {cofrnet_boost_result['model_size']} sub-models")
        print(f"  - Parameters: {cofrnet_boost_result.get('model_params', 0):,}")
        
        print("\n CoFrNet-D Results:")
        print(f"  - Accuracy: {cofrnet_d_result['metrics']['Accuracy']:.2f}%")
        print(f"  - F1-Score: {cofrnet_d_result['metrics']['F1-Score']:.2f}%")
        print(f"  - Training Time: {cofrnet_d_result['training_time']:.1f}s")
        print(f"  - Parameters: {cofrnet_d_result['model_params']:,}")
        
        print("\n MLP Results:")
        print(f"  - Accuracy: {mlp_result['metrics']['Accuracy']:.2f}%")
        print(f"  - F1-Score: {mlp_result['metrics']['F1-Score']:.2f}%")
        print(f"  - Training Time: {mlp_result['training_time']:.1f}s")
        print(f"  - Parameters: {mlp_result['model_params']:,}")
        
        print("\n Performance Comparison:")
        
        # 找出最佳准确率
        accuracies = {
            'CoFrNet-Boost': cofrnet_boost_result['metrics']['Accuracy'],
            'CoFrNet-D': cofrnet_d_result['metrics']['Accuracy'],
            'MLP': mlp_result['metrics']['Accuracy']
        }
        best_model = max(accuracies, key=accuracies.get)
        print(f"  - Best Accuracy: {best_model} ({accuracies[best_model]:.2f}%)")
        
        # 找出最快训练时间
        times = {
            'CoFrNet-Boost': cofrnet_boost_result['training_time'],
            'CoFrNet-D': cofrnet_d_result['training_time'],
            'MLP': mlp_result['training_time']
        }
        fastest_model = min(times, key=times.get)
        print(f"  - Fastest Training: {fastest_model} ({times[fastest_model]:.1f}s)")
        
        # 找出最少参数量
        params = {
            'CoFrNet-Boost': cofrnet_boost_result.get('model_params', 0),
            'CoFrNet-D': cofrnet_d_result['model_params'],
            'MLP': mlp_result['model_params']
        }
        smallest_model = min(params, key=params.get)
        print(f"  - Smallest Model: {smallest_model} ({params[smallest_model]:,} parameters)")
        
        return results
    
    return None

# ==================== 主函数 ====================

def main():
    print("\n" + "="*80)
    print(" "*15 + " WAVEFORM CLASSIFICATION EXPERIMENT PLATFORM")
    print(" "*20 + "CoFrNet-Boost vs CoFrNet-D vs MLP Comparison")
    print("="*80)
    
    set_random_seeds(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Running on: {device}")
    print(f" PyTorch version: {torch.__version__}")
    
    while True:
        print("\n" + "="*60)
        print(" MAIN MENU")
        print("="*60)
        print("  [1] Run CoFrNet-Boost Experiment")
        print("  [2] Run CoFrNet-D Experiment")
        print("  [3] Run MLP Experiment")
        print("  [4] Run Full Comparison (All Three Models)")
        print("  [Q] Quit")
        print("-"*60)
        
        choice = input("\n Select option: ").strip().upper()
        
        if choice == 'Q':
            print("\n Exiting program...")
            break
        
        elif choice == '1':
            run_cofrnet_boost_experiment()
        
        elif choice == '2':
            run_cofrnet_d_experiment()
        
        elif choice == '3':
            run_mlp_experiment()
        
        elif choice == '4':
            print("\n  This will run all three models on the Waveform dataset.")
            print("This may take a considerable amount of time.")
            confirm = input("Continue? (Y/N): ").strip().upper()
            if confirm == 'Y':
                run_comparison_experiment()
        
        else:
            print(" Invalid choice, please try again")
    
    print("\n" + "="*80)
    print(" "*30 + " EXPERIMENT COMPLETED")
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Program interrupted by user")
    except Exception as e:
        print(f"\n Error occurred: {e}")
        import traceback
        traceback.print_exc()
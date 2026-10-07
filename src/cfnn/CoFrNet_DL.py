import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys
import urllib.request
import warnings
warnings.filterwarnings('ignore')

class SingleLadder(nn.Module):

    def __init__(self, depth, epsilon=0.1):
        super(SingleLadder, self).__init__()
        self.depth = depth
        self.epsilon = epsilon
        # 使用单一参数张量
        self.weights = nn.Parameter(torch.randn(depth + 1) * 0.1)

    def safe_reciprocal(self, x):

        sign = torch.sign(x)
        abs_x = torch.abs(x)
        safe_abs = torch.clamp(abs_x, min=self.epsilon)
        return sign / safe_abs

    def forward(self, x_j):

        if self.depth == 0:
            return self.weights[0] * x_j
        
        # 从最深层开始
        result = self.weights[self.depth] * x_j
        
        for k in range(self.depth - 1, 0, -1):
            denominator = self.weights[k] * x_j + self.safe_reciprocal(result)
            result = denominator
        
        # 最终层: w0 * x_j + 1/result
        final_result = self.weights[0] * x_j + self.safe_reciprocal(result)
        return final_result

class FullLadder(nn.Module):

    def __init__(self, input_dim, depth, epsilon=0.1):
        super(FullLadder, self).__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.epsilon = epsilon
        # 权重矩阵: (depth+1, input_dim)
        self.weights = nn.Parameter(torch.randn(depth + 1, input_dim) * 0.1)
        
    def safe_reciprocal(self, x):

        sign = torch.sign(x)
        abs_x = torch.abs(x)
        safe_abs = torch.clamp(abs_x, min=self.epsilon)
        return sign / safe_abs

    def forward(self, x):

        if self.depth == 0:
            return F.linear(x, self.weights[0])

        # 从最深层开始
        result = F.linear(x, self.weights[self.depth])
        
        for k in range(self.depth - 1, 0, -1):
            a_k = F.linear(x, self.weights[k])
            denominator = a_k + self.safe_reciprocal(result)
            result = denominator
            
        # 最终层: w0^T * x + 1/result
        a_0 = F.linear(x, self.weights[0])
        final_result = a_0 + self.safe_reciprocal(result)
        return final_result

class CoFrNetDL(nn.Module):

    def __init__(self, input_dim, output_dim, num_full_ladders=25, 
                 max_diag_ladder_depth=12, max_full_ladder_depth=12, epsilon=0.1):
        super(CoFrNetDL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_full_ladders = num_full_ladders
        self.max_diag_ladder_depth = max_diag_ladder_depth
        self.max_full_ladder_depth = max_full_ladder_depth
        
        # 1. 对角梯子部分
        # 每个输入维度对应一个单特征梯子
        self.diag_ladders = nn.ModuleList()
        self.diag_ladder_depths = []
        
        for i in range(input_dim):

            depth = min(max_diag_ladder_depth, max(2, 5 + (i * 2) % 10))
            self.diag_ladder_depths.append(depth)
            self.diag_ladders.append(SingleLadder(depth, epsilon))
            
        # 2. 全连接梯子部分
        # 深度递增
        self.full_ladders = nn.ModuleList()
        self.full_ladder_depths = []
        
        for i in range(num_full_ladders):
            # 使用线性插值分配深度
            if num_full_ladders > 1:
                depth = 2 + int((i / (num_full_ladders - 1)) * (max_full_ladder_depth - 2))
            else:
                depth = 2
            depth = min(max_full_ladder_depth, max(2, depth))
            self.full_ladder_depths.append(depth)
            self.full_ladders.append(FullLadder(input_dim, depth, epsilon))
            
        # 3. 最终线性组合层
        combined_input_dim = input_dim + num_full_ladders
        self.output_layer = nn.Linear(combined_input_dim, output_dim)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x):

        batch_size = x.shape[0]
        
        # 1. 计算对角梯子输出
        diag_outputs = []
        for j in range(self.input_dim):
            x_j = x[:, j]  
            ladder_output = self.diag_ladders[j](x_j)
            diag_outputs.append(ladder_output.unsqueeze(1))
        
        diag_features = torch.cat(diag_outputs, dim=1)  
        
        # 2. 计算全连接梯子输出
        full_outputs = []
        for ladder in self.full_ladders:
            ladder_output = ladder(x)  
            full_outputs.append(ladder_output.unsqueeze(1))
        
        full_features = torch.cat(full_outputs, dim=1)  
        
        # 3. 拼接所有特征
        combined_features = torch.cat([diag_features, full_features], dim=1)
        
        # 4. 最终线性组合
        output = self.output_layer(combined_features)
        return output
    
    def get_feature_importance(self, x):

        device = next(self.parameters()).device
        
        with torch.no_grad():

            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            x = x.to(device)
            
            importance_dict = {}
            
            # 1. 对角梯子的特征重要性
            diag_importance = []
            for j in range(self.input_dim):
                x_j = x[:, j]
                ladder_output = self.diag_ladders[j](x_j)
                # 使用输出的平均绝对值作为重要性度量
                importance = torch.mean(torch.abs(ladder_output)).item()
                diag_importance.append(importance)
            
            importance_dict['diagonal_features'] = np.array(diag_importance)
            
            # 2. 全连接梯子的重要性
            full_importance = []
            for i, ladder in enumerate(self.full_ladders):
                ladder_output = ladder(x)
                importance = torch.mean(torch.abs(ladder_output)).item()
                full_importance.append(importance)
            
            importance_dict['interaction_features'] = np.array(full_importance)
            importance_dict['interaction_depths'] = np.array(self.full_ladder_depths)
            
            # 3. 总体特征重要性排序
            all_importance = np.concatenate([diag_importance, full_importance])
            importance_dict['total_importance'] = all_importance
            
            return importance_dict
def train_cofrnet_dl(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, 
                     weight_decay=1e-5, patience=15):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience//3, factor=0.5
    )
    
    # 训练状态
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_losses': [], 'val_losses': [], 'val_accuracies': [],
        'learning_rates': [], 'best_epoch': 0
    }
    
    print(f"\n训练配置:")
    print(f"设备: {device}")
    print(f"学习率: {learning_rate}")
    print(f"总轮数: {num_epochs}")
    print(f"早停耐心: {patience}")
    print(f"批次大小: {train_loader.batch_size}")
    print("-" * 60)
    
    # 主训练循环
    epoch_pbar = tqdm(range(num_epochs), desc="训练进度", position=0, leave=True)
    
    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        train_loss, train_samples, train_correct = 0.0, 0, 0
        
        train_pbar = tqdm(train_loader, desc=f"轮次 {epoch+1:3d}/{num_epochs} [训练]", 
                         position=1, leave=False, file=sys.stdout)
        
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            try:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # 检查NaN损失
                if torch.isnan(loss):
                    train_pbar.set_postfix({"状态": "NaN损失"})
                    continue
                    
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 统计信息
                train_loss += loss.item() * batch_x.size(0)
                train_samples += batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_y).sum().item()
                
                # 更新
                current_loss = train_loss / train_samples
                current_acc = train_correct / train_samples
                
                train_pbar.set_postfix({
                    "损失": f"{current_loss:.4f}",
                    "准确率": f"{current_acc:.4f}",
                    "学习率": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except Exception as e:
                train_pbar.set_postfix({"错误": f"{str(e)[:20]}"})
                continue
        
        train_pbar.close()
        
        if train_samples == 0:
            print("没有有效的训练样本")
            break
            
        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        val_pbar = tqdm(val_loader, desc=f"轮次 {epoch+1:3d}/{num_epochs} [验证]", 
                       position=1, leave=False, file=sys.stdout)
        
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                try:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    if not torch.isnan(loss):
                        val_loss += loss.item() * batch_x.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                        
                        # 更新验证进度
                        current_val_loss = val_loss / val_total
                        current_val_acc = val_correct / val_total
                        
                        val_pbar.set_postfix({
                            "损失": f"{current_val_loss:.4f}",
                            "准确率": f"{current_val_acc:.4f}"
                        })
                        
                except Exception as e:
                    val_pbar.set_postfix({"错误": f"{str(e)[:20]}"})
                    continue
        
        val_pbar.close()
        
        if val_total == 0:
            print("没有有效的验证样本")
            break
            
        # 计算epoch指标
        epoch_train_loss = train_loss / train_samples
        epoch_train_acc = train_correct / train_samples
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存历史记录
        history['train_losses'].append(epoch_train_loss)
        history['val_losses'].append(epoch_val_loss)
        history['val_accuracies'].append(epoch_val_acc)
        history['learning_rates'].append(current_lr)
        
        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"\n学习率调整: {old_lr:.2e} -> {new_lr:.2e}")
        
        # 更新主进度条
        epoch_pbar.set_postfix({
            "训练损失": f"{epoch_train_loss:.4f}",
            "训练准确率": f"{epoch_train_acc:.4f}",
            "验证损失": f"{epoch_val_loss:.4f}",
            "验证准确率": f"{epoch_val_acc:.4f}",
            "最佳损失": f"{best_val_loss:.4f}",
            "耐心": f"{patience_counter}/{patience}"
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"\n轮次 {epoch+1:3d} 总结:")
            print(f"  训练 - 损失: {epoch_train_loss:.6f}, 准确率: {epoch_train_acc:.6f}")
            print(f"  验证 - 损失: {epoch_val_loss:.6f}, 准确率: {epoch_val_acc:.6f}")
            print(f"  学习率: {current_lr:.2e}")
            print(f"  最佳验证损失: {best_val_loss:.6f}")
            print("-" * 60)
        
        # 早停检查
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            history['best_epoch'] = epoch + 1
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_cofrnet_dl.pth')
            epoch_pbar.set_description(f"训练进度 [最佳: {epoch+1}]")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停: 连续 {patience} 轮无改进")
                print(f"最佳验证损失: {best_val_loss:.6f} (第 {history['best_epoch']} 轮)")
                break
    
    epoch_pbar.close()
    
    # 加载最佳模型
    try:
        model.load_state_dict(torch.load('best_cofrnet_dl.pth'))
        print(f"\n训练完成: 已加载最佳模型 (验证损失: {best_val_loss:.6f})")
    except Exception as e:
        print(f"\n警告: 无法加载最佳模型: {e}")
        print("使用当前模型继续...")
    
    print()
    
    return model, history

def evaluate_cofrnet_dl(model, test_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("\n评估模型:")
    test_pbar = tqdm(test_loader, desc="测试中", leave=True)
    
    with torch.no_grad():
        for batch_x, batch_y in test_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            try:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
                # 更新当前准确率
                if len(all_predictions) > 0:
                    current_acc = accuracy_score(all_targets, all_predictions)
                    test_pbar.set_postfix({"准确率": f"{current_acc:.4f}"})
                
            except Exception as e:
                test_pbar.set_postfix({"错误": f"{str(e)[:20]}"})
                continue
    
    test_pbar.close()
    
    if len(all_predictions) == 0:
        return 0.0, [], []
        
    final_accuracy = accuracy_score(all_targets, all_predictions)
    print(f"最终测试准确率: {final_accuracy:.6f}")
    
    return final_accuracy, all_predictions, all_targets

def prepare_data(X, y, test_size=0.3, val_size=0.05, random_state=42, batch_size=128):

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 第一次划分：分离测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # 第二次划分：分离训练和验证集
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
        stratify=y_temp if len(np.unique(y_temp)) > 1 else None
    )
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 数据加载
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    
    return train_loader, val_loader, test_loader, scaler
def load_dataset(dataset_name):

    if dataset_name.lower() == 'waveform':
        print("正在加载 Waveform 数据集...")
        try:
            data = fetch_openml('waveform-5000', version=1, as_frame=True, parser='auto')
            X = data.data.values.astype(np.float32)
            y = data.target.values
            le = LabelEncoder()
            y = le.fit_transform(y)
            feature_names = list(data.feature_names)
            print(f"成功加载 Waveform 数据集: {X.shape}")
            return X, y, feature_names
        except Exception as e:
            print(f"加载 Waveform 数据集失败: {e}")
            raise
    
    elif dataset_name.lower() == 'magic':
        print("正在加载 MAGIC Gamma Telescope 数据集...")
        try:
            data = fetch_openml('MagicTelescope', version=1, as_frame=True, parser='auto')
            X = data.data.values.astype(np.float32)
            y = data.target.values
            le = LabelEncoder()
            y = le.fit_transform(y)
            feature_names = list(data.feature_names)
            print(f"成功加载 MAGIC 数据集: {X.shape}")
            return X, y, feature_names
        except Exception as e:
            print(f"加载 MAGIC 数据集失败: {e}")
            raise
    
    elif dataset_name.lower() in ['credit_card', 'creditcard']:
        print("正在加载 Credit Card Default 数据集...")
        
        local_files = ['default_of_credit_card_clients.xls']
        file_found = None
        
        for filename in local_files:
            if os.path.exists(filename):
                file_found = filename
                break
        
        if file_found:
            try:
                print(f"找到本地文件: {file_found}")
                df = pd.read_excel(file_found, header=1)
                X = df.iloc[:, 1:-1].values.astype(np.float32)
                y = df.iloc[:, -1].values
                feature_names = list(df.columns[1:-1])
                print(f"成功加载 Credit Card 数据集: {X.shape}")
                return X, y, feature_names
            except Exception as e:
                print(f"本地文件读取错误: {e}")
        
    
    elif dataset_name.lower() == 'cifar10':
        print("正在加载 CIFAR-10 数据集...")
        try:
            import torchvision
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                 download=True, transform=transform)
            
            print("转换CIFAR-10为表格格式...")
            
            # 训练集
            train_data, train_labels = [], []
            for i, (data, label) in enumerate(trainset):
                if i % 10000 == 0:
                    print(f"  处理训练样本 {i}/50000")
                train_data.append(data.numpy().flatten())
                train_labels.append(label)
            
            # 测试集
            test_data, test_labels = [], []
            for i, (data, label) in enumerate(testset):
                if i % 2000 == 0:
                    print(f"  处理测试样本 {i}/10000")
                test_data.append(data.numpy().flatten())
                test_labels.append(label)
            
            # 合并数据
            X = np.vstack([np.array(train_data), np.array(test_data)]).astype(np.float32)
            y = np.hstack([np.array(train_labels), np.array(test_labels)])
            feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
            
            print(f"成功加载 CIFAR-10 数据集: {X.shape}")
            return X, y, feature_names
        except Exception as e:
            print(f"加载 CIFAR-10 数据集失败: {e}")
            raise
    
    elif dataset_name.lower() == 'sentiment':
        print("正在加载 Sentiment Analysis 数据集...")
        
        try:
            print("尝试加载IMDB数据集...")
            from tensorflow.keras.datasets import imdb
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            vocab_size = 10000
            max_length = 500
            
            (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
            
            X_train = pad_sequences(X_train, maxlen=max_length)
            X_test = pad_sequences(X_test, maxlen=max_length)
            
            X = np.vstack([X_train, X_test]).astype(np.float32)
            y = np.hstack([y_train, y_test])
            feature_names = [f'word_pos_{i}' for i in range(max_length)]
            
            print(f"成功加载 IMDB情感数据集: {X.shape}")
            return X, y, feature_names
        except Exception as e:
            print(f"IMDB加载失败: {e}")
        
        # 尝试NLTK电影评论
        try:
            print("尝试加载NLTK电影评论...")
            import nltk
            from nltk.corpus import movie_reviews
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            nltk.download('movie_reviews', quiet=True)
            nltk.download('punkt', quiet=True)
            
            documents, labels = [], []
            for category in movie_reviews.categories():
                for fileid in movie_reviews.fileids(category):
                    documents.append(movie_reviews.raw(fileid))
                    labels.append(1 if category == 'pos' else 0)
            
            print("转换文本为TF-IDF特征...")
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', 
                                       min_df=2, max_df=0.8)
            X = vectorizer.fit_transform(documents).toarray().astype(np.float32)
            y = np.array(labels)
            feature_names = list(vectorizer.get_feature_names_out())
            
            print(f"成功加载 NLTK电影评论: {X.shape}")
            return X, y, feature_names
        except Exception as e:
            print(f"情感数据集加载失败: {e}")
            raise
    
    elif dataset_name.lower() == 'quora':
        print("正在加载 Quora Insincere Questions 数据集...")
        
        quora_files = ['train.csv']
        
        file_found = None
        for filename in quora_files:
            if os.path.exists(filename):
                file_found = filename
                break
        
        if file_found:
            try:
                print(f"找到Quora数据集: {file_found}")
                df = pd.read_csv(file_found)
                
                if len(df) > 50000:
                    df = df.sample(n=50000, random_state=42)
                    print(f"使用子集 {len(df)} 样本以加快处理")
                
                print("转换文本为TF-IDF特征...")
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                vectorizer = TfidfVectorizer(
                    max_features=4000, stop_words='english',
                    min_df=2, max_df=0.8, ngram_range=(1, 2)
                )
                
                X = vectorizer.fit_transform(df['question_text']).toarray().astype(np.float32)
                y = df['target'].values
                feature_names = list(vectorizer.get_feature_names_out())
                
                print(f"成功加载 Quora数据集: {X.shape}")
                return X, y, feature_names
            except Exception as e:
                print(f"Quora数据集处理错误: {e}")
        
        raise FileNotFoundError("Quora数据集未找到")
    
    else:
        available_datasets = ['waveform', 'magic', 'credit_card', 'cifar10', 'sentiment', 'quora']
        raise ValueError(f"未知数据集: {dataset_name}. 可用数据集: {available_datasets}")

def get_recommended_params_cofrnet_dl(dataset_name, input_dim):

    base_params = {
        'num_epochs': 80,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 128,
        'patience': 15,
        'num_full_ladders': 25,
        'max_diag_ladder_depth': 12,
        'max_full_ladder_depth': 12
    }
    
    # 调参
    if dataset_name.lower() == 'waveform':

        return {
            **base_params,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'num_full_ladders': 30,
            'max_diag_ladder_depth': 15,
            'max_full_ladder_depth': 15,
            'batch_size': 128
        }
    
    elif dataset_name.lower() == 'magic':

        return {
            **base_params,
            'num_epochs': 80,
            'learning_rate': 0.001,
            'num_full_ladders': 25,
            'max_diag_ladder_depth': 12,
            'max_full_ladder_depth': 12,
            'batch_size': 128
        }
    
    elif dataset_name.lower() in ['credit_card', 'creditcard']:

        return {
            **base_params,
            'num_epochs': 120,
            'learning_rate': 0.0008,
            'weight_decay': 1e-4,
            'num_full_ladders': 35,
            'max_diag_ladder_depth': 18,
            'max_full_ladder_depth': 20,
            'batch_size': 256
        }
    
    elif dataset_name.lower() == 'cifar10':

        return {
            **base_params,
            'num_epochs': 60,
            'learning_rate': 0.0005,
            'weight_decay': 1e-4,
            'num_full_ladders': 20,
            'max_diag_ladder_depth': 8,
            'max_full_ladder_depth': 10,
            'batch_size': 256
        }
    
    elif dataset_name.lower() == 'sentiment':

        return {
            **base_params,
            'num_epochs': 50,
            'learning_rate': 0.0005,
            'num_full_ladders': 25,
            'max_diag_ladder_depth': 10,
            'max_full_ladder_depth': 12,
            'batch_size': 256
        }
    
    elif dataset_name.lower() == 'quora':

        return {
            **base_params,
            'num_epochs': 50,
            'learning_rate': 0.0005,
            'num_full_ladders': 30,
            'max_diag_ladder_depth': 12,
            'max_full_ladder_depth': 15,
            'batch_size': 256
        }
    
    else:
        return base_params
def run_cofrnet_dl_experiment(dataset_name, custom_params=None):

    print(f"\n{'='*60}")
    print(f"在 {dataset_name.upper()} 数据集上运行 CoFrNet-DL 实验")
    print(f"{'='*60}")
    
    try:
        # 加载数据
        X, y, feature_names = load_dataset(dataset_name)
        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        
        print(f"数据集形状: {X.shape}, 类别数: {output_dim}")
        
        if custom_params is None:
            params = get_recommended_params_cofrnet_dl(dataset_name, input_dim)
        else:
            params = {**get_recommended_params_cofrnet_dl(dataset_name, input_dim), **custom_params}
        
        train_loader, val_loader, test_loader, scaler = prepare_data(
            X, y, batch_size=params['batch_size']
        )
        
        # 创建模型
        model = CoFrNetDL(
            input_dim=input_dim,
            output_dim=output_dim,
            num_full_ladders=params['num_full_ladders'],
            max_diag_ladder_depth=params['max_diag_ladder_depth'],
            max_full_ladder_depth=params['max_full_ladder_depth']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型已创建: {input_dim} 个输入 -> {output_dim} 个输出")
        print(f"总参数量: {total_params:,}")
        print(f"对角梯子: {input_dim}个")
        print(f"全连接梯子: {params['num_full_ladders']}个")
        
        # 训练模型
        print("\n开始训练...")
        trained_model, history = train_cofrnet_dl(
            model, train_loader, val_loader,
            num_epochs=params['num_epochs'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            patience=params['patience']
        )
        
        # 评估模型
        print("\n在测试集上评估...")
        test_accuracy, predictions, targets = evaluate_cofrnet_dl(
            trained_model, test_loader
        )
        
        print(f"\n最终测试准确率: {test_accuracy:.4f}")
        
        # 特征重要性分析（仅对中小规模数据集）
        analysis_results = None
        if input_dim <= 100:
            print("\n分析特征重要性...")
            try:
                sample_batch = next(iter(test_loader))
                sample_x = sample_batch[0][:100]
                
                importance_dict = trained_model.get_feature_importance(sample_x)
                
                # 单特征重要性
                diag_importance = importance_dict['diagonal_features']
                top_diag_indices = np.argsort(np.abs(diag_importance))[-5:][::-1]
                
                print("前5个最重要的单特征:")
                for i, idx in enumerate(top_diag_indices):
                    feature_name = feature_names[idx] if len(feature_names) > idx else f"特征_{idx}"
                    print(f"  {i+1}. {feature_name}: {diag_importance[idx]:.4f}")
                
                analysis_results = importance_dict
                
            except Exception as e:
                print(f"特征分析失败: {e}")
        else:
            print("跳过高维数据的特征分析")
        
        # 结果返回
        result = {
            'dataset': dataset_name,
            'test_accuracy': test_accuracy,
            'model': trained_model,
            'history': history,
            'analysis_results': analysis_results,
            'dataset_shape': X.shape,
            'num_parameters': total_params,
            'experiment_params': params
        }
        
        return result
        
    except Exception as e:
        print(f"实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():

    print("\n" + "="*60)
    print(" CoFrNet-DL 复现")
    print(" 基于论文: 'CoFrNets: Interpretable Neural Architecture")
    print("          Inspired by Continued Fractions'")
    print("="*60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n可用数据集:")
    datasets = ['waveform', 'magic', 'credit_card', 'cifar10', 'sentiment', 'quora']
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    
    while True:
        try:

            dataset_choice = input("\n请输入数据集名称或编号: ").strip().lower()
            
            if dataset_choice.isdigit():
                idx = int(dataset_choice) - 1
                if 0 <= idx < len(datasets):
                    dataset_choice = datasets[idx]
            
            if dataset_choice in datasets:
                print(f"\n正在 {dataset_choice} 上运行实验...")
                
                print("\n实验参数设置（直接回车使用默认值）:")
                
                use_custom = input("使用自定义参数? (y/N): ").strip().lower()
                
                if use_custom == 'y':
                    try:
                        custom_params = {}
                        custom_params['num_epochs'] = int(input("训练轮次 (默认80): ") or "80")
                        custom_params['learning_rate'] = float(input("学习率 (默认0.001): ") or "0.001")
                        custom_params['num_full_ladders'] = int(input("全连接梯子数量 (默认25): ") or "25")
                        custom_params['max_diag_ladder_depth'] = int(input("对角梯子最大深度 (默认12): ") or "12")
                        custom_params['max_full_ladder_depth'] = int(input("全连接梯子最大深度 (默认12): ") or "12")
                        custom_params['batch_size'] = int(input("批次大小 (默认128): ") or "128")
                    except ValueError:
                        print("参数输入错误，使用推荐参数")
                        custom_params = None
                else:
                    print("使用推荐参数...")
                    custom_params = None
                
                result = run_cofrnet_dl_experiment(dataset_choice, custom_params)
                
                if result:
                    print(f"\n实验成功完成!")
                    print(f"最终准确率: {result['test_accuracy']:.4f}")
                    
                    # 保存结果
                    import pickle
                    save_result = {k: v for k, v in result.items() if k != 'model'}
                    with open(f'cofrnet_dl_{dataset_choice}_result.pkl', 'wb') as f:
                        pickle.dump(save_result, f)
                    print(f"结果已保存到 'cofrnet_dl_{dataset_choice}_result.pkl'")
                    
                    # 与论文结果比较
                    paper_results = {
                        'waveform': 0.86, 'magic': 0.87, 'credit_card': 0.71,
                        'cifar10': 0.87, 'sentiment': 0.84, 'quora': 0.88
                    }
                    
                    if dataset_choice in paper_results:
                        paper_acc = paper_results[dataset_choice]
                        diff = result['test_accuracy'] - paper_acc
                        print(f"\n与论文结果比较:")
                        print(f"  论文准确率: {paper_acc:.4f}")
                        print(f"  我们的准确率: {result['test_accuracy']:.4f}")
                        print(f"  差异: {diff:+.4f}")
                        
                        if abs(diff) <= 0.02:
                            print("  -> 结果接近论文报告值")
                        elif diff > 0.02:
                            print("  -> 结果优于论文报告值")
                        else:
                            print("  -> 结果低于论文报告值")
                else:
                    print("实验失败!")
            else:
                print("无效的数据集选择!")
                continue
                
        except KeyboardInterrupt:
            print("\n\n用户中断，正在退出...")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
        
        continue_choice = input("\n是否运行另一个实验? (y/N): ").strip().lower()
        if continue_choice != 'y':
            break
    
    print("\n再见!")

def check_dependencies():

    required_packages = [
        ('torch', 'PyTorch'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(name)
    
    if missing_packages:
        print("缺少必需的包:")
        for package in missing_packages:
            print(f"    {package}")
        print("\n请: pip install " + " ".join([p.lower() for p in missing_packages]))
        return False
    
    return True

if __name__ == "__main__":
    print("检查依赖...")
    
    if not check_dependencies():
        print("\n请先安装缺少的依赖!")
        exit(1)
    
    print("所有依赖已找到!")
    
    try:
        main()
    except Exception as e:
        print(f"\n致命错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n请检查:是否有足够的磁盘空间和内存")

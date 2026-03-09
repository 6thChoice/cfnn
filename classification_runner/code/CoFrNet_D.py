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
        
        # 可学习的权重
        self.weights = nn.Parameter(torch.randn(depth + 1) * 0.1)
        
    def safe_reciprocal(self, x):
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        safe_abs = torch.clamp(abs_x, min=self.epsilon)
        return sign / safe_abs
    
    def forward(self, x_j):

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

            depth = min(max_depth, max(1, 5 + i % 10))  #可调
            self.ladder_depths.append(depth)
            
            # 单个梯形
            ladder = SingleLadder(depth, epsilon)
            self.ladders.append(ladder)
        
        # 最终层
        self.output_layer = nn.Linear(input_dim, output_dim)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):

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
    
    def get_feature_importance(self, x):

        device = next(self.parameters()).device  
        
        with torch.no_grad():
 
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            x = x.to(device)
            
            ladder_outputs = []
            for j in range(self.input_dim):
                x_j = x[:, j]
                ladder_output = self.ladders[j](x_j)
                # 使用平均绝对值作为重要性度量
                importance = torch.mean(torch.abs(ladder_output)).item()
                ladder_outputs.append(importance)
            
            return np.array(ladder_outputs)
    
    def get_ladder_weights(self):

        all_weights = []
        for ladder in self.ladders:
            all_weights.append(ladder.weights.detach().cpu().numpy())
        return all_weights

def train_cofrnet_d(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, 
                    weight_decay=1e-4, patience=15):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          patience=patience//3, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\n训练配置:")
    print(f"设备: {device}")
    print(f"总轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"批次大小: {train_loader.batch_size}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print("-" * 60)
    
    # 主训练循环
    epoch_pbar = tqdm(range(num_epochs), desc="训练进度", position=0, leave=True)
    
    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_samples = 0
        train_correct = 0
        
        # 进度条
        train_pbar = tqdm(train_loader, desc=f"轮次 {epoch+1:3d}/{num_epochs} [训练]", 
                         position=1, leave=False, file=sys.stdout)
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_pbar):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # 检查NaN损失
                if torch.isnan(loss):
                    train_pbar.set_postfix({"状态": "检测到NaN损失"})
                    continue
                    
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 统计
                train_loss += loss.item() * batch_x.size(0)
                train_samples += batch_x.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_y).sum().item()
                
                current_loss = train_loss / train_samples if train_samples > 0 else 0
                current_acc = train_correct / train_samples if train_samples > 0 else 0
                
                train_pbar.set_postfix({
                    "损失": f"{current_loss:.4f}",
                    "准确率": f"{current_acc:.4f}",
                    "学习率": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except Exception as e:
                train_pbar.set_postfix({"状态": f"错误: {str(e)[:20]}"})
                continue
        
        train_pbar.close()
        
        if train_samples == 0:
            print("没有处理有效的训练样本")
            break
            
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 验证进度条
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
                        current_val_loss = val_loss / val_total if val_total > 0 else 0
                        current_val_acc = val_correct / val_total if val_total > 0 else 0
                        
                        val_pbar.set_postfix({
                            "损失": f"{current_val_loss:.4f}",
                            "准确率": f"{current_val_acc:.4f}"
                        })
                        
                except Exception as e:
                    val_pbar.set_postfix({"状态": f"错误: {str(e)[:20]}"})
                    continue
        
        val_pbar.close()
        
        if val_total == 0:
            print("没有处理有效的验证样本")
            break
            
        # 计算轮次指标
        epoch_train_loss = train_loss / train_samples
        epoch_train_acc = train_correct / train_samples
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)
        
        # 更新主进度条
        epoch_pbar.set_postfix({
            "训练损失": f"{epoch_train_loss:.4f}",
            "训练准确率": f"{epoch_train_acc:.4f}",
            "验证损失": f"{epoch_val_loss:.4f}",
            "验证准确率": f"{epoch_val_acc:.4f}",
            "最佳验证": f"{best_val_loss:.4f}",
            "耐心": f"{patience_counter}/{patience}"
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"\n轮次 {epoch+1:3d} 总结:")
            print(f"  训练 - 损失: {epoch_train_loss:.6f}, 准确率: {epoch_train_acc:.6f}")
            print(f"  验证 - 损失: {epoch_val_loss:.6f}, 准确率: {epoch_val_acc:.6f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  最佳验证损失: {best_val_loss:.6f}")
            print("-" * 60)
        
        # 早停检查
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_cofrnet_d.pth')
            epoch_pbar.set_description(f"训练进度 [最佳: {epoch+1}]")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停: {patience}轮没有改进")
                print(f"最佳验证损失: {best_val_loss:.6f} 在轮次 {epoch+1-patience}")
                break
    
    epoch_pbar.close()
    
    # 加载最佳模型
    try:
        model.load_state_dict(torch.load('best_cofrnet_d.pth'))
        print(f"\n训练完成: 已加载最佳模型 (验证损失: {best_val_loss:.6f})")
    except:
        print("\n警告: 无法加载最佳模型，使用当前模型")
    
    print()  
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_epoch': len(train_losses) - patience_counter,
        'best_val_loss': best_val_loss
    }

def evaluate_model(model, test_loader):

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
                
                # 显示当前准确率
                if len(all_predictions) > 0:
                    current_acc = accuracy_score(all_targets, all_predictions)
                    test_pbar.set_postfix({"准确率": f"{current_acc:.4f}"})
                
            except Exception as e:
                test_pbar.set_postfix({"状态": f"错误: {str(e)[:20]}"})
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
    
    # 第一次：分离测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 第二次：从剩余数据中分离训练集和验证集
    val_size_adjusted = val_size / (1 - test_size)  
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
    
    elif dataset_name.lower() == 'credit_card' or dataset_name.lower() == 'creditcard':
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
                
                # 移除ID列并使用最后一列作为目标
                X = df.iloc[:, 1:-1].values.astype(np.float32)
                y = df.iloc[:, -1].values
                feature_names = list(df.columns[1:-1])
                
                print(f"成功加载 Credit Card 数据集: {X.shape}")
                return X, y, feature_names
                
            except Exception as e:
                print(f"读取本地文件错误: {e}")
        
    
    elif dataset_name.lower() == 'cifar10':
        print("正在加载 CIFAR-10 数据集...")
        try:
            import torchvision
            import torchvision.transforms as transforms
            
            # 表格化处理
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                 download=True, transform=transform)
            
            print("将CIFAR-10转换为表格格式...")
            
            # 转换为numpy数组并展平图像
            train_data = []
            train_labels = []
            for i, (data, label) in enumerate(trainset):
                if i % 10000 == 0:
                    print(f"处理训练样本 {i}/50000")
                train_data.append(data.numpy().flatten())
                train_labels.append(label)
            
            test_data = []
            test_labels = []
            for i, (data, label) in enumerate(testset):
                if i % 2000 == 0:
                    print(f"处理测试样本 {i}/10000")
                test_data.append(data.numpy().flatten())
                test_labels.append(label)
            
            # 合并训练和测试
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
            
            # 加载IMDB数据集
            vocab_size = 10000
            max_length = 500
            
            (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
            
            # 将序列填充到固定长度
            X_train = pad_sequences(X_train, maxlen=max_length)
            X_test = pad_sequences(X_test, maxlen=max_length)
            
            # 合并训练和测试
            X = np.vstack([X_train, X_test]).astype(np.float32)
            y = np.hstack([y_train, y_test])
            
            feature_names = [f'word_pos_{i}' for i in range(max_length)]
            
            print(f"成功加载 IMDB 情感数据集: {X.shape}")
            return X, y, feature_names
            
        except Exception as e:
            print(f"加载IMDB失败: {e}")
        
        # 尝试从NLTK加载电影评论
        try:
            print("尝试加载NLTK电影评论...")
            import nltk
            from nltk.corpus import movie_reviews
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            nltk.download('movie_reviews', quiet=True)
            nltk.download('punkt', quiet=True)
            
            documents = []
            labels = []
            
            for category in movie_reviews.categories():
                for fileid in movie_reviews.fileids(category):
                    documents.append(movie_reviews.raw(fileid))
                    labels.append(1 if category == 'pos' else 0)
            
            # 转换为TF-IDF特征
            print("将文本转换为TF-IDF特征...")
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', 
                                       min_df=2, max_df=0.8)
            X = vectorizer.fit_transform(documents).toarray().astype(np.float32)
            y = np.array(labels)
            feature_names = list(vectorizer.get_feature_names_out())
            
            print(f"成功加载 NLTK 电影评论: {X.shape}")
            return X, y, feature_names
            
        except Exception as e:
            print(f"加载情感数据集失败: {e}")
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
                
                # 使用前50k个样本
                if len(df) > 50000:
                    df = df.sample(n=50000, random_state=42)
                    print(f"使用{len(df)}个样本的子集以加快处理速度")
                
                print("将文本转换为TF-IDF特征...")
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                vectorizer = TfidfVectorizer(
                    max_features=4000,
                    stop_words='english',
                    min_df=2,
                    max_df=0.8,
                    ngram_range=(1, 2)
                )
                
                X = vectorizer.fit_transform(df['question_text']).toarray().astype(np.float32)
                y = df['target'].values
                feature_names = list(vectorizer.get_feature_names_out())
                
                print(f"成功加载 Quora 数据集: {X.shape}")
                return X, y, feature_names
                
            except Exception as e:
                print(f"处理Quora数据集错误: {e}")
        
        raise FileNotFoundError("未找到Quora数据集")
    
    else:
        raise ValueError(f"未知数据集: {dataset_name}. 可用的数据集: waveform, magic, credit_card, cifar10, sentiment, quora")
    
def run_experiment(dataset_name, max_depth=50, num_epochs=100, learning_rate=0.001):

    print(f"\n{'='*60}")
    print(f"在 {dataset_name.upper()} 数据集上运行 CoFrNet-D 实验")
    print(f"{'='*60}")
    
    try:
        # 加载数据
        X, y, feature_names = load_dataset(dataset_name)
        print(f"数据集形状: {X.shape}, 类别数: {len(np.unique(y))}")
        
        train_loader, val_loader, test_loader, scaler = prepare_data(
            X, y, batch_size=128 if X.shape[0] > 1000 else 32
        )
        
        # 创建模型
        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        
        # 根据数据集大小调整max_depth
        if input_dim > 1000:  # 对于高维数据
            max_depth = min(20, max_depth)
        
        model = CoFrNetD(input_dim=input_dim, output_dim=output_dim, max_depth=max_depth)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型已创建: {input_dim} 个输入 -> {output_dim} 个输出")
        print(f"总参数量: {total_params:,}")
        
        # 根据数据集调整训练参数
        if dataset_name.lower() in ['cifar10', 'sentiment', 'quora']:
            learning_rate = 0.0005  
            num_epochs = min(50, num_epochs)  
        
        print(f"训练参数: 学习率={learning_rate}, 轮次={num_epochs}")
        
        # 训练模型
        print("开始训练...")
        trained_model, history = train_cofrnet_d(
            model, train_loader, val_loader, 
            num_epochs=num_epochs, 
            learning_rate=learning_rate,
            patience=15
        )
        
        print("在测试集上评估...")
        test_accuracy, predictions, targets = evaluate_model(trained_model, test_loader)
        
        print(f"\n 最终测试准确率: {test_accuracy:.4f}")
        
        # 特征重要性分析（当然这仅对小数据集）
        if input_dim <= 100:
            print("\n分析特征重要性...")
            try:

                sample_batch = next(iter(test_loader))
                sample_x = sample_batch[0][:100]  
                
                # 获取特征重要性
                importance_scores = trained_model.get_feature_importance(sample_x)
                
                # 获取前5个最重要的特征
                top_indices = np.argsort(np.abs(importance_scores))[-5:][::-1]
                
                print("前5个最重要的特征:")
                for i, idx in enumerate(top_indices):
                    feature_name = feature_names[idx] if len(feature_names) > idx else f"特征_{idx}"
                    print(f"  {i+1}. {feature_name}: {importance_scores[idx]:.4f}")
                    
            except Exception as e:
                print(f"无法计算特征重要性: {e}")
                importance_scores = None
                top_indices = None
        else:
            print("跳过高维数据的特征重要性分析")
            importance_scores = None
            top_indices = None
        
        result = {
            'dataset': dataset_name,
            'test_accuracy': test_accuracy,
            'model': trained_model,
            'history': history,
            'feature_importance': importance_scores,
            'dataset_shape': X.shape,
            'num_parameters': total_params
        }
        
        if top_indices is not None and importance_scores is not None:
            result['top_features'] = [(feature_names[idx] if len(feature_names) > idx else f"特征_{idx}", 
                                     importance_scores[idx]) for idx in top_indices]
        
        return result
        
    except Exception as e:
        print(f"在 {dataset_name} 上运行实验时出错: {str(e)}")
        import traceback
        traceback.print_exc()  # 调试
        return None

def main():

    print("\n" + "="*60)
    print(" CoFrNet-D 复现")
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
                try:
                    epochs = int(input("训练轮次 (默认 80): ") or "80")
                    lr = float(input("学习率 (默认 0.001): ") or "0.001")
                    depth = int(input("最大梯形深度 (默认 30): ") or "30")
                except ValueError:
                    print("使用默认参数...")
                    epochs, lr, depth = 80, 0.001, 30
                
                result = run_experiment(
                    dataset_choice, 
                    max_depth=depth, 
                    num_epochs=epochs, 
                    learning_rate=lr
                )
                
                if result:
                    print(f"\n实验成功完成!")
                    print(f"最终准确率: {result['test_accuracy']:.4f}")
                    
                    # 保存结果
                    import pickle
                    with open(f'cofrnet_d_{dataset_choice}_result.pkl', 'wb') as f:

                        save_result = {k: v for k, v in result.items() if k != 'model'}
                        pickle.dump(save_result, f)
                    print(f"结果已保存到 'cofrnet_d_{dataset_choice}_result.pkl'")
                    
                    # 与论文结果比较
                    paper_results = {
                        'waveform': 0.69, 'magic': 0.76, 'credit_card': 0.66,
                        'cifar10': 0.38, 'sentiment': 0.80, 'quora': 0.75
                    }
                    
                    if dataset_choice in paper_results:
                        paper_acc = paper_results[dataset_choice]
                        diff = result['test_accuracy'] - paper_acc
                        print(f"\n与论文结果比较:")
                        print(f"  论文准确率: {paper_acc:.4f}")
                        print(f"  我们的准确率: {result['test_accuracy']:.4f}")
                        print(f"  差异: {diff:+.4f}")
                        
                        if abs(diff) <= 0.05:
                            print("  -> 结果接近论文报告值")
                        elif diff > 0.05:
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
        print("\n请检查是否有足够的磁盘空间和内存")

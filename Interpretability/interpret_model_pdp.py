# interpret_model_pdp.py (Corrected Version)

import torch
import joblib
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import sys
sys.path.append("/home/zxc/CodeBase/cofrnet")
from cfnet_regressor import CoFrNetRegressor

# 复用加载函数
def load_essentials(model_path='best_california_housing_regressor_3.pth', scaler_path='scaler.gz'):
    print("正在加载模型和相关组件...")
    try:
        regressor = CoFrNetRegressor.load_model(model_path)
    except FileNotFoundError:
        print(f"错误: 模型文件 {model_path} 未找到。")
        sys.exit(1)
    hparams = regressor.hparams
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"错误: 缩放器文件 {scaler_path} 未找到。")
        sys.exit(1)
    feature_names = fetch_california_housing().feature_names
    print("加载完成。")
    return regressor.ensemble_model, hparams, scaler, feature_names

def get_training_data():
    """加载并分割数据，以获取原始训练集，用于计算平均值"""
    print("正在加载原始数据用于PDP分析...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # 使用与训练脚本完全相同的分割参数
    # 第一次分割：从完整数据中分出训练集和测试集
    X_train_full, _, y_train_full, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 第二次分割：从大的训练集中分出最终的训练集和验证集
    # 我们只需要最终的 X_train 用于后续分析
    X_train, _, _, _ = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)
    
    return X_train

def plot_partial_dependence(model, feature_index, feature_name, X_train_original, scaler):
    """
    绘制单个特征的偏依赖图。
    X_train_original 是用于计算其他特征平均值的 *未标准化* 的训练数据。
    """
    print(f"正在为特征 '{feature_name}' 计算偏依赖关系...")
    model.eval()
    model.to("cpu")
    
    # 创建一个数据集副本，其中所有样本都取训练集的平均值
    X_avg = X_train_original.mean(axis=0)
    X_temp = np.tile(X_avg, (100, 1)) # 创建100个平均样本
    
    # 获取我们想变化的特征的值域（基于原始数据的分布）
    feature_values = np.linspace(
        np.percentile(X_train_original[:, feature_index], 5), 
        np.percentile(X_train_original[:, feature_index], 95), 
        100
    )
    
    # 将这100个样本中对应特征列的值，替换为我们想观察的值域
    X_temp[:, feature_index] = feature_values
    
    # 对生成的数据进行标准化
    X_scaled = scaler.transform(X_temp)
    X_tensor = torch.from_numpy(X_scaled).float()
    
    # 使用模型进行预测
    with torch.no_grad():
        predictions = model(X_tensor).numpy().flatten()
        
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(feature_values, predictions, color='forestgreen', linewidth=2.5)
    plt.title(f"Partial Dependence Plot for '{feature_name}'", fontsize=16)
    plt.xlabel(f"Value of {feature_name}", fontsize=12)
    plt.ylabel("Average Predicted House Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = f"plot3/pdp_{feature_name}.png"
    plt.savefig(save_path)
    print(f"偏依赖图已保存到: {save_path}")
    plt.close()

if __name__ == '__main__':
    model, hparams, scaler, feature_names = load_essentials()
    X_train = get_training_data()

    # 为最重要的几个特征绘制PDP图
    features_to_plot = ["MedInc", "AveRooms", "HouseAge"]
    
    for feature in features_to_plot:
        try:
            # 获取特征名称对应的索引
            feature_idx = feature_names.index(feature)
            plot_partial_dependence(model, feature_idx, feature, X_train, scaler)
        except ValueError:
            print(f"警告: 特征 '{feature}' 在数据集中未找到，跳过。")
    
    print("\nPDP图生成完毕。")
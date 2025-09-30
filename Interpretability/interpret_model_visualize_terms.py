# interpret_model_visualize_terms.py

import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

import sys
sys.path.append("/home/zxc/CodeBase/cofrnet")
from cfnet_regressor import CoFrNetRegressor

# 复用加载函数
def load_essentials(model_path='best_california_housing_regressor_2.pth', scaler_path='scaler.gz'):
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

def plot_feature_importance_for_term(term, feature_names, title=""):
    """可视化单个 PolynomialTerm 中特征的投影权重"""
    weights = term.projection.weight.detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=weights, y=feature_names, palette="vlag")
    plt.title(title, fontsize=16)
    plt.xlabel("Projection Weight", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = f"plot2/{title.replace(' ', '_').replace('->', '')}.png"
    plt.savefig(save_path)
    print(f"特征重要性图已保存到: {save_path}")
    plt.close()

def plot_polynomial_shape(term, title=""):
    """可视化学习到的多项式函数的形状"""
    coeffs = term.coeffs.detach().cpu().numpy().flatten()
    poly_func = np.poly1d(coeffs[::-1]) # np.poly1d 需要系数从高次到低次
    
    z = np.linspace(-1, 1, 400)
    output = poly_func(z)
    
    plt.figure(figsize=(8, 6))
    plt.plot(z, output, label="Learned Polynomial P(z)", color='dodgerblue', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel("Intermediate Variable 'z' (output of tanh)", fontsize=12)
    plt.ylabel("Output of Polynomial Term", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    save_path = f"plot2/{title.replace(' ', '_').replace('->', '')}.png"
    plt.savefig(save_path)
    print(f"多项式形状图已保存到: {save_path}")
    plt.close()

if __name__ == '__main__':
    model, hparams, scaler, feature_names = load_essentials()
    
    if not model.models:
        print("模型中没有已采纳的子模型，无法进行可视化。")
    else:
        print("\n开始生成微观构建单元的可视化图...")
        # 为了避免过多图片，我们只分析前2个子模型的前2个多项式项
        num_submodels_to_viz = min(2, len(model.models))
        
        for i in range(num_submodels_to_viz):
            cfnet = model.models[i]
            num_terms_to_viz = max(2, len(cfnet.terms))
            for k in range(num_terms_to_viz):
                term = cfnet.terms[k]
                base_title = f"CFNet_{i+1} -> Term_{k}"
                
                # 可视化特征重要性
                viz_title_importance = f"Feature Importance for {base_title}"
                plot_feature_importance_for_term(term, feature_names, title=viz_title_importance)
                
                # 可视化多项式形状
                viz_title_shape = f"Polynomial Shape for {base_title}"
                plot_polynomial_shape(term, title=viz_title_shape)
        print("\n可视化图生成完毕。")
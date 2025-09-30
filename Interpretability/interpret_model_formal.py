# interpret_model_formal.py

import torch
import joblib
from sklearn.datasets import fetch_california_housing

import sys
sys.path.append("/home/zxc/CodeBase/cofrnet")
from cfnet_regressor import CoFrNetRegressor

def load_essentials(model_path='best_california_housing_regressor.pth', scaler_path='scaler.gz'):
    """加载模型、超参数和数据缩放器"""
    print("正在加载模型和相关组件...")
    try:
        regressor = CoFrNetRegressor.load_model(model_path)
    except FileNotFoundError:
        print(f"错误: 模型文件 {model_path} 未找到。")
        print("请先运行 experiment_housing.py 脚本来训练和保存模型。")
        sys.exit(1)
        
    hparams = regressor.hparams
    
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"错误: 缩放器文件 {scaler_path} 未找到。")
        print("请修改 experiment_housing.py 以保存 scaler 并重新运行训练。")
        sys.exit(1)

    # 从 Scikit-learn 加载特征名称
    feature_names = fetch_california_housing().feature_names

    print("加载完成。")
    return regressor.ensemble_model, hparams, scaler, feature_names

def express_model_numerically(model, feature_names, hparams):
    """
    以数值和半形式化的方式打印出模型学到的内容。
    """
    alpha = hparams['boosting_learning_rate']
    num_models = len(model.models)

    print("\n" + "="*80)
    print(" 模型形式化与数值化表达")
    print("="*80)
    print(f"最终预测函数 F(X) = {alpha} * (CFNet_1(X) + ... + CFNet_{num_models}(X))")
    print(f"共包含 {num_models} 个子模型。")
    
    # 遍历每一个被采纳的 CFNet 子模型
    for i, cfnet in enumerate(model.models):
        print("\n" + "#"*60)
        print(f"##  分析子模型 CFNet_{i+1}")
        print("#"*60)
        
        # 打印连分式结构
        cfnet_expr = "P_0(X)"
        betas = [torch.nn.functional.softplus(rb).detach().cpu().numpy()[0] for rb in cfnet.raw_betas]
        
        temp_expr = ""
        for j in range(len(betas)):
            # 假设输出维度为1，直接取第一个元素
            beta_val = betas[j]
            temp_expr += f" + {beta_val:.4f} / (P_{j+1}(X)"
        temp_expr += ")" * len(betas)
        
        print(f"  CFNet_{i+1}(X) = P_0(X){temp_expr}")

        # 遍历 CFNet 中的每一个多项式项
        for k, term in enumerate(cfnet.terms):
            print(f"\n    --- 多项式项 P_{k} 详情 ---")
            
            # 1. 提取线性投影 W 和 b
            W = term.projection.weight.detach().cpu().numpy().flatten()
            b = term.projection.bias.detach().cpu().numpy()[0]
            
            print(f"    1. 内部线性组合 z = tanh( W * X_scaled + b )")
            for w, name in zip(W, feature_names):
                print(f"       W_({name}) = {w:.4f}")
            print(f"       b = {b:.4f}")
            
            # 2. 提取多项式系数 c
            coeffs = term.coeffs.detach().cpu().numpy().flatten()
            
            poly_expr_parts = []
            for d, c in enumerate(coeffs):
                if d == 0:
                    poly_expr_parts.append(f"{c:.4f}")
                elif d == 1:
                    poly_expr_parts.append(f"({c:.4f} * z)")
                else:
                    poly_expr_parts.append(f"({c:.4f} * z^{d})")
            poly_expr = " + ".join(poly_expr_parts)
            print(f"    2. 多项式变换 P_{k}(z) = {poly_expr}")

if __name__ == '__main__':
    # 确保 cfnet_regressor.py 和模型文件在当前目录或Python路径下
    model, hparams, scaler, feature_names = load_essentials()
    express_model_numerically(model, feature_names, hparams)
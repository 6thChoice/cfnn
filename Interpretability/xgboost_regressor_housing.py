import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# 设置随机种子以确保结果可复现
np.random.seed(42)

def load_and_split_data():
    """
    加载、分割并标准化加州住房数据集。
    此函数严格遵循您之前代码中的分割逻辑。
    """
    print("1. 正在加载和处理数据...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # XGBoost的Scikit-Learn接口需要一维的y
    # fetch_california_housing()返回的y已经是正确的形状，无需调整

    # 第一次分割：分出测试集 (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 第二次分割：从大的训练集中分出验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"数据准备完毕。训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    return X_train, y_train, X_val, y_val

if __name__ == '__main__':
    # --- 1. 数据准备 ---
    X_train, y_train, X_val, y_val = load_and_split_data()

    # --- 2. 模型超参数设置 ---
    # 为XGBoost选择一组常用的超参数
    hyperparameters = {
        'objective': 'reg:squarederror', # 回归任务的目标函数
        'n_estimators': 1000,             # 初始设置较多的树，通过早停找到最佳数量
        'learning_rate': 0.05,            # 学习率
        'max_depth': 5,                   # 树的最大深度
        'subsample': 0.8,                 # 训练每棵树时使用的样本比例
        'colsample_bytree': 0.8,          # 训练每棵树时使用的特征比例
        'n_jobs': -1,                     # 使用所有可用的CPU核心
        'random_state': 42,
        'early_stopping_rounds': 10,
    }

    # --- 3. 模型初始化与训练 ---
    print("\n2. 正在初始化并训练XGBoost模型...")
    
    model = xgb.XGBRegressor(**hyperparameters)

    # 使用验证集进行早停 (early stopping)
    # 这可以防止过拟合，并自动找到最佳的 n_estimators
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
         # 如果验证集上的RMSE在10轮内没有改善，则停止训练
        verbose=False # 不打印每一轮的评估结果，保持输出整洁
    )
    
    print("模型训练完成。")

    # --- 4. 在验证集上评估 ---
    print("\n3. 正在验证集上进行评估...")
    
    # 使用训练好的模型进行预测
    y_pred_val = model.predict(X_val)
    
    # 计算RMSE
    rmse_val = root_mean_squared_error(y_val, y_pred_val)

    # --- 5. 结果统计 ---
    print("\n" + "="*50)
    print("最终结果统计")
    print("="*50)
    print(f"在验证集上的RMSE: {rmse_val:.4f}")
    print(f"通过早停找到的最佳迭代次数: {model.best_iteration}")
    print("\n使用的超参数配置:")
    for key, value in hyperparameters.items():
        print(f"  - {key}: {value}")
    print("="*50)
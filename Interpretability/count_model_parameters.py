# count_model_parameters.py

import torch
import sys
sys.path.append("/home/zxc/CodeBase/cofrnet")
from cfnet_regressor import CoFrNetRegressor

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    计算一个 PyTorch 模型中所有可训练参数的总数。
    
    :param model: torch.nn.Module 的实例。
    :return: 可训练参数的总数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(model_path='best_california_housing_regressor_2.pth'):
    """
    主函数，用于加载模型并统计其参数量。
    """
    print(f"正在从 '{model_path}' 加载模型...")
    
    try:
        # 使用 CoFrNetRegressor 类提供的 load_model 方法加载完整的回归器对象
        regressor = CoFrNetRegressor.load_model(model_path)
    except FileNotFoundError:
        print(f"错误: 模型文件 '{model_path}' 未找到。")
        print("请确保模型文件与此脚本位于同一目录，或提供正确的文件路径。")
        sys.exit(1)
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        sys.exit(1)
        
    print("模型加载成功。")
    
    # 从加载的回归器对象中获取 PyTorch 模型本身 (EnsembleResCoFrNet)
    pytorch_model = regressor.ensemble_model
    
    # 调用函数计算参数量
    total_params = count_trainable_parameters(pytorch_model)
    
    print("\n" + "="*40)
    print("模型参数量统计")
    print("="*40)
    print(f"最终采纳的子模型 (CFNet) 数量: {regressor.best_model_size}")
    print(f"模型总的可训练参数量: {total_params:,}") # 使用逗号作为千位分隔符，方便阅读
    print("="*40)

if __name__ == '__main__':
    main()
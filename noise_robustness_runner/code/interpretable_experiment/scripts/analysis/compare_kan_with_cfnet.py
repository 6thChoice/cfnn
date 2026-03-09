"""
对比 KAN 与 CFNet 系列模型在噪声特征干扰实验上的表现
"""

import os
import sys
import json
import glob
from pathlib import Path
RUNNER_BASE = Path(__file__).resolve().parents[4]
BASE_RESULTS_DIR = str(RUNNER_BASE / "results")

# 设置工作目录为项目根目录
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def load_existing_results():
    """加载已有的 CFNet 实验结果"""
    results_dir = BASE_RESULTS_DIR

    # 查找 exp1_robustness 的结果
    exp1_dirs = glob.glob(os.path.join(results_dir, 'exp1_robustness', '2026*'))

    if not exp1_dirs:
        print("未找到 CFNet 实验结果，请先运行 run_exp1.py")
        return None

    # 取最新的结果
    latest_dir = max(exp1_dirs)
    summary_file = os.path.join(latest_dir, 'summary', 'robustness_report.md')

    results = {
        'result_dir': latest_dir,
        'summary_file': summary_file if os.path.exists(summary_file) else None
    }

    return results


def load_kan_results():
    """加载 KAN 实验结果"""
    results_dir = BASE_RESULTS_DIR

    # 查找 KAN 实验结果
    kan_dirs = glob.glob(os.path.join(results_dir, 'exp1_robustness_kan', '2026*'))

    if not kan_dirs:
        print("未找到 KAN 实验结果，请先运行 run_kan_robustness.py")
        return None

    # 取最新的结果
    latest_dir = max(kan_dirs)
    summary_file = os.path.join(latest_dir, 'kan_summary.json')

    kan_data = None
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            kan_data = json.load(f)

    return {
        'result_dir': latest_dir,
        'summary_file': summary_file,
        'data': kan_data
    }


def generate_comparison_report(cfnet_results, kan_results):
    """生成对比报告"""

    report = """# KAN vs CFNet 噪声特征干扰实验对比报告

## 实验设置

- **数据集**: 噪声特征干扰数据集
  - 真实特征: 4个
  - 噪声特征: 4个
  - 冗余特征: 2个
  - 欺骗性特征: 1个
  - 总特征数: 11个
- **训练配置**: 50 epochs, lr=0.001, batch_size=64
- **公平性保证**: KAN 与 CFNet_Standard 参数量对齐

"""

    # KAN 结果
    if kan_results and kan_results['data']:
        data = kan_results['data']
        report += f"""## KAN 实验结果

- **结果目录**: `{kan_results['result_dir']}`
- **测试 MSE**: {data.get('test_mse', 'N/A'):.6f}
- **参数量**: {data.get('parameters', 'N/A'):,}

### 特征选择准确率
"""
        fsa = data.get('feature_selection_accuracy', {})
        for k, v in fsa.items():
            report += f"- {k}: {v}\n"

        report += "\n### 噪声抑制比\n"
        nsr = data.get('noise_suppression_ratio', {})
        for k, v in nsr.items():
            report += f"- {k}: {v:.4f}\n"

    # CFNet 结果提示
    if cfnet_results:
        report += f"""
## CFNet 系列实验结果

- **结果目录**: `{cfnet_results['result_dir']}`
- **详细报告**: `{cfnet_results.get('summary_file', 'N/A')}`

请参考上述路径中的详细报告获取各模型的具体指标。

"""

    # 对比总结
    report += """## 对比说明

### 公平性对比维度

1. **参数量对齐**
   - KAN 的隐藏层维度通过算法自动调整
   - 目标参数量以 CFNet_Standard (depth=4, poly_degree=4) 为锚点
   - 确保模型复杂度相当

2. **数据一致性**
   - 使用相同的数据生成器 (NoisyFeaturesGenerator)
   - 相同的随机种子 (random_seed=42)
   - 相同的训练/验证/测试划分

3. **训练配置一致**
   - 相同的学习率 (0.001)
   - 相同的训练轮数 (50 epochs)
   - 相同的早停耐心值 (15)

### 评估指标

1. **测试 MSE**: 预测精度
2. **特征选择准确率**: 模型能否正确识别真实特征
3. **噪声抑制比**: 真实特征重要性 / 噪声特征重要性

## 结果解读

KAN 使用 B-样条基函数替代传统线性权重，在以下方面表现不同:
- **函数表达能力**: 样条函数提供局部自适应的非线性
- **可解释性**: 每个连接都有可学习的单变量函数
- **训练动态**: 需要不同的优化策略

## 文件位置

- **KAN 实验脚本**: `xai/interpretable/run_kan_robustness.py`
- **KAN 结果**: `interpretable/results/exp1_robustness_kan/`
- **CFNet 实验脚本**: `xai/interpretable/run_exp1.py`
- **CFNet 结果**: `interpretable/results/exp1_robustness/`
"""

    return report


def main():
    """主函数"""
    print("="*80)
    print("KAN vs CFNet 对比报告生成")
    print("="*80)

    # 加载结果
    cfnet_results = load_existing_results()
    kan_results = load_kan_results()

    if not kan_results:
        print("\n错误: 未找到 KAN 实验结果!")
        print("请先运行: python xai/interpretable/run_kan_robustness.py")
        return

    # 生成报告
    report = generate_comparison_report(cfnet_results, kan_results)

    # 保存报告
    report_dir = str(Path(BASE_RESULTS_DIR) / "comparisons")
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, 'kan_vs_cfnet_comparison.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n对比报告已保存: {report_path}")
    print("\n" + "="*80)
    print(report)


if __name__ == '__main__':
    main()
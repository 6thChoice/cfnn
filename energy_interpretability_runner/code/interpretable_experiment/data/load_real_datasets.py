"""
真实数据集加载器
支持多种真实世界数据集
"""

from .base import BaseDataGenerator
from typing import Dict, Any
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class RealDatasetLoader(BaseDataGenerator):
    """真实数据集加载器"""

    SUPPORTED_DATASETS = {
        'energy_efficiency': {
            'openml_id': 42319,
            'target_column': 'Y1',  # 采暖负荷
            'task': 'regression',
            'n_features': 8,
            'feature_names': [
                'X1',  # 相对紧凑度
                'X2',  # 表面积
                'X3',  # 墙面积
                'X4',  # 屋顶面积
                'X5',  # 总高度
                'X6',  # 朝向
                'X7',  # 窗户面积
                'X8'   # 窗户分布
            ],
            # 领域知识：特征重要性评分 (1=低, 2=中, 3=高)
            'domain_importance': {
                'X1': 3,  # 高 - 直接影响热交换
                'X2': 3,  # 高 - 直接影响热交换
                'X3': 2,  # 中 - 间接影响
                'X4': 2,  # 中 - 间接影响
                'X5': 2,  # 中 - 间接影响
                'X6': 1,  # 低 - 仅影响采光
                'X7': 2,  # 中 - 影响散热
                'X8': 1   # 低 - 次要因素
            }
        }
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                - dataset_name: 数据集名称
                - target_column: 目标列名
                - test_split: 测试集比例
                - val_split: 验证集比例
                - random_seed: 随机种子
                - normalize: 是否标准化 (默认 True)
        """
        super().__init__(config)

        self.dataset_name = config['dataset_name']
        self.target_column = config.get('target_column', None)
        self.normalize = config.get('normalize', True)

        # 加载数据集信息
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}. "
                f"Supported datasets: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        self.dataset_info = self.SUPPORTED_DATASETS[self.dataset_name]

        # 数据存储
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = None

    def load_dataset(self):
        """加载数据集"""
        print(f"Loading dataset: {self.dataset_name}")

        # 尝试多种方式加载数据集
        df = None

        runner_base = Path(__file__).resolve().parents[4]
        data_dir = runner_base.parent / "data"
        openml_cache = data_dir / "openml"
        data_dir.mkdir(parents=True, exist_ok=True)
        openml_cache.mkdir(parents=True, exist_ok=True)

        # 方法1: 尝试从 OpenML 加载（使用名称而不是 ID）
        try:
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            dataset = fetch_openml(
                name='energy',
                version=1,
                as_frame=True,
                parser='auto',
                data_home=str(openml_cache)
            )
            df = dataset.frame
            print(f"Loaded from OpenML (name)")
        except Exception as e:
            print(f"OpenML load failed (name): {e}")

        # 方法2: 尝试从 UCI URL 加载
        if df is None:
            try:
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
                df = pd.read_excel(url)
                # 重命名列以匹配预期格式
                df.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2']
                print(f"Loaded from UCI URL")
            except Exception as e:
                print(f"UCI URL load failed: {e}")

        # 方法3: 使用本地 CSV（如果存在）
        if df is None:
            try:
                local_path = str(data_dir / 'energy_efficiency.csv')
                if os.path.exists(local_path):
                    df = pd.read_csv(local_path)
                    print(f"Loaded from local file")
                else:
                    raise FileNotFoundError(f"Local data file not found: {local_path}")
            except Exception as e:
                print(f"Local file load failed: {e}")

        if df is None:
            raise RuntimeError("Failed to load dataset from all sources")

        # 获取特征列
        feature_cols = self.dataset_info['feature_names']

        # 获取目标列
        if self.target_column is None:
            # 使用第一个以 Y 开头的列
            y_cols = [col for col in df.columns if col.startswith('Y')]
            self.target_column = y_cols[0] if y_cols else 'Y1'

        # 提取特征和目标
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[self.target_column].values.astype(np.float32)
        self.feature_names = feature_cols

        print(f"Dataset loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")

    def normalize_data(self):
        """标准化数据"""
        if self.normalize:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
            print(f"Data normalized: mean={self.X.mean():.4f}, std={self.X.std():.4f}")

    def generate_features(self) -> np.ndarray:
        """
        生成特征矩阵

        Returns:
            X: 特征矩阵
        """
        if self.X is None:
            self.load_dataset()
            self.normalize_data()

        return self.X

    def generate_target(self, X: np.ndarray) -> np.ndarray:
        """
        生成目标变量

        Args:
            X: 特征矩阵

        Returns:
            y: 目标变量
        """
        if self.y is None:
            self.load_dataset()

        return self.y

    def split_data(self, X: np.ndarray, y: np.ndarray):
        """
        划分数据集

        与基类不同，这里使用 sklearn 的 train_test_split
        """
        # 先划分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_split,
            random_state=self.seed
        )

        # 从剩余数据中划分出验证集
        val_ratio = self.val_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.seed
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征信息

        Returns:
            feature_info: 特征信息字典
        """
        if self.feature_names is None:
            self.load_dataset()

        domain_importance = self.dataset_info.get('domain_importance', {})

        return {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_types': ['continuous'] * len(self.feature_names),
            'domain_importance': domain_importance,
            'dataset_name': self.dataset_name,
            'target_column': self.target_column,
            'dataset_info': self.dataset_info
        }


def create_real_dataset_loader(config: Dict[str, Any]) -> RealDatasetLoader:
    """
    创建真实数据集加载器

    Args:
        config: 配置字典

    Returns:
        loader: 数据加载器实例
    """
    return RealDatasetLoader(config)

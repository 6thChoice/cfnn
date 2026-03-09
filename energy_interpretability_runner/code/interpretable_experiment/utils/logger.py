"""
日志工具
提供统一的日志记录功能
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(save_dir: str, exp_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    设置日志器

    Args:
        save_dir: 日志保存目录
        exp_name: 实验名称
        level: 日志级别

    Returns:
        logger: 配置好的日志器
    """
    # 创建日志器
    logger = logging.getLogger(exp_name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建日志目录
    os.makedirs(save_dir, exist_ok=True)

    # 日志文件路径
    log_file = os.path.join(save_dir, f'{exp_name}.log')

    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class ProgressLogger:
    """
    进度日志器
    用于记录多阶段实验的进度
    """

    def __init__(self, logger: logging.Logger, total_phases: int):
        """
        Args:
            logger: 日志器
            total_phases: 总阶段数
        """
        self.logger = logger
        self.total_phases = total_phases
        self.current_phase = 0

    def start_phase(self, phase_name: str):
        """开始一个新阶段"""
        self.current_phase += 1
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"[Phase {self.current_phase}/{self.total_phases}] {phase_name}")
        self.logger.info(f"{'='*80}\n")

    def info(self, message: str):
        """记录信息"""
        self.logger.info(message)

    def success(self, message: str):
        """记录成功信息"""
        self.logger.info(f"✓ {message}")

    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(f"⚠ {message}")

    def error(self, message: str):
        """记录错误"""
        self.logger.error(f"✗ {message}")

    def finish(self):
        """完成所有阶段"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"All phases completed!")
        self.logger.info(f"{'='*80}\n")

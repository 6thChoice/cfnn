#!/bin/bash
# 监控方案 1 实验进度

echo "===================================="
echo "方案 1 实验进度监控"
echo "===================================="
echo ""

# 检查进程是否还在运行
if ps -p 4173485 > /dev/null 2>&1; then
    echo "✓ 实验进程正在运行 (PID: 4173485)"
else
    echo "✗ 实验进程已结束"
fi
echo ""

# 显示最新日志
echo "最近日志："
echo "------------------------------------"
tail -50 /tmp/exp1_run.log | grep -E "(Training|Test MSE|Stage|Analyzing|Report)" || tail -20 /tmp/exp1_run.log
echo ""

# 检查结果目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_BASE="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULT_DIR=$(find "${RUNNER_BASE}/results/exp1_robustness" -type d -name "20*" | sort | tail -1)

if [ -n "$RESULT_DIR" ]; then
    echo "结果目录: $RESULT_DIR"
    echo "------------------------------------"
    echo "模型检查点:"
    ls -lh "$RESULT_DIR/checkpoints/" 2>/dev/null || echo "  (暂无)"
    echo ""
    echo "分析结果:"
    ls -lh "$RESULT_DIR/metrics/" 2>/dev/null || echo "  (暂无)"
    echo ""
    echo "报告:"
    ls -lh "$RESULT_DIR/summary/" 2>/dev/null || echo "  (暂无)"
else
    echo "未找到结果目录"
fi

echo ""
echo "===================================="
echo "监控完成"
echo "===================================="

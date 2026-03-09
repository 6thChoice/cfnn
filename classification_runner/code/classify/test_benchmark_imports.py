"""
测试benchmark脚本是否可以正确导入和运行基础功能
"""

import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))



def test_imports():
    """测试所有模块导入"""
    print("Testing imports...")

    try:
        from benchmark_utils import (
            create_model, count_parameters, measure_model_memory,
            measure_inference_memory, benchmark_inference, ConvergenceTracker,
            save_benchmark_results, create_model_with_param_limit
        )
        print("✓ benchmark_utils imported successfully")
    except Exception as e:
        print(f"✗ benchmark_utils import failed: {e}")
        return False

    try:
        from kan_classification_base import KAN, count_parameters
        print("✓ kan_classification_base imported successfully")
    except Exception as e:
        print(f"✗ kan_classification_base import failed: {e}")
        return False

    try:
        from cfnet_complicate import CFNet
        from hybrid import HybridRationalNet
        print("✓ cfnet_complicate and hybrid imported successfully")
    except Exception as e:
        print(f"✗ cfnet_complicate import failed: {e}")
        return False

    return True


def test_model_creation():
    """测试模型创建"""
    print("\nTesting model creation...")

    from benchmark_utils import create_model, count_parameters

    input_dim, output_dim = 40, 3

    models_to_test = [
        ('CFNet', {'depth': 5, 'poly_degree': 3}),
        ('Hybrid', {'num_units': 10, 'unit_degree': 10}),
        ('KAN', {'hidden_dim': 32, 'grid_size': 10, 'spline_order': 3}),
        ('MLP', {'hidden_dim': 128, 'num_layers': 2}),
    ]

    for model_type, config in models_to_test:
        try:
            model, info = create_model(model_type, input_dim, output_dim, config)
            params = count_parameters(model)
            print(f"✓ {model_type}: {params:,} parameters")
        except Exception as e:
            print(f"✗ {model_type} creation failed: {e}")
            return False

    return True


def test_memory_measurement():
    """测试内存测量"""
    print("\nTesting memory measurement...")

    import torch
    from benchmark_utils import create_model, measure_model_memory

    try:
        model, _ = create_model('CFNet', 40, 3, {'depth': 3, 'poly_degree': 3})
        mem_stats = measure_model_memory(model, batch_size=32, input_dim=40, device='cpu')
        print(f"✓ Memory measurement works")
        print(f"  Model params: {mem_stats['model_params_mb']:.4f} MB")
        return True
    except Exception as e:
        print(f"✗ Memory measurement failed: {e}")
        return False


def test_inference_benchmark():
    """测试推理速度基准"""
    print("\nTesting inference benchmark...")

    import torch
    from benchmark_utils import create_model, benchmark_inference

    try:
        model, _ = create_model('CFNet', 40, 3, {'depth': 3, 'poly_degree': 3})
        results = benchmark_inference(
            model, input_dim=40, batch_sizes=[1, 32],
            device='cpu', num_warmup=2, num_iters=5
        )
        print(f"✓ Inference benchmark works")
        for bs, perf in results.items():
            print(f"  Batch {bs}: {perf['samples_per_sec']:.1f} samples/s")
        return True
    except Exception as e:
        print(f"✗ Inference benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convergence_tracker():
    """测试收敛追踪器"""
    print("\nTesting convergence tracker...")

    from benchmark_utils import ConvergenceTracker

    try:
        tracker = ConvergenceTracker()
        for epoch in range(10):
            tracker.record(epoch, train_loss=0.5-epoch*0.04,
                          val_loss=0.6-epoch*0.03, val_acc=0.7+epoch*0.02, lr=0.001)

        epoch_80 = tracker.get_convergence_epoch(0.80)
        speed = tracker.get_avg_improvement_speed(5)

        print(f"✓ Convergence tracker works")
        print(f"  Epoch to 80%: {epoch_80}")
        print(f"  Avg speed: {speed:.4f}")
        return True
    except Exception as e:
        print(f"✗ Convergence tracker failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("CFNet Benchmark Test Suite")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Memory Measurement", test_memory_measurement),
        ("Inference Benchmark", test_inference_benchmark),
        ("Convergence Tracker", test_convergence_tracker),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name:30s}: {status}")

    all_passed = all(success for _, success in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))


if __name__ == "__main__":
    main()
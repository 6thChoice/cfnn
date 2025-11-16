# cfnn
【工作中】连分网络 cfnn

# 现有模型定义
1. 结合了 boost 想法的 cfnn-boost
2. 结合了 moe 想法的 cfnn-boost


# cfnn-moe
```python

from cfnet_regressor_moe_iterative import CoFrNetRegressor_MoE
moe_hparams = {
        'task_name': 'Task',
        'input_dim': 1,
        'output_dim': 1,
        'shallow_depth_per_cofrnet': 4,
        'polynomial_degree': 4,
        'learning_rate_adam': 1e-2,
        'weight_decay': 1e-5,
        'batch_size': 128,
        'epochs_per_model': 150,
        'early_stopping_patience': 15,
        'max_experts': 10,
    }
moe_regressor = CoFrNetRegressor_MoE(moe_hparams)
moe_regressor.train_iterative(
            train_data=(X_train, y_train),
            val_data=(X_train, y_train),
            log_filepath=f"logs/log.json",
            model_save_path=f"models/model.pth"
        )
```

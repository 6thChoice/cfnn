# Data Cache (Submit Codebase)

此目录用于缓存需要联网下载的数据集。默认结构：

- `openml/`：OpenML 数据缓存
- `hf/`：HuggingFace datasets 缓存
- `torchvision/`：TorchVision 数据缓存

运行下载脚本：

```bash
python download_datasets.py
```

你也可以只下载部分数据集，脚本内有开关参数。

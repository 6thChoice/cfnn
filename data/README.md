# Data Cache (Submit Codebase)

此目录用于缓存需要联网下载的数据集。默认结构：

- `openml/`：OpenML 数据缓存
- `hf/`：HuggingFace datasets 缓存
- `torchvision/`：TorchVision 数据缓存

当前仓库已内置一份可直接用于 `classification_runner` 中 CIFAR-10 实验的
`torchvision/cifar10/cifar-10-batches-py/` 数据。

项目内没有找到可直接复用的 OpenML 与 HuggingFace 本地缓存，因此这两部分
在首次运行 `waveform`、`magic`、`credit_card`、`sentiment`、`quora`
相关脚本时仍可能需要联网下载。

运行下载脚本：

```bash
python download_datasets.py
```

你也可以只下载部分数据集，脚本内有开关参数。

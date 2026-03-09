from pathlib import Path

# Optional: comment out datasets you don't need
DOWNLOAD_OPENML = True
DOWNLOAD_HF = True
DOWNLOAD_TORCHVISION = True

DATA_DIR = Path(__file__).resolve().parent
OPENML_CACHE = DATA_DIR / "openml"
HF_CACHE = DATA_DIR / "hf"
TORCHVISION_DIR = DATA_DIR / "torchvision"

for p in [OPENML_CACHE, HF_CACHE, TORCHVISION_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def download_openml():
    from sklearn.datasets import fetch_openml

    # Classification datasets
    fetch_openml(name="waveform-5000", version=1, as_frame=False, parser="liac-arff", data_home=str(OPENML_CACHE))
    fetch_openml(name="MagicTelescope", version=1, as_frame=False, parser="liac-arff", data_home=str(OPENML_CACHE))
    fetch_openml(name="default-of-credit-card-clients", version=1, as_frame=True, parser="auto", data_home=str(OPENML_CACHE))

    # Energy Efficiency (regression)
    fetch_openml(name="energy", version=1, as_frame=True, data_home=str(OPENML_CACHE))


def download_hf():
    from datasets import load_dataset

    load_dataset("imdb", cache_dir=str(HF_CACHE))
    load_dataset("UKPLab/insincere-questions", cache_dir=str(HF_CACHE))


def download_torchvision():
    import torchvision

    torchvision.datasets.CIFAR10(root=str(TORCHVISION_DIR / "cifar10"), train=True, download=True)
    torchvision.datasets.CIFAR10(root=str(TORCHVISION_DIR / "cifar10"), train=False, download=True)


if __name__ == "__main__":
    if DOWNLOAD_OPENML:
        print("Downloading OpenML datasets...")
        download_openml()
    if DOWNLOAD_HF:
        print("Downloading HuggingFace datasets...")
        download_hf()
    if DOWNLOAD_TORCHVISION:
        print("Downloading TorchVision datasets...")
        download_torchvision()

    print("Done.")

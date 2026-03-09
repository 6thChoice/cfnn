from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
SUBMIT_BASE = CODE_DIR.parents[1]
DATA_DIR = SUBMIT_BASE / "data"
OPENML_CACHE = DATA_DIR / "openml"
HF_CACHE = DATA_DIR / "hf"
TORCHVISION_DIR = DATA_DIR / "torchvision"

for p in [DATA_DIR, OPENML_CACHE, HF_CACHE, TORCHVISION_DIR]:
    p.mkdir(parents=True, exist_ok=True)

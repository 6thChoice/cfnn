#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-.venv}"

"${python_bin}" -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.5.1+cu124 \
  torchvision==0.20.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

echo "Environment ready. Activate it with: source ${venv_dir}/bin/activate"

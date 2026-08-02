#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}/experiments:${PWD}/src:${PYTHONPATH:-}"

python figures/create_cfnn_hybrid_training_stability.py
python figures/create_sharp_response_mechanism.py
python figures/create_ai4science_fano_3d_response_recovery.py
python figures/create_ai4science_microstrip_recovery.py

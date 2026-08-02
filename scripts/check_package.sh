#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src experiments figures data
python figures/create_cfnn_hybrid_training_stability.py >/tmp/cfnn_nmi_stability_check.json
python figures/create_sharp_response_mechanism.py >/tmp/cfnn_nmi_sharp_check.json
python figures/create_ai4science_fano_3d_response_recovery.py >/tmp/cfnn_nmi_fano_check.json
python figures/create_ai4science_microstrip_recovery.py >/tmp/cfnn_nmi_microstrip_check.json
echo "Package checks passed."

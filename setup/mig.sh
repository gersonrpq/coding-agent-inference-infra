#!/usr/bin/env bash
# Split the single H100 PCIe into 2 x MIG 3g.40gb (one SGLang worker each).
# Idempotent. Run on the server BEFORE the NVIDIA device plugin starts (MIG mode and
# instances need an idle GPU) and again after every reboot: instances do not persist.
set -euo pipefail

if [[ "$(nvidia-smi -L | grep -c '^GPU ')" -ne 1 ]]; then
  echo "expected exactly one physical GPU; on 2x H100 nodes do not use MIG, skip this script"; exit 1
fi

if [[ "$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader)" != "Enabled" ]]; then
  sudo nvidia-smi -i 0 -mig 1
fi

have=$(nvidia-smi -L | grep -c 'MIG 3g.40gb' || true)
if [[ "$have" -lt 2 ]]; then
  sudo nvidia-smi mig -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -cgi 9,9 -C
fi
nvidia-smi -L

#!/usr/bin/env bash
# Samples the GPU once per second into a CSV until killed (or for SECONDS_TO_RUN seconds).
#   script/gpu_sampler.sh out.csv [seconds]
# With MIG enabled nvidia-smi reports utilization.gpu as [N/A]; power.draw and memory.used are valid and
# power is the load indicator used for the cost report.
set -euo pipefail
out=${1:?usage: gpu_sampler.sh out.csv [seconds]}
secs=${2:-0}
echo "timestamp,power_w,util_gpu,mem_used_mib,temp_c" > "$out"
if [[ "$secs" -gt 0 ]]; then
  timeout "$secs" nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu \
    --format=csv,noheader,nounits -l 1 >> "$out" || true
else
  exec nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu \
    --format=csv,noheader,nounits -l 1 >> "$out"
fi

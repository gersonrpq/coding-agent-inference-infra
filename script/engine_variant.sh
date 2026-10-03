#!/usr/bin/env bash
# Changes SGLang knobs of the LIVE cluster for an engine experiment and waits for both workers (~3 min).
# Run ON THE SERVER. It patches the ConfigMap sglang-config only; the next setup/launch_cluster.sh restores the repo values.
#
#   script/engine_variant.sh RADIX_EVICTION_POLICY=lru
#   script/engine_variant.sh CHUNKED_PREFILL_SIZE=2048
#   script/engine_variant.sh RADIX_EVICTION_POLICY=lfu CHUNKED_PREFILL_SIZE=4096     # back to the repo defaults
#   script/engine_variant.sh show
# Knobs: RADIX_EVICTION_POLICY, CHUNKED_PREFILL_SIZE, MAX_QUEUED_REQUESTS, HICACHE_WRITE_POLICY, MAMBA_FULL_MEMORY_RATIO, MAX_NUM_SEQS.
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
NS=gpu-serving
show() { kubectl -n $NS get cm sglang-config -o jsonpath='{range .data}{end}' >/dev/null; kubectl -n $NS get cm sglang-config -o json | python3 -c "import json,sys; d=json.load(sys.stdin)['data']; [print(f'{k}={v}') for k,v in sorted(d.items())]"; }
[[ "${1:-show}" == "show" ]] && { show; exit 0; }
patch='{"data":{'; sep=''; changed=0
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  cur=$(kubectl -n $NS get cm sglang-config -o jsonpath="{.data.$k}")
  [[ -z "$cur" ]] && { echo "unknown knob $k" >&2; exit 1; }
  [[ "$cur" == "$v" ]] && { echo "$k already $v"; continue; }
  patch+="$sep\"$k\":\"$v\""; sep=','; changed=1; echo "$k: $cur -> $v"
done
patch+='}}'
[[ $changed == 0 ]] && { echo "nothing to change"; exit 0; }
kubectl -n $NS patch cm sglang-config --type merge -p "$patch" >/dev/null
kubectl -n $NS rollout restart deploy/sglang-worker-0 deploy/sglang-worker-1 >/dev/null
kubectl -n $NS rollout status deploy/sglang-worker-0 --timeout=900s
kubectl -n $NS rollout status deploy/sglang-worker-1 --timeout=900s
show | grep -E "RADIX|CHUNKED|MAX_QUEUED|HICACHE_WRITE|MAMBA|MAX_NUM_SEQS"

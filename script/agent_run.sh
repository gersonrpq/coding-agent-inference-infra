#!/usr/bin/env bash
# Runs the agentic generator once per session count and collects what sweep.sh collects (GPU samples, Prometheus snapshot, cost).
# Run ON THE SERVER (it needs kubectl, the repository as the agents' workspace, and .env).
#
#   script/agent_run.sh TAG "12 20" [DURATION] [extra agentgen args...]
#   script/agent_run.sh fin2-agents "12 20" 360
#
# Uses the virtual key LITELLM_KEY_LOADGEN (script/make_keys.py), or the master key if there is none. Output: metrics/runs/TAG/N<k>/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
tag=${1:?TAG}; counts=${2:?"session counts, e.g. \"12 20\""}; duration=${3:-360}; shift $(( $# >= 3 ? 3 : $# ))
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
set -a; source .env; set +a

lite=$(kubectl -n gpu-serving get svc litellm -o jsonpath='{.spec.clusterIP}')
prom=$(kubectl -n gpu-serving get svc prometheus -o jsonpath='{.spec.clusterIP}')
base=metrics/runs/$tag
mkdir -p "$base"
{ echo "tag=$tag generator=agentgen counts=$counts duration=$duration extra=$*"; echo "date=$(date -Is)"; script/cap_variant.sh show; } > "$base/setup.txt"

for n in $counts; do
  dir=$base/N$n
  mkdir -p "$dir"
  echo "== agents N=$n, ${duration}s =="
  script/gpu_sampler.sh "$dir/gpu.csv" & sampler=$!
  set +e
  python3 script/agentgen.py --url "http://$lite:4000" --sessions "$n" --duration "$duration" --warmup "${WARMUP:-60}" \
    --ramp "${RAMP:-30}" --seed "$(( ${SEED:-800} + n ))" --out "$dir" "$@" 2> "$dir/progress.log"
  set -e
  kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true
  python3 script/prom_snapshot.py --prom "http://$prom:9090" --window "$((duration + 10))" --out "$dir/prometheus.json" || true
  python3 script/cost_report.py --run "$dir" --price "${PRICE:-3.29}" > /dev/null 2>&1 || true
  sleep "${COOLDOWN:-30}"
done
python3 script/analyze_runs.py --runs "$base" || true

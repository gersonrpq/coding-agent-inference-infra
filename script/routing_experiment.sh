#!/usr/bin/env bash
# Routing experiment (notes/routing-experiment.md): the same conversations and load under several placement policies.
# Run ON THE SERVER.
#
#   script/routing_experiment.sh TAG N "lb shuffle latency aff affload lb2" [DURATION]
#
# A name may carry a trailing number to repeat a variant (lb2 = lb again, to measure run-to-run noise).
# The cap on in-flight requests is whatever the cluster has (LiteLLM middleware, 12 + K); every variant uses the same seed,
# so the same conversations are replayed. Afterwards: script/router_variant.sh affload (the repo default).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
set -a; source .env; set +a
tag=${1:?TAG}; n=${2:?sessions N}; variants=${3:?"variants"}; duration=${4:-240}; shift $(( $# >= 4 ? 4 : $# ))
lite=$(kubectl -n gpu-serving get svc litellm -o jsonpath='{.spec.clusterIP}')
prom=$(kubectl -n gpu-serving get svc prometheus -o jsonpath='{.spec.clusterIP}')
base=metrics/runs/$tag
mkdir -p "$base"
{ echo "tag=$tag N=$n variants=$variants duration=$duration seed=${SEED:-300}"; echo "date=$(date -Is)"; script/cap_variant.sh show; } > "$base/setup.txt"

for name in $variants; do
  v=${name%%[0-9]*}
  script/router_variant.sh "$v" > "$base/$name.variant.txt"
  sleep 8
  dir=$base/$name; mkdir -p "$dir"
  echo "== $name  (policy $v)  N=$n  ${duration}s"
  script/gpu_sampler.sh "$dir/gpu.csv" & sampler=$!
  set +e
  python3 script/loadgen.py --url "http://$lite:4000" --sessions "$n" --duration "$duration" --warmup "${WARMUP:-60}" \
    --ramp "${RAMP:-20}" --profile "${PROFILE:-paper}" --seed "$(( ${SEED:-300} + n ))" \
    --out "$dir" "$@" 2> "$dir/progress.log"
  set -e
  kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true
  python3 script/prom_snapshot.py --prom "http://$prom:9090" --window "$((duration + 10))" --out "$dir/prometheus.json" || true
  python3 script/cost_report.py --run "$dir" --price "${PRICE:-3.29}" > /dev/null 2>&1 || true
  sleep "${COOLDOWN:-20}"
done
python3 script/analyze_runs.py --runs "$base" --order "$variants" || true

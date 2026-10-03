#!/usr/bin/env bash
# Shows or changes the cap on concurrent requests of the LIVE cluster: LiteLLM's admission middleware, set in
# litellm/config.yaml (`max_in_flight_requests_per_worker` = 12 running + K). Run ON THE SERVER. Restarts LiteLLM (~1 min);
# the next `setup/launch_cluster.sh` puts the repo value back.
#
#   script/cap_variant.sh show        # current cap and the other live switches
#   script/cap_variant.sh 14          # cap = 14 (K = 2, the repo default)
#   script/cap_variant.sh K 4         # same thing written as K: cap = 12 + 4
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
NS=gpu-serving

show() {
  kubectl -n $NS get cm litellm-config -o jsonpath='{.data.config\.yaml}' | grep -E '^\s+(max_in_flight_requests_per_worker|max_queued_requests_per_worker|routing_strategy):'
  kubectl -n $NS get deploy litellm -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}={.value}{"\n"}{end}' | grep -E 'ADMISSION|PLACEMENT|MAX_TOKENS'
}

case "${1:-show}" in
  show) show; exit 0 ;;
  K) cap=$((12 + ${2:?K})) ;;
  *) cap=$1 ;;
esac
[[ "$cap" =~ ^[0-9]+$ ]] || { echo "usage: cap_variant.sh show | <cap> | K <k>" >&2; exit 1; }

tmp=$(mktemp)
kubectl -n $NS get cm litellm-config -o jsonpath='{.data.config\.yaml}' > "$tmp"
sed -i -E "s/^(\s+max_in_flight_requests_per_worker:)\s*[0-9]+/\1 $cap/" "$tmp"
kubectl -n $NS create cm litellm-config --from-file=config.yaml="$tmp" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
rm -f "$tmp"
kubectl -n $NS rollout restart deploy/litellm >/dev/null
kubectl -n $NS rollout status deploy/litellm --timeout=300s >/dev/null
echo "== cap $cap"; show

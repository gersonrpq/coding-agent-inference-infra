#!/usr/bin/env bash
# Switches how requests are assigned to workers on the LIVE cluster (notes/routing-experiment.md). Run ON THE SERVER.
# Restarts LiteLLM only (~1 min). The next setup/launch_cluster.sh restores the repo values.
#
#   script/router_variant.sh lb        # LiteLLM least-busy (the baseline of the routing experiment)
#   script/router_variant.sh shuffle   # LiteLLM simple-shuffle (random)
#   script/router_variant.sh latency   # LiteLLM latency-based-routing
#   script/router_variant.sh affload   # control/place.py affload: session affinity unless the worker is busier by PLACEMENT_SLACK; batch -> least loaded (the repo default)
#     (the experiment-only policies ll / aff / random were removed from the code; their runs stay in metrics/runs/er-*)
#   script/router_variant.sh show
set -euo pipefail
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
NS=gpu-serving
show() {
  echo "routing_strategy: $(kubectl -n $NS get cm litellm-config -o jsonpath='{.data.config\.yaml}' | grep -E '^\s+routing_strategy:' | awk '{print $2}')"
  kubectl -n $NS get deploy litellm -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}={.value}{"\n"}{end}' | grep -E 'PLACEMENT'
}
v=${1:?usage: router_variant.sh lb|shuffle|latency|affload|show}
case "$v" in
  show) show; exit 0 ;;
  lb)      strategy=least-busy;            policy=litellm ;;
  shuffle) strategy=simple-shuffle;        policy=litellm ;;
  latency) strategy=latency-based-routing; policy=litellm ;;
  affload) strategy=least-busy;            policy=affload ;;
  *) echo "unknown variant $v" >&2; exit 1 ;;
esac
tmp=$(mktemp)
kubectl -n $NS get cm litellm-config -o jsonpath='{.data.config\.yaml}' > "$tmp"
sed -i -E "s/^(\s+routing_strategy:)\s*[a-z-]+/\1 $strategy/" "$tmp"
kubectl -n $NS create cm litellm-config --from-file=config.yaml="$tmp" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
rm -f "$tmp"
kubectl -n $NS set env deploy/litellm PLACEMENT_POLICY="$policy" ${PLACEMENT_SLACK:+PLACEMENT_SLACK="$PLACEMENT_SLACK"} >/dev/null
kubectl -n $NS rollout restart deploy/litellm >/dev/null
kubectl -n $NS rollout status deploy/litellm --timeout=300s >/dev/null
echo "== variant $v"; show

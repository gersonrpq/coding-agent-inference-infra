#!/usr/bin/env bash
# Runs the tenant probe on the cluster with the virtual keys (no restart needed: the limits belong to the keys).
# Run ON THE SERVER from ~/final_project, not during a load test. Needs script/make_keys.py to have been run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
set -a; source .env; set +a
: "${LITELLM_KEY_TENANT_ACME:?run script/make_keys.py first}"; : "${LITELLM_KEY_TENANT_BETA:?run script/make_keys.py first}"
export GATEWAY_URL="http://$(kubectl -n gpu-serving get svc litellm -o jsonpath='{.spec.clusterIP}'):4000"
mkdir -p metrics/logs
python3 metrics/probes/tenant_probe.py | tee metrics/logs/tenant_probe.log
echo TENANT_PROBE_DONE

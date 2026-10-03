#!/usr/bin/env bash
# Builds (or updates) the whole application on an existing K3s node.
# Idempotent: safe to re-run; only changed config triggers a restart.
#
# Order (each stage gates the next):
#   namespace -> NVIDIA plugin -> Mooncake -> SGLang workers -> engine smoke test
#   -> Prometheus/Grafana -> LiteLLM -> LiteLLM->SGLang smoke test
#
# Needs: kubectl, envsubst, python3 and a .env with LITELLM_MASTER_KEY and
# SUPERLINKED_API_KEY. K3s itself is installed by setup/lambda_k3s.sh.
#
# Optional env: SGLANG_READY_TIMEOUT (default 3000s, first model download is slow),
#               IMAGE_PULL_TIMEOUT (default 1800s; the SGLang image is ~20 GB and Mooncake uses it),
#               SKIP_SMOKE=1 to skip the two smoke tests.
#
# Fresh node, in order: setup/lambda_k3s.sh (K3s + MIG + device plugin), setup/sync_lambda.sh,
# copy .env to the server, then this script. After a reboot run setup/mig.sh first.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

NS=gpu-serving
SGLANG_READY_TIMEOUT="${SGLANG_READY_TIMEOUT:-3000}"
IMAGE_PULL_TIMEOUT="${IMAGE_PULL_TIMEOUT:-1800}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

step() { printf '\n=================== %s ===================\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v kubectl >/dev/null || die "kubectl not found"
command -v envsubst >/dev/null || die "envsubst not found (apt install gettext-base)"
command -v python3 >/dev/null || die "python3 not found"
[[ -r "$KUBECONFIG" ]] || die "cannot read $KUBECONFIG (run as a user who can, or sudo chmod 644)"
[[ -f .env ]] || die ".env not found (needs LITELLM_MASTER_KEY and SUPERLINKED_API_KEY)"

set -a
# shellcheck disable=SC1091
source .env
set +a
: "${LITELLM_MASTER_KEY:?missing in .env}"
: "${SUPERLINKED_API_KEY:?missing in .env}"

if [[ -z "${GRAFANA_ADMIN_PASSWORD:-}" ]]; then
  GRAFANA_ADMIN_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')
  printf '\nexport GRAFANA_ADMIN_PASSWORD=%s\n' "$GRAFANA_ADMIN_PASSWORD" >> .env
  warn "GRAFANA_ADMIN_PASSWORD was not in .env: generated one and saved it there (Grafana user: admin)"
fi
export GRAFANA_ADMIN_PASSWORD

if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
  POSTGRES_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_hex(16))')   # hex: safe inside a URL
  printf '\nexport POSTGRES_PASSWORD=%s\n' "$POSTGRES_PASSWORD" >> .env
  warn "POSTGRES_PASSWORD was not in .env: generated one and saved it there"
fi
export POSTGRES_PASSWORD

kubectl get nodes >/dev/null || die "cluster not reachable; run setup/lambda_k3s.sh first"

# Applies a manifest and records <tag> in CHANGED when kubectl reports an existing
# object as "configured" (not "created"/"unchanged"), so consumers that only read
# config at start-up can be restarted afterwards.
CHANGED=""
apply_track() {  # apply_track <tag> <kubectl apply args...>
  local tag=$1 out; shift
  out=$(kubectl apply "$@")
  echo "$out"
  if grep -q ' configured$' <<<"$out"; then CHANGED="$CHANGED $tag"; fi
}
changed() { [[ " $CHANGED " == *" $1 "* ]]; }

# Renders a Secret manifest with envsubst and applies it. kubectl reports stringData
# Secrets as "configured" on every run, so change detection uses a hash of the rendered
# manifest kept as an annotation (not reversible; no secret value is stored in clear).
# apply_secret <manifest> <secret name> <tag recorded in CHANGED> <envsubst variable list>
apply_secret() {
  local file=$1 name=$2 tag=$3 vars=$4 manifest hash old
  manifest=$(envsubst "$vars" < "$file")
  hash=$(sha256sum <<<"$manifest" | cut -d' ' -f1)
  old=$(kubectl get secret -n "$NS" "$name" -o jsonpath='{.metadata.annotations.serving-content-hash}' 2>/dev/null || true)
  kubectl apply -f - <<<"$manifest"
  kubectl annotate --overwrite -n "$NS" "secret/$name" "serving-content-hash=$hash"
  if [[ -n "$old" && "$old" != "$hash" ]]; then CHANGED="$CHANGED $tag"; fi
}

# Runs a tiny chat-completion request from inside a pod (no secrets leave the pod).
# smoke_chat <deployment> <url> <model> [env var holding a bearer token]
smoke_chat() {
  local deploy=$1 url=$2 model=$3 key_env=${4:-} py
  py=$(kubectl exec -n "$NS" "deploy/$deploy" -- sh -c 'command -v python3 || command -v python')
  kubectl exec -i -n "$NS" "deploy/$deploy" -- "$py" - "$url" "$model" "$key_env" <<'PY'
import json, os, sys, urllib.request
url, model, key_env = sys.argv[1:4]
headers = {"Content-Type": "application/json"}
if key_env:
    headers["Authorization"] = "Bearer " + os.environ[key_env]
body = {"model": model, "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with the word ok."}]}
req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
r = json.load(urllib.request.urlopen(req, timeout=180))
print("  response ok: finish_reason=%s usage=%s" % (r["choices"][0]["finish_reason"], r.get("usage")))
PY
}

# ---------------------------------------------------------------------------
# 1. Namespace
# ---------------------------------------------------------------------------
step "1. Namespace"
kubectl apply -f cluster/namespace.yaml

# ---------------------------------------------------------------------------
# 2. NVIDIA device plugin
# ---------------------------------------------------------------------------
step "2. NVIDIA device plugin"
kubectl apply -f cluster/nvidia/device-plugin.yaml
echo "waiting for the node to advertise >= 2 nvidia.com/gpu ..."
for _ in $(seq 1 60); do
  gpus=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)
  [[ "${gpus:-0}" -ge 2 ]] && break
  sleep 5
done
[[ "${gpus:-0}" -ge 2 ]] || die "node advertises ${gpus:-0} nvidia.com/gpu, need 2 (is the device plugin running?)"
echo "node advertises $gpus nvidia.com/gpu"
if command -v nvidia-smi >/dev/null; then
  physical=$(nvidia-smi -L | wc -l)
  if [[ "$physical" -ne "$gpus" ]]; then
    [[ "$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | head -1)" == "Enabled" ]] || warn "$physical physical GPU(s) vs $gpus advertised and MIG is off: run setup/mig.sh first"
  fi
fi

# ---------------------------------------------------------------------------
# 2b. PostgreSQL (LiteLLM virtual keys, limits and spend)
# ---------------------------------------------------------------------------
step "2b. PostgreSQL"
apply_secret cluster/postgres/secret.yaml postgres-secrets postgres '${POSTGRES_PASSWORD}'
kubectl apply -f cluster/postgres/postgres.yaml
kubectl rollout status -n "$NS" statefulset/postgres --timeout="${IMAGE_PULL_TIMEOUT}s"

# ---------------------------------------------------------------------------
# 3. Mooncake master (SGLang's HiCache L3 connects to it on startup)
# ---------------------------------------------------------------------------
step "3. Mooncake master"
kubectl apply -f cluster/mooncake/master.yaml
kubectl rollout status -n "$NS" deploy/mooncake-master --timeout="${IMAGE_PULL_TIMEOUT}s"

# ---------------------------------------------------------------------------
# 4. SGLang workers + services, then engine smoke test
# ---------------------------------------------------------------------------
step "4. SGLang workers"
apply_track sglangcfg -f cluster/sglang/config.yaml
kubectl apply -f cluster/sglang/services.yaml
apply_track sglangpod -f cluster/sglang/worker-0.yaml
apply_track sglangpod -f cluster/sglang/worker-1.yaml
if changed sglangpod; then
  warn "SGLang pod template changed: workers use Recreate, so they restart (model reload)."
elif changed sglangcfg; then
  # The ConfigMap is read as env at container start and does not change the pod template.
  warn "sglang-config changed: restarting both workers (model reload)."
  kubectl rollout restart -n "$NS" deploy/sglang-worker-0 deploy/sglang-worker-1
fi
echo "waiting for the workers to load the model (up to ${SGLANG_READY_TIMEOUT}s) ..."
kubectl rollout status -n "$NS" deploy/sglang-worker-0 --timeout="${SGLANG_READY_TIMEOUT}s"
kubectl rollout status -n "$NS" deploy/sglang-worker-1 --timeout="${SGLANG_READY_TIMEOUT}s"

if [[ "$SKIP_SMOKE" != "1" ]]; then
  step "5. Engine smoke test (direct to each SGLang worker)"
  model=$(kubectl get cm -n "$NS" sglang-config -o jsonpath='{.data.MODEL_NAME}')
  echo "model: $model"
  echo "worker 0:"; smoke_chat sglang-worker-0 "http://localhost:30000/v1/chat/completions" "$model"
  echo "worker 1:"; smoke_chat sglang-worker-1 "http://localhost:30001/v1/chat/completions" "$model"
fi

# "Ready" (model loaded, port open) is not "warm" (lazy kernels, empty prefix cache). LiteLLM is applied only after
# the workers pass this gate, so the gateway never routes to a cold replica. SKIP_WARM=1 skips it.
if [[ "${SKIP_WARM:-0}" != "1" ]]; then
  step "5b. Warm-up gate (cold vs warm TTFT per worker)"
  model=$(kubectl get cm -n "$NS" sglang-config -o jsonpath='{.data.MODEL_NAME}')
  python3 script/warm_workers.py --model "$model" --json "metrics/probes/warm_workers.json" \
    || python3 script/warm_workers.py --model "$model" --json "metrics/probes/warm_workers.json" \
    || { echo "workers did not reach a stable warm TTFT; not exposing them (SKIP_WARM=1 to override)"; exit 1; }
fi

# ---------------------------------------------------------------------------
# 6. Prometheus & Grafana
# ---------------------------------------------------------------------------
# Prometheus mounts litellm-secrets (the master key it scrapes LiteLLM with), so the Secret must exist
# before it: creating it only in the LiteLLM stage left Prometheus in ContainerCreating on a fresh node.
apply_secret cluster/litellm/secret.yaml litellm-secrets litellm '${LITELLM_MASTER_KEY} ${SUPERLINKED_API_KEY} ${POSTGRES_PASSWORD}'

step "6. Prometheus"
python3 monitoring/build_dashboards.py   # keeps grafana-dashboards.yaml in sync with its generator
apply_track prometheus -f cluster/monitoring/prometheus-config.yaml
apply_track prometheus -f cluster/monitoring/alert-rules.yaml
kubectl apply -f cluster/monitoring/prometheus.yaml
# The config is mounted with subPath, so a changed ConfigMap needs a pod restart.
# Note: Prometheus storage is emptyDir, a restart drops its history.
if changed prometheus; then kubectl rollout restart -n "$NS" deploy/prometheus; fi

step "6. Grafana"
apply_secret cluster/monitoring/grafana-secret.yaml grafana-secrets grafana '${GRAFANA_ADMIN_PASSWORD}'
apply_track grafana -f cluster/monitoring/grafana-datasource.yaml
apply_track grafana -f cluster/monitoring/grafana-dashboard-provider.yaml
kubectl apply -f cluster/monitoring/grafana-dashboards.yaml   # dashboards reload by themselves
kubectl apply -f cluster/monitoring/grafana.yaml
if changed grafana; then kubectl rollout restart -n "$NS" deploy/grafana; fi
kubectl apply -f cluster/monitoring/services.yaml
kubectl rollout status -n "$NS" deploy/prometheus --timeout="${IMAGE_PULL_TIMEOUT}s"
kubectl rollout status -n "$NS" deploy/grafana --timeout="${IMAGE_PULL_TIMEOUT}s"

# ---------------------------------------------------------------------------
# 7. LiteLLM (gateway + control plane callbacks), then end-to-end smoke test
# ---------------------------------------------------------------------------
step "7. LiteLLM"

# control/inspect.py is mounted as security_inspect.py (that is the module name
# referenced by litellm/config.yaml); the rest keep their names.
sec_out=$(kubectl create configmap litellm-security -n "$NS" \
  --from-file=security_inspect.py=control/inspect.py \
  --from-file=admission.py=control/admission.py \
  --from-file=fleet_state.py=control/fleet_state.py \
  --from-file=place.py=control/place.py \
  --from-file=__init__.py=control/__init__.py \
  --dry-run=client -o yaml | kubectl apply -f -)
echo "$sec_out"
if grep -q ' configured$' <<<"$sec_out"; then CHANGED="$CHANGED litellm"; fi

apply_track litellm -f cluster/litellm/config.yaml
kubectl apply -f cluster/litellm/deployment.yaml
kubectl apply -f cluster/litellm/service.yaml
# Env vars, mounted ConfigMaps and Python modules are only read at process start.
if changed litellm; then kubectl rollout restart -n "$NS" deploy/litellm; fi
kubectl rollout status -n "$NS" deploy/litellm --timeout="${IMAGE_PULL_TIMEOUT}s"

if [[ "$SKIP_SMOKE" != "1" ]]; then
  step "8. LiteLLM -> SGLang smoke test (through the gateway, with auth)"
  smoke_chat litellm "http://localhost:4000/v1/chat/completions" "qwen-coding-local" LITELLM_MASTER_KEY
fi

# ---------------------------------------------------------------------------
step "Done"
kubectl get pods -n "$NS" -o wide
cat <<'EOF'

Next:
  setup/start_forward.sh    # grafana :3000, prometheus :9090, litellm :4000
  (after any LiteLLM/Prometheus restart, stop_forward.sh + start_forward.sh: forwards stick to the old pod)
EOF

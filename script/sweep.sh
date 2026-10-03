#!/usr/bin/env bash
# Runs the load generator once per session count and collects everything for each run.
# Run ON THE SERVER (it needs kubectl, nvidia-smi and the .env with LITELLM_MASTER_KEY).
#
#   script/sweep.sh TAG PROFILE "8 12 16 20" [DURATION] [extra loadgen args...]
#   script/sweep.sh e1-knee paper "8 12 16 20 24 32 48" 480
#   script/sweep.sh e3-priority paper "16 24" 480 --batch-share 0.2
#
# Per run, in metrics/runs/TAG/N<k>/: records.jsonl, summary.json (verdict), config.json, gpu.csv, cost.json,
# prometheus.json. An idle GPU baseline is taken once into metrics/runs/TAG/idle.csv.
# Each N uses a different --seed so a run never re-sends the conversations of the previous one (they could still
# be in the cache and inflate the hit rate); the shared system prompt is identical on purpose, as in production.
# Stops early after STOP_AFTER_FAILS (default 2) consecutive runs that fail the verdict: higher N would only repeat the collapse.
# Env: STOP_AFTER_FAILS, SEED (base, default 100), PRICE (USD/h, default 3.29), COOLDOWN (s between runs, default 45), WARMUP (default 120).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag=${1:?TAG}; profile=${2:?PROFILE light|paper}; counts=${3:?"session counts, e.g. \"8 12 16\""}
duration=${4:-480}; shift $(( $# >= 4 ? 4 : $# ))
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
set -a; source .env; set +a
: "${LITELLM_MASTER_KEY:?missing in .env}"

lite=$(kubectl -n gpu-serving get svc litellm -o jsonpath='{.spec.clusterIP}')
prom=$(kubectl -n gpu-serving get svc prometheus -o jsonpath='{.spec.clusterIP}')
base=metrics/runs/$tag
mkdir -p "$base"
{
  echo "tag=$tag profile=$profile counts=$counts duration=$duration extra=$*"
  echo "date=$(date -Is)"
  kubectl -n gpu-serving get cm sglang-config -o jsonpath='{.data}'; echo
  kubectl -n gpu-serving get deploy litellm -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}={.value}{"\n"}{end}' | grep -E 'FLEET|BATCH|GATEWAY|ADMISSION'
} > "$base/setup.txt"

echo "== idle GPU baseline (30 s) =="
script/gpu_sampler.sh "$base/idle.csv" 30

fails=0
for n in $counts; do
  dir=$base/N$n
  mkdir -p "$dir"
  echo "== N=$n sessions, profile=$profile, ${duration}s =="
  script/gpu_sampler.sh "$dir/gpu.csv" & sampler=$!
  set +e
  python3 script/loadgen.py --url "http://$lite:4000" --sessions "$n" --duration "$duration" \
    --warmup "${WARMUP:-120}" --profile "$profile" --seed "$(( ${SEED:-100} + n ))" --out "$dir" "$@" 2> "$dir/progress.log"
  rc=$?
  set -e
  kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true
  python3 script/prom_snapshot.py --prom "http://$prom:9090" --window "$((duration + 10))" --out "$dir/prometheus.json" || true
  python3 script/cost_report.py --run "$dir" --price "${PRICE:-3.29}" --idle-csv "$base/idle.csv" > /dev/null || true
  python3 - "$dir" <<'PY'
import json, sys
d = sys.argv[1]
s = json.load(open(d + "/summary.json")); v = s.get("verdict", {})
i = s["classes"].get("interactive", {})
c = {}
try: c = json.load(open(d + "/cost.json"))
except Exception: pass
print(f"   N={s['sessions']:>3} pass={v.get('pass')} ttft p50/p99={i.get('ttft_p50') and round(i['ttft_p50'],1)}/{i.get('ttft_p99') and round(i['ttft_p99'],1)}s "
      f"shed={i.get('shed_rate')} {i.get('shed_by_reason')} served/min={i.get('served_per_min')} "
      f"$/1Mout={c.get('usd_per_1M_completion_tokens')} avgW={c.get('avg_power_w')}")
PY
  echo "   (exit code of loadgen: $rc; 0 = verdict pass, 2 = fail)"
  if [[ "$rc" == "0" ]]; then fails=0; else fails=$((fails + 1)); fi
  if [[ "$fails" -ge "${STOP_AFTER_FAILS:-2}" ]]; then
    echo "   stopping: $fails consecutive failing runs (set STOP_AFTER_FAILS to change)"; break
  fi
  sleep "${COOLDOWN:-45}"
done
echo "done: $base"

#!/usr/bin/env bash
# Saves the evidence the course lists under metrics/: the scrapes, the hop counters, the eviction counters and the
# 429 and 503 counts. Run ON THE SERVER (read-only, safe during a run):  script/collect_evidence.sh [TAG]
# Output: metrics/evidence/<TAG>/ (default TAG = date).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
set -a; source .env; set +a
NS=gpu-serving
tag=${1:-$(date +%Y%m%d-%H%M)}
out=metrics/evidence/$tag
mkdir -p "$out"
lite=$(kubectl -n $NS get svc litellm -o jsonpath='{.spec.clusterIP}')

curl -s -m 15 -H "Authorization: Bearer $LITELLM_MASTER_KEY" "$lite:4000/metrics/" > "$out/litellm_metrics_full.txt"
grep -E '^(orch_|litellm_admission_)' "$out/litellm_metrics_full.txt" | grep -v '_created' > "$out/gateway_orch_metrics.txt"
for w in 0 1; do
  port=$((30000 + w))
  curl -s -m 15 "localhost:$port/metrics" | grep -E '^sglang:(num_running_reqs|num_queue_reqs|num_used_tokens|token_usage|cache_hit_rate|num_retracted_reqs|evicted_tokens_total|hicache_|backuped_tokens_total|storage_prefetch|prefill_effective_tokens_total|time_to_first_token_seconds_(sum|count))' > "$out/sglang_worker$w.txt"
done
# eviction counters per worker (engine's own)
grep -hE 'evicted_tokens_total|hicache_dropped_tokens_total' "$out"/sglang_worker*.txt > "$out/evict_count.txt" || true
# hops: placement counts a move of a session from one worker to the other (orch_hops_*); they are in gateway_orch_metrics.txt
# 429 and 503: counts by reason, from the gateway counters
{ echo "# orch_requests_shed_total (status_code 429 = tenant limits, 503 = capacity); LiteLLM's own proxy status counters below"
  grep -E '^orch_requests_shed_total' "$out/litellm_metrics_full.txt" | grep -v ' 0.0$' || true
  grep -E '^litellm_proxy_total_requests_metric_total' "$out/litellm_metrics_full.txt" | grep -E 'status_code="(429|503|408|500)"' || true
} > "$out/status_429_503.txt"
echo "saved to $out:"; wc -l "$out"/*.txt | sed 's/^/  /'

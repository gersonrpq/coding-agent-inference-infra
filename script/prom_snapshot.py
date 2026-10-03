#!/usr/bin/env python3
"""Reads what the cluster itself measured during a run (Prometheus, over the run window) into a JSON file.

    python3 script/prom_snapshot.py --prom http://10.43.241.86:9090 --window 480 --out snapshot.json

Window is "the last N seconds", so run it right after the load generator ends. It includes the warmup.
"""
import argparse
import json
import urllib.parse
import urllib.request

QUERIES = {
    "admitted": 'sum by (priority_class) (increase(litellm_orch_requests_admitted_total[{w}s]))',
    "shed_by_reason": 'sum by (reason, priority_class) (increase(litellm_orch_requests_shed_total[{w}s]))',
    "gateway_in_flight_max": 'max(max_over_time(litellm_litellm_admission_admitted_requests[{w}s]))',
    "gateway_ttft_p99_s": 'histogram_quantile(0.99, sum by (le, priority_class) (increase(litellm_orch_request_ttft_seconds_bucket[{w}s])))',
    "engine_ttft_p99_s": 'histogram_quantile(0.99, sum by (le, worker) (increase(sglang:time_to_first_token_seconds_bucket[{w}s])))',
    "engine_queue_max": 'max by (worker) (max_over_time(sglang:num_queue_reqs{{priority=""}}[{w}s]))',
    "engine_running_max": 'max by (worker) (max_over_time(sglang:num_running_reqs{{priority=""}}[{w}s]))',
    "kv_used_max": 'max by (worker) (max_over_time(sglang:full_token_usage[{w}s]))',
    "retracted_max": 'max by (worker) (max_over_time(sglang:num_retracted_reqs[{w}s]))',
    "prefill_tokens_by_source": 'sum by (mode, worker) (increase(sglang:prefill_effective_tokens_total[{w}s]))',
    "placement_by_worker": 'sum by (policy, worker, priority_class) (increase(litellm_orch_placement_total[{w}s]))',
    "placement_sessions": 'sum by (result) (increase(litellm_orch_placement_session_total[{w}s]))',
    "hicache_backup_tokens": 'sum by (pool, worker) (increase(sglang:hicache_backup_tokens_total[{w}s]))',
}


def query(base, expr):
    url = base + "/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    if data.get("status") != "success":
        raise RuntimeError(data)
    return [{"labels": x["metric"], "value": float(x["value"][1])} for x in data["data"]["result"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prom", required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = {}
    for name, expr in QUERIES.items():
        try:
            result[name] = query(a.prom, expr.format(w=a.window))
        except Exception as e:  # keep going: one wrong metric name must not lose the others
            result[name] = {"error": repr(e)[:200]}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print("snapshot written:", a.out)


if __name__ == "__main__":
    main()

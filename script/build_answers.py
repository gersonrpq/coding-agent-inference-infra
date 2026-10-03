#!/usr/bin/env python3
"""Regenerates the section "Answers with evidence" of ARCHITECTURE.md from the saved scrapes and runs (nothing is typed by hand).

    python3 script/build_answers.py            # rewrites the text between the markers ANSWERS:BEGIN / ANSWERS:END in ARCHITECTURE.md

Sources: metrics/evidence/final2 (gateway and engine scrape after the simplification, end of the agent and regime-B runs),
final4, final5 and final6, metrics/logs/*.log (probes), metrics/runs/*/summary.json and prometheus.json, cluster/ manifests.
"""
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = lambda *p: os.path.join(ROOT, "metrics", "evidence", *p)
L = lambda n: os.path.join(ROOT, "metrics", "logs", n)
R = lambda *p: os.path.join(ROOT, "metrics", "runs", *p)
NOISE = re.compile(r'(engine_type|model_name|moe_ep_rank|pp_rank|tp_rank|dp_rank|cache_type|attn_cp_rank|attn_cp_size|pp_size|pid|hashed_api_key|client_ip|api_provider|end_user|model_id|org_alias|team|team_alias|user|user_email|requested_model|status_code_class)="[^"]*",?')


def lines(path, *patterns, limit=40):
    out = []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#") or "_created" in line or "_bucket" in line:
            continue
        if any(re.search(p, line) for p in patterns):
            line = NOISE.sub("", line).replace(",}", "}").replace("{,", "{")
            out.append(line[:200])
    return out[:limit]


def block(ls):
    return "```text\n" + "\n".join(ls) + "\n```\n"


def nonzero(ls):
    """Drop the counters that are still 0 and the series of mechanisms that were removed (ttft_cut)."""
    return [l for l in ls if "ttft_cut" not in l and not (l.startswith("orch_requests_shed_total") and l.endswith(" 0.0"))]


def per_worker(name, *patterns):
    out = []
    for w in (0, 1):
        out.append(f"# worker {w}")
        out += lines(os.path.join(f2, f"sglang_worker{w}.txt"), *patterns)
    return out


def soak_series(name):
    d = json.load(open(E("final-soak", name + ".json")))["data"]["result"]
    return {r["metric"].get("worker", "all"): max(float(v) for _, v in r["values"]) for r in d}


def value(path, name_re):
    for line in open(path):
        m = re.match(name_re + r" ([0-9.eE+-]+)$", line.strip())
        if m:
            return float(m.group(1))
    return 0.0


def summary(tag, n, cls="interactive"):
    return json.load(open(R(tag, f"N{n}", "summary.json")))["classes"][cls]


def prom(tag, n):
    return json.load(open(R(tag, f"N{n}", "prometheus.json")))


S = []
add = S.append
f2, f4, f6 = E("final2"), E("final4"), E("final6")

# ---------------------------------------------------------------- 1
modes = {}
for w in (0, 1):
    for m in ("input", "device_hit", "host_hit", "storage_hit"):
        modes[m] = modes.get(m, 0) + value(os.path.join(f2, f"sglang_worker{w}.txt"), r'sglang:prefill_effective_tokens_total\{[^}]*mode="%s"[^}]*\}' % m)
tot = sum(modes.values())
add("### 1. What is the app, and which tokens are shared vs unique?\n")
add("**The app** is a tool-using coding agent, track B: **pi** (the real client, `app/`), a think -> tool -> observe -> answer loop whose context grows every step. **Shared across sessions:** the system prompt and the tool schemas (and, for `agentgen.py`, the project instructions): about 6-10K identical tokens, the prefix the radix cache serves. **Unique per session:** the history and the tool results (48 % and 28 % of a production prompt in the paper). Where the prompt tokens came from, cumulative since the workers started (every run and probe, cold starts and regime B included; `sglang:prefill_effective_tokens_total`):\n")
add(block(per_worker("p", r"prefill_effective_tokens_total")))
add("Of %.1f M prompt tokens, **%.0f %% were read from the GPU radix cache, %.0f %% from host RAM, %.0f %% from Mooncake and only %.0f %% were computed again**: the history is reused, the new tail is what costs. (`metrics/evidence/final2/`)\n" % (
    tot / 1e6, 100 * modes["device_hit"] / tot, 100 * modes["host_hit"] / tot, 100 * modes["storage_hit"] / tot, 100 * modes["input"] / tot))

# ---------------------------------------------------------------- 2
add("### 2. What dies at guardrails vs admit vs place vs queue?\n")
add("| Layer | What dies there | Status | Counter |\n| --- | --- | --- | --- |\n"
    "| guardrails (`inspect.py`) | wrong model, role, content part, tool schema, `max_tokens` <= 0, `n` > 1, priority out of range | 400 / 403 / 413 (fail-closed: 503) | `orch_overflow_decisions_total{status=\"400\"}` (stay) |\n"
    "| the cap (LiteLLM middleware) | the 15th request in flight | 503 `queue_full`, in 0.0 s | `litellm_admission_rejected_requests_total{reason=\"queue_full\"}` |\n"
    "| admit (`should_shed`) | a new prefix at KV >= 95 %, batch while the tail is bad or the last places are reserved | 503 `kv_pressure` / `batch_pressure` / `batch_share` | `orch_requests_shed_total{reason}` |\n"
    "| tenant key | a key over its own limit | 429 (never overflows) | `litellm_proxy_total_requests_metric_total{api_key_alias,status_code=\"429\"}` |\n"
    "| place (`pick`) | no worker answers its metrics | 503 `no_healthy_worker` | `orch_requests_shed_total{reason=\"no_healthy_worker\"}` |\n"
    "| queue / engine | nothing is refused after admission: at most K = 2 wait in SGLang's priority queue; the engine did not retract in any run | | `sglang:num_queue_reqs`, `sglang:num_retracted_reqs` |\n")
add("Scrape after the agent, regime-B and batch runs (`metrics/evidence/final2/`):\n")
add(block(nonzero(lines(os.path.join(f2, "gateway_orch_metrics.txt"), r"orch_requests_shed_total", r"orch_overflow_decisions_total", r"litellm_admission_rejected"))))

# ---------------------------------------------------------------- 3
sk = prom("fin-soak", 16)
add("### 3. Where do I prevent work that will time out?\n")
add("At the door, with a **cap of 12 + K = 14 requests in flight** (LiteLLM's admission middleware; K = 2 may wait in SGLang's priority queue). The next request gets a 503 in 0.0 s (`metrics/logs/cap_probe.log`) instead of a long wait; during the 15-minute soak the engine queue never went above 2 per worker and the interactive TTFT p99 was **%.1f s** against the 20 s SLO. `timeout_queue` (an estimate from stale gauges) is implemented and off; a **first-token cut was built, verified and then removed** because LiteLLM's `stream_timeout` is a read timeout of the whole stream and SGLang is silent while a tool call's arguments are generated (decisions 59, 61; `metrics/probes/gap_probe.py`).\n" % summary("fin-soak", 16)["ttft_p99"])
add(block(open(L("cap_probe.log")).read().strip().splitlines()[:9]))
add("Engine queue maximum during the soak (`metrics/runs/fin-soak/N16/prometheus.json`): " + ", ".join(f"worker {x['labels']['worker']}: {int(x['value'])}" for x in sk["engine_queue_max"]) + ". Figure: `plots/final_soak.png`.\n")

# ---------------------------------------------------------------- 4
add("### 4. Where do I protect KV?\n")
add("**By sizing first:** the pool of a worker holds 435,199 tokens (6.64 sequences of 64K) and the engine runs at most `MAX_NUM_SEQS = 6`, so running sequences cannot take more than 88 % of it. **At the door:** `kv_pressure` refuses a *new* prefix when the fleet KV is >= 95 % (a cached prefix costs no new KV). **After admission:** the engine retracts the least urgent request (`--retraction-policy priority`); the gateway never preempts. The probe puts 12 requests of ~60K tokens in at once (`metrics/probes/kv_probe.py`):\n")
add("```text\n" + "\n".join(l[:190] for l in open(L("kv_probe_default.log")).read().splitlines() if l.startswith(("KV used", "long requests"))) + "\n```\n")
add("The KV peaks at 83 %, with **0 retractions**: the 95 % threshold cannot be reached by running sequences, and the engine does not need to preempt. To see the shed path, the threshold was lowered to 0.70 on the deployment (the gateway decides on a snapshot polled every 15 s, so the requests are sent 20 s after the KV filled):\n")
add("```text\n" + "\n".join(l[:200] for l in open(L("kv_probe_0.70.log")).read().splitlines() if l.startswith(("new-prefix", "reused-prefix", "KV used", "extra requests", "orch_requests_shed_total{priority_class=\"interactive\""))) + "\n```\n")

# ---------------------------------------------------------------- 5
b4, b0 = summary("fin4-batch", 20), summary("fin4-batch", 20, "batch")
c = open(os.path.join(f2, "gateway_orch_metrics.txt")).read()
def mean(cls):
    s = re.search(r'orch_request_ttft_seconds_sum\{priority_class="%s"\} ([0-9.eE+-]+)' % cls, c)
    n = re.search(r'orch_request_ttft_seconds_count\{priority_class="%s"\} ([0-9.eE+-]+)' % cls, c)
    return float(s.group(1)) / float(n.group(1))
add("### 5. Where do I prioritize interactive traffic? (p99 spread)\n")
add("In three places: the engine orders its own waiting queue by `priority` (`--enable-priority-scheduling`, forwarded in `extra_body.priority`); admission refuses batch first (`batch_pressure`: fleet TTFT p99 > 4x p50 and > 10 s) and keeps the last 4 places for interactive calls (`batch_share`: batch enters only while fewer than 10 requests of any class are in flight). With 20 %% batch sessions at N = 20 (`fin4-batch`): interactive **p99 %.1f s, %.1f %% refused**; batch p99 %.1f s, %.1f %% refused. Mean TTFT by class over the whole scrape: interactive **%.2f s**, batch **%.2f s** (`orch_request_ttft_seconds`). Dashboard 03 has the p99 spread panel. Figure: `plots/final_batch.png`.\n" % (b4["ttft_p99"], 100 * b4["shed_rate"], b0["ttft_p99"], 100 * b0["shed_rate"], mean("interactive"), mean("batch")))
add(block(nonzero(lines(os.path.join(E("final4"), "gateway_orch_metrics.txt"), r'orch_requests_shed_total\{[^}]*batch', r'orch_request_ttft_seconds_(sum|count)'))))

# ---------------------------------------------------------------- 6
add("### 6. Where do I stop one tenant from owning the GPU?\n")
add("Each tenant or client has its own **LiteLLM virtual key** (PostgreSQL) with its own limits (`max_parallel_requests`, tpm, rpm); an over-limit key gets a 429, which never overflows. `tenant-acme` is limited to 3 in flight and sends 6 at once while `tenant-beta` works (`metrics/probes/tenant_probe.py`):\n")
add(block(open(L("tenant_probe.log")).read().strip().splitlines()[:16]))

# ---------------------------------------------------------------- 7
add("### 7. Where do I hop, and what is not copied?\n")
add("Placement (`place.py`) knows the worker that served the session's previous call and records a **hop** when it moves the session (src != dst). The gateway moves nothing: SGLang's hierarchical cache lets the destination read the prefix from host RAM or Mooncake instead of recomputing it (2.28 s against 4.48 s cold for a 47K prefix, probe 5). **Not copied:** the weights, the decode state of running requests, the sampling parameters and the scheduler state; for this hybrid model a prefix is reusable on the other worker only once its KV pages *and* a recurrent-state snapshot are in Mooncake. Same worker (`kept`): nothing to do.\n")
add(block(lines(os.path.join(f2, "gateway_orch_metrics.txt"), r"orch_hops_total", r"orch_hop_tokens_total", r"orch_placement_session_total", r"orch_placement_total")))
add("Engine side of the same events (worker 0): " + "; ".join(l.split("{")[0].split(":")[1] + " " + l.rsplit(" ", 1)[1] for l in lines(os.path.join(f2, "sglang_worker0.txt"), r"storage_prefetch_hit_tokens_total", r"hicache_backup_tokens_total\{[^}]*pool=\"kv\"")) + ".\n")

# ---------------------------------------------------------------- 8
add("### 8. Where do I evict, and what becomes a ghost if I skip it?\n")
add("Eviction happens in the engine, tier by tier (GPU radix cache `lru` -> host RAM -> Mooncake, whose master evicts above 95 % of its pool). We count it; we do not drive it. **Ghosts:** (a) the gateway's memory of seen prefixes (30 min) can say \"cached\" for a prefix the engine already evicted, so `kv_pressure` lets in a request that costs a full prefill; (b) a Mooncake key whose owner restarted. `hicache_dropped_tokens_total` counts what was evicted without a lower-tier copy:\n")
ev = []
for w, ls in enumerate([lines(os.path.join(f2, "sglang_worker0.txt"), r"evicted_tokens_total", r"hicache_dropped_tokens_total"), lines(os.path.join(f2, "sglang_worker1.txt"), r"evicted_tokens_total", r"hicache_dropped_tokens_total")]):
    ev += [f"# worker {w}"] + [l.replace("{}", "") for l in ls]
add(block(ev))

# ---------------------------------------------------------------- 9
w0 = open(os.path.join(ROOT, "cluster", "sglang", "worker-0.yaml")).read()
flags = re.findall(r"^\s+- (--[a-z0-9-]+)\s*(?:#.*)?$", w0, re.M)
add("### 9. Where does the engine scheduler sit vs my admit / place / queue?\n")
add("Two boxes (diagram in *Architecture diagrams*). **Gateway** (LiteLLM + `control/`): whether a request enters (cap, `should_shed`), which worker (`pick`), nothing else; it holds no queue. **Engine** (SGLang): waiting queue ordered by priority, continuous batching, chunked prefill (`--chunked-prefill-size 4096`), radix cache, KV allocation, retraction. The gateway never reimplements the scheduler. Engine flags used: `" + "`, `".join(flags) + "`. Maximum of the engine gauges during the 15-minute soak (Prometheus range queries saved in `metrics/evidence/final-soak/`):\n")
run, que = soak_series("engine_running"), soak_series("engine_queue")
add(block([f"sglang:num_running_reqs{{priority=\"\"}} max over the soak: worker 0 = {run.get('0', 0):.0f}, worker 1 = {run.get('1', 0):.0f}   (limit MAX_NUM_SEQS = 6)",
           f"sglang:num_queue_reqs{{priority=\"\"}}   max over the soak: worker 0 = {que.get('0', 0):.0f}, worker 1 = {que.get('1', 0):.0f}   (K = 2 may wait in the whole fleet)"] + lines(os.path.join(f2, "sglang_worker0.txt"), r"num_retracted_reqs")))

# ---------------------------------------------------------------- 10
tab = subprocess.run(["python3", os.path.join(ROOT, "script", "analyze_runs.py"), "--runs", R("fin-knee")], capture_output=True, text=True).stdout
tab = "\n".join(l for l in tab.splitlines() if l.startswith("|"))
add("### 10. What limited concurrency on this GPU for this app?\n")
add("Not the slots and not the KV. The expected limiter (KV) was refuted: the pool never went above 62 % in the load runs and 83-85 % in the deliberate worst case. What limited the system was **the cost of each call under load**: long prefills interleave with the decode of the other sequences, so the decode speed of one sequence falls from 48.8 tok/s alone to about 30 at N = 16-24 and 15 at N = 28 (`notes/findings.md`, \"Engine configuration review\"), and the extra requests are refused at the cap. Knee runs (`fin-knee`, columns include `decode tok/s` and `engine queue max`):\n")
add(tab + "\n")

# ---------------------------------------------------------------- 11
rules = json.load(open(os.path.join(f6, "alert_rules.json")))["data"]["groups"][0]["rules"]
add("### 11. Four production alerts\n")
add("`cluster/monitoring/alert-rules.yaml`, loaded by Prometheus (no Alertmanager: they show at `/alerts` and in dashboard 08):\n")
add("| Alert | Fires when | For | State when saved |\n| --- | --- | --- | --- |")
cond = {"CapacityShedRateHigh": "more than 5 % of requests are refused for capacity (`queue_full` or `kv_pressure`)", "InteractiveTTFTSLOBreach": "interactive TTFT p99 above the 20 s SLO",
        "KVPressureWithRetractions": "KV above 90 % while the engine retracts requests", "StaleOrMissingTelemetry": "the fleet snapshot is older than 45 s or a worker's /metrics is down"}
for r in rules:
    add(f"| `{r['name']}` | {cond[r['name']]} | {int(r['duration'] // 60)} min | {r['state']} (health {r['health']}) |")
add("\n(`InteractiveTTFTSLOBreach` was pending when this was saved because the KV probe, minutes before, queued twelve 60K-token prefills.) `/api/v1/rules`: `metrics/evidence/final6/alert_rules.json`.\n")

# ---------------------------------------------------------------- 12
ag = [summary("fin2-agents", n) for n in (12, 20)]
pin = sum(a["prompt_tokens"] for a in ag)
pout = sum(a["completion_tokens"] for a in ag)
cached = sum(a["cached_tokens_reported"] for a in ag)
add("### 12. If I scale, which pool: prefill tokens or decode slots?\n")
add("**Prefill.** With real tool-using agents (`fin2-agents`, N = 12 and 20) the model read **%.1f M prompt tokens (%.0f %% from cache) and wrote %.0f K tokens: %.0f prompt tokens for each output token**. The calls are prefill-heavy and decode-light (median prompt 21-24K tokens, median output 57-68), the cost of a call is its uncached tail, and the pressure at a turn boundary after an idle session is a prefill (27 %% of the first prompt of a turn from cache after 120 s of idle, `plots/final_cache_regimes.png`). Decode slots do not saturate with a healthy TTFT. So the pool to grow is the **prefill** one (more compute for uncached tokens, a bigger prefix cache); not \"another replica of the same size\", which duplicates 17.6 GiB of weights and starts with an empty cache.\n" % (pin / 1e6, 100 * cached / pin, pout / 1e3, pin / pout))

# ---------------------------------------------------------------- 13
add("### 13. What would I change at 10x traffic, and which three knobs are the wrong next move?\n")
add("See *At 10x traffic*. Short version: with the knee at about 16 synthetic sessions (48 served calls/min) or more than 20 real agents, 10x is 160-200 sessions: first policy (per-key budgets, batch to overflow earlier, a larger host and Mooncake cache), then add GPUs for the **prefill** pool. **Wrong next moves:** (1) raise `max_num_seqs` above 6 or move the KV to fp8 to count more sessions (the pool does not hold more 64K sequences; retraction and re-prefill follow; fp8 is excluded by decision 17); (2) lengthen the gateway queue or the deadlines (the K experiments: a longer queue lowers served calls per minute and hides overload); (3) add a worker of the same size on the same GPU (it splits the same HBM and SMs), or re-enable a first-token cut as a fix for TTFT (it breaks every long tool call).\n")

text = "\n".join(S)
p = os.path.join(ROOT, "ARCHITECTURE.md")
d = open(p).read()
begin, end = "<!-- ANSWERS:BEGIN -->", "<!-- ANSWERS:END -->"
section = f"{begin}\n{text}\n{end}"
if begin in d:
    d = d[:d.index(begin)] + section + d[d.index(end) + len(end):]
else:
    anchor = "# Alerts (production)"
    d = d.replace(anchor, "# Answers with evidence\n\nThe questions of the course's last part, each with its answer and the scrape or log it comes from. This section is generated by `script/build_answers.py` from `metrics/`: rerun it after new runs instead of editing it by hand.\n\n" + section + "\n\n" + anchor, 1)
open(p, "w").write(d)
print("answers section written:", len(text), "characters")

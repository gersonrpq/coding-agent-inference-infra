#!/usr/bin/env python3
"""Generates cluster/monitoring/grafana-dashboards.yaml (ConfigMap grafana-dashboards).

Run from the repo root after editing:

    python3 monitoring/build_dashboards.py

Every panel has a description (the "i" icon in Grafana) with two parts:
DEFINITION (what the number is, unit, source) and WHY IT MATTERS (the decision
or failure it helps explain). Each dashboard starts with an "About" panel that
says which design question it answers and where in the repo / which scrape to
point at.

Metric naming (see prometheus-config.yaml): SGLang keeps its native names
(`sglang:*`) and both workers are told apart by the `worker` label ("0", "1"),
so one query covers the whole fleet (`sum(...)`) or each worker
(`... by (worker)`). LiteLLM and Mooncake metrics are renamed with a job prefix:
LiteLLM's own `litellm_x` becomes `litellm_litellm_x`, our control-plane metrics
(`orch_*`, defined in control/fleet_state.py) become `litellm_orch_*`, and
Mooncake gets `mooncake_*`.
"""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "cluster" / "monitoring" / "grafana-dashboards.yaml"
DS = {"type": "prometheus", "uid": "prometheus"}
R = "$__rate_interval"

# --- metric name shortcuts -------------------------------------------------
PROXY_TOTAL = "litellm_litellm_proxy_total_requests_metric_total"
PROXY_FAILED = "litellm_litellm_proxy_failed_requests_metric_total"
QUEUE_REJECT = "litellm_litellm_admission_rejected_requests_total"
HOOK_SHED = "litellm_orch_requests_shed_total"
HOOK_ADMITTED = "litellm_orch_requests_admitted_total"
CLASS_LAT = "litellm_orch_request_latency_seconds"
CLASS_TTFT = "litellm_orch_request_ttft_seconds"
SNAP_AGE = "litellm_orch_fleet_snapshot_age_seconds"
FLEET_TTFT = "litellm_orch_fleet_ttft_seconds"
EST_WAIT = "litellm_orch_fleet_estimated_queue_wait_seconds"
GW_IN_FLIGHT = "litellm_litellm_admission_admitted_requests"   # LiteLLM's own admission middleware: requests in flight
GW_CAP = "vector(14)"                                              # max_in_flight_requests_per_worker = 12 running + K = 2 (litellm/config.yaml)
PLACED = "litellm_orch_placement_total"
PLACE_SESS = "litellm_orch_placement_session_total"
HOPS = "litellm_orch_hops_total"
HOP_TOKENS = "litellm_orch_hop_tokens_total"
OVERFLOW = "litellm_orch_overflow_decisions_total"
BATCH_SEL = '{priority_class="batch"}'
INTERACTIVE_SEL = '{priority_class="interactive"}'


def sg(name, sel=""):
    """SGLang metric selector. Worker identity is the `worker` label."""
    return f"sglang:{name}{{{sel}}}" if sel else f"sglang:{name}"


def by_worker(expr, legend="worker {{worker}}"):
    """One target whose series already carry the `worker` label."""
    return [(expr, legend)]


def hq(q, metric, rng=R, by=""):
    """Histogram quantile over SGLang buckets: whole fleet, or per label with by='worker'."""
    group = f"le, {by}" if by else "le"
    return f"histogram_quantile({q}, sum by ({group}) (rate({sg(metric + '_bucket')}[{rng}])))"


def lq(q, metric, sel="", by=""):
    """Histogram quantile over a gateway histogram (full metric name)."""
    group = f"le, {by}" if by else "le"
    return f"histogram_quantile({q}, sum by ({group}) (rate({metric}_bucket{sel}[{R}])))"


def avg_rate(metric_sum, metric_count, rng=R):
    return f"sum(rate({metric_sum}[{rng}])) / sum(rate({metric_count}[{rng}]))"


def desc(defn, why):
    return f"DEFINITION: {defn}\n\nWHY IT MATTERS: {why}"


# --- panel builders -------------------------------------------------------
class Dash:
    def __init__(self, uid, title, refresh="10s"):
        self.uid, self.title, self.refresh = uid, title, refresh
        self.panels = []
        self.x = 0
        self.y = 0
        self.row_h = 0
        self.next_id = 1

    def _place(self, w, h):
        if self.x + w > 24:
            self.y += self.row_h
            self.x, self.row_h = 0, 0
        pos = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        return pos

    def _targets(self, targets):
        return [
            {"refId": chr(65 + i), "expr": e, "legendFormat": l, "datasource": DS}
            for i, (e, l) in enumerate(targets)
        ]

    def _add(self, panel, w, h):
        panel["id"] = self.next_id
        self.next_id += 1
        panel["datasource"] = DS
        panel["gridPos"] = self._place(w, h)
        self.panels.append(panel)

    def row(self, title):
        self.y += self.row_h
        self.x, self.row_h = 0, 0
        self.panels.append({
            "type": "row", "title": title, "collapsed": False, "panels": [],
            "id": self.next_id, "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1},
        })
        self.next_id += 1
        self.y += 1

    def about(self, markdown, h=7):
        self._add({
            "type": "text", "title": "About this dashboard",
            "options": {"mode": "markdown", "content": markdown},
        }, 24, h)

    def stat(self, title, expr, unit, defn, why, steps=None, w=4, h=4,
             mappings=None, decimals=None, graph=True):
        steps = steps or [{"color": "green", "value": None}]
        defaults = {
            "unit": unit,
            "color": {"mode": "thresholds"},
            "thresholds": {"mode": "absolute", "steps": steps},
            "mappings": mappings or [],
        }
        if decimals is not None:
            defaults["decimals"] = decimals
        self._add({
            "type": "stat", "title": title, "description": desc(defn, why),
            "targets": self._targets([(expr, "")]),
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {
                "colorMode": "value", "graphMode": "area" if graph else "none", "textMode": "auto",
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            },
        }, w, h)

    def ts(self, title, targets, unit, defn, why, stack=False, w=12, h=8,
           threshold=None, ymax=None, mappings=None):
        custom = {
            "drawStyle": "line", "lineWidth": 1, "fillOpacity": 25 if stack else 8,
            "showPoints": "never", "spanNulls": True,
            "stacking": {"mode": "normal" if stack else "none", "group": "A"},
        }
        defaults = {"unit": unit, "custom": custom, "min": 0, "mappings": mappings or []}
        if ymax is not None:
            defaults["max"] = ymax
        if threshold is not None:
            custom["thresholdsStyle"] = {"mode": "line"}
            defaults["thresholds"] = {
                "mode": "absolute",
                "steps": [{"color": "green", "value": None}, {"color": "red", "value": threshold}],
            }
        self._add({
            "type": "timeseries", "title": title, "description": desc(defn, why),
            "targets": self._targets(targets),
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {
                "legend": {"displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull", "max"]},
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
        }, w, h)

    def build(self):
        return {
            "uid": self.uid, "title": self.title, "editable": True, "graphTooltip": 1,
            "refresh": self.refresh, "schemaVersion": 41,
            "time": {"from": "now-30m", "to": "now"}, "timezone": "",
            "tags": ["gpu-serving"], "templating": {"list": []},
            "annotations": {"list": []}, "links": [], "panels": self.panels,
        }


def GREEN_RED(bad):
    return [{"color": "green", "value": None}, {"color": "red", "value": bad}]


def GREEN_YELLOW_RED(y, r):
    return [{"color": "green", "value": None}, {"color": "yellow", "value": y}, {"color": "red", "value": r}]


def RED_GREEN(ok):
    return [{"color": "red", "value": None}, {"color": "green", "value": ok}]


FIRING = [{"type": "value", "options": {"0": {"text": "ok"}, "1": {"text": "FIRING"}}}]

# With --enable-priority-scheduling SGLang emits the total under priority="" and a
# per-priority breakdown under priority="<n>". Totals must select priority="" or they
# are counted twice.
RUNNING = sg("num_running_reqs", 'priority=""')
WAITING = sg("num_queue_reqs", 'priority=""')
CHAT_SEL = 'endpoint="/v1/chat/completions"'
KV_USED_PCT = f"100 * {sg('kv_used_tokens')} / {sg('max_total_num_tokens')}"
# Counters only exist after their first event, and sum(<empty>) + x is empty, so every
# term that may be absent after a restart is wrapped in `or vector(0)`.
ALL_REQS = f"((sum(rate({PROXY_TOTAL}[{R}])) or vector(0)) + (sum(rate({QUEUE_REJECT}[{R}])) or vector(0)))"
OK_REQS = f'(sum(rate({PROXY_TOTAL}{{status_code=~"2.."}}[{R}])) or vector(0))'
HOOK_SHED_RATE = f"sum(rate({HOOK_SHED}[{R}]))"
HOOK_ADMIT_RATE = f"sum(rate({HOOK_ADMITTED}[{R}]))"


# ==========================================================================
# 01 - Cluster overview
# ==========================================================================
def d01():
    d = Dash("gpu-cluster", "01 - Cluster", "5s")
    d.about(
        "**Purpose.** One-screen health of the whole stack: is it up, how loaded is it, is KV memory safe, "
        "is the cache working.\n\n"
        "**Application.** A tool-using coding agent (Pi) calling an OpenAI-compatible endpoint. Each request has a "
        "*shared prefix* (system prompt + tool definitions, identical across requests, served from the radix/HiCache) "
        "and a *unique suffix* (repo files, conversation history, tool results). The cache-hit panels measure how much "
        "of each prompt is shared.\n\n"
        "**Where to point.** Engine flags and capacity knobs: `cluster/sglang/worker-*.yaml`, `cluster/sglang/config.yaml` "
        "(MAX_NUM_SEQS, MAX_MODEL_LEN, MAX_MEM_FRAC). Policy: `control/admission.py`. Gateway limits: `cluster/litellm/config.yaml`. "
        "Raw scrapes: `curl <worker>:30000/metrics` (engine), `litellm:4000/metrics` (gateway + `orch_*`), "
        "`mooncake-master:9003/metrics`.\n\n"
        "**What limits concurrency on this GPU?** The engine runs at most MAX_NUM_SEQS = 6 requests per worker (12 in the fleet) and the gateway admits 12 + K = 14. "
        "KV was the expected limiter, but in the measured runs it peaked at 55-60 % of the pool: the binding limit was the cost of each call under load "
        "(long prefills interleaving with decode, cache misses). See *KV used* here and *KV-implied concurrency ceiling* (dashboard 06).",
        h=8)
    d.row("System at a glance")
    d.stat("Scrape targets healthy", "100 * sum(up) / count(up)", "percent",
           "Percentage of Prometheus scrape targets (litellm, sglang x2, mooncake, k8s API) that responded to the last scrape.",
           "A target that is down means every panel based on it is stale or empty. Check this first when a dashboard looks wrong.",
           RED_GREEN(100), decimals=0)
    d.stat("Requests / s", ALL_REQS, "reqps",
           "Requests per second reaching the gateway: proxied requests of any status plus requests rejected by LiteLLM's "
           "queue middleware (which rejects before the proxy route and never appears in proxy_total_requests).",
           "The offered load. Everything else (queues, KV, latency) should be read relative to it.",
           decimals=2)
    d.stat("Success ratio", f"100 * {OK_REQS} / {ALL_REQS}", "percent",
           "2xx responses divided by all requests, including queue-middleware 503s.",
           "The headline reliability number. A drop means shedding, rate limiting or engine errors.",
           [{"color": "red", "value": None}, {"color": "yellow", "value": 90}, {"color": "green", "value": 99}],
           decimals=1)
    d.stat("Rejected / s (429 + 503)",
           f'(sum(rate({PROXY_TOTAL}{{status_code=~"429|503"}}[{R}])) or vector(0)) + (sum(rate({QUEUE_REJECT}[{R}])) or vector(0))', "reqps",
           "Rate of requests refused by the platform: 429 (LiteLLM parallel-request limit) and 503 (admission hook or queue middleware).",
           "Separates deliberate load protection from engine failures. 429 never overflows; 503 may overflow to the external API.",
           GREEN_YELLOW_RED(0.01, 1), decimals=2)
    d.stat("In flight (cap 14)", GW_IN_FLIGHT, "short",
           "Requests in flight in the gateway, counted by LiteLLM's admission middleware (cap 14 = 12 running (2 workers x MAX_NUM_SEQS) + K = 2 waiting). It also counts the Prometheus scrape in progress.",
           "The cap is the fleet's concurrency budget; when it is reached, the next request is refused at once with a 503 (queue_full).",
           GREEN_YELLOW_RED(10, 14))
    d.stat("Engine queue", f"sum({WAITING})", "short",
           "Requests admitted by the gateway and waiting in SGLang's priority queue (the gateway holds none).",
           "With K = 2 this stays at 0-2; more means places are over-allocated.",
           GREEN_YELLOW_RED(2, 6))
    d.stat("Running on engines", f"sum({RUNNING})", "short",
           "Requests actively executing on the GPUs (SGLang num_running_reqs, both workers).",
           "Compare with MAX_NUM_SEQS x workers: at the ceiling, new work waits.",
           GREEN_YELLOW_RED(30, 40))
    d.stat("Waiting on engines", f"sum({WAITING})", "short",
           "Requests accepted by SGLang but not yet scheduled (num_queue_reqs). This is the engine queue, not the gateway queue.",
           "Persistent waiting means the engines are saturated; it is also the input to the admission hook's wait estimate.",
           GREEN_YELLOW_RED(1, 10))
    d.stat("KV used (worst worker)", f"max({KV_USED_PCT})", "percent",
           "kv_used_tokens / max_total_num_tokens on the most loaded worker.",
           "KV exhaustion forces the engine to retract running requests. The admission hook sheds new prefixes at 95%.",
           GREEN_YELLOW_RED(80, 95), decimals=1)
    d.stat("TTFT p95 (fleet)", hq(0.95, "time_to_first_token_seconds"), "s",
           "95th percentile time to first token measured by SGLang, both workers combined.",
           "TTFT is what an interactive user feels. Read it only under steady load: a cold replica inflates it.",
           GREEN_YELLOW_RED(2, 5), decimals=2)
    d.stat("Prefix cache hit rate", f"100 * avg({sg('cache_hit_rate')})", "percent",
           "Share of prompt tokens served from the prefix cache (GPU radix + host + Mooncake), averaged over workers.",
           "The direct measure of how much of the agent's prompt is shared prefix vs unique. High = cheap prefill.",
           decimals=1)
    d.stat("Mooncake pool used", "100 * mooncake_master_allocated_bytes / mooncake_master_total_capacity_bytes", "percent",
           "Allocated bytes divided by total segment capacity of the shared KV store.",
           "Above the master's eviction watermark (95%) Mooncake starts evicting KV that workers may still reference.",
           GREEN_YELLOW_RED(80, 95), decimals=1)
    d.stat("LLM backend p95",
           f"histogram_quantile(0.95, sum by (le) (rate(litellm_litellm_llm_api_latency_metric_bucket[{R}])))", "s",
           "95th percentile time spent in the upstream model call, as measured by LiteLLM.",
           "Distinguishes slow engines from slow gateway: compare with end-to-end latency in dashboard 02.", decimals=2)

    d.row("Traffic and load")
    d.ts("Requests by outcome",
         [(f"sum by (status_code) (rate({PROXY_TOTAL}[{R}]))", "HTTP {{status_code}}"),
          (f"sum(rate({QUEUE_REJECT}[{R}]))", "HTTP 503 (queue middleware)")], "reqps",
         "Requests per second by HTTP status, plus queue-middleware rejections shown separately because they bypass the proxy route.",
         "Shows at a glance whether a spike of load turned into successes or into refusals.", stack=True)
    d.ts("Engine load per worker",
         by_worker(RUNNING, "running w{{worker}}") + by_worker(WAITING, "waiting w{{worker}}"), "short",
         "Running = requests on GPU; waiting = engine queue, per worker.",
         "Imbalance between workers means routing is not spreading load; waiting > 0 means the engine is saturated.")
    d.ts("KV utilization per worker", by_worker(KV_USED_PCT), "percent",
         "KV tokens in use as a percentage of the KV pool, per worker. Red line = 95% admission threshold.",
         "The protected resource. Crossing the line makes the hook refuse new (uncached) prefixes.", threshold=95, ymax=100)
    d.ts("KV available tokens per worker", by_worker(sg("kv_available_tokens")), "short",
         "Free KV token slots per worker.",
         "Absolute headroom: divide by the tokens a typical request needs to see how many more requests fit.")
    return d


# ==========================================================================
# 02 - Success & failures
# ==========================================================================
def d02():
    d = Dash("gpu-success-failures", "02 - Success & Failures")
    d.about(
        "**Purpose.** Every request outcome, split by *who refused it and why*.\n\n"
        "**Status code contract.** `429` = a limit of the caller's own key (LiteLLM virtual keys: max_parallel_requests, tpm, rpm) or LiteLLM's parallel-request limit: the caller should back off, it **never** overflows. "
        "`503` = capacity (hook shed or LiteLLM queue middleware): the only code allowed to overflow to the external API. "
        "`500`/OOM stay local.\n\n"
        "**Two sources of 503.** LiteLLM's queue middleware (`litellm_admission_rejected_requests_total`, rejects *before* the proxy route, "
        "so it is absent from `proxy_total_requests`) and our hook (`orch_requests_shed_total`, `control/admission.py`).\n\n"
        "**Where to point.** `cluster/litellm/config.yaml` (max_parallel_requests, max_in_flight/queued), `control/admission.py` (HTTPException 503), "
        "`control/inspect.py` (fail-closed 503). Scrape: `litellm:4000/metrics`.", h=7)
    d.row("Totals since gateway start (reset when the LiteLLM pod restarts)")
    d.stat("Total requests", f"sum({PROXY_TOTAL}) + (sum({QUEUE_REJECT}) or vector(0))", "short",
           "Cumulative requests seen by the gateway, including queue-middleware rejections.",
           "Denominator for every ratio below.", graph=False)
    d.stat("2xx", f'sum({PROXY_TOTAL}{{status_code=~"2.."}}) or vector(0)', "short",
           "Cumulative successful responses.", "Goodput: work that actually completed.", graph=False)
    d.stat("429 rate limit", f'sum({PROXY_TOTAL}{{status_code="429"}}) or vector(0)', "short",
           "Cumulative 429 responses: a key over its limit or LiteLLM's per-deployment max_parallel_requests safety net.",
           "A 429 means 'slow down' and must not be overflowed to another provider; the per-key view is in dashboard 03.", GREEN_RED(1), graph=False)
    d.stat("503 via hook", f'sum({PROXY_TOTAL}{{status_code="503"}}) or vector(0)', "short",
           "Cumulative 503s raised from the pre-call hooks (admission shed or security_inspect fail-closed).",
           "Our own policy decisions; break them down by reason in dashboard 03.", GREEN_RED(1), graph=False)
    d.stat("503 via queue middleware", f"sum({QUEUE_REJECT}) or vector(0)", "short",
           "Cumulative requests rejected by LiteLLM's ASGI admission middleware (queue_timeout or queue_full).",
           "Blind to priority and tenant; if this is non-zero the gateway cap is binding before our policy is.",
           GREEN_RED(1), graph=False)
    d.stat("Other 4xx/5xx", f'sum({PROXY_TOTAL}{{status_code!~"2..|429|503"}}) or vector(0)', "short",
           "Cumulative responses with any status other than 2xx, 429 or 503.",
           "Genuine errors (bad request, engine 500). These should be zero in a healthy run.",
           GREEN_RED(1), graph=False)

    d.row("Outcomes")
    d.ts("HTTP outcomes / s",
         [(f"sum by (status_code) (rate({PROXY_TOTAL}[{R}]))", "HTTP {{status_code}}"),
          (f"sum(rate({QUEUE_REJECT}[{R}]))", "HTTP 503 (queue middleware)")], "reqps",
         "Responses per second by status. Queue-middleware 503s are a separate series because they bypass the proxy route.",
         "Shows how load turns into 200s versus protective refusals over time.", stack=True)
    d.ts("Success ratio", [(f"100 * {OK_REQS} / {ALL_REQS}", "2xx / all")], "percent",
         "2xx divided by all requests, including queue-middleware rejections.",
         "Trend of reliability under load; the SLO view.", ymax=100)
    d.ts("Rejections by source / reason",
         [(f"sum by (reason) (rate({QUEUE_REJECT}[{R}]))", "queue middleware: {{reason}}"),
          (f"sum by (reason) (rate({HOOK_SHED}[{R}]))", "admission hook: {{reason}}"),
          (f'sum(rate({PROXY_TOTAL}{{status_code="429"}}[{R}]))', "429 rate limit")], "reqps",
         "Refusals per second grouped by who refused (middleware, hook, rate limiter) and why. Hook reasons: timeout_queue, kv_pressure, batch_pressure.",
         "Tells you which control is acting. A policy that never fires, or a blind limit that fires first, are both design findings.", stack=True)
    d.ts("Failures by exception class",
         [(f"sum by (exception_class) (rate({PROXY_FAILED}[{R}]))", "{{exception_class}}")], "reqps",
         "Client-facing failures per second grouped by exception class (LiteLLM post_call_failure_hook).",
         "Separates rate limiting (RateLimitError), shedding (HTTPException) and real engine/backend errors.", stack=True)
    d.ts("Proxy total latency p50 / p95 / p99",
         [(lq(q, "litellm_litellm_request_total_latency_metric"), f"p{int(q*100)}") for q in (0.5, 0.95, 0.99)], "s",
         "End-to-end latency of successful requests measured by LiteLLM (wall seconds, includes generation).",
         "Latency depends on output length: compare across runs only with the same token counts (see tokens/s in dashboard 06).")
    d.ts("LLM backend latency p50 / p95 / p99",
         [(lq(q, "litellm_litellm_llm_api_latency_metric"), f"p{int(q*100)}") for q in (0.5, 0.95, 0.99)], "s",
         "Time inside the upstream SGLang call only.",
         "Subtract from total latency to see gateway overhead and queueing.")
    d.ts("LiteLLM overhead (avg)",
         [(avg_rate("litellm_litellm_overhead_latency_metric_sum", "litellm_litellm_overhead_latency_metric_count"), "avg overhead")], "ms",
         "Average time LiteLLM itself adds per request, excluding the model call (includes our pre-call hooks).",
         "Confirms the control plane (guard, admission) is cheap; it must not become the bottleneck.")
    return d


# ==========================================================================
# 03 - Gateway & admission
# ==========================================================================
def d03():
    d = Dash("gpu-gateway-admission", "03 - Gateway & Admission", "5s")
    d.about(
        "**Purpose.** What the gateway let in, queued or refused, *why*, and the signals it decided on.\n\n"
        "**Pipeline.** LiteLLM auth → `control/inspect.py` (guard: model, roles, content parts, tools, output cap) → "
        "`control/admission.py` (checks, batch share, placement) → SGLang; the in-flight cap is LiteLLM's own middleware.\n\n"
        "**A cap, not a queue.** The gateway admits 14 requests at once: 12 running (2 workers × MAX_NUM_SEQS) plus a queue allowance K = 2 that waits in SGLang's priority queue; the next request is refused at once by LiteLLM's admission middleware (503 `queue_full`). "
        "Nothing waits at the gateway. Targets: interactive TTFT p99 <= 20 s (the SLO, measured and bounded by K = 2; a first-token cut was removed because it breaks long tool calls), batch 120 s.\n\n"
        "**Shed reasons:** 503 (the only code that may overflow): `queue_full` (the 14-request cap, LiteLLM's middleware), `kv_pressure` (KV ≥ 95% and the system-prompt+tools prefix was not seen recently), "
        "`batch_pressure` (fleet TTFT p99 > 4× p50 over the last 60 s: batch is sacrificed first), `timeout_queue` (optional estimate, off by default). "
        "429 (never overflows): a virtual key over its limit (LiteLLM). `no_healthy_worker` is answered by placement; a stale snapshot is only reported, never a shed (decision 62).\n\n"
        "**Where to point.** Policy: `control/admission.py`; placement: `control/place.py`; inputs: `control/fleet_state.py` (cached FleetSnapshot, never Prometheus per request). "
        "Scrape: `litellm:4000/metrics`, series `orch_*`. LiteLLM's own admission middleware is only a safety net (limits above capacity).\n\n"
        "**Where does the engine scheduler sit?** Below this layer (dashboards 05/06): the gateway admits, queues and refuses; SGLang schedules, batches and retracts, "
        "using the priority forwarded in each request.", h=10)
    d.row("Gateway state")
    d.stat("In flight (cap 14)", GW_IN_FLIGHT, "short",
           "Requests in flight in the gateway (cap 14 = 12 running + K = 2).",
           "The cap is the concurrency budget of the fleet; at the limit the next request is refused at once, nothing waits at the gateway.",
           GREEN_YELLOW_RED(10, 14), w=6)
    d.stat("Engine queue", f"sum({WAITING})", "short",
           "Requests admitted and waiting in SGLang's priority queue (the gateway holds none).",
           "Bounded by K = 2 on purpose; more means places are over-allocated.",
           GREEN_YELLOW_RED(2, 6), w=6)
    d.stat("Hook shed ratio", f"100 * {HOOK_SHED_RATE} / ({HOOK_SHED_RATE} + {HOOK_ADMIT_RATE})", "percent",
           "Requests shed by control/admission.py divided by requests it evaluated (shed + admitted), recent window.",
           "The share of load the policy refuses. Too high = capacity problem; zero under heavy load = the policy never sees pressure.",
           GREEN_YELLOW_RED(1, 5), w=6, decimals=1)
    d.stat("Safety-net rejections", f"sum({QUEUE_REJECT}) or vector(0)", "short",
           "Cumulative requests rejected by LiteLLM's own admission middleware (limits set above fleet capacity).",
           "Should stay 0: if non-zero, the safety net fired before our policy did.", GREEN_RED(1), w=6, graph=False)
    d.ts("In flight against the cap",
         [(GW_IN_FLIGHT, "in flight"), (GW_CAP, "cap (12 running + K = 2)")], "short",
         "Requests in flight in the gateway against the cap of 14.",
         "Shows how close the fleet runs to the cap. The batch/interactive split of the old counter no longer exists: both share the cap.")

    d.row("Admission decisions")
    d.ts("Admitted vs shed / s by priority class",
         [(f"sum by (priority_class) (rate({HOOK_ADMITTED}[{R}]))", "admitted {{priority_class}}"),
          (f"sum by (priority_class) (rate({HOOK_SHED}[{R}]))", "shed {{priority_class}}")], "reqps",
         "Per second, how many requests the hook admitted and how many it shed, split into interactive and batch.",
         "Verifies the priority policy: under strain batch should be shed first while interactive keeps being admitted.")
    d.ts("Shed / s by reason",
         [(f"sum by (reason, priority_class) (rate({HOOK_SHED}[{R}]))", "hook: {{reason}} ({{priority_class}})"),
          (f"sum by (reason) (rate({QUEUE_REJECT}[{R}]))", "queue middleware: {{reason}}")], "reqps",
         "Refusals per second by reason. Hook = our policy; middleware = LiteLLM built-in queue.",
         "The 'sheds by reason' view: each reason maps to a different protection (timeouts, KV, interactive latency).", stack=True)
    d.ts("Shed totals (increase over selected range)",
         [(f"sum by (reason) (increase({HOOK_SHED}[$__range]))", "hook: {{reason}}"),
          (f"sum by (reason) (increase({QUEUE_REJECT}[$__range]))", "queue middleware: {{reason}}")], "short",
         "Number of refusals during the dashboard's time range, per reason.",
         "Use it to report 'how many requests did each protection stop' for a run.")
    d.row("Interactive vs batch (p99 spread)")
    d.ts("Latency p50 / p99 by priority class",
         [(lq(0.5, CLASS_LAT, by="priority_class"), "p50 {{priority_class}}"),
          (lq(0.99, CLASS_LAT, by="priority_class"), "p99 {{priority_class}}")], "s",
         "Gateway-measured end-to-end latency of successful requests, per priority class (interactive: priority ≤ 5, batch: > 5).",
         "If priority works, interactive tails stay low while batch absorbs the delay.")
    d.ts("p99 spread (batch − interactive)",
         [(f"{lq(0.99, CLASS_LAT, sel=BATCH_SEL)} - {lq(0.99, CLASS_LAT, sel=INTERACTIVE_SEL)}", "p99 batch − p99 interactive")], "s",
         "Difference between the batch and interactive 99th-percentile latency. Positive = interactive is protected.",
         "A spread near zero under load means prioritisation is not doing anything; negative means batch is faster than interactive.")
    d.ts("TTFT p50 / p99 by priority class (streaming)",
         [(lq(0.5, CLASS_TTFT, by="priority_class"), "p50 {{priority_class}}"),
          (lq(0.99, CLASS_TTFT, by="priority_class"), "p99 {{priority_class}}")], "s",
         "Time to first token as seen by the gateway for streaming requests, per priority class.",
         "TTFT is the interactive SLO. Only streaming requests report it.")

    d.row("Signals the hook decides on (control/fleet_state.py)")
    d.ts("KV used per replica vs 95% shed threshold",
         [("100 * litellm_orch_replica_kv_used_ratio", "{{replica}}")], "percent",
         "KV used ratio per replica exactly as the control plane last scraped it.",
         "kv_pressure sheds new prefixes when the worst replica reaches the line.", threshold=95, ymax=100)
    d.ts("Estimated queue wait vs timeout/2", [(EST_WAIT, "estimated wait")], "s",
         "Waiting requests × fleet TTFT p50, as computed by the hook (fleet_state.estimated_queue_wait_s). Red line = interactive deadline/2 = 10 s.",
         "Input of the optional timeout_queue rule (off by default: the gauges are stale by seconds; the in-flight cap and K = 2 replaced it).", threshold=10)
    d.ts("Fleet TTFT p50 / p99 seen by the hook",
         [(FLEET_TTFT + '{quantile="0.5"}', "p50"), (FLEET_TTFT + '{quantile="0.99"}', "p99")], "s",
         "TTFT quantiles the control plane computes from the SGLang histogram over the last 60 s (0 when fewer than 20 samples).",
         "batch_pressure triggers when p99 exceeds 4× p50; this is that comparison.")
    d.ts("Telemetry age (reported, never a shed)", [(SNAP_AGE, "seconds since last scrape")], "s",
         "Seconds since the control plane last scraped the workers successfully (-1 = never).",
         "Decisions made on old data are blind. The poller runs every 15 s; above ~45 s the snapshot is stale (shed reason declared, not yet enforced).",
         threshold=45)
    d.ts("Engine waiting queue per replica", [("litellm_orch_replica_queue_depth", "{{replica}}")], "short",
         "SGLang waiting-queue depth per replica as last scraped by the control plane.",
         "The waiting_total input of the wait estimate.")
    d.ts("Engine running per replica", [("litellm_orch_replica_running_requests", "{{replica}}")], "short",
         "Requests running on each replica as last scraped by the control plane.",
         "Capacity in use per replica, the decode-slot view used for placement and scaling.")
    d.row("Tenants and the overflow gate")
    d.ts("Requests by key and status / s",
         [(f"sum by (api_key_alias, status_code) (rate({PROXY_TOTAL}[{R}]))", "{{api_key_alias}} HTTP {{status_code}}")], "reqps",
         "Requests per second by virtual key (tenant or client) and HTTP status, from LiteLLM's own metrics.",
         "Who uses the GPU and who hit a limit (429). Clients use their own key; the master key is only for administration.", stack=True)
    d.ts("Overflow gate decisions / s",
         [(f"sum by (decision, status, forwarded) (rate({OVERFLOW}[{R}]))", "{{decision}} (HTTP {{status}}, forwarded: {{forwarded}})")], "reqps",
         "After a failed call the gate decides: leave (503/529) or stay (429, 500, slice_oom, guard refusals). 'forwarded: no' means the external model was not called.",
         "A 429 must never show up as leave. Forwarding to the overflow model is off, so 'leave' counts what would have overflowed.", stack=True)
    return d


# ==========================================================================
# 04 - Router
# ==========================================================================
def d04():
    d = Dash("gpu-router", "04 - Router")
    d.about(
        "**Purpose.** How requests are spread across the two SGLang deployments and whether each deployment is healthy.\n\n"
        "**Placement.** `control/place.py`, policy `affload` (default): keep a session on the worker that served its previous call, unless that worker has 3 or more requests in flight than the other; "
        "batch goes to the least loaded worker. The hook rewrites the model to the per-worker alias (`qwen-coding-w0/w1`), so LiteLLM has one deployment to choose from; no retries (`num_retries: 0`). "
        "With `PLACEMENT_POLICY=litellm` LiteLLM's `least-busy` decides instead.\n\n"
        "**Where to point.** `control/place.py`, `cluster/litellm/config.yaml` (`router_settings`, `model_list`). Scrape: `litellm:4000/metrics` (`litellm_deployment_*`, `orch_placement_*`, `orch_hops_*`).\n\n"
        "**Hop / what is not copied.** A hop is a call served by a different worker than its session's previous call (src != dst); the destination reads the prefix from Mooncake (dashboard 07) instead of recomputing it. "
        "Only KV of cached prefixes (and the recurrent-state snapshot) is shared; weights and the in-flight decode state are never transferred.", h=7)
    dep = "sum by (api_base) (rate({m}[" + R + "]))"
    d.row("Distribution across workers")
    d.ts("Requests / s per deployment", [(dep.format(m="litellm_litellm_deployment_total_requests_total"), "{{api_base}}")], "reqps",
         "Requests per second sent to each upstream SGLang deployment.",
         "With `affload` the lines need not match (sessions stay where their history is); a persistent gap with idle capacity on the other worker means the slack (3) is too high.", stack=True)
    d.ts("Load share per worker",
         [("100 * sum by (api_base) (litellm_litellm_deployment_total_requests_total) / "
           "scalar(sum(litellm_litellm_deployment_total_requests_total))", "{{api_base}}")], "percent",
         "Cumulative share of requests each deployment has received since gateway start.",
         "About 50/50 for identical workers under many sessions; a large skew indicates a placement or health problem.", ymax=100)
    d.ts("Success vs failure per deployment",
         [(dep.format(m="litellm_litellm_deployment_success_responses_total"), "ok {{api_base}}"),
          (dep.format(m="litellm_litellm_deployment_failure_responses_total"), "fail {{api_base}}")], "reqps",
         "Upstream responses per second per deployment, split into success and failure.",
         "Failures on one deployment only point at that worker (OOM, restart) rather than at the gateway.")
    d.ts("Latency per output token (avg)",
         [(f"sum by (api_base) (rate(litellm_litellm_deployment_latency_per_output_token_sum[{R}])) / "
           f"sum by (api_base) (rate(litellm_litellm_deployment_latency_per_output_token_count[{R}]))", "{{api_base}}")], "s",
         "Average seconds per generated token for each deployment.",
         "A token-normalised speed measure; wall seconds alone cannot be compared across different output lengths or models.")
    d.row("Placement policy (control/place.py)")
    d.ts("Requests / s by policy and worker",
         [(f"sum by (policy, worker, priority_class) (rate({PLACED}[{R}]))", "{{policy}} {{worker}} ({{priority_class}})")], "reqps",
         "Requests the custom policy placed, per second, by worker and class.", "Shows the balance the policy achieves and that batch is spread by load.", stack=True)
    d.stat("Session stickiness", f'100 * sum(increase({PLACE_SESS}{{result="kept"}}[$__range])) / (sum(increase({PLACE_SESS}{{result="kept"}}[$__range])) + sum(increase({PLACE_SESS}{{result="moved"}}[$__range])))', "percent",
           "Share of follow-up calls that stayed on their session's worker (kept) against those that moved, over the dashboard range.",
           "The mechanism behind the cache gain: LiteLLM's least-busy keeps ~50 %, affload ~90-95 %.", GREEN_YELLOW_RED(70, 40), w=6, decimals=1)
    d.ts("Sessions: new / kept / moved",
         [(f"sum by (result) (rate({PLACE_SESS}[{R}]))", "{{result}}")], "reqps",
         "Interactive calls by what happened to their session: new, kept on the same worker, or moved.",
         "A rising 'moved' means the load term overrides affinity (one worker is much busier).", stack=True)
    d.ts("In flight per worker (placement counters)", [("litellm_orch_placement_inflight", "worker {{worker}}")], "short",
         "Requests in flight on each worker as counted by the gateway.", "The load term of affload: moves happen when the gap reaches the slack (3).")
    d.row("Hops (src != dst)")
    d.ts("Hops / s by direction vs calls that stayed",
         [(f"sum by (src, dst) (rate({HOPS}[{R}]))", "hop {{src}} -> {{dst}}"), (f"sum(rate({PLACE_SESS}{{result=\"kept\"}}[{R}]))", "stayed on the same worker")], "reqps",
         "Calls placed on another worker than their session's previous call (hop) against calls that stayed on it (no hop).",
         "The hop rate is the cost of imperfect placement: each hop reads its prefix from Mooncake.", stack=True)
    d.ts("Estimated prompt tokens of hop calls / s",
         [(f"sum by (src, dst) (rate({HOP_TOKENS}[{R}]))", "{{src}} -> {{dst}}")], "short",
         "Estimated prompt tokens (about 4 characters per token) of the calls that hopped, per second.",
         "How much history the destination has to read from the lower cache tiers instead of recomputing; the engine's own load-back counters are in dashboard 07.")
    d.row("Worker availability and ramp (control/place.py)")
    d.ts("Worker answers its metrics (1) / is down (0)", [("litellm_orch_worker_available", "worker {{worker}}")], "short",
         "1 if the worker's /metrics answered the last scrape of the control plane, 0 if it did not (restarting, crashed).",
         "A worker at 0 is never picked by placement; with both at 0 the request is refused with 503 no_healthy_worker.", ymax=1.2, w=8)
    d.ts("Ramp factor of each worker", [("litellm_orch_worker_ramp_factor", "worker {{worker}}")], "short",
         "Weight of the worker in placement: 1 = full, 0.25 right after it comes back, rising to 1 over about 60 s while the fleet TTFT p99 is below 10 s.",
         "Answers 'slam it at 100 % or ramp while p99 holds': a returning worker is brought back gradually.", ymax=1.2, w=8)
    d.ts("Workers that came back (ramps started)", [(f"sum by (worker) (increase(litellm_orch_worker_ramps_total[$__range]))", "worker {{worker}}")], "short",
         "Times each worker went from down to answering again during the dashboard range.",
         "A worker that keeps coming back is flapping: look at its pod and its logs.", w=8)
    d.row("Health")
    d.stat("Deployment state", "litellm_litellm_deployment_state", "short",
           "Router health per deployment: 0 healthy, 1 partial outage, 2 complete outage.",
           "A deployment in outage is skipped by the router, halving capacity without any other visible symptom.",
           [{"color": "green", "value": None}, {"color": "yellow", "value": 1}, {"color": "red", "value": 2}],
           mappings=[{"type": "value", "options": {"0": {"text": "healthy"}, "1": {"text": "partial"}, "2": {"text": "outage"}}}],
           w=8, graph=False)
    d.stat("Cooldowns", "sum(litellm_litellm_deployment_cooled_down_total) or vector(0)", "short",
           "Number of times the router put a deployment into cooldown.", "Repeated cooldowns mean a flapping worker.",
           GREEN_RED(1), w=8, graph=False)
    d.stat("Failed fallbacks", "sum(litellm_litellm_deployment_failed_fallbacks_total) or vector(0)", "short",
           "Fallback attempts (e.g. to overflow) that also failed.", "Overflow must have a model and a limiter; failures here show it is not a safety net.",
           GREEN_RED(1), w=8, graph=False)
    return d


# ==========================================================================
# 05 - Queue depth (gateway vs engine)
# ==========================================================================
def d05():
    d = Dash("gpu-sglang-queue", "05 - Queue Depth by Worker", "5s")
    d.about(
        "**Purpose.** The three different places a request can wait, per pod, so a delay can be attributed to the right layer.\n\n"
        "1. **In-flight cap** (LiteLLM's admission middleware): the gateway holds nothing; it counts 12 + K requests and refuses the next one at once.\n"
        "2. **Engine waiting queue** (SGLang `num_queue_reqs`): accepted by the engine, not yet scheduled; ordered by priority "
        "(`--enable-priority-scheduling`).\n"
        "3. **Engine running** (`num_running_reqs`): on the GPU, in the continuous batch.\n\n"
        "**Scheduler boundary.** The gateway only decides *whether* and *where* a request enters. Batching, prefill/decode interleaving, radix cache, "
        "KV allocation and retraction are SGLang's scheduler: this repo does not re-implement them.\n\n"
        "**Where to point.** Engine flags in `cluster/sglang/worker-*.yaml` (`--max-running-requests`, `--chunked-prefill-size`, `--retraction-policy`). "
        "Scrape: `curl <worker>:30000/metrics | grep num_`.", h=7)
    d.row("Three places a request can wait")
    d.ts("1. In flight against the cap (gateway)",
         [(GW_IN_FLIGHT, "in flight"), (GW_CAP, "cap")], "short",
         "Requests in flight in the gateway, against the cap (12 + K).",
         "At the cap the next request is refused at once; nothing waits at the gateway.", w=8)
    d.ts("2. Engine waiting (SGLang)", by_worker(WAITING), "short",
         "Requests accepted by SGLang but not yet scheduled on the GPU, per worker.",
         "The queue that actually reflects GPU saturation.", w=8)
    d.ts("3. Engine running (GPU)", by_worker(RUNNING), "short",
         "Requests currently executing in the engine's continuous batch, per worker.",
         "Bounded by --max-running-requests (MAX_NUM_SEQS) and by KV capacity; whichever is lower is the real concurrency limit.", w=8)
    d.row("Engine pressure")
    d.ts("Waiting by priority (SGLang)",
         [('sum by (priority) (sglang:num_queue_reqs{priority!=""})', "priority {{priority}}")], "short",
         "Engine waiting queue broken down by request priority. Lower number is served first; -9223372036854775808 = no priority sent.",
         "Shows whether interactive (low number) requests overtake batch inside the engine.", stack=True)
    d.ts("Running by priority (SGLang)",
         [('sum by (priority) (sglang:num_running_reqs{priority!=""})', "priority {{priority}}")], "short",
         "Requests on GPU broken down by priority.", "Who actually holds the decode slots.", stack=True)
    d.ts("Retracted / paused requests",
         by_worker(sg("num_retracted_reqs"), "retracted w{{worker}}") + by_worker(sg("num_paused_reqs"), "paused w{{worker}}"), "short",
         "Retracted = running requests the engine evicted from the GPU because KV ran out (policy: priority). Paused = temporarily stopped.",
         "Retraction is the engine's last line of defence against KV exhaustion and wastes the work already done; the gateway should shed before this.")
    d.ts("Engine queue wait p50 / p95 / p99 (fleet)",
         [(hq(q, "queue_time_seconds"), f"p{int(q*100)}") for q in (0.5, 0.95, 0.99)], "s",
         "Time between SGLang accepting a request and starting its prefill.",
         "Direct measure of engine queueing: at most K = 2 requests wait here, because the gateway holds nothing.")
    d.ts("Engine queue wait p95 per worker", [(hq(0.95, "queue_time_seconds", by="worker"), "worker {{worker}}")], "s",
         "95th percentile engine queue wait per worker.", "Reveals a single overloaded worker hidden by the fleet average.")
    d.ts("PD-disaggregation queues (unused in unified mode)",
         [(f"sum({sg(q)})", q) for q in (
             "num_prefill_bootstrap_queue_reqs", "num_prefill_inflight_queue_reqs",
             "num_decode_prealloc_queue_reqs", "num_decode_transfer_queue_reqs", "num_decode_host_receive_queue_reqs")], "short",
         "Queues that only exist when prefill and decode run on separate workers.",
         "Must stay 0 until PD disaggregation is introduced; non-zero would mean a misconfiguration.")
    return d


# ==========================================================================
# 06 - SGLang engine
# ==========================================================================
def d06():
    d = Dash("gpu-sglang-performance", "06 - SGLang / Engine")
    d.about(
        "**Purpose.** The inference engine's own view: latency, throughput, cache and KV sizing. (The course template calls this the "
        "vLLM dashboard; this deployment uses SGLang.)\n\n"
        "**Shared vs unique tokens.** `prompt_tokens_total` is everything the agent sent; `cached_tokens_total` is the part served from cache "
        "(shared prefix: system prompt + tool definitions + earlier turns). The *uncached* remainder is the unique work done in prefill.\n\n"
        "**What limited concurrency on this GPU?** *KV-implied concurrency ceiling* below: KV capacity divided by the measured KV tokens per running request. "
        "Compare with MAX_NUM_SEQS: the smaller number is the real limit.\n\n"
        "**Cold replicas.** Do not quote TTFT right after a restart: weights loaded does not mean warm (cache empty, CUDA graphs, first-request costs).\n\n"
        "**Where to point.** `cluster/sglang/worker-*.yaml`, `cluster/sglang/config.yaml`. Scrape: `curl <worker>:30000/metrics`.", h=8)
    d.row("Latency")
    d.ts("TTFT p50 / p95 / p99 per worker",
         [(hq(q, "time_to_first_token_seconds", by="worker"), f"w{{{{worker}}}} p{int(q*100)}") for q in (0.5, 0.95, 0.99)], "s",
         "Time from request arrival to the first generated token, per worker (SGLang histogram).",
         "The interactive latency SLO. Quote it only from a warm, steady-state run, and with the prompt/output token counts.")
    d.ts("Inter-token latency (avg)",
         by_worker(f"rate({sg('inter_token_latency_seconds_sum')}[{R}]) / rate({sg('inter_token_latency_seconds_count')}[{R}])"), "s",
         "Average time between consecutive generated tokens.", "Decode speed as the user perceives streaming; grows with batch size.")
    d.ts("End-to-end request latency (avg)",
         by_worker(f"rate({sg('e2e_request_latency_seconds_sum')}[{R}]) / rate({sg('e2e_request_latency_seconds_count')}[{R}])"), "s",
         "Average total time SGLang spends on a request, from arrival to last token.",
         "Depends on output length; interpret together with generated tokens/s.")
    d.row("Throughput")
    d.ts("Generation throughput (tok/s)", by_worker(sg("gen_throughput")), "short",
         "Decode tokens per second reported by the scheduler.", "Total engine output rate; saturation shows as a plateau while concurrency keeps rising.")
    d.ts("Tokens / s by direction",
         by_worker(f"sum by (worker) (rate({sg('prompt_tokens_total')}[{R}]))", "prompt w{{worker}}")
         + by_worker(f"sum by (worker) (rate({sg('generation_tokens_total')}[{R}]))", "generated w{{worker}}"), "short",
         "Prompt tokens processed vs tokens generated per second.",
         "Prefill load (prompt) and decode load (generated) are different resources; this is the input for deciding which pool to scale.")
    d.ts("Engine HTTP responses (chat completions)",
         by_worker(f"sum by (worker, status_code) (rate({sg('http_responses_total', CHAT_SEL)}[{R}]))", "w{{worker}} HTTP {{status_code}}"), "reqps",
         "Responses per second as the engine itself reports them, per status.",
         "Independent of LiteLLM's accounting; useful to confirm the gateway and engine agree.")
    d.row("Cache, KV and utilisation")
    d.ts("Prefix cache hit rate", by_worker(f"100 * {sg('cache_hit_rate')}"), "percent",
         "Share of prompt tokens served from the prefix cache.",
         "Measures the shared-prefix share of the agent's prompts; a drop means the shared prefix is changing or being evicted.", ymax=100)
    d.ts("Cached vs uncached prompt tokens / s",
         by_worker(f"sum by (worker, cache_source) (rate({sg('cached_tokens_total')}[{R}]))", "w{{worker}} cached ({{cache_source}})")
         + by_worker(f"sum by (worker) (rate({sg('prompt_tokens_total')}[{R}])) - sum by (worker) (rate({sg('cached_tokens_total')}[{R}]))",
                     "w{{worker}} uncached (unique)"), "short",
         "Per second: prompt tokens served from cache (device = GPU radix, host = HiCache L2, storage = Mooncake L3) and tokens that had to be prefilled.",
         "Splits the prompt into shared (cached) and unique (uncached) tokens; uncached tokens are the real prefill cost.", stack=True)
    d.ts("Avg KV tokens per running request",
         by_worker(f"{sg('kv_used_tokens')} / ({RUNNING} > 0)", "w{{worker}}"), "short",
         "KV tokens in use divided by running requests (only while something is running).",
         "The measured KV footprint of this application's requests, used to size concurrency instead of guessing.")
    d.ts("KV-implied concurrency ceiling",
         by_worker(f"{sg('max_total_num_tokens')} / ({sg('kv_used_tokens')} / ({RUNNING} > 0))", "w{{worker}}"), "short",
         "KV pool size in tokens divided by the measured KV tokens per running request.",
         "How many requests of this shape fit in KV. If lower than MAX_NUM_SEQS, KV (not slots) limits concurrency on this GPU.")
    d.ts("Token usage", by_worker(sg("token_usage")), "percentunit",
         "Fraction of the KV pool in use (SGLang token_usage).", "Same signal as KV utilisation, straight from the engine.", ymax=1)
    d.ts("Forward occupancy", by_worker(sg("fwd_occupancy")), "percentunit",
         "Fraction of time the GPU forward pass is busy.", "Low occupancy with queued requests points at CPU/scheduler limits, not GPU.")
    d.ts("Scheduler idle %", by_worker(f"100 * rate({sg('scheduler_idle_seconds_total')}[{R}])"), "percent",
         "Share of time the scheduler had nothing to run.", "Near 100% = engine underused; near 0% = saturated.", ymax=100)
    d.ts("HTTP requests active", by_worker(sg("http_requests_active")), "short",
         "HTTP requests currently open on the engine's server.", "Includes streaming connections that are still open.")
    return d


# ==========================================================================
# 07 - Mooncake / HiCache
# ==========================================================================
def d07():
    d = Dash("gpu-mooncake-kv", "07 - Mooncake KV")
    d.about(
        "**Purpose.** The shared KV store (our hop store) and the cache hierarchy feeding it: GPU radix (L1) → host RAM (L2) → Mooncake (L3).\n\n"
        "**What moves.** SGLang HiCache publishes KV pages to Mooncake as they are written (`write_through`) and loads them into another worker's cache instead of "
        "recomputing the prefix. KV *tensors of cached prefixes* are the only thing shared: model weights and the decode state of running requests are never copied. "
        "This is not prefill/decode disaggregation.\n\n"
        "**Eviction.** The Mooncake master evicts above `--eviction_high_watermark_ratio 0.95`. A worker's cache index can still reference a prefix whose KV was "
        "evicted (a 'ghost'); the symptom is load-back/get misses after evictions. SGLang's radix tree evicts with LRU.\n\n"
        "**Hops** are recorded by the gateway (`orch_hops_*`, dashboard 04); the load-back and backup panels here are the engine's side of the same event. "
        "A prefix is reusable on the other worker only after its KV pages and a recurrent-state snapshot are in Mooncake.\n\n"
        "**Where to point.** `cluster/mooncake/master.yaml`, HiCache flags in `cluster/sglang/worker-*.yaml`. Scrape: `mooncake-master:9003/metrics`.", h=9)
    d.row("Pool capacity")
    d.stat("Pool used", "100 * mooncake_master_allocated_bytes / mooncake_master_total_capacity_bytes", "percent",
           "Allocated bytes divided by total capacity of the Mooncake segments.",
           "Eviction starts above 95%; staying near that line means constant churn.", GREEN_YELLOW_RED(80, 95), w=6, decimals=1)
    d.stat("Keys stored", "mooncake_master_key_count", "short",
           "Number of KV entries held by the master.", "Growth without bound with a full pool means high turnover of unique prefixes.", w=6)
    d.stat("Active clients", "mooncake_master_active_clients", "short",
           "SGLang workers currently connected to the master (expected: 2).",
           "Fewer than expected means a worker lost its KV store and silently fell back to recomputing.", RED_GREEN(2), w=6, graph=False)
    d.stat("Allocated", "mooncake_master_allocated_bytes", "bytes",
           "Bytes of segment memory currently in use.", "Absolute size behind the percentage.", w=6)
    d.ts("Pool bytes: allocated vs capacity",
         [("mooncake_master_allocated_bytes", "allocated"), ("mooncake_master_total_capacity_bytes", "capacity")], "bytes",
         "Used and total bytes of the shared KV store over time.", "Shows how fast the pool fills under a given workload.")
    d.ts("Keys", [("mooncake_master_key_count", "keys"), ("mooncake_master_soft_pin_key_count", "soft-pinned")], "short",
         "Total keys and keys protected from eviction (soft-pinned).", "Pinned keys reduce what eviction can reclaim.")
    d.row("Evictions")
    d.ts("Evictions / s",
         [(f"rate(mooncake_master_successful_evictions_total[{R}])", "successful"),
          (f"rate(mooncake_master_attempted_evictions_total[{R}])", "attempted")], "ops",
         "Eviction operations per second (attempted vs successful).",
         "Evictions free space but can remove KV that a worker still expects (ghosts). Attempted ≫ successful means eviction cannot keep up.")
    d.ts("Evicted bytes / s",
         [(f"rate(mooncake_master_evicted_size_bytes[{R}])", "total"), (f"rate(mooncake_master_evicted_size_bytes_mem[{R}])", "memory tier")], "Bps",
         "Bytes of KV evicted per second.", "The cost of running the pool full: data written then discarded.")
    d.ts("Put failures / s",
         [(f"rate(mooncake_master_put_start_failures_total[{R}])", "put_start failures"),
          (f"rate(mooncake_master_put_start_alloc_failures_total[{R}])", "alloc failures")], "ops",
         "Writes Mooncake refused per second.", "Refused writes mean prefixes are not being shared: usually a full pool.")
    d.row("HiCache tiers (GPU -> host RAM -> Mooncake)")
    d.ts("HiCache host (L2) usage",
         by_worker(f"100 * {sg('hicache_host_used_tokens')} / {sg('hicache_host_total_tokens')}"), "percent",
         "Fill level of each worker's host-RAM cache tier.", "A full L2 pushes prefixes out to Mooncake or drops them.", ymax=100)
    d.ts("Backup to L3 (tokens / s)", by_worker(f"rate({sg('hicache_backup_tokens_total')}[{R}])"), "short",
         "KV tokens per second written back to Mooncake.", "The 'hop out' volume: how much cache each worker publishes.")
    d.ts("Load-back from lower tiers (tokens / s)", by_worker(f"rate({sg('load_back_tokens_total')}[{R}])"), "short",
         "Prefix tokens per second reloaded from host/Mooncake instead of recomputed.",
         "The 'hop in' volume: the benefit of the shared store. Zero means workers are not reusing each other's prefixes.")
    d.ts("HiCache dropped tokens / s", by_worker(f"rate({sg('hicache_dropped_tokens_total')}[{R}])"), "short",
         "Tokens that could not be backed up per second.", "Cache work that was lost; non-zero under load points at bandwidth or capacity limits.")
    d.ts("Radix eviction (tokens / s)", by_worker(f"sum by (worker) (rate({sg('evicted_tokens_total')}[{R}]))"), "short",
         "Tokens per second the engine evicts from its GPU radix cache (policy lru).",
         "What leaves L1 under pressure; sessions that come back later read it from L2/L3 or recompute it.")
    d.ts("Mooncake get hit ratio",
         [(f"100 * rate(mooncake_valid_get_nums_[{R}]) / rate(mooncake_total_get_nums_[{R}])", "valid / total")], "percent",
         "Share of Mooncake lookups that found data.", "Low hit ratio after evictions is the symptom of ghost entries.", ymax=100)
    return d


# ==========================================================================
# 08 - Pods / health / alerts / scaling
# ==========================================================================
def d08():
    d = Dash("gpu-pods-scaling", "08 - Pods / Replicas / Scaling")
    d.about(
        "**Purpose.** Are the pods alive, would we page someone, and what would we scale.\n\n"
        "**Scaling today.** None: v0.1.0 has a fixed pool of 2 workers (no autoscaling, decision 62) and no extra GPU capacity. If we scaled, the question is "
        "*which pool*: **prefill tokens** (prompt tokens/s, engine queue wait: scale when long uncached prompts queue) or **decode slots** "
        "(running requests, inter-token latency, KV: scale when slots or KV are full). Adding another replica of the same size does not help when the "
        "cache is full of unique prefixes. Both signals are below.\n\n"
        "**Alerts.** The four rules of `cluster/monitoring/alert-rules.yaml` (loaded by Prometheus; visible at `/alerts`; no Alertmanager, so nothing is sent). "
        "The four stat panels below evaluate the same conditions (red = it would fire), and the last panel of the section shows the alert states Prometheus keeps.\n\n"
        "**Not covered here:** per-pod CPU/memory of the SGLang workers and replica counts (no kube-state-metrics / cAdvisor is scraped); "
        "only the LiteLLM process exposes its own CPU and memory.", h=8)
    d.row("Alerts (red = would fire; same conditions as cluster/monitoring/alert-rules.yaml)")
    d.stat("ALERT capacity shed rate", f'((sum(rate({HOOK_SHED}{{reason="kv_pressure"}}[5m])) + sum(rate({QUEUE_REJECT}{{reason="queue_full"}}[5m]))) / clamp_min(sum(rate({HOOK_SHED}[5m])) + sum(rate({HOOK_ADMITTED}[5m])) + sum(rate({QUEUE_REJECT}[5m])), 0.001)) > bool 0.05', "short",
           "1 if more than 5 % of requests are refused for capacity (the 14-request cap, queue_full, or kv_pressure); rule CapacityShedRateHigh, for 5 m.",
           "The fleet is at its limit: look at KV, the in-flight cap and the overflow share; do not lengthen the queue.",
           GREEN_RED(1), w=6, graph=False, mappings=FIRING)
    d.stat("ALERT interactive TTFT SLO", f'histogram_quantile(0.99, sum by (le) (rate({CLASS_TTFT}_bucket{INTERACTIVE_SEL}[5m]))) > bool 20', "short",
           "1 if the interactive TTFT p99 is above the 20 s SLO; rule InteractiveTTFTSLOBreach, for 10 m.",
           "Priority is no longer protecting the user: look at batch share, retractions and the cache hit rate.",
           GREEN_RED(1), w=6, graph=False, mappings=FIRING)
    d.stat("ALERT KV pressure + retractions", f"((max({KV_USED_PCT}) > bool 90) * (max(max_over_time({sg('num_retracted_reqs')}[5m])) > bool 0))", "short",
           "1 if KV is above 90 % while the engine retracts requests; rule KVPressureWithRetractions, for 5 m.",
           "The engine is re-doing prefill work: shed new prefixes earlier or lower the in-flight cap.",
           GREEN_RED(1), w=6, graph=False, mappings=FIRING)
    d.stat("ALERT stale or missing telemetry", f'((max({SNAP_AGE}) > bool 45) + (min(up{{job="sglang"}}) == bool 0)) > bool 0', "short",
           "1 if the control plane has not scraped the workers for 45 s or a worker's /metrics is down; rule StaleOrMissingTelemetry, for 2 m.",
           "Every admission decision is running on old data, or a worker is down.",
           GREEN_RED(1), w=6, graph=False, mappings=FIRING)
    d.ts("Alert states kept by Prometheus (ALERTS)", [("sum by (alertname, alertstate) (ALERTS)", "{{alertname}} {{alertstate}}")], "short",
         "Pending and firing alerts as Prometheus evaluates the rules.", "Pending = the condition is true but not for long enough yet; firing = it would page.", w=24, h=5)
    d.row("Targets")
    d.stat("SGLang workers up", 'sum(up{job="sglang"})', "short",
           "Number of SGLang workers whose metrics endpoint answered the last scrape (expected: 2).",
           "A worker that is scraped is serving; a worker that is up but cold is not yet ready (see warm states in the design).",
           RED_GREEN(2), w=6, graph=False)
    d.stat("Scrape targets up", "sum(up)", "short", "Total healthy scrape targets.", "Quick overall liveness.", w=6, graph=False)
    d.stat("LiteLLM up", 'up{job="litellm"}', "short", "1 if the gateway's /metrics answered.", "Gateway down = the whole app is down.",
           steps=RED_GREEN(1), w=6, graph=False)
    d.stat("Mooncake up", 'up{job="mooncake-master"}', "short", "1 if the Mooncake master's metrics answered.",
           "Workers keep serving without it but lose cross-worker KV reuse.", steps=RED_GREEN(1), w=6, graph=False)
    d.ts("up by job", [("up", "{{job}}")], "short", "Scrape success (1) or failure (0) per job over time.",
         "Shows when and for how long a component dropped out.", ymax=1.2)
    d.ts("Scrape duration by job", [("scrape_duration_seconds", "{{job}}")], "s",
         "How long each scrape took.", "Slow scrapes foreshadow a stuck process (e.g. a blocked event loop).")
    d.row("Scaling signals: which pool?")
    d.ts("Prefill pressure: prompt tokens/s and engine queue wait p95",
         by_worker(f"sum by (worker) (rate({sg('prompt_tokens_total')}[{R}]))", "prompt tok/s w{{worker}}")
         + [(hq(0.95, "queue_time_seconds"), "queue wait p95 (s)")], "short",
         "Prompt tokens processed per second per worker, with the fleet's engine queue wait p95.",
         "Scale the *prefill* pool when prompt tokens/s is high and requests queue before starting; more decode slots would not help.")
    d.ts("Decode pressure: running requests, KV used, inter-token latency",
         by_worker(RUNNING, "running w{{worker}}") + by_worker(KV_USED_PCT, "KV % w{{worker}}")
         + [(f"1000 * sum(rate({sg('inter_token_latency_seconds_sum')}[{R}])) / sum(rate({sg('inter_token_latency_seconds_count')}[{R}]))",
             "inter-token latency (ms)")], "short",
         "Running requests, KV percentage and average inter-token latency (ms).",
         "Scale the *decode* pool when slots or KV are full and tokens slow down. If KV is full of unique prefixes, a same-size replica does not fix it.")
    d.ts("Running vs waiting (fleet)", [(f"sum({RUNNING})", "running"), (f"sum({WAITING})", "waiting")], "short",
         "Total requests on GPU versus waiting, both workers.", "Saturation signal for any future autoscaler.")
    d.row("LiteLLM process")
    d.ts("CPU (cores)", [(f"rate(litellm_process_cpu_seconds_total[{R}])", "litellm")], "short",
         "CPU cores used by the LiteLLM process (limit in the Deployment: 2).",
         "The gateway runs our policy in-process; CPU saturation here would add latency to every request.")
    d.ts("Memory (RSS)", [("litellm_process_resident_memory_bytes", "litellm")], "bytes",
         "Resident memory of the LiteLLM process (limit: 2 GiB).", "Growth over time points at a leak in a callback or in request logging.")
    return d


# ==========================================================================
FILES = {
    "01-cluster.json": d01,
    "02-success-failures.json": d02,
    "03-gateway-admission.json": d03,
    "04-router.json": d04,
    "05-sglang-queue.json": d05,
    "06-sglang-performance.json": d06,
    "07-mooncake-kv.json": d07,
    "08-pods-scaling.json": d08,
}


def main():
    lines = [
        "# GENERATED by monitoring/build_dashboards.py - edit that script, not this file.",
        "apiVersion: v1",
        "kind: ConfigMap",
        "metadata:",
        "  name: grafana-dashboards",
        "  namespace: gpu-serving",
        "data:",
    ]
    for name, fn in FILES.items():
        body = json.dumps(fn().build(), indent=2)
        lines.append(f"  {name}: |")
        lines.extend("    " + l for l in body.splitlines())
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

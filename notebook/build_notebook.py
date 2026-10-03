#!/usr/bin/env python3
"""Builds notebook/part5_queue.ipynb (course Part 5 and the proof of the design decisions).

    python3 notebook/build_notebook.py && python3 notebook/run_notebook.py && python3 notebook/export_markdown.py

The notebook reads only files of this repository (metrics/runs/*, metrics/probes/*, cluster/*), so it runs anywhere.
If a Prometheus is reachable (PROM_URL, default http://localhost:9090) it also scrapes the live cluster.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Part 5: who runs next, and what you do not run

`admit -> place -> your queue -> engine waiting / running / preempt` on **one H100 PCIe split into two MIG instances**
(two SGLang workers, `Qwen/Qwen3.5-9B` bf16, 6 sequences of 64K per worker) behind LiteLLM with our control plane.

Every answer below is read from data of the runs (`metrics/runs/<experiment>/<variant>/`: `records.jsonl` is the raw
per-call data, `prometheus.json` the scrape taken at the end of the run, `timeline.csv` per minute) or from the
manifests. The load comes from `script/loadgen.py`: closed-loop agent sessions (shared system+tools prefix, context that
grows per step, tool pauses). **N** is the number of sessions working at once, **K = 2** the requests allowed to wait
beyond the 12 that run. Run it from the repository root: `python3 notebook/run_notebook.py` (a Markdown copy, readable without a server, is `notebook/part5_queue.md`).
""")

code("""
import json, glob, os, re, urllib.request
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
while not (ROOT / "metrics").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
RUNS = ROOT / "metrics" / "runs"
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)

def load(exp, variant):
    d = RUNS / exp / variant
    return {"summary": json.load(open(d / "summary.json")),
            "prom": json.load(open(d / "prometheus.json")),
            "records": pd.read_json(d / "records.jsonl", lines=True),
            "timeline": pd.read_csv(d / "timeline.csv")}

def series(prom, key, label):
    # one value per label combination -> {label value: value}
    return {s["labels"].get(label, "-"): s["value"] for s in prom.get(key, [])}

def live(query):
    # scrape the live Prometheus when it is reachable (tunnel to :9090); otherwise None
    url = os.environ.get("PROM_URL", "http://localhost:9090") + "/api/v1/query?query=" + urllib.parse.quote(query)
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return json.load(r)["data"]["result"]
    except Exception:
        return None
import urllib.parse
print("repo root:", ROOT.name)
print("runs:", sorted(p.name for p in RUNS.iterdir()))
""")

md("""
## 1. Who sits in our queue, and who in the engine's waiting queue?

We do not queue at the gateway. Measured first with a gateway queue of 64 (`eq-ref-*`) and then with a counter
(`eq-k*`): a longer queue only keeps the engine in its slow regime (decision 34). The gateway only **caps the requests in flight at
12 + K = 14** (LiteLLM's own admission middleware, `cluster/litellm/config.yaml`); the next request gets an immediate 503. The K = 2
admitted requests beyond the 12 running ones wait in **SGLang's priority queue** (`--enable-priority-scheduling`). The runs
below (`fin-*`) are the final configuration under 360 s runs.
""")
code("""
exp, var = "fin-knee", "N24"
r = load(exp, var)
prom = r["prom"]
rows = {
  "requests in flight in the gateway, max (cap 14)": series(prom, "gateway_in_flight_max", "x"),
  "engine waiting queue, max, per worker": series(prom, "engine_queue_max", "worker"),
  "engine running, max, per worker (limit 6)": series(prom, "engine_running_max", "worker"),
  "engine retracted (preempted), max, per worker": series(prom, "retracted_max", "worker"),
}
for k, v in rows.items():
    print(f"{k:55s}", v)
""")
md("""
Reading: at N = 24 (heavy overload) the cap was reached (14 in flight) and the engine queue stayed at 2-3 on a worker. The K = 2
allowance is fleet-wide, not per worker: when one worker has fewer than 6 running the other can hold both waiting requests (and a
gauge that is stale by seconds can show a little more), which is why placement matters for queueing (section 2). Every refused
request was refused at the door, none waited at the gateway. The soak shows the same over time (`plots/final_soak.png`, below).
""")

md("""
## 2. Waiting / running / preempted, and `orch_replica_queue_depth`: which pod, under which mix?
""")
code("""
sheds = prom["shed_by_reason"]
print("sheds by reason (requests during the measured window):")
print(pd.DataFrame([{"reason": s["labels"]["reason"], "class": s["labels"]["priority_class"], "n": s["value"]} for s in sheds if s["value"]]))
# per worker snapshot of the three states, across the four routing policies of the N = 24 experiment
out = []
for v in ["lb", "shuffle", "aff", "affload"]:
    p = load("er-n24", v)["prom"]
    for w in ("0", "1"):
        out.append({"policy": v, "worker": w,
                    "waiting max": series(p, "engine_queue_max", "worker").get(w),
                    "running max": series(p, "engine_running_max", "worker").get(w),
                    "retracted max": series(p, "retracted_max", "worker").get(w)})
pd.DataFrame(out).pivot(index="policy", columns="worker")
""")
md("""
Reading: the mix is 100 % interactive sessions of the `paper` profile at N = 24. `aff` (static hash) piled 7 requests in
worker 0's queue and 1 in worker 1's: that is the pod-level answer to *which pod*, and the reason the placement policy
matters for queueing. With `affload` the queues are 3/2. **Nothing was preempted** in any run (`retracted max` 0): KV
never filled (see 6).
""")

md("""
## 3. A 32K retrieval and a short agent decode are both ready: who goes first?

Two levels, as in the design:
1. **Gateway (us):** both take a place if one is free; if none is free, the interactive one still has the reserved
   places (batch may use only 75 % of them) and both are refused at once otherwise. Priority is forwarded to the engine
   in `extra_body.priority`.
2. **Engine (SGLang):** once both are inside, the engine decides: priority scheduling orders its waiting queue (lower
   number first) and **chunked prefill** cuts the 32K prefill into chunks of `--chunked-prefill-size` so decode steps of
   the short request interleave between chunks. We do not interleave ourselves.
""")
code("""
import yaml
cfg = yaml.safe_load(open(ROOT / "cluster/sglang/config.yaml"))["data"]
w0 = open(ROOT / "cluster/sglang/worker-0.yaml").read()
flags = re.findall(r"^\\s+- (--[a-z0-9-]+)\\s*(?:#.*)?$", w0, re.M)
print("ConfigMap sglang-config:")
for k in ("MODEL_NAME","KV_CACHE_DTYPE","MAX_MODEL_LEN","MAX_NUM_SEQS","MAX_MEM_FRAC","MAMBA_FULL_MEMORY_RATIO","CHUNKED_PREFILL_SIZE","RADIX_EVICTION_POLICY","HICACHE_WRITE_POLICY"):
    print(f"  {k:26s}{cfg.get(k)}")
print("engine flags in worker-0.yaml:", flags)
""")
md("""
Evidence for the interleaving cost: decode per sequence fell from **48.8 tok/s idle to 26.6 (N = 20) and 15.4 (N = 28)**
(`notes/findings.md`, "Engine configuration review"), i.e. long prefills do slow every decode on the same worker;
chunked prefill (4096) bounds how long. That is why the capacity limiter turned out to be the per-call cost under load,
not the slot count.
""")

md("""
## 4. PagedAttention vs radix cache: which one saved memory, which saved compute?

Paging removes fragmentation (a sequence uses only the blocks it needs): it is always on. The **radix cache** saves
*compute*, and memory only for what is shared across sessions (system prompt + tools, about 14 % of a prompt). The history
is per session. Below, where the prompt tokens of the `affload` run were served from:
""")
code("""
src = prom["prefill_tokens_by_source"]
df = pd.DataFrame([{"worker": s["labels"]["worker"], "mode": s["labels"]["mode"], "tokens": s["value"]} for s in src])
tot = df.pivot(index="worker", columns="mode", values="tokens").fillna(0)
tot["total"] = tot.sum(axis=1)
share = (tot.div(tot["total"], axis=0) * 100).round(1)
print("prompt tokens by where they came from (device = GPU radix, host = HiCache L2, storage = Mooncake L3, input = recomputed):")
print(tot.round(0)); print(share)
""")
code("""
# the same run as seen by the client: share of each prompt read from cache, split by whether the call stayed on the session's worker
rec = r["records"]
print(rec.columns.tolist())
""")
md("""
Reading: most of the prompt is **not recomputed** (it comes from the GPU radix cache, then host RAM, then Mooncake); the
radix cache is the structure that saved compute on this shared-history mix, paging only avoided waste. Prefix reuse across
sessions is limited to system+tools, so a worker's real capacity is *how many sessions' histories fit*.
""")

md("""
## 5. Placement experiment: what the routing policy did to the cache and to the queue

The policy decides how many calls land where their history is. Table of the N = 24 experiment (one run per variant, `lb` twice;
`metrics/runs/er-n24/comparison.md`):
""")
code("""
print(open(RUNS / "er-n24" / "comparison.md").read())
""")
code("""
order = ["lb", "lb2", "shuffle", "latency", "aff", "affload"]
rows = []
for v in order:
    s = load("er-n24", v)["summary"]["classes"]["interactive"]
    rows.append({"policy": v, "served/min": load("er-n24", v)["timeline"]["served"].iloc[1:].mean(), "stickiness": s["worker_stickiness"]})
d = pd.DataFrame(rows).set_index("policy")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
d["served/min"].plot.bar(ax=ax[0], color="#4477aa", title="served calls per minute (N = 24)")
d["stickiness"].plot.bar(ax=ax[1], color="#cc6677", title="share of calls that stay on the session's worker")
plt.tight_layout(); plt.show()
""")
code("""
from IPython.display import Image, display
for f in ("plots/routing_n24.png", "plots/routing_n20.png"):
    if (ROOT / f).exists():
        display(Image(filename=str(ROOT / f)))
""")

md("""
## 6. KV full after admit: do we shed at the door, or does the engine preempt?

Both, in order. **At the door:** `kv_pressure` refuses a *new* prefix when the fleet KV is at 95 % or more. **After admit:**
the engine retracts the least urgent running request (`--retraction-policy priority`) and re-prefills it later; the gateway
never preempts. In these runs the KV did not fill, so neither fired:
""")
code("""
for v in ["lb", "shuffle", "aff", "affload"]:
    p = load("er-n24", v)["prom"]
    kv = series(p, "kv_used_max", "worker")
    shed = sum(s["value"] for s in p["shed_by_reason"] if s["labels"]["reason"] == "kv_pressure")
    ret = sum(series(p, "retracted_max", "worker").values())
    kv_txt = {k: round(x, 2) for k, x in kv.items()}
    print(f"{v:8s} KV used max per worker {kv_txt}  kv_pressure sheds {shed:.0f}  retracted {ret:.0f}")
""")
md("""
The binding limit at this load was therefore **not memory** (KV peaked near 55-60 %) but the speed of the work: prefill
interference and cache misses (section 3). With 6 sequences of 64K per worker the KV pool holds 435,199 tokens (6.64 sequences);
the recurrent states of this hybrid model (46 per worker) take the rest of the pool.

**The worst case, on purpose** (`metrics/probes/kv_probe.py`): 12 requests of about 60K tokens at once, then the same with the shed
threshold lowered to 0.70 so that the door acts:
""")
code("""
for name in ("kv_probe_default.log", "kv_probe_0.70.log"):
    print("-----", name)
    for line in (ROOT / "metrics/logs" / name).read_text().splitlines():
        if line.startswith(("KV used", "long requests", "new-prefix", "reused-prefix", "extra requests", "orch_requests_shed_total{priority_class=\\"interactive\\"")):
            print(line[:210])
""")
md("""
The KV peaks at 83-85 % with **zero retractions**: 6 running sequences x 64K cannot take more than 88 % of the pool, so after admission
the engine never has to preempt, and the 95 % threshold cannot be reached by running sequences. With the threshold lowered, a request
with a **new prefix is refused in 0.1 s with a 503 `kv_pressure`** and one that reuses a seen prefix is admitted: shedding at the door
protects the KV when it is full; the engine preempts only if it ever overcommits (it did not).
""")

md("""
## 7. Client gone (aborted): who frees the KV, and how?

LiteLLM runs with `cancel_on_disconnect: true`: when the client leaves, LiteLLM closes the upstream connection and **SGLang
aborts the request and frees its KV blocks**. The gateway's place is returned in the failure event, and a reaper recovers
a leaked place after 330 s. Verified with a probe that watches the engine (`metrics/logs/conn.log`, `conn4.log`):
""")
code("""
for name in ("conn.log", "conn4.log"):
    text = (ROOT / "metrics/logs" / name).read_text()
    for line in text.splitlines():
        if line.startswith("VERDICT") or "VERDICT" in line or line.startswith("== s"):
            print(f"{name}: {line.strip()}")
""")
md("""
Results of that probe: a streaming client that leaves while its request runs (s1) and a non-streaming one (s2) both end with
the upstream closed and generation stopped early; and when LiteLLM itself cuts a queued request at its `stream_timeout` (s4) the request is removed from SGLang (verified live: 408 at 4.2 s). That first-token cut
was later removed because the same timeout fires in the middle of a stream while a tool call's arguments are generated (decisions 59, 61). The
log of s3 (a client that leaves while its request is queued in SGLang) has no verdict line, so it is not claimed here.
""")

md("""
## 8. After a worker returns: slam it at 100 % or ramp?

"Ready" (model loaded, port open) is not "warm". `script/warm_workers.py` (stage 5b of `launch_cluster.sh`) measures the
first-token time of a unique 4K-token prompt **cold**, warms the worker with prompt shapes the agent sends, and measures
again; LiteLLM is applied only after both workers pass. The SLO is quoted from the **warm** number.
""")
code("""
for name, what in (("warm_workers.json", "workers up for 140 min (already warm)"), ("warm_after_restart.json", "worker 1 just restarted")):
    f = ROOT / "metrics/probes" / name
    if f.exists():
        print(f"{name}: {what}")
        print(pd.DataFrame(json.load(open(f))))
    else:
        print(name, "not present")
""")
md("""
**At run time** a worker that comes back is not slammed to 100 %: `place.py` gives it 25 % of its weight and raises it to 100 %
over 60 s, pausing while the fleet TTFT p99 is above 10 s, and never picks it while its metrics do not answer. The probe restarts
worker 1 under a constant load of new sessions (`metrics/probes/ramp_probe.py`; columns: requests, served by worker 0 and 1, errors,
share of worker 1, and the gauges `orch_worker_available` and `orch_worker_ramp_factor` of worker 1):
""")
code("""
print((ROOT / "metrics/logs/ramp_probe.log").read_text())
""")
code("""
from IPython.display import Image, display
if (ROOT / "plots/final_ramp.png").exists():
    display(Image(filename=str(ROOT / "plots/final_ramp.png")))
""")
md("""
Reading: 3 requests failed when the pod died (the ones in flight); then nothing went to worker 1 for four minutes and there were no more
errors; when it answered again its share started at 0.13 and climbed to 0.54 over about 125 s, the ramp factor staying at 0.25 for ~45 s
because the cold worker pushed the fleet p99 above the hold.
""")

md("""
## 9. Hop: when does KV move, and is the destination warm?

`src == dst`: the prefix is already on the device, nothing moves (`orch_hops_local_total`). `src != dst`: the other worker
reads the prefix that this one published to Mooncake instead of recomputing it. Probe 5 (direct to the workers, 47K-token prefix):
""")
code("""
print(pd.DataFrame([
  {"case": "cold prefill, worker 0", "TTFT s": 4.48, "cached tokens": 0},
  {"case": "same prefix, same worker (src == dst)", "TTFT s": 0.94, "cached tokens": 46976},
  {"case": "same prefix on the OTHER worker (src != dst, via Mooncake L3)", "TTFT s": 2.28, "cached tokens": 46912},
]))
tot = sum(s["value"] for s in prom["placement_sessions"])
print({s["labels"]["result"]: round(s["value"]/tot, 3) for s in prom["placement_sessions"]}, "<- affload: new / kept / moved sessions")
""")
md("""
A hop costs about 1.3 s more than a local hit and saves about 2.2 s against recomputing. The gateway records every hop
(`orch_hops_total`, `orch_hop_tokens_total`, `orch_hop_cached_tokens_total`) and prints a `HOP` line; the dictionary below is from
a 200 s run with `lb` at N = 16 (`metrics/evidence/after-ev-n16/hop_dictionary.jsonl`):
""")
code("""
hops = pd.read_json(ROOT / "metrics/evidence/after-ev-n16/hop_dictionary.jsonl", lines=True)
print(len(hops), "hops;", f"{hops.cached_tokens.sum() / hops.tokens.sum():.0%}", "of their", int(hops.tokens.sum()), "prompt tokens were read from cache on the destination")
print(hops.groupby(["src", "dst"]).agg(n=("tokens", "size"), tokens=("tokens", "sum"), cached=("cached_tokens", "sum")))
hops.head(5)
""")
md("""
Eviction and 429/503 counts of the same run (`metrics/evidence/after-ev-n16/`):
""")
code("""
ev = ROOT / "metrics/evidence/after-ev-n16"
for name in ("evict_count.txt", "status_429_503.txt"):
    print(f"--- {name}"); print("\\n".join(l[:170] for l in (ev / name).read_text().splitlines()))
""")
md("""
Tenant limits (429, never an overflow) were checked with `metrics/probes/tenant_probe.py`:
""")
code("""
print((ROOT / "metrics/logs/tenant_probe.log").read_text())
""")
md("""
## 10. The finished system under load: the final tests

Synthetic sessions (`script/loadgen.py`) with the final configuration (K = 2, `affload`; the runs F1-F5 had a first-token cut of 20 s that fired once and was later removed). The knee, the replicate
against LiteLLM's `least-busy` (same conversations), the 20 % batch run, regime B (120 s idle) and a 15-minute soak:
""")
code("""
import json
rows = []
for tag, n in [("fin-knee", 16), ("fin-knee", 20), ("fin-knee", 24), ("fin-rep-affload", 20), ("fin-rep-lb", 20),
               ("fin-batch", 20), ("fin-regimeB", 48), ("fin-soak", 16)]:
    v = json.load(open(RUNS / tag / f"N{n}" / "summary.json"))["classes"]["interactive"]
    rows.append({"run": tag, "N": n, "served/min": v["served_per_min"], "TTFT p50": round(v["ttft_p50"], 1), "TTFT p99": round(v["ttft_p99"], 1),
                 "refused %": round(100 * v["shed_rate"], 1), "abandoned": f"{v['calls_abandoned']}/{v['calls']}",
                 "cached %": round(100 * v["cached_tokens_reported"] / v["prompt_tokens"]), "stickiness": v["worker_stickiness"]})
pd.DataFrame(rows)
""")
code("""
from IPython.display import Image, display
for f in ("plots/final_knee.png", "plots/final_soak.png", "plots/final_cache_regimes.png", "plots/final_batch.png", "plots/final_agents.png"):
    if (ROOT / f).exists():
        display(Image(filename=str(ROOT / f)))
""")
md("""
Reading: refusals stay below 5 % up to N = 16; above it the cap refuses 38-70 % of the attempts but the served ones keep their TTFT p99
between 7 and 14 s against the 20 s SLO. Paired at N = 20, `affload` beats `least-busy` on every number. A session that rests 120 s
reads only 27 % of the first prompt of a turn from cache. The soak is flat: queue <= 2, KV <= 57 %, nothing leaked.
""")

md("""
## 11. Real agents: tool-using sessions (`script/agentgen.py`)

`loadgen.py` is synthetic (random text, no tools, forced output lengths). `agentgen.py` runs real sessions: the model reads files of this
repository through three tools (`read_file`, `list_dir`, `grep`), decides when it has enough and answers; the history is what happened.
""")
code("""
import glob
paths = sorted(glob.glob(str(RUNS / "fin2-agents" / "N*" / "summary.json")))
if not paths:
    print("no agentic runs saved yet (metrics/runs/fin2-agents)")
for path in paths:
    s = json.load(open(path))
    v = s["classes"]["interactive"]
    print(path.split("/")[-2], "| served/min", v["served_per_min"], "| TTFT p50/p99", round(v["ttft_p50"], 1), round(v["ttft_p99"], 1),
          "| refused %", round(100 * v["shed_rate"], 1), "| cached %", round(100 * v["cached_tokens_reported"] / v["prompt_tokens"]))
    print("   ", json.dumps(s["agent"]))
""")

md("""
## 12. Live scrape (only if a Prometheus is reachable)
""")
code("""
queries = {
  "requests in flight (cap 14)": "litellm_litellm_admission_admitted_requests",
  "engine waiting per worker": "sglang:num_queue_reqs{priority=\\"\\"}",
  "engine running per worker": "sglang:num_running_reqs{priority=\\"\\"}",
  "shed by reason": "sum by (reason, status_code) (litellm_orch_requests_shed_total)",
  "rejected by the cap": "sum by (reason) (litellm_litellm_admission_rejected_requests_total)",
  "hops": "sum by (src, dst) (litellm_orch_hops_total)",
  "overflow decisions": "sum by (decision, status) (litellm_orch_overflow_decisions_total)",
  "requests by key": "sum by (api_key_alias, status_code) (litellm_litellm_proxy_total_requests_metric_total)",
  "evicted tokens": "sglang:evicted_tokens_total",
}
for name, q in queries.items():
    res = live(q)
    if res is None:
        print(f"{name:28s} Prometheus not reachable (open the tunnel to :9090 to scrape live)"); continue
    print(f"{name:28s}", [(x["metric"].get("worker") or x["metric"].get("reason") or x["metric"].get("decision") or x["metric"].get("src") or x["metric"].get("api_key_alias") or "", x["value"][1]) for x in res][:8])
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebook/part5_queue.ipynb")
print("wrote notebook/part5_queue.ipynb")

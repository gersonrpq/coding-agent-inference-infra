# Glossary

The canonical list of terms used in `notes/`, `ARCHITECTURE.md` and the plots. Every experiment protocol links here instead of redefining them.
Numbers are the ones of the served configuration: one H100 PCIe 80 GB split by MIG into two workers, `Qwen/Qwen3.5-9B` bf16.

## The load

| Term | Meaning |
| --- | --- |
| **Session** | One simulated coding agent (like one `pi` session): a loop of an LLM call, a short pause for a tool, the next call with a longer context, and a longer pause between user turns. |
| **Call / attempt** | A call is one LLM request of a session; an **attempt** is one try of it (a refused call is retried, the load generator makes up to 3 attempts). |
| **N** | The load: how many sessions work **at the same time**. N = 20 means 20 agents working at once. |
| **Synthetic vs agentic load** | `loadgen.py` is synthetic: it reproduces the shape of an agent session with random text, no tools and forced output lengths. `agentgen.py` runs real agents: the model calls tools and the history is what happened. |
| **Profile `paper` / `light`** | The shape of the prompts: `paper` starts at 30-52K tokens (the median production call is ~68K, Liu et al.); `light` starts at 4-20K. |
| **Regime A / B** | A = dense: 20 s between turns (measures the compute limit: slots and prefill). B = realistic: 120 s idle between turns (measures how many **open** sessions the caches hold). |
| **Interactive / batch** | Priority class of a request: `priority <= 5` is interactive (a human waits), `> 5` is batch (a background sweep). Lower number = more urgent. |
| **Shared vs unique tokens** | The system prompt + tool schemas (~9-10K tokens) are shared by every session; the history and tool results are unique per session. |

## Capacity and the queue

| Term | Meaning |
| --- | --- |
| **Worker** | One SGLang engine process on one MIG instance (39.5 GiB). There are two. |
| **Running (seqs)** | Requests the engine is decoding at once: `MAX_NUM_SEQS = 6` per worker, **12 in the fleet**. |
| **In flight / the cap** | Requests the gateway has accepted and not yet answered. LiteLLM's admission middleware allows at most `12 + K` = 14 (`max_in_flight_requests_per_worker`); the next one is refused at once. (Earlier notes call each unit a "place" or a "slot".) |
| **K** | The queue allowance: how many requests may **wait beyond the 12 running ones**. K = 2 (decision [43](../ARCHITECTURE.md#decision-43)). They wait in SGLang's priority queue, not at the gateway. |
| **Queue** | There is exactly one: the **engine's waiting queue** (SGLang `num_queue_reqs`, ordered by priority). The gateway holds nothing; the original gateway queue (`r0`, up to 64 waiters) was removed from the code (decision [46](../ARCHITECTURE.md#decision-46)). |
| **Refusal / shed** | An immediate answer "not now" instead of serving: 503 for capacity reasons, 429 for tenant limits. Each carries a `shed-reason`. |
| **Shed reasons** | `queue_full` (the 14-request cap; earlier `decode_capacity`), `kv_pressure` (new prefix while KV >= 95 %), `batch_pressure` (fleet TTFT p99 > 4x p50 and > 10 s, batch only), `batch_share` (batch is admitted only while fewer than 10 requests of any class are in flight), `timeout_queue` (optional estimate, off). A key over its limit gets a 429 from LiteLLM. |
| **Abandoned call** | A call refused on all its attempts: the session gave up on it. |
| **Overflow / gate** | Overflow is the external model (`Qwen3.8-27B-FP8` on Superlinked) that may receive requests refused with 503/529. The **gate** decides stay or leave after the local result: 429, 500 and `slice_oom` stay. Forwarding is off (the provider's wallet is exhausted); the gate counts what would leave. |
| **Tenant / virtual key** | A caller with its own LiteLLM key (stored in PostgreSQL) and its own limits (`max_parallel_requests`, tpm/rpm); over its limit it gets a 429. |

## Latency and goodput

| Term | Meaning |
| --- | --- |
| **TTFT** | Time to first token: queue wait plus prefill. What the user waits before the answer starts. |
| **p50 / p99** | Median and 99th percentile: 1 call in 100 is slower than p99. |
| **SLO** | Our target: interactive TTFT p99 of served calls <= 20 s (decision [16](../ARCHITECTURE.md#decision-16)), refusals <= 1 % in the original verdict. It is a measured target, bounded by K = 2; the first-token cut that was meant to enforce it was removed (decisions [59](../ARCHITECTURE.md#decision-59), [61](../ARCHITECTURE.md#decision-61)). |
| **Served per minute (goodput)** | Calls that completed per minute: the useful throughput. |
| **Shed rate** | Refused attempts divided by attempts (retries count as attempts, so it is higher than the share of calls that end refused). |
| **Knee** | The N from which the system starts refusing a lot or breaks the SLO. |
| **Cold / warm** | A freshly started worker pays for lazy kernels, allocator growth and empty caches: its first requests are slower. "Ready" (model loaded, port open) is not "warm". The SLO is quoted from the warm number. |
| **Decode tok/s** | Output tokens per second of one sequence (about 48.8 alone, 26.6 at N = 20, 15.4 at N = 28: prefills interleave with decode). |

## Cache and placement

| Term | Meaning |
| --- | --- |
| **Radix cache (L1)** | SGLang's prefix cache in GPU memory; eviction policy `lru`. |
| **HiCache (L2) / Mooncake (L3)** | Host-RAM tier of each worker / the pool shared by both workers. A prefix lives there after eviction from the GPU. |
| **Cached / recomputed share** | Fraction of the prompt tokens read from a cache tier (GPU, host, Mooncake) versus computed again by the engine. |
| **Recurrent state** | The fixed-size per-sequence state of the linear-attention layers of this hybrid model (~47 MB, 46 per worker). A prefix is reusable on the other worker only once its KV pages **and** a state snapshot are in L3. |
| **Ramp** | A worker that comes back after being down is given a fraction of its weight (25 %) that grows to 100 % over 60 s, paused while the fleet TTFT p99 is above 10 s, instead of receiving its full share at once. |
| **Shed (placement)** | What `pick` returns when no worker answers its metrics: a 503 `no_healthy_worker`. |
| **Hop** | A call served by a different worker (`dst`) than the session's previous call (`src`): the destination reads the prefix from L3 instead of recomputing it. `src == dst` means nothing moves. |
| **Stickiness** | Share of a session's calls that land on the same worker as its previous call. |
| **Placement policy** | How a request chooses a worker: `lb` (LiteLLM least-busy), `shuffle` (random), `latency` (lowest latency), `aff` (hash of the session), `affload` (the session's worker unless it is busier by 3 requests; batch to the least loaded). `affload` is the default. |
| **Eviction / ghost** | Evicting is dropping a cached prefix from a tier. A ghost is something remembered as cached that is not (the gateway's prefix memory after an engine eviction, or a Mooncake key whose owner restarted). |

## Method

| Term | Meaning |
| --- | --- |
| **Pre-registered** | The hypotheses and the decision rule of an experiment are written before running and never edited; changes go in dated notes at the end. |
| **Run / variant / tag** | A run is one load generator execution; a variant is a setting under comparison; a tag names an experiment folder in `metrics/runs/<tag>/<variant>/`. |
| **Noise** | Difference between two runs of the same setting. A difference smaller than that is not a result. |

# Findings (lab notebook)

A file to remember **what we found out, with what evidence, and what it implies**. It is not the design
document (`ARCHITECTURE.md`); it holds the raw data and the lessons, in order. Each entry: date, finding, evidence,
consequence. Rule: whenever something is discovered, it is added here.

Machine: 1 x H100 PCIe 80 GB (81559 MiB), K3s, SGLang 0.5.21, LiteLLM 1.105.0. Model: `Qwen/Qwen3.5-9B` bf16, KV bf16.

## Terms and reading guide

Terms (N, K, place, refusal, TTFT, p99, SLO, knee, stickiness, hop, ...) are defined in [`glossary.md`](glossary.md). The index of this folder is [`README.md`](README.md).

This file is chronological: entries are never rewritten, only added. Names that appear in old entries and **no longer exist in the code**: the gateway queue
and its reference run `r0`, `ADMISSION_MODE` (`queue`/`gate`), `GATEWAY_QUEUE_MAX`, the variants `q0` and `gate`, the metrics `orch_gateway_queue_*` (decision [46](../ARCHITECTURE.md#decision-46)).
Where an old entry says "slot", read "place".

| If you look for... | Read |
| --- | --- |
| the machine, memory, speed, the hop with Mooncake | Hardware and GPU, Memory and capacity, Speed, Hop with Mooncake |
| how the control plane is wired and what the native queues do | Native queues, How the control plane is plugged into LiteLLM |
| how K = 2 was chosen | Queue experiment (reference runs), Queue size K at overload, Choosing K |
| why the engine, not the queue, is the limit | Engine configuration review |
| which worker should serve a call | Placement, Routing experiment at N = 24, at N = 20 |
| mechanisms checked live (cut, connection close, tenants, warm-up, hops) | Does LiteLLM close the upstream connection?, Live checks of the cut, Tenant limits / cold worker / hop dictionary, First scrape of the deployed control plane |
| cost | Cost in LiteLLM, verified |
| the overflow model | Overflow and where fallbacks acts, Direct test of the overflow endpoint |

## Hardware and GPU

- **2026-10-02 · MIG works.** `nvidia-smi mig -lgip` offers 2 `3g.40gb` instances (39.5 GiB, 46 SMs each; together 92 of 114 SMs). To enable: `sudo nvidia-smi -i 0 -mig 1` + `sudo nvidia-smi mig -cgi 9,9 -C`. **The instances do not survive a reboot** → `setup/mig.sh`.
- **The device plugin with MIG needs `privileged: true`.** With only `CAP_SYS_ADMIN` it fails with "Insufficient Permissions" when reading the instance memory.
- **Privileged workers see both MIG instances.** Left alone, both use instance 0 and the second dies with OOM (15 MiB free). On top of that, the NVIDIA runtime rewrites `NVIDIA_VISIBLE_DEVICES` to `void` for PID 1 (only `kubectl exec` sees the real value, which fooled my first fix). Solution: `CUDA_VISIBLE_DEVICES` = 0 / 1 in each manifest.
- **The `mlx5_0` NIC exists and is active (RoCE).** Mooncake initialises the RDMA transport and works between the two workers on the same host.
- Disk 968 GB free, RAM 221 GB.

## Start-up and operation

- **The images are huge**: SGLang is 15.2 GB (more than 5 min to pull). The script gave up at 300 s → `IMAGE_PULL_TIMEOUT` (1800 s) for every rollout.
- **Ordering bug in `launch_cluster.sh`**: Prometheus mounts the Secret `litellm-secrets`, which was created later → on a fresh machine Prometheus stayed in `ContainerCreating`. The Secret is now created first.
- **A change in `sglang-config` alone does not restart the workers** (it does not change the pod template). The script now restarts them.
- **SGLang's `/health` takes ~1.0 s** because it runs a 1-token generation. The startup probe with the default timeout (1 s) always failed although the server was fine → `timeoutSeconds: 20`.
- **Cold start**: weights 20-31 s, scheduler ready at ~105-115 s, decode CUDA graphs 27 s; first answer at ~2 min. The first time: +19 GB download per model (shared cache in `/var/lib/gpu-serving/huggingface`).
- **Prefill CUDA graphs cost 5.4 GB and 209 s per worker** and one died with OOM during capture. Disabled (`--cuda-graph-backend-prefill disabled`); decode graphs cost 0.04 GB.
- `pkill -f launch_cluster.sh` over ssh kills itself (its own command line contains the pattern). Use a script on the server (`/tmp/relaunch.sh`).

## Memory and capacity (measured)

- **bf16 weights = 17.62 GiB** (`mem usage` in the log). RedHat FP8-dynamic = 12.68 GiB (14.0 GB on disk, because it keeps the linear layers, embeddings, lm_head and vision in bf16).
- **Static pool** = 39.15 − 0.15×39.5 = 33.2 GiB with `--mem-fraction-static 0.85`; 5.9 GiB stay outside for activations.
- **The recurrent state of the hybrid model is large**: with the default ratio (`--mamba-full-memory-ratio 0.9`) SGLang reserved 7.34 GiB (152 states × 47 MB) and left only **268,567 KV tokens = 4.1 sequences of 64K**.
- **With ratio 0.17**: 46 states (2.20 GiB), **435,199 KV tokens = 6.64 sequences of 64K** per worker. (The prediction was ~435K.)
- Each recurrent state weighs ~47 MB. With 46 states per worker, only ~46 distinct prefixes have a "resume point" on the GPU.
- **Host (HiCache)**: KV pool in RAM ~835K tokens per worker and 15 GB of recurrent states. **Mooncake**: 2 × 16 GB = 32 GB; with page size 1 it stores 2 keys per token (K and V): 47,037 tokens → 94,102 keys.
- With bf16 KV it is 32 KiB per token (the same value as the 27B with FP8 KV).

## Speed (measured, one request alone)

| Case | Tokens | Time |
| --- | ---: | ---: |
| Cold prefill | 62,648 | **7.34 s** (~8.5K tok/s) |
| Cold prefill | ~47,000 | **4.0-4.35 s** (~11K tok/s) |
| Local GPU hit (same prompt) | 46,976 cached | **0.94 s** |
| Prefix only in the host tier of the same worker | 46,912 cached | 2.29 s |
| **Hop from Mooncake (other worker)** | 46,912 cached | **2.28 s** |

- There was **one 13.9 s stall** when the GPU cache started evicting to the host (only once). If it hits an interactive request it breaks the SLO.
- The local hit (0.94 s) is not free: 47K tokens of context still cost ~1 s for the prefill of "the rest" and for reading.

## Hop with Mooncake (the central test)

- **HiCache does attach to the hybrid model** (`UnifiedRadixCache hybrid_ssm=True hicache_attached=True`). SGLang's 2025 note that listed it as future work is out of date.
- **With `write_back` the other worker never hit** (3 attempts): a prefix that is live on the GPU is not in Mooncake until it is evicted.
- **With `write_through` the KV is published at once**, but worker 1 still did not hit: the **recurrent state** was missing (only 13 states published).
- **The recurrent state is published when it leaves its 46-entry pool.** After pushing it out with 80 short prompts: `mamba_backed` 25 → 183 and worker 1 read **46,904 of 46,930 tokens from Mooncake** (`storage_hit`), TTFT 2.28 s against 4.48 s cold.
- Consequence: the other worker can only reuse a conversation **after** its state has been published. `MAMBA_FULL_MEMORY_RATIO` controls that too. I did not re-test `write_back` with this procedure.
- Useful metrics: `sglang:prefill_effective_tokens_total{mode=device_hit|host_hit|storage_hit|input}`, `sglang:hicache_backup_tokens_total{pool=kv|mamba}`, `sglang:hicache_host_used_tokens`. The internal prefetch counters (`l3_demand_requests`, `declined_*`) exist in the code but are **not exported**.

## Model and client

- **The paper (arXiv 2608.00101)**: the **median prompt of one call is 68K tokens**, not "tokens per task". Median output 247; the cache hits 45% on the 1st call of a turn, 86% on the 2nd, 92-94% afterwards, and drops ~26% when the turn changes (idle 2-10 min). 64K is a memory-driven cap, not "enough". An automatic web summary claimed the opposite and was wrong. (2026-10-03: ARCHITECTURE no longer cites the paper to justify 64K, because its 68K median describes Copilot production traffic and does not connect with our measured 21-24K median / 55K maximum; the justification there is now our own measurements. The paper stays here as background, and for the cache-cliff observation.)
- **The model thinks by default.** A trivial question ("17*23") spent all 200 tokens reasoning (3.8 s); with `chat_template_kwargs: {"enable_thinking": false}` it answers in 4 tokens and 0.15 s. It goes through the gateway both as a top-level field and inside `extra_body`. **Decision: the client (the coding agent) chooses**; the gateway does not touch it. For load balancing, remember that an answer with thinking can be 50× longer.
- Quality: the log warns `Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0` → a reason not to use FP8 KV.

## SLO

- **TTFT < 20 s** (queue wait + prefill), p99 assumed, initial value to be validated under stress. Interactive deadline 20 s, maximum slot wait 10 s. Decode does not count.
- Worst case inside the SLO with no load: 10 s of wait + 7.3 s of cold 64K prefill ≈ 17 s (margin ~3 s).
- `orch_request_ttft_seconds` only exists for **streaming** requests.

## Measurement tools

- **Under MIG `nvidia-smi` gives no GPU utilization** (`utilization.gpu` = `[N/A]`); it does give power (79 W idle with both models loaded), memory and temperature. Power is the load indicator. DCGM is not installed either.
- **GPU price: 3.29 USD/hour** (user-provided). Superlinked overflow cost: $0.25 per million input tokens, $2.00 per million output tokens.
- The cluster metrics exist with these names: `litellm_orch_*` (admitted, shed by reason, queues, TTFT and latency per class), `sglang:num_queue_reqs`, `num_running_reqs`, `num_retracted_reqs`, `full_token_usage`, `time_to_first_token_seconds`, `prefill_effective_tokens_total{mode}`.
- The load generator and the cost tools are in `script/` (see its README); results go to `metrics/runs/`.
- An admission shed now carries the `shed-reason` header so the client can tell `timeout_queue`, `kv_pressure`, `batch_pressure` and `decode_capacity` apart.

## Load-generator smoke test (2026-10-02, 3 sessions, `light` profile, 100 s measured)

Not a capacity measurement: it only validates the tools. Data in `metrics/runs/smoke/N3/`.

- **Defect fixed after the test**: `sweep.sh` used the same seed for every N, so a run could re-send the conversations of the previous one (still cached) and inflate the hits. The seed now changes with N; the system prompt stays common on purpose.
- **It works end to end**: 24 attempts, 24 served, 0 shed; TTFT p50/p95/p99 = 0.51 / 1.12 / 1.86 s. The client and gateway figures agree in order of magnitude (gateway p99 3.7 s includes the first cold call).
- **The generator's token estimate is well calibrated**: 3.9 tokens per word gives contexts within ~1% of the real value (29,324 estimated against 29,466 real).
- **Decode measured: ~48 tok/s per sequence** (one sample: 83 tokens in 1.7 s), matching the 50 estimate. The sweep will give the distribution.
- **The hop happened on its own, unprovoked**: the shared system prefix (10K tokens) produced `storage_hit` of ~19,900 tokens on worker 1 (read from Mooncake) and `device_hit` of hundreds of thousands of tokens on both workers. The common prefix is where the hop pays off naturally.
- **Power**: 251 W with 3 sessions against 79.7 W idle.
- **Cost at low load (corrected, 2026-10-02):** GPU = 3.29 USD/h x 100 s = **0.0914 USD**. That same bill divided by each token kind gives 0.121 USD per million input tokens (all), 1.745 USD per million **uncached** input tokens, 8.58 USD per million output tokens; they are three views of the *same* bill and are not added. **93.1% of the input tokens came from the cache** (705,344 of 757,701) and the slots were busy only 19.9% of the time. Comparison with overflow (0.25 / 2.00 USD per million): with no cache discount it would be 0.211 USD (the GPU, 0.43x, would be cheaper); **if overflow charged cached tokens at 10%, it would be 0.052 USD and the GPU would be 1.76x dearer**. My first reading ("the local GPU is 2.3 times cheaper even with 3 sessions") left out that assumption and was premature. The answer depends on two things we do not know yet: whether Superlinked discounts the cache, and what utilization we reach (the sweep gives the curve).
- The median output was 320 tokens (the lognormal sampling with `ignore_eos` gives somewhat more than 247).
- Shed reasons and queues appear in `prometheus.json` (all zero here).
- **Slot time in the smoke test**: sum of TTFT = 16 s against sum of decode = 223 s. Slot occupancy is almost all decode (93%), even though the *tokens* are almost all input. This matters when splitting costs.

## Native queues in LiteLLM and SGLang (2026-10-02, read in the pods' code)

Reason: the user preferred that LiteLLM and SGLang manage the wait with their parameters rather than a hand-written queue. This is what the versions that run offer (LiteLLM 1.105.0, SGLang 0.5.21):

| Collapse risk with a queue | Native parameter | Where | Used today |
| --- | --- | --- | --- |
| Unbounded engine waiting queue: latency grows without limit | `--max-queued-requests N` | SGLang | **no** (unbounded) |
| Queue full: whom to reject? | with `--enable-priority-scheduling`, SGLang aborts the **least-preferred and newest** waiting request if the incoming one is strictly better ("The request is aborted by a higher priority request"); otherwise it rejects the incoming one ("The request queue is full"). Both with **HTTP 503** | SGLang (`_abort_on_queued_limit`) | priority yes, limit no |
| An urgent request waits because batch holds everything | `--priority-scheduling-preemption-threshold`, `--retraction-policy priority`; running-request preemption is on unless `--disable-priority-preemption` | SGLang | retraction yes, default threshold |
| Order of the engine queue | `--schedule-policy` (`fcfs`, `lpm`, `priority`, `hrrn`, `shortest-prefill-first`…) | SGLang | priority enabled |
| A burst fills the gateway | `max_in_flight_requests_per_worker`, `max_queued_requests_per_worker`, `admission_queue_timeout_seconds`: a **FIFO** queue with a semaphore in an HTTP middleware, 503 "Worker at capacity" (`queue_full` / `queue_timeout`), counter `litellm_admission_rejected_requests_total` | LiteLLM | yes, as a safety net (128 / 32 / 10 s) |
| Waiting too long for the first token | the router's `stream_timeout` (exists; its effect on TTFT is **untested**) | LiteLLM | no |
| A 429 that must not overflow | `max_parallel_requests` | LiteLLM | high on purpose |
| Retries that multiply the load | `num_retries: 0` | LiteLLM | yes |
| The client leaves and the engine keeps going | `cancel_on_disconnect` | LiteLLM | yes |

- **LiteLLM's priority `Scheduler` (`/queue/chat/completions`) is no use for this**: its `poll` returns true as soon as there are *healthy* deployments (not in cooldown), so it only waits when none is; it does not queue by load.
- **LiteLLM's admission middleware** matches our queue in the basics (in-flight limit, bounded queue, maximum time, 503), but it is FIFO, acts before the model or the priority is known, and is per process.
- **SGLang exports the queue per priority**: `sglang:num_queue_reqs{priority=""}` is the total and `priority="<n>"` the breakdown (the poller in `fleet_state.py` already sums only the `""` series). A gate could therefore count only the requests ahead of a given priority.
- **What SGLang does surprisingly well**: a bounded queue with priority and eviction of the least-preferred request, and 503 (fit for overflow).
- **What neither does** and justifies our own code: rejecting by *estimated wait time* (half the deadline), reserving slots for interactive, rejecting on KV pressure at the door, seeing the whole fleet (SGLang's queue is per worker: with bad placement one can have a full queue and the other room to spare) and publishing the shed reason.
- Candidates to compare with the current queue (Q0): **Q1** SGLang decides (`--max-queued-requests` per worker + priority preemption, our admission only tags the priority, LiteLLM as a safety net) and **Q2** LiteLLM's FIFO queue (12 in flight, 64 queued, 10 s) with priority inside SGLang. Plan in `notes/sweep-experiment.md`, section 9.

## How the control plane is plugged into LiteLLM (2026-10-02)

- There is no separate service: `control/*.py` goes in the ConfigMap `litellm-security`, mounted at `/app/security` and visible through `PYTHONPATH`. `config.yaml` registers two callbacks (guard and admission); the **queue is not a callback but a library** imported by `admission.py` (singleton `GATEWAY` created at import, from the variables `FLEET_MAX_IN_FLIGHT`, `BATCH_SHARE`, `GATEWAY_QUEUE_MAX`).
- The wait happens **inside LiteLLM, before the router**: `await GATEWAY.acquire()` leaves the coroutine suspended with the client connection open; the success and failure events give the slot back.
- **One replica and one process** (verified: `replicas=1`, a single `litellm` process). With two replicas or `--num_workers 2` each process would have its own 12 slots and the real capacity would silently double. Scaling the gateway out would need a shared store (Redis) or a consistent router.
- **Importing the same module under two names breaks** (`Duplicated timeseries in CollectorRegistry: orch_gateway_in_flight`): reproduced in the pod. That is why the state lives in modules that are only imported as `security.*`, and why the tests load them the same way.

## Occupancy model (2026-10-02, before the sweep)

- A session only occupies a slot while a call is in progress. In the smoke test: 9.9 s per call (TTFT ~0.6 s + ~20.7 ms per token = 48 tok/s) and 2.38 of 12 slots busy for 3 sessions, i.e. **0.79 slots per session** with 8 s pauses.
- Saturation depends on the pause between turns: N = 12 / occupancy = 17 (8 s pause), **20 (20 s)**, 30 (60 s), **46 (120 s)**, 94 (300 s). My initial hypothesis of "~16 sessions" did not account for the pauses and was corrected.
- Consequence: "how many sessions can it take?" has no single answer. With dense agents compute rules (slots and prefill, N≈20); with agents that rest as in the paper (80% of the time idle) cache memory rules (open sessions). That is why the experiment has two regimes: `notes/sweep-experiment.md`.
- The load generator now splits TTFT by call position (session start, turn start, within turn), flags the runs that did not reach a steady state and writes a per-minute timeline.

## Cost in LiteLLM, verified (2026-10-02)

- With the new rates in `litellm/config.yaml`, the metric `litellm_litellm_spend_metric_total{model="Qwen/Qwen3.5-9B"}` read **0.0001266 USD** after two test requests and the start-up smoke test. Expected: 1629×4.8e-8 + 2×1.59e-6 = 8.14e-5; (30×4.8e-8 + 1600×9e-9 + 2×1.59e-6) = 1.90e-5; smoke 16 + 16 tokens = 2.62e-5; total 1.266e-4. **It matches to 4 digits.**
- LiteLLM **does apply the cache rate** when the response carries `prompt_tokens_details.cached_tokens` (the second identical request reported 1,600 cached tokens; the first, none). If the response does not report them, it charges the whole prompt at the input rate.
- The metric is called `litellm_litellm_spend_metric_total` (prefix duplicated by Prometheus' renaming).
- The rates are still **provisional**: they are recomputed with the throughput measured in the sweep.

## Unit tests (2026-10-02)

- 58 tests in `tests/` (`python3 -m unittest discover -s tests -t .`), they run in under 1 s and need no cluster. They cover the queue (priority, interactive reservation, deadlines, leaked slots), admission (each shed is a 503 with its reason and header; each event gives the slot back), the guard and the fleet state. Since then 11 more tests cover the load generator's analysis (69 in total).
- **Mutation check**: I deliberately broke six things (release that does not wake waiters, batch ignoring its limit, KV that never sheds, shed with 429, output cap removed, a latency tail that is never bad) and each made between 1 and 6 tests fail. I then restored the code and the 58 went back to OK. The load generator's tests were mutation-checked too (warmup not excluded, steady state always true, sheds never failing a run).
- They **do not cover** the real LiteLLM integration (hook order, header propagation), the SGLang poller, or anything about the engine.
- Lesson: `control/inspect.py` has the same name as a standard-library module; that is why the tests load it by path and do not put `control/` on `sys.path`.

## Queue mechanism probes, first results (2026-10-02, SGLang queue limit of 3 per worker, gate mode)

- **m1: the engine's "queue full" reaches the client as HTTP 500, not 503.** Body: `litellm.APIConnectionError: APIConnectionError: OpenAIException - The request queue is full.` Under our status-code table a 500 stays local (it never overflows), so an engine-side refusal would have to be mapped to 503 before it is useful.
- **`--max-queued-requests` bounds simultaneous arrivals, not the queue behind the running requests.** 24 calls arriving together with a limit of 3 per worker got only 4 served and 20 refused within ~3 s, although 12 running places plus 6 queue places existed. SGLang applies the limit when a request arrives, counting requests not yet scheduled, so a burst is refused while slots are idle. The limit is only safe far above any burst. The bound on the queue is better enforced with a counter at the gateway (see `notes/queue-experiment.md`, note of this date).
- **m1b was inconclusive for priority:** the batch wave left 4 places free, so the 4 interactive calls (TTFT 0.2-3.1 s) did not need to jump anything. To be repeated with the engine full.
- **m3 (client leaves while queued): the queued request disappeared within 3 s** (queued 1 → 0) while all 12 places stayed busy: the abort path works for queued requests (confirmed in detail in the section below).
- **m2 did not run:** my script picked the LiteLLM pod that was still terminating after a rollout (exit 137). Pods must be selected by `Running` and newest creation time.

## Does LiteLLM close the upstream connection? (2026-10-02, `metrics/probes/connection_close.py`)

Setup: idle engine, `gate` variant with an unbounded engine queue and a 4 s first-token cut. Evidence is what SGLang does afterwards, not the HTTP status.

| Scenario | Result |
| --- | --- |
| **s1** streaming client leaves at ~3 s while running (`max_tokens` 800) | SGLang `running` 1 → 0 within ≤0.5 s of the disconnect (a zombie would run ~17 s); no tokens credited |
| **s2** non-streaming client leaves while running | `running` 1 → 0 within ~1.5 s; 215 of 800 tokens generated |
| **s3** client leaves while its request is queued in SGLang | the queued request disappears (queue 3 → 2) with all 12 places still busy |
| **s4** LiteLLM itself cuts a queued interactive call at `stream_timeout` = 4 s | the client gets a clean **HTTP 408** `litellm.Timeout: Request timed out` after **4.2 s**, before any stream byte; SGLang **never prefilled its 1,200-token prompt** and, when a place freed at ~12 s, gave it to a lower-priority batch request instead (the cut request would have been first in line): it was not served to nobody |

- **Conclusion: LiteLLM does close the upstream call**, both when the client leaves and when its own timeout fires. A cut request does not become a zombie. The disconnect path (`cancel_on_disconnect: true`) is safe to rely on.
- **The cut arrives as 408, not 503.** Under the status-code table in `ARCHITECTURE.md` a 408 never overflows, so it needs mapping to a 503 (with `shed-reason: ttft_cut`) before it can feed the overflow. Not done yet; candidate: `async_post_call_failure_hook` returning an `HTTPException`, to be tested.
- **My own script misled me once:** its verdict used `sglang:generation_tokens_total`, which SGLang increments when a request *finishes*, so it read 0 in all four scenarios and printed "the cut removed the request" before there was evidence. The conclusion above rests on the log (no 1,200-token prefill; the freed place went to a batch request), not on that counter.
- **SGLang's gauges are stale by several seconds:** `num_queue_reqs` / `num_running_reqs` change in steps (every ~3-10 s under decode load), not continuously. An abort looked like a 3-5 s lag in s3 and s4 although the request was already gone. Any admission rule that reads the engine queue (the estimate rule, the gate) works on data that is old by that much plus the poller's 15 s.

## Multimodal support (2026-10-02)

- The guard used to refuse **every** list-valued `content` ("Multimodal content is not enabled"), which also refused text-only messages sent as `[{"type": "text", ...}]` by clients that use content parts: a risk for pi, not yet checked against pi itself.
- Qwen3.5-9B is multimodal (the log shows `Qwen3_5ForConditionalGeneration`, 27 vision blocks in the quantization config), so the policy changed: text and `data:` images are accepted, with limits (see `ARCHITECTURE.md` decision [33](../ARCHITECTURE.md#decision-33)).
- **Live results through the gateway:** a 64x64 red PNG: HTTP 200, answer "Red.", 89 prompt tokens; a 512x512 blue PNG: HTTP 200, "Blue", 281 prompt tokens (text prompt included); an `http://` image URL: HTTP 400; text parts as a list: HTTP 200. An image therefore costs on the order of a few hundred prompt tokens at these sizes; large screenshots were not measured.
- Remote image URLs are refused because the engine would fetch them from inside the cluster (SSRF); the client must send a `data:` URI.

## Queue experiment, reference runs `r0` (2026-10-02; 240 s runs, 60 s warm-up, profile `paper`, pause 20 s, current gateway queue)

Data: `metrics/runs/eq-ref-n20/`, `metrics/runs/eq-ref-n28/`. Short runs: they rank, they do not certify.

| | N = 20 | N = 28 |
| --- | ---: | ---: |
| served per minute | 44.7 | **29.0** |
| TTFT p50 / p99 of served calls | 7.0 / 12.5 s | 12.6 / **20.0 s** |
| shed rate (all `timeout_queue`) | 6.3 % | **56.5 %** |
| how long a refused user waited to be told no (p95) | 10.8 s | 10.2 s (all of them > 5 s) |
| calls abandoned after 2 retries | 0 | 23 of 110 |
| engine running / queue / KV (max) | 6 + 6 / 0-1 / 55 % | 6 + 6 / 1-2 / 63 % |

- **Hypotheses.** H2 holds: the failure shows up as sheds while served TTFT p99 stays inside the SLO (12.5 s at N=20). H1 is slightly optimistic: N=20 already sheds 6 % (about 2 % in the last two minutes, once the cold start is over); the knee for this profile is near N ≈ 18-20.
- **The live override works:** the sheds are `timeout_queue`, the reason of the 64-place gateway queue; with the limits left at the counter's `GATEWAY_QUEUE_MAX=0` they would have been `decode_capacity`.
- **Goodput falls with load** (44.7 → 29.0 calls/min). It is not saturation, it is lost efficiency, and the cache explains it:

| Prompt tokens served from… | N = 20 | N = 28 |
| --- | ---: | ---: |
| GPU cache | 28.5 % | 24.1 % |
| host RAM | 34.3 % | 8.8 % |
| Mooncake | 10.0 % | 14.4 % |
| **recomputed (new prefill)** | **27.1 %** | **52.8 %** |
| share of a within-turn call served from cache | 84 % | 52 % |
| same for the first call of a turn | 61 % | 36 % |

- **More than half of the prompt is recomputed at N=28**, so TTFT is dominated by prefill, not by waiting for a place: the queue length K cannot fix a cost that comes from cache misses. The suspected cause is routing: `least-busy` sends the calls of one session to any worker, so the history cached on one worker is useless when the next call lands on the other (a chance of about 1 in 2 per call with two workers). The load generator now records which worker served each call and reports `worker_stickiness` (share of consecutive calls of a session that stay on the same worker) to test this; the K sweep below is the first run that carries it. If it is about 50 %, prefix/session affinity (`control/place.py`) is the biggest single lever and it should come before tuning K.
- A refused user waits ~10 s to be refused: the gateway queue holds a request for half the deadline and only then says no. A counter (`K`) refuses at once.

## Queue size K at overload (2026-10-02; N = 28, 240 s runs; `metrics/runs/eq-k-n28/`, reference `eq-ref-n28`)

| Queue | served/min | TTFT p50 | TTFT p99 | shed | wait to be refused (p95) | worker stickiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r0` (current gateway queue, waits up to 10 s) | 29.0 | 12.6 s | 20.0 s | 56.5 % | 10.2 s (all late) | – |
| K = 4 (16 places) | 24.0 | 12.3 s | 23.1 s | 87.2 % | **0.1 s** | 0.64 |
| K = 10 (22 places) | 27.7 | 22.9 s | 36.2 s | 58.5 % | 0.1 s | 0.58 |
| K = 16 (28 places) | 28.7 | 28.2 s | 44.6 s | 0 % | – | 0.74 |
| K = 24 (36 places) | 20.7 | 39.9 s | 60.3 s | 0 % | – | – |

- **Goodput does not depend on K** (24-29 calls/min; K = 24 is lower only because its queue was so deep that fewer calls finished inside 240 s). **A longer queue adds latency, not capacity**: TTFT p50 rises from 12 s to 40 s while the served rate stays flat. This is what a queue does beyond saturation.
- **Hypothesis HQ1 (1.65 s per queue position) is refuted.** Between K = 4 and K = 24 the TTFT p50 grows by ≈ 2.7 s per queued position per worker on average (3.5, 1.8, 2.9 over the three steps), not 1.65 s. The model used the nominal service rate (12 slots / 9.9 s per call = 1.2 calls/s); at N = 28 the fleet only finished **0.47 calls/s**, because every call costs much more than in the smoke test (52.8 % of the prompt is recomputed, see the reference runs). **HQ2 is refuted too**: K = 10 gives p99 36 s, not 12-16 s; no K meets p99 ≤ 20 s at this load. HQ3 holds in its first half (flat goodput) and fails in the second (sheds fall with K, but only by turning refusals into latency). HQ4 and HQ5 were not tested here.
- **Where K does help is the quality of the "no"**: a counter refuses in 0.1 s, the current queue refuses after 10 s. For the same served rate and a similar TTFT, K = 4 tells a user to go elsewhere instantly instead of making them wait ten seconds.
- **`worker_stickiness` is 0.58-0.74** (a coin toss between two workers would give 0.5): between a quarter and two fifths of the follow-up calls of a session land on the other worker than the previous one. That is part of the cache misses, not all of it: the within-turn cached share was only 52 % at N = 28.
- **The lever is the cost of a call, not the queue.** The per-call cost is dominated by recomputed prefill. Candidates, in the order of effort: (1) the radix eviction policy is `lfu` (set in the worker args), which keeps frequently used prefixes and evicts a session's recent history first, the opposite of what an agent loop needs; `lru` needs one worker restart to test; (2) session/prefix affinity in placement (`control/place.py`); (3) the chunk size and prefill/decode interference; (4) cache capacity (`--hicache-size` 32 GiB per worker, Mooncake 16 GiB).
- **Consequence for the design:** the queue allowance should be small (K of the order of 0-4) with an immediate 503 for the rest, and the overflow takes what is refused; the SLO is then met by keeping the per-call cost down, which is the cache work above.

## pi against the cluster (2026-10-02, pi 0.74.2, `app/run.sh`)

- **pi's global config would have bypassed the gateway**: `~/.pi/agent/settings.json` has `defaultProvider: superlinked` (the external API). `settings.json` also repeats a `providers` block that disagrees with `models.json` (context window 32768 vs 65536). The `app/` directory carries its own config and selects the provider explicitly.
- **pi asked for more output than the guard allows**: its default `maxTokens` is 16384, the guard caps output at 4096, so the first request was refused with `400 Maximum output is 4096 tokens`. The app config sets `maxTokens: 4096` (the product cap stays).
- **It works end to end**: `Reply with the single word: ready` -> `ready`; creating `fib.py` and running it -> `fib(10) = 55`, 20 s, 8 model calls, no guard rejection and no admission shed.
- **The agent loop reuses its history**: SGLang prefilled 1,578 new tokens on the first call and then only 37-264 new tokens per call with 1,536 -> 2,560 cached. **pi's real system prompt plus tools is ~1.5K tokens, about 6x smaller than the 10K shared prefix assumed by `loadgen.py`** (which came from the Copilot paper). Real pi contexts grow with the files it reads; the synthetic profile is the heavier case.
- **The local tunnel was dead**: `localhost:4000` listened (an `ssh -L` process) but the server-side `kubectl port-forward` had stuck to a LiteLLM pod that no longer existed after a restart (HTTP 000). `setup/start_forward.sh` on the server fixes it; it has to be rerun after every LiteLLM restart.
- `pi --help` shows `PI_CODING_AGENT_DIR` (config directory) and `PI_CODING_AGENT_SESSION_DIR`; `apiKey` in `models.json` can be a literal, an environment variable name or a `!command`.

## Engine configuration review: chunked prefill, speculative decoding, topology (2026-10-02, analysis only, nothing changed)

Asked by the user while the K experiment ran: is the chunked-prefill setting right, would speculative decoding be worth it, and is the architecture sound given two replicas on one GPU sharing a cache hierarchy (L1 GPU per worker, L2 host RAM per worker, L3 Mooncake shared). Evidence is the existing runs (`records.jsonl`: decode rate = completion tokens / (total - TTFT)).

**Measured: decode collapses under load.** Median decode speed per sequence of served calls (calls with at least 100 output tokens):

| Run | Decode tok/s per sequence (p50 / p10) | Median uncached prompt per call | Cached share of the prompt |
| --- | ---: | ---: | ---: |
| smoke, N = 3 | **48.8** / 45.6 | 2,109 | 86.7 % |
| `r0`, N = 20 | 26.6 / 18.6 | 3,214 | 72.9 % |
| `r0`, N = 28 | **15.4** / 9.8 | 32,431 | 43.5 % |
| K = 4, N = 28 | 15.0 / 8.8 | 36,244 | 32.3 % |
| K = 16, N = 28 | 15.1 / 9.5 | 19,835 | 33.9 % |

- This is the **mechanism behind the failed capacity model**. A call that decodes ~350 tokens at 15 tok/s takes ~23 s plus its TTFT, ~36 s in total, not the 9.9 s measured when idle; 12 slots / 36 s ≈ 0.33 calls/s against 1.2 nominal (observed 0.47 because many calls are cheaper). The slowdown is the interference of long prefills with decode in the same engine, and the more prefill is recomputed (cache misses) the worse it gets.
- TTFT under load has a **floor of ~7-8 s even for calls with only 500-8,000 uncached tokens** (queue wait included) and grows to a median of 12.5-14.9 s above 8,000 uncached tokens (p90 ≈ 35 s).

**Chunked prefill (`--chunked-prefill-size 4096`).** Necessary in principle: without chunking a cold 45K-token prefill (~4-5 s) would hold the whole worker and stop every running decode. But 4096 is not proven right for this load: each engine step carries a 4096-token chunk (~0.3-0.5 s) and the six running sequences emit one token per such step while a prefill is in progress, which is consistent with the 3x drop above. The trade-off is TTFT of the new request (larger chunk, faster prefill) against decode speed of the running ones (smaller chunk, less stall). Hypotheses (to test, not now): **HC-1** a smaller chunk (1024-2048) raises decode tok/s under load and the served rate, at a modest TTFT cost; **HC-2** `--enable-mixed-chunk` (prefill and decode in the same batch) gives most of that gain without a smaller chunk; **HC-3** the best chunk matters less once the cache hit rate is fixed, because there are fewer prefills to interleave. Test: N = 20 and 28, chunk in {1024, 2048, 4096, 8192} plus mixed-chunk, one worker restart each; metric: decode tok/s p50, served/min, TTFT p99.

**Speculative decoding.** Decode is 93 % of slot time when idle and almost all of it under load, so a faster decode raises capacity directly (saturation N = slots / occupancy). Qwen3.5 has a native multi-token-prediction head (the quantization config lists MTP layers) and SGLang documents MTP for these hybrid models together with the second-generation Mamba radix cache. Expected effect (estimate, not measured): 1.5-2x tokens per step per sequence, because decode here is memory-bandwidth-bound (the 17.6 GiB of weights are read each step at the instance's ~1 TB/s; 19 ms ≈ the measured 48 tok/s). Costs and risks: (1) every draft token needs its own recurrent-state slot, ~47 MB each, so 3 draft tokens x 6 sequences ≈ 0.85 GiB per worker taken from a pool that is already tight (KV 13.3 GiB, 46 states); (2) it needs `--mamba-scheduler-strategy extra_buffer` and a larger page size, which changes cache semantics and has **not been shown to work with the HiCache/Mooncake hop we proved** with the default strategy; (3) the extra MTP weights. Verdict: promising for capacity, but it puts the proven hop at risk, so it is worth one controlled experiment **after** the cache work, with the hop probe (`metrics/probes/probe5.py`) re-run as the gate. Hypothesis **HS-1**: decode tok/s per sequence x1.5 or more at N = 20; **HS-2**: the hop still works; if HS-2 fails the feature is not adopted.

**Topology: two MIG instances on one GPU.** What the split costs, against a single worker on the whole GPU (columns marked "estimate" come from first principles, not from a run):

| | Two workers (now) | One worker, whole GPU (estimate) |
| --- | --- | --- |
| Weights in HBM | 2 x 17.6 GiB = 35.2 GiB (44 % of 79.6) | 17.6 GiB (22 %) |
| KV pool | 2 x 13.3 GiB, two separate caches | ~45 GiB in one cache (about 3.4x), no cross-worker misses, no hop needed |
| Memory bandwidth per worker | ~1 TB/s (half) → decode ≈ 48 tok/s measured | ~2 TB/s → decode ≈ 95 tok/s |
| SMs used | 92 of 114 (19 % idle) | 114 |
| Prefill speed | ~8.5-11K tok/s measured | ~25K tok/s (≈ 114/46) |
| Cache routing problem | `worker_stickiness` 0.58-0.74, a quarter to two fifths of follow-up calls miss their history | none |
| Failure isolation, placement and hop | yes (what the course asks to show) | no |

- **Honest reading:** for production on one H100 the better architecture is one worker; the two-worker layout is justified by the course (at least two workers, placement, hop) and by failure isolation, and its cost is large and measurable: a lower cache hit rate, half the decode speed per sequence, duplicated weights. That should be said in the presentation, with the table above and ideally one measured data point (optional experiment **E-W**: reconfigure MIG to one `7g.80gb` instance, ~15 min, same load sweep).
- **Prefill/decode split across the two instances** (one prefill-only, one decode-only, KV moved through Mooncake) is the other option the course lists and it removes the interference directly. It looks attractive here because the damage comes from prefill stalling decode, but the decode instance must hold all KV of the running sequences and the prefill instance becomes the bottleneck when the hit rate is low (≈ 3 s of prefill per call at 32K uncached tokens, ~0.33 calls/s per prefill instance). Hypothesis **HP-1**: it beats the colocated layout only if the cache hit rate is good (median uncached per call below ~8K tokens). Not now.
- **Cache hierarchy:** the host tier is small against the machine (32 GiB per worker of 221 GB RAM) and the radix eviction policy is `lfu`; at N = 28 the host tier served 8.8 % of the prompt against 34.3 % at N = 20. Both are cheap to change (`--hicache-size`, `--radix-eviction-policy lru`) and come before the options above.

**Queue of engine experiments (in this order, one variable each):** E-L `lfu` -> `lru`; E-C chunk size and mixed chunk; affinity in placement; then, optionally, E-S speculative decoding (gated by the hop probe), E-P prefill/decode split, E-W single worker.

## Placement: does the shared cache make session affinity unnecessary? (2026-10-02, evidence so far)

The user's argument: with a hierarchical cache shared by both workers (host RAM per worker, Mooncake across them) routing to the worker that holds the prefix is not worth a custom router; LiteLLM's `least-busy` is enough if it is justified. Data (the four K runs at N = 28, calls inside a turn, previous call of the session a few seconds earlier; `metrics/runs/eq-k-n28`):

| The call lands on… | Calls | Prompt served from cache | Median recomputed tokens |
| --- | ---: | ---: | ---: |
| the same worker as the previous call | 170 | 44.3 % | 16,424 |
| the other worker | 100 | 26.7 % | 32,209 |

- The shared tiers do rescue part of the history (26.7 % cached on the other worker, not ~0), so the argument has support. They do not equalise it: 17.6 points less cache and twice the recomputed tokens.
- **Not conclusive.** These runs used `lfu` and a long queue (TTFT ~25 s for both groups, so the TTFT difference is hidden), and sessions that switched worker may differ from those that did not. After the move to `lru` the gap may change.
- **Cheap test, planned after the queue number and `lru`:** same conversations and load, 4 min each, `routing_strategy` `least-busy` (now), `simple-shuffle`, and a ~25-line session-affinity hook. Decision rule fixed beforehand: adopt affinity only if it improves served per minute or TTFT p99 by more than the run-to-run noise; otherwise keep `least-busy` and report this comparison as its justification. `least-busy` already uses in-flight requests per worker as its score, so queue depth is a placement criterion and not only an admission input.
- Changing the strategy is a one-line change of `router_settings.routing_strategy` in `cluster/litellm/config.yaml` plus a LiteLLM restart (~1 min).

## Overflow and where `fallbacks` acts (2026-10-02)

`router_settings.fallbacks` (commented out in the repo config, to be enabled with the overflow gate) lives at the router level, so it only fires when the **call to a deployment fails** (an SGLang 503/529, a connection error). Requests refused by our admission (instant 503 `decode_capacity`, `kv_pressure`, `timeout_queue`, `ttft_cut`) are refused in a pre-call hook, before the router, and never reach that fallback. Consequence for "who receives a 503": with fallbacks alone only engine-side failures leave; admission sheds return a 503 to the client. Accepted for now (the user's reading: the request had already been admitted); sending admission sheds to the overflow would be a small hook change (rewrite the model alias to the overflow deployment for capacity reasons, never for 429/500, only under a context cap). Not done.

## Choosing K: the decisive sweep around the knee (2026-10-02; K = 0, 2, 4, 6 at N = 24 and N = 20; `metrics/runs/eq-k2-n24`, `eq-k2-n20`; plots `plots/queue_k_n24.png`, `plots/queue_k_n20.png`)

| K | N = 24: served/min | TTFT p99 | refusals | calls abandoned | N = 20: served/min | TTFT p99 | refusals | calls abandoned |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 37.7 | 7.2 s | 76 % | 45 % | **45.7** | 6.2 s | 61 % | 23 % |
| 2 | **40.0** | 15.7 s | 71 % | 38 % | 43.3 | 12.3 s | 48 % | 14 % |
| 4 | 37.0 | 18.3 s | 62 % | 28 % | 36.3 | 17.1 s | 23 % | 6 % |
| 6 | 29.0 | **29.4 s** | 65 % | 30 % | 30.7 | **28.2 s** | 4 % | 1 % |

("refusals" are attempts, including the load generator's two retries after each refusal; "abandoned" are calls that never got served after the retries. Every refusal is immediate: `shed wait p95` 0.1-0.5 s.)

- **Result of the rule fixed beforehand** ("the largest K whose served TTFT p99 is ≤ 20 s at both N"): **K = 4** (18.3 s at N = 24, 17.1 s at N = 20); K = 6 fails at both (29.4 s, 28.2 s).
- **Hypotheses.** HK1 (p99 grows linearly with K): holds, about +2.9 s per unit of K at N = 24 and +3.5 at N = 20; the intercept is lower at N = 20 (6.2 s) than at N = 24 (7.2 s) and at N = 28 (≈ 15.7 s extrapolated earlier, which was too high: K = 0 measures 7 s). HK2 (K = 0 has the lowest p99 and the most refusals, goodput independent of K): the p99 and refusal parts hold, **goodput does depend on K and falls as K grows** (45.7 → 30.7 at N = 20). HK3 (K ≤ 4 at N = 20 and K ≤ 2 at N = 24 keep p99 ≤ 20 s): K ≤ 4 holds at both, so the N = 24 case was less strict than predicted.
- **Why goodput falls with K:** a queued request still needs a long prefill and slows the decoding of the others, so a longer queue keeps the engine in the slow regime (decode 15-17 tok/s) longer. With a small K the engine stays out of it and finishes more calls per minute; the refused users retry within seconds.
- **The trade-off is real and has two sides:** a larger K means fewer calls abandoned (K = 4: 6 % at N = 20; K = 0: 23 %) but a lower served rate and a TTFT closer to the SLO. K = 6 buys almost no abandonment relief at N = 20 (1 %) for a p99 of 28 s.
- **Recommendation: K = 2** rather than the rule's 4, for three reasons visible in the table: (1) the same goodput as K = 0 over the two loads (83.3 calls/min added up) with 5-14 points fewer refusals; (2) a p99 margin of 4-8 s under the SLO against 2-3 s for K = 4, which with short single runs and no repeat is inside the noise; (3) with the first-token cut as the backstop, the margin decides how many calls get cut late. The pre-registered rule did not contain goodput because HQ3 expected it to be flat; that assumption failed, so a dated note in the protocol records the deviation. K = 4 stays as the upper bound that still meets the SLO. The final value is confirmed with one full 8-minute run.
- **Engine check:** the maximum SGLang queue per worker followed the counter (K/2 per worker), so the gateway counter bounds the engine queue as designed.

## Live checks of the cut and of placement (2026-10-02, K = 2 variant, `metrics/probes/cut_mapping.py`, `placement_check.py`)

- **First-token cut, warm workers:** 12 long calls fill the running places and a 13th is admitted into SGLang's queue. It is cut and answered **HTTP 503, `shed-reason: ttft_cut`, after 4.2 s** (`TTFT_CUT_S=4`; body "No first token within 4s"), the 12 long calls all finish with 200, and `orch_requests_shed_total{reason="ttft_cut"}` goes from 0 to 1. The 408 → 503 mapping therefore works in the cluster.
- **A cold worker trips the cut:** the first attempt ran just after the workers restarted; six of the calls were cut because a freshly started worker needs more than 4 s for a first token. This is the "ready ≠ warm" problem again and the reason warm-up requests belong in the readiness procedure.
- **Placement through the aliases:** under `aff` and `affload`, 8 sessions x 4 calls were spread over both workers (6/2 and 5/3) and every session stayed on one worker; all statuses 200.
- My script printed nothing for the placement metrics (no `-i` on `kubectl exec`), so `litellm_orch_placement_*` is checked in the Prometheus snapshot of the routing experiment instead. A first run of the `affload` check died with exit code 137 because it entered a LiteLLM pod that was terminating after a rollout; the rerun waits for the new pod to be Ready.

## Direct test of the overflow endpoint (2026-10-02, from this computer, `curl`, key from `.env` not printed)

- `GET /v1/models` -> **200**, 60 models. The overflow model is listed as **`Qwen/Qwen3.8-27B-FP8`** (plus variants `:h100-256k`, `:no-spec`, `:thinking`, ...), the same id as in `cluster/litellm/config.yaml`.
- Chat with the id `Qwen3.8-27B-FP8` (no `Qwen/` prefix) -> **404 `model_not_found`**; with `Qwen/Qwen3.6-27B-FP8` -> 404 (not offered).
- Chat with `Qwen/Qwen3.8-27B-FP8`: the first call produced **no answer in 60 s** (curl timeout, probably a cold start of the model); the second answered **HTTP 500 `transport_failure`: "Inference completed without a valid conserved billing settlement"**, `credits_charged: 2`, after 3.4 s. The error comes from Superlinked's billing/settlement step, not from our request or key (the key is accepted and credits are charged).
- Consequence: the overflow path is not usable today, independent of our gateway. It also confirms why the gate keeps **500 on the stay side**: an external 500 must not be treated as capacity. The gate still measures the share that would leave (`OVERFLOW_ENABLED=0`); a live overflow test waits until the endpoint answers 200.

## Routing experiment at N = 24 (2026-10-02, `metrics/runs/er-n24/`, protocol `notes/routing-experiment.md`)

Six runs, 240 s each, K = 2, `lru`: served per minute / TTFT p99 / cached share of the prompt / stickiness: `lb` 40.0 / 8.9 s / 73 % / 0.52; `lb` repeat 40.0 / 11.1 s / 74 % / 0.50; `shuffle` 48.3 / 14.0 s / 78 % / 0.53; `latency` 29.7 / 30.7 s / 76 % / 0.49; `aff` 40.0 / 27.1 s / 83 % / 0.96; **`affload` 54.0 / 9.2 s / 87 % / 0.95**. Figure: `plots/routing_n24.png`.

- LiteLLM's `least-busy` keeps a session on its previous worker only half of the time (0.50-0.52, the same as random choice): it balances load but ignores the session, so every other follow-up call lands where its history is not.
- Keeping the session on its worker (`affload`) raises the cached share from 73 % to 87 % and the served rate by 35 %; a pure hash (`aff`) gets the cache but loses the balance (60/40 split, one engine queue of 7) and its tail, so the load term of `affload` is what makes affinity usable.
- `latency-based-routing` is the worst: it chases the latency its own choices create (engine queues 8/8, p99 30.7 s).
- The shared cache hierarchy narrows the penalty of a miss (cached share on the other worker is 60-74 % here against 27 % before `lru`), but it does not remove it: recomputation was 27 % of the prompt with `lb` and 13 % with `affload`.
- Next: N = 20 for `lb`, `shuffle`, `affload` to confirm near the knee.

## Tenant limits, a cold worker, and the hop dictionary (2026-10-02, after the second deployment)

- **Tenant limits (429), `metrics/probes/tenant_probe.py`, log `metrics/logs/tenant_probe.log`:** with `TENANT_MAX_CONCURRENCY=2`, tenant `acme` sent 5 long calls at once: **2 answered 200 and 3 answered 429 `tenant_concurrency` in 0.3-0.4 s**, while tenant `beta`, at the same time, was served (200). Counters: `orch_requests_shed_total` with reason `tenant_concurrency` and status 429: 3; `orch_tenant_requests_total` acme admitted 2 / refused 3, beta admitted 1; and the overflow gate decided `stay` for all three (`orch_overflow_decisions_total` decision `stay`, status 429: 3): a 429 never leaves.
- **A cold worker is slower, and the warm gate sees it (`metrics/probes/warm_after_restart.json`):** worker 1 was restarted and measured as soon as Kubernetes called it ready: first 4K-token prompt **1.76 s cold**, then warm-up rounds of 1.06, 0.33 and 0.29 s, then **0.32 s warm** (5.5x faster). The earlier gate result (0.30 s cold, 0.29 s warm) was on workers that had been up for 140 minutes. So "ready" is not "warm", and quoting the TTFT of the first request after a restart would overstate the SLO number by about 5x at this prompt size.
- **Hop dictionary (`metrics/evidence/after-ev-n16/hop_dictionary.jsonl`)**, from a 200 s run with `lb` at N = 16: 40 hops (a call served by a different worker than its session's previous one), 1,815,245 prompt tokens, of which 1,488,194 (**82%**) were read from cache on the destination: a hop moves ~45-50K tokens of history and the destination reads 82 % of it from the cache tiers instead of recomputing it (the other 18 % is recomputed: the part of the history that was not yet published). Fields: `src`, `dst`, `tokens`, `cached_tokens`, `prefix` (hash of system+tools), `session`, `backend` (`mooncake`).
- **Eviction counters** (`metrics/evidence/after-ev-n16/evict_count.txt`): `sglang:evicted_tokens_total` 12.5M tokens on worker 0 (worker 1 restarted: 0.55M); `hicache_dropped_tokens_total` is 0 for both reasons (`host_pressure`, `write_through_unbacked_eviction`): nothing was dropped without a lower-tier copy.
- **429 and 503 counts** (`status_429_503.txt`): 18 `decode_capacity` 503s in that run (2.7 % of the attempts) and no 429 outside the probe.
- That run (`metrics/runs/ev-n16`, `lb`, N = 16): 45.9 served/min, TTFT p99 6.4 s, 2.7 % refused.

## Routing experiment at N = 20 (2026-10-02, `metrics/runs/er-n20/`, figure `plots/routing_n20.png`)

`lb` 53.7 served/min, p99 9.5 s, cached 82 %, stickiness 0.55, 30 % of attempts refused (9 abandoned calls); `shuffle` 41.7, p99 19.3 s, 51 % refused (18 abandoned), engine queue 1/7; `affload` 54.0, **p99 7.6 s**, cached **88 %**, stickiness 0.88, but 42 % refused (19 abandoned).

- Near the knee the policies tie on throughput; the gain of `affload` is in the tail and the cache, and it comes with more refusals. At N = 24 it won by 35 %: the benefit of affinity grows with overload, which is when recomputing a missed prefix costs the most.
- `shuffle` is the only one that clearly loses at both loads (random placement leaves one engine queue at 7 while the other is at 1).
- Open: whether the extra refusals of `affload` are real (a repeat at N = 20 would tell) and why (hypotheses in `notes/routing-experiment.md`).

## First scrape of the deployed control plane (2026-10-02 20:18, during the first N = 20 run, policy `lb`)

The deployment (`launch_cluster.sh`, 20:13-20:14) put the new code in the cluster; `litellm_orch_*` on `/metrics/` (the path needs the trailing slash, `/metrics` answers a redirect) showed:

- **Hops:** `orch_hops_total` 22 (w0 -> w1) + 22 (w1 -> w0) against `orch_hops_local_total` 19 + 21: with `least-busy`, 52 % of the follow-up calls (44 of 84) ran on the other worker than the session's previous call, the same figure as the stickiness of 0.52 measured from the client side. The hop calls carried 902,729 + 916,311 prompt tokens, of which 766,316 + 758,871 (**85 %**) were read from cache (`orch_hop_cached_tokens_total`): the hierarchy makes a hop mostly warm, but 15 % of those tokens were recomputed.
- **Overflow gate:** `orch_overflow_decisions_total{decision="leave",forwarded="no",status="503"}` = 129 = exactly the `decode_capacity` sheds (129): every refusal was a candidate to leave and none was forwarded (forwarding is off).
- **Tenants:** `orch_tenant_requests_total{tenant="default_user_id",outcome="admitted"}` 259: without an `x-tenant-id` header every caller is the master key's user, so tenants are not separated; the header (used by `metrics/probes/tenant_probe.py`) names them.
- **Places:** `orch_gateway_in_flight` 12 of `orch_gateway_capacity` 14 (batch limit 10).
- **Alerts:** Prometheus loaded the four rules (`/api/v1/rules`, all `health: ok`); under this load `CapacityShedRateHigh` and `InteractiveTTFTSLOBreach` were **pending**, the other two inactive.
- **Warm gate** (`metrics/probes/warm_workers.json`): cold 0.297 s / warm 0.292 s on worker 0 and 0.303 / 0.296 s on worker 1: both passed, but the workers had been up for 140 minutes, so cold and warm do not differ. A real cold-to-warm contrast needs a freshly restarted worker (to do).

## Review of the guard and of admission (2026-10-02, reading `control/inspect.py` and `control/admission.py` against what the runs showed)

**Guard (`inspect.py`): what still makes sense.** Model alias allow-list (only `qwen-coding-local`, so a client cannot pick a worker alias), message roles, content parts (text and `data:` images; remote URLs refused against SSRF), image count and size, tool count and schema size, the 4096 output cap, fail-closed 503. All of them are cheap, stateless and protect something real (GPU time, memory of the gateway, the engine). The context-window check stays with LiteLLM.

**Gaps found and fixed (unit-tested, with a mutation check):**
- **The priority was self-declared and unbounded.** Admission and the engine trust `priority`, so any client could send `priority=-1000` and jump the engine's queue (and, with `--retraction-policy priority`, get others retracted). Now a caller declares a class: 1-10 is accepted, anything else is a 400 `invalid_priority`, and every interactive value (<= 5) is normalised to 5, so inside the interactive class the order is arrival. Limit that remains: with one master key nothing proves that a caller is entitled to "interactive"; that needs a key or tenant per caller.
- **`n` > 1 (and `best_of`) cost several sequences but counted as one place.** Now a 400 `multiple_choices_not_supported`.
- **A 503 from the guard could have left for the overflow model.** Its fail-closed answer is a 503, and the overflow gate treated every 503 as capacity. The gate now keeps anything with `source: inspect` (a request the guard could not vet must not be handed to another model).

**Left as it is, on purpose or for later:**
- `MAX_TOKENS_POLICY=reject` answers 400 to pi's default 16384 (the app config sets 4096 to avoid it). `clamp` would serve such a request with 4096, which protects the same thing without failing a client that did nothing wrong; recommended, awaiting the user's decision.
- **`kv_pressure` and `batch_pressure` never fired in any run** (KV peaked at 55-60 %, 0 sheds): they are insurance and have no live evidence. A probe that sends unique 60K prompts to push KV past 95 % would show them (and the engine's retraction); not done.
- **Tenant limits are off by default**, because all our traffic is one tenant and a cap would shed our own load. "Stop one tenant owning the GPU" therefore needs limits per tenant (`TENANT_LIMITS`) once there are tenants; the mechanism is proven by the probe (2 x 200, 3 x 429).
- `DEADLINE_S` (interactive 20 s, batch 120 s) and a client-supplied `timeout` now only feed the optional `timeout_queue` estimate; the real 20 s bound is the first-token cut (`TTFT_CUT_S`), which applies to interactive only.

## Why not HAMi? (2026-10-02)

HAMi is a Kubernetes layer that gives containers a fraction of a GPU (`nvidia.com/gpumem`, `nvidia.com/gpucores`) by intercepting CUDA calls in the container (a software limit), plus a scheduler extender and an admission webhook. It is what the course's engine used to slice a GPU. We did not use it, by reasoning (it was **not tested**):
- **The GPU has MIG, which isolates in hardware.** A MIG instance gets its own share of HBM, memory controllers, L2 and SMs; two workers cannot slow each other through bandwidth or cache. HAMi limits memory and caps SM use over time, but the two workers still share the memory bandwidth, the L2 and the SM scheduler, so a prefill burst on one worker shows up as latency on the other. Every plot here (TTFT p99, decode tok/s) would carry that interference.
- **We already saw the software-sharing behaviour.** The first cluster used NVIDIA time-slicing (the closest relative of HAMi's sharing): no memory isolation, kernels interleaved in time, `MAX_MEM_FRAC = 0.485` per worker and a startup race ("loaded weights leave no GPU memory for the KV cache") when both workers loaded at once. HAMi would fix the memory cap, not the compute and bandwidth interference. Decision [4](../ARCHITECTURE.md#decision-4).
- **Less to run.** MIG is done by the NVIDIA device plugin we already have (`MIG_STRATEGY=single`); HAMi adds a scheduler, a webhook and its own device plugin to the cluster.
- **What MIG costs:** a fixed geometry (resizing means draining the GPU), and the `3g.40gb` x 2 layout uses 92 of the 114 SMs (about 19 % of the compute is left unused). HAMi could give each worker a share of all the SMs.
- **When HAMi is the better answer:** a GPU without MIG (L4, A10, L40S, RTX), more slices than MIG allows, or fractional requests that change often. On this H100 and with latency as the thing we measure, MIG is the cleaner experiment.

## Final load tests F1-F5 (2026-10-02, protocol `notes/final-load-tests.md`, runs `metrics/runs/fin-*`)

The final configuration (K = 2, `affload`, first-token cut 20 s, `lru`, bf16) under the synthetic sessions of `loadgen.py`:

| run | N | served/min | TTFT p50 | TTFT p99 | refused attempts | abandoned calls | cached share | stickiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fin-knee` | 16 | 47.0 | 1.1 s | 9.3 s | 3.7 % | 1 of 236 | 88 % | 0.918 |
| `fin-knee` | 20 | 56.8 | 1.7 s | 7.7 s | 38.4 % | 23 of 307 | 88 % | 0.924 |
| `fin-knee` | 24 | 46.4 | 2.3 s | 13.7 s | 70.1 % | 124 of 356 | 82 % | 0.803 |
| `fin-rep-affload` | 20 | 50.4 | 1.9 s | 10.3 s | 48.8 % | 33 of 285 | 88 % | 0.853 |
| `fin-rep-lb` | 20 | 43.6 | 2.7 s | 11.5 s | 53.0 % | 51 of 269 | 74 % | 0.54 |
| `fin-batch` | 20 | 43.2 | 0.9 s | 7.4 s | 1.4 % | 0 of 216 | 91 % | 0.965 |
| `fin-regimeB` | 48 | 38.4 | 4.5 s | 14.9 s | 84.7 % | 310 of 502 | 66 % | 0.884 |
| `fin-soak` | 16 | 48.57 | 1.0 s | 7.1 s | 1.7 % | 1 of 681 | 90 % | 0.935 |

- **The knee for "refusals below 5 %" is N = 16.** From N = 20 the system refuses 38-53 % of the attempts but still serves 44-57 calls per minute with TTFT p99 under 14 s: the cap (K = 2) keeps the served ones inside the SLO and pushes the rest back at once.
- **Paired at N = 20 (same conversations), `affload` beats LiteLLM `least-busy` on every number:** +16 % served, p99 10.3 s against 11.5 s, refusals 49 % against 53 %, cached share 88 % against 74 % (stickiness 0.85 against 0.54). The extra refusals seen for `affload` in the first N = 20 comparison were noise: two runs of the same policy at N = 20 differ by 11 % in served per minute when the conversations differ.
- **A session that rests loses its GPU cache (regime B).** With 120 s of idle between turns, only 27.5 % of the prompt of the first call of a turn is read from cache (88.7 % in regime A), and its TTFT p50 is 5.3 s; once the turn is under way the share is back to 71 %. This is the cliff the paper describes between 2 and 10 idle minutes, here at 2 minutes because HiCache has to bring the history back from host RAM or Mooncake. N = 48 open sessions is beyond the capacity (85 % refused).
- **Priority works, and `batch_pressure` is too eager.** With 20 % batch sessions the interactive ones are untouched (1.4 % refused, p99 7.4 s), while batch gets 98 % refusals: 373 by `batch_pressure`. The rule compares the fleet TTFT p99 with 4x the p50, and in this workload p99 is 7-8x p50 in normal operation (p50 about 1 s, p99 about 7 s), so it is almost always true. Batch is sacrificed while the interactive p99 sits at a third of the SLO.
- **The soak is flat.** 15 minutes at N = 16: 48.6 served/min, p99 7.1 s, queue <= 2 per worker, KV used <= 57 %, in flight reaching the cap of 14 a few times and back to 0 at the end (`plots/final_soak.png`, series in `metrics/evidence/final-soak/`).
- **KV is not the limiter at this load.** The pool never went above 62 % in any run, `kv_pressure` never fired and the engine never retracted: the hypothesis "KV first" of the capacity section was not confirmed (the limiter is the cost of each call under load, decisions [34](../ARCHITECTURE.md#decision-34) and [36](../ARCHITECTURE.md#decision-36)).

## After the simplification: the cap probe and the keys (2026-10-02, `metrics/logs/cap_probe.log`, `tenant_probe.log`)

- **LiteLLM's own cap behaves like the counter.** 14 long streams in flight (`litellm_admission_admitted_requests` = 14): the 15th request is refused with a **503 in 0.0 s**, body "Worker at capacity: 14 in-flight, 0 queued requests. Retry later." (no `shed-reason` header: the load generator now classifies that text as `queue_full`); when the 14 finish the gauge returns to 0 and a new request is served in 0.4 s (the place is released at the end of the stream).
- **The scrape was NOT refused with the cap full**: `GET /metrics/` answered 200 in 0.04 s while 14 were in flight, so the predicted blind spot (the middleware counting `/metrics`) did not happen in LiteLLM 1.105; `/health/liveliness` answered 200 as predicted.
- **Per-key limits (virtual keys on PostgreSQL):** `tenant-acme` (`max_parallel_requests` 3) sent 6 simultaneous calls: 3 x 200 and 3 x 429 in 0.02-0.03 s, while `tenant-beta` was served; the counters `litellm_proxy_total_requests_metric_total` carry `api_key_alias`.
- PostgreSQL costs one small pod; LiteLLM started in about one minute with the schema migration.

## The demo timeout: `APIConnectionError: Timeout on reading data from socket` (2026-10-03, user's first run of `app/demo/demo.sh`)

**Symptom.** pi fails whenever it reaches the point where the model writes a file; LiteLLM logs `aiohttp ... SocketTimeoutError: Timeout on reading data from socket` -> `litellm.APIConnectionError ... OpenAIException`, in `async_data_generator()` (the 200 had already been sent), repeating every 25-30 s (pi retries).

**Evidence.** (1) In the LiteLLM log six errors at 08:47:41-08:49:59. (2) In the worker log, for the request that failed at 08:49:59 (started 08:49:38 with 23,040 cached + 45 new tokens): the worker decoded continuously at 51.5 tok/s from 08:49:39 to 08:49:59 (`#full token` 23,120 -> 24,160, about 1,040 tokens) and the error fires at exactly 20 s: the model was working, nothing reached LiteLLM. (3) `metrics/probes/gap_probe.py`, direct to a worker with a `write_file` tool and a request for a 120-line file: first chunks at 0.07-0.43 s, then **no chunk for 20.59 s** (1,121 tokens), then the whole argument in one chunk (`metrics/logs/gap_probe.log`).

**Cause.** SGLang's streaming parser for the Qwen3.5 tool format (`--tool-call-parser qwen3_coder`) does not stream the arguments of a tool call: it sends the call's header and then stays silent until the argument is complete. LiteLLM's `stream_timeout` is not a time-to-first-token limit: it is applied as the **socket read timeout of the whole stream**, so any silent gap longer than `TTFT_CUT_S` (20 s) kills the call. A file of about 1,000 tokens takes 20 s to generate at 51 tok/s (a 4,096-token answer would be silent for 80 s idle and up to 270 s under load).

**Why the tests did not see it.** `loadgen.py` streams plain text token by token (no gap); `agentgen.py`'s tool calls are short (read_file/grep arguments; outputs p95 about 700 tokens, about 14 s); the first-token cut was verified with text. The demo is the first traffic that writes a large tool-call argument. The cut works for what it was verified for (no first token within 20 s -> 503 `ttft_cut`) but cannot be told apart, in LiteLLM, from a silent gap in the middle of a stream.

**Options.** (1) `TTFT_CUT_S=0`: the cut is off; the bound on TTFT then comes from K = 2 (at most 2 requests wait) and was already met without the cut firing (it fired once in the whole N = 24 run). (2) Raise it above the worst silent gap (4,096 tokens under load: about 270 s), which removes the cut. (3) Enforce the first-token limit differently (outside LiteLLM's socket timeout); not available in the hook API without holding the stream. Recommended: (1).

## KV full, and a worker that comes back (2026-10-03; `metrics/logs/kv_probe_*.log`, `ramp_probe.log`; protocol in `notes/final-load-tests.md`, decision [60](../ARCHITECTURE.md#decision-60))

- **The KV cannot overcommit with this sizing.** 12 requests of about 60K tokens at once: the worst worker peaked at 83.0-84.6 % of its pool, 0 retractions, all served. Six running sequences of 64K are 88 % of the 435K-token pool at most, so after admission the engine never needs to preempt; the 95 % threshold of `kv_pressure` is a backstop that running sequences alone cannot reach.
- **The shed path works, but the gateway sees the KV 15 s late.** With the threshold lowered to 0.70, a new prefix got a 503 `kv_pressure` in 0.1 s ("Fleet KV usage is 83 % and the prefix is new") and a reused prefix was admitted. The first attempt did not shed: the request was sent while the engine showed 71-75 %, but the gateway decides on the fleet snapshot polled every 15 s, which was still lower. Decisions on gauges are always late; the probe now waits for the snapshot.
- **A worker that dies and returns (V8):** 3 requests failed (the ones in flight), then nothing was sent to it for 4 minutes with no further error; when it answered again its share of new requests went 0.13 -> 0.36 -> 0.54 as the ramp factor went 0.25 -> 1.0, which took about 125 s because the clock waited ~45 s while the cold worker kept the fleet p99 above the 10 s hold (`plots/final_ramp.png`).
- **`kv_pressure` and `batch_pressure` now both have live evidence** (the first at a lowered threshold; the second fired 397 times in the batch runs).

## Superlinked overflow, probed again (2026-10-03)

- Four tiny calls from the server (key not printed). `Qwen/Qwen3.8-27B-FP8`: 500 `transport_failure`, "Inference completed without a valid conserved billing settlement", `credits_charged: 1`. `Qwen/Qwen3.5-4B`: **402 `INSUFFICIENT_CREDITS`, "the org wallet is exhausted"**. Variants `:thinking`, `:no-spec`, `:h100-256k`: 503 "Generation route is unavailable". `Qwen/Qwen3.6-27B-FP8`: 404 (not served there).
- **Consequence:** the 500 of 2026-10-02 was the same billing problem, not our configuration (URL, key and model id are valid). Overflow stays built with forwarding OFF; decision [62](../ARCHITECTURE.md#decision-62) records it together with the closing scope (no KEDA, no `stale_telemetry` shed).

## The demo, run end to end (`metrics/logs/demo_run.log`)

- `app/demo/demo.sh` with pi through the gateway: **71 s**, four files written into `app/todo-app/` (`index.html` 32 lines, `style.css` 166, `app.js` 203, `README.md`; `node --check` accepts `app.js`). The gateway counters moved by exactly what pi did: 2 requests admitted, both interactive, placed by `affload` on one worker (1 new session, 1 kept). Places in use afterwards: 0 of 14.
- Two problems found while running it, both fixed in `app/demo/demo.sh`: (1) `tunnel.sh status` only recognises tunnels it opened, so with the user's own `ssh -L` already holding ports 4000/3000/9090 the script failed with "Address already in use"; it now skips the tunnel when the gateway already answers on `localhost:4000`. (2) `pi -p` waits for end-of-file on stdin; started from a background job with an open stdin it sat idle for 10 minutes with no connection to the gateway. Run it from a terminal, or with `< /dev/null`.
- The run used the master key because this machine's `.env` has no `LITELLM_KEY_PI_DEMO` (the script warns); `python3 script/make_keys.py` creates pi's own key.

## Open questions (state at the end of the notebook)

| Question | State |
| --- | --- |
| How many sessions does the system sustain before the SLO breaks? | Measured near N = 16-24 in the routing and final tests (`metrics/runs/fin-*`, `notes/final-load-tests.md`). |
| What does a GPU hour buy per session and per million tokens, and when is overflow cheaper? | Answered in ARCHITECTURE "Cost model"; to refresh with the final runs (`script/cost_report.py`). |
| How long does the recurrent state take to reach Mooncake under real traffic? | Visible in the hop dictionary (82 % of the hop tokens were cached on the destination); not measured as a time. |
| Does LiteLLM's `stream_timeout` bound the time to first token? | **Yes**, verified live: 503 `ttft_cut` after 4.2 s with a 4 s cut ("Live checks of the cut"). |
| Does the overflow endpoint work? | **No** on 2026-10-02 (HTTP 500 from Superlinked's billing step); repeat when it answers 200. |
| Are the extra refusals of `affload` near the knee real? | Checked by the replicate in the final tests (`fin-rep-affload` against `fin-rep-lb`). |

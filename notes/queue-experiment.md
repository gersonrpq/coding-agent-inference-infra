# Experiment EQ: how long can the queue be and still meet the TTFT SLO?

**Status:** DONE. Result: K = 2 (decisions [42](../ARCHITECTURE.md#decision-42)-[43](../ARCHITECTURE.md#decision-43)). The code it varied (the gateway queue, `ADMISSION_MODE`, `GATEWAY_QUEUE_MAX`, the variants `q0` and `gate`) was removed afterwards (decision [46](../ARCHITECTURE.md#decision-46)); the runs `metrics/runs/eq-*` and this protocol stay as the evidence. Terms: [`glossary.md`](glossary.md).

Protocol written **before measuring** (2026-10-02). Hypotheses and decision rules are not edited after seeing data: later changes go in dated notes at the end. Results in `metrics/runs/`, interpretation in `notes/findings.md`, decisions in the log of `ARCHITECTURE.md`. It replaces the "Q1/Q2" sketch in `notes/sweep-experiment.md` section 9 with the design the user described; the sweep protocol there (A1/A2/B) still defines the regimes and the verdict.

**Terms.** See [`glossary.md`](glossary.md) (N, K, place, refusal, TTFT, p99, SLO).

## 1. What exists today (verified in the repo files)

| Layer | Today |
| --- | --- |
| **SGLang** (each worker) | `--max-running-requests 6` → **12 running in the fleet**; waiting queue **unbounded**; `--enable-priority-scheduling --schedule-low-priority-values-first` (lower number = more urgent, default 5); `--retraction-policy priority`; chunked prefill 4096 |
| **LiteLLM** | `routing_strategy least-busy`, `num_retries 0`, `cancel_on_disconnect`, per-worker `max_parallel_requests` (now 40; a lower value answers 429, which never overflows), its own FIFO middleware as a safety net (128 / 32 / 10 s) |
| **Our admission** | 12 slots, waits ≤ 10 s, 9 for batch, queue ≤ 64; sheds with a 503 `timeout_queue` / `kv_pressure` / `batch_pressure` / `decode_capacity` |

Consequence: with our slot queue the engine's own queue is never used (12 gateway slots = 12 running), so SGLang's priority ordering has almost nothing to order.

## 2. The design under test (the user's)

- **SGLang waits and orders.** Up to `K` requests may wait *beyond the 12 running*, ordered by priority, bounded by `--max-queued-requests` (when it is full SGLang evicts the least-preferred request and answers 503).
- **LiteLLM only decides whether it fits the SLO.** It holds nothing (`ADMISSION_MODE=gate`). If a request would not get its first token within the SLO, it is cut and answered **503**, so the overflow can take it (`TTFT_CUT_S` → `stream_timeout`).
- Overflow itself is out of scope here (not implemented, not billed); a 503 is counted as "would overflow".

The question: **what is the largest `K` such that requests that enter the queue still start within the 20 s TTFT SLO?**

## 3. Hypotheses (fixed before measuring)

Service model. At saturation the fleet finishes calls at μ = 12 slots / 9.9 s per call ≈ **1.2 calls/s**, i.e. **0.6/s per worker** (measured call time in the smoke test). A request that finds `m` requests ahead of it in its worker's queue waits ≈ m / 0.6 = **1.65 s per position**, plus its own prefill (≈1 s with a cache hit, up to ≈7 s cold for 64K):

```text
TTFT(position m) ≈ 1.65 s x m + prefill            TTFT ≤ 20 s  =>  m ≤ (20 - prefill) / 1.65
cold prefill 7 s -> m ≤ 7.9 per worker (≈16 fleet)      cached prefill 1 s -> m ≤ 11.5 per worker (≈23 fleet)
```

- **HQ1.** TTFT of a queued request grows linearly with its position (≈1.65 s per position per worker). The largest queue that keeps the worst case under 20 s is **K ≈ 16 in the fleet (8 per worker)**.
- **HQ2.** The user's **K = 10 (5 per worker)** meets the SLO with margin: p99 TTFT ≈ 12-16 s at an overloaded N. **K = 16** is borderline (p99 ≈ 18-22 s). **K = 24** fails (p99 > 20 s) unless the first-token cut is on.
- **HQ3.** Goodput (calls served per minute) rises with K until the fleet is saturated, then flattens; sheds fall as K grows. The best `K` is the largest one that still meets the SLO for served calls; beyond it extra capacity only adds latency.
- **HQ4.** The first-token cut guarantees served TTFT ≤ ~20 s by construction, but with a large K many requests are cut **after waiting the full 20 s**: a late 503, worse for the user than an immediate one. So the cut is a backstop, and K is the real control.
- **HQ5.** Queues are per worker and LiteLLM routes by `least-busy`, so one worker's queue fills before the other's: the effective K is lower than 2 x k, and shed appears earlier than the fleet arithmetic predicts.
- **HQ6.** With priority scheduling, interactive requests jump ahead of queued batch ones, and a full queue evicts batch (503 "aborted by a higher priority request") before interactive. Interactive TTFT with 20% batch is then close to its TTFT without batch.
- **HQ7.** A bounded queue of K ≈ 10 matches what the current gateway queue does (it waits ≤ 10 s ≈ 12 positions). The gain is not capacity but removing code and letting the engine order by priority.

## 4. Part 1: mechanism probes (minutes; they decide whether the design is feasible)

Run with `metrics/probes/queue_mechanism.py` after `script/queue_variant.sh gate 3` (SGLang queue of 3 per worker, nothing held in the gateway).

| Probe | What it does | What we need to see | If not… |
| --- | --- | --- | --- |
| **m1** | 24 simultaneous long calls (12 running + 6 queue = 18 fit) | the 6 extra come back as **503** with the engine's message | if 500: map the error to 503 (hook or exception mapping) before anything else |
| **m1b** | fill with batch (priority 8), then send interactive (priority 5) | interactive start before queued batch; evicted batch get 503 "aborted by a higher priority request" | priority in the engine does not do what we assume: revisit |
| **m2** | saturate, then one interactive call with `TTFT_CUT_S=4` | cut near 4 s, a clean **503/timeout status before any stream byte**, and the engine queue shrinks | if the cut arrives as an SSE error after HTTP 200 the status cannot be 503: use non-streaming for the cut, or a different mechanism |
| **m3** | a queued call whose client disconnects | the engine queue shrinks within seconds | if not: queued dead requests waste queue positions; need an explicit abort |

## 5. Part 2: the queue-size experiment

**Constant**: model, KV, `MAX_NUM_SEQS=6`, `write_through`, profile `paper`, regime A (20 s pause), thinking off, all sessions interactive, seeds as in the sweep, warm-up 120 s + 360 s measured, 45 s between runs. Workers are restarted when `K` changes (fresh caches in every run).

**Reference (Q0)**: the current queue, at N = 16, 24, 32. It gives the knee of the system and the reference to beat (this is phase A1 of `notes/sweep-experiment.md`).

**Variable**: `K` per worker ∈ {2, 5, 8, 12} (fleet 4, 10, 16, 24), `ADMISSION_MODE=gate`, stateless checks off (`ADMISSION_CHECKS=""`) so the effect of the queue alone is visible, no cut. Then the best `K` again **with the cut at 20 s** and **with the estimate rule on**, to measure what each adds.

**N for Part 2**: the first N at which Q0 fails the verdict (overload), so the queue is actually used. The best `K` is then checked at one N below the knee (must pass) and at overload with 20% batch (priority).

**Selection at overload** (shed is unavoidable when demand exceeds capacity, so the plain verdict is not the right test): among the `K` whose **served** TTFT p99 ≤ 20 s, pick the one with the highest goodput; break ties by the lowest `shed_wait_p95` (how long a refused user waits to hear no) and the fewest `late sheds` (> 5 s).

Runs: Q0 x3 (26 min) + K x4 (4 x 9 min + 4 restarts of ~3 min) + best K with cut / with rule / with batch / below the knee (4 x 9 min) ≈ **2 h** (about 6.6 USD of GPU time, billed whether we measure or not).

## 6. Measurements

From `loadgen.py`: served TTFT p50/p95/p99 (also by call position), shed rate and reason (`engine_queue_full`, `engine_evicted`, `ttft_cut`, admission reasons), **`shed_wait_p50/p95`** and `shed_late_over_5s`, goodput (`served_per_min`), abandoned calls, steady-state flag, timeline. From Prometheus: maximum engine queue and running per worker (imbalance), retractions, KV, cache hits per tier. GPU power and the cost report as in the sweep.

## 7. Decision rules

| Observation | Decision |
| --- | --- |
| m1/m2/m3 all behave as needed | proceed to Part 2 |
| m1 gives 500 instead of 503 | add the mapping (a LiteLLM hook that turns the engine's queue-full error into a 503 with `shed-reason`) before Part 2 |
| m2: the cut arrives after HTTP 200 | the "cut and 503" idea needs another mechanism for streaming; Part 2 runs without the cut and the gap is documented |
| p99 grows ~1.65 s per position (HQ1) | the model is right; **K* = floor((20 - 7) / 1.65) per worker**, reduced by the margin the p99 needs |
| The best K is ≈ the length of the current gateway queue (HQ7) | move the waiting to SGLang (delete `GATEWAY.acquire` from the hot path, keep the estimate rule as the gate) |
| Gate mode sheds earlier than predicted (HQ5) | placement imbalance: `control/place.py` becomes the priority |
| Served p99 > 20 s for every K ≥ 1 | the queue cannot be used at all under this SLO; K = 0 (shed immediately) plus overflow |
| Cut at 20 s gives many late sheds | do not rely on the cut; lower K until late sheds vanish |
| Interactive with 20% batch is much worse than without (HQ6 false) | priority in the engine is insufficient: reserve slots (BATCH_SHARE) again or preempt more aggressively (`--priority-scheduling-preemption-threshold`) |

## 8. Threats to validity

- `--max-queued-requests` counts requests that arrive in the same scheduler tick before they are scheduled; a burst can be refused while slots are free. Values below the burst size would be misleading; the smallest K tested is 2 and m1 shows the effect.
- Queues are per worker; the fleet K is not a single number (HQ5).
- Restarting workers between variants gives each run a cold cache, so TTFT of the early minutes is higher than in the Q0 runs that follow one another; warm-up (120 s) mitigates but does not remove it. Q0 is also run once after a restart to measure the difference.
- A dozen runs, one replicate of the key ones: p99 near the knee rests on few observations (see the sweep protocol).
- LiteLLM's `stream_timeout` semantics on streaming are exactly what m2 tests; the cut is not assumed.

## 9. Execution log

| Step | Date | Result |
| --- | --- | --- |
| m1, m1b, m3 (queue limit 3 per worker) | 2026-10-02 | engine refusal arrives as **500**; `--max-queued-requests` refuses bursts while slots are free; m3 abort works; m1b inconclusive; m2 failed (script picked a terminating pod) |
| `r0` reference N=20, N=28 (short runs) | 2026-10-02 | N=20: 6.3 % shed, TTFT p99 12.5 s, 44.7 served/min; N=28: 56.5 % shed, p99 20.0 s, **29.0 served/min** (goodput falls); 52.8 % of the prompt recomputed at N=28: cache misses dominate, routing suspected; details in `notes/findings.md` |
| K sweep at N=28 (K = 4, 10, 16, 24) | 2026-10-02 | goodput flat 21-29 calls/min; TTFT p50 12 -> 40 s; p99 23, 36, 45, 60 s; HQ1 and HQ2 refuted (≈ 2.7 s per position, not 1.65); a counter refuses in 0.1 s vs 10 s; K = 24 jammed; details in `notes/findings.md` |
| Part 1b s1-s4 | 2026-10-02 | HC1, HC2, HC3 confirmed; HC4 resolved as **408** (clean status before streaming), not 503; details in `notes/findings.md` |

## 10. Later notes (dated; what is above is not rewritten)

**2026-10-02 (before running): Part 1b, does LiteLLM close the upstream connection?** The user asked for a dedicated experiment on whether LiteLLM really closes connections. It matters because the whole "cut and answer 503" idea only frees capacity if SGLang finds out: an HTTP error to the client is worth nothing if SGLang keeps generating for nobody (a *zombie* request that holds a running slot or a queue place). Probe: `metrics/probes/connection_close.py`, on an otherwise idle engine, variant `gate 3 4 ""`.

| Scenario | Action | Evidence of an abort | Evidence of a zombie |
| --- | --- | --- | --- |
| s1 | streaming client leaves at 3 s while running (`max_tokens=800`, `ignore_eos`) | running 1 → 0 within a few seconds; tokens generated ≪ 800 | running stays 1 until ~17 s; ~800 tokens generated |
| s2 | same, non-streaming | same | same |
| s3 | client leaves while queued in SGLang (12 running + queued) | queued count drops right after | the request starts when a slot frees and generates its tokens |
| s4 | LiteLLM cuts a queued interactive call at `TTFT_CUT_S=4` | the client gets its error at ≈4 s and the queued count drops; total tokens = the batch only (14 x 600) | an extra ~200 tokens appear when a slot frees |

Hypotheses (fixed now). **HC1**: s1 and s2 abort within ~1-3 s (`cancel_on_disconnect: true` is set, and the earlier design assumed it). **HC2**: s3 drops the queued request, because the connection closes before it ever reaches the model. **HC3 (the uncertain one)**: in s4 the client gets an error near 4 s, but whether LiteLLM closes the upstream stream when its own timeout fires is not guaranteed: if it does not, the cut leaves zombies and cannot be used as the queue's backstop. **HC4**: if the error arrives after HTTP 200 (stream already open) the status cannot be a 503; the probe prints the status it actually got.

Decision rules. s1-s3 abort → the disconnect path is safe, keep `cancel_on_disconnect`. s4 zombie → do not use `stream_timeout` as the cut; implement the cut ourselves (a task that cancels the upstream call) or accept that K alone bounds the queue. s4 aborts but arrives as HTTP 200 + SSE error → the cut works for capacity but not as a 503: map it, or use the cut only for non-streaming clients, and say so in the design.

**2026-10-02 (after the first probes): the queue bound moves to the gateway.** m1 showed that SGLang's `--max-queued-requests` is evaluated at arrival over requests that are not scheduled yet, so it refuses bursts while places are free, and that its refusal reaches the client as a 500. The variable `K` of Part 2 is therefore **not** implemented with that flag. It is implemented with a counter at the gateway that holds nothing: `FLEET_MAX_IN_FLIGHT = 12 + K` and `GATEWAY_QUEUE_MAX = 0`, so a request is either admitted (and then waits, ordered by priority, in SGLang's queue) or refused at once with our own 503 `decode_capacity`. SGLang's limit stays unbounded (`MAX_QUEUED_REQUESTS=1000000`). Consequences: the engine's eviction of low priority by high priority is not used (the counter refuses newcomers instead), `BATCH_SHARE` keeps limiting how many of the `12 + K` places batch may take, and no 500 mapping is needed. `script/queue_variant.sh` gets a `count K` mode for this.

**2026-10-02 (results of Part 1b).** HC1 (s1, s2 abort within ~1.5 s): confirmed. HC2 (a queued request is dropped on disconnect): confirmed. **HC3 (the cut frees the upstream): confirmed**, with the log as evidence (no prefill of the marker prompt; the freed place went to a batch request). **HC4: the cut is answered with HTTP 408 before any stream byte**, so the status is controllable and not an SSE error after a 200; but 408 does not overflow, so a mapping to 503 `ttft_cut` is needed. Consequences for Part 2: the first-token cut can be used as the backstop; the queue bound is the gateway counter (note above), and the 408→503 mapping is a prerequisite only for the overflow (out of scope now), not for measuring K. The staleness of SGLang's gauges (seconds) is a threat to every rule that reads the engine queue; the counter at the gateway is exact.

**2026-10-02 (shortened plan).** Where the time went: each run of the sweep protocol is 8.75 min, and every change of `K` cost a restart (workers ~3 min when the SGLang flag changed, LiteLLM ~1 min). Two changes remove most of it. (1) `K` is a gateway counter, so SGLang is never restarted. (2) With `ADMISSION_ALLOW_OVERRIDE=1` a request may carry `fleet_capacity` / `gateway_queue_max` / `batch_share` in its body and the live queue changes with the first request, so **no restart at all** between values of K (`GatewayQueue.configure`, tested). Part 2 is run with short runs (240 s: 60 s warm-up + 180 s measured) to compare K, then the chosen K is confirmed with one full run:

| Step | What | Time |
| --- | --- | --- |
| 1 | `r0` (reference) at N = 20 and 28 to find where the system is overloaded | 2 x 4.3 min |
| 2 | `r0, 4, 10, 16, 24` at that N (short runs) | 5 x 4.3 min |
| 3 | best K confirmed: full 8 min run, again with 20% batch | 2 x 9 min |
| total | | ≈ 50 min (was ≈ 2 h) |

Short runs trade accuracy for time: with ~180 s measured the p99 rests on few observations. They are used to rank K, not to declare a capacity; the confirmation run is. Step 3 also repeats the best K once to see the noise.

**The cut and the 408.** Requests the gateway knows will not be served (counter full, estimate rule) are already refused at once with a 503 and a reason. Requests that were admitted and then waited too long are cut by LiteLLM as a 408; `async_post_call_failure_hook` now turns that 408 into a 503 `ttft_cut` (LiteLLM 1.105 lets a failure hook replace the error; tested with a stand-in timeout, **not yet seen against the live cluster**). Caveat: `stream_timeout` is an httpx read timeout, so it also applies between later chunks, not only to the first token; with a 20 s value a decode stall that long is cut too.

**2026-10-02 (results of Part 2, step 2).** At N = 28 no K meets the SLO and goodput is independent of K: the queue converts refusals into latency without adding capacity. **HQ1 refuted** (the nominal service rate of 1.2 calls/s does not hold under load: the fleet finished 0.47 calls/s because 52.8 % of the prompt was recomputed), **HQ2 refuted** (K = 10 gives p99 36 s), HQ3 half refuted, HQ7 supported in a different way (the useful property of a counter is the instant refusal, not capacity). Decision rule applied from section 7: the queue is not the lever, the cost of a call is. Next experiments, one variable at a time: radix eviction `lfu` -> `lru`; then session/prefix affinity (`control/place.py`); the best queue allowance (small K, instant 503) is confirmed afterwards at the knee of the improved system, not at N = 28 of the current one.

**2026-10-02 (pre-registration, Part 3: choosing the number).** The user's goal is one number: *how many requests may wait, such that the SLO (TTFT p99 ≤ 20 s) holds and the system does not break.* The K sweep at N = 28 did not give it (no K met the SLO; fit: p99 ≈ 15.7 s + 1.86 s x K, so K ≈ 2 at that load, extrapolated from four short runs at one overloaded N). Decisive experiment, run before reading anything else: **K = 0, 2, 4, 6 at N = 24 and N = 20** (240 s runs; `eq-k2-n24`, `eq-k2-n20`), counter mode, stateless checks off, no cut.

Hypotheses (fixed now). **HK1:** p99 TTFT of served calls is linear in K, with a lower intercept at N = 20 than at N = 28 (less cache thrash), so the SLO-compatible K is larger at N = 20 than at N = 24. **HK2:** K = 0 has the lowest p99 and the highest shed rate; goodput (served/min) does not depend on K. **HK3:** at N = 20, K ≤ 4 keeps p99 ≤ 20 s; at N = 24, K ≤ 2.

Selection rule. The number is `K* = the largest K whose served TTFT p99 is ≤ 20 s at BOTH N = 24 and N = 20`, because overload is when the queue is full and the bound matters. Ties: lower shed rate, then lower `shed_wait_p95`. It is then confirmed with one full 8-minute run at N = 24 (and at N = 28 as a stress), and the number is reported with the measured intercept and slope, not as a bare value.

Where the guarantee comes from. A measured fit (p99 ≈ a + b x K) says how large K can be; it is not a guarantee. The guarantee is **K plus the first-token cut**: with `TTFT_CUT_S = 20` an admitted interactive call that has not produced its first token after 20 s is cut and answered with a 503 `ttft_cut` (LiteLLM closes the upstream call, verified in Part 1b), so the TTFT of every *served* call is ≤ 20 s by construction, and K only controls how many calls are cut late instead of refused at once. The cut and its 408 → 503 mapping must be verified live with K* before the claim is made. Caveat: `stream_timeout` is a read timeout, so it also cuts a stream that stalls for 20 s between tokens.

**2026-10-02 (results of Part 3).** Rule result: **K\* = 4** (largest K with p99 ≤ 20 s at both N = 24 and N = 20: 18.3 s and 17.1 s; K = 6 fails with 29.4 s and 28.2 s). HK1 holds (about +3 s of p99 per unit of K); HK2 half fails: **goodput falls as K grows** (N = 20: 45.7, 43.3, 36.3, 30.7 calls/min for K = 0, 2, 4, 6), which the rule did not anticipate (HQ3 had assumed it flat). Deviation, stated openly: the rule would give 4, but K = 2 has the same goodput as K = 0 across the two loads, fewer refusals than K = 0, and a 4-8 s margin under the SLO against 2-3 s for K = 4, so **K = 2 is recommended and K = 4 is the upper bound**; the choice is the user's and is confirmed with a full-length run. Tables and plots: `notes/findings.md` "Choosing K", `plots/queue_k_n24.png`, `plots/queue_k_n20.png`. Still open: the live check of the 20 s first-token cut with the 408 -> 503 mapping.

**2026-10-02 (decision).** The user chose **K = 2**. The 408 → 503 mapping of the first-token cut was verified live (503 `ttft_cut` after 4.2 s). Repo defaults changed accordingly (ARCHITECTURE decisions [42](../ARCHITECTURE.md#decision-42)-[43](../ARCHITECTURE.md#decision-43)). A confirmation run at full length (8 min) is still pending and goes with the final load tests.

**2026-10-02 (code removal).** The gateway queue was deleted from the code (ARCHITECTURE decision [46](../ARCHITECTURE.md#decision-46)): `gateway_queue.py` is now `places.py` (counter only), and the variants `q0`/`gate` no longer exist. This protocol, its runs (`metrics/runs/eq-*`) and the reference `r0` stay as the evidence behind K = 2; reproducing `r0` would need the code from the git history before this change.

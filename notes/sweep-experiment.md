# Experiment E1: how many sessions can the system take?

**Status:** PARTLY SUPERSEDED. It defined the regimes (A dense, B realistic idle) and the verdict that the later experiments reuse; its full sweep was replaced by the queue and routing experiments and by the final tests (`notes/final-load-tests.md`). Words that changed: "slot" is now "place", and the gateway queue it assumes no longer exists (decision [46](../ARCHITECTURE.md#decision-46)). Terms: [`glossary.md`](glossary.md).

Protocol written **before measuring** (2026-10-02). The hypotheses and the decision rules are not edited after seeing the data: if something changes, a dated note is added at the end. Results in `metrics/runs/<tag>/`, interpretation in `notes/findings.md`, decisions in the log of `ARCHITECTURE.md`.

## 1. Questions

| # | Question | How it is answered |
| --- | --- | --- |
| P1 | What is the largest number of sessions N with TTFT p99 ≤ 20 s and sheds ≤ 1%? | `loadgen.py` verdict per run |
| P2 | How does it break: shedding cleanly (503) or letting TTFT blow up? | sheds by reason, TTFT p99 and its evolution per minute |
| P3 | Which resource saturates first: slots, KV, cache or prefill? | `prometheus.json` (queues, KV, retractions, hits per tier) |
| P4 | Are the two workers loaded evenly? | `engine_running_max` and `engine_queue_max` per worker |
| P5 | What does a token cost at that point, and when is overflow the better option? | `cost.json` |
| P6 | With agents that rest as in the real world, how many **open** sessions fit? | regime B (phase 3) |

## 2. Capacity model (the hypothesis, with numbers)

A session occupies a slot only while a call is in progress. Measured in the smoke test: a call lasts ~9.9 s (TTFT ~0.6 s + ~350 tokens at 48 tok/s) and the session spends ~79% of its time inside a call. With pauses between turns the occupancy drops:

```text
occupancy per session = calls_per_turn x t_call / (calls_per_turn x (t_call + t_tool) + pause_between_turns)
N_saturation          = 12 slots / occupancy
```

| Median pause between turns | Occupancy (slots per session) | 12 slots saturate at N = |
| ---: | ---: | ---: |
| 8 s (smoke test) | 0.72 | 17 |
| **20 s (regime A)** | **0.61** | **20** |
| 60 s | 0.39 | 30 |
| **120 s (regime B)** | **0.26** | **46** |
| 300 s | 0.13 | 94 |

The paper reports that in multi-turn sessions the user is idle ~80% of the time, so the real world looks like regime B, not A. That is why there are **two regimes**: A measures the compute limit (slots and prefill) with dense load and is quick; B measures the **memory/cache** limit with open sessions that rest, and is long.

**Hypotheses (fixed before measuring)**

- **H1.** Regime A: the verdict passes up to N ≈ 16-20 and fails from N ≈ 24.
- **H2.** The failure shows up first as **sheds** (`timeout_queue` after 10 s of waiting), not as TTFT > 20 s: the maximum wait is 10 s and a cold 64K prefill is ~7 s, so a served TTFT should stay below ~17 s. Therefore the TTFT p99 of the served calls should stay < 20 s almost until the collapse.
- **H3.** The TTFT of the first call of a turn (`turn_start`) is worse than that of the calls inside the turn, because of the cold cache after the pause. In regime A the difference is small; in B it is large.
- **H4.** Decode with 6 sequences at once is slower than the 48 tok/s measured with a single one (more KV to read). If so, the real knee sits below H1.
- **H5.** Regime B: the capacity in open sessions is not 46 (the slot limit) but ~40-50 because of the cache (GPU ~8 + host ~15 per worker at ~55K tokens): beyond that, every turn start pays a full prefill.
- **H6.** The split between workers is uneven (LiteLLM's `least-busy` does not know the prefixes): there will be runs where one worker reaches 6 running and has a queue while the other has room.

## 3. Design

**Held constant** (and saved in each run's `setup.txt`): model `Qwen/Qwen3.5-9B` bf16, KV bf16, `MAX_NUM_SEQS=6`, `FLEET_MAX_IN_FLIGHT=12`, `MAMBA_FULL_MEMORY_RATIO=0.17`, `write_through`, interactive deadline 20 s (maximum wait 10 s), gateway queue 64; context profile `paper` (30-52K at start, compaction above 60K), shared 10K system prompt, output with median 247 (`ignore_eos`), thinking off, all sessions interactive (priority 5), 2 retries after a 503. **Nobody else uses the GPU.**

**Each run:** a different seed per N (so it does not re-send conversations from the previous run that may still be cached; the system prompt is common, as in production); staggered start over 60 s; 120 s of warm-up discarded; 360 s measured. 45 s pause between runs.

**Phases** (B and C are not launched until A has been read)

| Phase | Regime | N | Time | Purpose |
| --- | --- | --- | ---: | --- |
| **A1 coarse** | A (20 s pause) | 8, 16, 24, 32 | ~35 min | locate the interval where the verdict changes; stops only if 2 in a row fail |
| **A2 refine** | A | 3 values inside the interval + a **replicate** of the largest N that passes | ~35 min | fix the knee and measure variability |
| **B realistic** | B (120 s pause, warm-up 480 s, measurement 900 s) | ~2x, 3x and 5x the maximum N of A, capped at 96 | ~75 min | cache limit and cost of a cold turn start |

GPU cost: the machine is billed by the hour (3.29 USD/h) whether we measure or not; A1+A2 ≈ 70 min ≈ 3.8 USD, phase B ≈ 75 min ≈ 4.1 USD. There is no extra budget, only time.

**What is NOT varied here** (later experiments, only if the results call for them): `FLEET_MAX_IN_FLIGHT` (E2), batch sessions (E3), the `light` profile (E4), thinking on (E5). Vary one thing at a time.

## 4. What is measured

- **Primary (verdict):** interactive-class TTFT p99 ≤ 20 s **and** shed rate ≤ 1%; plus `steady_state` (second-half median ≤ 1.5x the first half's; if not, the run was in growing overload and the p99 is not a stable regime).
- **By call position** (`by_position`): session start, turn start, within turn; TTFT p50/p99 and the fraction served from cache.
- **From the cluster** (`prometheus.json`, run window): sheds by reason, maximum gateway and per-worker queue, slots in use, maximum KV, retractions, cache hits per tier (GPU / host / Mooncake), backups to Mooncake, TTFT p99 as seen by the gateway and by the engine.
- **From the GPU:** power, memory, temperature every second (`gpu.csv`; utilization does not exist under MIG).
- **Cost** (`cost.json`): per million input tokens (all and uncached), output, per thousand calls, per session-hour, slot occupancy, and the equivalent overflow bill with and without cache discount.
- **Timeline** (`timeline.csv`): calls, served, sheds and TTFT per minute.

## 5. Decision rules (what we change depending on the result)

| If we observe… | Then… | Change |
| --- | --- | --- |
| It passes up to N ≥ 16 and the failure arrives as sheds with TTFT p99 < 20 s (H1+H2) | admission does its job | leave the policy alone; the declared capacity is 80% of the maximum N; continue with phase B |
| TTFT p99 > 20 s with sheds ≤ 1% (silent queueing) | the wait rule is too permissive | shed on **predicted** TTFT (`estimated wait + uncached tokens / 8.5K tok/s`) in `admission.py`; first measure the calibration of that prediction with this data |
| Sheds > 1% with slot occupancy < 80% and one worker at 6 while the other has room (H6) | placement imbalance, not lack of capacity | top priority to `control/place.py` (`p2c` or `prefix_then_load`) before tuning anything else |
| `turn_start` with TTFT p99 ≥ 3x that of `within_turn` and cache served < 50% at `turn_start` (H3) | cache residency problem | in this order: affinity placement; raise `--hicache-size` (today 32 GiB per worker, there are 221 GB of RAM) and the Mooncake pool; review `MAMBA_FULL_MEMORY_RATIO`; if that is not enough, the turn-start SLO is declared separately |
| `kv_used_max` ≥ 0.95 or retractions > 0 below the knee | KV does not hold 6 x 64K in practice | lower `MAX_NUM_SEQS` to 5 (and `FLEET_MAX_IN_FLIGHT` to 10, `max_parallel_requests` to 10, dashboard thresholds) |
| The verdict fails with slot occupancy < 60% (H4) | batched decode or concurrent prefill is slower than measured alone | measure tok/s per sequence with a full batch from `records.jsonl`; try a smaller `--chunked-prefill-size` and 4 sequences per worker; recompute the LiteLLM rates |
| The collapse is abrupt (a single N goes from OK to shedding > 20%) | there is no gradual degradation | more conservative alert thresholds (the capacity-shed alert at 2-3%, not 5%) |
| Replicates of the same N differ > 20% in TTFT p99 | the knee is a band, not a point | report the maximum N as an interval and repeat the doubtful run once more |
| Regime B: cache coverage falls and turn-start TTFT > 20 s at an N below the slot figure (H5) | the real limit is open sessions | declare the capacity in open sessions (lower than the slot one) and prioritise HiCache/Mooncake and affinity |
| Cost at the knee | | recompute LiteLLM's `input/output/cache_read_input_token_cost` with the measured tokens/s; compute the crossover with overflow (with and without cache discount) |

## 6. Threats to validity and controls

- **Synthetic text:** compute does not depend on meaning, but it does depend on length and on repeated prefixes; the token calibration (3.9 per word) is verified (±1%).
- **Order effect:** ascending runs start with the previous run's cache full. Control: different seeds, a replicate of the largest N that passes, and `setup.txt` with the exact configuration.
- **Sampling noise:** with ~400 calls per run near the knee, the p99 depends on ~4 observations. Control: replicates and `steady_state`.
- **The generator shares the machine with LiteLLM** (26 CPUs; up to 96 threads): watch that `progress.log` shows no gaps without calls.
- **Forced output (`ignore_eos`):** it makes the load deterministic but the history contains meaningless text; it does not affect compute.
- **One GPU, one day:** power/temperature variations are not controlled; they are recorded.
- **Pauses are compressed** in regime A (20 s instead of minutes); that is why regime B with long pauses exists.

## 7. How it is run

```bash
bash setup/sync_lambda.sh                       # local: uploads script/, tests/, control/
ssh ... 'cd ~/final_project && script/sweep.sh e1-a1 paper "8 16 24 32" 480'
ssh ... 'tail -f ~/final_project/metrics/runs/e1-a1/N16/progress.log'   # one line every 30 s
bash script/fetch_results.sh                    # local: brings metrics/runs/
```

Phase B: `WARMUP=480 script/sweep.sh e1-b paper "<N...>" 1380 --turn-idle 120 --ramp 120`.

## 8. Execution log (filled in as it runs)

| Run | Date | Regime | N | Pass? | TTFT p99 (s) | Sheds | Notes |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| smoke | 2026-10-02 | A (8 s pause, light profile) | 3 | yes | 1.86 | 0% | only validates the tools |

## 9. Later notes (dated; what is above is not rewritten)

**2026-10-02 (before launching A1): queue variants.** The user asked that the wait be managed by LiteLLM and SGLang through their parameters instead of our own queue. The code was investigated (see `notes/findings.md`, "Native queues") and it was decided to **measure it rather than opine**: A1/A2 run as they are with the current queue (**Q0**) as the baseline, and then it is repeated with the same N and seeds under:

- **Q1, SGLang decides:** a high `FLEET_MAX_IN_FLIGHT` and admission in "tag only" mode (it still forwards `priority` to the engine, but neither sheds nor queues); `--max-queued-requests` per worker (proposal: 8) and `--priority-scheduling-preemption-threshold` (proposal: 1); LiteLLM as a safety net; also try `stream_timeout: 20` as a TTFT cap.
- **Q2, LiteLLM's FIFO queue:** our own admission in "tag only" mode; `max_in_flight_requests_per_worker: 12`, `max_queued_requests_per_worker: 64`, `admission_queue_timeout_seconds: 10`; priority is applied inside SGLang.

Comparison: maximum N in regime A, and **with 20% batch sessions** (where priority matters), interactive TTFT p99 and shed rate per class; the reason and shape of the shed; lines of code that could be removed from `control/`. Hypotheses (fixed now): Q1 matches Q0 in maximum N but with a worse split when the workers are unbalanced (H6); Q2 passes the verdict without batch and fails it with batch because its FIFO queue does not tell classes apart; neither replaces shedding on estimated wait time, so if the TTFT p99 of Q1/Q2 exceeds 20 s with few sheds, that part of `admission.py` stays.

**2026-10-02 (user's design, refines Q1): "LiteLLM only decides whether it fits the SLO; SGLang queues and orders by priority".** Q1 becomes a *gate-only* admission:

- SGLang does the waiting: `--enable-priority-scheduling --schedule-low-priority-values-first` (already on) orders its queue, `--max-queued-requests` bounds it (it evicts the least-preferred request and answers 503), and a higher-priority arrival can preempt running work.
- The gateway does **not** hold requests (`GATEWAY.acquire` skipped, `FLEET_MAX_IN_FLIGHT` no longer a queue). It only answers "does this request fit the SLO?" and sheds with a 503 (overflow-eligible) if not; it keeps tagging the priority and forwarding it as `extra_body.priority`.
- The gate predicts TTFT as `queue wait + uncached prefill`. The queue wait should count only the work **ahead of this priority**: SGLang exports `sglang:num_queue_reqs` per priority (`priority=""` is the total, `priority="<n>"` the breakdown), so an interactive request is not penalised by a long batch queue it will jump over.
- Known weaknesses to measure, not assume: (1) the fleet snapshot is polled every 15 s, so a burst sees an empty queue and all of it is admitted; a gateway-side in-flight counter (no queue, just a count) or a faster poll would fix that; (2) SGLang has no queue timeout, so an admitted request stuck in the engine queue can exceed the SLO unless `stream_timeout` (untested) or a client disconnect cuts it; (3) the slot reservation for interactive (`BATCH_SHARE`) is replaced by priority ordering and preemption, which wastes the preempted work.
- Extra hypothesis (fixed now): H7. Gate-only passes the verdict at the same N as Q0 when traffic is all-interactive, and does better than Q0 with 20% batch (priority acts inside the engine without wasting slots), but its p99 TTFT shows more bursts because of weakness (1).

**2026-10-02 (later): the Q1/Q2 comparison above is superseded by `notes/queue-experiment.md`**, which turns the user's gate-only design into its own experiment (mechanism probes first, then queue size K at overload). The sweep (A1/A2/B) keeps defining the regimes, the verdict and the Q0 reference.

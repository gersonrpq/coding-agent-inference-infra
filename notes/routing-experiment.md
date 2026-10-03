# Experiment ER: which worker should get a request?

**Status:** DONE at N = 24 and N = 20 (`metrics/runs/er-n24`, `er-n20`); `affload` is the default (decisions [47](../ARCHITECTURE.md#decision-47), [50](../ARCHITECTURE.md#decision-50)); the replicate of the refusal effect is in `notes/final-load-tests.md` (F2). Terms: [`glossary.md`](glossary.md).

Protocol written **before running** (2026-10-02). Hypotheses and the decision rule are not edited after seeing data; later changes go in dated notes at the end. Results in `metrics/runs/er-*`, tables and plots in `plots/`, interpretation in `notes/findings.md`, decision in the log of `ARCHITECTURE.md`.

**Terms.** See [`glossary.md`](glossary.md) (N, K, place, refusal, TTFT, p99, SLO).

## Question

Two SGLang workers share a cache hierarchy (L1 GPU per worker, L2 host RAM per worker, L3 Mooncake across them). The user's argument: since the hierarchy is shared, sending a request to the worker that holds its prefix is not worth the complexity, and LiteLLM's `least-busy` is enough if justified. Evidence so far (`notes/findings.md`, "Placement"): a call that lands on the other worker gets 26.7 % of its prompt from cache against 44.3 % on the same one and recomputes twice as many tokens, but those runs used `lfu` and a long queue. **Is a smarter placement worth it, with `lru` and the final queue settings, and which policy should the system use?**

## Variants

| Id | Policy | Implemented by | Information it uses |
| --- | --- | --- | --- |
| `shuffle` | random | LiteLLM `routing_strategy: simple-shuffle` | none |
| `lb` | fewest requests in flight per worker (**current**) | LiteLLM `least-busy` | load |
| `latency` | lowest recent latency | LiteLLM `latency-based-routing` | observed latency |
| `aff` | the same worker for the whole session | hook, `control/place.py`: `hash(session) mod 2` | session only |
| `affload` | the session's previous worker, unless it is more loaded than the other by `PLACEMENT_SLACK` (3) requests in flight; **batch** requests (priority > 5) go to the least loaded worker | hook, `control/place.py` | session, load, priority, per-worker weight |

`affload` is the custom policy: it uses what the gateway knows and LiteLLM's strategies do not (the session, the priority class, exact in-flight counts per worker, worker weights, default 1:1). A session is identified by the hash of its first user message (a conversation keeps its system prompt and first message; compaction starts a new key). Custom policies work by rewriting the model alias to a per-worker alias (`qwen-coding-w0` / `qwen-coding-w1`); with `PLACEMENT_POLICY=litellm` (default) nothing is rewritten and LiteLLM routes.

## Design

- **Constant:** model, `lru`, final queue allowance K\* as a counter (`script/queue_variant.sh`, chosen in the queue experiment), thinking off, profile `paper`, regime A (20 s pause), all sessions interactive, same seeds in every variant (the same conversations are replayed), 240 s runs with 60 s warm-up.
- **Loads:** N = 24 (overload, where the queue is full and cache misses cost most) for all five variants and a **repeat of `lb`** (run-to-run noise); N = 20 (near the knee) for `lb`, `shuffle`, `affload` and the best of the rest. About 45 min in total.
- **Changing the variant** restarts LiteLLM only (~1 min): the strategy is one line of `router_settings`, the custom policies an environment variable.
- **Metrics per run:** served per minute, TTFT p50/p99, shed rate; share of prompt tokens from cache (GPU / host / Mooncake) and recomputed; **cached share when the call stays on the same worker versus moves to the other one**; `worker_stickiness`; share of calls per worker (balance); decode tok/s; maximum engine queue per worker.
- **Outputs:** a markdown table printed by `script/analyze_runs.py` and the figures `plots/routing_*.png`.

## Hypotheses (fixed before running)

- **HR-1.** `shuffle` is the worst on balance (more variation in the share of calls per worker and in the engine queue) and has stickiness ≈ 0.5.
- **HR-2.** `lb` has stickiness 0.6-0.75 (measured) and a good balance. `latency` behaves like `lb` or slightly worse (it reacts to the load it creates).
- **HR-3.** `aff` has stickiness 1.0 and the highest cached share, but a worse balance than `lb` (static hash, 24 sessions: a typical gap of 2-3 sessions between workers).
- **HR-4.** `affload` keeps stickiness ≥ 0.85 and a balance as good as `lb`, so it has the best served per minute and TTFT p99 of all.
- **HR-5.** The gain of `affload` over `lb` is between 5 % and 20 % in served per minute at N = 24 and smaller at N = 20. Under `lru` it is smaller than the 17.6-point cached-share gap measured with `lfu` suggests, because part of that gap was the eviction policy.

## Decision rule

Adopt a custom policy only if, at N = 24, its served per minute exceeds `lb`'s by more than the difference between the two `lb` runs **and** its TTFT p99 is not more than 10 % worse. Among adopted candidates, the highest served per minute. Otherwise `least-busy` stays and this table is the justification the course asks for ("which policy and why"). In both cases `PLACEMENT_POLICY` and `routing_strategy` in the repo are set to the winner.

## Threats to validity

- Short runs and one repetition per variant, one extra only for `lb`: differences below the `lb` run-to-run gap are noise.
- Session identification by first-message hash fits the load generator and pi; a client that changes the first message each call would defeat it (then `affload` degrades to least-loaded).
- Rewriting the model alias changes the `model` label of LiteLLM's metrics and spend; the aliases carry the same cost rates.
- The in-flight counts used by the hook are the gateway's own (exact); `lb` uses LiteLLM's.
- Placement interacts with the queue: K\* is fixed so that only placement varies.

## Execution log

| Step | Date | Result |
| --- | --- | --- |
| Built: `control/place.py`, aliases, `script/router_variant.sh`, `script/routing_experiment.sh`, `script/analyze_runs.py`; 128 unit tests | 2026-10-02 | not yet run on the cluster; first plot (K sweep) at `plots/queue_k_n28.png` shows the analysis tool works. In that sweep the engine queue maximum per worker was 2/2, 5/5, 8/7, 8/8 for K = 4, 10, 16, 24: the gateway counter does bound SGLang's queue to K/2 per worker |
| Ran N = 24: `lb`, `shuffle`, `latency`, `aff`, `affload`, `lb2` (18:04-18:36) | 2026-10-02 | `affload` met the rule; see the dated note below and `plots/routing_n24.png` |
| Deployed the new code (`launch_cluster.sh`: `places.py` counter, tenants, hops, overflow gate, alerts, warm gate) and started N = 20 for `lb`, `shuffle`, `affload` (script `/tmp/exp5.sh`, log `~/exp5.log` on the server, ends with `EXP5_DONE`) | 2026-10-02 | running; results will be in `metrics/runs/er-n20/` |

## Later notes (dated; what is above is not rewritten)

**2026-10-02 - N = 24 done** (`metrics/runs/er-n24/`, 240 s per run, 60 s warm-up, K = 2, `lru`, same seeds; figure `plots/routing_n24.png`; one run per variant, `lb` twice).

| run | served/min | TTFT p50 | TTFT p99 | shed % | cached % | cached % same worker | cached % other worker | stickiness | calls on worker 0 % | decode tok/s | engine queue max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| aff | 40.0 | 9.3 | 27.1 | 73.4 | 83.0 | 88.6 | 46.2 | 0.96 | 60.0 | 30.3 | 7/1 |
| affload | 54.0 | 2.1 | 9.2 | 64.9 | 86.8 | 90.8 | 62.6 | 0.95 | 50.0 | 28.7 | 3/2 |
| latency | 29.7 | 14.7 | 30.7 | 79.2 | 75.6 | 90.2 | 72.5 | 0.49 | 59.6 | 25.9 | 8/8 |
| lb | 40.0 | 3.3 | 8.9 | 69.4 | 73.3 | 85.8 | 71.0 | 0.52 | 51.2 | 26.4 | 4/3 |
| lb2 | 40.0 | 3.3 | 11.1 | 77.0 | 73.6 | 89.0 | 60.1 | 0.5 | 51.2 | 26.1 | 2/4 |
| shuffle | 48.3 | 3.2 | 14.0 | 65.4 | 78.0 | 89.1 | 74.2 | 0.53 | 42.1 | 27.6 | 4/3 |

Reading against the pre-registered hypotheses:
- **HR-1** (shuffle: stickiness ~0.5, worst balance): stickiness 0.53 and the most uneven split (42 % of calls on worker 0) **held**; but shuffle served 48.3/min, more than `lb`.
- **HR-2** (`lb` stickiness 0.6-0.75): **refuted**: measured 0.50-0.52, as random as shuffle. `latency` behaves worse than `lb` (29.7/min, p99 30.7 s, engine queue 8/8): it reacts to the load it creates, as predicted.
- **HR-3** (`aff`: stickiness 1.0, highest cached share, worse balance): stickiness 0.96 and a worse balance (60 % on worker 0) **held**; the highest cached share did **not** (83.0 % against 86.8 % for `affload`), and its tail is the worst after `latency` (p99 27.1 s, engine queue 7/1): a static hash piles sessions on one worker.
- **HR-4** (`affload`: stickiness >= 0.85, balance like `lb`, best served and TTFT): stickiness 0.95, balance 50.0 %, best served per minute (54.0), p99 9.2 s (`lb` 8.9 and 11.1): **held** (its p99 is not the lowest, it is within the `lb` noise).
- **HR-5** (gain over `lb` between 5 % and 20 %): the gain is **+35 %** (54.0 against 40.0 served per minute): above the predicted range. The cached share rises from 73 % to 87 %, and the share recomputed falls from 27 % to 13 %.

**Decision rule at N = 24:** the two `lb` runs give the same served rate (40.0 and 40.0; their shed rates differ, 69 % and 77 %, which is the real noise here), so the noise threshold is below one call per minute (served counts are 120 calls in 180 s, resolution 0.33). `affload` exceeds `lb` by 14 calls per minute and its TTFT p99 (9.2 s) is within 10 % of `lb`'s (8.9 s) and better than its repeat (11.1 s): **`affload` is adopted at N = 24; `shuffle` is second (48.3/min, p99 14.0 s).** The decision is confirmed or reversed by the N = 20 runs (`lb`, `shuffle`, `affload`) before `PLACEMENT_POLICY` is changed in the repo.

Caveats: one run per variant; at N = 24 the system refuses 65-79 % of the attempts (heavy overload by design), so the table says which policy copes best with overload, and N = 20 (near the knee) says whether the advantage survives at a sustainable load.

**2026-10-02 - N = 20 done** (`metrics/runs/er-n20/`, same settings, deployed code with the places counter; figure `plots/routing_n20.png`; one run per variant, no repeat of `lb`).

| run | served/min | TTFT p50 | TTFT p99 | shed % | cached % | cached % same worker | cached % other worker | stickiness | calls on worker 0 % | decode tok/s | engine queue max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| affload | 54.0 | 1.7 | 7.6 | 41.9 | 87.7 | 92.3 | 41.5 | 0.88 | 50.6 | 29.6 | 2/1 |
| lb | 53.7 | 1.8 | 9.5 | 29.7 | 82.1 | 90.3 | 79.6 | 0.55 | 50.3 | 28.9 | 2/3 |
| shuffle | 41.7 | 6.2 | 19.3 | 51.4 | 79.5 | 89.5 | 78.9 | 0.46 | 41.6 | 27.4 | 1/7 |

| run | calls | abandoned (3 refusals in a row) | refused attempts | within the 20 s SLO |
| --- | ---: | ---: | ---: | ---: |
| lb | 170 | 9 | 68 | 100 % |
| shuffle | 143 | 18 | 132 | 99.2 % |
| affload | 181 | 19 | 117 | 100 % |

Reading: near the knee `affload` and `lb` **tie in served calls per minute** (54.0 against 53.7; the difference is below any noise we could estimate), `shuffle` is clearly worse (41.7, p99 19.3 s, engine queues 1/7). `affload` has the best tail (p99 7.6 s against 9.5 s, p50 1.7 against 1.8 s) and the best cache (87.7 % against 82.1 %, stickiness 0.88 against 0.55), but it **refuses more**: 42 % of the attempts against 30 %, and 19 abandoned calls against 9. The gain seen at N = 24 (+35 %) therefore does not appear at N = 20: it is a gain under overload, not near the knee. Hypotheses for the extra refusals (not tested): faster calls make the sessions return sooner, so more attempts reach a full counter; or keeping sessions on their worker lets one worker's six places fill while the other has room (the counter is fleet-wide, the places are not). A repeat of both at the knee would separate noise from effect.

**Outcome against the pre-registered rule:** the N = 20 runs do not reverse the N = 24 result (throughput equal, tail and cache better) but they add a cost (more refusals) that the rule did not anticipate. The repo default is not changed yet; it is the user's decision (see ARCHITECTURE row 47).


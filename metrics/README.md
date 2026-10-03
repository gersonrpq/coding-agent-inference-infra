# metrics/

| Path | What it is |
| --- | --- |
| `runs/<experiment>/<variant>/` | Load-test runs. `records.jsonl` is the raw per-call data (one JSON line per attempt), `summary.json` the derived numbers, `prometheus.json` the cluster scrape over the run, `timeline.csv` per minute, `gpu.csv` power and memory, `cost.json` the GPU cost. `comparison.md` in an experiment folder is the table across its variants. |
| `evidence/<tag>/` | What the course lists: `gateway_orch_metrics.txt` and `litellm_metrics_full.txt` (scrape of the gateway), `sglang_worker{0,1}.txt` (engine counters), the hop counters (`orch_hops_total`, `orch_hop_tokens_total` in `gateway_orch_metrics.txt`; the `after-ev-n16` folder also has the hop dictionary of the earlier recorder: one line per hop with `src`, `dst`, `tokens`, `cached_tokens`, `prefix`, `session`, `backend`), **`evict_count.txt`** (`sglang:evicted_tokens_total`, `hicache_dropped_tokens_total`), **`status_429_503.txt`** (counts of 429 and 503 by reason). Produced by `script/collect_evidence.sh`. |
| `probes/` | One-off checks of a mechanism (hop, connection close, first-token cut, placement, tenant 429, warm-up). `warm_workers.json` (workers already warm) and `warm_after_restart.json` (freshly restarted worker: the cold-vs-warm contrast). |
| `logs/` | Console logs of the long jobs and probes. |

## Index of the run directories (`runs/`)

| Directory | What it is |
| --- | --- |
| `smoke/` | first check of the load generator (3 sessions, 100 s); not a capacity measurement |
| `eq-ref-n20`, `eq-ref-n28` | queue experiment, reference: the first design, which held waiting requests at the gateway |
| `eq-k-n28`, `eq-k2-n20`, `eq-k2-n24` | queue experiment: the allowance K (0, 2, 4, 6 ...) at N = 28, 20 and 24; K = 2 was chosen from these |
| `er-n20`, `er-n24` | routing experiment: placement policies compared with the same conversations |
| `ev-n16` | N = 16 run used for the hop dictionary and the eviction counters (`evidence/after-ev-n16/`) |
| `fin-knee` | final tests: the knee at N = 16, 20, 24 |
| `fin-rep-affload`, `fin-rep-lb` | paired replicate at N = 20: `affload` against LiteLLM `least-busy` |
| `fin-batch` | 20 % batch sessions (made with the first version of the cap) |
| `fin-regimeB` | sessions that rest 120 s between turns (N = 48) |
| `fin-soak` | 15 minutes at N = 16 |
| `fin2-n20`, `fin2-regimeB`, `fin2-batch` | checks after the cap moved to LiteLLM's own middleware (same seeds as the runs above) |
| `fin3-batch`, `fin4-batch` | the two corrections of the batch rules; `fin4-batch` is the final one |
| `fin5-knee`, `fin5-soak` | the overload point (N = 24) and the 15-minute soak repeated on the final code, with the seeds of `fin-knee` and `fin-soak` (V9) |
| `fin2-agents` | real tool-using agents (`script/agentgen.py`) at N = 12 and 20 |

Protocols and results: `notes/queue-experiment.md`, `notes/routing-experiment.md`, `notes/final-load-tests.md`.

## Index of the scrapes (`evidence/`)

`after-ev-n16/` (hop dictionary and eviction counters of `ev-n16`); `final/` through `final6/` are scrapes saved at successive points of the final tests (`final2/` is the one quoted in the answers of ARCHITECTURE.md, `final6/` also holds the alert rules and their state); `final7/` is the scrape after the repeats on the final code; `final-soak/` is the Prometheus series of the 15-minute soak.

Interpretation of every number is in `notes/findings.md`; decisions in the ARCHITECTURE decision log.

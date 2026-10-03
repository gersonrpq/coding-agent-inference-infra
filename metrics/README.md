# metrics/

| Path | What it is |
| --- | --- |
| `runs/<experiment>/<variant>/` | Load-test runs. `records.jsonl` is the raw per-call data (one JSON line per attempt), `summary.json` the derived numbers, `prometheus.json` the cluster scrape over the run, `timeline.csv` per minute, `gpu.csv` power and memory, `cost.json` the GPU cost. `comparison.md` in an experiment folder is the table across its variants. |
| `evidence/<tag>/` | What the course lists: `gateway_orch_metrics.txt` and `litellm_metrics_full.txt` (scrape of the gateway), `sglang_worker{0,1}.txt` (engine counters), the hop counters (`orch_hops_total`, `orch_hop_tokens_total` in `gateway_orch_metrics.txt`; the `after-ev-n16` folder also has the hop dictionary of the earlier recorder: one line per hop with `src`, `dst`, `tokens`, `cached_tokens`, `prefix`, `session`, `backend`), **`evict_count.txt`** (`sglang:evicted_tokens_total`, `hicache_dropped_tokens_total`), **`status_429_503.txt`** (counts of 429 and 503 by reason). Produced by `script/collect_evidence.sh`. |
| `probes/` | One-off checks of a mechanism (hop, connection close, first-token cut, placement, tenant 429, warm-up). `warm_workers.json` (workers already warm) and `warm_after_restart.json` (freshly restarted worker: the cold-vs-warm contrast). |
| `logs/` | Console logs of the long jobs and probes. |

Interpretation of every number is in `notes/findings.md`; decisions in the ARCHITECTURE decision log.

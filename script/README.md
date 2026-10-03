# script/

Tools for the capacity and cost tests. Standard library only (they run on the server's Python 3.10).

| File | What it does |
| --- | --- |
| `loadgen.py` | Closed-loop SYNTHETIC coding-agent sessions (random code-like text, no tools, forced output lengths) against the LiteLLM gateway; one JSON line per attempt, `summary.json`, verdict vs the 20 s TTFT SLO |
| `gpu_sampler.sh` | Samples power, memory and temperature once per second (utilization is `[N/A]` under MIG) |
| `prom_snapshot.py` | Reads what the cluster measured over the run (sheds by reason, queues, KV, cache hits by tier) |
| `cost_report.py` | GPU cost per 1M tokens / per call / per session-hour, and the same traffic billed at the overflow API rates |
| `sweep.sh` | Runs all of the above for a list of session counts |
| `fetch_results.sh` | Copies `metrics/runs/` from the server to this repo |
| `cap_variant.sh` | Show or change the cap on requests in flight (`max_in_flight_requests_per_worker` = 12 + K) on the live cluster |
| `agentgen.py` | Real agent sessions: the model calls tools (`read_file`, `list_dir`, `grep` on this repository) and the history is what really happened; same record format as `loadgen.py` |
| `make_keys.py` | Creates the LiteLLM virtual keys (pi, loadgen, tenants) through the gateway and saves them in `.env` |
| `routing_experiment.sh`, `router_variant.sh` | Placement policies (`lb`, `shuffle`, `latency`, `aff`, `affload`) compared at one load / switch the live policy |
| `engine_variant.sh` | Engine knobs on the live cluster (radix eviction, chunk size) |
| `analyze_runs.py` | Table (markdown) and figure from a runs directory: `--runs metrics/runs/er-n24 --plot plots/x.png` |
| `warm_workers.py` | Cold vs warm first-token time per worker; exit 1 if a worker is not warm (stage 5b of `launch_cluster.sh`) |
| `plot_final.py` | The figures of the final tests (`plots/final_knee/soak/cache_regimes/batch/agents/ramp.png`) from `metrics/runs/fin*` and `metrics/evidence/final-soak/` |
| `build_answers.py` | Regenerates the ARCHITECTURE section "Answers with evidence" (scrapes, probe logs, tables) from `metrics/`; nothing in it is typed by hand |
| `collect_evidence.sh` | Saves scrapes, eviction counters and the 429/503 counts into `metrics/evidence/<tag>/` |

```bash
bash setup/sync_lambda.sh                                   # local
ssh ... 'cd ~/final_project && script/sweep.sh e1-knee paper "8 12 16 20 24" 480'
bash script/fetch_results.sh                                # local
```

Shapes and assumptions are in the `loadgen.py` docstring and in `ARCHITECTURE.md` ("Load test"). Results land in
`metrics/runs/<tag>/N<sessions>/`; interpretations go in `notes/findings.md`.

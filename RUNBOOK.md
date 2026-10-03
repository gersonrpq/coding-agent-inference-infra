# RUNBOOK: start, check, use and stop the cluster

How to bring the whole system up from nothing (or from a rebooted machine), check that it works, use it, and stop paying for it.
Everything runs on **one GPU server** (1 x H100 PCIe 80 GB, K3s); your computer only edits, syncs and opens a tunnel.
The architecture is in [`README.md`](README.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md); the rules for working here are in [`CLAUDE.md`](CLAUDE.md).

## 0. What you need

| Where | What |
| --- | --- |
| your computer | this repo, `pi` installed (`pi --version`), `ssh`, `rsync`, `python3`; a `.env` (git-ignored, copy [`.env.example`](.env.example)) with `LAMBDA` (`user@host`), `LAMBDA_SSH_KEY`, `LITELLM_MASTER_KEY`, `SUPERLINKED_API_KEY` |
| the server | an NVIDIA GPU with MIG (H100), NVIDIA driver and container toolkit, internet (images and the model, ~40 GB the first time) |
| the server's `~/final_project/.env` | `LITELLM_MASTER_KEY`, `SUPERLINKED_API_KEY`; `launch_cluster.sh` adds `GRAFANA_ADMIN_PASSWORD` and `POSTGRES_PASSWORD` the first time. `sync_lambda.sh` never copies `.env`: put it there once (`scp`, then `chmod 600`) |

Never print or commit `.env`. Go to the server with:

```bash
set -a; source .env; set +a
ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA"          # then:  cd ~/final_project
```

## 1. Is it already up? (30 seconds)

On the server:

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl -n gpu-serving get pods              # expect Running: litellm, postgres-0, prometheus, grafana, mooncake-master, sglang-worker-0, sglang-worker-1
nvidia-smi -L                                # expect two "MIG 3g.40gb" devices
bash script/cap_variant.sh show              # the cap (14) and the live switches
```

If everything is `Running` jump to section 4 (tunnel). If not, continue.

## 2. Start from a stopped or rebooted machine

K3s starts by itself after a reboot, but **the MIG instances do not survive a reboot**. Order matters (MIG needs an idle GPU):

```bash
cd ~/final_project
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
bash setup/mig.sh                                                    # 1. 2 x 3g.40gb (idempotent)
kubectl -n kube-system rollout restart ds/nvidia-device-plugin-daemonset   # 2. so the node advertises 2 x nvidia.com/gpu again
bash setup/launch_cluster.sh                                         # 3. everything else, in order, idempotent
```

`launch_cluster.sh` stages: namespace, device plugin (waits for 2 GPUs), **PostgreSQL**, Mooncake, SGLang workers (waits for the model, up to `SGLANG_READY_TIMEOUT` = 3000 s; about 3 minutes with the model cached), engine smoke test, **warm-up gate**, Prometheus and Grafana (regenerates the dashboards), LiteLLM (waits for its schema migration, about a minute), gateway smoke test. It ends with `Done` and the list of pods. `SKIP_SMOKE=1` and `SKIP_WARM=1` skip the checks.

## 3. Start on a brand-new machine (once)

```bash
# on your computer
bash setup/sync_lambda.sh                  # rsync the repo to $LAMBDA:~/final_project (never deletes, never copies .env)
scp -i "$LAMBDA_SSH_KEY" .env "$LAMBDA":~/final_project/.env       # the server's own .env
# on the server
cd ~/final_project && bash setup/lambda_k3s.sh     # K3s with the nvidia runtime, MIG, device plugin
bash setup/launch_cluster.sh                       # as above; the first run pulls ~20 GB of images and the model
```

## 4. Open the tunnel and create the keys

The gateway, Grafana and Prometheus live inside the cluster; `kubectl port-forward` exposes them on the server and an `ssh -L` brings them to your computer. The server-side forwards stick to **one pod**: rerun them after every LiteLLM or Prometheus restart.

```bash
app/demo/tunnel.sh up        # restarts the server-side forwards and opens ssh -L 4000 (gateway), 3000 (Grafana), 9090 (Prometheus)
app/demo/tunnel.sh status    # tunnel up / down;   app/demo/tunnel.sh down  to close it
```

Clients use their own **virtual key**, not the master key. The keys live in PostgreSQL (its volume survives restarts) and in the `.env` of the machine that created them:

```bash
python3 script/make_keys.py                    # creates pi-demo, loadgen, tenant-acme, tenant-beta; appends LITELLM_KEY_* to .env (never printed)
python3 script/make_keys.py --rotate --only pi-demo     # if the key exists in the database but not in this .env
```

Run it once on the server (the load generators there read the server's `.env`) and once on your computer through the tunnel (pi reads `LITELLM_KEY_PI_DEMO` from yours).

## 5. Check that it works

```bash
curl -s localhost:4000/health/liveliness            # on your computer, through the tunnel: "I'm alive!"
# on the server
python3 script/warm_workers.py                      # cold vs warm first-token time per worker; exit 0 = both warm
python3 metrics/probes/cap_probe.py                 # needs GATEWAY_URL and the keys in the environment; 14 in flight, the 15th gets a 503
```

Grafana: `http://localhost:3000` (user `admin`, password `GRAFANA_ADMIN_PASSWORD` in the server's `.env`). Start with dashboard **03 - Gateway & Admission**. Prometheus alerts: `http://localhost:9090/alerts`.

## 6. Use it

```bash
app/run.sh                    # interactive pi through the gateway (own config in app/pi/agent, own key)
app/demo/demo.sh              # pi builds a todo app into app/todo-app/ and prints what the cluster did (do not run it during a load test)
```

Load tests (on the server, with the repo as the agents' workspace): `script/sweep.sh TAG paper "16 20 24" 360` (synthetic sessions), `script/agent_run.sh TAG "12 20" 360` (real tool-using agents), `script/routing_experiment.sh`. Results land in `metrics/runs/TAG/`; bring them home with `bash script/fetch_results.sh`, tabulate with `python3 script/analyze_runs.py --runs metrics/runs/TAG`, plot with `python3 script/plot_final.py`. Long jobs: write `/tmp/x.sh`, `scp` it, `setsid nohup bash /tmp/x.sh > ~/x.log 2>&1 < /dev/null &` and poll the log.

## 7. Update the code in a running cluster

```bash
bash setup/sync_lambda.sh                       # your computer: push
bash setup/launch_cluster.sh                    # the server: applies what changed and restarts only what needs it
bash setup/stop_forward.sh; WITH_ENGINES=0 bash setup/start_forward.sh     # the server: forwards stick to the old pod
```

`launch_cluster.sh` also resets every live variant (`cap_variant.sh`, `router_variant.sh`, `engine_variant.sh`) to the repo values.

## 8. Stop

The GPU server is billed per hour (3.29 USD) **whether it is busy or not**: stop the instance in the provider's console when you finish. To only stop the workloads and keep the machine:

```bash
kubectl -n gpu-serving scale deploy --all --replicas=0       # workers, LiteLLM, Prometheus, Grafana, Mooncake
kubectl -n gpu-serving scale statefulset/postgres --replicas=0   # keeps its volume (the keys)
# start again: bash setup/launch_cluster.sh
```

Prometheus keeps its history in an `emptyDir`: a restart loses it. Save what you need first (`script/collect_evidence.sh TAG`, `script/fetch_results.sh`).

## 8b. Cost and budgets

The GPU is billed per hour (3.29 USD) busy or not; the cost model and its derivation are in [`ARCHITECTURE.md`](ARCHITECTURE.md), *Cost model*.

- **Cost of a load run:** `python3 script/cost_report.py --run metrics/runs/fin-soak/N16 --price 3.29` prints the GPU bill of the window and the same bill per million tokens, per 1,000 calls and per session-hour, and compares it with the outside provider's rates.
- **What LiteLLM books per request:** the three rates in `cluster/litellm/config.yaml` (`input_cost_per_token`, `cache_read_input_token_cost`, `output_cost_per_token`, in dollars of GPU time). They are provisional: recompute them from the tokens per second served at the sustainable load if the sweep changes.
- **Spend per client:** metric `litellm_litellm_spend_metric_total` with label `api_key_alias` (Prometheus; no Grafana panel, it is only there to be queried). It counts since the LiteLLM pod started.
- **Comparison with an outside provider:** the same traffic at the provider's rates (0.25 USD per million input, 2.00 per million output) was priced on the short, lightly loaded smoke run. If the provider charges full price for cached tokens, the local bill is 0.43 times theirs (cheaper); if it charges cached tokens at 10 %, the local bill is 1.76 times theirs (dearer). It flips on that assumption, and the local cost per call falls as the GPU gets busier. `cost_report.py` prints both cases.
- **Limit a client's budget:** each client has its own virtual key (`script/make_keys.py`). LiteLLM keys accept `max_budget` (dollars) and `budget_duration` (for example `"30d"`); add them next to `max_parallel_requests` in the limits of `script/make_keys.py` and recreate the key (`--rotate --only ALIAS`), or update an existing key through the admin API (`/key/update`). A key over its budget is refused by LiteLLM. This is **not tested** in this cluster; check it with a very small budget before relying on it.

## 9. When something goes wrong

| Symptom | Cause and fix |
| --- | --- |
| a worker pod crashes once at start with "Loaded weights leave no GPU memory for the KV cache" | both workers load at once on the shared GPU; it recovers on the container restart |
| a worker is `Pending` | the node does not advertise 2 GPUs: MIG instances are missing (`bash setup/mig.sh`, restart the device plugin) |
| both workers on MIG instance 0, the second runs out of memory | `CUDA_VISIBLE_DEVICES` must be pinned per worker in the manifests (it is); do not remove it |
| `localhost:4000` does not answer after a restart | forwards stick to the old pod: `app/demo/tunnel.sh up` (or `setup/stop_forward.sh` + `start_forward.sh` on the server) |
| LiteLLM pod `Running` but not Ready for a few minutes | it migrates its PostgreSQL schema on start (the startup probe allows 7 minutes) |
| `curl /metrics` answers 307 | use `/metrics/` with the trailing slash |
| pi: `APIConnectionError: Timeout on reading data from socket` while writing a file | a read timeout shorter than the silent gap of a big tool call (SGLang sends nothing while it generates the arguments). The first-token cut that did this was removed (decisions [59](ARCHITECTURE.md#decision-59), [61](ARCHITECTURE.md#decision-61)); LiteLLM's own `timeout` is 300 s. See `notes/findings.md` |
| pi: `400 Maximum output is 4096 tokens` | not with `app/run.sh` (its config asks for 4096); the guard clamps larger requests anyway |
| 503 with "Worker at capacity: 14 in-flight" | the cap is full, by design; retry |
| 429 on a tenant key | that key is over its own limit |
| `jupyter nbconvert` fails | use `python3 notebook/run_notebook.py` |

## 10. Known pending work

See the Roadmap at the end of [`ARCHITECTURE.md`](ARCHITECTURE.md). At the time of writing only the git commit and the GitHub link remain. Deliberately out of scope (decision [62](ARCHITECTURE.md#decision-62)): KEDA/autoscaling and a `stale_telemetry` shed; the real overflow is blocked because the Superlinked wallet is exhausted (HTTP 402 `INSUFFICIENT_CREDITS` on 2026-10-03). The ramp of a returning worker is built (decision [60](ARCHITECTURE.md#decision-60), `plots/final_ramp.png`).

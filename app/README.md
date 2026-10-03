# app/

The application of the project: **pi**, a coding agent, used through our serving path. It is a launcher, not a new
agent: `app/run.sh` points pi at the gateway and nothing else.

```text
pi (your machine) -- localhost:4000 --> LiteLLM -> security_inspect -> admission -> place -> SGLang (2 workers) -> answer
```

## Use

1. Open the tunnel to the gateway (you do this; the server side is `setup/start_forward.sh`):
   `ssh -i <key> -L 4000:127.0.0.1:4000 <user>@<server>`
2. Have a virtual key for pi: `python3 script/make_keys.py` (through the tunnel) saves `LITELLM_KEY_PI_DEMO` in `.env` (git-ignored). Without it `run.sh` falls back to the master key with a warning.
3. `app/run.sh` (interactive), `app/run.sh -p "do X"` (one shot), `app/run.sh --thinking high` (any pi option).

## What it configures, and why

pi reads its provider list from `PI_CODING_AGENT_DIR`; `run.sh` sets it to `app/pi/agent`, so your own `~/.pi` is not
touched. The reasons for a separate config (all found by running pi 0.74.2 against the cluster):

| Setting | Value here | Why |
| --- | --- | --- |
| default provider | `litellm` | a global pi config may default to an external API, which would **bypass the gateway** |
| `maxTokens` | **4096** | pi's default is 16384; the guard caps output at 4096 and answers `400 Maximum output is 4096 tokens` |
| `contextWindow` | 65536 | the model's `--context-length` |
| `input` | text, image | the model is multimodal and the guard accepts `data:` images |
| `compat.thinkingFormat` | `qwen` | pi sends `enable_thinking` according to its thinking level; the **client** chooses (ARCHITECTURE decision [22](../ARCHITECTURE.md#decision-22)) |
| default thinking level | `off` | keeps the decode short; `--thinking low|medium|high` turns it on per session |
| `apiKey` | the name `LITELLM_KEY_PI_DEMO` | pi resolves it from the environment, so the key is never written to a file; it is pi's own virtual key (limit: 4 requests at once), not the master key |
| `PI_CODING_AGENT_SESSION_DIR` | `app/.sessions` (git-ignored) | sessions stay out of your home directory |

Every client has its own virtual key (LiteLLM with PostgreSQL, created by `script/make_keys.py`): pi uses `LITELLM_KEY_PI_DEMO`, the
load generators use `LITELLM_KEY_LOADGEN`, and the master key is only for administration.

## Verified

`app/run.sh -p "Reply with the single word: ready"` answers `ready`; a coding task (create `fib.py`, run it) was
completed through the cluster in 20 s with 8 model calls, the first with ~1.6K new prompt tokens and the rest with
37-264 new tokens on top of a cached history. Details: `notes/findings.md`.

## Demo: pi builds a todo app through the cluster (`app/demo/`)

One command shows the whole path working with the real client:

```bash
app/demo/demo.sh                 # tunnel -> pi builds the todo app in app/todo-app -> what the cluster did
DEMO_CLEAN=1 app/demo/demo.sh    # start from an empty app/todo-app/
app/demo/demo.sh --interactive   # pi interactive in app/todo-app/, you type the task
app/demo/tunnel.sh up|status|down
```

| File | Role |
| --- | --- |
| `app/demo/tunnel.sh` | starts the server-side port-forwards and one `ssh -L` for the gateway (4000), Grafana (3000) and Prometheus (9090); reads `LAMBDA` and `LAMBDA_SSH_KEY` from `.env` |
| `app/demo/prompt.md` | the task given to pi (a dependency-free todo list: add, toggle, edit, delete, filters, counter, `localStorage`) |
| `app/demo/demo.sh` | opens the tunnel if needed, snapshots the gateway counters, runs `pi -p` with the prompt inside `app/todo-app/`, checks the files, prints the counter differences |
| `app/demo/gateway_stats.py` | before/after of `orch_*`: admitted and refused requests, requests per worker, hops, tenants, overflow decisions |
| `app/todo-app/` | the folder pi writes into. It holds the generated app (`index.html`, `style.css`, `app.js`, `README.md`) |

Every call of the demo is guarded, admitted, placed and queued like any other (that is the point: the app is the real client), and it is the
interactive class. While it runs, Grafana (`http://localhost:3000`, dashboard *03 - Gateway & Admission*) shows the places in use and the
placement per worker. Do not run it during a load test: it is one more uncontrolled session in the measurement.

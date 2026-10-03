#!/usr/bin/env python3
"""Live check of a worker coming back: it must not be slammed to 100 %.

Keeps 6 requests in flight at all times through the gateway, every one from a NEW session (so placement chooses a worker for each, by load,
and the ramp is visible), restarts worker 1 at T_KILL seconds (`kubectl rollout restart`, strategy Recreate: it is down for about 3 minutes)
and records for every request when it was sent, which worker served it and the status. Every 5 s it also reads the gateway's gauges
`orch_worker_available` and `orch_worker_ramp_factor`.

What to look at: (1) before the restart the share of worker 1 is about half; (2) after the poller notices (up to 15 s) nothing is sent to it
and the errors stop; (3) when it answers again its ramp factor goes from 0.25 to 1 in about 60 s (it waits while the fleet p99 is above 10 s)
and its share of new requests climbs with it instead of jumping to 50 %.
Run on the server (needs kubectl, GATEWAY_URL and the keys in the environment). Do not run it during a load test.
"""
import http.client
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.parse

URL = urllib.parse.urlparse(os.environ.get("GATEWAY_URL", "http://localhost:4000"))
KEY = os.environ.get("LITELLM_API_KEY") or os.environ.get("LITELLM_KEY_LOADGEN") or os.environ["LITELLM_MASTER_KEY"]
MASTER = os.environ["LITELLM_MASTER_KEY"]
DURATION = float(os.environ.get("DURATION_S", "460"))
T_KILL = float(os.environ.get("T_KILL_S", "40"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "6"))
WORDS = "alpha beta gamma delta kernel buffer socket thread mutex lambda vector matrix parser token cursor module import export".split()
rows, gauges = [], []
T0 = time.time()
stop = threading.Event()


def one(i):
    rnd = random.Random(i)
    text = " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 9999)) for _ in range(120))
    body = json.dumps({"model": "qwen-coding-local", "max_tokens": 300, "temperature": 0.7, "stream": True, "ignore_eos": True, "priority": 5,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": f"[session {i}] {text}"}]})
    t = time.time() - T0
    try:
        c = http.client.HTTPConnection(URL.hostname, URL.port or 80, timeout=120)
        c.request("POST", "/v1/chat/completions", body, {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        r = c.getresponse()
        worker = (r.getheader("x-litellm-model-api-base") or "").split("//")[-1].split(":")[0]
        r.read()
        rows.append((t, r.status, worker, r.getheader("shed-reason")))
    except Exception as e:
        rows.append((t, 0, "error", repr(e)[:60]))


def loop(k):
    n = 0
    while not stop.is_set():
        one(k * 100000 + n)
        n += 1


def read_gauges():
    while not stop.is_set():
        try:
            c = http.client.HTTPConnection(URL.hostname, URL.port or 80, timeout=10)
            c.request("GET", "/metrics/", headers={"Authorization": "Bearer " + MASTER})
            text = c.getresponse().read().decode()
            g = {}
            for m in re.finditer(r'^orch_worker_(available|ramp_factor)\{worker="(w\d)"\} ([0-9.]+)$', text, re.M):
                g[f"{m.group(1)}_{m.group(2)}"] = float(m.group(3))
            gauges.append((round(time.time() - T0), g))
        except Exception:
            pass
        time.sleep(5)


def kill():
    time.sleep(T_KILL)
    print(f"[{time.time() - T0:.0f}s] restarting worker 1", flush=True)
    subprocess.run(["kubectl", "-n", "gpu-serving", "rollout", "restart", "deploy/sglang-worker-1"], check=False, capture_output=True)


threads = [threading.Thread(target=loop, args=(k,), daemon=True) for k in range(CONCURRENCY)]
threads += [threading.Thread(target=read_gauges, daemon=True), threading.Thread(target=kill, daemon=True)]
for t in threads:
    t.start()
time.sleep(DURATION)
stop.set()
time.sleep(20)

print("window  requests  w0  w1  errors   share_w1   gauges (available_w1, ramp_factor_w1)")
for start in range(0, int(DURATION), 15):
    win = [r for r in rows if start <= r[0] < start + 15]
    ok = [r for r in win if r[1] == 200]
    w1 = sum(1 for r in ok if r[2].endswith("1"))
    w0 = sum(1 for r in ok if r[2].endswith("0"))
    err = sum(1 for r in win if r[1] != 200)
    g = [g for t, g in gauges if start <= t < start + 15]
    last = g[-1] if g else {}
    print(f"{start:5d}s  {len(win):8d}  {w0:2d}  {w1:2d}  {err:6d}   {'-' if not ok else round(w1 / len(ok), 2)}   "
          f"({last.get('available_w1', '?')}, {last.get('ramp_factor_w1', '?')})")
print("errors by status:", {s: sum(1 for r in rows if r[1] == s) for s in sorted({r[1] for r in rows if r[1] != 200})})

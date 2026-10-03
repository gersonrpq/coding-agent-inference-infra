#!/usr/bin/env python3
"""Live check of the KV protection: what happens when the KV pool is as full as the cap allows?

Sends 12 long streaming requests at once (distinct system prompts, about 60K tokens each: 6 per worker, the engine's limit) and samples,
every second, the KV used of each worker and the engine's retraction counter. Then, while they run, sends a request with a NEW
prefix and a request that reuses the prefix of one of the 12 (admission's `kv_pressure` sheds only new prefixes).

By construction the pool holds 6.64 sequences of 64K per worker and MAX_NUM_SEQS is 6, so running sequences alone cannot take more than
6 x 64K / 435K = 88 % of the pool: the default 95 % threshold cannot be reached after admission. Run it twice:
  default threshold (0.95)                -> shows the KV peak and that nothing is retracted or shed
  KV_SHED_THRESHOLD=0.70 on the deployment -> shows the shed path live: the new prefix gets 503 kv_pressure, the reused one is admitted
Run where the gateway is reachable (GATEWAY_URL, the keys in the environment) and the SGLang metrics are on localhost:30000/30001 (the server).
"""
import http.client
import json
import os
import random
import threading
import time
import urllib.parse

URL = urllib.parse.urlparse(os.environ.get("GATEWAY_URL", "http://localhost:4000"))
KEY = os.environ.get("LITELLM_API_KEY") or os.environ.get("LITELLM_KEY_LOADGEN") or os.environ["LITELLM_MASTER_KEY"]
WORDS = "alpha beta gamma delta kernel buffer socket thread mutex lambda vector matrix parser token cursor module import export".split()


def prompt(seed, tokens):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 999)) for _ in range(int(tokens / 3.9)))


def call(system_seed, tokens, max_tokens, results, name):
    body = json.dumps({"model": "qwen-coding-local", "max_tokens": max_tokens, "temperature": 0.7, "stream": True, "ignore_eos": True,
                       "priority": 5, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": "Project notes %d:\n%s" % (system_seed, prompt(system_seed, tokens))},
                                    {"role": "user", "content": "Summarize the notes in one sentence. [%s]" % name}]})
    c = http.client.HTTPConnection(URL.hostname, URL.port or 80, timeout=600)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body, {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    r = c.getresponse()
    headers = dict(r.getheaders())
    data = r.read()
    results.append((name, r.status, round(time.time() - t0, 1), headers.get("shed-reason"), data[:100].decode("utf-8", "ignore") if r.status != 200 else ""))


def sglang(port):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/metrics")
    text = c.getresponse().read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key in ("sglang:kv_used_tokens", "sglang:max_total_num_tokens", "sglang:num_retracted_reqs", "sglang:num_running_reqs", "sglang:num_queue_reqs"):
            if line.startswith(key) and 'priority=' not in line.replace('priority=""', ""):
                out[key.split(":")[1]] = float(line.rsplit(" ", 1)[1])
    return out


results, samples = [], []
stop = threading.Event()


def sampler():
    while not stop.is_set():
        row = {}
        for w, port in (("w0", 30000), ("w1", 30001)):
            try:
                m = sglang(port)
                row[w] = (round(100 * m["kv_used_tokens"] / m["max_total_num_tokens"], 1), int(m.get("num_running_reqs", 0)), int(m.get("num_queue_reqs", 0)), int(m.get("num_retracted_reqs", 0)))
            except Exception:
                row[w] = None
        samples.append((round(time.time() - T0, 0), row))
        time.sleep(1)


T0 = time.time()
threading.Thread(target=sampler, daemon=True).start()
LONG_TOKENS = int(os.environ.get("LONG_TOKENS", "1500"))     # long decodes keep the KV full for ~50 s, longer than the 15 s the gateway's snapshot lags
threads = [threading.Thread(target=call, args=(1000 + i, 60000, LONG_TOKENS, results, f"long{i}")) for i in range(12)]
for t in threads:
    t.start()
    time.sleep(0.3)
print("12 long requests sent; waiting for the KV to fill ...")
TRIGGER = float(os.environ.get("KV_TRIGGER_PCT", "72"))     # send the two extra requests while the KV is this full (or after 75 s)
deadline = time.time() + float(os.environ.get("FILL_WAIT_S", "75"))
while time.time() < deadline:
    last = samples[-1][1] if samples else {}
    if any(v and v[0] >= TRIGGER for v in last.values()):
        break
    time.sleep(0.5)
time.sleep(float(os.environ.get("TRIGGER_DELAY_S", "20")))   # the gateway decides on a snapshot polled every 15 s: let it see the full KV
print(f"extra requests sent at {time.time() - T0:.0f} s, KV now {samples[-1][1] if samples else '?'}")
extra = []
call(2000, 500, 16, extra, "new-prefix")          # a prefix nobody has sent
call(1003, 60000, 16, extra, "reused-prefix")      # the SAME system prompt as long3 (same seed and size): admission has seen its hash
for t in threads:
    t.join()
stop.set()

peak = max((max(v[0] for v in row.values() if v) for _, row in samples if any(row.values())), default=0)
retracted = max((max(v[3] for v in row.values() if v) for _, row in samples if any(row.values())), default=0)
print(f"KV used, worst worker, peak: {peak} %   engine retractions seen: {retracted}")
print("every 10 s (second, {worker: kv %, running, waiting, retracted}):")
for t, row in samples[::10]:
    print(" ", int(t), row)
print("long requests:", sorted({r[1] for r in results}), "statuses; slowest", max(r[2] for r in results), "s")
for name, status, secs, reason, note in extra:
    print(f"{name}: HTTP {status} in {secs} s  shed-reason={reason} {note}")

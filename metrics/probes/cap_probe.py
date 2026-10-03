#!/usr/bin/env python3
"""Live check of the cap on requests in flight: LiteLLM's own admission middleware (max_in_flight_requests_per_worker = 14).

Opens 14 long streaming requests, then sends a 15th and a /metrics scrape while the 14 are running:
  - the 15th must be refused AT ONCE with a 503 (nothing waits at the gateway);
  - the scrape of /metrics/ is still served (we had predicted a 503; the measurement shows 200);
  - when the 14 finish, a new request is served again (the place is released at the end of the stream).
Run where the gateway is reachable (GATEWAY_URL) with LITELLM_KEY_LOADGEN (or LITELLM_API_KEY) and LITELLM_MASTER_KEY in the environment.
Do not run it during a load test.
"""
import http.client
import json
import os
import threading
import time
import urllib.parse

URL = urllib.parse.urlparse(os.environ.get("GATEWAY_URL", "http://localhost:4000"))
KEY = os.environ.get("LITELLM_API_KEY") or os.environ.get("LITELLM_KEY_LOADGEN") or os.environ["LITELLM_MASTER_KEY"]
MASTER = os.environ["LITELLM_MASTER_KEY"]
CAP = int(os.environ.get("CAP", "14"))


def post(max_tokens, results=None, name=None):
    body = json.dumps({"model": "qwen-coding-local", "max_tokens": max_tokens, "temperature": 0.7, "stream": True, "ignore_eos": True,
                       "priority": 5, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": f"[{name}] Write a long essay about queues. " + "alpha " * 30}]})
    c = http.client.HTTPConnection(URL.hostname, URL.port or 80, timeout=300)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body, {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    r = c.getresponse()
    status, headers = r.status, dict(r.getheaders())
    data = r.read()
    out = (name, status, round(time.time() - t0, 2), headers.get("shed-reason") or headers.get("retry-after"), data[:120].decode("utf-8", "ignore") if status != 200 else "")
    if results is not None:
        results.append(out)
    return out


def get(path, key):
    c = http.client.HTTPConnection(URL.hostname, URL.port or 80, timeout=20)
    t0 = time.time()
    c.request("GET", path, headers={"Authorization": "Bearer " + key})
    r = c.getresponse()
    r.read()
    return r.status, round(time.time() - t0, 2)


def gauge():
    c = http.client.HTTPConnection(URL.hostname, URL.port or 80, timeout=20)
    c.request("GET", "/metrics/", headers={"Authorization": "Bearer " + MASTER})
    r = c.getresponse()
    text = r.read().decode()
    return r.status, {l.split()[0]: l.split()[1] for l in text.splitlines() if l.startswith("litellm_admission_") and not l.startswith("#")}


print(f"cap under test: {CAP}")
print("gauge before:", gauge())
held = []
threads = [threading.Thread(target=post, args=(600, held, f"hold{i}")) for i in range(CAP)]
for t in threads:
    t.start()
    time.sleep(0.15)
time.sleep(3)                                          # all of them are running now
print("gauge with the cap full:", gauge())
extra = post(16, None, "extra")
print(f"request {CAP + 1}: HTTP {extra[1]} in {extra[2]} s  (expected 503, at once)  {extra[3] or ''} {extra[4]}")
print("GET /metrics/ with the cap full: HTTP %s in %s s  (served: the cap does not refuse the scrape)" % get("/metrics/", MASTER))
print("GET /health/liveliness with the cap full: HTTP %s in %s s  (exempt, must be 200)" % get("/health/liveliness", MASTER))
for t in threads:
    t.join()
print("the %d held requests: statuses %s" % (len(held), sorted({h[1] for h in held})))
after = post(16, None, "after")
print(f"a new request after they finished: HTTP {after[1]} in {after[2]} s  (expected 200: the place came back)")
print("gauge after:", gauge())

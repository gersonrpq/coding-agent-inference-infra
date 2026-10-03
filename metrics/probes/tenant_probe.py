#!/usr/bin/env python3
"""Live check of per-tenant limits with LiteLLM virtual keys: one tenant cannot own the GPU, and a 429 is not an overflow.

Needs the keys of script/make_keys.py: `tenant-acme` (max_parallel_requests 3) and `tenant-beta`. Run it where the gateway is
reachable (GATEWAY_URL, default http://localhost:4000) with LITELLM_KEY_TENANT_ACME and LITELLM_KEY_TENANT_BETA in the
environment. Tenant acme sends 6 long calls at once: 3 should run (200) and 3 be refused with a 429 by LiteLLM; tenant beta,
at the same time, is not affected. Then LiteLLM's counters are read per key alias.
"""
import http.client
import json
import os
import re
import threading
import time
import urllib.parse

URL = urllib.parse.urlparse(os.environ.get("GATEWAY_URL", "http://localhost:4000"))
KEYS = {"acme": os.environ["LITELLM_KEY_TENANT_ACME"], "beta": os.environ["LITELLM_KEY_TENANT_BETA"]}


def call(tenant, out, i):
    body = json.dumps({"model": "qwen-coding-local", "max_tokens": 200, "temperature": 0.7, "ignore_eos": True, "priority": 5,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": f"[{tenant}-{i}] Write a long essay about queues. " + "alpha " * 30}]})
    c = http.client.HTTPConnection(URL.hostname, URL.port or 80, timeout=120)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body, {"Authorization": "Bearer " + KEYS[tenant], "Content-Type": "application/json"})
    r = c.getresponse()
    r.read()
    out.append((tenant, i, r.status, round(time.time() - t0, 2)))


def metrics():
    c = http.client.HTTPConnection(URL.hostname, URL.port or 80, timeout=10)
    c.request("GET", "/metrics/", headers={"Authorization": "Bearer " + os.environ.get("LITELLM_MASTER_KEY", KEYS["acme"])})
    return c.getresponse().read().decode()


results = []
threads = [threading.Thread(target=call, args=("acme", results, i)) for i in range(6)] + \
          [threading.Thread(target=call, args=("beta", results, 0))]
for t in threads:
    t.start()
    time.sleep(0.05)
for t in threads:
    t.join()

print("tenant  call  status  seconds")
for tenant, i, status, secs in sorted(results):
    print(f"{tenant:6s}  {i:4d}  {status:6d}  {secs}")
print("acme statuses:", sorted(r[2] for r in results if r[0] == "acme"), "| beta statuses:", [r[2] for r in results if r[0] == "beta"])
print("--- LiteLLM counters per key alias")
for line in metrics().splitlines():
    if line.startswith("litellm_proxy_total_requests_metric_total") and 'api_key_alias="tenant-' in line:
        m = re.search(r'api_key_alias="([^"]+)".*status_code="(\d+)".*\} ([0-9.]+)$', line)
        if m:
            print(f"  {m.group(1)} HTTP {m.group(2)}: {m.group(3)}")

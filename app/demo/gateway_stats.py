#!/usr/bin/env python3
"""What did the cluster do for this session? Reads the gateway's /metrics before and after the demo and prints the difference.

    LITELLM_MASTER_KEY=... python3 app/demo/gateway_stats.py snapshot /tmp/before.json
    LITELLM_MASTER_KEY=... python3 app/demo/gateway_stats.py diff /tmp/before.json

Standard library only. Series printed: admitted and refused requests (by reason), requests per worker (placement), hops, places
in use, tenant requests and overflow decisions.
"""
import json
import os
import re
import sys
import urllib.request

URL = os.environ.get("GATEWAY_URL", "http://localhost:4000")
WANTED = ("orch_requests_admitted_total", "orch_requests_shed_total", "orch_placement_total", "orch_placement_session_total",
          "orch_hops_total", "orch_hops_local_total", "orch_hop_tokens_total", "orch_hop_cached_tokens_total",
          "orch_tenant_requests_total", "orch_overflow_decisions_total", "orch_gateway_in_flight")


def scrape() -> dict:
    req = urllib.request.Request(URL + "/metrics/", headers={"Authorization": "Bearer " + os.environ["LITELLM_MASTER_KEY"]})
    text = urllib.request.urlopen(req, timeout=15).read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(orch_[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if m and m.group(1) in WANTED:
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def main() -> None:
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "snapshot":
        json.dump(scrape(), open(path, "w"))
        return
    before, after = json.load(open(path)), scrape()
    print("What the cluster did during the demo (gateway counters, after - before):")
    shown = False
    for key in sorted(after):
        if key.startswith("orch_gateway_in_flight"):
            continue
        delta = after[key] - before.get(key, 0.0)
        if delta:
            print(f"  {key:110s} +{delta:g}")
            shown = True
    if not shown:
        print("  (no change: did the demo reach the gateway?)")
    print(f"  places in use now: {after.get('orch_gateway_in_flight', 0):g} of 14")


main()

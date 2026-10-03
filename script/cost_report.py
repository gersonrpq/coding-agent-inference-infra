#!/usr/bin/env python3
"""GPU cost of a load-generator run, from its summary.json and the GPU power CSV.

    python3 script/cost_report.py --run metrics/runs/x/N12 --price 3.29

Cost model (assumptions are printed): the GPU is rented by the hour, so what a run costs is price x time; what
changes with load is how many tokens that hour yields. Energy is reported separately (power x time) because
the hourly price already includes it. Overflow comparison: the same traffic billed at the Superlinked rates in
cluster/litellm/config.yaml ($0.25 per 1M input, $2.00 per 1M output), assuming no prefix-cache discount.
"""
import argparse
import csv
import json
import os
import time

IN_PER_M, OUT_PER_M = 0.25, 2.00


def read_power(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                ts = time.mktime(time.strptime(r["timestamp"].strip().split(".")[0], "%Y/%m/%d %H:%M:%S"))
                rows.append((ts, float(r["power_w"])))
            except (ValueError, KeyError):
                pass
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--price", type=float, default=3.29, help="USD per GPU hour")
    p.add_argument("--slots", type=int, default=12, help="running slots of the fleet (2 workers x MAX_NUM_SEQS)")
    p.add_argument("--idle-csv", help="GPU CSV taken with no load, for the idle power baseline")
    a = p.parse_args()
    s = json.load(open(os.path.join(a.run, "summary.json")))
    inter = s["classes"].get("all") or {}
    t0, win = s["start_epoch"] + s["warmup_s"], s["window_s"]
    power = [w for ts, w in read_power(os.path.join(a.run, "gpu.csv")) if t0 <= ts <= t0 + win]
    idle = [w for _, w in read_power(a.idle_csv)] if a.idle_csv and os.path.exists(a.idle_csv) else []
    hours = win / 3600
    gpu_cost = a.price * hours
    tin, tcached, tout = inter.get("prompt_tokens", 0), inter.get("cached_tokens_reported", 0), inter.get("completion_tokens", 0)
    tunc = max(0, tin - tcached)
    served = inter.get("served", 0)
    avg_w = sum(power) / len(power) if power else None
    rows = [json.loads(l) for l in open(os.path.join(a.run, "records.jsonl"))]
    busy = sum(r["total"] for r in rows if s["warmup_s"] <= r["t"] < s["duration_s"] and r["status"] == 200 and not r["error"])
    out = {
        "sessions": s["sessions"], "window_s": win, "price_per_hour": a.price,
        "gpu_cost_usd_for_window": round(gpu_cost, 4),
        "avg_power_w": avg_w and round(avg_w, 1), "idle_power_w": idle and round(sum(idle) / len(idle), 1),
        "energy_wh": avg_w and round(avg_w * hours, 2),
        "slot_utilization": round(busy / (win * a.slots), 3),
        "served_calls": served, "prompt_tokens": tin, "cached_prompt_tokens": tcached, "uncached_prompt_tokens": tunc,
        "cached_share": tin and round(tcached / tin, 3), "completion_tokens": tout,
        "usd_per_1k_served_calls": served and round(gpu_cost / served * 1000, 4),
        "usd_per_1M_prompt_tokens_all": tin and round(gpu_cost / tin * 1e6, 3),
        "usd_per_1M_prompt_tokens_uncached": tunc and round(gpu_cost / tunc * 1e6, 3),
        "usd_per_1M_completion_tokens": tout and round(gpu_cost / tout * 1e6, 3),
        "usd_per_session_hour": round(a.price / s["sessions"], 4),
    }
    # the three per-token figures above are the SAME bill divided by different counts: they are not additive
    for label, cached_price in (("no_cache_discount", 1.0), ("cached_at_10pct", 0.1)):
        bill = ((tunc + cached_price * tcached) * IN_PER_M + tout * OUT_PER_M) / 1e6
        out["overflow_bill_" + label + "_usd"] = round(bill, 4)
        out["local_over_overflow_" + label] = bill and round(gpu_cost / bill, 2)
    out["notes"] = ["gpu cost = hourly price x window (the machine is billed whether busy or not)",
                    "the per-token figures divide the whole bill by one kind of token each; do not add them",
                    "overflow: same traffic at the Superlinked rates; whether cached tokens are discounted is unknown, so both cases are shown",
                    "local is cheaper than overflow when local_over_overflow_* < 1",
                    "slot_utilization = served-call seconds / (window x slots); low utilization makes every per-token cost look high"]
    with open(os.path.join(a.run, "cost.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

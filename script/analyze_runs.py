#!/usr/bin/env python3
"""Compares the runs found in a directory (one sub-directory per run) as a markdown table and a figure.

    python3 script/analyze_runs.py --runs metrics/runs/er-n24 --order "lb shuffle latency aff affload" [--plot plots/routing_n24.png]

A run directory needs summary.json and records.jsonl (script/loadgen.py); prometheus.json is used when present.
Columns: served per minute, TTFT p50/p99 of served calls, shed rate, share of prompt tokens served from cache, the
same split by whether the call stayed on the worker of the session's previous call, worker stickiness, share of calls
on worker 0 (balance), decode tok/s per sequence and the maximum engine queue per worker.
"""
import argparse
import json
import math
import os


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, max(0, math.ceil(p / 100 * len(v)) - 1))] if v else None


def served(records):
    return [r for r in records if r.get("status") == 200 and not r.get("error") and r.get("ttft") is not None]


def cached_by_continuity(records):
    """Cached share of calls inside a turn, split by whether the session's previous call used the same worker."""
    last, groups = {}, {True: [0, 0], False: [0, 0]}
    for r in sorted(served(records), key=lambda r: r["t"]):
        w = r.get("worker")
        if w is None or r.get("cached_tokens") is None or not r.get("prompt_tokens"):
            continue
        s = r["session"]
        if s in last and r["call"] > 1:
            g = groups[last[s] == w]
            g[0] += r["cached_tokens"]
            g[1] += r["prompt_tokens"]
        last[s] = w
    share = lambda g: round(100 * g[0] / g[1], 1) if g[1] else None
    return share(groups[True]), share(groups[False]), groups[True][1] > 0, groups[False][1] > 0


def summarize_run(path, warmup):
    summary = json.load(open(os.path.join(path, "summary.json")))
    c = summary["classes"]["interactive"]
    recs = [json.loads(l) for l in open(os.path.join(path, "records.jsonl"))]
    recs = [r for r in recs if r["t"] >= warmup]
    ok = served(recs)
    pt = sum(r["prompt_tokens"] or 0 for r in ok)
    ca = sum(r["cached_tokens"] or 0 for r in ok if r.get("cached_tokens") is not None)
    workers = [r["worker"] for r in ok if r.get("worker")]
    first = sorted(set(workers))[0] if workers else None
    dec = [r["completion_tokens"] / (r["total"] - r["ttft"]) for r in ok
           if (r.get("completion_tokens") or 0) >= 100 and r["total"] - r["ttft"] > 0.5]
    same, other, _, _ = cached_by_continuity(recs)
    queue = None
    try:
        prom = json.load(open(os.path.join(path, "prometheus.json")))
        queue = {x["labels"]["worker"]: x["value"] for x in prom["engine_queue_max"]}
    except Exception:
        pass
    return {
        "served_per_min": c["served_per_min"], "ttft_p50": c["ttft_p50"], "ttft_p99": c["ttft_p99"],
        "shed_pct": round(100 * c["shed_rate"], 1), "cached_pct": round(100 * ca / pt, 1) if pt else None,
        "cached_same": same, "cached_other": other, "stickiness": c.get("worker_stickiness"),
        "share_w0": round(100 * workers.count(first) / len(workers), 1) if workers else None,
        "decode_tps": round(pct(dec, 50), 1) if dec else None, "engine_queue_max": queue,
    }


def table(rows):
    f = lambda x, d=1: "-" if x is None else (round(x, d) if isinstance(x, float) else x)
    out = ["| run | served/min | TTFT p50 | TTFT p99 | shed % | cached % | cached % same worker | cached % other worker | stickiness | calls on worker 0 % | decode tok/s | engine queue max |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, r in rows:
        q = "-" if not r["engine_queue_max"] else "/".join(str(int(v)) for _, v in sorted(r["engine_queue_max"].items()))
        out.append(f"| {name} | {f(r['served_per_min'])} | {f(r['ttft_p50'])} | {f(r['ttft_p99'])} | {f(r['shed_pct'])} | {f(r['cached_pct'])} | "
                   f"{f(r['cached_same'])} | {f(r['cached_other'])} | {f(r['stickiness'], 2)} | {f(r['share_w0'])} | {f(r['decode_tps'])} | {q} |")
    return "\n".join(out)


def plot(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n, _ in rows]
    panels = [("Served calls per minute (higher is better)", "served_per_min", None),
              ("TTFT p99 of served calls, s (SLO 20 s)", "ttft_p99", 20),
              ("Prompt served from cache, %", "cached_pct", None),
              ("Calls that stayed on the worker of the previous call, %", "stickiness", None),
              ("Calls on worker 0, % (50 = balanced)", "share_w0", 50),
              ("Decode speed per sequence, tok/s", "decode_tps", None)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (label, key, ref) in zip(axes.flat, panels):
        vals = [(r[key] * 100 if key == "stickiness" and r[key] is not None else r[key]) or 0 for _, r in rows]
        bars = ax.bar(names, vals, color="#3b6ea5")
        if ref is not None:
            ax.axhline(ref, color="#c0392b", linewidth=1.2, linestyle="--")
        ax.set_title(label, fontsize=9)
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--order", default="")
    p.add_argument("--warmup", type=float, default=60)
    p.add_argument("--plot")
    p.add_argument("--title", default="")
    a = p.parse_args()
    found = [d for d in sorted(os.listdir(a.runs)) if os.path.exists(os.path.join(a.runs, d, "summary.json"))]
    order = [d for d in a.order.split() if d in found] + [d for d in found if d not in a.order.split()]
    rows = [(d, summarize_run(os.path.join(a.runs, d), a.warmup)) for d in order]
    md = table(rows)
    print(md)
    open(os.path.join(a.runs, "comparison.md"), "w").write(md + "\n")
    if a.plot:
        plot(rows, a.plot, a.title or os.path.basename(a.runs.rstrip("/")))
        print("figure:", a.plot)


if __name__ == "__main__":
    main()

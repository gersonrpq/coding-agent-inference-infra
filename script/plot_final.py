#!/usr/bin/env python3
"""Figures of the final load tests (plots/final_*.png), from metrics/runs/fin-* and metrics/evidence/final-soak/.

    python3 script/plot_final.py

  final_soak.png     the 15-minute soak: engine queue and running per worker, KV used, requests in flight, served and TTFT per minute
  final_knee.png     the knee: served per minute, TTFT p99 and refusals against the number of sessions N (and the replicate at N = 20)
  final_cache_regimes.png   share of the prompt read from cache at the start of a turn and inside a turn, regime A (20 s pauses) against B (120 s)
  final_batch.png    the four 20 %-batch runs (decisions 57): interactive refusals and what batch got, as the admission rules changed
  final_agents.png   synthetic sessions against real tool-using agents at N = 20: refusals, TTFT p99 and output length
  final_ramp.png     worker 1 restarted under load: availability, ramp factor and its share of the new requests (metrics/logs/ramp_probe.log)
Needs matplotlib (the analysis machine only; the server does not run this).
"""
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "metrics", "runs")
SOAK = os.path.join(ROOT, "metrics", "evidence", "final-soak")
OUT = os.path.join(ROOT, "plots")


def summary(tag, n):
    return json.load(open(os.path.join(RUNS, tag, f"N{n}", "summary.json")))["classes"]["interactive"]


def series(name):
    """{label: (minutes since the start, values)} from a saved Prometheus range query."""
    data = json.load(open(os.path.join(SOAK, name + ".json")))["data"]["result"]
    out = {}
    for r in data:
        label = r["metric"].get("worker") or r["metric"].get("reason") or r["metric"].get("alertname") or "all"
        t0 = r["values"][0][0]
        out[label] = ([(t - t0) / 60 for t, _ in r["values"]], [float(v) for _, v in r["values"]])
    return out


def soak():
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for lab, (x, y) in sorted(series("engine_queue").items()):
        ax[0][0].plot(x, y, label=f"worker {lab}")
    ax[0][0].set_title("Engine waiting queue (SGLang, K = 2 allowed)")
    ax[0][0].set_ylabel("requests")
    for lab, (x, y) in sorted(series("engine_running").items()):
        ax[0][1].plot(x, y, label=f"worker {lab}")
    ax[0][1].axhline(6, color="red", ls="--", lw=1, label="MAX_NUM_SEQS = 6")
    ax[0][1].set_title("Engine running")
    for lab, (x, y) in sorted(series("kv_used_pct").items()):
        ax[0][2].plot(x, y, label=f"worker {lab}")
    ax[0][2].axhline(95, color="red", ls="--", lw=1, label="kv_pressure threshold")
    ax[0][2].set_ylim(0, 100)
    ax[0][2].set_title("KV used (% of the pool)")
    inflight = series("in_flight")
    for lab, (x, y) in inflight.items():
        ax[1][0].plot(x, y, label="in flight (counter)")
    ax[1][0].axhline(14, color="red", ls="--", lw=1, label="cap 12 + K = 14")
    ax[1][0].set_title("Requests in flight in the gateway")
    rows = list(csv.DictReader(open(os.path.join(RUNS, "fin-soak", "N16", "timeline.csv"))))
    m = [int(r["minute"]) for r in rows]
    ax[1][1].bar(m, [int(r["served"]) for r in rows], color="#4477aa", label="served")
    ax[1][1].bar(m, [int(r["shed"]) for r in rows], bottom=[int(r["served"]) for r in rows], color="#cc6677", label="refused")
    ax[1][1].set_title("Attempts per minute (N = 16 sessions)")
    ax[1][1].set_xlabel("minute of the run")
    ax[1][2].plot(m, [float(r["ttft_p50"] or "nan") for r in rows], label="TTFT p50")
    ax[1][2].plot(m, [float(r["ttft_p99"] or "nan") for r in rows], label="TTFT p99")
    ax[1][2].axhline(20, color="red", ls="--", lw=1, label="SLO 20 s")
    ax[1][2].set_title("Interactive TTFT per minute (s)")
    ax[1][2].set_xlabel("minute of the run")
    for a in ax.flat:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    ax[0][0].set_xlabel("minutes")
    fig.suptitle("Soak: 15 minutes, N = 16 agent sessions, final configuration (affload, K = 2, first-token cut 20 s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "final_soak.png"), dpi=130)


def knee():
    ns = [16, 20, 24]
    pts = [(n, summary("fin-knee", n)) for n in ns]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(ns, [v["served_per_min"] for _, v in pts], "o-", label="affload (first runs)")
    rep = summary("fin-rep-affload", 20), summary("fin-rep-lb", 20)
    ax[0].plot([20], [rep[0]["served_per_min"]], "s", color="C0", label="affload replicate (seed 400)")
    ax[0].plot([20], [rep[1]["served_per_min"]], "^", color="C3", label="least-busy (seed 400)")
    ax[0].set_title("Served calls per minute")
    ax[1].plot(ns, [v["ttft_p99"] for _, v in pts], "o-", label="TTFT p99")
    ax[1].plot([20], [rep[0]["ttft_p99"]], "s", color="C0", label="affload replicate")
    ax[1].plot([20], [rep[1]["ttft_p99"]], "^", color="C3", label="least-busy")
    ax[1].axhline(20, color="red", ls="--", lw=1, label="SLO 20 s")
    ax[1].set_title("Interactive TTFT p99 (s)")
    ax[2].plot(ns, [100 * v["shed_rate"] for _, v in pts], "o-", label="refused attempts")
    ax[2].plot([20], [100 * rep[0]["shed_rate"]], "s", color="C0", label="affload replicate")
    ax[2].plot([20], [100 * rep[1]["shed_rate"]], "^", color="C3", label="least-busy")
    ax[2].set_title("Refused attempts (%)")
    for a in ax:
        a.set_xlabel("sessions N")
        a.set_xticks(ns)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle("The knee with the final configuration (360 s runs, 60 s warm-up)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "final_knee.png"), dpi=130)


def cache_regimes():
    a = summary("fin-soak", 16)["by_position"]
    b = summary("fin-regimeB", 48)["by_position"]
    labels = ["start of a turn", "inside a turn"]
    xa = [100 * (a[k]["cached_share"] or 0) for k in ("turn_start", "within_turn")]
    xb = [100 * (b[k]["cached_share"] or 0) for k in ("turn_start", "within_turn")]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.35
    ax.bar([i - w / 2 for i in range(2)], xa, w, label="regime A: 20 s between turns (soak, N = 16)")
    ax.bar([i + w / 2 for i in range(2)], xb, w, label="regime B: 120 s idle (N = 48)")
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of the prompt read from cache (%)")
    ax.set_ylim(0, 100)
    ax.set_title("A session that rests loses its GPU cache")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "final_cache_regimes.png"), dpi=130)


def klass(tag, n, cls):
    return json.load(open(os.path.join(RUNS, tag, f"N{n}", "summary.json")))["classes"][cls]


def batch():
    runs = [("fin-batch", "F3\nold counter,\neager rule"), ("fin2-batch", "V4\nnative cap,\nno batch rule"),
            ("fin3-batch", "V4b\nfloor +\nwrong share"), ("fin4-batch", "V4c (final)\nfloor + share\nof any class")]
    inter = [100 * klass(t, 20, "interactive")["shed_rate"] for t, _ in runs]
    served = [klass(t, 20, "batch")["served_per_min"] for t, _ in runs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(range(4), inter, color=["#999999", "#cc6677", "#cc6677", "#4477aa"])
    ax[0].axhline(10, color="red", ls="--", lw=1, label="target <= 10 %")
    ax[0].set_title("Interactive attempts refused (%)")
    ax[1].bar(range(4), served, color=["#999999", "#cc6677", "#cc6677", "#4477aa"])
    ax[1].set_title("Batch calls served per minute")
    for a in ax:
        a.set_xticks(range(4))
        a.set_xticklabels([l for _, l in runs], fontsize=8)
        a.grid(alpha=0.3, axis="y")
    ax[0].legend(fontsize=8)
    fig.suptitle("20 % batch sessions, N = 20: protecting interactive without starving batch")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "final_batch.png"), dpi=130)


def agents():
    labels = ["synthetic\nloadgen N = 20", "real agents\nagentgen N = 20"]
    syn, real = summary("fin2-n20", 20), klass("fin2-agents", 20, "interactive")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].bar(range(2), [100 * syn["shed_rate"], 100 * real["shed_rate"]], color=["#999999", "#4477aa"])
    ax[0].set_title("Attempts refused (%)")
    ax[1].bar(range(2), [syn["ttft_p99"], real["ttft_p99"]], color=["#999999", "#4477aa"])
    ax[1].axhline(20, color="red", ls="--", lw=1, label="SLO 20 s")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Interactive TTFT p99 (s)")
    rows = [json.loads(l) for l in open(os.path.join(RUNS, "fin2-agents", "N20", "records.jsonl"))]
    ct = sorted(r["completion_tokens"] for r in rows if r["status"] == 200 and r["completion_tokens"] is not None)
    syn_rows = [json.loads(l) for l in open(os.path.join(RUNS, "fin2-n20", "N20", "records.jsonl"))]
    sct = sorted(r["completion_tokens"] for r in syn_rows if r["status"] == 200 and r["completion_tokens"] is not None)
    q = lambda a, p: a[int(p * (len(a) - 1))]
    ax[2].bar([0, 1], [q(sct, .5), q(ct, .5)], width=0.35, label="median", color="#4477aa")
    ax[2].bar([0.4, 1.4], [q(sct, .95), q(ct, .95)], width=0.35, label="p95", color="#88aacc")
    ax[2].set_xticks([0.2, 1.2])
    ax[2].set_title("Output tokens per call")
    ax[2].legend(fontsize=8)
    for a in ax[:2]:
        a.set_xticks(range(2))
    for a, lab in ((ax[0], labels), (ax[1], labels), (ax[2], labels)):
        a.set_xticklabels(lab, fontsize=8)
        a.grid(alpha=0.3, axis="y")
    fig.suptitle("The real agent is lighter than the synthetic load: short tool calls, not forced outputs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "final_agents.png"), dpi=130)


def ramp():
    rows = []
    for line in open(os.path.join(ROOT, "metrics", "logs", "ramp_probe.log")):
        m = re.match(r"\s*(\d+)s\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.-]+)\s+\(([0-9.?]+), ([0-9.?]+)\)", line)
        if m:
            share = None if m.group(6) == "-" else float(m.group(6))
            rows.append((int(m.group(1)), share, float(m.group(7)), float(m.group(8)), int(m.group(5))))
    t = [r[0] + 7.5 for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, [r[1] if r[1] is not None else float("nan") for r in rows], "o-", color="#4477aa", label="share of the requests served by worker 1")
    ax[0].axhline(0.5, color="gray", ls=":", lw=1)
    ax[0].set_ylim(0, 0.7)
    ax[0].set_title("Worker 1 restarted at 40 s under a constant load of new sessions")
    ax[0].legend(fontsize=8)
    ax[1].step(t, [r[2] for r in rows], where="mid", color="#cc6677", label="orch_worker_available (1 = answers)")
    ax[1].plot(t, [r[3] for r in rows], "o-", color="#228833", label="orch_worker_ramp_factor")
    ax[1].bar(t, [r[4] / 3 for r in rows], width=6, color="#999999", alpha=0.6, label="failed requests (/3)")
    ax[1].set_xlabel("seconds since the start of the probe")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "final_ramp.png"), dpi=130)


if __name__ == "__main__":
    soak()
    knee()
    cache_regimes()
    batch()
    agents()
    ramp()
    print("plots/final_soak.png, final_knee.png, final_cache_regimes.png, final_batch.png, final_agents.png, final_ramp.png")

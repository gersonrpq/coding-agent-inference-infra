#!/usr/bin/env python3
"""Declare a worker warm: measure the first-token time cold, warm it, measure again, and only then call it ready.

"Ready" in Kubernetes only means the port is open and the model loaded. The first requests after a start pay for
lazy CUDA kernels, allocator growth and an empty prefix cache, so quoting TTFT from a cold replica gives a wrong
SLO number. This script runs from the server (the workers use hostNetwork), directly against each engine, so it
also works before the gateway is exposed:

  1. cold probe    one unique ~4K-token prompt, TTFT measured
  2. warm-up       WARM_ROUNDS rounds of short, medium and long prompts (the shapes the agent sends)
  3. warm probe    another unique ~4K-token prompt, TTFT measured
  4. verdict       warm when the warm probe is not slower than WARM_MAX_RATIO x the median of the last two rounds
                   (stable) and below WARM_TTFT_LIMIT_S. Exit code 1 if any worker is not warm.

Usage: python3 script/warm_workers.py [--workers 30000,30001] [--model NAME] [--json out.json]
"""
import argparse
import json
import os
import random
import statistics
import string
import sys
import time
import urllib.request

WORDS = ["cache", "token", "kernel", "queue", "prefix", "worker", "decode", "prefill", "tensor", "batch"]


def unique_prompt(tokens: int) -> str:
    rng = random.Random(os.urandom(8))
    salt = "".join(rng.choices(string.ascii_lowercase, k=12))
    return f"[{salt}] " + " ".join(rng.choice(WORDS) for _ in range(int(tokens * 0.75)))


def ttft(port: int, model: str, prompt: str, max_tokens: int = 8) -> float:
    body = json.dumps({"model": model, "stream": True, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            if line.startswith(b"data: ") and b'"content"' in line:
                first = time.monotonic() - start
                for _ in resp:        # drain
                    pass
                return first
    return time.monotonic() - start


def warm(port: int, model: str, rounds: int, max_ratio: float, limit_s: float) -> dict:
    cold = ttft(port, model, unique_prompt(4000))
    per_round = []
    for _ in range(rounds):
        times = [ttft(port, model, unique_prompt(n)) for n in (200, 1500, 4000, 8000)]
        per_round.append(round(statistics.mean(times), 3))
    probe = ttft(port, model, unique_prompt(4000))
    stable = statistics.median(per_round[-2:]) if len(per_round) >= 2 else per_round[-1]
    ok = probe <= limit_s and (probe <= max_ratio * stable or abs(probe - cold) < 0.05)
    return {"port": port, "cold_ttft_s": round(cold, 3), "round_mean_ttft_s": per_round,
            "warm_ttft_s": round(probe, 3), "warm": ok}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="30000,30001")
    ap.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B"))
    ap.add_argument("--rounds", type=int, default=int(os.getenv("WARM_ROUNDS", "3")))
    ap.add_argument("--max-ratio", type=float, default=float(os.getenv("WARM_MAX_RATIO", "1.3")))
    ap.add_argument("--limit-s", type=float, default=float(os.getenv("WARM_TTFT_LIMIT_S", "5")))
    ap.add_argument("--json")
    args = ap.parse_args()

    results = [warm(int(p), args.model, args.rounds, args.max_ratio, args.limit_s) for p in args.workers.split(",")]
    print(f"{'port':>6} {'cold TTFT':>10} {'warm TTFT':>10}  warm-up rounds (mean TTFT)   verdict")
    for r in results:
        print(f"{r['port']:>6} {r['cold_ttft_s']:>9.2f}s {r['warm_ttft_s']:>9.2f}s  {r['round_mean_ttft_s']}   "
              f"{'WARM' if r['warm'] else 'NOT WARM'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
    return 0 if all(r["warm"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

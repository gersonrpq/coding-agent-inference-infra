#!/usr/bin/env python3
"""Closed-loop synthetic coding-agent load generator for the LiteLLM gateway (stdlib only).

Each session imitates an agent such as pi: a system prompt shared by every session, a history that grows
call after call, tool time between calls and user idle time between turns. Requests are streaming chat
completions sent to the gateway (never to SGLang), so guard, admission and queue are all in the path.

Shapes come from the Copilot-trace paper (arXiv 2608.00101) and from ARCHITECTURE.md:
  calls per turn      lognormal, median 4.5
  output tokens       lognormal, median 247, clipped to 1000 (thinking sessions: median 800, up to 4000)
  tool result tokens  lognormal around --profile's median, 0.5K..8K
  prompt length       --profile light (4-20K start) or paper (30-52K start), compaction above --max-prompt

Records every attempt (one JSON line) and writes summary.json. Verdict: p99 TTFT of the interactive class
<= --slo-ttft and its shed rate <= --max-shed (ARCHITECTURE.md decision 16).

    LITELLM_MASTER_KEY=... python3 script/loadgen.py --url http://10.43.46.95:4000 \
        --sessions 12 --duration 480 --warmup 120 --profile paper --out metrics/runs/x/N12
"""
import argparse
import http.client
import json
import math
import os
import random
import sys
import threading
import time
import urllib.parse

WORDS = ("alpha beta gamma delta kernel buffer socket thread mutex lambda vector matrix parser token cursor "
         "module import export return yield async await class struct enum trait impl handler router queue cache "
         "index table column shard replica leader follower commit rebase branch merge deploy rollback metric trace").split()

PROFILES = {
    "light": {"initial": (4000, 20000), "tool_median": 1500},
    "paper": {"initial": (30000, 52000), "tool_median": 2000},
}


def lognorm(rnd, median, sigma, lo, hi):
    return min(hi, max(lo, rnd.lognormvariate(math.log(median), sigma)))


def position(r):
    """Where a call sits in its session: the first call ever, the first call of a later turn (the cache may
    have gone cold during the user's idle time), or a call inside a turn (history was used a few seconds ago)."""
    if r["turn"] == 1 and r["call"] == 1:
        return "session_start"
    return "turn_start" if r["call"] == 1 else "within_turn"


def pct(values, p):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, max(0, math.ceil(p / 100 * len(v)) - 1))]


class Text:
    """Synthetic code-like text. tokens_per_word is calibrated for the Qwen tokenizer (digits split)."""

    def __init__(self, tokens_per_word):
        self.tpw = tokens_per_word

    def make(self, rnd, tokens):
        n = max(1, int(tokens / self.tpw))
        return " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 999)) for _ in range(n))


class Recorder:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.fh = open(path, "a", buffering=1)
        self.rows = []

    def add(self, row):
        with self.lock:
            self.rows.append(row)
            self.fh.write(json.dumps(row) + "\n")

    def snapshot(self):
        with self.lock:
            return list(self.rows)


def classify_shed(status, body_text, header_reason):
    """Why a request was refused. Admission sets the `shed-reason` header; refusals that come from the engine
    or from a timeout have no header, so they are recognised by status and message."""
    if header_reason:
        return header_reason
    text = (body_text or "").lower()
    if status in (503, 529):
        if "worker at capacity" in text:         # LiteLLM's admission middleware: the cap of 14 in flight
            return "queue_full"
        if "queue is full" in text:
            return "engine_queue_full"
        if "higher priority" in text:
            return "engine_evicted"
        return "upstream_503"
    if status in (408, 504) or "timeout" in text or "timed out" in text:
        return "timeout"
    return None


def one_call(host, port, key, body, timeout):
    t0 = time.time()
    rec = {"worker": None, "status": None, "ttfb": None, "ttft": None, "total": None, "prompt_tokens": None,
           "completion_tokens": None, "cached_tokens": None, "shed_reason": None, "retry_after": None,
           "error": None, "text": ""}
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        resp = conn.getresponse()
        rec["status"] = resp.status
        rec["ttfb"] = time.time() - t0
        rec["worker"] = resp.getheader("x-litellm-model-api-base")   # which SGLang worker LiteLLM routed to
        if resp.status != 200:
            rec["error"] = resp.read(2000).decode("utf8", "replace")[:300]
            rec["shed_reason"] = classify_shed(resp.status, rec["error"], resp.getheader("shed-reason"))
            rec["retry_after"] = resp.getheader("retry-after")
        else:
            text = []
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                usage = ev.get("usage")
                if usage:
                    rec["prompt_tokens"] = usage.get("prompt_tokens")
                    rec["completion_tokens"] = usage.get("completion_tokens")
                    details = usage.get("prompt_tokens_details") or {}
                    rec["cached_tokens"] = details.get("cached_tokens")
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    c, r = d.get("content"), d.get("reasoning_content")
                    if (c or r) and rec["ttft"] is None:
                        rec["ttft"] = time.time() - t0
                    if c:
                        text.append(c)
            rec["text"] = "".join(text)
            if rec["ttft"] is None:
                rec["error"] = "stream ended without tokens"
    except Exception as e:  # network error, timeout, reset
        rec["error"] = repr(e)[:200]
    finally:
        conn.close()
    rec["total"] = time.time() - t0
    return rec


class Run:
    def __init__(self, args):
        self.a = args
        u = urllib.parse.urlparse(args.url)
        self.host, self.port = u.hostname, u.port or 80
        self.key = os.environ.get("LITELLM_API_KEY") or os.environ.get("LITELLM_KEY_LOADGEN") or os.environ.get("LITELLM_MASTER_KEY", "")   # its own virtual key if there is one
        if not self.key:
            sys.exit("no key: set LITELLM_KEY_LOADGEN (python3 script/make_keys.py), LITELLM_API_KEY or LITELLM_MASTER_KEY")
        os.makedirs(args.out, exist_ok=True)
        self.rec = Recorder(os.path.join(args.out, "records.jsonl"))
        self.text = Text(args.tok_per_word)
        self.t0 = time.time()
        self.stop_at = self.t0 + args.duration
        self.counters = {"compactions": 0, "turns": 0}
        self.clock = threading.Lock()
        # one system prompt (+tools) shared by every session: the prefix the gateway tracks
        self.system = self.text.make(random.Random(args.seed), args.system_tokens)
        self.system = "You are a coding agent. Tools: read, edit, bash, grep.\n" + self.system

    def sleep(self, seconds):
        end = min(self.stop_at, time.time() + seconds)
        while time.time() < end:
            time.sleep(min(0.5, end - time.time()))

    def session(self, sid):
        a = self.a
        rnd = random.Random(f"{a.seed}-{sid}")
        prof = PROFILES[a.profile]
        is_batch = rnd.random() < a.batch_share
        thinking = rnd.random() < a.thinking_share
        cls = "batch" if is_batch else "interactive"
        priority = 8 if is_batch else 5
        self.sleep(rnd.uniform(0, a.ramp))
        initial = rnd.uniform(*prof["initial"])
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": "Task and repository context:\n" + self.text.make(rnd, initial)}]
        ctx = a.system_tokens + initial
        turn = 0
        while time.time() < self.stop_at:
            turn += 1
            with self.clock:
                self.counters["turns"] += 1
            if turn > 1:
                messages.append({"role": "user", "content": "Next request:\n" + self.text.make(rnd, 150)})
                ctx += 150
            n_calls = int(round(lognorm(rnd, 4.5, 0.8, 1, 25)))
            for call in range(1, n_calls + 1):
                if time.time() >= self.stop_at:
                    return
                if thinking:
                    max_tokens = int(lognorm(rnd, 800, 0.8, 64, 4000))
                else:
                    max_tokens = int(lognorm(rnd, 247, 0.9, 16, 1000))
                if ctx + max_tokens > a.max_prompt:      # client-side compaction, as pi does
                    keep = rnd.uniform(8000, 15000)
                    messages = [messages[0], {"role": "user", "content": "Summary of the work so far:\n" + self.text.make(rnd, keep)}]
                    ctx = a.system_tokens + keep
                    with self.clock:
                        self.counters["compactions"] += 1
                body = {"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7,
                        "stream": True, "stream_options": {"include_usage": True}, "priority": priority}
                if not thinking:
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                if a.ignore_eos:
                    body["ignore_eos"] = True
                served = None
                for attempt in range(1, a.retries + 2):
                    t_start = time.time()
                    r = one_call(self.host, self.port, self.key, body, a.timeout)
                    row = {"t": round(t_start - self.t0, 3), "session": sid, "cls": cls, "thinking": thinking,
                           "turn": turn, "call": call, "attempt": attempt, "ctx_est": int(ctx), "max_tokens": max_tokens}
                    row.update({k: v for k, v in r.items() if k != "text"})
                    self.rec.add(row)
                    if r["status"] == 200 and not r["error"]:
                        served = r
                        break
                    if r["status"] in (503, 529) and attempt <= a.retries and time.time() < self.stop_at:
                        try:
                            wait = float(r["retry_after"] or 2)
                        except ValueError:
                            wait = 2.0
                        self.sleep(min(10.0, max(1.0, wait)) + rnd.uniform(0, 1))
                        continue
                    break
                if served is None:
                    self.sleep(rnd.uniform(1, 3))
                    continue
                reply = served["text"] or self.text.make(rnd, min(served["completion_tokens"] or 100, 300))
                messages.append({"role": "assistant", "content": reply})
                tool = lognorm(rnd, prof["tool_median"], 0.7, 500, 8000)
                messages.append({"role": "user", "content": "Tool output:\n" + self.text.make(rnd, tool)})
                ctx = (served["prompt_tokens"] or ctx) + (served["completion_tokens"] or 0) + tool
                self.sleep(lognorm(rnd, a.tool_time, 0.8, 0.2, 15))
            self.sleep(lognorm(rnd, a.turn_idle, 0.8, 2, 120))

    def progress(self):
        last = 0
        while time.time() < self.stop_at:
            time.sleep(30)
            rows = [r for r in self.rec.snapshot() if r["t"] >= last]
            last = time.time() - self.t0
            ok = [r for r in rows if r["status"] == 200 and r["ttft"] is not None]
            shed = sum(1 for r in rows if r["status"] in (503, 529))
            p99 = pct([r["ttft"] for r in ok], 99)
            print(f"[{last:6.0f}s] last30s attempts={len(rows)} ok={len(ok)} shed={shed} "
                  f"ttft_p99={'-' if p99 is None else round(p99, 1)}", file=sys.stderr, flush=True)

    def go(self):
        a = self.a
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump({**vars(a), "start_epoch": self.t0}, f, indent=1)
        threads = [threading.Thread(target=self.session, args=(i,), daemon=True) for i in range(a.sessions)]
        for t in threads:
            t.start()
        threading.Thread(target=self.progress, daemon=True).start()
        for t in threads:
            t.join(timeout=max(1, self.stop_at + a.timeout - time.time()))
        return self.summarize()

    def summarize(self):
        a = self.a
        a_warm, a_dur = a.warmup, a.duration
        rows = [r for r in self.rec.snapshot() if a.warmup <= r["t"] < a.duration]
        window = max(1.0, a.duration - a.warmup)
        out = {"sessions": a.sessions, "profile": a.profile, "window_s": window, "classes": {},
               "compactions": self.counters["compactions"], "turns_started": self.counters["turns"],
               "start_epoch": self.t0, "warmup_s": a.warmup, "duration_s": a.duration}
        for cls in ("interactive", "batch", "all"):
            rs = [r for r in rows if cls == "all" or r["cls"] == cls]
            if not rs:
                continue
            ok = [r for r in rs if r["status"] == 200 and not r["error"] and r["ttft"] is not None]
            shed = [r for r in rs if r["status"] in (503, 529)]
            ok_ids = {id(r) for r in ok}
            shed_ids = {id(r) for r in shed}
            other = [r for r in rs if id(r) not in ok_ids and id(r) not in shed_ids]
            reasons = {}
            for r in shed:
                k = r.get("shed_reason") or "unknown"
                reasons[k] = reasons.get(k, 0) + 1
            ttft = [r["ttft"] for r in ok]
            tot = [r["total"] for r in ok]
            calls = {}
            for r in rs:
                key = (r["session"], r["turn"], r["call"])
                calls[key] = calls.get(key, False) or (id(r) in ok_ids)
            pin = sum(r["prompt_tokens"] or 0 for r in ok)
            cached = sum(r["cached_tokens"] or 0 for r in ok if r["cached_tokens"] is not None)
            pout = sum(r["completion_tokens"] or 0 for r in ok)
            # Do consecutive calls of one session land on the same worker? Cached history only helps if they do.
            last, same, seen = {}, 0, 0
            for r in sorted(ok, key=lambda r: r["t"]):
                if r.get("worker") is None:
                    continue
                if r["session"] in last:
                    seen += 1
                    same += last[r["session"]] == r["worker"]
                last[r["session"]] = r["worker"]
            by_pos = {}
            for name in ("session_start", "turn_start", "within_turn"):
                sub = [r for r in ok if position(r) == name]
                pt = sum(r["prompt_tokens"] or 0 for r in sub)
                ct = sum(r["cached_tokens"] or 0 for r in sub)
                by_pos[name] = {"n": len(sub), "ttft_p50": pct([r["ttft"] for r in sub], 50),
                                "ttft_p99": pct([r["ttft"] for r in sub], 99),
                                "cached_share": round(ct / pt, 3) if pt else None}
            mid = a_warm + (a_dur - a_warm) / 2
            first = [r["ttft"] for r in ok if r["t"] < mid]
            second = [r["ttft"] for r in ok if r["t"] >= mid]
            out["classes"][cls] = {
                "worker_stickiness": round(same / seen, 3) if seen else None,
                "by_position": by_pos,
                "ttft_p50_first_half": pct(first, 50), "ttft_p50_second_half": pct(second, 50),
                "attempts": len(rs), "served": len(ok), "shed": len(shed), "shed_by_reason": reasons,
                "other_errors": len(other), "other_error_samples": sorted({(r["error"] or str(r["status"]))[:80] for r in other})[:3],
                "shed_rate": round(len(shed) / len(rs), 4),
                # how long the user waits to be told no: an immediate 503 and a 503 after 20 s are very different
                "shed_wait_p50": pct([r["total"] for r in shed], 50), "shed_wait_p95": pct([r["total"] for r in shed], 95),
                "shed_late_over_5s": sum(1 for r in shed if r["total"] > 5),
                "calls": len(calls), "calls_abandoned": sum(1 for v in calls.values() if not v),
                "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95), "ttft_p99": pct(ttft, 99),
                "ttft_max": max(ttft) if ttft else None,
                "ttft_within_slo": round(sum(1 for t in ttft if t <= a.slo_ttft) / len(ttft), 4) if ttft else None,
                "total_p50": pct(tot, 50), "total_p99": pct(tot, 99),
                "prompt_tokens": pin, "cached_tokens_reported": cached, "completion_tokens": pout,
                "served_per_min": round(len(ok) / window * 60, 2),
                "prompt_tok_per_s": round(pin / window), "completion_tok_per_s": round(pout / window, 1),
            }
        inter = out["classes"].get("interactive")
        if inter:
            out["verdict"] = {
                "slo_ttft_s": a.slo_ttft, "max_shed": a.max_shed,
                "ttft_p99_ok": inter["ttft_p99"] is not None and inter["ttft_p99"] <= a.slo_ttft,
                "shed_ok": inter["shed_rate"] <= a.max_shed,
            }
            f, sec = inter["ttft_p50_first_half"], inter["ttft_p50_second_half"]
            # a queue that keeps growing shows up as a median that keeps rising: the run did not reach a steady state
            out["verdict"]["steady_state"] = None if not f or not sec else sec <= 1.5 * f
            out["verdict"]["pass"] = out["verdict"]["ttft_p99_ok"] and out["verdict"]["shed_ok"]
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(out, f, indent=1)
        self.timeline()
        return out

    def timeline(self):
        """One row per minute over the whole run (warmup included): the shape of the run, for the plots."""
        buckets = {}
        for r in self.rec.snapshot():
            buckets.setdefault(int(r["t"] // 60), []).append(r)
        with open(os.path.join(self.a.out, "timeline.csv"), "w") as f:
            f.write("minute,attempts,served,shed,ttft_p50,ttft_p99\n")
            for m in sorted(buckets):
                rs = buckets[m]
                ok = [r["ttft"] for r in rs if r["status"] == 200 and not r["error"] and r["ttft"] is not None]
                shed = sum(1 for r in rs if r["status"] in (503, 529))
                f.write(",".join(str(x) for x in (m, len(rs), len(ok), shed,
                                                  "" if not ok else round(pct(ok, 50), 3),
                                                  "" if not ok else round(pct(ok, 99), 3))) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("LOADGEN_URL", "http://127.0.0.1:4000"))
    p.add_argument("--model", default="qwen-coding-local")
    p.add_argument("--sessions", type=int, required=True)
    p.add_argument("--duration", type=float, default=480, help="total seconds including warmup")
    p.add_argument("--warmup", type=float, default=120, help="seconds excluded from the statistics")
    p.add_argument("--ramp", type=float, default=60, help="sessions start spread over this many seconds")
    p.add_argument("--profile", choices=sorted(PROFILES), default="paper")
    p.add_argument("--batch-share", type=float, default=0.0, help="fraction of sessions with priority 8")
    p.add_argument("--thinking-share", type=float, default=0.0, help="fraction of sessions that leave thinking on")
    p.add_argument("--system-tokens", type=int, default=10000)
    p.add_argument("--max-prompt", type=int, default=60000)
    p.add_argument("--tool-time", type=float, default=1.5, help="median tool seconds between calls")
    p.add_argument("--turn-idle", type=float, default=20, help="median user idle seconds between turns (compressed)")
    p.add_argument("--retries", type=int, default=2, help="re-sends after a 503/529 shed")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--tok-per-word", type=float, default=3.9)
    p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    p.add_argument("--slo-ttft", type=float, default=20.0)
    p.add_argument("--max-shed", type=float, default=0.01)
    p.add_argument("--fleet-capacity", type=int, default=None, help=argparse.SUPPRESS)   # deprecated, ignored: the cap lives in the gateway config
    p.add_argument("--gateway-queue-max", type=int, default=None, help=argparse.SUPPRESS)   # deprecated: the gateway holds nothing; ignored
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    s = Run(a).go()
    print(json.dumps({k: s[k] for k in ("sessions", "profile", "window_s", "compactions")}), file=sys.stderr)
    for cls, v in s["classes"].items():
        print(f"{cls:11s} attempts={v['attempts']} served={v['served']} shed={v['shed']} {v['shed_by_reason']} "
              f"abandoned={v['calls_abandoned']}/{v['calls']} ttft p50/p95/p99="
              f"{v['ttft_p50'] and round(v['ttft_p50'], 2)}/{v['ttft_p95'] and round(v['ttft_p95'], 2)}/"
              f"{v['ttft_p99'] and round(v['ttft_p99'], 2)}s in-slo={v['ttft_within_slo']}")
    print("verdict:", s.get("verdict"))
    sys.exit(0 if s.get("verdict", {}).get("pass") else 2)


if __name__ == "__main__":
    main()

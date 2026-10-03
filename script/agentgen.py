#!/usr/bin/env python3
"""Agentic load generator: sessions that really use tools, driven by what the model really answers.

`loadgen.py` is a synthetic generator: its prompts are random words, it sends no `tools` field, it forces the output length and
scripts every next call. That reproduces the SHAPE of an agent session (shared prefix, growing history, pauses) but not its content
or its behaviour. This generator closes the gap: every session is a real agent loop against the real model.

  system prompt   instructions + the repository's CLAUDE.md as project context (a real ~5K-token shared prefix) + 3 tool schemas
  tools           read_file, list_dir, grep, executed here on this repository (read-only, confined to the repo root)
  turn            a user task ("where is the placement policy implemented?"); the model decides which tools to call, the tool
                  results go back as `tool` messages, and the turn ends when the model answers without a tool call
                  (or after --max-steps calls)
  history         grows with what the model asked for and what the tools returned (reading ARCHITECTURE.md adds ~5K tokens per call),
                  and is compacted client-side near --max-prompt, as pi does
  outputs         natural: nothing is forced (no ignore_eos); thinking is off unless the client asks (pi default)

It writes the same files as loadgen.py (records.jsonl, summary.json, timeline.csv, config.json), so analyze_runs.py, prom_snapshot.py
and cost_report.py work on its runs, plus an `agent` block in summary.json (steps per turn, tool calls by name, malformed calls).
Run it where the repository is (the server: ~/final_project):

    LITELLM_KEY_LOADGEN=... python3 script/agentgen.py --url http://10.43.0.1:4000 --sessions 12 --duration 360 --warmup 60 --out metrics/runs/x/N12

Standard library only.
"""
import argparse
import http.client
import json
import math
import os
import random
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loadgen  # noqa: E402  (shares the recorder, the summary and the shed classification with the synthetic generator)

ROOT = os.path.dirname(HERE)
SKIP_DIRS = {".git", "metrics", "plots", "__pycache__", ".sessions", "node_modules", "todo-app"}
TEXT_SUFFIXES = (".md", ".py", ".yaml", ".yml", ".sh", ".txt", ".json", ".ipynb")
READ_CHARS = 20000        # one tool result is capped here (~5K tokens), as agents do
GREP_MAX = 40

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file of the repository. Returns numbered lines. Use offset/limit to page through long files.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "path relative to the repository root"},
            "offset": {"type": "integer", "description": "first line to read, 1-based (default 1)"},
            "limit": {"type": "integer", "description": "number of lines to read (default 300)"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List the entries of a directory of the repository (directories end with /).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "directory, default '.'"}}}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search a regular expression in the text files under a path. Returns 'file:line: text' for up to 40 matches.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string", "description": "file or directory, default '.'"}}, "required": ["pattern"]}}},
]

TASKS = [
    "Where is the worker placement policy implemented and how does it decide which worker serves a request? Read the code and explain it in five bullets.",
    "Explain how admission decides to refuse a request: list every refusal reason, its threshold and the file where it lives.",
    "Find every place where the output token cap (max_tokens) is set or enforced and list file and line for each.",
    "Read the part of notes/findings.md about the routing experiment and tell me which policy won and why.",
    "How does the batch share work in admission? Which setting controls it and what does a refused batch request receive?",
    "What does the synthetic load generator send to the gateway? Read script/loadgen.py and describe the shape of its requests.",
    "List the Kubernetes manifests under cluster/ and say in one line what each one deploys.",
    "Read the decision log in ARCHITECTURE.md (rows 40 to 50) and summarize the three decisions that matter most for performance.",
    "Explain how the warm-up gate works: what script/warm_workers.py measures and when launch_cluster.sh uses it.",
    "Where is the hierarchical cache configured for the SGLang workers? Name the flags and what each one does.",
    "Summarize the guard in control/inspect.py: what it refuses and with which status codes.",
    "Read the unit tests of the placement policy and list the behaviours they pin down.",
]
FOLLOWUPS = [
    "Now explain that to a new teammate in one short paragraph.",
    "Which of those points is the riskiest, in your opinion? Check the code before answering.",
    "Is there a test that covers what you just described? Find it.",
]


def _confine(path):
    full = os.path.realpath(os.path.join(ROOT, path or "."))
    if full != ROOT and not full.startswith(ROOT + os.sep):
        raise ValueError("path outside the repository")
    parts = os.path.relpath(full, ROOT).split(os.sep)
    if any(p in SKIP_DIRS for p in parts):
        raise ValueError("path not available")
    return full


def run_tool(name, args):
    """Executes one tool call; always returns text (errors are returned to the model as text, like a real agent)."""
    try:
        if name == "read_file":
            full = _confine(args["path"])
            offset = max(1, int(args.get("offset") or 1))
            limit = max(1, int(args.get("limit") or 300))
            with open(full, errors="replace") as f:
                lines = f.read().splitlines()
            chunk = lines[offset - 1: offset - 1 + limit]
            text = "\n".join(f"{offset + i:5d}  {l}" for i, l in enumerate(chunk))
            if len(text) > READ_CHARS:
                text = text[:READ_CHARS] + f"\n[truncated: {len(lines)} lines in the file, use offset/limit]"
            return text or "[empty]"
        if name == "list_dir":
            full = _confine(args.get("path") or ".")
            return "\n".join(sorted(e + ("/" if os.path.isdir(os.path.join(full, e)) else "")
                                    for e in os.listdir(full) if e not in SKIP_DIRS)) or "[empty]"
        if name == "grep":
            rx = re.compile(args["pattern"])
            base = _confine(args.get("path") or ".")
            files = [base] if os.path.isfile(base) else [os.path.join(d, f) for d, ds, fs in os.walk(base)
                                                          if not set(os.path.relpath(d, ROOT).split(os.sep)) & SKIP_DIRS
                                                          for f in fs if f.endswith(TEXT_SUFFIXES)]
            hits = []
            for fp in sorted(files):
                with open(fp, errors="replace") as f:
                    for n, line in enumerate(f, 1):
                        if rx.search(line):
                            hits.append(f"{os.path.relpath(fp, ROOT)}:{n}: {line.strip()[:200]}")
                            if len(hits) >= GREP_MAX:
                                return "\n".join(hits) + "\n[more matches not shown]"
            return "\n".join(hits) or "no matches"
        return f"unknown tool {name}"
    except Exception as e:  # a bad path, a bad regex, a bad argument: the model sees the error and may retry
        return f"error: {e}"


def agent_call(host, port, key, body, timeout):
    """One streaming chat completion; returns the record plus the assistant text and the tool calls the model made."""
    t0 = time.time()
    rec = {"worker": None, "status": None, "ttfb": None, "ttft": None, "total": None, "prompt_tokens": None,
           "completion_tokens": None, "cached_tokens": None, "shed_reason": None, "retry_after": None, "error": None,
           "text": "", "tool_calls": [], "finish": None}
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    calls = {}
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        resp = conn.getresponse()
        rec["status"] = resp.status
        rec["ttfb"] = time.time() - t0
        rec["worker"] = resp.getheader("x-litellm-model-api-base")
        if resp.status != 200:
            rec["error"] = resp.read(2000).decode("utf8", "replace")[:300]
            rec["shed_reason"] = loadgen.classify_shed(resp.status, rec["error"], resp.getheader("shed-reason"))
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
                    rec["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if ch.get("finish_reason"):
                        rec["finish"] = ch["finish_reason"]
                    c, r, tc = d.get("content"), d.get("reasoning_content"), d.get("tool_calls")
                    if (c or r or tc) and rec["ttft"] is None:
                        rec["ttft"] = time.time() - t0
                    if c:
                        text.append(c)
                    for t in tc or []:
                        cur = calls.setdefault(t.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        cur["id"] = t.get("id") or cur["id"]
                        fn = t.get("function") or {}
                        cur["name"] += fn.get("name") or ""
                        cur["arguments"] += fn.get("arguments") or ""
            rec["text"] = "".join(text)
            rec["tool_calls"] = [calls[i] for i in sorted(calls)]
            if rec["ttft"] is None:
                rec["error"] = "stream ended without tokens"
    except Exception as e:
        rec["error"] = repr(e)[:200]
    finally:
        conn.close()
    rec["total"] = time.time() - t0
    return rec


class AgentRun(loadgen.Run):
    def __init__(self, args):
        super().__init__(args)
        with open(os.path.join(ROOT, "CLAUDE.md"), errors="replace") as f:
            project = f.read()
        self.system = ("You are a coding agent working in a software repository. Use the tools to inspect files before you "
                       "answer, do not guess, and keep your answers short and concrete. Always cite file paths.\n\n"
                       "Project instructions (from CLAUDE.md):\n" + project)
        self.agent = {"turns_done": 0, "turns_cut": 0, "steps": [], "tool_calls": {}, "bad_calls": 0, "tool_chars": [], "compactions": 0}

    def _count(self, key, n=1):
        with self.clock:
            self.agent[key] += n

    def session(self, sid):
        a = self.a
        rnd = random.Random(f"{a.seed}-{sid}")
        is_batch = rnd.random() < a.batch_share
        cls, priority = ("batch", 8) if is_batch else ("interactive", 5)
        self.sleep(rnd.uniform(0, a.ramp))
        messages = [{"role": "system", "content": self.system}]
        ctx = len(self.system) // 4
        turn = 0
        while time.time() < self.stop_at:
            turn += 1
            task = rnd.choice(TASKS) if turn == 1 or rnd.random() < 0.5 else rnd.choice(FOLLOWUPS)
            messages.append({"role": "user", "content": task})
            ctx += len(task) // 4
            finished = False
            for step in range(1, a.max_steps + 1):
                if time.time() >= self.stop_at:
                    return
                if ctx + a.max_tokens > a.max_prompt:       # client-side compaction, as pi does
                    last = next((m["content"] for m in reversed(messages) if m["role"] == "assistant" and m.get("content")), "")
                    messages = [messages[0], {"role": "user", "content": "Summary of the work so far:\n" + (last or "(nothing yet)")[:6000]},
                                {"role": "user", "content": task}]
                    ctx = len(messages[0]["content"]) // 4 + 2000
                    self._count("compactions")
                body = {"model": a.model, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "max_tokens": a.max_tokens,
                        "temperature": 0.7, "stream": True, "stream_options": {"include_usage": True}, "priority": priority,
                        "chat_template_kwargs": {"enable_thinking": False}}
                served = None
                for attempt in range(1, a.retries + 2):
                    t_start = time.time()
                    r = agent_call(self.host, self.port, self.key, body, a.timeout)
                    row = {"t": round(t_start - self.t0, 3), "session": sid, "cls": cls, "thinking": False, "turn": turn,
                           "call": step, "attempt": attempt, "ctx_est": int(ctx), "max_tokens": a.max_tokens,
                           "tools": [c["name"] for c in r["tool_calls"]]}
                    row.update({k: v for k, v in r.items() if k not in ("text", "tool_calls")})
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
                    break                                  # the turn is lost; the session moves on after its pause
                msg = {"role": "assistant", "content": served["text"] or ""}
                if served["tool_calls"]:
                    msg["tool_calls"] = [{"id": c["id"] or f"call_{sid}_{turn}_{step}_{i}", "type": "function",
                                          "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                                         for i, c in enumerate(served["tool_calls"])]
                messages.append(msg)
                ctx = (served["prompt_tokens"] or ctx) + (served["completion_tokens"] or 0)
                if not served["tool_calls"]:
                    finished = True
                    self._count("turns_done")
                    with self.clock:
                        self.agent["steps"].append(step)
                    break
                for c, tc in zip(served["tool_calls"], msg["tool_calls"]):
                    try:
                        args = json.loads(c["arguments"] or "{}")
                        result = run_tool(c["name"], args)
                    except ValueError:
                        self._count("bad_calls")
                        result = "error: the arguments were not valid JSON"
                    with self.clock:
                        self.agent["tool_calls"][c["name"]] = self.agent["tool_calls"].get(c["name"], 0) + 1
                        self.agent["tool_chars"].append(len(result))
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    ctx += len(result) // 4
                self.sleep(loadgen.lognorm(rnd, a.tool_time, 0.6, 0.1, 5))
            if not finished:
                self._count("turns_cut")
            self.sleep(loadgen.lognorm(rnd, a.turn_idle, 0.8, 2, 120))

    def summarize(self):
        out = super().summarize()
        steps, chars = self.agent["steps"], self.agent["tool_chars"]
        out["agent"] = {
            "turns_finished": self.agent["turns_done"], "turns_cut_or_lost": self.agent["turns_cut"],
            "steps_per_finished_turn_p50": loadgen.pct(steps, 50), "steps_per_finished_turn_max": max(steps) if steps else None,
            "tool_calls_by_name": self.agent["tool_calls"], "malformed_tool_calls": self.agent["bad_calls"],
            "tool_result_tokens_p50": round(loadgen.pct(chars, 50) / 4) if chars else None,
            "tool_result_tokens_p95": round(loadgen.pct(chars, 95) / 4) if chars else None,
            "compactions": self.agent["compactions"],
        }
        with open(os.path.join(self.a.out, "summary.json"), "w") as f:
            json.dump(out, f, indent=1)
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("LOADGEN_URL", "http://127.0.0.1:4000"))
    p.add_argument("--model", default="qwen-coding-local")
    p.add_argument("--sessions", type=int, required=True)
    p.add_argument("--duration", type=float, default=360, help="total seconds including warmup")
    p.add_argument("--warmup", type=float, default=60, help="seconds excluded from the statistics")
    p.add_argument("--ramp", type=float, default=30, help="sessions start spread over this many seconds")
    p.add_argument("--batch-share", type=float, default=0.0, help="fraction of sessions with priority 8")
    p.add_argument("--max-steps", type=int, default=12, help="tool calls allowed per user turn")
    p.add_argument("--max-tokens", type=int, default=4096, help="output cap asked for (the gateway cap)")
    p.add_argument("--max-prompt", type=int, default=60000)
    p.add_argument("--tool-time", type=float, default=0.6, help="median seconds between a tool result and the next call")
    p.add_argument("--turn-idle", type=float, default=20, help="median user idle seconds between turns (compressed)")
    p.add_argument("--retries", type=int, default=2, help="re-sends after a 503/529 shed")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--slo-ttft", type=float, default=20.0)
    p.add_argument("--max-shed", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.profile = "agent"
    a.tok_per_word = 3.9
    a.system_tokens = 100          # the synthetic system prompt of the base class is replaced
    s = AgentRun(a).go()
    for cls, v in s["classes"].items():
        print(f"{cls:11s} attempts={v['attempts']} served={v['served']} shed={v['shed']} {v['shed_by_reason']} "
              f"abandoned={v['calls_abandoned']}/{v['calls']} ttft p50/p95/p99="
              f"{v['ttft_p50'] and round(v['ttft_p50'], 2)}/{v['ttft_p95'] and round(v['ttft_p95'], 2)}/"
              f"{v['ttft_p99'] and round(v['ttft_p99'], 2)}s in-slo={v['ttft_within_slo']}")
    print("agent:", json.dumps(s["agent"]))
    print("verdict:", s.get("verdict"))
    sys.exit(0 if s.get("verdict", {}).get("pass") else 2)


if __name__ == "__main__":
    main()

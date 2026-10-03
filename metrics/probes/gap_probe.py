#!/usr/bin/env python3
"""Why does a streamed tool call that writes a long file go silent? (found 2026-10-03, notes/findings.md "The demo timeout")

Asks the model, through SGLang's streaming chat endpoint, to write a ~120-line file with a `write_file` tool and records when each
chunk arrives. With `--tool-call-parser qwen3_coder` SGLang sends the first chunks at once and then NOTHING until the whole
`content` argument has been generated (20.6 s for 1,121 tokens at ~51 tok/s), and then the argument in one chunk. Any socket read
timeout shorter than that gap (LiteLLM's `stream_timeout`, which `TTFT_CUT_S` sets) kills the stream.
Run on the server, direct to a worker (diagnosis only; clients must go through LiteLLM):  python3 metrics/probes/gap_probe.py 30001
"""
import http.client, json, time, sys
port = int(sys.argv[1]) if len(sys.argv) > 1 else 30001
body = {"model": "Qwen/Qwen3.5-9B", "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 4096, "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [{"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file",
                   "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}],
        "messages": [{"role": "user", "content": "Create app.js for a todo list web app (add, toggle, edit, delete, filters, localStorage) with at least 120 lines of readable code. Use the write_file tool."}]}
c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
t0 = time.time()
c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
r = c.getresponse()
events, last = [], t0
kinds = {}
usage = None
while True:
    line = r.readline()
    if not line:
        break
    line = line.strip()
    if not line.startswith(b"data:"):
        continue
    p = line[5:].strip()
    if p == b"[DONE]":
        break
    ev = json.loads(p)
    now = time.time()
    if ev.get("usage"):
        usage = ev["usage"]
    for ch in ev.get("choices") or []:
        d = ch.get("delta") or {}
        k = "tool_calls" if d.get("tool_calls") else ("content" if d.get("content") else ("reasoning" if d.get("reasoning_content") else "other"))
        kinds[k] = kinds.get(k, 0) + 1
        events.append((round(now - t0, 2), round(now - last, 2), k))
        last = now
print("port", port, "total s", round(time.time() - t0, 1), "chunks", len(events), kinds, "completion_tokens", usage and usage.get("completion_tokens"))
if events:
    print("first chunk at", events[0][0], "s; largest gap between chunks", max(e[1] for e in events), "s")
    big = [e for e in events if e[1] > 5]
    print("gaps > 5 s:", big[:6])
    print("first 5 chunks:", events[:5], "last 3:", events[-3:])

#!/usr/bin/env python3
"""Live check of custom placement: do the calls of one session stay on one worker, and are sessions spread over both?

Run inside the litellm pod next to script/loadgen.py, with the repo default policy (`affload`, or `script/router_variant.sh affload`).
Eight sessions (eight different first messages) make four short calls each; the worker that served each call is read from
LiteLLM's `x-litellm-model-api-base` header.
"""
import os
import sys

sys.path.insert(0, "/tmp")
import loadgen  # noqa: E402

KEY = os.environ["LITELLM_MASTER_KEY"]
seen = {}
for call in range(4):
    for s in range(8):
        msgs = [{"role": "user", "content": f"Session {s}: summarise item {s * 7}."}]
        for k in range(call):
            msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": f"and item {k}?"}]
        b = {"model": "qwen-coding-local", "max_tokens": 8, "stream": True, "stream_options": {"include_usage": True},
             "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs}
        r = loadgen.one_call("127.0.0.1", 4000, KEY, b, 60)
        w = (r["worker"] or "?").split("//")[-1].split(":")[0]
        seen.setdefault(s, []).append((r["status"], w))
for s, v in seen.items():
    print(f"session {s}: statuses={sorted({x[0] for x in v})} workers={[x[1][-1] for x in v]}")
workers = {x[1] for v in seen.values() for x in v}
sticky = all(len({x[1] for x in v}) == 1 for v in seen.values())
print(f"sessions spread over {sorted(workers)}; every session stayed on one worker: {sticky}")

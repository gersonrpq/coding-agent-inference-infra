#!/usr/bin/env python3
"""Does LiteLLM close the upstream connection (so SGLang aborts the request) when ...

  s1  a streaming client disconnects while its request is RUNNING
  s2  a non-streaming client disconnects while its request is RUNNING
  s3  a client disconnects while its request is QUEUED in SGLang
  s4  LiteLLM itself cuts a QUEUED request because its first token took longer than stream_timeout (TTFT_CUT_S)

The status code is not the evidence. The evidence is what SGLang does afterwards: its running and queued counts
and the number of tokens it keeps GENERATING. Requests use `ignore_eos`, so an aborted request is recognisable
(far fewer tokens than max_tokens) and a "zombie" one (served to nobody) is too (all of them).

Run inside the litellm pod with script/loadgen.py next to it (same recipe as queue_mechanism.py); needs the
variant `script/queue_variant.sh gate 3 4 ""` (engine queue 3 per worker, 4 s first-token cut):

    kubectl -n gpu-serving exec $POD -- python3 /tmp/connection_close.py s1
"""
import http.client
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/tmp")
import loadgen  # noqa: E402

KEY = os.environ["LITELLM_MASTER_KEY"]
WORKERS = ["sglang-worker-0:30000", "sglang-worker-1:30001"]
TOKENS = 800                    # ~17 s of decode at 48 tok/s: long enough to see an abort


def metrics():
    run = queued = gen = 0.0
    for w in WORKERS:
        txt = urllib.request.urlopen(f"http://{w}/metrics", timeout=10).read().decode()
        def val(name, label=True):
            pat = r'sglang:%s(?:_total)?\{[^}]*%s[^}]*\} ([0-9.e+]+)' % (name, 'priority=""' if label else "")
            m = re.search(pat, txt)
            return float(m.group(1)) if m else 0.0
        run += val("num_running_reqs")
        queued += val("num_queue_reqs")
        m = re.search(r'sglang:generation_tokens_total\{[^}]*\} ([0-9.e+]+)', txt)
        gen += float(m.group(1)) if m else 0.0
    return run, queued, gen


def sampler(stop, rows, t0):
    while not stop.is_set():
        rows.append((round(time.time() - t0, 1),) + metrics())
        time.sleep(0.5)


def body(stream=True, priority=5, max_tokens=TOKENS):
    return {"model": "qwen-coding-local", "max_tokens": max_tokens, "temperature": 0.7, "stream": stream,
            "priority": priority, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Write a long essay about queues. " + "alpha " * 40}]}


def open_request(stream, **kw):
    c = http.client.HTTPConnection("127.0.0.1", 4000, timeout=120)
    c.request("POST", "/v1/chat/completions", body=json.dumps(body(stream, **kw)),
              headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    return c


def disconnect(c):
    try:
        c.sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    c.close()


def observe(label, action, seconds=40):
    """Run `action(t0)` while sampling SGLang; report running/queued over time and the tokens generated."""
    rows, stop, t0 = [], threading.Event(), time.time()
    base = metrics()
    th = threading.Thread(target=sampler, args=(stop, rows, t0))
    th.start()
    events = action(t0)
    time.sleep(seconds)
    stop.set()
    th.join()
    gen = rows[-1][3] - base[2]
    print(f"\n== {label}")
    print("   events:", events)
    print("   t(s)  running queued tokens_generated_since_start")
    last = None
    for t, r, q, g in rows:
        state = (r, q)
        if state != last:                      # print only the changes
            print(f"   {t:5.1f} {r:7.0f} {q:6.0f} {g - base[2]:8.0f}")
            last = state
    print(f"   total tokens generated: {gen:.0f}")
    return gen


def s1(stream=True):
    def act(t0):
        c = open_request(stream)
        if stream:
            c.getresponse().readline()                    # first bytes: it is running
        time.sleep(3.0)
        disconnect(c)
        return {"client_disconnected_at_s": round(time.time() - t0, 1)}
    gen = observe(f"{'s1' if stream else 's2'}: {'streaming' if stream else 'non-streaming'} client leaves at 3 s while running (max_tokens={TOKENS})", act)
    print(f"   VERDICT: {'upstream closed (generation stopped early)' if gen < TOKENS * 0.5 else 'ZOMBIE: SGLang kept generating for nobody'}")


def s3():
    def act(t0):
        batch = []
        for _ in range(14):                              # 12 running + 2 queued (k=3 per worker leaves room)
            c = open_request(True, priority=8, max_tokens=600)
            batch.append(c)
        time.sleep(2.5)
        extra = open_request(True, priority=5, max_tokens=200)   # waits in SGLang's queue
        time.sleep(1.0)
        before = metrics()
        disconnect(extra)
        time.sleep(0.2)
        for c in batch:
            threading.Thread(target=lambda c=c: c.getresponse().read(), daemon=True).start()
        return {"queued_before_leave": before[1], "left_at_s": round(time.time() - t0, 1)}
    observe("s3: client leaves while its request is QUEUED in SGLang", act, seconds=30)


def s4():
    """The cut request has a long, unique prompt (about 1,200 tokens) so SGLang's log shows whether it was EVER
    prefilled: every other request in the scenario has a prompt of roughly 100 tokens. The token counter is not
    used as evidence: SGLang adds to it when a request finishes, not while it runs."""
    marker = "zombie-marker " * 400

    def act(t0):
        batch = [open_request(True, priority=8, max_tokens=600) for _ in range(14)]
        time.sleep(2.5)
        t = time.time()
        b = dict(body(True, 5, 200), stream_options={"include_usage": True})
        b["messages"] = [{"role": "user", "content": marker}]
        print(f"   [cut request sent at {time.strftime('%H:%M:%S')}]")
        r = loadgen.one_call("127.0.0.1", 4000, KEY, b, 120)
        print(f"   [client answered at {time.strftime('%H:%M:%S')}]")
        for c in batch:
            threading.Thread(target=lambda c=c: c.getresponse().read(), daemon=True).start()
        return {"interactive_status": r["status"], "answered_after_s": round(time.time() - t, 1), "ttft": r["ttft"],
                "error": (r["error"] or "")[:110]}
    observe("s4: LiteLLM cuts a QUEUED request at TTFT_CUT_S. Look in the SGLang log for a prefill of ~1,200 tokens: "
            "if it appears after the cut, SGLang served the request to nobody", act, seconds=35)


if __name__ == "__main__":
    {"s1": lambda: s1(True), "s2": lambda: s1(False), "s3": s3, "s4": s4}[sys.argv[1]]()

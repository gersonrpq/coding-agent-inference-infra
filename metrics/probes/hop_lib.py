import json, random, re, time, urllib.request
W = {0: "http://localhost:30000", 1: "http://localhost:30001"}
WORDS = ("alpha beta gamma delta kernel buffer socket thread mutex lambda vector matrix parser token cursor "
         "module import export return yield async await class struct enum trait impl handler router queue cache "
         "index table column shard replica leader follower commit rebase branch merge deploy rollback metric trace").split()

def make_prompt(seed, target_tokens):
    rnd = random.Random(seed)
    n = int(target_tokens / 1.25)
    body = " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 999)) for _ in range(n))
    return "Repository notes follow.\n" + body + "\nReply with the single word ok."

def chat(w, prompt, max_tokens=1):
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(W[w] + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return dict(worker=w, seconds=round(dt, 2), prompt_tokens=u["prompt_tokens"], cached_tokens=cached)

def modes(w):
    txt = urllib.request.urlopen(W[w] + "/metrics", timeout=30).read().decode()
    out = {}
    for m in re.finditer(r'sglang:prefill_effective_tokens_total\{[^}]*mode="(\w+)"[^}]*\} ([0-9.e+]+)', txt):
        out[m.group(1)] = out.get(m.group(1), 0) + float(m.group(2))
    for key in ("hicache_host_used_tokens",):
        m = re.search(r'sglang:%s\{[^}]*\} ([0-9.e+]+)' % key, txt)
        if m: out[key] = float(m.group(1))
    return out

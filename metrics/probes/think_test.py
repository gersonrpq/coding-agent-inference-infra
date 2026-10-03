import json, os, time, urllib.request
def call(extra):
    body = {"model": "qwen-coding-local", "max_tokens": 200, "temperature": 0,
            "messages": [{"role": "user", "content": "What is 17*23? Answer briefly."}]}
    body.update(extra)
    req = urllib.request.Request("http://localhost:4000/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["LITELLM_MASTER_KEY"]})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=120))
    m = r["choices"][0]["message"]
    print(json.dumps(extra), "->", round(time.time()-t, 2), "s | usage:", r["usage"].get("completion_tokens"),
          "reasoning:", (r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"),
          "| reasoning_content chars:", len(m.get("reasoning_content") or ""), "| content:", (m.get("content") or "")[:60].replace("\n"," "))
call({})
call({"chat_template_kwargs": {"enable_thinking": False}})
call({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})

import sys; sys.path.insert(0, "/tmp")
from hop_lib import *
def bk(w):
    txt = urllib.request.urlopen(W[w] + "/metrics", timeout=30).read().decode()
    g = lambda pool: sum(float(x) for x in re.findall(r'hicache_backup_tokens_total\{[^}]*pool="%s"[^}]*\} ([0-9.e+]+)' % pool, txt))
    return {"kv_backed": g("kv"), "mamba_backed": g("mamba")}
S_ = make_prompt(501, 15000)
print("w0 backups before", bk(0), flush=True)
print("S -> w0 cold  :", chat(0, S_), flush=True)
time.sleep(12)
print("w0 backups after S", bk(0), flush=True)
t = time.time()
for i in range(80):                       # 80 distinct short prompts: more than the 46 recurrent states
    chat(0, make_prompt(600 + i, 1500))
print("80 short prompts done in %.0fs" % (time.time() - t), "backups", bk(0), flush=True)
time.sleep(15)
print("S -> w1 (cross):", chat(1, S_), flush=True)
time.sleep(3)
print("w1 modes", modes(1), flush=True)
print("S -> w0 (owner):", chat(0, S_), flush=True)
print("DONE", flush=True)

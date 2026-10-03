import sys; sys.path.insert(0, "/tmp")
from hop_lib import *
Q = make_prompt(101, 20000)          # already prefilled on worker 0 by probe1
print("w0 start", modes(0), flush=True)
for i in range(12):
    F = make_prompt(200 + i, 15000)
    r = chat(0, F)
    m = modes(0)
    print("filler", i, r, "host_used", m.get("hicache_host_used_tokens"), flush=True)
time.sleep(25)
print("w0 after fillers", modes(0), flush=True)
print("w1 before Q", modes(1), flush=True)
print("Q -> w1:", chat(1, Q), flush=True)
print("w1 after Q", modes(1), flush=True)
print("Q -> w0 (where does it hit?):", chat(0, Q), flush=True)
print("w0 final", modes(0), flush=True)
print("DONE", flush=True)

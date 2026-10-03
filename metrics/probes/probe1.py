import sys; sys.path.insert(0, "/tmp")
from hop_lib import *
print("w0 before", modes(0)); print("w1 before", modes(1))
Q = make_prompt(101, 20000)
print("Q -> w0 (cold):", chat(0, Q))
time.sleep(20)
print("w0 after", modes(0)); print("w1 after", modes(1))

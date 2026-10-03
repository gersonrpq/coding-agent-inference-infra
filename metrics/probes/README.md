# Probes

Scripts used to measure the cluster by hand (2026-10-02). They talk to the workers directly (`hop_lib.py`
uses `localhost:30000/30001`), so run them **inside a worker pod**:

```bash
P=$(kubectl -n gpu-serving get pod -l serving.gpu-worker=0 -o jsonpath='{.items[0].metadata.name}')
kubectl -n gpu-serving cp metrics/probes/hop_lib.py $P:/tmp/hop_lib.py
kubectl -n gpu-serving cp metrics/probes/probe5.py  $P:/tmp/probe5.py
kubectl -n gpu-serving exec $P -- python3 /tmp/probe5.py
```

- `probe1`-`probe3`: first cross-worker attempts with `write_back` (all missed; kept as evidence).
- `probe4`, `probe5`: with `write_through`; probe5 is the one that proved the Mooncake hop.
- `think_test.py`: thinking on/off through the LiteLLM gateway (run inside the `litellm` pod).
- Prompt sizes are tokens counted by the server; `make_prompt(seed, target)` overshoots by ~3x, so use `target = wanted / 3`.
Results: `notes/findings.md` and DESIGN.md "Probes".
- `connection_close.py` (s1-s4): does LiteLLM close the upstream connection when the client leaves or when its own timeout fires? s1 and s2 (a client that leaves while its request runs) can be repeated on the current system; s3 and s4 were run with an SGLang-side queue bound that no longer exists, so their evidence is the saved log (`metrics/logs/conn.log`, `conn4.log`; for s4 the SGLang log, not the token counter).
- `gap_probe.py`: why a long tool call goes silent (20.6 s for a 120-line file, 68 s for 300 lines) and why the first-token cut was removed (decisions 59, 61).
- `kv_probe.py`: 12 requests of ~60K tokens at once: how full the KV gets after admission (peak 83 %, no retraction) and, with `KV_SHED_THRESHOLD=0.70`, the `kv_pressure` shed path (new prefix refused, reused prefix admitted).
- `ramp_probe.py`: worker 1 restarted under load: nothing is sent to it while down and it comes back with a ramp (`orch_worker_available`, `orch_worker_ramp_factor`).
- `placement_check.py`: sessions stay on one worker under `affload`.
- `cap_probe.py`: the cap of 14 in flight (LiteLLM's middleware): the 15th request is refused at once. `tenant_probe.py` + `run_tenant_probe.sh`: per-key limits (needs `script/make_keys.py`).
- Removed (decisions 57 and 61) because they exercised mechanisms that no longer exist (the SGLang-side queue bound, `queue_variant.sh gate`, the first-token cut): `queue_mechanism.py` (m1-m3), `cut_mapping.py` and the `run_*.sh` drivers of the connection-close and mechanism probes. Their results are kept in `notes/findings.md` and `metrics/logs/`.


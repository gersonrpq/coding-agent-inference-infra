# Console logs of the experiment drivers

Raw output of the shell drivers that ran on the server (copied from `~/*.log`), kept because the mechanism probes
print their evidence to the console rather than to a run directory.

| File | Driver | What it holds |
| --- | --- | --- |
| `probes.log` | `metrics/probes/run_mechanism_probes.sh` | probes m1, m1b, m3 (and the failed m2) with the SGLang queue limit of 3 |
| `conn.log`, `conn4.log` | `run_connection_close.sh`, `run_connection_close_s4.sh` | scenarios s1-s4 and the marker-prompt rerun of s4 (does LiteLLM close the upstream call?) |
| `exp1.log` | live checks + `queue_experiment.sh eq-ref-n20/n28` | image support, reference runs `r0` |
| `exp2.log`, `exp3.log` | `queue_experiment.sh eq-k-n28` and `eq-k2-n24/n20` | the K sweeps (tables at the end of each) |
| `smoke.log` | `sweep.sh smoke` | the first load-generator smoke run |

The earlier hop probes (`probe1.py`-`probe5.py`) printed to the terminal; their results are transcribed in
`notes/findings.md` ("Hop with Mooncake") and `DESIGN.md` ("Probes").

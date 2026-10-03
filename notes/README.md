# notes/

The lab notebook and the protocols of the experiments. `ARCHITECTURE.md` is the deliverable and holds the decisions; this folder holds the **evidence and the reasoning behind them**.
Terms are defined once, in [`glossary.md`](glossary.md). Everything here is in English.

## How these files work

- **`findings.md`** is chronological. Every discovery is added with its date, evidence and consequence; entries are not rewritten. Start with its reading guide.
- **Protocols (`*-experiment*.md`, `final-load-tests.md`)** are **pre-registered**: the hypotheses and the decision rule are written before running and never edited. What happened goes in the execution log and in the dated "Later notes" at the end.
- Every number points to its raw data in `metrics/runs/<tag>/<variant>/` (`records.jsonl` is the raw per-call data) or `metrics/evidence/` and `metrics/probes/`.
- A result becomes a **decision** only when it has a row in the ARCHITECTURE decision log; the table below says which.

## Files

| File | What it is | State | Decisions it produced |
| --- | --- | --- | --- |
| [`glossary.md`](glossary.md) | Definitions: load, capacity, latency, cache, placement, method | canonical | |
| [`findings.md`](findings.md) | Lab notebook: hardware, memory, speed, the hop, mechanisms checked live, every experiment's reading | living | most of them |
| [`sweep-experiment.md`](sweep-experiment.md) | E1: how many sessions fit; defines regime A (dense) and B (idle) and the verdict | partly superseded | 28 |
| [`queue-experiment.md`](queue-experiment.md) | EQ: how long a queue still meets the SLO; mechanism probes; the K sweep | done | 30-34, 42-43 |
| [`routing-experiment.md`](routing-experiment.md) | ER: which worker serves a call (shuffle, least-busy, latency, affinity, affinity + load) | done at N = 24 and 20 | 40-41, 47, 50 |
| [`final-load-tests.md`](final-load-tests.md) | F1-F5: the final configuration under the knee, a replicate, batch, regime B and a soak; V1-V6: the checks after the simplification | done | 50, 55-58 |

## The story in six steps

1. **Size it** (`findings.md`: Hardware, Memory, Speed): one H100 as two MIG instances, a 9B model in bf16, 6 sequences of 64K per worker.
2. **Prove the hop** (Hop with Mooncake): the other worker reads a prefix from L3 (2.28 s against 4.48 s cold), once its recurrent state has been published.
3. **Choose the queue** (`queue-experiment.md`): a longer queue keeps the engine in its slow regime and lowers served calls per minute, so the gateway holds nothing and admits `12 + K` with K = 2; a first-token cut that was meant to bound the rest turned out to break long tool calls and was removed (decisions [59](../ARCHITECTURE.md#decision-59), [61](../ARCHITECTURE.md#decision-61)).
4. **Find the real limit** (Engine configuration review): decode per sequence falls from 48.8 to 15.4 tok/s as prefills interleave, so the cost of a call, not the slot count, limits the system.
5. **Choose the placement** (`routing-experiment.md`): `least-busy` keeps a session on its worker only half of the time; `affload` keeps it unless that worker is busier by 3 requests.
6. **Verify the finished system** (`final-load-tests.md`): the same configuration as the repo, under load, with the scrapes, the hop dictionary and the 429/503 counts in `metrics/evidence/`.

## Conventions

- Dates are 2026-10-02 unless stated. Sizes are in GiB (an "80 GB" H100 is 79.6 GiB).
- "Predicted" marks something not yet confirmed by a log or a scrape. A verdict is checked against the raw records, not only against a derived counter.
- Names that no longer exist in the code (the gateway queue, `r0`, `ADMISSION_MODE`, `q0`, `gate`) appear in old entries; decision [46](../ARCHITECTURE.md#decision-46) explains the removal.

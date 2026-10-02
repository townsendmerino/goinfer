# R21 — swap growth under scenario D: result (2026-10-01, nobara-pc, same day as the rule)

Rule: `PREREGISTERED.md`, committed before the run (`857551ec`). Runner `run.sh`; raw samples `samples.tsv` (every 2 s: epoch, swap used kB,
MemAvailable kB, Cached kB), arm boundaries `marks.txt`, the table below as computed `RESULT.txt`. Serve binary: this tree's `cmd/serve` at
`-tags cuda` (built from `cuda/`), the 26B q4_0 GGUF, `-backend cuda`, no `-moe-cache-experts`.

| window | s | swap used (GB) start → peak | Δ peak | Cached gain |
|---|---|---|---|---|
| idle (three of them) | 30 each | 13.20 → 13.20 (idle3: 13.27 → 13.27) | 0.000 | ~0 |
| B: `cat` a 16.8 GB GGUF (no goinfer) | 12 | 13.201 → 13.201 | **0.000** | +14.1 GB |
| A: scenario D (declines to CPU, loads, +30 s) | 74 | 13.201 → 13.406 | **+0.205** | −30.9 GB (the page cache was evicted to make room) |

Idle slope 0.40 MB/s at worst (0.030 GB over A's 74 s). dA − s = 0.175 GB.

**Outcome by the registered rule: AMBIGUOUS, parked.** dA − s is below the 0.5 GB bar for "goinfer's footprint", and dB is nowhere near 0.5 × dA for "the kernel's".
No change was made to the swap guard.

What the pass does and does not say:
- goinfer's load of the 26B on the CPU fallback grew swap by about 0.2 GB here, not the 1.9 GB of the cold-user run. The swap guard (`+512 MB`) did not trip
  and printed `armed` and its baseline, so it ran and was not exceeded. Whatever grew 1.9 GB in the v0.19.0 run is **not reproduced**.
- The control's page-cache fill (+14 GB) moved swap by nothing, so on this box in this state page cache growth alone does not explain swap growth: the cold-user
  run's "may be the kernel" explanation is **not supported** by this pass either. It is also not ruled out for a box with less free memory (here MemAvailable stayed ≥ 37.8 GB).
- One pass each, with the 26B's files warm in the page cache from earlier runs today (A read warm, B cold-ish). A second, cold-cache pass would be the way to
  separate more; not run, because this one is not close to the bars in either direction and R21 asked for no fix until measured.

R21 stays open as a measurement with an ambiguous result, not closed.

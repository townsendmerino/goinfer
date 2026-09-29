# MC3 per-pass prefill: how much of a 4-client served cell is prefill passes — pre-registration, 2026-09-28

The first step of the "per-pass prefill cost cut" follow-on in
[`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md) (MC3 follow-ons).
- **Why it is owed:** the item rests on an estimate. Newcomer suffix prefills were put at "~⅓ of a 4-client 1.5B
  cell", from W7's own numbers, before chunked prefill shipped (`-prefill-chunk 512`, default since 2026-09-27).
- **What this measures:** that share, directly. Nothing is built unless the share is large. **Written before the run.**

## 1. The instrument

- **073be561:** `ResidentBatchStats` gains wall time with the resident held. `RunNs` is time inside decode runs,
  `ExclusiveNs` inside exclusive sections, and `PrefillNs` / `PrefillPasses` inside the sections that are prefill
  passes (chunked `PrefillLast` and the final seed). `TestMC3_longPrefillChunksWhileOthersDecode` checks the pass
  count against the resident's calls and that the durations nest; zeroing the counter turns it red.
- **9e1d31c8:** `GET /admin/status` reports the counters per MC3 model. The driver reads them over serve's admin socket
  immediately before and after the cell, and differences them. So load, the warm request and first-use pipeline
  compiles are excluded.
  - An exploratory smoke showed why that matters: the server's lifetime counters carried 1.49 s of pre-cell prefill.
  - The differenced numbers add up: prefill + runs + bookkeeping + 0.06 s idle = the cell's wall clock.
  - The smoke used 16-token replies. Its share is not quoted.

## 2. The run

*(Hashes are the ones on `main`. The binary was built before a rebase renamed the commits; the only commit
interleaved, 9a48035e, adds a CUDA test and docs, so the Metal binary's sources are 9e1d31c8's.)*

- **Binary:** `serve-metal-9e1d31c8`, Metal resident, int4, `-embed-int4=false`. Since 9ccf7fb1 the int4-embedding
  default makes the Metal resident decline to the CPU; a cell whose log does not show `decode path: metal-resident`
  is void. Otherwise serve's defaults: 4 generations, `-prefill-chunk 512`.
- **Workload:** `bench_w7_plain.py`'s W7, 4 clients × 4 growing chat turns, 128 max tokens, greedy, fixed nonces.
  One warm request, then the cell.
- **Models:** Qwen2.5-Coder-1.5B-Instruct Q4_K_M (graded) and Qwen2.5-7B-Instruct Q4_K_M (reported), both from
  `~/models`. 3 repetitions each, each on a fresh server.
- **Mac night queue**, ~30 min: [`run-mc3-prefill-attr.sh`](mc3-prefill-attr-2026-09-28/run-mc3-prefill-attr.sh),
  driver [`mc3_prefill_attr.py`](mc3-prefill-attr-2026-09-28/mc3_prefill_attr.py).

## 3. The decision rule, on the 1.5B's median prefill share (ΔPrefillNs ÷ the cell's wall clock)

- **< 0.15 → close the cut.** The estimated ~⅓ does not hold with chunked prefill, and the follow-on is struck with
  this record.
- **≥ 0.25 → register the cut** as its own item. It gets a per-pass cost breakdown (passes, prompt tokens prefilled,
  time per pass and per token) and a bar, before any code.
- **0.15–0.25 → the owner's call**, with the numbers.
- **Fewer than 2 valid 1.5B cells → inconclusive**; the run is fixed and re-queued.
- **Reported, not graded:** the 7B's share, runs and bookkeeping time, idle time, passes per cell, and the lifetime
  counters beside the differenced ones.

## 4. Results (Mac night queue, 2026-09-28 21:17–21:27 PDT): the share is 0.151. CLOSED by the owner

The Mac night queue ran `run-mc3-prefill-attr.sh` unattended. It was idle-gated, on AC power, and every cell was on
the Metal resident (`decode path: metal-resident (int4)` in each server log). Raw data:
[`results.json`](mc3-prefill-attr-2026-09-28/results.json), [`run.log`](mc3-prefill-attr-2026-09-28/run.log) and the
six `server-*.log`.

| model | rep | cell wall (s) | prefill (s) / passes | decode runs (s) | bookkeeping / idle (s) | **prefill share** | aggregate tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5B | 1 | 19.78 | 3.01 / 24 | 16.46 | 0.001 / 0.31 | **0.152** | 155.3 |
| 1.5B | 2 | 20.00 | 3.00 / 24 | 16.80 | 0.001 / 0.19 | **0.150** | 153.6 |
| 1.5B | 3 | 19.80 | 2.99 / 24 | 16.61 | 0.000 / 0.20 | **0.151** | 155.2 |
| 7B | 1 | 66.76 | 10.15 / 24 | 56.16 | 0.001 / 0.45 | 0.152 | 46.0 |
| 7B | 2 | 66.59 | 10.19 / 24 | 56.19 | 0.001 / 0.21 | 0.153 | 46.1 |
| 7B | 3 | 67.45 | 10.20 / 24 | 57.04 | 0.001 / 0.21 | 0.151 | 45.6 |

- **The graded reading:** the 1.5B's median prefill share is **0.151**, just inside the owner's-call band
  [0.15, 0.25). The three repetitions span 0.150–0.152, and the 7B reads the same (0.151–0.153).
- **The estimate it tested does not hold.** Newcomer prefills are ~15% of a 4-client cell, not the ~⅓ estimated before
  chunked prefill. Decode runs are ~84%.
- **Per pass:** each cell prefilled 609 new prompt tokens in 24 passes. 9,884 prompt tokens were sent and 9,275 reused
  from the slots' prefixes, so each pass averages ~125 ms (1.5B) and ~425 ms (7B).
- **What a cut could buy:** it removes per-pass overhead, not the prefill compute itself, so its ceiling is a fraction
  of the 15%.

**Owner decision, 2026-09-28: close the per-pass prefill cost cut.** The follow-on is struck in
`task-concurrency-2026-09.md`. The instrument stays: the counters print at shutdown and are on `/admin/status`, so a
later workload with longer or more frequent newcomer prompts can be read the same way.

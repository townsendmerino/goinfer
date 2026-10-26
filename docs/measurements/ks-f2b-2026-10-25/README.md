# KS-F2b: #239's attendQueryP schedule on the other families, Mac CPU (2026-10-25)

Francis's order of 12:30 (relayed), item 2: time #239's new `attendQueryP` schedule on the families it was held back from. It fans
the heads out over the attention workers and scores eight keys at once as independent f64 chains. The families are Qwen3.5/3.6,
Granite-4.0-H, Nemotron-H, LFM2 and Llama 4. Graded at the merge rule, per family, with the memory cost stated. Any default change is
Francis's call. This is coordinated with mac-kapitsa, who owns KS-F2: its pointers are used here, and its code is untouched.

## Registration (before any timed reading)

- **Arms, two builds, one harness:**
  - **A:** `origin/main` at `ba8225aa`, with today's `attendQueryP`.
  - **B:** branch `ks-f2b-2026-10-25` at `48598110`, with f8a1eda2's `attendQueryP` body restored. Its helpers (`dotsF64x8`,
    `scalarHeadWorkers`, `decodeScratch.headScoresPool`) are already on main for gpt-oss.
  - **Harness:** both run `decoder/ksf2_speed_test.go:TestKSF2_cpuDecodeSpeed`, which uses exported API only and is unchanged on the
    branch. CPU, int4, a prompt of `depth` pseudo-random ids, then 32 greedy tokens timed from the first to the last (ms a token).
- **Bit-identity (done, before timing):**
  - `TestKSF2_queryPFannedBitIdentical` on B: 768 runs (ring, window and global; softcap 0 and 30; GOMAXPROCS 1 and 8) equal word
    for word to main's code, kept verbatim as `attendQueryPRef`. 132 of them are in cells the fan-out rule takes, and all 132 fanned
    out. This is `TestKSF2_queryPKeepsTheOldPath` inverted.
  - mac-kapitsa's `gate identity` at f8a1eda2 read IDENTICAL for qwen3_5_moe, granitemoehybrid, lfm2, nemotron_h and llama4_text,
    64 steps, on the Mac CPU and on nobara amd64.
- **Cells on the Mac (M1 Pro, CPU).** Real checkpoints this Mac's CPU path can load:
  - **Qwen3.5-0.8B** (`~/models/qwen3.5-0.8b`) at depths 128 and 1,024. Fit check: 3.6 GB peak resident; about 20 s a run at 1,024.
  - **LFM2.5-2.6B** (`~/models/lfm2.5-2.6b`) at depths 128 and 1,024. Fit check: 8.7 GB peak resident; its prefill runs token by token
    at about 180 ms a token, so about 3 minutes a run at 1,024.
- **Not covered on this Mac, with the reason:**
  - **Granite-4.0-H:** no checkpoint on the Mac; only the tiny fixture.
  - **Nemotron-H:** only the 9B's Metal `.giw`, and a 9B on the CPU path fails the fit guard here, as the 7B did for R-06.
  - **Qwen3.6:** the 35B-A3B is off-limits on the Mac's CPU.
  - **Llama 4:** no checkpoint small enough exists.
  - Those cells need nobara (its session) or a pulled small checkpoint. That is for Francis.
- **Instrument:** `run_ksf2b.py`, under the timing lock (exclusive). Per cell: 2 warm runs of each arm, then 10 valid pairs, A B then
  B A alternately.
- **The void rule:** a run is void when the 1-minute load at its start is over 4.0, when reclaimable memory at its start is under
  the cell's fit-check peak plus 2 GiB, or when the machine pages or swaps during it (`vm_stat`). A void pair is re-run, up to 6 a
  cell.
- **Grade per cell:** median A ÷ B (the speedup), and how many pairs B is faster. **Meets the merge rule** iff the median is ≥ 1.05×
  with at least 9 of 10 faster. A family meets it when all its cells do.
- **Timing (stated before starting):**
  - Qwen3.5-0.8B: about 2 min at depth 128 and about 8 min at 1,024, by day.
  - LFM2.5-2.6B: about 12 min at depth 128 and about 75 min at 1,024. Its 8.7 GB peak needs about 10.7 GB reclaimable, which the Mac
    does not have with apps open, so it goes on the Mac's night runner, gated on memory.
- **Memory cost of B:**
  - **Added:** score buffers per cache, at most `maxAttnWorkers` (6) × 2 × the key count × 4 bytes. That is 1.57 MB at 32k, the same
    bound #239 states for gpt-oss.
  - **Not added:** nothing else (no weights, no KV).
  - **Longest context that fits:** unaffected beyond that bound.

## Readings

Raw records: `docs/measurements/ks-f2b-2026-10-25/readings-qwen35.jsonl` (every run: arm, ms a token, load, reclaimable, paging,
validity). The instrument is `run_ksf2b.py` and the queued night job is `run-lfm.sh`, both beside this file.

**Qwen3.5-0.8B (2026-10-25, 13:37–13:48, the lock held exclusively; every valid run started at a 1-minute load of 2.68 or below with
9.4 GB or more reclaimable, and did not page):**

| depth | A (main), ms a token | B (fanned), ms a token | median A ÷ B | pairs | B faster | void pairs | grade |
|---|---|---|---|---|---|---|---|
| 128 | 20.64–21.31 | 18.48–19.13 | **1.113×** | 1.080–1.134 | **10 of 10** | 1 (paged 3 pages) | **meets the merge rule** |
| 1,024 | 37.44–38.74 | 20.41–21.40 | **1.833×** | 1.787–1.872 | **10 of 10** | 0 | **meets the merge rule** |

Qwen3.5 meets the merge rule at both depths, and the gain grows with depth: attention is a larger part of the token there.

**LFM2.5-2.6B:** queued on the Mac's night runner (`ksf2b-lfm25`, not before 22:00, gated on its 8.7 GB need); readings follow.

**Not covered on the Mac:** Granite-4.0-H, Nemotron-H, Qwen3.6 and Llama 4 (see the registration).

**LFM2.5-2.6B, forced run, 2026-10-26 (01:27–01:36): no reading, every run void.**
- **Why it ran forced:** the night runner held `ksf2b-lfm25` all evening. Free memory read 7.2 GB, under its 8.9 GB need plus the
  runner's 2 GB headroom. At Francis's "force it" (01:26), the coordinator dropped it from the queue and started `run-lfm.sh`
  directly under the timing lock (label `ksf2b-lfm25-forced`). So it started below the runner's headroom rule. The harness's own
  void rule still applied, as registered.
- **Result:** all 18 runs (4 warm, then 7 attempts at pair 0) were void.
  - **Memory:** reclaimable read 6.49 GiB at the first run and 9.59–10.04 GiB after, under the cell's 10.7 GiB line (8.7 GB plus
    2 GiB).
  - **Paging:** 17 of the 18 runs paged during the run, 4 to 183 pages.
  - The harness stopped after 6 void pairs ("over 6 void pairs"), before depth 1,024.
- **Not graded:** voided runs are not a reading. For the record only, the voided depth-128 runs read A 94.8–99.2 and B 92.2–96.7 ms
  a token.
- The log is beside this file: `lfm25-forced-2026-10-26.log`.
- **The cause is the cell's own load:** the 2.6B loads from its Hugging Face directory at an 8.7 GB peak, which leaves the Mac
  paging. A pre-quantized CPU `.giw` would load mapped, at a lower peak.

**LFM2.5-2.6B re-run from a CPU `.giw`, registered 2026-10-26 (01:42) before it runs, on Francis's (a).**
- **The bundle:** `cmd/prequant -quant int4 -target cpu-arm64` of `~/models/lfm2.5-2.6b`, written to
  `~/goinfer-bench/lfm-giw/lfm2.5-2.6b.int4.cpu-arm64.v17.giw` on the internal SSD (2.08 GB; kept out of `~/models` by the session's
  brief).
- **Fit check:** both arms load it mapped, at a **2.19 GB peak** against the directory's 8.7 GB. Decode reads 91–92 ms a token on
  both arms.
- **The cell's need becomes 2.2 GB:** a run voids under 2.2 GB plus 2 GiB reclaimable, or on any paging, as registered.
- Everything else is as registered: the same two binaries, the same harness and rules, depths 128 and 1,024, 10 valid pairs each,
  graded at the merge rule.
- **Cost:** about 6 min at depth 128 and about 42 min at 1,024, under the timing lock.

**The `.giw` re-run's result (01:40–01:49): no complete reading. The paging check voided half the runs while memory was ample.**
- **Memory was not the problem:** reclaimable read 8.78–9.50 GiB through the run, about twice the 4.2 GiB line.
- **Paging was:** 8 of 24 runs recorded machine-wide pageouts during the run, 1 to 29 pages each (at most about 0.5 MB). The
  registered rule voids a run on any paging at all. The harness stopped after 6 void pairs, at depth 128 pair 3, so depth 1,024 did
  not run.
- **Not graded:** 3 valid pairs, B faster in all 3, at 1.017×, 1.027× and 1.022×. That is too few pairs to grade.
- **What it shows:** with this process far from memory pressure, the "any paging" line counts background paging by other processes.
  It cannot complete on this Mac in an ordinary state. Changing the line is a void-rule change, so it is Francis's call.
- The log is beside this file: `lfm25-giw-2026-10-26.log`.

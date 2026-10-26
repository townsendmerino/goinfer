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

## nobara: the four families the Mac's CPU cannot load

Francis's order of 14:01 (relayed): time the schedule on nobara for Granite-4.0-H, Nemotron-H, Qwen3.6 and Llama 4. It ships on by
default from #247, untimed on these four, so this run is their confirming record. A family that reads slower than the old schedule
(median below 1.0, or fewer than 9 of 10 pairs faster) is a DECISION NEEDED for Francis, since the default is already on.

### Registration (2026-10-25, before any timed reading)

- **Arms, two builds, one harness:**
  - **A:** `21245f4f4`, main before #247: the old `attendQueryP` for these families.
  - **B:** `0058f0be9`, main after #247: the fanned schedule. Between the two, `decoder/` differs in `attendQueryP`'s body (the fan-out
    and `dotsF64x8`), its counter `attendQueryPFanned`, one line of `decoder/scratch.go`, and tests.
  - **Harness:** `decoder/ksf2_speed_test.go:TestKSF2_cpuDecodeSpeed` (unchanged between the two), built once per arm with
    `go test -c` and run one process a run. CPU, `--quant int4`, a prompt of `depth` pseudo-random ids, 32 greedy tokens timed first to
    last (ms a token).
- **Bit-identity, before timing:** `TestKSF2_queryPFannedBitIdentical` and `TestKSF2_familiesTakeTheFannedSchedule` on B, on nobara
  (amd64). mac-kapitsa's `gate identity` on nobara at f8a1eda2 (night job `ks-f2-nobara` and its granite re-run `ks-f2-g1b-granite`,
  2026-10-25) is the family-level record; it is cited, not re-run.
- **Cells:** depths 128 and 1,024 for each model:
  - **Granite-4.0-H:** `~/models/granite/granite-4.0-h-tiny-Q8_0.gguf`.
  - **Nemotron-H:** `~/models/nemotron-h-4b-instruct-128k`.
  - **Qwen3.6:** `~/models/qwen3.6-35b-a3b-q4_k_m.gguf`.
  - **Llama 4:** `~/models/llama4-scout-gguf/Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf`. Expected NOT to fit: int4 of 109B parameters is
    about 60 GB, on a 62 GB box. The fit check decides; a refusal leaves Llama 4 uncovered on both machines, and that goes to Francis.
- **Fit check, before any timed run:** one B run per model at depth 128 and 2 tokens, under `systemd-run --scope` with `MemoryMax=52G`
  and no swap, so a load that does not fit is killed rather than swapping the box. It gives each model's peak resident memory (the
  cell's need) and its prefill rate, from which the cells' wall time is stated before they start.
- **Instrument:** `run_ksf2b_linux.py` beside this file: `run_ksf2b.py`'s design on Linux, under the timing lock (exclusive). Per cell:
  2 warm runs of each arm, then 10 valid pairs, A B then B A alternately.
- **The void rule:** a run is void when the 1-minute load at its start is over 8.0 (half nobara's 16 threads, as CLAUDE.md's), when
  MemAvailable at its start is under the cell's need plus 2 GiB, or when the test process swaps (its own VmSwap peak, sampled every
  0.5 s, above 0). The system's swap-in and swap-out page count over the run is recorded as context only: nobara carries swap that
  no run owns. A void pair is re-run, up to 6 a cell.
- **Grade per cell:** median A ÷ B and how many of the 10 pairs B is faster. **Meets the merge rule** iff the median is at least 1.05×
  with at least 9 of 10 faster. **Slower than the old schedule** (DECISION NEEDED) iff the median is below 1.0 or fewer than 9 of 10
  pairs are faster. Between the two (median 1.0–1.05 with 9 or 10 faster): neither, reported as such.
- **Where it runs:** by day when a family's two cells together come in under about 2 hours; a family over that goes on nobara's night
  queue as its own job of 3 hours or less.
- **Memory cost of B:** as on the Mac: score buffers per cache, at most `maxAttnWorkers` (6, a constant, so the same on nobara's 16
  threads) × 2 × the key count × 4 bytes; nothing else.

### Fit check (2026-10-25, 16:11–16:17, the exclusive lock; `nobara/fit.log`, `nobara/fit.jsonl`)

Both arms built at the registered commits. One B run per model, depth 128, 2 tokens:

| model | peak resident (VmHWM) | swap | prefill of 128 | outcome |
|---|---|---|---|---|
| Granite-4.0-H tiny Q8_0 | 12,635 MiB | 0 | 29.4 s (0.23 s a token) | fits |
| Nemotron-H 4B | 16,352 MiB | 0 | 18.8 s (0.15 s a token) | fits |
| Qwen3.6-35B-A3B q4_k_m | — | — | about 250 s (about 2 s a token) | ran, then stopped on a stop token after 1 of 3 tokens: the harness's run fails |
| Llama-4-Scout Q2_K | — | — | — | refused by the fit guard |

- **Granite-4.0-H and Nemotron-H:** CPU prefill is token by token here, so a depth-1,024 run is mostly prefill. Per run, about 250 s
  (Granite) and 160 s (Nemotron) at 1,024, and 42 s and 29 s at 128. Over 24 runs a cell (4 warm, 20 timed), that is about 2 h 10 min
  for Granite's two cells and 1 h 30 min for Nemotron's. Each goes on nobara's night queue as its own job (`run-nobara.sh <family>`)
  at priority 60. That is ahead of ks-o4-gfid (90), and both end before mac-tack's 22:30 screen if they start by about 18:00.
- **Need, for the void rule:** 13 GiB (Granite) and 17 GiB (Nemotron).
- **Qwen3.6 and Llama 4 are not timed:** see "Untimed, by Francis's decision" below.

**Untimed, by Francis's decision (2026-10-25, 16:29, option b):** Qwen3.6-35B-A3B and Llama-4-Scout are not timed on either
machine. The default stays on. The record for the qwen3_5 attention path is the Mac's Qwen3.5-0.8B reading above (1.113× and 1.833×,
10 of 10).
- **Qwen3.6-35B-A3B, why:**
  - Its CPU prefill on nobara is about 2 s a token, so one depth-1,024 run is about 34 minutes and the cell about 13 hours. The
    depth-128 cell alone would be about 1 h 45 min.
  - The harness's random-id prompt also draws a stop token at once, so timing it at all would need an instrument change in both
    arms (ignore the stop token).
- **Llama 4 (Scout, 109B), why:** the fit guard refuses it on nobara's 62 GB, and no checkpoint small enough exists for the Mac.

### Readings on nobara

Raw records: `nobara/readings-<family>.jsonl` (every run: arm, ms a token, load at its start, MemAvailable, VmHWM, the process's swap
peak, the system's swap pages, validity) and the night jobs' logs `nobara/night-<family>.log`.

**Granite-4.0-H tiny (2026-10-25, 16:28–18:24, night job `ksf2b-nobara-granite`):**
- Every run started at a 1-minute load of 2.52 or below, with 58.2 GiB or more available, and did not swap. No void runs.
- Peak resident memory: 12,647 MiB.

| depth | A (old), ms a token | B (fanned), ms a token | median A ÷ B | pairs | B faster | void pairs | grade |
|---|---|---|---|---|---|---|---|
| 128 | 236.6–238.0 | 235.8–237.0 | **1.0033×** | 1.0004–1.0060 | **10 of 10** | 0 | below the merge rule; not slower |
| 1,024 | 252.4–253.4 | 241.1–242.4 | **1.0455×** | 1.0441–1.0474 | **10 of 10** | 0 | below the merge rule; not slower |

- B is faster in every pair at both depths, but by less than the merge rule's 1.05×. Granite-4.0-H is mostly Mamba-2 layers, so
  attention is a small part of its token.
- Not slower than the old schedule (the DECISION NEEDED trigger), so the default stands on this record.

**Nemotron-H 4B, attempt 1 (2026-10-25, 18:40–19:48, night job `ksf2b-nobara-nemotron`): VOID, both cells.**
- Both cells hit the 6-void-pair limit (`nobara/readings-nemotron-1-void.jsonl`, `nobara/night-nemotron-1-void.log`).
- The void rule caught the harness's own residue: a Nemotron run keeps all 16 threads busy, and the next run started at a 1-minute
  load of 8–12, over the 8.0 bar. Granite's runs never took the load past 2.5.
- These runs are not graded.

**Amendment before attempt 2 (2026-10-25, 20:10, no reading taken since):** `run_ksf2b_linux.py` gains an idle gate. Before each
run it waits, up to 5 minutes, for a 1-minute load of 4.0 or less, and records the wait (`settled_s`). The void rule is unchanged
(8.0 at the run's start). The cost is about a minute a run: Nemotron's two cells are now about 2 h 5 min, re-queued as
`ksf2b-nobara-nemotron-2`.

# G46: Gemma 3's tensor-core GEMM prefill between 4,096 and 12,160 keys, read against the 0.99 bar (pre-registration, 2026-10-21)

<!-- citations-at: 7cb04cd1c -->

> **Status 2026-11-24: a record of 2026-10-21, committed late.** The registration and the read were made on 2026-10-21 on the CUDA
> box (an RTX 2070 SUPER, 8 GB, 16 CPU cores) and stayed on a local branch; they are committed here unchanged except that the box is
> named by its hardware. The original commits, the registration e5bb7a79c and the read 7cb04cd1c, are on the branch
> `archive/g46-2026-10-21`, and this record's citations are checked at 7cb04cd1c. At the pin the test is
> `cuda/g46_gemm_fidelity_test.go:TestG46_gemma3GEMMFidelityLadder`; it is now in the backend/cuda package's KV3 tiled test file under
> the same name (the package moved), ported without a change to what it reads and not re-run since; the run script below builds it
> from there. Its prompts are encoded from the tree's own sources (`kv0TextTokens`), so a re-run on a later tree
> reads different text.

> **Re-read 2026-11-24** on main `7d4b58916`, the port run as registered: AMBIGUOUS (11 cells PASS, seed 9001 at 12,160 parked at 0.990694; the controls held). Today's cells beside these: `docs/measurements/g46-gemma3-gemm-fidelity-2026-11-24.md`.

**Registered before the run** (decided 2026-10-21: run it by day on the CUDA box, `docs/queue-correctness.md` G46). The rule and the bar are
`docs/tasks/task-kv-architecture-2026-10.md` (KV3) and `docs/measurements/kv3-tiled-exact-2026-10-13.md` amendment 1: the batched prefill's
last-row logits against the per-token path, **cosine 0.99 or better and equal argmax**. That record read the default arm at 1,200 tokens (0.9963),
at 12,000 (0.990073) and at 12,400 (0.986525, FAIL, past the bound where the exact row kernels run); the lever's own gate stops at 3,900 tokens.
Nothing between has been read.

## Question

Does the tensor-core GEMM (`GOINFER_CUDA_FAST_PREFILL`'s projections, on above the 512-token floor) keep Gemma-3-4B's prefill within the bar at every
depth the batched prefill serves below the shared-memory bound?

## Design

- **Machine and build.** An RTX 2070 SUPER 8 GB (16 CPU cores), driver 595.91.07 (the current anchor). The test binary is built by `run-g46.sh` from the
  committed tree; its commit and the Go version are written to `g46-gemma3-gemm-fidelity-2026-10-21/run/start.txt`.
- **Checkpoint.** `~/models/gemma-3-4b-it-Q4_K_M.gguf` (local disk, not the archive), loaded `Quant: int4`, f32 KV, resident context 12,760, as G4 loads it.
- **Greedy.** Nothing is sampled: the reading is last-row logits and their argmax.
- **Cells.** Two prompts (corpus stream offsets 4242, as G4, and 9001) x six depths: 4,096, 6,144, 8,192, 10,240, 12,000 and 12,160 tokens. 12,160 is the last
  length below the bound (`splitKVRequired` is false at each; the test fails if it is not). Each cell is the default arm: the batched prefill
  (`PrefillLast`) of the first d tokens, graded against the per-token path's last-row logits at row d.
- **One reference per prompt.** The per-token pass over the longest prompt reads logits at each depth's last row (`Forward` there, `ForwardNoLogits`
  elsewhere), so six references cost one pass. The per-token path does not depend on the levers.
- **Controls, in the same process.** At the shortest and longest depth of each prompt, a levers-off arm (fast attention and tensor-core GEMM off) must
  read 0 logits differing and 0 tensor-core launches (the exact class, as G4's levers-off arm). A nonzero control voids the run's attribution of any
  difference to the GEMM; it fails the test. Each default cell must also launch the tensor-core GEMM (else the cell read nothing about it) and no
  row-kernel group.
- **Interleaving.** The comparison is within one process and one prompt: per-token reference, then default and control arms, nothing timed against anything.
  No A B B A is needed; the reference is the single other arm.
- **Reused readings.** None reused. The 2026-10-14 reading at (seed 4242, first 12,000 tokens), cosine 0.990073, was taken on a different build
  (`kv3-tiled-2026-10`); this run re-reads that cell, and **if the corpus stream is prefix-stable and the GEMM unchanged it should read the same to six
  digits.** That agreement is reported, not graded.

## Pre-registered reading

Per cell: **PASS** cosine >= 0.992 and equal argmax. **FAIL** cosine < 0.99 or a different argmax. **AMBIGUOUS** cosine in [0.990, 0.992) with equal
argmax: the 0.992 is a parking line, not a new bar: the 2026-10-14 reading sat 0.00007 above 0.99, and a cell that near the bar is not a pass or a fail.

Whole run: **PASS** if all 12 default cells PASS and both controls hold. **FAIL** if any default cell FAILs or any control does not hold. Otherwise
**AMBIGUOUS, parked**: reported cell by cell with no conclusion; the owner decides what a parked cell means. A FAIL registers no remedy here: it
names the first depth under the bar, and the owner decides whether the lever's depth range, the bar or the kernel is what changes. The bar is not moved.

## Gate shape (CLAUDE.md "Gate tiers", the noise-registry template)

- **Tier:** day. **Testing tier:** 1 (one small real model; tier 0 cannot answer it: the question is the real checkpoint's logits at depth).
- **Instrument:** `TestG46_gemma3GEMMFidelityLadder` (`cuda/g46_gemm_fidelity_test.go`), the cheapest instrument that reads the bar: G4's own reading
  (cosine and argmax against the per-token path) at more depths.
- **Stopping rule:** fixed, all 12 cells and both controls; no early stop. A load failure, a NaN or a decline stops the test with that named.
- **Noise:** the kernels are deterministic, so a cell repeats bit for bit on the same build and card; a second run of an unchanged cell is not needed
  (CLAUDE.md "never re-measure an unchanged arm"). Timings in the log are reported, never graded.
- **Cell validity:** `run-g46.sh` records the load average, `MemAvailable` and free VRAM at the start (`run/start.txt`). The run is VOID if the 1-minute
  load is above 8.0 (half the box's 16 cores) or `MemAvailable` is under the load's need plus 2 GB (about 5 GB: the 2.5 GB file, the host copy and
  the margin). The desktop compositor's CUDA context (about 16 MiB) is counted in the VRAM figure.
- **Fit check:** the load is G4's (same file, int4, f32 KV, 12,760 against 13,000 tokens of context) which ran on this card on 2026-10-14; the test's own
  load is the check and fails within seconds if refused. Nothing queues behind it.
- **Never-swap:** not a registered bar here; the test process's `VmHWM` and peak `VmSwap` are sampled every 5 s from `/proc/<pid>/status` and recorded
  (`run/proc-samples.txt`, `run/peak.txt`). System swap is context only.
- **Timing lock:** the run holds the exclusive timing lock (`timing_lock.py run --label g46`); nothing else runs on the card beside it.

## Estimated wall time

About 10 minutes, against the owner's 40: the design shares one per-token pass per prompt. Per prompt: the per-token pass over 12,160 rows at about 95 rows/s
(G4: 130 s for 12,400), about 130 s; six default batched prefills, about 3 s at 4,096 to about 20 s at 12,160 (G4: 23 s at 12,400), about 65 s together; two levers-off
controls, slower than default, about 70 s together (a guess: G4 gives no levers-off batched time). Twice, plus a 20 s load: about 9 to 10 minutes. The log
prints a heartbeat every 30 s in the per-token pass (rows done, elapsed) and a line per cell.

Reproduce: `bash docs/measurements/g46-gemma3-gemm-fidelity-2026-10-21/run-g46.sh` from the repo root.

## Results (appended when read)

Run 2026-10-21, 16:21:50 to 16:31 PDT (550 s, against about 10 minutes estimated), by day, under the exclusive timing lock. Commit `f61f25fd` (the test as
registered in `e5bb7a79`, unchanged), Go 1.27.3, driver 595.91.07. At the start: load 2.55, `MemAvailable` 56.6 GB, 7,382 MiB of VRAM free (valid: under
8.0 and well above the need). Test process: peak `VmHWM` 6.07 GB (6,071,272 kB), `VmSwap` 0 in all 110 samples. Records: `g46-gemma3-gemm-fidelity-2026-10-21/run/`
(`g46.log`, `start.txt`, `proc-samples.txt`, `peak.txt`); the driver's log is `~/goinfer-logs/g46-2026-10-21.log`.

| depth | seed 4242: cosine | argmax | cell | seed 9001: cosine | argmax | cell |
|---|---|---|---|---|---|---|
| 4,096 | 0.994150 | equal | PASS | 0.995975 | equal | PASS |
| 6,144 | 0.996105 | equal | PASS | 0.996998 | equal | PASS |
| 8,192 | 0.995950 | equal | PASS | 0.991072 | equal | AMBIGUOUS |
| 10,240 | **0.987131** | equal | **FAIL** | 0.995030 | **differs** | **FAIL** |
| 12,000 | 0.996317 | equal | PASS | 0.996074 | equal | PASS |
| 12,160 | 0.995662 | equal | PASS | 0.990061 | equal | AMBIGUOUS |

Every default cell launched the tensor-core GEMM (1,904 launches at 4,096 to 5,712 at 12,000 and 12,160) and no row-kernel group; every logit differs from the
per-token path in every default cell (worst 2.0 to 3.37). **All four controls held:** levers off, 0 of 262,144 logits differing and 0 tensor-core launches at
4,096 and 12,160 on both prompts, so the differences above are the levers', and with no fused attention for head dim 256 that is the tensor-core GEMM.
Per-token reference: 135.9 s and 136.2 s for 12,160 rows.

**Verdict, by the registered rule: FAIL.** The first depth under the bar is **10,240 tokens, on both prompts**: seed 4242 by cosine (0.987131), seed 9001 by
argmax (cosine 0.995030, a different top token). Two further cells are in the parking band (0.991072 and 0.990061). As registered, no remedy is
proposed here; the bar is not moved, and what changes (the lever's depth range, the bar, or the kernel) is the owner's decision.

Reported, not graded:
- **The cosine does not fall with depth.** On both prompts the worst cell is at 10,240, and 12,000 reads better than either neighbour; seed 9001 dips at 8,192
  and 12,160 too. The GEMM's distance from the per-token path moves by up to 0.009 between neighbouring depths of one prompt, so it depends on the prompt
  content at least as much as on its length. A depth cut-off fitted to these twelve cells would be fitted to two prompts' content.
- **The 2026-10-14 cell did not reproduce, and was not the same prompt.** Seed 4242 at 12,000 read 0.996317 here, against 0.990073 on `kv3-tiled-2026-10`.
  The registration expected agreement "if the corpus stream is prefix-stable"; it is not stable across trees: `kv0TextTokens` encodes the tree's own
  `decoder/*.go` sources, so every commit that edits them changes every prompt. The two readings are of different text, and say nothing about the GEMM
  having changed. The same holds for re-running this record: the prompts are those of commit `f61f25fd`'s sources.
- The run script stopped at the test's non-zero exit before writing `run/peak.txt` (`set -e`); the peak was computed from `run/proc-samples.txt` after the run,
  and the script now records the peaks and then exits with the test's status.

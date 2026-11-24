# CPU KV row pad, full implementation: registration, 2026-11-24 05:59 PST, before any timed run

**Why.** The trimmed prototype A/B read ADOPT (`docs/measurements/cpu-kv-rowpad-2026-11-23/README.md`, § Result). This measures the full
implementation (every reader of the CPU KV layout takes the padded stride; branch `cpu-kv-rowpad-2026-11-23`), adds depth 128, settles
whether the 16-key QK kernel ships under the pad, and re-measures the f16 read on the kernel that would ship. Its outputs are
bit-identical padded and unpadded (`TestKVRowPad_everyReaderBitIdentical` and the other `TestKVRowPad_*` tests).

**Instrument.** `BenchmarkDecodeAtDepth/grouped_on` with `GOINFER_BENCH_FILL=1`, 32 timed tokens a cell, one process a cell, A B B A ×
5 blocks = 10 pairs a comparison, `run_ab.py` (`docs/measurements/cpu-qk-multichain-2026-11-23/`, `--pad-b` for arm B's pad) under the
exclusive lock, its void rule unchanged (start load over 8.0 or swap growth: void, re-run in place, listed). Binaries of one goinfer
commit: **K16** on aikit `f16-acc64-2026-11-23` (the 16-key f32 QK kernel; the f16 kernels), **K4** on aikit `k4-variant-ab-2026-11-23`
(the same with the f32 16-key dispatch off). The f16 kernels are the same in both. Every padded Phi-3 f32 cell must report `rowpad 16`.

| id | comparison (B ÷ A) | depths | role |
|---|---|---|---|
| R1 | Phi-3 f32, K16, padded ÷ unpadded | 3,900, 128 | the pad (default question) |
| R2 | Phi-3 f32, padded, K16 ÷ K4 | 3,900, 128 | the 16-key kernel under the pad (QK dispatch) |
| R3 | Phi-3 f32, K4, padded ÷ unpadded | 3,900 | report: the pad with today's kernel |
| F1 | Phi-3, K16, f16 ÷ f32 padded | 3,900 | the f16 read against the best f32 |
| C1 | 0.5B f32, K16, pad on ÷ off (not applied: A/A) | 3,900 | control |

**Bars, each fixed now:**
- **R1, the pad as the CPU default** (Francis's call, brought with this reading and the pad's memory, 128 B per token per layer, Phi-3
  16 MiB at 4,096 positions): **CONFIRMED** if R1 at 3,900 is ≥ 1.15 with ≥ 9/10 pairs faster **and** R1 at 128 ≥ 0.98 (no short-context
  cost); **NOT CONFIRMED** if R1 at 3,900 is under 1.05 or R1 at 128 under 0.98; between, PARKED.
- **R2, the 16-key QK kernel under the pad** (the four-condition rule for a bit-identical speed change): **SHIP** with the pad if R2 at
  3,900 is faster on ≥ 9/10 pairs and R2 at 128's median ≥ 0.98; otherwise it does not ship. The 0.5B's 16-key reading is carried from
  `docs/measurements/cpu-qk-multichain-2026-11-23/` (1.099×, 8/8 at 3,900; 1.000× at 128), not re-measured. If the pad does not become
  the default, the 16-key kernel does not ship (it read 0.948× on unpadded Phi-3).
- **F1, f16 KV as a CPU opt-in** (the lossless opt-in bar; fidelity passed on both models): **PASS** at median ≥ 1.05 with ≥ 9/10 pairs.
- **C1** outside 0.98–1.02 voids the session's readings (the instrument moved), not an option.
- **Prediction:** R1 at 3,900 about 1.17, at 128 about 1.00; R2 at 3,900 about 1.12; R3 about 1.00; F1 about 1.40.

**Cost:** R1 and R2 at both depths (four comparisons), R3, F1 and C1 at 3,900: about 105 min, under the exclusive lock, by day, on the
Ryzen 7 3700X box (16 threads), after the timed runs queued ahead of it. **Stopping rule:** fixed N, no early stop.

## Amendment, written 2026-11-24 09:59 PST before any timed run (after the review of #1297)

1. **F0 added, a decision cell: the f16 in-place read against the widen read it replaces,** same binary (K16), same commit,
   `GOINFER_BENCH_F16WIDEN=1` on arm A (`kvF16WidenForTest`). The in-place read is the default path for every f16 CPU cache, so it ships
   only by the four-condition rule on its own: bit-identical (`TestKVF16_decodeReadsInPlace`, `TestKVF16_F1Exactness` through the new
   read); scratch and memory not higher (it allocates no widen scratch; each cell's peak RSS is reported); faster on ≥ 9/10 pairs; it
   replaces the widen read (`kvF16WidenForTest` is a test-only switch back). **Cells:** Phi-3 f16 and the 0.5B f16 at 3,900 (decision,
   ≥ 9/10 pairs faster each), the 0.5B f16 at 128 (report, median ≥ 0.98 expected). Every arm-A cell must report `f16inplace 0` and every
   arm-B cell a positive count. Failing F0 splits the in-place read out of #1297 and drops it.
2. **R3 removed** (report-only; the kernel microbench put the 4-key kernel flat under a pad), to keep the run under 2 h.
3. **ADOPT outcome, stated now:** if Francis makes the pad the CPU default, `kvRowPadForTest` and `kvRowPadAlign` go and the pad becomes the
   code path (`Architecture.kvRowPad` with the 4 KiB rule); if it stays opt-in, it becomes a `decoder.Options` field with a per-model
   accessor. `kvF16WidenForTest` stays a test seam only while the widen read is the A/B's control, and goes with the widen read.
4. **One eligibility rule:** `Architecture.kvRowPad` decides the pad for both the cache (`KVCache.setRowPad`, which panics if a cache it
   accepts holds a ring, MLA latent or hybrid state) and the plan's price; `TestKVRowPad_planPricesThePad` checks price against held bytes
   on every fixture at both precisions.

**Revised cells and cost:** R1 (3,900, 128), R2 (3,900, 128), F1 (3,900), F0 (Phi-3 3,900; 0.5B 3,900 and 128), C1 (3,900): about 110 min.

## Result, read 2026-11-24 12:23 PST

200 cells, 10:34–12:22 PST under the exclusive lock, binaries `dec-rowpad-K16-a4e1c68bf` and `dec-rowpad-K4-a4e1c68bf` (#1297's head).
Start load 0.50–0.99 on every cell, no void cells. Every cell's own report matched its arm: R1's padded cells `rowpad 16` and its
unpadded ones `0`; R2's every cell `16`; F1's f32 arm `16` with no in-place read, its f16 arm unpadded (Phi-3's f16 row is 6 KiB) with
in-place reads; F0's widen arm `f16inplace 0` and its in-place arm a positive count on every cell; C1 never padded. Cells
`ab/*-cells.tsv`, per-cell logs `ab/logs.tar.gz`, driver log `ab/run.log`, graded by `docs/measurements/cpu-kv-rowpad-2026-11-23/ab/grade.py`.

| id | comparison (B ÷ A) | depth | A tok/s | B tok/s | median | pairs faster | bar | reading |
|---|---|---|---|---|---|---|---|---|
| R1 | Phi-3 f32 K16, padded ÷ unpadded | 3,900 | 2.921 | 3.442 | **1.180** | 10/10 | ≥ 1.15, 9/10 | pass |
| R1 | | 128 | 8.971 | 9.085 | **1.012** | 10/10 | ≥ 0.98 | pass |
| R2 | Phi-3 f32 padded, K16 ÷ K4 | 3,900 | 3.125 | 3.450 | **1.103** | 10/10 | ≥ 9/10 | pass |
| R2 | | 128 | 9.086 | 9.097 | **1.001** | 6/10 | median ≥ 0.98 | pass |
| F1 | Phi-3 K16, f16 ÷ f32 padded | 3,900 | 3.447 | 4.789 | **1.389** | 10/10 | ≥ 1.05, 9/10 | pass |
| F0 | Phi-3 f16, in-place ÷ widen | 3,900 | 1.500 | 4.782 | **3.190** | 10/10 | ≥ 9/10 | pass |
| F0 | 0.5B f16, in-place ÷ widen | 3,900 | 17.79 | 23.91 | **1.343** | 10/10 | ≥ 9/10 | pass |
| F0 | | 128 | 54.77 | 55.48 | 1.012 | 10/10 | report (≥ 0.98) | — |
| C1 | 0.5B f32, pad not applied (A/A) | 3,900 | 24.10 | 24.23 | 1.008 | 7/10 | 0.98–1.02 | in band |

**Readings, by the registered bars:**
- **R1: CONFIRMED.** The pad speeds Phi-3's f32 CPU decode 1.180× at 3,900 and costs nothing at 128 (1.012×). Making it the CPU default
  is Francis's call, with its memory: 128 B per token per layer, Phi-3 16 MiB at 4,096 positions (+0.52%). On ADOPT the switch goes
  (amendment item 3).
- **R2: SHIP with the pad.** The 16-key QK kernel is 1.103× the 4-key one under the pad at 3,900, 10/10, and level at 128 (1.001×);
  with the 0.5B's carried 1.099× (8/8) it ships if, and only if, the pad becomes the default (unpadded Phi-3 read 0.948×).
- **F0: PASS (four-condition).** The f16 in-place read is 3.19× (Phi-3) and 1.34× (0.5B) the widen read at 3,900, 10/10 each, peak RSS
  not higher on any cell (Phi-3 4.768 GB both arms; 0.5B 0.987–0.988 GB), bit-identical, and it replaces the widen read: it stays in #1297.
- **F1: PASS (lossless opt-in).** f16 KV decodes 1.389× the best f32 (padded, 16-key) on Phi-3 at 3,900, 10/10, at 4.77 GB peak RSS
  against 6.24 GB; with fidelity passed on both models (`docs/measurements/cpu-kv-f16-read-2026-11-23/README.md`), `--kv f16` on the CPU
  is a faster opt-in for Phi-3. Making it `auto`'s CPU pick is Francis's call.
- **Against the prediction** (R1 1.17 / 1.00, R2 1.12, F1 1.40): all within 0.02 but R2 (1.103 against 1.12); F0's size was not predicted.

**After the reading (12:31 PST):** the A/B seams named above were taken out of package globals, as the tree's Go rules ask for a new test seam:
the pad's switch is `Architecture.kvRowPadAlign` (0, the default, never pads; the benchmark's `GOINFER_BENCH_KVPAD=1` sets it to
`kvRowPadAlignDefault`), the in-place read count is `KVCache.f16InPlaceReads`, and with F0 passed the widen read and
`kvF16WidenForTest` are gone (`TestKVF16_decodeReadsInPlace` now checks the in-place read against the F1 reference). The binaries this
reading measured are `a4e1c68bf`; the change after it moves no arithmetic.

# aikit S-05 — the −8 centering folded into the SDOT accumulator (arm64 W4A8 decode kernel), and the Mac DECODE SPLIT table

`docs/tasks/red-october.md` R9; aikit `docs/task-simd-audit.md` §S-05. Box: `apple-m1pro` (M1 Pro,
6P+2E, 16 GB, macOS), goinfer `3ea2f93d`+, aikit working tree on `340ce19` (v1.46.0 + S-05).
Models: `qwen2.5-coder-0.5b`/`1.5b`-instruct and `qwen2.5-7b`-instruct q4_K_M GGUFs under
`~/models`, `Backend: "cpu"`, `Quant: "int4"`, greedy, 128-token prompt (R9's decision-cell
depth), 24 decode tokens. The box was NOT idle: load ~4 (VSCode, two Claude Code sessions,
WindowServer), ~100 MB free, 4.1 GB compressed, 2.6 of 3 GB swap in use at the start.

**Standing, so it is not re-derived:** S-05's own stop rule fired 2026-09-05 (goinfer ahead of
Ollama on the CPU prefill marginal), so the fold was never written for prefill. It is funded here
as a **decode** lever on the premise that the M=1 kernel sits at 91–97% of its per-core issue
ceiling and the fold is the one µop lever left in it (9 → 7 SIMD µops per row-group), bit-identical
by the int32 identity `Σ(nib−8)·act = Σnib·act − 8·Σact`. The Linux half of R9
(`cpu-decode-attribution-2026-09-22-linux.md`) found the amd64 token's remaining matmul cost
DRAM/kernel-bound, not fan-out-bound; whether the Mac is the same decides whether a single-core
kernel win reaches the token — step 0 below measures that first.

## Step 0 — the Mac DECODE SPLIT table (`TestR9_decodeAttribution`, shipped `GOINFER_DECODE_TIMING`)

ms/token; `sample`/`logitProc`/`embed` ≤ 0.10 everywhere, as before.

| component | 0.5B | 0.5B share | 1.5B | 1.5B share | 7B |
|---|---:|---:|---:|---:|---|
| **forward** | **10.3** | 100% | **20.3** | 100% | not run — see below |
| attention | 2.85 | 27.7% | 5.41 | 26.7% | |
| · q/k/v matmuls | 0.69 | 6.7% | 1.92 | 9.5% | |
| · rope + KV + scores/softmax/AV core | 1.64 | 15.9% | 2.25 | 11.1% | |
| · o matmul | 0.52 | 5.0% | 1.24 | 6.1% | |
| MLP | 5.97 | 58.0% | 12.41 | 61.1% | |
| · gate+up matmuls | 3.29 | 31.9% | 7.31 | 36.0% | |
| · **activation** | 0.81 | 7.9% | **1.32** | **6.5%** | |
| · down matmul | 1.85 | 18.0% | 3.75 | 18.5% | |
| LM head | 1.26 | 12.2% | 2.19 | 10.8% | |
| residual | 0.18 | 1.7% | 0.26 | 1.3% | |

The coarse rows reproduce the 2026-09-20 Mac record to within 1% (0.5B 10.3 vs 10.2; 1.5B 20.3 vs
20.4), which is the instrument's own sanity check before the finer rows are read.

**The 1.5B needed the fit guard bypassed** (`GOINFER_NO_FIT_GUARD=1`): the guard priced it at
~4.9 GB (2.1 resident + 1.8 KV + 1.0 checkpoint read) against a box with ~100 MB free. The 1.8 GB
KV is sized for the full context and a 152-token run touches almost none of it; the 2.1 GB of
anonymous weights fit inside the 4.6 GB of reclaimable inactive file cache. Run under the same
external swap monitor + automated kill discipline as the 2026-09-20 record's 7B (kill at +400 MB
swap growth): **swap did not move — 2594.62 MB before and after** — and the run took 6 s.

**The 7B was not run.** Same discipline, kill at +300 MB: swap grew **+452 MB inside the first
10 s of the load** (2578 → 3030 MB, against a 3072 MB swap file at the time) and the monitor
killed it. The 2026-09-20 record ran the 7B with 6.4–7.1 GB available; today's box has ~1/10 of
that. This is the memory note's "any fit-guard bypass here is a real risk" in numbers, and the
7B column stays empty rather than carrying a swap-spiral artefact. It needs a quieter box, not a
bigger threshold.

### (a) The activation share — no flip

1.32 ms of 20.3 = **6.5% on the 1.5B**, under the 8% bar for flipping `activationFanoutEnabled`.
The mechanism the Linux record found does not reproduce here: the 1.5B's fanned-out SwiGLU runs
at 28 × 8960 = 250,880 elements in 1.32 ms = **5.3 ns/element effective**, *faster* than the
0.5B's serial 6.9 ns/element (116,736 in 0.81 ms) — on Linux the fanned-out rate was 3× the serial
one. The `silubench` tuning that set the arm64 default stands; `TestR9_cpuTuningAB` was not run
for this knob because its precondition (≥ 8%) is not met.

### (b) The MLP matmul rate against the read ceiling — fan-out-bound, not bandwidth-bound

gate+up+down on the 1.5B: 2 × 8960 × 1536 × 28 + 8960 × 1536 × 28 = 1.156 G MACs, at 0.625
bytes/MAC (int4 + f32 scale per 32) = **722 MB of weights per token, in 11.06 ms = 65 GB/s — 54% of
the 121 GB/s six-thread read ceiling** (`completed/task-w4a8-neon-bandwidth.md`), and 104.5 GMAC/s
across six workers = **17.4 GMAC/s per worker against the kernel's 42–44 hot**. Neither at
bandwidth nor at the kernel's issue rate: this is aikit S-02's shape (six workers delivering ~2.4
cores' worth), the same finding the Linux half made in its own terms. So the pre-registration
below expects the fold's 1.29× to be mostly hidden at the token, and the record is written to say
so before the end-to-end number exists.

## The kernel — built, gated, measured (aikit `linalg/dot_w4a8_fold_arm64.s`)

`dotW4A8SplitHalf4RowFold` is `dotW4A8SplitHalf4Row` with the two per-row `VSUB.16B` gone and
`MOVI #0` replaced by a register copy of `corr_g`; `w4a8LaneCorrNeg8` computes
`corr[4g+l] = −8·Σ act` over lane l's eight k's (the row kernel's own SDOT lane mapping) once per
activation row — two SDOTs against a vector of int8 −8 per group. `MatmulBTW4A8Row4Into` (the
M=1 path every row4-resident tensor takes at decode) computes `corr` into a new `Workspace`
scratch and dispatches the fold kernel; `SetW4A8RowFold(false)` selects the old kernel for the
in-process A/B. The batched q‖k‖v span (goinfer R-06, opt-in, parked) keeps the old kernel.

**Encodings.** Ten new `SDOT` words (new `Rm`: the raw nibbles V1/V2 instead of the centered
V3/V4; and V31 for the pre-pass) and four `SCVTF` words computed from the encoding formula,
self-checked against the four words the existing kernel carries, and cross-checked word for word
against clang's assembler (`-march=armv8.2-a+dotprod`, `llvm-objdump`): **14/14 equal**.

**Gates, all exact `==`:** `TestW4A8LaneCorrNeg8_matchesScalar` (nGroups 1..37, random + the
int8 extremes); `TestDotW4A8SplitHalf4RowFold_bitIdenticalToSplitHalf4Row` — against BOTH the
kernel it replaces and canonical `dotW4A8FoldSDOT`, nGroups 1..20 and the production 48/280, ten
random activations plus all-(−128), all-127 and alternating; `TestMatmulBTW4A8Row4Into_foldBitIdenticalToUnfolded`
— the dispatch, fold on vs off, serial and six-way, at 1536×8960, 8960×1536 and a small shape. The
existing `TestMatmulBTW4A8Row4Into_bitIdenticalToMatmulBTW4A8Into`,
`TestWeightMatW4A8_MConsistentAcrossRow4Dispatch`, `TestMatmulBTW4A8_MConsistent` and the tile
gates now run through the fold and pass.

**Canaries, both red as required:** the pre-pass constant −8 → −7 (`MOVI #0xF8` → `#0xF9`) fails
all four gates including the raw-kernel one; a one-ULP perturbation (`Float32bits+1`) on the fold
path's row-0 output fails the three dispatch-level gates (the raw-kernel gate cannot see a Go-side
change, which is the correct shape). Both reverted; gates green again.

**Measured, `AIKIT_HARNESS=1`, this box** (`TestW4A8RowFold_hotAB` / `_streamedAB`):

| regime | shape | baseline | fold | ratio | pairs |
|---|---|--:|--:|--:|---|
| hot, one quad, L1-resident (min of 3) | K=1536 | 138.0 ns, 44.5 GMAC/s | 107.0 ns, 57.4 | **1.290×** | — |
| hot, one quad | K=8960 | 804.0 ns, 44.6 | 626.0 ns, 57.3 | **1.284×** | — |
| streamed 12-matrix bank, **single core** | 1536×8960 | 3.89 ms/pass, 42.4 GMAC/s | 3.07, 53.8 (33.6 GB/s) | **1.268×** | fold wins 3/3 |
| streamed, single core | 8960×1536 | 3.92, 42.2 | 3.02, 54.6 (34.2 GB/s) | **1.296×** | 3/3 |
| streamed, **six workers** | 1536×8960 | 1.64, 100.5 | 1.54, 107.3 (67.1 GB/s) | 1.068× | 3/3 |
| streamed, six workers | 8960×1536 | 1.76, 93.7 | 1.69, 97.5 (60.9 GB/s) | 1.040× | 3/3 |

Correction pre-pass: 19 ns (K=1536) / 100 ns (K=8960) per activation row = 0.008% / 0.042% of
one projection at the fold rate — free, as claimed.

**Reading.** The single-core kernel gains exactly what the µop count said — 9 → 7 is 1.286× and the
hot measurement is 1.284–1.290× — and it keeps that gain in the streamed regime because a single
core at 34 GB/s is nowhere near the 71.9 GB/s one thread can pull: **the decode kernel is issue-bound
on one core, and the S-01 read-back's "`MOVI #0` is not a rename-time zero idiom" is confirmed a
second time** (had it been, 9 → 8 would have measured ~1.13×). At six workers the same matmul
gains 4–7%: the fan-out is what step 0(b) said it was.

**Decision, by the registered rule (ship on any single-core win): SHIP.**

## End-to-end — PRE-REGISTERED BEFORE THE RUN

`TestR9_s05FoldAB` (goinfer `decoder/r9_s05_fold_ab_test.go`): the 1.5B, depth 128, 24 tokens,
`linalg.SetW4A8RowFold` flipped in-process, ABBA, 3 pairs, paired ratio with a win count, and the
matmul terms (q/k/v + o + gate+up + down + LM head = 16.4 of the 20.3 ms) reported beside the
token. Bit-identity is not re-argued here: the gates above hold the kernel exactly equal, and
goinfer's `TestForwardN_matchesSequential` runs after the bump as the cross-repo check.

- **Expected:** ~0 to a few percent — step 0(b) puts the token's matmuls fan-out-bound; the isolated
  six-worker streamed rows above gained 4–7% on the matmul alone, and a real token has other work
  between calls that the isolated pass does not.
- **Band:** paired OFF/ON ≥ 1.03× on 3/3 pairs → *reaches the token*; 1.015–1.03× or a split win
  count → *ambiguous, parked* (recorded, not claimed); < 1.015× → *does not reach the token, as
  predicted*. None of these changes the ship decision, which is the single-core rule above; the
  band decides only what R9's row is allowed to say about the token.
- **The number that would refute the reading:** a token gain at or above the single-core kernel
  gain (≥ 1.2×) — that would mean the matmuls were kernel-bound after all and 0(b) was misread.

## End-to-end — measured (`TestR9_s05FoldAB`, 1.5B, depth 128, fit guard bypassed under the monitor, swap flat)

| run | pairs | fold ON ms/tok | OFF ms/tok | paired OFF/ON | wins | per-pair range | matmul terms |
|---|--:|--:|--:|--:|---|---|--:|
| pre-registered | 3 | 19.69 | 21.44 | **1.089×** | 3/3 | 1.008 / 1.203 / 1.055 | 15.82 → 17.37 = 1.098× |
| replicate 1 | 6 | 19.24 | 21.11 | **1.097×** | 6/6 | 1.031–1.250 | 15.44 → 17.10 = 1.107× |
| replicate 2 | 6 | 19.33 | 20.26 | **1.048×** | 6/6 | 1.018–1.086 | 15.54 → 16.45 = 1.059× |

**Verdict by the band: reaches the token.** The pre-registered run clears it (1.089×, 3/3), but its
mean leans on one OFF-arm burst (23.79 ms on a box at load ~4), so the two replicates were added
with the same statistic and no change to the bar: 1.097× (6/6, every pair ≥ 1.031) and 1.048×
(6/6, two pairs at 1.018/1.021). Across the three runs the fold wins **15 of 15 pairs**; the ON arm
sits at 19.1–19.5 ms/token in every pair while OFF spreads 19.9–21.3 with bursts, so the honest
size is **~1.05–1.10× on the token, ~1.06–1.11× on the matmul terms** — 1.5B CPU decode from the
standing ~49 tok/s to ~52. The refuting number (≥ 1.2×) did not appear: the matmuls are still
fan-out-bound, and the kernel shows roughly 0.4× of itself at the token.

**The expectation was wrong, in the direction the campaign's scoreboard says to expect least.**
Pre-registered: "~0 to a few percent". Measured: 5–10%. The reading in 0(b) — 54% of the read
ceiling, 17.4 GMAC/s per worker — was right about the regime and wrong about what a faster kernel
does inside it: each worker's share is a *serial* stream where the kernel is issue-bound, and
shortening that stream shortens the barrier wait behind the slowest worker too; the isolated
six-worker rows (4–7%) had already said as much and were discounted for "other work between
calls" that turns out not to matter. Recorded as an under-prediction, the same way aikit's own
scoreboard records its over-predictions.

**Cross-repo bit-identity, checked after the fold and not assumed:** with goinfer built against
the working aikit (fold on by default), `TestForwardN_matchesSequential` is bit-identical across
**911,616 (K=6) and 19,447,808 (K=128) logits** and `TestSpeculativeGreedyParity` passes —
`decode == batched prefill == speculative verify` survives the kernel swap. `TestMoEExpertMajor_bitIdentical`
**skipped** (no MoE asset on this box) — a skip, not a pass; the box that has the asset owes it.

## What this establishes, and what it does not

- Established: the fold is bit-identical (three exact gates, both canaries red), 1.284–1.290× on the
  hot kernel and 1.268–1.296× single-core streamed, free to precompute, and worth 1.05–1.10× on the
  1.5B decode token on the M1 Pro at depth 128 — shipped by the registered single-core rule, with
  the token number reported beside it, not gated on.
- Established: the Mac DECODE SPLIT table for 0.5B and 1.5B with the MLP inner split; the Mac's
  activation fan-out does NOT lose (6.5% of the token, faster per element than serial), so
  `activationFanoutEnabled` stays on for arm64 by measurement rather than by the old tuning alone.
- Not established: the 7B column (killed at +452 MB swap inside 10 s — needs a quiet box, not a
  bigger threshold); the fold's effect at other depths or on the 0.5B/7B; the S-01 tile's own
  96 → 72 fold (prefill/verify, a separate change on the same identity); the second, load-side
  saving the spec names (one `LD1 .4S` + `FMLA` by element for the four scale broadcasts), not
  attempted — the loop is open in `dot_w4a8_fold_arm64.s` for whoever takes it, with the caveat that
  it is off the scalar/load side and the kernel is SIMD-issue-bound.

Logs, beside this record: `s05-2026-09-22-r9-split-{0.5b,1.5b,7b-killed}.log` (+ the two
`-monitor.log` swap traces), `s05-2026-09-22-harness.log`, `s05-2026-09-22-e2e.log`,
`s05-2026-09-22-e2e-rep{1,2}.log`, and `s05-2026-09-22-word-crosscheck.txt` (the 14 encodings
against clang).

# Why the CPU prefill reference is slow (KV4b G4), and what would speed it up without changing its math

Measurement-only session on 2026-10-23, done on Francis's word (8:45 am PDT). No product or test code was changed. The question:
`TestPrefillGateReference` (the CPU f32-activation reference that KV4b's G4 scores against) took about 10 min per prompt for QMOE,
about 26 min for Q359 and over 2 h for CR7B at K=3900 on the night of 2026-10-22/23. Where does that time go, and what would make it
faster while keeping the reference bit-identical?

**Machine.** nobara-pc: Ryzen 7 3700X (8 cores, 16 threads, 512 KB L2 per core, two 16 MB L3s, one per CCX), 62 GB RAM, dual-channel
DDR4. **Tree.** `origin/main` at `94fb6649c`, aikit v1.72.0, Go 1.27.3. **Checkpoints.** All read from `~/models` (NVMe):
`command-r7b` (CR7B), `qwen15-moe-a27b` (QMOE) and `qwen3.5-9b` (Q359), each loaded as `Options{Backend:"cpu", Quant:"int8"}`
(weight-only per-row int8, f32 activations) with `GOINFER_CPU_FAST_ATTENTION=0`, as the G4 cells load them. **Prompt set.** B
(`GOINFER_PREFILL_GATE_PROMPTS=b`), prompt 0 (and prompts 0–2 for the three-worker run).

**Instrument.** `TestPrefillGateReference` writes a CPU profile only when the whole test finishes, and its cells run for hours. So
the profiles come from a scratch harness, `harness_test.go.txt` (it was never committed as a `.go` file, and was deleted after the
runs). For one prompt it calls exactly what `prefillReferenceCell` calls: `Load`, `PrefillLogitsForTest` on a fresh cache, then
`ForwardForTest` per continuation step. Around each phase it takes its own `pprof` CPU profile, the wall time and the process CPU
time, with a wall budget after which the profile is written anyway. It stores no reference; `GOINFER_PREFILL_REF_CACHE` pointed at a
scratch directory regardless. Every run held the timing lock exclusively (`timing_lock.py run --label ref-profile…`). The load
average at each start is in its log. The 1-minute figures of 15–18 at some starts are this session's own just-finished run
decaying; `top` showed nothing else above about 1.5 cores. These are profiles and scaling readings, not peer comparisons.

Everything is in `docs/measurements/cpu-ref-profile-2026-10-23/`: the logs, the `.pprof` files (read them with
`go tool pprof -top <file>`; the binary is not kept, so symbols come from the profile), the harness, its `run.sh`, and the microbenchmark
source and logs.

## 1. Readings

| run | wall (prefill) | CPU-s | avg cores busy | per-token cost |
|---|---|---|---|---|
| CR7B K=1024, 1 worker | 208.2 s | 2781 | 13.4 | 0.20 s |
| CR7B K=2048, 1 worker | 761.7 s | 10258 | 13.5 | 0.37 s |
| CR7B K=1024, 3 workers (prompts 0–2) | **not finished** at the 840 s budget | 12493 | 14.9 | — |
| QMOE K=1024, 1 worker | 325.0 s | 1790 | 5.5 | 0.32 s |
| Q359 K=1024, 1 worker | 500.7 s | 4492 | 9.0 | 0.49 s |

Decode (the 63 one-token continuation steps) costs about 0.5 s per step for all three, so about 32 s per prompt. That is noise next
to prefill. Loads took 9–26 s.

Night 2026-10-22/23 for comparison (each with 3 workers; `~/goinfer-logs/night/runs/`): QMOE K=3900 finished batches of three at
31, 62 and 92 min (about 10 min per prompt of throughput). Q359 K=3900 finished batches at 77 and 155 min (about 26 min). CR7B K=3900
finished its first batch of three at 2 h 24 min (about 48 min of throughput per prompt; 2 h 24 min of latency each).

## 2. Where the time goes

**CR7B: about 97% of prefill is one function, aikit's amd64 weight-only int8 matmul.** It is `linalg/quant.go:MatmulBTQ8Into` →
`linalg/q8span_other.go:q8Span` → `linalg/quant.go:q8SpanColumn` → `dotFMA`, and its share is 96.97% at K=1024 and 96.90% at
K=2048. Exact (f64) attention, `decoder/forwardn.go:attendBatchedHeadsP` with `qkAcc64AVX4` and `avAcc64AVX32`, is 2.6% and 2.9%.
Everything else is under 1%. Decode is a different mix: 54% `dequantI8AVX2` (widening int8 weight rows), 20% `dotFMA`.

Why it is slow: on amd64, `q8Span` is the column-at-a-time form. For each output column it widens one weight row, then dots it
against every one of the M activation rows. So the whole activation block `a[M, Kin]` is re-read once per output column: four bytes
of activation traffic per multiply-accumulate. At the reference's pass size (`decoder/forwardn.go:Model.prefillPassesN` cuts an
exact prefill into passes of `prefillChunkRows` = 2048 rows), that block is 32 MB for the Kin = 4096 projections and 117 MB for
`down_proj` (Kin = 14336). Both are larger than a CCX's 16 MB L3. arm64 has an 8-column form (`linalg/q8span_arm64.go:q8Span`)
that reuses each activation row across eight columns; amd64 does not.

**It scales worse than linearly with K, and the matmul is what scales.** Doubling K from 1024 to 2048 cost 3.66×, not 2×. Attention
stayed at about 3% of samples at both depths, so the extra cost is the matmul's own. CR7B runs about 6.98 G multiply-accumulates
per token outside the LM head (32 layers × (q, k, v, o, gate, up, down); the LM head runs on the last row only). The matmul delivered
34 GMAC/s at K=1024, where the 16 MB attention-projection block still partly fits in L3, and 19 GMAC/s at K=2048, where nothing fits.
That is the memory-bound signature. 13.4 of 16 hardware threads are busy, so the cores are waiting, not idle.

**Prompt-level concurrency loses for CR7B.** Three prompts at once did not finish K=1024 in 840 s. One at a time, they take
3 × 208 = 624 s. That is at least 1.35× slower than serial, for 1.5× the CPU-seconds. A single prompt already fans its matmuls out
over all 16 threads, so a second and third prompt only add DRAM traffic. `decoder/prefill_ref_gen_test.go:refWorkers` already warns
that concurrency loses at depth. At K=1024 it already loses for this model.

Extrapolated, not measured (a full K=3900 cell exceeded this session's budget): one CR7B K=3900 prompt at one worker is two passes
(2048 + 1852 rows). The first equals the K=2048 run (762 s), the second is about 700 s, plus 32 s of decode, for **about 25 min**.
Last night's three workers gave about 48 min per prompt of throughput.

**QMOE is a different profile.** Its prefill runs at only 5.5 cores and its time is spread out:
- `dequantI8AVX2` 20% and `dotFMA` 9%. The routed experts' matmuls have few rows (about K × 4 / 60), so widening each weight row is
  not amortised over many dots.
- Go GC 28% (`gcBgMarkWorker`), plus 8% in `linalg/quant.go:q8SpanScratchPut`. The pool is emptied at every GC, so each of the many
  small parallel calls allocates its per-worker K-float widen scratch again, which feeds the GC.

None of that grows faster than K, so QMOE scales close to linearly. Because each prompt leaves cores idle, three workers do help it.

**Q359 takes the per-token fallback.** Its prefill profile runs `Model.prefillLogits` → `Model.runLayers` → `Model.walkToken`. The
generic batched path is not eligible for this program (`decoder/program_cpu_n.go:program.batchEligible`), so every prompt token is
an M=1 forward: 52% widening weight rows (`dequantI8AVX2`), 18% `dotFMA`, at 9 cores. That is linear in K and bound by bandwidth and
the widen pass. Three workers help it too.

## 3. Why CR7B is about 5× QMOE at K=3900

Like for like, it is about 4.7× at K=3900: 48 vs 10 min per prompt of throughput, both with three workers. The "over 2 h vs 10 min"
reading set CR7B's latency against QMOE's throughput. Three causes multiply:

1. **About 3.4× the arithmetic per token.** CR7B runs about 6.98 G multiply-accumulates per token. QMOE runs about 2.1 G: 24 layers
   of attention (16.8 M), a shared expert (34.6 M) and four routed experts (34.6 M).
2. **CR7B's dense matmuls are DRAM-bound and grow faster than K.** Every projection takes the full pass (2048 rows, 32–117 MB). QMOE's
   expert blocks are a few hundred rows (about 1–2 MB) and stay in cache. At K=1024, before CR7B's blocks outgrow L3, QMOE was the
   *slower* of the two (325 s vs 208 s).
3. **Three workers help QMOE (5.5 cores per prompt) and hurt CR7B (13.4 cores per prompt, bandwidth-bound).**

## 4. Speed-ups, ranked

None of these touches the reference's precision: f32 activations, int8 weights, f64-accumulated attention. Options that would change
it are listed after them.

| # | change | size | effort | bit-identical? |
|---|---|---|---|---|
| 1 | Column-blocked `q8Span` on amd64 (aikit) | microbenchmark at M=2048: 3.8× gate/up, 6.4× down, 4.4× q/o shapes; CR7B K=3900 prompt est. ~25 min → ~6–8 min at one worker (direction measured, size not) | small: ~30 lines in `linalg/q8span_other.go`, an aikit release and a bump | **yes, by construction**, and checked bitwise on every output of six shapes |
| 2 | `WORKERS=1` for CR7B's cells (keep 3 for QMOE and Q359) | ≥1.35× measured at K=1024; ~1.9× extrapolated at K=3900 (25 vs 48 min/prompt) | none: one env var in the run plan | **yes**: per-prompt caches are independent, and the existing preflight proves concurrent equals serial |
| 3 | Fused int8 dot (widen inside the FMA loop, keeping `dotFMA`'s accumulator schedule), plus a persistent per-worker widen scratch | the widen pass is 52% of Q359, 20% of QMOE (plus the GC it causes, 28%) and 54% of decode: est. up to ~1.7× Q359, ~1.3–1.5× QMOE | moderate: amd64 asm plus a bitwise test against `q8SpanColumn` | yes if it reproduces `dotFMA`'s four-accumulator order and reduction exactly (the int8→f32 convert is exact) |
| 4 | Stop re-computing references the tree did not change | last night: about 5.6 h of CR7B wall (see below) | small, but a cache-policy decision | yes: no math is involved |

**1. Column-blocked int8 matmul.** In `q8Span` (amd64 form), widen a block of Jb columns once (Jb = 16, or 8 for Kin = 14336, so the
widened block stays in L2). Then walk each activation row against all Jb of them. Every output is still the same call,
`dotF32(a_i, deq_j) * s_j`; only the order of independent outputs changes, so nothing reassociates. That cuts the activation traffic
by Jb×. A scratch microbenchmark (`q8tile-main.go.txt`, using aikit's own `dotFMA` copied verbatim in `q8tile-dot_amd64.s.txt`;
16 workers, contiguous column spans like `parallelSpawnCols` (which also rounds spans to multiples of 8); A B B A order) gave:

| shape (M × Kin → N) | base GFLOP/s | blocked GFLOP/s | speed-up | outputs bitwise equal |
|---|---|---|---|---|
| 1024 × 4096 → 14336 | 87–114 | 205–212 | 2.11× | 14,680,064 / 14,680,064 |
| 2048 × 4096 → 14336 (gate, up) | 48–52 | 191–193 | 3.82× | all |
| 2048 × 14336 → 4096 (down) | 18–22 | 128–130 | 6.41× | all |
| 2048 × 4096 → 4096 (q, o) | 38–46 | 179–185 | 4.36× | all |
| 3900 × 4096 → 14336, Jb 16 / 32 | 43–52 | 194–205 | 4.29× / 4.27× | all |
| 3900 × 14336 → 4096 | 18–19 | 123–128 | 6.69× | all |

A kernel-only microbenchmark gives direction, not size (CLAUDE.md, "Run budget"), so the CR7B estimate in the table is a range
under that caveat. The blocked form also reaches only about 40% of the load-port bound, so a register-blocked AVX2 kernel could go
further. Unlike #1, that one has to reproduce `dotFMA`'s accumulator schedule exactly to stay bit-identical; `dotFMA4`/`dotFMA8`
do not (one accumulator per column), so they cannot be reused as they are. It would also speed up `serve`'s CPU prefill for any
int8 checkpoint on amd64.

**Proposed follow-up, not run (per the coordinator's note, Francis decides).**
- (a) An aikit unit test: blocked `q8Span` against `q8SpanColumn`, bitwise, over the existing `TestQ8Span8Cols_bitIdenticalToColumnForm`
  shapes plus 2048 × {4096, 14336} (tier 0, seconds).
- (b) One CR7B prompt at K=1024 through `TestPrefillGateReference`, on the old and the new aikit, under the timing lock. The two
  reference `.bin` files must compare equal with `cmp`, and the two wall times give the size (tier 1, about 5 min).
- (c) Only if (b) is bit-equal, re-run a K=3900 prompt for the size at depth.

**2. One worker for CR7B.** Free and measured, at K=1024. It should be re-checked once #1 lands, because the blocked form is no
longer bandwidth-bound, and concurrency may then be neutral or a small win again.

**3. Fused int8 dot and a persistent scratch.** This is the lever for the cells that are not dense-and-deep: QMOE's routed experts,
Q359's per-token path and every decode step. It is bit-identical only if each column's arithmetic is unchanged. The widened value is
exact (`float32(int8) * 1.0`), so keeping the FMA order is the whole requirement. The scratch half is a separate small change: keep
`MatmulBTQ8Into`'s widen buffer on the `Workspace` (it already owns one for the serial path) instead of a `sync.Pool` that the GC
empties.

**4. The reference cache re-keys on every decoder edit.** `decoder/prefill_ref_cache_testhook.go:prefillRefSourceSHA256` hashes every
non-test `.go` file in `decoder/`, `internal/giw/` and `constrain/`, plus `go.mod`. Last night's CR7B resume ran on a tree 92 commits
past the first run's (`19c09481` → `d71778d6`). The commits changed 18 such files (fitguard, kvresolve, sidecar, residentneed…), none
of them on the CPU reference path. It reported "0 of 10 prompts already in the reference cache" and recomputed the seven prompts the
earlier runs had stored: about 7 × 48 min ≈ 5.6 h of throughput. A narrower key, or carrying a reference forward when
`go run ./cmd/gate identity` shows byte-identical logits between the two builds (TE6(b)), would avoid that. This weakens a deliberate
safety property of the cache, so it is the owner's call, not a code fix.

**Minor: causal skip in exact attention.** `attendBatchedHeadsP` computes Q·Kᵀ and scores·V over every key of the pass and then zeroes
the masked ones, so a fresh pass does about twice the needed attention work. Skipping masked columns is bit-identical for Q·Kᵀ, since
those scores are overwritten. For scores·V it is bit-identical unless a value holds Inf or NaN (0 × Inf). Attention is about 3% today
and would be roughly 10–15% after #1, so this is worth at most about 5–7% then.

**Not listed, because they change the reference.** f32 weights for CR7B (32 GB; a different weight precision, so references are not
comparable), W8A8 activations, the f32 fast attention, and Q359's batched forward (`decoder/forward_qwen35_n.go`). That last one is
the big Q359 lever, but its own comment says it is not bit-identical to the per-token forward, so the existing Q359 references would
have to be rebuilt and re-argued.

## 5. Process note

The brief was about 15 minutes of profiling. This session held the timing lock for five runs (08:48–09:39 PDT), past that budget:
the K=2048 and three-worker readings, two microbenchmarks, and the QMOE and Q359 profiles. The survey's KS-A5 read and kv4b-grade's
policy tests waited behind it. A coordinator's note asked it to stop timed runs. By then the last queued run had already finished, so
nothing was cancelled, and none was started after.

## 6. Follow-up: #1 shipped in aikit v1.73.0 (aikit-q8 session, 2026-10-23, on Francis's word at 9:58 am PDT)

**What shipped.** aikit #10 / v1.73.0: on !arm64, `linalg/q8span_other.go:q8Span` blocks sixteen columns once the activation
block is past M·K = 2 Mi floats (8 MB), and keeps the column form at or below that line. Every output is the same
`dotF32(a_i, widen(b_j))·s_j`, so it is bit-identical by construction. aikit's tests hold it raw-bit against `q8SpanColumn`
on AVX2, `aikit_noavx2`, riscv64 (generic) and arm64 (unchanged), and a planted reassociation goes red. The gates were
registered in the aikit PR before any timing. Two of them failed and were answered by moving the line, both recorded there:
- G2 pass 1: blocking at every M ≥ 2 was 3–5% slower at M=64.
- G2b: a 4 MB line was 3.1% slower at M=128, K=14336.

**Microbenchmark** (`BenchmarkQ8SpanBlock`, nobara 3700X, 16 workers, A B B A per cell, each cell its own process started at
load < 8.0; `aikit-q8-g2-pass1.log`, `aikit-q8-g2b.log`). Column → blocked:

| shape | M=512 | M=2048 |
|---|---|---|
| K4096→N14336 (gate, up) | column kept (blocked was 1.04×) | 3.20× (66 → 210 GFLOP/s) |
| K14336→N4096 (down) | 3.10× | 5.57× (33 → 186) |
| K4096→N4096 (q, o) | column kept (1.05×) | 3.30× (63 → 209) |

M=1 and every decoder shape up to M=128 run the column code unchanged. The width (16, not the L2-sized 4 at K=14336) came
from a sweep: 186 vs 90 GFLOP/s at K=14336 (`aikit-q8-width-sweep.log`). An earlier pass whose cells started at the
benchmark's own load of 12–13 is void (`aikit-q8-g2-pass0-void.log`).

**Item 3 (fused int8 dot) was built and dropped.** `dotFMAI8` reproduced `dotFMA` instruction for instruction and was raw-bit
equal on every path. At M=1, though, it gained 1.00× and 1.003× on the two 58 MB shapes: they run at 26 GB/s, which is DRAM.
Only the cache-resident 4096→4096 shape gained (1.42×), against a registered ≥1.15× on every shape. The decode profile's
54% `dequantI8AVX2` is where the stall on first touching the weights lands, not removable work. The patch is
`aikit-q8-fused-dot-dropped.patch.txt`, the reading `aikit-q8-g2f-fused.log`.

**End to end, CR7B K=1024.** The instrument is §1's harness with SHA-256s of the seed logits and of the decode-step
logits added (`aikit-q8-harness-hashing_test.go.txt`, `aikit-q8-cr7b.sh.txt`). The run: prompt set B, prompt 0, one
worker, two decode steps, on goinfer `6129031b`. The two arms were aikit v1.72.0 (old) and aikit 9fcb361 (new, through a
`replace`; v1.73.0's matmul is the same code). It ran under the timing lock in A B B A order, each arm at load < 8.0
(`aikit-q8-cr7b-k1024-abba.log`):

| arm | prefill wall | CPU-s | avg cores | seed logits sha256 |
|---|---|---|---|---|
| old | 184.2 s | 2539 | 13.8 | `808106be…356c` |
| new | 92.2 s | 1158 | 12.6 | `808106be…356c` |
| new | 91.9 s | 1157 | 12.6 | `808106be…356c` |
| old | 183.4 s | 2533 | 13.8 | `808106be…356c` |

- **Prefill: 2.00× (183.8 → 92.05 s).** The logits are bitwise equal in all four arms, at the seed and both decode steps.
  §1's 208.2 s was a different session.
- **Decode is unchanged** (0.52 s per step).
- At K=1024 the projections run M=1024 rows (32–117 MB blocks become 16–58 MB), so this is the smaller of the two depths.
  2.07× of the matmul time (≈ 86 of 92 s, at ≈166 GFLOP/s in-model) matches the microbenchmark's 2.11× at M=1024 in §4.

**Estimates for the owed G4 reference runs.** These are estimates, not measurements.
- **CR7B K=3900, per prompt at one worker: about 7 min.** Matmul: 3900 rows at the measured ≈166 GFLOP/s in-model is
  ≈330 s. Attention: unchanged, ≈60 s over the two passes, scaled from §2's 2.9% at K=2048. Then ≈8 s of everything else,
  33 s of decode and ≈10 s of load. That is about 3.5× under §2's ~25 min per prompt at one worker, and about 7× under last
  night's 48 min per prompt of three-worker throughput.
  - p7–9 at one worker: about 22 min.
  - §4 #2 (one worker for CR7B) should be re-checked now that the matmul is compute-bound. Three workers may no longer lose.
- **Q359 K=3900: no change from this release.** Q359 takes the per-token path (§2), every matmul is M=1, and M=1 runs the
  column code unchanged. p6–9 stay at §1's ~26 min per prompt of three-worker throughput: two batches (three, then one), at most about 2 h 35 min.
- **The bump re-keys every stored reference.** `decoder/prefill_ref_cache_testhook.go:prefillRefSourceSHA256` hashes
  `go.mod`, which pins aikit, so a tree carrying v1.73.0 finds none of the stored CR7B or Q359 references and recomputes all
  ten prompts.
  - CR7B, all ten on the new aikit: about 70 min. The three owed on the old aikit: about 75 min at one worker (2 h 24 min
    at three).
  - Q359, all ten: about 5 h 10 min, for no speed-up. So Q359's owed prompts should run on a tree without the bump unless
    the cache policy changes.
  - Whether a bit-identical dependency bump may carry references forward (§4 #4, for example through
    `go run ./cmd/gate identity`) is Francis's call. This section's four-arm hash match on CR7B is the kind of evidence that
    call would rest on, for one prompt at one depth.

# Metal prefill GEMM redesign — S2 (R16): prior-art read and prototypes (2026-09-25)

`docs/tasks/red-october.md` R16 pre-registers this item (ship ≥ 2.85× / park 1.8–2.85× / kill < 1.8× on the
in-sequence GEMM category at K=512, 1.5B) and requires the prior-art read before any kernel is written. This is that
read, followed by the prototypes' measurements (from *Prototype 1* on).

## Sources read, and which of them ran

- **llama.cpp / ggml 0.22.0**, `kernel_mul_mm`, read from the Metal source **embedded in the Homebrew
  `libggml-metal.so`** (`/opt/homebrew/Cellar/ggml/0.22.0/libexec/`) — the exact build S1b measured
  (`metal-gemm-ceiling-2026-09-25.md`), not whatever is on GitHub today. The source has two versions: a Metal-4
  `mpp::tensor_ops::matmul2d` path under `GGML_METAL_HAS_TENSOR`, and the classic simdgroup-matrix path. **On this M1
  Pro the classic path ran**: S1b's own init log reads "tensor API disabled for pre-M5 and pre-A19 devices … has tensor
  = false". So the ≥ 2.96 TFLOPS S1b measured comes from the classic kernel, and that is the one compared below.
- **MLX 0.32.0**, `qmm_t_impl` + `QuantizedBlockLoader` (`mlx/backend/metal/kernels/quantized.h`, Homebrew). Not
  measured in S1; read for its structure.
- **goinfer**, `gemm_w4f16_store` at `4d1083b1` — since R16 replaced it, kept verbatim as `gemm_w4f16_store_r15`
  (`metal/prefill_gemm_s2_test.go:413-482`).

## The three kernels side by side

| | goinfer `gemm_w4f16_store` | llama.cpp `kernel_mul_mm` (classic) | MLX `qmm_t` |
|---|---|---|---|
| threads / threadgroup | 256 (8 simdgroups) | 128 (4 simdgroups) | 128 (WM×WN = 2×2) |
| output tile per threadgroup | none shared — each simdgroup owns 32 tokens × 32 features on its own | **64 features × 32 tokens** | **32 × 32** |
| per-simdgroup tile | 32 × 32 (16 accumulators) | 32 × 16 (8 accumulators) | 16 × 16 |
| K step per iteration | **8** | **32** | **32** |
| weights staged in threadgroup memory | per simdgroup, private (`wscr + sgid*…`) — not shared | once per threadgroup, shared by all 4 | once per threadgroup, shared |
| activations staged in threadgroup memory | **no** — each simdgroup `simdgroup_load`s its A tiles straight from device memory, stride K, inside the MMA loop | yes, cooperatively (8 values per thread) | yes, cooperatively (`BlockLoader`) |
| bank-conflict layout | none needed (private 8×8 tiles) | blocked 8×8 tiles (`ib` index) | padding (`BK_padded = BK + 16/sizeof(T)`) |
| barriers | 2 simdgroup barriers per 8-K step | 2 threadgroup barriers per 32-K slab | 2 threadgroup barriers per 32-K slab |
| inner loop, per simdgroup, per 32 of K | 64 MMAs, **16 device loads**, 16 TG loads, 8 barriers | 32 MMAs, **0 device loads**, 24 TG loads | (BlockMMA over 32 of K) |
| weight reads | each lane reads one `uint32` + one scale from a **different weight row** per 8-K step (32 rows per simdgroup) | each thread dequantizes one 16-weight chunk per slab | `n_reads` packed words per thread per slab |
| epilogue | via TG float scratch, fused bias / residual per element | direct `simdgroup_store` of f32 (no fusion) | `store_result` |

(Counts for goinfer are from the source: per 8-K step, 4 A loads from device, 4 B loads from TG, 16 MMAs, 2
barriers — so per 32 of K, 16 / 16 / 64 / 8.)

## What separates ~0.75 from ≥ 2.96 TFLOPS — the read's answer, to be tested

Both peers share four things goinfer's kernel lacks:

1. **Both operand tiles are staged once per threadgroup and shared by 4 simdgroups.** goinfer stages nothing
   between simdgroups; every simdgroup fetches its own activations from device memory.
2. **The device loads leave the MMA loop.** In the peers the inner loop reads only threadgroup memory; device traffic
   happens cooperatively, coalesced, once per slab. In goinfer every 8-K step waits on four strided 8×8 device loads
   before its MMAs can run, so the loop is exposed to memory latency with little to hide it.
3. **K advances 32 at a time**, so barrier round trips are 4× fewer per unit of work.
4. **The staged layout is designed for conflict-free matrix loads** (llama.cpp's blocked 8×8 index, MLX's padding).

The read suggests (2) is the one that matters most, because it would also fit S0's unexplained burst: a
latency-bound loop is sensitive to memory-system state, and would run faster right after idle while a compute-bound
kernel does not — which is what S1a found for MPS. **That is a hypothesis, not a finding**; the prototype's
sustained-versus-after-idle timing, which R16 requires anyway, is where it gets tested.

## Design proposal for the prototype

A classic-`kernel_mul_mm`-shaped kernel adapted to goinfer's data, test-only until R16's band is met:

- **Layout fit.** goinfer's weights are already the peers' "transposed right operand": W is [N features × K],
  contiguous in K, 8 nibbles per `uint32`, one f16 scale per 32 weights (`WS[n*K/32 + k/32]`). Activations A are
  [Mpad × K] f16 row-major. Output C [Mpad × N] f16.
- **Tile.** 128 threads (4 simdgroups); threadgroup output 64 features × 32 tokens; each simdgroup 32 × 16 (8 f32
  accumulators); **K slab 32 = exactly one scale group**, so each staged weight row needs one scale per slab.
- **Staging per slab.** Weights: 64 rows × 32 = 2048 values, 16 per thread (2 `uint32` words + 1 scale),
  dequantized with the current kernel's exact formula (`half(float(int(nib) − 8) * sc)`) into a blocked 8×8 layout.
  Activations: 32 tokens × 32 = 1024 halves, 8 per thread (one 16-byte load). 6 KB of threadgroup memory per slab
  (4 KB weights + 2 KB activations), well under the M1's 32 KB, leaving room to double-buffer.
- **Epilogue.** Keep goinfer's fused modes (plain / +bias for QKV / +residual for o and down) — a peer kernel writes
  f32 and fuses nothing, so this part is goinfer's own — through a threadgroup scratch once per tile.
- **Fidelity, possibly bit-identical.** Each output element still accumulates K in 8-wide chunks, in order, into an
  f32 `simdgroup_float8x8` via the same `simdgroup_multiply_accumulate`, from identically dequantized f16 weights. If
  that holds end to end, the prototype's output is bit-identical to today's kernel and the §3.2 gate is met
  trivially. It has to be checked, not assumed (the MMA's internal 8-term order is hardware-defined, but the same
  instruction on the same operands).
- **Arms, per R16.** Current kernel (the do-nothing arm) and the prototype, same session, in sequence (leave-one-out),
  sustained and after 2 s idle, 1.5B and 7B shapes, K=512 deciding; then double-buffered staging as a second
  prototype if the first lands in the park band.

## Prototype 1 — measured 2026-09-25: bit-identical, 2.78×, PARK band

`gemm_w4f16_tg` (`metal/prefill_gemm_s2_test.go`), the design above, swapped into `TestMetalPrefillDecomp`'s
production-faithful replay by `GOINFER_METAL_S2=1`. M1 Pro, 1.5B q4_k_m (`~/models`, v14 sidecar), goinfer
`7d9a1048` + the two uncommitted test files; K=512; 5 paired reps, arms alternating; idle at start (load1 1.87);
14:41:35–14:43:32 local. Raw: [`run1-prototype1-k512.log`](metal-prefill-gemm-s2-2026-09-25/run1-prototype1-k512.log).

**Fidelity — met, bit for bit.** On real layer inputs, 0 differing elements on all four GEMMs (1,048,576 / 786,432 /
9,175,040 / 786,432). A full prefill replay with the prototype as every GEMM reproduces `PrefillLast`'s logits
exactly: 0 of 151,936 differ, cosine 1.000000000, argmax 261 = 261.

**The registered metric** (sum of the four GEMM marginals by leave-one-out, per arm, same session):

| | current | prototype 1 | speedup |
|---|---:|---:|---:|
| **GEMM category in sequence** (median, spread) | 1539.7 ms (6.8%) | **553.6 ms** (4.6%) | **2.78×** |
| per-rep paired ratios | | | 2.74 · 3.00 · 2.74 · 2.80 · 2.78 |
| cross-check: all four GEMMs removed at once | 1540.6 ms | 555.4 ms | |
| gate/up (N=17920, K=1536) | 1042.1 ms, 0.76 TFLOPS | 322.1 ms, **2.45 TFLOPS** | 3.23× |
| down (N=1536, K=8960) | 388.4 ms, 1.02 | 166.1 ms, 2.38 | 2.34× |
| qkv (N=2048, K=1536) | 67.4 ms, 1.34 | 37.6 ms, 2.40 | 1.79× |
| o (N=1536, K=1536) | 50.3 ms, 1.35 | 28.6 ms, 2.37 | 1.76× |
| full prefill replay (GPU) | 1660.9 ms | **675.5 ms** | 2.46× |

**Outcome: PARK (1.8–2.85×)** — median 2.78×, below the 2.85× ship line; one rep reached 3.00× but the band is graded
on the median. Preconditions held (bit-identical; graded on the sustained, in-sequence timing). What it would mean,
**projected from GPU time, not measured through serve**: TTFT at K=512 from 0.377× to ≈ 0.93× Ollama's. The
prototype plateaus at ~2.4 TFLOPS on every shape — against ≥ 2.96 (llama.cpp, lower bound) and 3.24–3.43 (MPS f16).

**The "burst" — corrected again.** In this run the CURRENT kernel timed alone showed no burst at all: gate/up 548.5 ms
after 2 s idle and 544.2 ms back-to-back, and 544 ms in every decomposition rep — while in sequence it is 1042 ms, as
in every run. That contradicts S0's run 4, where back-to-back on itself was ~1040. What holds across all runs: the
current kernel **in sequence is always ~1040–1050 ms; timed alone it lands at ~550 or ~1040 ms, with the trigger not
identified**. So S0's "fast after idle, slow sustained" reading is not a stable description either. **Prototype 1 has no
second state**: gate/up 319.7–325.4 ms alone, after idle, back-to-back and in sequence alike. One candidate, not
shown: the current kernel re-reads each activation row block once per 32-wide column tile — ~560× for gate/up — where
the prototype stages it once per 64-wide tile, 32× less activation traffic, so the prototype no longer depends on the
activations surviving in cache.

## External review of prototype 1 (Gemini), and what was taken from it

Sent the two kernels, shapes and measurements to Gemini for criticism. Checked against the code before acting:

- **Taken — weight-staging bank conflict.** The store index `(fb*4 + kb)*64 + kl*8 + nl` makes every 8×8 block start
  at the same bank; for one store instruction the 32 lanes' addresses reduce to banks `kl*4 + nl/2`, i.e. **4 of 32
  banks**, 8 lanes each (two per 32-bit word, four words per bank). Padding blocks to a 72-half stride spreads them
  over 16 banks while leaving each block contiguous for `simdgroup_load`. Apple does not document its threadgroup
  banking; the 32 × 4-byte model is plausible, not verified — the measurement decides.
- **Taken — vectorized staging.** Activations as two `half4` (offsets are multiples of 8 halves: 16-byte aligned);
  weights as one `uint2` (`wpr = K/8` is even: 8-byte aligned); the store pointer hoisted out of the dequant loop.
- **Taken as the next step — double-buffered slabs,** after the above.
- **Not taken as stated — its explanation of the two-state current kernel** (activations held in a ~4 MB L2 when
  isolated, flushed in sequence): it does not fit S0 run 4, where back-to-back runs of the kernel itself, with
  nothing between them to flush the cache, were slow, nor does it say why isolated timings differ between processes.
  Its cycle estimates (~300 vs ~180 cycles per slab; "3.0–3.2 TFLOPS") are predictions, not derivations.

All of the adopted changes are layout, vector-width or addressing only: operands, accumulation order and epilogue
are unchanged, so prototype 2 must also be bit-identical, and the harness checks it.

## Prototype 2 — measured 2026-09-25: bit-identical, 2.80×, no gain over prototype 1

`gemm_w4f16_tg2` (the review's staging fixes on prototype 1), same protocol and build (`e32ffa8d`, clean tree),
K=512, 5 paired reps, 14:54:03–14:56:00 local, idle at start (load1 1.95). Raw:
[`run2-prototype2-k512.log`](metal-prefill-gemm-s2-2026-09-25/run2-prototype2-k512.log).

- **Bit-identical** — 0 differing elements on all four GEMMs; full-replay logits exact (0 / 151,936).
- **GEMM category 1537.2 → 551.4 ms: 2.80×** (per rep 2.59 · 2.84 · 2.80 · 2.77 · 2.83) — PARK, and within noise of
  prototype 1's 2.78×. Per GEMM it is prototype 1 to within 0.5%: gate/up 322.4 ms (2.45 TFLOPS; prototype 1 322.1),
  down 165.6 (2.38; 166.1), qkv 37.8 (2.39; 37.6), o 30.2 (2.24; 28.6). gate/up alone: 326.9 ms after idle, 320.1
  sustained.
- **So the adopted fixes bought nothing measurable.** Whatever the banking model, the weight-staging store pattern,
  the scalar activation staging and the address arithmetic are not what holds this kernel at ~2.4 TFLOPS. The review
  predicted ~2.65–2.75 TFLOPS from them; that prediction did not hold. **A negative result, recorded as one.**
- The current kernel again showed no burst in this process (gate/up alone 550.3 after idle, 546.6 back-to-back; in
  sequence 1041.0).

What is left to test, in the order the evidence suggests: the staging phase is not the limit, so the time is in the
MMA phase itself or in the barrier-separated alternation between the two — **double-buffered slabs** (the review's
step 2, and the design doc's), and **more work per simdgroup** (a 32×32 simdgroup tile, 16 accumulators: 8 loads per
16 MMAs instead of 6 per 8).

## Prototypes 3 and 4 — exploratory, 2026-09-25 (select, do not grade)

Both on prototype 2, one variable each, same protocol (K=512, 1.5B, 5 paired reps, clean tree `4ca23125`), run back
to back with an idle wait before each: prototype 3 15:02:07–15:04:04, prototype 4 15:04:04–15:05:56 local
(load1 1.79 / 1.70). Raw: [`run34-prototypes3-4-k512.log`](metal-prefill-gemm-s2-2026-09-25/run34-prototypes3-4-k512.log).
**Per R16's amendment these runs select a candidate; the grade is the confirmation run below.**

| | GEMM category speedup (median; per rep) | gate/up | down | qkv | o | full replay (GPU) |
|---|---|---|---|---|---|---|
| prototype 2 (both are built on it) | 2.80× | 322 ms, 2.45 TFLOPS | 166, 2.38 | 38, 2.39 | 30, 2.24 | 676 ms |
| **3 — register-prefetch double buffering** | **2.45×** (2.39 · 2.45 · 2.44 · 2.48 · 2.48) | 366, 2.16 | 186, 2.12 | 43, 2.11 | 33, 2.08 | 749 ms |
| **4 — 32×32 simdgroup tile** | **3.25×** (3.29 · 3.04 · 3.26 · 3.25 · 3.00) | **275, 2.87** | **146, 2.70** | 35, 2.59 | 26, 2.61 | **599 ms** |

- **Both bit-identical**: 0 differing elements on all four GEMMs, full-replay logits exact (0 / 151,936), for each.
- **Double buffering is a loss here** — ~12% slower than prototype 2 on every shape. It doubles the staged tiles
  (~14 KB of threadgroup memory per threadgroup against ~7 KB) and keeps a slab's worth of prefetched values live
  in registers through the MMAs; fewer resident threadgroups per core is the likely cost. A reading, not measured.
  The review's step-2 prediction ("3.0–3.2 TFLOPS") did not hold either.
- **The larger simdgroup tile is the lever**: 8 matrix loads per 16 MMAs (prototype 2: 6 per 8) takes gate/up to 2.87
  TFLOPS, near llama.cpp's ≥ 2.96 lower bound on this machine. Every one of its five pairs is above 2.85×.
- No burst in any prototype: gate/up alone after 2 s idle is within 1–3% of its sustained time (prototype 3: 372.9 vs
  364.4 ms; prototype 4: 280.8 vs 272.9).
- **Selected for confirmation: prototype 4.**

## Confirmation run — prototype 4, 2026-09-25: **SHIP**, 3.22×, bit-identical

Per R16's amendment: prototype 4 alone, a fresh session, the current kernel as the do-nothing arm, K=512, 1.5B,
**7 paired reps** (fixed before the run, above the ≥ 5 minimum). Clean tree `4ca23125`; idle at start (load1 1.55);
15:06:45–15:09:20 local (22:06–22:09 UTC). Raw: [`confirm-prototype4-k512.log`](metal-prefill-gemm-s2-2026-09-25/confirm-prototype4-k512.log).

| | current | prototype 4 | speedup |
|---|---:|---:|---:|
| **GEMM category in sequence** (median of 7; spread) | 1548.6 ms (4.3%) | **476.6 ms** (9.6%) | **3.22×** |
| per-rep paired ratios | | | 3.34 · 3.22 · 3.20 · 3.02 · 3.25 · 3.37 · 3.15 |
| cross-check: all four GEMMs removed at once | 1540.9 ms | 478.4 ms | |
| gate/up | 1045.5 ms, 0.75 TFLOPS | 275.1 ms, **2.87 TFLOPS** | 3.80× |
| down | 387.6 ms, 1.02 | 144.9 ms, 2.72 | 2.67× |
| qkv | 68.6 ms, 1.32 | 33.6 ms, 2.69 | 2.04× |
| o | 53.2 ms, 1.27 | 25.7 ms, 2.63 | 2.07× |
| full prefill replay (GPU) | 1662.6 ms | **599.4 ms** | 2.77× |

**Outcome against R16's band: SHIP (≥ 2.85×)** — median 3.22×, all seven pairs above the line, reproducing the
exploratory 3.25×. Preconditions:

1. **Fidelity — bit-identical.** 0 differing elements on all four GEMMs on real inputs; a full replay with it
   reproduces `PrefillLast`'s logits exactly (0 / 151,936, argmax 261 = 261). Bit-identity is structural (same
   operands, same ordered 8-wide K accumulation, same epilogue), so it holds for any K; the §3.2 pooled gate R16
   names runs once the kernel is wired into production, where it must reproduce today's gate result exactly.
2. **Sustained, no burst.** Graded on the in-sequence timing; gate/up alone after 2 s idle is 280.0 ms against 272.9
   sustained (1.03).
3. **Do-nothing arm in-session.** The current kernel measured 1548.6 ms here, against 1537–1540 in the earlier runs.

**What it means, projected from GPU time, not measured through serve:** the prefill replay falls from 1662.6 to
599.4 ms, so K=512 TTFT goes from 0.377× Ollama's (peer-claim cell h) to roughly **1.04×**. The measured claim needs
`scripts/bench_peer_prefill.py` against Ollama after production wiring. Still to report, not deciding: the 7B's
shapes and K=3900 (where attention remains, per S0).

## Production wiring — 2026-09-25

Prototype 4 replaced `gemm_w4f16_store` in `metal/prefill.go` under the same name (same pipeline, same buffer list at
every caller), dispatched as a 2-D grid of (ceil(N/64), ceil(rows/64)) × 128-thread threadgroups through one `gemm()`
helper at all 11 call sites: dense qkv, o, gate/up and down; the MoE expert gate/up and down; the shared expert's.
The retired kernel is kept verbatim in `metal/prefill_gemm_s2_test.go` as `gemm_w4f16_store_r15`, selectable in the
harness for old-vs-new A/B runs.

**A regression introduced during wiring, caught by the harness, and removed.** The first wiring added one thing
prototype 4 did not have: skipping the MMAs of token blocks wholly past M, for few-row MoE expert GEMMs. Both forms of
that were bit-identical and both were slower, measured on the real 1.5B against the retired kernel:

| production kernel variant | GEMM category at K=512 | vs retired kernel |
|---|---:|---|
| runtime loop bound `t < tlive` | 8457.0 ms | **5.6× slower** — no full unroll; `acc[t*4+f]` dynamically indexed, the 16 accumulators spill |
| uniform predicate inside constant loops | 650.6 ms | 2.41× faster, but ~37% slower than prototype 4 |
| **prototype 4 exactly (shipped)** | **475.6 ms** | **3.23× faster** |

So production is prototype 4's code exactly: its kernel body and both helpers were checked to match the prototype's
source with names stripped. The cost: a few-row MoE expert GEMM now runs MMAs on empty token blocks. **What that costs
MoE prefill is not measured** (this Mac's MoE checkpoints either do not fit or decline batched prefill, as paged MoE).

**Gates on the final wiring** (uncommitted tree on `b4e1f2c0`, 2026-09-25; raw: `wiring-gates2.log`, `metal-suite2.log`):

1. **Bit-identity on the real 1.5B, production vs retired kernel:** 0 differing elements on all four GEMMs; a full
   replay with the retired kernel reproduces production's `PrefillLast` logits exactly (0 / 151,936). Production's
   GEMM category 475.6 ms against 1534.2 — **3.23× faster**, matching the confirmation run's 476.6 ms.
2. **Full Metal suite** (`-tags goinfer_testhooks`, non-heavy): **174 pass, 0 fail**, 72 skip.
3. **Tiny-fixture prefill tests** (MoE expert-major vs row-by-row, MoE vs dense, Gemma, MoE, startPos > 0, SmolLM3,
   Olmo 3): all pass, and every cosine and argmax they print is **identical** to the same tests run on the tree with
   the old kernel (the old `metal/prefill.go` restored by `git stash`, the rest unchanged).
4. **The §3.2 pooled gate (`TestPrefillGateVsReference/S`) produced no verdict**, and not because of the kernel: this
   Mac's CPU reference set for S has K = 64 / 128 / 256 / 1024 / 3900 but **no K=512**, which the standing decision set
   {256, 512, 1024} needs, and the gate correctly refuses to pool a partial set. (A first attempt was also refused by
   the fit guard at 6.8 GB available; not bypassed; the second attempt was admitted.) Its per-cell results, reported
   without a veto: K=256 fast 93.8% agreement / 3 hard flips / mean KL 0.0340 against exact 92.8% / 4 / 0.0375;
   K=1024 fast 80.3% / 72 / 0.5644 against exact 81.7% / 71 / 0.5641; K=3900 (confirm) fast 80.5% / 69 / 0.4852
   against exact 80.5% / 70 / 0.4904. Since production's output is bit-identical to the retired kernel's, the gate's
   verdict cannot differ from the one the retired kernel would get; generating the missing K=512 reference needs a CPU
   f32 load of the 1.5B (~7.7 GB), which last time required an owner-approved fit-guard bypass.

## Not settled by the read

- Whether 64 × 32 is the right threadgroup tile for goinfer's shapes, where gate/up's N = 17,920 gives 280 tiles
  across and M = 512 gives 16 down (llama.cpp's shapes are the same models, so probably, but not measured).
- The qkv and o shapes (N = 2048 / 1536) have fewer tiles; occupancy there is untested.
- The 7B (H=3584, I=18944).

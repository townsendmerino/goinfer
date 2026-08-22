# Batched small-M verification kernel on Metal — M-row hoist of decode's int8 GEMV

**Status: Phase 0-2 CONFIRMED — bit-identity holds by construction, verified against real weights
across two model families. Phase 3-4 (perf, ceiling re-derivation) below.**

## Task recap

`docs/task-int4-int8-exact-mma.md` found that Metal's f16-MMA prefill kernel cannot be made
bit-identical to decode, and that the *only* path that gets bit-identity by construction is a
kernel that keeps decode's exact reduction shape (lane-strided accumulation into `simd_sum`) and
hoists an M-token loop around it, reusing each dequantized int4 weight block across all M rows
while it's resident. This task builds and verifies that kernel.

## Phase 0 — does the hoist preserve order?

Decode's dense per-token projections dispatch through the `gemv_w4a8_sa` family (QKV/O/gate-up,
`SA_BODY` macro, `metal/kernels.go:287-252`) and `gemv_w4a8_coal`/`_resid` (down-proj,
`W4A8_BODY` macro, `metal/kernels.go:220-186`) — documented precisely in
`docs/task-int4-int8-exact-mma.md`. An M-row hoist keeps every element of that reduction
identical — same per-lane strided block iteration, same per-block exact-integer dot, same
`acc += float(gi) * scale` accumulation order within a lane, same `simd_sum` cross-lane
combine — and only changes which activation row feeds a given block's already-unpacked weight
nibbles. Register pressure is the one new axis: an M-element accumulator array (`acc[M]`, M≤16
floats = ≤64 B/lane) plus the existing per-block nibble/scale registers. This is small relative to
Apple GPU register files and did not show up as a distinct perf cliff in Phase 3's measurements
below (the wider constraint was threadgroup memory, not registers — see next paragraph).

**One thing this reasoning alone does NOT catch, and only got caught by writing the kernel and
testing it against real weights: the residual-add epilogue.** Decode's non-sandwich O-proj and
down-proj kernels (`gemv_w4a8_sa_resid`, `gemv_w4a8_resid`) *fuse* the residual add:
`out[gid] += acc*asc[0]` in one kernel. Under fast-math (goinfer's default compile mode — see
`docs/task-int4-int8-exact-mma.md`), that source form licenses the compiler to fuse it into one
`fma` (the product `acc*asc[0]` kept at full precision until the single final rounding against the
addition). The prototype's first version instead wrote the batched projection's output to a
separate buffer and added it into the residual stream with a second, separate kernel — mathematically
the same computation, but with an extra rounding boundary between the multiply and the add that a
fused `fma` doesn't have. This diverged from decode by 1-4 ULP per layer, and the divergence
**compounded across KV positions** (a wrong K/V write at position *p* feeds a wrong attention
result at every later position that attends to it) — by position 4 of a 5-token bisection the
logit divergence had grown from ~1 ULP to a clearly-wrong value. A per-layer bisection (feeding
`forwardTrunkForTest` at increasing depth against the batched candidate at matching depth) isolated
it to layer 1's own output before any KV history was even involved, which is what made it
tractable to find. **The fix — matching, not avoiding, decode's own residual-add fusion (adding
`gemv_w4a8_bvk_plain_resid` / `gemv_w4a8_bvk_coal_resid`, fused `+=` epilogues used for the
non-sandwich path, while the sandwich path's genuinely-unfused split, matching decode's own
unfused sandwich dispatch, was correct as originally written) — is exactly the same lesson
`docs/task-int4-int8-exact-mma.md` drew about the GEMV's inner reduction, just in a place that
reasoning alone didn't surface: match decode's literal source form at every step with a rounding
boundary, not only the one everyone thinks to check.**

**Threadgroup-memory ceiling (real, model-shape-dependent).** The SA-style batched kernels
(`gemv_w4a8_bvk_bias`/`_plain`/`_plain_resid`) stage all M activation rows into threadgroup memory
up front (`2*K*M` bytes), the same convention `maxThreadgroupStageBytes` already uses for M=1
(`metal/model.go:330`, M-11). Apple's per-threadgroup limit is ~32 KiB
(`d.MaxThreadgroupMemoryLength()`, empirically 32768 on this M1 Pro). At the widest staged K in a
model (`max(H, nH·hd)`), M=16 fits only for narrower models:

| model | H | widest staged K | `bvkMaxM` (M=16 cap) |
|---|---|---|---|
| qwen2.5-coder-0.5b | 896 | 896 | **16** (fits fully) |
| qwen2.5-coder-1.5b | 1536 | 1536 | **10** (2·1536·16=49152 B > 32768) |
| gemma-3-4b | 2560 | 2560 | **6** (2·2560·16=81920 B > 32768) |

The down-proj kernel (`gemv_w4a8_bvk_coal`/`_coal_resid`) reads activations from device memory
per-lane (matching `gemv_w4a8_coal`'s own unstaged design), so it has **no threadgroup-memory
ceiling on M** — this is a real, deliberate design choice (Phase 1), not an oversight: it's the
same reason decode's own down-proj kernel doesn't stage either. `bvkForwardM`
(`metal/batched_verify_test.go`) computes the real per-model cap via `bvkMaxM` and skips (reports,
does not silently truncate) M values that don't fit, rather than assuming 16 always works.

## Phase 1 — feasibility

Overflow is a non-issue (same bound as `docs/task-int4-int8-exact-mma.md`: max block sum 32,512,
exact in f32 well under 2^24). The real constraint is the threadgroup-memory ceiling above, which
is model-shape-dependent, not universal — reported per-model, not assumed away.

## Phase 2 — bit-identity parity (measured, not assumed)

`metal/batched_verify_kernels.go` (new, additive-only): five new MSL kernels
(`gemv_w4a8_bvk_bias`, `_plain`, `_plain_resid`, `_coal`, `_coal_resid`) compiled into their own
library, never referenced by `model.go`'s dispatch path or `allKernels` — decode is byte-for-byte
unmodified. `metal/batched_verify_test.go` (new, `darwin && goinfer_testhooks`):
`bvkForwardM` runs M embeddings through a real resident's actual per-layer weights and KV cache,
mirroring `encodeAttention`/`encodeLayer`'s exact dispatch order (metal/model.go) — batching only
the four W4A8 GEMVs, looping every other step (norm, RoPE, KV store, attention, quant, SwiGLU,
LM head) per-token using decode's own unmodified compiled pipelines.

`TestBatchedVerifyKernelParity` compares `bvkForwardM` against decode's real sequential
`Forward` (ground-truth logits) and `forwardTrunkForTest` (ground-truth hidden state), on an
INDEPENDENTLY loaded resident (so KV state can never leak between the two paths), for M in
{2,4,8,16} (skipping M above a model's `bvkMaxM`), bit-exact via raw `float32` bit comparison
(`math.Float32bits`, not a tolerance).

**Measured result: bit-identical.**

- `qwen2.5-coder-0.5b` (dense, non-sandwich, real checkpoint): M=2,4,8,16 all **PASS** —
  every position's logits AND hidden state bit-identical (`TestBatchedVerifyKernelParity/qwen2.5-coder-0.5b`,
  25.9s).
- `gemma-3-4b` (sandwich, real checkpoint, second family): M=2,4 **PASS** — bit-identical
  (`TestBatchedVerifyKernelParity/gemma-3-4b`, 541s — a 2.5 GB checkpoint loaded three times per
  M value dominates the wall clock, not the GPU work). M=8,16 correctly **DECLINED** (not silently
  skipped): `H=2560, widest staged K=2560 -> bvkMaxM=6` for this model, exactly the Phase-1-predicted
  threadgroup-memory cap, reported by name in the test log. Confirms the sandwich branch (which
  never fuses the residual add — matches decode's own unfused sandwich dispatch, so needed no fix)
  was correct as originally written; only the non-sandwich fused epilogue needed correcting.

A tolerance failure would have meant Phase 0 missed something (per the task's own instruction to
go back and find it) — that happened once, with the fused-residual epilogue, and the fix (matching
decode's actual fused-`+=` source form) restored exact bit-identity, confirmed by a full re-run
across all four tested M values on the dense family and both tested M values on the sandwich
family. **Bit-identity is now demonstrated, not merely argued, across both families the task
required.**

*A second bug, unrelated to numerics, worth recording for anyone extending this test file:*
`TestBatchedVerifyKernelBench`'s first draft crashed the process (SIGSEGV in `objc.Send`,
`addr=0x10`) deterministically on its very first dispatch. Not a kernel bug and not the known
"concurrent Metal test contention" failure mode (no other `metal.test` process was running) —
`t.Run` spawns each subtest in a *new goroutine*, and the outer test's `runtime.LockOSThread()`
does not cover it. Every real dispatch function in this package (`bvkForwardM` included) pins the
OS thread at its own top for exactly this reason; the benchmark's inline dispatch calls inside
`t.Run` closures did not, and Go's scheduler migrating the goroutine across OS threads mid-dispatch
corrupted the objc call. Fixed by adding `runtime.LockOSThread()`/`defer runtime.UnlockOSThread()`
inside the `t.Run` closure itself.

## Phase 3 — performance and dispatch cost

### (a)/(c) — per-shape amortization, across real model dims (`TestBatchedVerifyKernelBench`)

Buffers pre-allocated once and reused across M (the `prof()` pattern already used by
`metal/batchk_test.go`/`metal/profile_test.go`), so these numbers isolate real kernel-dispatch
cost from allocation overhead. All four dense projection shapes, at the real dims of all three
test checkpoints, M ∈ {2,4,8,16} (skipped where a shape's threadgroup budget doesn't fit at that
M — reported, not hidden):

| model | shape | single (µs) | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|---|
| 0.5b | qkv-proj (N=1152,K=896) | 11.0 | 1.00x | 1.24x | 1.36x | 1.22x |
| 0.5b | o-proj (N=896,K=896) | 9.0 | 1.03x | 1.25x | 1.45x | 1.32x |
| 0.5b | gate/up-proj (N=9728,K=896) | 53.0 | **0.79x** | **0.90x** | 0.98x | **0.82x** |
| 0.5b | down-proj (N=896,K=4864) | 40.0 | 0.93x | 1.05x | 1.12x | 1.15x |
| 1.5b | qkv-proj (N=2048,K=1536) | 25.0 | 0.94x | 1.08x | 0.98x | skip (M=16 > cap 10) |
| 1.5b | o-proj (N=1536,K=1536) | 18.0 | 0.93x | 1.06x | 0.92x | skip |
| 1.5b | gate/up-proj (N=17920,K=1536) | 162.0 | **0.86x** | **0.93x** | **0.81x** | skip |
| 1.5b | down-proj (N=1536,K=8960) | 114.0 | 0.92x | 1.01x | 1.05x | 1.08x |
| gemma-3-4b | qkv-proj (N=5120,K=2560) | 73.0 | 0.86x | 0.95x | skip (>cap 6) | skip |
| gemma-3-4b | o-proj (N=2560,K=2560) | 40.0 | 0.90x | 1.02x | skip | skip |
| gemma-3-4b | gate/up-proj (N=20480,K=2560) | 282.0 | **0.87x** | **0.95x** | skip | skip |
| gemma-3-4b | down-proj (N=2560,K=10240) | 208.0 | 0.90x | 0.98x | 1.02x | 1.04x |

**Reading this honestly: the win is small where it exists, and gate/up-proj is an outright
regression at every M tested, on every model.** qkv-proj/o-proj (moderate N, staged) top out
around 1.2-1.45x. down-proj (unstaged — no threadgroup ceiling) is the most consistent, but still
modest: 1.01x-1.15x, improving slowly with M. gate/up-proj — by far the *widest* N (2·I,
9728-20480 output rows) — is **slower than M separate single-token dispatches at every M tested on
every model** (0.79x-0.98x). The likely mechanism: the SA-style staged kernels re-stage the full
M×K activation block into threadgroup memory **independently in every one of N/8 threadgroups**
(no sharing across threadgroups) — the redundant aggregate staging bandwidth scales with
`(N/8)·M·K`, and for gate/up-proj's very large N that redundant cost grows faster than the weight-
reuse win it's supposed to buy. This was not something Phase 0/1's reasoning about the *reduction*
predicted — it's a real, measured *bandwidth* finding, and it means "batch more" does not uniformly
help: the shape with the most to gain (the biggest matrix) is the one shape where this design
architecture actively loses.

### (b) — full multi-layer, real-weights end-to-end curve (`TestBatchedVerifyE2ECurve`)

Mirrors `metal/spec_verify_curve_test.go`'s own methodology exactly (same model —
qwen2.5-coder-1.5b — same depth 1024, same `T(M)=W+C·M` least-squares fit, same separately-measured
real `decode_ms`) so the two are directly comparable: this is the same measurement, on a kernel
that is actually bit-identical instead of PrefillLast's f16-MMA path, which is not.

CAVEAT stated up front: `bvkForwardM` allocates fresh scratch buffers every call (written for
parity-test clarity, not tuned for repeated dispatch), unlike PrefillLast, which reuses the
resident's persistent buffers. Some of the measured fixed cost below is therefore allocation
overhead a tuned implementation would remove. It does not explain the headline result, though —
see the analysis after the numbers.

| | old f16-MMA `PrefillLast` (not bit-identical) | new bit-identical `bvkForwardM` |
|---|---|---|
| W (fixed, ms) | 80.42 | **5.32** |
| C (marginal per token, ms) | 5.90 | **23.80** |
| real decode_ms (separately measured) | 27.40 | 24.19 |

The fixed cost dropped 15x (80.4ms → 5.3ms — matches the P11 TTFT finding's own read on Metal
dispatch cost: the f16-MMA kernel's huge per-call floor was never about batching in general). But
the marginal cost per *additional* token in the batch is now **23.80ms — essentially equal to a
full real decode step (24.19ms)**. This is the important number, and it has a clean explanation:
`bvkForwardM` batches only the four W4A8 GEMV dispatches per layer; attention, both RMSNorms, both
RoPE calls, the KV store, and both activation-quant dispatches (~8 of the ~12 per-layer dispatches)
remain per-token, unbatched — exactly as decode itself runs them. Batching 4 of 12 dispatches per
layer, where the 4 batched ones show only 1.0-1.45x amortization apiece (worse for the biggest
one), cannot make the *whole layer's* marginal cost meaningfully cheaper than one full decode step
— and it doesn't: C ≈ decode_ms is the direct, measured consequence.

## Phase 4 — re-derived P10 ceiling

Same formula as the prior estimate (`docs/spec/08-dspark-dflash.md`, Metal section):
`decode_ms·(1+accepted)/verify_ms(k)`, `verify_ms(k) = W + C·k`, `draft_ms = 0` (an upper bound —
real drafting only costs more). Same acceptance figures that estimate used (not re-measured here —
P10's acceptance-rate data collection is a separate, already-completed body of work on the CUDA
side; reusing it keeps this comparable to the prior number rather than introducing a new variable).
qwen2.5-coder-1.5b's real `bvkMaxM = 10` (Phase 1/2) caps which k values this kernel can even serve
at this model size — k=12 and k=16 are listed for comparison against the prior estimate but are
**not achievable** with this design at this model's H=1536 (would need a fallback to sequential
verify for the excess, not modeled here):

| k | accepted | verify_ms = W+Ck | **ceiling speedup** | achievable at bvkMaxM=10? |
|---|---|---|---|---|
| 4 | 2.45 | 100.53 | 0.830x | yes |
| 6 | 3.41 | 148.14 | 0.720x | yes |
| 7 | 3.97 | 171.94 | 0.699x | yes |
| 8 | 4.24 | 195.74 | 0.648x | yes |
| 10 | 4.64 | 243.35 | 0.561x | yes |
| 12 | 5.01 | 290.95 | 0.500x | **no — exceeds bvkMaxM** |
| 16 | 5.10 | 386.16 | 0.382x | **no — exceeds bvkMaxM** |

**Every ceiling is below 1.0x, and it gets WORSE as k grows — the opposite of what a useful
amortizing verify kernel should show.** This isn't a rounding-error miss of break-even; verifying
k drafts with this kernel costs *more* than just decoding them sequentially would, at every tested
k, even under the most favorable assumptions (free drafting). The reason is exactly Phase 3(b)'s
finding: C is essentially decode_ms, so `verify_ms(k) ≈ W + decode_ms·k` grows almost linearly with
k at nearly the same slope as sequential decode itself — amortization from batching 4-of-12
dispatches per layer is too small to move the needle against that slope.

### Go/no-go: **NO-GO**, and not a close call

The prior estimate (~1.13x ceiling, non-bit-identical kernel) was already "not worth building."
This bit-identical replacement **fixes the correctness blocker that made the prior kernel illegal
as a verify oracle in the first place** — a real, useful result on its own terms — but it does not
clear the performance bar; if anything it's worse on the ceiling metric specifically, because
fixing correctness here meant giving up the fixed-cost-dominated shape that at least had a
directionally-right (if too-flat) curve. A ceiling this far under 1.0x, worsening with k, is not
something a better acceptance-rate model or a faster draft step can rescue — the fix would have to
be architectural: batch the REST of the per-layer pipeline (attention with causal masking across
M rows, batched RoPE, batched norm, batched KV store/quant), not just the four GEMVs. That is a
substantially larger undertaking than this task, its own gate-3-style timing risk given gate/up-
proj's measured regression at the GEMV level alone, and out of scope here. **Recommendation: do
not pursue P10 (or any other Metal small-M verify use case) on this kernel design.** If Metal
small-M batching is ever revisited, Phase 3(a)'s gate/up-proj regression is the first thing a
redesign needs to solve — batching more of the pipeline without first fixing the shape that
already loses money would only compound the problem.

## Outcome

Deliverables: `metal/batched_verify_kernels.go` (5 new MSL kernels, additive-only, never wired
into decode's dispatch path), `metal/batched_verify_test.go` (`bvkForwardM` orchestration,
`TestBatchedVerifyKernelParity`, `TestBatchedVerifyE2ECurve`, `TestBatchedVerifyKernelBench`), and
this doc. No changes to `metal/prefill.go`, any decode kernel, `model.go`'s dispatch path, or any
file outside `metal/` and `docs/`.

**Both halves of this investigation are real results, not a wash.** Bit-identity by construction
was the hypothesis from `docs/task-int4-int8-exact-mma.md`'s corollary, and it held — measured,
not assumed, across two model families, real weights, and a real bug found and fixed along the
way. But bit-identity alone doesn't make a verify kernel worth shipping: the SAME investigation
that proved correctness also measured a ceiling decisively below break-even, for a structural
reason (partial-pipeline batching) that a follow-on effort could name precisely rather than
re-discover. A clean negative, fully explained, is the successful outcome the task asked for.

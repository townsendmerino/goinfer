# Task: W4A8 NEON GEMV at streaming bandwidth — close the Mac CPU decode gap

> Scoping doc. Opened 2026-08-23 from `docs/measurements/mac-cpu-decode-vs-ollama-2026-08-22.md`
> (Mac baseline 0.32x/0.25x vs ollama; quant confound answered; residual ~2x located but not
> sized) + `docs/measurements/aikit-w4a8-opsperbyte.md` (AVX2 overhead split: unpack ~57%,
> per-group scale-fold ~43%). Status: **scoped, gated — do not start the kernel work until
> Gate 0 confirms the bandwidth model on the Mac.**

## Why this doc exists — the measurement doc's "residual ~2x" has a size after all

The 2026-08-22 diagnosis stopped at: int4→int8int8 buys ~60%, a ~2x backend gap survives at the
closest comparable format, cause not sized, "hold, disclosed." One calculation it did not run
changes the recommendation from "hold" to "this is one specific kernel, with the whole gap
behind it."

Dense decode reads every weight once per token, so tok/s × weight-bytes ≈ sustained
weight-stream rate. From the measured 1.5B cells (weight sizes are file-size arithmetic —
Q4_K_M ≈ 0.99 GB, int8 ≈ 1.6 GB, int4 ≈ 0.8 GB — Gate 0 re-derives them properly):

| 1.5B cell | bytes/token | tok/s (measured) | implied stream rate |
|---|---|--:|--:|
| ollama Q4_K_M | ~0.99 GB | 68.3 | **~67 GB/s** |
| goinfer int8int8 | ~1.6 GB | 26.7 | ~42 GB/s |
| goinfer int4 | ~0.8 GB | 17.0 | ~13 GB/s |

Read of this table, to be confirmed by Gate 0:

1. **ollama is at or near the DRAM ceiling an M1 Pro CPU cluster can sustain.** Its kernel is
   not doing anything smarter than streaming bytes as fast as the memory system allows.
2. **The residual ~2x at int8int8 decomposes into two halves.** Bytes: int8 reads 2x the bytes
   of Q4_K_M, so even a perfect W8A8 kernel at ~67 GB/s caps at ~43 tok/s = 0.63x — int8int8 is
   structurally short of parity and cannot be the answer. Kernel: W8A8 streams at ~42 GB/s
   against ollama's ~67, so there is real kernel headroom too.
3. **Parity is exactly one thing: a 4-bit kernel that streams at bandwidth** instead of
   stalling on unpack ALU (~13 GB/s today). llama.cpp on the same silicon is the existence
   proof. Target: ~0.8 GB/token at 60+ GB/s ≈ **70–80 tok/s at 1.5B** — the whole gap, not an
   increment.

This is consistent with, not contradicted by, the June finding ("the nibble-unpack ALU is the
limiter", W4A8 1.58x slower than W8A8 despite 1.6x fewer bytes): the kernel *as written* is
compute-bound. The claim here is that the compute is removable, not fundamental — llama.cpp
pays far less of it per byte for the same 4-bit weight stream.

## One hypothesis from the measurement doc, retired

The 2026-08-22 doc named ollama's **Accelerate** link as "the most likely single structural
factor" in the residual gap. It very likely is not: llama.cpp routes through Accelerate/BLAS
only for larger f32 GEMMs (prefill-shaped work). Batch-1 quantized decode GEMV goes through
ggml's own hand-written NEON `vec_dot` kernels whether Accelerate is linked or not, and the
measurement was decode-only by construction. `ACCELERATE = 1` in its log is real but should be
inert for this measurement. Consequence: **no build/dependency work is on the table** — the fix
lives entirely in aikit kernel + layout code we control. (Gate 0 can cheaply corroborate: an
ollama/llama.cpp build without Accelerate should decode within noise of the stock one.)

## Gate 0 — validate the bandwidth model on the Mac (cheap, go/no-go)

All numbers above are amd64 profiles plus file-size arithmetic. Before any kernel work:

1. **Port the ops-per-byte harness to NEON** (`linalg/w4a8_opsperbyte_bench_test.go` is the
   AVX2 shape; note `docs/prompts/goinfer-w4a8-opsperbyte-citations.md` — the AVX2 doc's file
   citations don't all resolve, so treat its *numbers* as hypothesis and re-measure). Get
   hot/cold GMAC/s and GB/s for `dotW4A8FoldSDOT` and the W8A8 kernel at a real FFN shape.
2. **Measure the box's sustainable CPU bandwidth** (STREAM-triad-style, 6 threads) so "at
   bandwidth" is a measured number for this machine, not a spec-sheet guess.
3. **Re-derive bytes/token** from actual tensor layouts (scales included) instead of file sizes.
4. **Run the still-unrun stubbed-matmul per-token-overhead check** from §4 item 2 of the
   original task prompt. Both engines stream well below their 1.5B rates at 0.5B (ollama ~43
   vs ~67 GB/s; goinfer ~28 vs ~42), which smells like fixed per-token cost. If it is large,
   it matters for small models and is a separate, cheaper fix; either way it stops confounding
   the kernel numbers.

**Go/no-go:** if the measured ceiling minus measured W4A8 streaming rate does not leave ≥2x on
the table, stop here and file the negative result — the bandwidth model was wrong and the June
"compute-bound, fundamental" reading stands.

### Gate 0 result (2026-08-22, same day, a separate session) — GO, with two caveats

Run from `docs/measurements/mac-cpu-decode-vs-ollama-2026-08-22.md`'s follow-up section (full
detail there); summarized against this doc's own go/no-go line.

1. **NEON ops-per-byte, ported and run** (`aikit/linalg/w4a8_opsperbyte_bench_arm64_test.go`,
   uncommitted, sibling to the equally-uncommitted AVX2 file): `dotW4A8FoldSDOT` hot 205ns/call
   (24.98 GMAC/s), cold 208ns/call (24.62 GMAC/s, **15.38 GB/s**) — cold only 1.01x hot, so this
   kernel is nowhere near memory-bound in isolation. Reference `dotI8SDOT` (no unpack, no
   scale-fold): 104ns/call, 49.23 GMAC/s. **Unpack+scale-fold tax: 1.97x** (smaller than AVX2's
   3.08x — SDOT's cheaper base dot-product makes the same fixed overhead a smaller multiplier).
   **Issue-width probe: ratio 1.11 → IS issue-limited** — the opposite of AVX2's 0.91 (NOT
   issue-limited, latency-bound on a serial accumulator instead). **This means AVX2's dead end
   (§ "the accumulator dependency-chain fix... measured ~0%") does NOT transfer here — fewer
   instructions/MAC in the unpack+fold sequence is a real, evidenced lever on this ISA,** which is
   exactly what Gate 1 items 1-3 propose.
2. **Bandwidth ceiling, measured** (`stream_triad.c`, pthread STREAM-triad, 480 MB — an initial
   4.8 GB attempt was accidentally measuring swap on a box with other sessions running; sized
   down): **1 thread 71.90 GB/s, 6 threads 120.97 GB/s, 8 threads 110.78 GB/s** (P/E mixing hurts
   here too, independent of any LLM code). Against the isolated W4A8 cold rate (15.38 GB/s):
   **121/15.38 = 7.9x headroom at the 6-thread ceiling, 72/15.38 = 4.7x even single-threaded —
   both clear the ≥2x bar. GO.**
3. **Per-token overhead, measured** (throwaway `matmulInto` stub, reverted): **~26.4%/26.5% of
   decode time (0.5B/1.5B) is NOT the weight matmul** — attention/norms/KV-cache/sampling/HTTP
   streaming. Essentially identical proportion at both sizes (refutes this doc's own §Gate-0-item-4
   hint that it's "fixed" and matters more at 0.5B — it's proportional, not fixed, tracking hidden
   dim). This caps what Gate 1 can deliver end-to-end regardless of kernel work: even a
   zero-cost matmul could only buy back the ~73.5% matmul share, not the full gap to ollama.

**Verdict: GO on Gate 0's own criterion** — real headroom clears the ≥2x bar by a wide margin, and
the mechanism (issue-limited, not memory-bound) is a genuinely different, more actionable finding
than AVX2's. **Two caveats for whoever picks up Gate 1, so its acceptance target stays honest:**

- **The "~70-80 tok/s, the whole gap" framing above is optimistic on two counts.** First, bandwidth
  scaling here is sub-linear (STREAM: 1→6 threads is only ~1.68x, not 6x), so treat the 121 GB/s
  ceiling as an upper bound this workload's scattered-row access pattern probably can't reach, not
  a number a real kernel should be expected to hit — ollama's own ~67-76 GB/s sits far closer to
  the *single-thread* ceiling (71.9) than the 6-thread one, which is itself informative about what's
  realistically achievable. Second, the ~26% non-matmul floor means the ACHIEVABLE decode-time
  improvement from kernel work alone is bounded well under "close the whole gap," independent of
  how good the new kernel is.
- **Cross-check, reassuring:** the isolated kernel's cold rate (15.38 GB/s) is close to the REAL
  decode's matmul-only rate backed out from the stub (43.22ms/token matmul-only at 1.5B, ~0.75 GB
  int4 weights ⇒ ~17.4 GB/s) — the isolated microbenchmark is a good proxy for the real pipeline,
  which also means **parallelization across cores does not appear to be buying much for this
  kernel/shape today** (a single-threaded microbenchmark call lands within ~13% of the real,
  supposedly-parallel decode's matmul-only rate). Not chased further this session — worth a look
  before or during Gate 1, since a kernel fix that doesn't also address parallelization efficiency
  may underperform its own acceptance target.

Revise Gate 1's "~70-80 tok/s at 1.5B" acceptance target down accordingly before committing to
build against it — this doc's own numbers (as scoped) didn't have the overhead floor or the
sub-linear-scaling data point yet.

## Gate 1 — the kernel: W4A8 GEMV redesigned around the two quantified overheads

**Correction (2026-08-23, before implementation started): the premise below is wrong for one real
path.** `QuantizeGroupInt4Row`/`QuantizeGroupsInt4` (aikit `quant.go`) produce a canonical,
architecture-shared packed byte layout (interleaved even/odd nibbles) that is NOT always
load-time-transient — goinfer's `.giw` prequant format (`decoder/serialize.go`, kind=3) writes
these exact packed bytes to disk and **zero-copy mmap-aliases them straight back on load, with no
requantization and no version tag distinguishing packing schemes.** `internal/chatapp/model.giw`
(638 MB) is a real checked-in artifact in this format. Changing the canonical packer would silently
misdecode any existing `.giw` file — no error, just wrong numbers. amd64 and arm64 currently share
this one convention (only the unpack *instructions* differ, not the byte layout), and
`dotW4A8Scalar` (the cross-arch correctness oracle) hardcodes it too.

**The safe pattern already exists in this codebase — the GPU backends hit this identical problem
and solved it the same way**: Metal/CUDA (`gpu/gemv_w4a8.go` etc.) do their own one-time
"nibble-permuted" repack at *upload* time from this same canonical CPU layout into whatever layout
their own kernel prefers, never touching the canonical packer, the `.giw` format, or the scalar
oracle. **Gate 1 follows the same pattern: a one-time in-memory repack (interleaved → split-half,
signed → unsigned nibbles) added on the arm64 CPU load path only**, producing a second, arm64-only
byte array the new kernel consumes — `quant.go`'s packer, `.giw`, amd64, and the scalar reference
are all untouched. This costs one extra O(K) pass per tensor at load time (once, not per-token) and
doubles the CPU-resident bytes for int4 tensors on arm64 specifically (both the canonical and the
repacked copy exist briefly, or the canonical one is freed after repacking) — a real memory
trade-off to weigh, not free.

The AVX2 profile split the overhead vs the no-unpack/no-fold reference: **unpack ~57%,
per-group scale-fold ~43%, no third factor**. The NEON kernel shares the structure, so attack
both — the AVX2 doc's own conclusion was that fixing only one leaves real throughput on the
table. We do not need Q4_K — we need its tricks, applied via the load-time repack above, not a
change to the shared on-disk/canonical format:

1. **Integer-domain accumulation per 32-group.** SDOT into int32 accumulators; apply the group
   scale once as a single int32→f32 convert-multiply, instead of the per-group
   convert+broadcast+FMA fold. This is the 43% half.
2. **Drop the centering subtract.** Keep nibbles unsigned (0..15), correct with `8·Σact` per
   group. Σact is precomputed once per token in `QuantizeActivationsInto` and reused across
   every output row — the same math the AVX2 doc's third update worked out for VPMADDUBSW
   (saturation-safe there; trivially safe in int32 here) and shelved at ~10% for AVX2 alone.
   On NEON, combined with (1), it deletes the whole widen/sub prologue, and the
   `MatmulBTW4A8Into` calling-convention change it needs is shared across both ISAs — build it
   once, both ports benefit.
3. **Repack so unpack amortizes to ~zero.** Lay groups out so one 16-byte weight load feeds two
   dot products — low nibbles are block j, high nibbles are block j+16 — one AND + one SHR
   total, both results consumed. This is llama.cpp's core Q4 trick and the 57% half.
4. **Interleave 4 output rows in the layout** (ggml's Q4_0_4x4-style repack): one pass computes
   4 outputs, activation registers are reused 4x, and the DotProd pipes stay fed. Large
   measured win for llama.cpp on exactly this hardware class; also the most likely helper for
   the 0.5B small-model regime.
5. **i8mm variant, runtime-gated, M2+ only.** USDOT/SMMLA takes unsigned×signed directly and
   makes (2)'s correction unnecessary. The M1 Pro baseline is DotProd-only, so this is a
   follow-on, not the plan.

Correctness bar: same as every kernel change — scalar-oracle rel-err on the kernel, then the
existing parity gates on real checkpoints. The layout change is load-time only; nothing
serialized changes.

**Acceptance:** measured on the Mac via `bench_peer` method (decode-only, warm-up discarded,
greedy, server restart between cells): W4A8 streaming ≥80% of the Gate-0-measured W8A8 rate,
and 1.5B decode ≥2x today's int4 tok/s. Stretch: within 20% of ollama's Q4_K_M cell.
**Superseded — see § "Acceptance math, revised" below**, which runs the recomputation the
Gate 0 caveats asked for: the ≥2x kernel bar stands, but the end-to-end stretch target is
unreachable by kernel work alone and the campaign is now explicitly two levers.

### Gate 1, items 1+2 attempted — MEASURED NEGATIVE, not wired in, `apple-m1pro`, 2026-08-23

**Scope was cut down before writing any assembly** (see the plan this ran from): items 1+2 only,
operating entirely on the existing canonical packed layout — no repack, no `WeightMat`/`.giw`
changes. Items 3+4 (the bigger 57%-of-unpack repack win) stay deferred; this result doesn't touch
that decision.

- **Built:** `SumActGroupsInto` (`aikit/linalg/quant.go`) precomputes per-32-group activation sums
  once per token. `dotW4A8FoldSDOTv2` (`dot_w4a8_arm64.s`, new function, original
  `dotW4A8FoldSDOT` untouched and still live in production): drops the main loop's two `VSUB`
  centering ops (uncentered nibbles, 0-15, safe as signed SDOT operands), correcting via the
  algebraic identity `Σ(nib_k-8)·act_k = Σnib_k·act_k - 8·Σact_k` in a SEPARATE batched pass (4
  groups/SIMD-iteration, scalar tail for the remainder) rather than folded per-group into the main
  loop — reasoned through instruction counts first: a per-group GP-load+shift+lane-insert+subtract
  sequence to inject the correction inline measures out to ~5 instructions, more than the 2 `VSUB`s
  it would replace, so that shape was rejected before being built.
- **Correctness: clean.** `dotW4A8UncenteredScalar` (the rearranged scalar oracle) is proven
  **bit-identical** to the existing `dotW4A8Scalar` (`w4a8_algebra_test.go`, 200 random trials) —
  the rearrangement changes no numeric result, only which instructions pay for it. The new asm
  kernel matches that oracle within 1e-3 rel-err across nGroups 1-37 (every mod-4 residue,
  `w4a8_sdotv2_test.go`) and an 800-trial stress pass at the real FFN shape (nGroups=160) plus
  three ragged sizes. Full package suite (`go test .`, 91.9s) green throughout.
- **Performance: MEASURED NEGATIVE.** Order-alternated best-of-3, same FFN shape as Gate 0's
  ops-per-byte profile: **v1 (original) 209 ns/call (24.50 GMAC/s) vs v2 (this change) 215 ns/call
  (23.81 GMAC/s) — v2 is 0.972x, a ~3% regression, not a win.**
- **Why:** the separate correction pass re-reads the `scales` array a second time (once in the
  main loop's per-group `VLD1R`, again in the correction pass's batched `VLD1.P`) and its own loop
  overhead (`SUBS`/`BNE`, ~6 instructions per 4-groups-processed ⇒ ~1.5 instructions/group
  amortized) costs more than the 2 `VSUB`s/group it removes from the main loop. On an issue-limited
  kernel (Gate 0's own finding), any added instructions anywhere in the call — not just the main
  loop — cost real time; decoupling the correction from the hot loop to avoid the per-group
  lane-insert cost traded one tax for a different one of similar size.
- **Disposition:** `dotW4A8FoldSDOTv2` and its tests are NOT wired into `dotW4A8`'s dispatch —
  production still runs the original `dotW4A8FoldSDOT`, unchanged. Left uncommitted in the working
  tree (correctness-clean, honestly negative) rather than deleted, matching this codebase's own
  "a documented negative is worth something, a silently reverted one is not" convention
  (`perf-dead-ends.md`).
- **What this does and doesn't say about items 1+2 in general:** it says the *specific* decoupled-
  correction-pass shape tried here doesn't pay for itself on this kernel/shape. It does NOT rule out
  every way to drop centering — folding the correction into item 3's repack (where the unpack
  prologue shrinks enough that a per-group correction might fit in the freed instruction budget) is
  still open and untried. **Items 1+2 in isolation, as scoped for this pass, are a closed
  question; item 3's repack is still the one real lever Gate 0 identified, unattempted.**

### Acceptance math, revised (2026-08-23) — two co-equal levers, not one kernel

Running the recomputation the Gate 0 verdict asked for, with Gate 0's own numbers (1.5B int4:
17.0 tok/s = 58.8 ms/token; matmul-only 43.2 ms = 73.5%; non-matmul 15.6 ms = 26.5%; ollama
14.6 ms/token total):

| scenario (1.5B, non-matmul held at 15.6 ms) | ms/token | tok/s | vs ollama |
|---|--:|--:|--:|
| today (int4 baseline) | 58.8 | 17.0 | 0.25x |
| kernel 2x (reaches `dotI8SDOT`'s ~49 GMAC/s class) | 37.2 | ~27 | 0.39x |
| kernel bandwidth-saturated (~60 GB/s × 0.75 GB ≈ 12.5 ms) | 28.1 | ~36 | 0.52x |
| matmul → 0 (unreachable bound) | 15.6 | ~64 | 0.94x |

Two consequences, both hard:

1. **Parity is out of reach for kernel work alone.** Even a zero-cost matmul lands under
   ollama's 68.3 tok/s. The original "~70-80 tok/s, the whole gap" framing in § "Why this doc
   exists" was wrong — written before the overhead floor was measured, corrected here.
2. **The non-matmul 15.6 ms/token is a co-equal lever, and it looks like a defect, not a
   floor: it alone exceeds ollama's entire 14.6 ms token budget** — ollama fits its matmuls
   AND its attention/norms/KV/sampling/streaming inside less time than goinfer spends on just
   the non-matmul part. Same shape at 0.5B (7.7 ms non-matmul vs ollama's 9.2 ms total). Gate
   0 measured it proportional to hidden dim, which points at attention/norm/KV compute rather
   than fixed serving overhead — at depth ~130 on a 1.5B, 15.6 ms of non-matmul work is a lot.
   Nothing in this doc has profiled *inside* that 26.5% yet.

Illustrative both-levers case: bandwidth-saturated kernel (12.5 ms) + non-matmul cut to ~5 ms
→ ~57 tok/s ≈ 0.84x. That is what "close the gap" realistically means now.

### Next steps, resequenced (2026-08-23) — three cheap probes before item 3's plumbing

Items 1+2 as scoped: closed (measured negative above). Item 3 stays the live kernel lever —
the mechanism of the negative (issue-limited, instructions cost wherever they land) argues
*for* an instruction-removing repack, not against it. But three cheaper things now come first:

1. **Break down the 26.5%.** Same stub technique as Gate 0 item 3, one level finer: attention
   vs norms vs KV append vs sampler vs detokenize/stream, per token. ~Half a day. Decides
   whether the non-matmul lever is a second campaign, one bad loop, or a real floor. Given
   revised-math consequence 2, this is currently the highest-information-per-hour item in the
   doc.
2. **Chase the parallelization anomaly Gate 0 flagged and left.** A single-threaded
   microbenchmark landing within ~13% of the real, supposedly-parallel decode's matmul-only
   rate says the fork-join matmul is buying almost nothing on this box. If confirmed, fixing it
   multiplies with any kernel win and may dwarf item 3 alone (6 threads at even 60% efficiency
   beats a 2x single-thread kernel). Also decides whether item 4's multi-row shape is
   optional or load-bearing.
3. **Item 3 via harness first, plumbing last** (the AVX2 round's own diagnostic pattern):
   repack the weights *in the benchmark harness only*, write the split-half kernel against
   that layout, measure GMAC/s in isolation. Design the layout jointly with item 4's 4-row
   interleave — they are one layout decision, and probe 2's result feeds it. The item-2
   centering correction gets one more look here: with the unpack prologue gone, the per-group
   correction may fit in the freed issue budget (left open by the negative above). Fund the
   arm64 load-path repack in `WeightMat` only after the harness shows the win.

### Probes 1+2 result (2026-08-23, same day) — the 26.5% is ~99% one thing; the parallelization
### anomaly was a byte-estimate artifact, not a dead fork-join

**Probe 1 — the 26.5% breakdown.** Independent (not cumulative) env-gated stubs against the
matmul-real baseline, per component: `normalize()` (norms), `attendBatchedHeads` + the two
`ropeAt` calls (attention math), `Sampler.SampleWithInfo` (sampler) — all decode-only,
`bench_peer` method, both model sizes. KV-append measured separately (below) after its stub
crashed batched prefill (the naive per-call stub skips the write real prefill history depends
on reading back within the same batched pass — not chased further, a cheaper isolated
micro-benchmark settles it instead).

| component | 0.5B cost | 0.5B % of total | 1.5B cost | 1.5B % of total |
|---|--:|--:|--:|--:|
| norm | 0.59 ms | 2.0% | 0.40 ms | 0.7% |
| **attention (RoPE+QK^T+softmax+AV)** | **8.47 ms** | **28.7%** | **15.16 ms** | **26.0%** |
| sampler | 0.51 ms | 1.7% | −2.11 ms (noise) | −3.6% (noise) |
| KV-append (isolated, see below) | ~0.004 ms | ~0.01% | ~0.004 ms | ~0.01% |
| **non-matmul floor (Gate 0's own number)** | **7.34 ms** | **24.9%** | **15.33 ms** | **26.3%** |

**Attention alone accounts for 99-115% of the entire non-matmul floor at both sizes** (8.47/7.34
and 15.16/15.33 — the >100% at 0.5B is cross-cell noise, not a real inconsistency; norm and
sampler are both small and roughly tied with each other, indistinguishable from measurement noise
at this granularity). **Answer to "is the non-matmul lever a second campaign or one bad loop":
one thing, not several.** KV-append confirmed negligible by a separate isolated Go benchmark
(steady-state `append` across 28 layers, kvDim=256: 4.29 µs/token — ~0.03% of the non-matmul
budget, nowhere near worth chasing).

**This is not a bug — it's `causalAttention`'s own documented tradeoff.** The f64 accumulation
path (`acc64 := true` in `attention.go`) exists specifically so decode is bit-identical to batched
prefill/speculative-verify (the comment there: "old dense path... flipped ~11% of argmaxes... f64
is order-independent ⇒ exact"). It is real, deliberate, and load-bearing for spec-decode
correctness — not a loop someone forgot to optimize. **What this changes: the highest-leverage
lever in this whole doc may not be a W4A8 kernel at all** — it's whether attention's cost can drop
for configurations that don't need spec-decode bit-identity (an f32 fast path, gated the way
`--metal-fast-prefill` already gates a different exactness/speed tradeoff elsewhere in this
codebase). That is a new, separate, real design question this probe surfaces — not scoped or
started here; flagging it as the standout candidate for a follow-up task doc of its own.
**(Now scoped: `docs/task-attention-decode-cost.md`, 2026-08-23 — invariant enumeration, an
A1/A2/A3 ladder starting with bit-identity-preserving restructuring, and acceptance math tied
back to this doc's revised table.)**

**Probe 2 — the parallelization anomaly.** Chased with a direct benchmark of the REAL parallel
`MatmulBTW4A8Into` (not the single-call kernel), first at the ops-per-byte harness's borrowed
27B-model shape ([17408,5120]: 1w 15.26 GB/s → 6w 55.98 GB/s, 3.67x, 61% efficiency), then at the
**actual 1.5B model's real gate/up-proj shape** ([8960,1536], from the GGUF's own metadata:
hidden=1536, FFN=8960, 28 layers) with a cold, 8-distinct-matrix rotation (ruling out
cache-residency as a confound — result was within 1% of a single reused matrix, matching Gate
0's own single-call hot/cold finding): **1 worker 14.65 GB/s → 6 workers 40.58 GB/s (2.77x, 46%
efficiency) → 8 workers 42.69 GB/s (2.91x).**

**The flagged "~13% agreement" was a byte-estimate artifact, not a dead fork-join.** The
concerning framing rested on ~0.8 GB/token (file-size arithmetic). A proper per-tensor accounting
from the real GGUF metadata — dense layers (qkv/o/gate/up/down, 1.31B params × 0.625 B/param
int4+scale) plus the LM head (151936×1536, int8-pinned, tied embedding, streamed once per token)
— comes to **~1.05 GB/token**, about 31% more than the earlier estimate. Recomputing the
matmul-only implied rate with this number: 1.05 GB / 43.22 ms = **24.4 GB/s**, not ~17.4 GB/s.
Against the isolated benchmark's 40.6 GB/s (6 workers, real shape): **real decode achieves ~60%
of isolated parallel efficiency — a real, worth-noting gap, but not the "single-thread-equivalent,
parallelism buying almost nothing" picture the rough estimate implied.** (For calibration: normal
fork-join overhead at ~7 matmuls/layer × 28 layers ≈ 196 fork-joins/token would need to cost
~4-5 µs each to account for the whole gap on its own — plausible-adjacent but not confirmed; the
remainder is more likely real (different weight matrix touched per call, 196 distinct large
allocations vs this probe's 8-matrix rotation) than a single dominant cause. Not fully resolved —
recorded as "real but partially unexplained," not chased further given the two probes together
already answered the higher-priority question (item 3's kernel work still matters, multiplied by
whatever fraction of this gap is closeable, but attention dominates the non-kernel side by a wide
margin).

**Net effect on prioritization:** item 3's kernel work is still worth doing (Gate 0's issue-limited
finding stands, unaffected by either probe) but is no longer the obvious single highest-leverage
item in this doc — attention's cost is comparable in size and, being a documented tradeoff rather
than an unoptimized kernel, may have a cheaper path to a partial win (a gated fast path) than a
from-scratch NEON kernel redesign.

## Format follow-on — `.giw` and the repacked layout (decision recorded 2026-08-23, deferred)

Scoped in conversation before Gate 1 started; recorded here so the design isn't re-derived.
Context: the Gate 1 correction above already establishes that the canonical packed layout is
load-bearing on disk (`.giw` kind 3 zero-copy mmap-aliases it; `internal/chatapp/model.giw`
is a checked-in artifact; the scalar oracle hardcodes it), and that Gate 1 therefore ships as
an arm64-only load-time repack, GPU-style. This section is about the step after that: whether
the repacked layout eventually goes *on disk*.

**Decision: a new weightMat kind, opt-in via a `cmd/prequant` target flag — never a wholesale
replacement of kind 3 — and only after the layout is harness-final and the load-time repack
has shipped and been measured.** Reasons, so they don't get relitigated:

- **Why on disk at all (eventually):** three real costs of the load-time repack. It erodes the
  prequant fast-load win (an extra O(K) pass per int4 tensor at every load — the same class of
  work `cmd/prequant` exists to hoist); the repacked copy is anonymous memory, not
  page-cache-backed mmap (matters for the 16 GB Mac paging story); and — decisive — the paged
  MoE path (`expertPager`, gemma4) reads experts directly off the read-only `.giw` mapping
  with no load-time-transform option at all, so the new kernel can never apply to paged
  experts unless disk layout equals kernel layout.
- **Why not a wholesale swap:** the layout is ISA-specific (NEON split-half + 4-row interleave
  is not what AVX2 or VNNI wants, and is not the GPU layout). llama.cpp shipped exactly this
  (Q4_0_4_4/8_8 aarch64 file types) and later removed them in favor of runtime repack —
  ISA-specific files are a mess as an interchange format. `.giw` is better placed than GGUF
  was (it's `cmd/prequant` output, typically built on the target box), but shipped artifacts
  like the embedded chat model would become per-ISA, so the portable canonical kind stays the
  default.
- **GPU interaction:** precedent already in-tree — GPU backends repack at upload from the
  canonical layout. A bundle in the arm64 kind either gets repacked at upload (note: this
  reintroduces work on the Mellum2 direct-int4-upload fast path, 66s→13s — measure before
  accepting) or prequant simply emits the arm64 kind only for CPU-target bundles.
- **Fold-ins, so the format is cut once:** (a) `docs/task-giw-f16-scales.md` rides along — the
  new kernel applies each group scale once at the int32 boundary, so f16 scales widened there
  cost ~nothing, unify CPU/GPU scale precision, and retire that task without its own version
  bump; (b) whatever centering convention the harness-final layout lands on (uncentered
  nibbles + Σact correction, or precomputed per-group zero-points) is part of the kind
  definition; (c) `giwVersion` bump with version-aware read, matching the `GINFB` v1/v2 house
  pattern — the machinery exists, no breaking change actually needed.
- **Numerics caveat (kernel-side, but belongs in the same decision):** integer-domain
  accumulation changes summation order vs today's per-group f32 fold, so CPU W4A8 parity
  goldens shift — cosine re-gate shape (as in the f16-scales task), independent of what the
  disk format does.

**Sequencing rule:** harness-final layout → arm64 load-time repack shipped, load cost and RAM
delta measured → only then cut the kind, with the paged-MoE need as the forcing function. A
serialized format frozen around an unmeasured kernel layout is the one clearly wrong order.

## Zero-cost item, ship regardless of the gates

Surface the already-measured fact: on Apple Silicon CPU decode, `-quant int8int8` is ~60%
faster than the int4 default at roughly double the weight RAM. Doc it where a CPU-only Mac
user will see it (env-vars/quant docs, maybe the README perf note). Whether to make it the
darwin/arm64 CPU *default* is a product call — RAM cost is real — but the guidance costs
nothing today. If Gate 1 ships, this note gets retired.

## Item-3 campaign — Step 0 quiet-box baseline (2026-08-23)

Per `docs/prompts/w4a8-item3-harness.md`'s Step 0: `bench_peer` method, 2 runs each, quiet box
(1-min load average settled to 2.73 after closing other VS Code windows/sessions; the campaign's
own two prior load-contamination incidents made this worth waiting out rather than skipping),
against the tagged aikit v1.25.0 (fresh `go build`, no `go.work` override) — a fresh serve binary,
not the one used for the A1 close-out.

| model | quant | run 1 | run 2 | mean | vs ollama |
|---|---|--:|--:|--:|--:|
| 0.5B, depth 128 | int4 | 40.49 tok/s | 41.68 tok/s | **41.09 tok/s** | 0.38x (ollama 109.0) |
| 1.5B, depth 128 | int4 | 21.63 tok/s | 21.73 tok/s | **21.68 tok/s** | 0.32x (ollama 68.3) |
| 0.5B, depth 128 | int8int8 | 83.83 tok/s | 86.67 tok/s | **85.25 tok/s** | 0.78x (ollama 109.0) |
| 1.5B, depth 128 | int8int8 | 37.51 tok/s | 37.61 tok/s | **37.56 tok/s** | 0.55x (ollama 68.3) |

**Certifies the A1 close-out's load-caveated int4 numbers, on a quiet box.** 21.68 tok/s (1.5B)
clears the ≥21 tok/s bar cleanly — the campaign doc's load-5.9 re-measurement (20.1 tok/s mean)
is confirmed as ambient-load noise, not a real regression; these numbers, not that one, are the
correct A1-era figures now that both exist. 41.09 tok/s (0.5B) likewise slightly exceeds the
override-era 40.71 tok/s figure.

**Upgrades the zero-cost `int8int8` guidance from projection to measurement.** The item-3 brief
projected ~40 tok/s (~0.59x ollama) at 1.5B post-A1; measured **37.56 tok/s (0.55x)** — close to
projection, real headroom still on the table for a kernel that closes the rest of the gap. These
four cells are this campaign's *before* baseline: the item-3 harness's projected end-to-end numbers
get compared against this table, not the load-5.9 one.

## Non-goals — measured or reasoned dead ends, do not reopen without new evidence

- **An Accelerate/vecLib/AMX kernel path.** Retired above; decode GEMV never went through it.
- **Chasing int4-vs-int8int8 as its own finding.** Same ALU limit the June spike characterized;
  Gate 1 subsumes it.
- **Wiring `SetParallelWidth` to a CLI flag for this.** Measured neutral for goinfer on this
  box (6 vs 8 threads within spread). The E-core mechanism is real for ggml's threading model,
  not ours.
- **3-bit / rotation formats.** Shelved 2026-06-14 with two independent sinks; nothing here
  changes that. If anything, Gate 1's unpack-amortized layout would be the prerequisite for
  ever revisiting.
- **The accumulator dependency-chain fix.** Measured ~0% on AVX2; do not port it on a hunch.
- **The decoupled Σact-correction pass (Gate 1 items 1+2 shape).** Measured 0.972x on
  `apple-m1pro`; full record above. The only sanctioned retry is the folded-into-repack
  variant inside item 3's harness, per the resequenced next steps.
- **Replacing `.giw` kind 3 with an ISA-specific layout.** See § Format follow-on — new
  opt-in kind or nothing.
- **The CUDA depth curve.** Tracked separately, as before.

## Done looks like

Gate 0: done (GO — result recorded above). Gate 1 items 1+2: done (measured negative, recorded
above). Remaining, in the resequenced order: the non-matmul breakdown and the parallelization
probe, each landing their numbers in this doc; then item 3+4 as a harness-measured GMAC/s
number *before* any `WeightMat` plumbing; then (if the harness says go) the arm64 load-time
repack, parity green, and a before/after `bench_peer` table against the REVISED acceptance
math — both levers reported side by side, kernel and non-matmul. The format follow-on ships
last or never, per its own sequencing rule. A negative result at any step costs the same to
obtain and is worth as much.

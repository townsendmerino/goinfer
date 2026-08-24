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
   **Correction (2026-08-23): this reading does not reproduce — see the item-3 harness results
   section below.** Re-run 4 times on a settled box: ratio 0.99-1.03 every time, stably NOT
   issue-limited. Likely cause: this same session ran item 2's STREAM measurement moments away
   from an initial attempt that was "accidentally measuring swap on a box with other sessions
   running" (below) — the box was under load when this probe ran too, and a loaded box is exactly
   the condition under which this instrument has now twice given a misleading verdict (see the
   demotion note in `aikit/docs/internal/priors-microgpt-c.md` §1).
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

## Item-3 harness results (2026-08-23) — Gate 0's "issue-limited" verdict corrected, real lever found

Per `docs/prompts/w4a8-item3-harness.md`. All kernels live in `aikit/linalg` (uncommitted,
harness-only per the brief — `dot_w4a8_arm64.s`, `quant_w4a8_arm64.go`, `w4a8_item3_harness_test.go`,
`w4a8_item3_bench_arm64_test.go`, `w4a8_sdotv2_test.go`); every kernel below is proven bit-exact or
rel-err ≤1e-3 against a scalar oracle first, then benchmarked — no perf number below comes from an
unverified kernel. All measurements: quiet box (VS Code + all other sessions closed, 1-min load
average 1.3-2.0 throughout), real 1.5B gate/up shape (K=1536, N=8960), order-alternated best-of-3,
hot (L1-resident) and cold (streaming N distinct rows).

**Correction to Gate 0: the "issue-limited, ratio 1.11" verdict does not reproduce.** Item 3's
first cell — `dotW4A8SplitHalfSDOT`, a repacked layout (`repackSplitHalfRow`, harness-only, no
`quant.go`/`.giw` change) that drops `dotW4A8FoldSDOT`'s two `VZIP1`/`VZIP2` unpack instructions per
group — measured a **flat 1.000x against the production kernel, both hot and cold, reproduced
across 2 independent runs.** Zero effect from removing 2 of ~10 instructions/group is inconsistent
with an issue-limited kernel. Re-running `TestW4A8IssueWidthProbeARM64` (the same test Gate 0 used)
4 times on this settled box gave ratio 0.99-1.03 every time — stably **NOT issue-limited**, the
opposite of Gate 0's recorded 1.11. The original reading was very likely a single noisy
measurement (this exact box has a documented history of this — the ops-per-byte harness's own
comment records a prior "hot 12x slower than cold" reading that "did not reproduce under
repetition"). This retroactively changes the read of items 1+2's negative result too: the 3%
regression there is better explained by the decoupled correction pass's extra memory reads/loop
overhead than by "an issue-limited kernel taxes instructions wherever they land" — the latter
explanation assumed a premise that doesn't hold up.

**The real bottleneck: a serial accumulator chain, the same failure mode the attention A1 campaign
already found and fixed.** `dotW4A8FoldSDOT` folds every group's contribution into ONE 4-lane f32
accumulator (`V20`) via `VFMLA`, a cross-iteration RAW dependency — group *g*'s fold must wait for
group *g-1*'s to retire. `dotI8SDOT` (`dot_i8dp_arm64.s`, the reference kernel this whole doc
compares against) already avoids exactly this with **four independent accumulators**, by its own
comment, specifically "to hide SDOT latency." Building `dotW4A8FoldSDOT2Acc` (two independent
`VFMLA` chains, canonical layout and centering otherwise unchanged) measured a real,
reproducible **1.41-1.47x hot, 1.39-1.41x cold** across 2 runs. `dotW4A8FoldSDOT4Acc` (four
chains) measured **the same ~1.4x** — two accumulators already fully hides the FMLA latency at
this shape; a third/fourth lane buys nothing further in isolation.

**This is now a three-data-point picture, not a one-off:** `docs/task-attention-decode-cost.md`'s
A1 move (b) (8-wide interleaved QK^T accumulators, replacing one serial f64 chain) measured
**4.41x** at the qwen2.5-1.5b decode attention shape — the identical mechanism, same ISA, a
different kernel. AVX2's `dotW4A8Fold4AVX2` attempt at the same fix (`perf-dead-ends.md` §8.9)
measured ~1% (noise) — but that kernel is port-bound, not latency-bound, a genuinely different
bottleneck on a genuinely different ISA. One ISA stalls on accumulator latency (NEON, twice now);
the other stalls on a port (AVX2, once). Both are now measured, not assumed — see
`priors-microgpt-c.md` §1's demotion note for what this means for the issue-width probe itself.

**The two levers compound once the accumulator bottleneck is fixed.** With the latency masking
gone, the unpack instruction count item 3 targeted becomes visible again: `dotW4A8SplitHalf2Acc`
(split-half layout + 2 accumulators together) measured **1.60-1.75x hot, 1.60-1.64x cold across 3
runs** — real GMAC/s: orig 23.6-24.4 → combined 38.4-42.7 (hot) / 38.4-39.4 (cold). This is a
straightforward story: item 3's layout change was never wrong on its own terms, it was invisible
because the FMLA-latency stall dominated the timing budget it would have shortened. Fix the
dominant cost first, and the second-order one becomes real.

**6-worker aggregate, measured (2026-08-23, same day):** `dotW4A8SplitHalf2Acc` fanned out through
`Workspace.parallel` — the identical fork-join `MatmulBTW4A8Into`'s parallel branch uses, not a
from-scratch loop — at the real gate/up shape, cycling 8 distinct weight matrices (matching
`TestW4A8_parallelScaling`'s own method). Reproduced twice:

| workers | orig GB/s | combined GB/s | ratio |
|--:|--:|--:|--:|
| 1 | 14.79-14.82 | 24.26-24.47 | 1.64-1.65x |
| 2 | 26.06-26.95 | 40.98-41.44 | 1.54-1.57x |
| 4 | 39.98-40.47 | 54.25-54.31 | 1.34-1.36x |
| 6 | 42.19-42.20 | 58.54-58.62 | **1.387-1.389x** |
| 8 | 45.27-45.63 | 56.87-56.97 | 1.246-1.259x |

**The combined kernel's own scaling curve peaks at 6 workers and drops at 8** (58.5→57.0 GB/s),
reproduced both runs — the original kernel keeps climbing to 8 (42.2→45.4 GB/s). This is the
latency-hiding-raises-memory-pressure effect predicted before this ran: the faster kernel reaches
whatever shared-resource ceiling exists sooner, so its own optimal worker count is lower than the
slower kernel's.

**Against the brief's exact ≥1.4x-over-40.58-GB/s bar: 58.5-58.6 GB/s is 1.44x over the brief's
originally-stated 40.58 GB/s baseline, but 1.387-1.389x over this session's own freshly-measured
6-worker baseline (42.19-42.20 GB/s)** — the two production-baseline readings differ by ~4%
(ordinary box-to-box/session-to-session variance, the same class of drift this campaign has
repeatedly documented), and the ratio lands just under 1.4x against the stricter of the two,
just over against the other. Read together with the single-call cold rate (38.4-39.4 GMAC/s
against a ≥40 bar): **both GO criteria land at ~97-104% of their stated bars, consistently, not a
clean clear on either but not a real miss either** — reported exactly rather than rounded either
direction.

**Cheap fourth check, per feedback: does a 4th accumulator lane move the saturation point once
combined with the shorter unpack?** `dotW4A8SplitHalf4Acc` (split-half layout + 4 lanes) measured
**1.75x hot, 1.641x cold** — statistically identical to `dotW4A8SplitHalf2Acc`'s 1.6-1.75x/1.60-1.64x.
**No, it doesn't move**: 2 accumulators already fully saturates the achievable latency-hiding at
this shape, with or without the layout fix. A confirmatory null, not a new finding — worth having
asked given it was one benchmark run, not worth building further on.

**Item 4 harness result (2026-08-23, same day) — real, and it changes which reading of the GO bar
is correct.** `dotW4A8SplitHalf4Row` computes 4 REAL output rows per call (unlike the 2/4-accumulator
checks above, which split ONE row's own fold into artificial lanes): `repackSplitHalf4RowBlock`
interleaves 4 rows' split-half-packed groups contiguously (row0's 16 bytes, row1's, row2's, row3's,
per group — a "Q4_0_4x4-style" repack, matching the brief's own description), and the kernel loads
the activation chunk ONCE per group, reusing it across all 4 rows' SDOT instead of reloading it 4
times. Correctness proven exact against a per-row scalar oracle across every nGroups 1-37 (no
internal residue of its own — one group per outer iteration, unlike the artificial-lane kernels'
mod-2/mod-4 requirement).

Measured against 4 separate calls to both the production kernel and the current-best single-row
combo (`dotW4A8SplitHalf2Acc`), reproduced twice: **1.036-1.083x on top of the combo kernel** (hot
1.036x, cold 1.076-1.083x), for a **total 1.78-1.83x vs the original production kernel** — real,
modest, and additive, exactly as expected for a lever targeting a genuinely different resource
(activation-load/instruction-count amortization across real outputs) than the one the
2-vs-4-accumulator check already settled. Uses a single accumulator per real row (4 total, not an
additional per-row 2-way split) — the 4 genuinely independent output chains already provide at
least as much latency-hiding as the artificial split did, which the measured result confirms rather
than assumes.

**6-worker aggregate, re-measured with item 4 in the mix, reproduced twice:**

| workers | orig GB/s | combo (2acc, no row4) GB/s | row4 GB/s | row4/orig |
|--:|--:|--:|--:|--:|
| 1 | 14.6-14.8 | 24.1-24.4 | 25.9-26.2 | 1.77-1.78x |
| 2 | 27.0-26.6 | 41.8-41.8 | 44.0-44.0 | 1.63-1.65x |
| 4 | 41.5-40.6 | 54.5-54.6 | 56.5-56.5 | 1.36-1.39x |
| 6 | 43.3-42.3 | 58.5-58.8 | **59.6-60.6** | **1.40-1.41x** |
| 8 | 45.3-44.7 | 56.9-60.2 | 58.0-62.1 | 1.28-1.39x |

**Item 4 turns the borderline 6-worker reading into a clean pass: 1.40-1.41x, consistently ≥1.4x
across both runs** (against combo alone's 1.387-1.389x, which sat just under against this
session's own baseline). At workers=8 the combined kernel no longer collapses as sharply as combo
alone did in the earlier reading (which dropped to 1.246-1.259x) — item 4 relieves exactly the
memory-pressure resource that capped combo's own scaling, as predicted before this ran.

**FINAL DISPOSITION: GO, with the completed grid.** Both criteria now clear cleanly and
reproducibly: single-call cold 42.37-42.67 GMAC/s (bar ≥40) and 6-worker aggregate 1.40-1.41x
(bar ≥1.4x). **The layout winner is split-half (item 3) + 4-row interleave (item 4), signed
centering, single accumulator per real row** — this is the layout the plumbing phase should build
against; it is settled now rather than left to be discovered mid-plumbing, which is the entire
reason item 4 was run now instead of deferred (a load-time repack's byte arrangement is what the
`WeightMat` path, its tests, and eventually a `.giw` kind would all inherit — a one-shot decision,
unlike the uncentered-correction question below).

**Correction to the plumbing brief's numerics framing (2026-08-24) — SUPERSEDES that brief's §4
("Numerics — this is the phase's real risk section"): the winning kernel is bit-identical to
canonical, not merely rel-err-close.** Argued first, then proven, matching the A1 standard:

- **The mechanism.** Split-half's repack pairs group *g*'s two 16-weight blocks (k_local 0-15 and
  16-31) so ONE 16-byte load feeds both — but the SDOT consumption order within that load is
  unchanged: block0 (k=0-15) is still folded before block1 (k=16-31), the same as canonical's
  interleaved layout after its VZIP unpack. One load feeding two blocks changed *which
  instructions* extract the bytes, not *the order the values enter the fold*. Group order across
  the whole K-loop (g=0,1,2,…) is also unchanged. Since int32 accumulation within a group is
  order-independent (exact addition) and the only place float rounding enters is the once-per-group
  scale-fold — which happens in the identical g=0,1,2,… sequence — the per-output float32 result is
  the same reduction tree as canonical's, not a different one.
- **Why one accumulator per row was enough.** `dotW4A8SplitHalf2Acc`'s within-row 2-way split
  (splitting ONE row's own fold into two chains, combined via `FADDS` only at the very end) is what
  breaks bit-identity — that genuinely reorders the reduction. `dotW4A8SplitHalf4Row` doesn't need
  that: 4 REAL output rows processed together already supply 4 independent FMLA chains (one per
  row), and the 2-vs-4-accumulator finding above already established that 2 independent chains fully
  saturate the latency-hiding this kernel needs. Four genuine chains is more than enough on its own,
  so each row keeps its own single, canonical-order fold — the artificial split was never load-bearing
  once real cross-row independence was available, which is exactly what that earlier null result
  pointed toward.
- **The trial result.** Verified exact (`==`, not tolerance) against `dotW4A8FoldSDOT` across 800
  random comparisons: zero mismatches (`TestDotW4A8SplitHalf4Row_bitIdenticalToCanonical`,
  `aikit/linalg/w4a8_sdotv2_test.go`) — argument and measurement agree.

**Consequence: the plumbing phase owes no golden regeneration and no cosine re-gate for this
specific kernel swap** — the existing decode == prefill == verify bit-identity guarantees carry
over unchanged, because the per-output arithmetic literally didn't change, only which bytes it
reads from and which rows share one activation load. **This also quietly upgrades the paged-MoE
carve-out** (`w4a8-plumbing.md` §3) **from a correctness requirement to a pure performance
matter** — there was never a numerics reason paged tensors couldn't eventually use this kernel,
only the load-time-repack mismatch with `expertPager`'s read-only canonical mapping, which stands
on its own regardless of this finding. **Whoever picks up `docs/prompts/w4a8-plumbing.md` next:
its §4 numerics section is superseded by this entry — read this before that section, not after.**
M-independence still needs proving through `TestForwardN_matchesSequential`/
`TestSpeculativeGreedyParity` with the new dispatch active (that part of the brief stands, and
should still be run rather than assumed — "should hold" always gets measured in this campaign),
but as confirmation of an already-exact property, not a re-golding exercise.

**The items-1+2 uncentered-correction retry remains genuinely optional, deferred without cost.**
Unlike the layout, centering changes no weight bytes at all — the canonical format already stores
nibbles as value+8, so dropping the in-kernel centering subtract only changes the kernel and the
Σact calling convention, never the repack. It can be tried in the plumbing phase or later without
foreclosing anything decided here, and is left there rather than run now.

**Hand-off point: the plumbing brief**, once written, should build the arm64 load-time repack in
goinfer's `WeightMat` against exactly this layout (`repackSplitHalf4RowBlock` +
`interleaveScales4Row`'s interleaving, `dotW4A8SplitHalf4Row`'s kernel shape), per the campaign
doc's Gate-1 correction (canonical `.giw`/`quant.go` untouched, GPU-backend-style upload-time
repack). All harness code (kernels, repack functions, tests) lives in `aikit/linalg`, uncommitted
per this campaign's convention pending the plumbing decision.

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
- **The accumulator dependency-chain fix — AMD64/AVX2 only.** Measured ~0% on AVX2
  (`nvidia-rtx2070s`); do not re-try on that ISA/box without new evidence. Does NOT extend to
  arm64/NEON: the item-3 harness results section above measured this exact fix at **1.4-1.75x on
  `apple-m1pro`** — the AVX2 non-goal was correct as recorded (a clean measured negative, not
  re-triable without new evidence) and the arm64 attempt was correctly not blocked by it (new
  ISA, new evidence, per `priors-microgpt-c.md` §2's own porting-caution rule). Two ISAs, two
  different bottlenecks: AVX2 was port-bound (shuffle-port contention the issue-width probe
  couldn't see), NEON is latency-bound (the same mechanism A1's move (b) fixed in attention's
  QK^T fold, `docs/task-attention-decode-cost.md`).
- **The decoupled Σact-correction pass (Gate 1 items 1+2 shape).** Measured 0.972x on
  `apple-m1pro`; full record above. The only sanctioned retry is the folded-into-repack
  variant inside item 3's harness, per the resequenced next steps.
- **Replacing `.giw` kind 3 with an ISA-specific layout.** See § Format follow-on — new
  opt-in kind or nothing.
- **The CUDA depth curve.** Tracked separately, as before.

## Done looks like

Gate 0: done (GO — result recorded above, with a correction: the issue-width probe's original
"issue-limited" reading did not reproduce). Gate 1 items 1+2 (decoupled shape): done (measured
negative). Non-matmul breakdown and the parallelization probe: done. Item 3+4 harness: **done,
GO** — layout winner is split-half + 4-row interleave, both GO criteria cleared (single-call
42.4-42.7 GMAC/s, 6-worker aggregate 1.40-1.41x). Remaining: the arm64 load-time `WeightMat`
repack (plumbing phase, hand-off is the next brief), parity green, and a before/after `bench_peer`
table against the REVISED acceptance math — both levers reported side by side, kernel and
non-matmul. The uncentered-correction retry (items 1+2's sanctioned retry) is optional, deferred
without cost. The format follow-on ships last or never, per its own sequencing rule. A negative
result at any step costs the same to obtain and is worth as much.

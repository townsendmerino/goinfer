# Task: decode attention's f64 cost — cut the ~26% floor without breaking bit-identity

> Scoping doc. Opened 2026-08-23 from `docs/task-w4a8-neon-bandwidth.md` § "Probes 1+2 result"
> (attention = 99-115% of the non-matmul decode floor at both model sizes) — the follow-up that
> probe flagged as "the standout candidate for a task doc of its own." Sibling to the W4A8
> campaign, not part of it: this lever is independent of and multiplicative with that doc's
> kernel work. **Status: Gate A0 CLOSED, all four items answered, 2026-08-23 — GO. A1 (thread
> across heads + interleave independent outputs + fix AV's memory order) is cleared to build.**

## The measurement (probe 1, 2026-08-23, `apple-m1pro`, depth ~130)

| | 0.5B | 1.5B |
|---|--:|--:|
| attention (RoPE+QK^T+softmax+AV), per token | 8.47 ms (28.7% of decode) | 15.16 ms (26.0%) |
| everything else non-matmul (norms+sampler+KV-append) | noise-level | noise-level |
| ollama's ENTIRE token, same model, same box | 9.17 ms | 14.64 ms |

goinfer's attention alone costs about what ollama spends on a whole token. And this is at depth
~130 — attention cost scales with nKeys while the weight matmuls don't, so this same path is the
prime suspect for the long-context deficit (ollama ahead 5.54x at 32k — figure and the
"FA-class attention rewrite" lever both live in `docs/completed/task-decode-attention-fa.md`,
citation corrected 2026-08-23 from an earlier plan-still-slow attribution). That FA campaign
was scoped **GPU-only and explicitly excluded CPU** (task-decode-attention-fa.md:189, "Not in
scope: ... CPU") — so the CPU side of this lever is genuinely new territory, not a re-tread.
**Whatever this doc wins at depth 130 multiplies at depth 32k.**

## Why it costs this much — mechanism hypothesis, for Gate A0 to confirm

Not a forgotten loop. Decode deliberately routes through `attendBatchedHeads` at K=1 with
`acc64 := true` (`decoder/attention.go` ~130 and its comment block) so that decode is
bit-identical to batched prefill/verify. The cost structure, from reading
`decoder/forwardn.go:709` (`attendBatchedHeads`) and the arithmetic:

- **The work is tiny; the rate is the problem.** At 1.5B, depth ~130: QK^T + AV ≈ 2 × 28 layers
  × 12 Q-heads × 130 keys × 128 dims ≈ 11M MACs, plus ~44K exps. 15.16 ms for 11M MACs is
  **~1.4 ns/MAC** — three orders of magnitude below the same box's int8 dot rate, and right at
  the speed of a serial f64 FMA dependency chain (~4 cycles/MAC at ~3 GHz). The floor is not
  f64 arithmetic per se; it is *unoverlapped* f64 arithmetic.
- **Single-threaded.** `attendBatchedHeads` runs `for kvh { for g { ... } }` serially — no
  goroutines anywhere in the function. 12 independent query heads × 28 layers of exploitable
  parallelism, none used.
- **Serial chains per output.** `MatmulBTAcc64Strided` runs "the SAME sequential f64 reduction
  as `MatmulBTAcc64`" (comment at decoder/forwardn.go:819). Whether it interleaves *independent
  outputs* (different keys' dots in QK, different dims in AV — legal without touching any
  single output's order) is an aikit-internal question Gate A0 must answer by reading the
  kernel.
- **AV's element reads are 1KB-strided.** The scores·V call reads `vals` with element stride
  `kvDim` (decoder/forwardn.go:919) — each successive f64 MAC touches a new cache line. The f32 path's
  gather/transpose that fixed this is skipped on the acc64 path (and the f32 path itself is
  test-only today: "every live caller passes useAcc64=true").
- RoPE and softmax ride along in the measured number but are O(heads·hd) and O(nKeys)
  respectively — almost certainly minor. Gate A0 splits them out.

## The invariant, precisely — enumerate before designing

What `acc64` actually buys (all from the attention.go comment block, verbatim claims):

1. **Same-model speculative decoding**: batched verify must reproduce sequential greedy
   exactly. The old f32 dense path flipped ~11% of argmaxes and left spec acceptance at 0.93.
   Gates: `TestSpeculativeGreedyParity`, `TestForwardN_matchesSequential`.
2. **decode == prefill == verify bit-identity** for both dense and MoE, including tree
   attention (05).
3. **MoE router stability**: attention output feeds the router; f32 QK^T reassociation
   (~4.6e-5) flips a top-k expert at a near-tie and cascades.
4. **Confirmed by an explore pass (2026-08-23):** the consumer list is these two gates plus
   several docs and measurement logs that cite the bit-identity guarantee. A1 touches none of
   them; any A2/A3 change must sweep those docs too, per the parity-coverage policy.
5. The root cause of the f32 problem is stated precisely: **f32's QK^T/AV reduction order is
   M-dependent (K=1 decode ≠ M=K verify) at every aikit version; f64 is order-independent ⇒
   exact.** This is the sentence the whole design space pivots on: the invariant is
   *M-independence of each output's reduction*, and f64 is the current mechanism for it, not
   the requirement itself.

## The option ladder — cheapest-invariant-preserving first

**A1 — bit-identical restructuring: change no numerics at all.** Three independent moves, all
preserving every output's exact f64 fold order, so every gate above passes unchanged and no
golden shifts:

- **(a) Thread across query heads** (and layers' two KV heads): each head's outputs are
  computed by an unchanged serial fold; distributing heads across cores reorders nothing
  within any output. Up to ~6x on this box at 12 heads. **Depth-aware, not heads-only** (Gate
  A0's depth curve below): at short depth each QK/AV call is too small (~16.6K MACs) for
  within-call splitting to pay for its own fork-join, so heads-parallel is the whole story —
  but past whatever depth makes one call's own work clear a real threshold (~1M MACs by 8192
  keys), (b)'s within-call independent-axis split becomes a second, additive axis. Heads-parallel
  alone caps near 6x; the depth regime where attention is 90%+ of the token is exactly where more
  than 6x is worth having.
- **(b) Interleave independent outputs inside the f64 kernel**: run 4-8 keys' dots (QK) or
  dims' folds (AV) as concurrent accumulator chains in one pass. Each output's own summation
  order is untouched; the FMA latency chain stops serializing the whole call. Classic 3-5x on
  latency-bound loops.
- **(c) Fix AV's memory order**: iterate keys-outer/dims-inner with hd accumulators — the
  per-dim sums see keys in exactly today's ascending order (bit-identical by construction —
  `docs/task-decode-splitkv-attention.md:36`'s own principle: "The split that IS bit-identical:
  split the independent axes, never the reduction"),
  while V rows are read contiguously once instead of 1KB-strided per element.

(a)×(b) multiply; plausible combined ceiling 5-15x → attention ~1-3 ms. **This is the default
plan.** It also speeds up batched verify and CPU prefill for free (same kernel, same fold).

**A2 — an M-independent-order f32 kernel, only if A1 measures short.** Build the aikit kernel
the comment says has never existed: f32 QK^T/AV whose per-output reduction order is fixed
regardless of M, used by BOTH decode and verify — the invariant (decode==verify) is preserved
by construction, in f32. Goldens shift once (cosine re-gate + regold, the f16-scales-task
shape); spec parity gates still pass. MoE router stability (item 3) needs its own check — the
near-tie flip argument is about f32 *error* vs the f64 reference, not about M-dependence, so
A2 may be dense-only. Roughly another 2-4x over A1's f64.

**A3 — the gated FA-class f32 fast path (the plan-still-slow lever), last resort.** Precedent
exists and is exactly on point: `--metal-fast-prefill` (internal/serveapp/main.go:371) already
gates a documented not-bit-identical speed path, off by default, with the divergence spelled
out in the flag help. A `--cpu-fast-attention` twin would be off by default, refused or
ignored when speculative decoding is active, and excluded for MoE unless item 3 clears. Only
worth its config surface if A1(+A2) leave real money on the table — at ~1-3 ms post-A1, they
probably don't at decode depths; the 32k regime is where A3 might still earn its keep
(revisit with A1's long-context numbers in hand).

## Gate A0 — confirm the mechanism (cheap, before any kernel work)

1. Split the 15.16 ms: QK vs softmax vs AV vs RoPE, via the same env-gated-stub method as the
   W4A8 doc's probe 1. Confirms where (b) and (c) should land first.
2. Read `MatmulBTAcc64Strided` in aikit: does it already interleave independent outputs? If
   yes, (b) is spent and A1's ceiling drops — know before promising.
3. Single-call microbenchmark of `attendBatchedHeads` at the real 1.5B shape (nKV=2, group=6,
   hd=128) at nKeys = 128 / 512 / 2048 / 8192 — the depth curve is the baseline the
   long-context claim gets measured against later.
4. Confirm single-threadedness empirically (CPU utilization during a decode with matmuls
   stubbed) — the code read says yes; make the number say it too.

**Go/no-go:** if A0 finds the time is NOT dominated by chain-serialized/strided f64 matmul
work (e.g. softmax/exp dominates, or aikit already interleaves and the ceiling is small), stop
and re-scope — the ladder above assumes the mechanism the arithmetic points to.

### Gate A0, partial results (2026-08-23, same day, `apple-m1pro`) — items 3+4 answered

An isolated benchmark of `attendBatchedHeads` alone (real 1.5B shape: nKV=2, group=6, hd=128;
K=1 decode query against 130 keys, f64 path), run before this doc landed and folded in:

- **One layer: 404.1 µs ⇒ 11.10 ms/token across 28 layers** — independently consistent with
  the HTTP-stub method's 15.16 ms. **Correction (component split below): the gap is NOT
  RoPE** — RoPE measures at 0.053 ms/token, far too small to explain a ~4 ms difference. The
  real explanation is almost certainly the stub's averaging over the actual 64-token
  generation, where depth grows 130→193 (this benchmark holds depth fixed at 130): QK+AV scale
  ~linearly with nKeys, so the true per-token average over that range lands meaningfully above
  the depth-130 figure — roughly consistent with the gap once scaled (~161 average depth ⇒
  ~12.4 ms QK+AV alone, plus softmax/RoPE/gather bookkeeping, lands close to 15.16 ms).
- **The rate hypothesis holds almost exactly:** ~399K MACs + ~1,560 exps per layer at
  404.1 µs ⇒ **~1.0 ns/MAC** — serial-f64-FMA-chain speed, as § "Why it costs this much"
  predicted (~1.4 ns/MAC from the coarser stub number). The mechanism is confirmed at the
  order-of-magnitude level that matters: unoverlapped f64 chains, not exp/softmax cost.
- **Single-threadedness structurally confirmed:** no goroutines, no `sync.WaitGroup`, no
  worker pool anywhere in `attendBatchedHeads` or its call sites — plain nested loops over
  kv-head/query-head/position/dim.

### Gate A0 item 1, answered (2026-08-23, same day) — QK and AV are ~equal and dominant; softmax
### and RoPE are noise

Isolated benchmarks of each component alone, real 1.5B shape, depth 130, 28 layers/token:

| component | ns/token | ms/token | share |
|---|--:|--:|--:|
| QK (`MatmulBTAcc64Strided`, keys-strided) | 5,144,401 | 5.14 | 47.1% |
| AV (`MatmulBTAcc64Strided`, vals-strided) | 5,342,013 | 5.34 | 48.9% |
| softmax (max/exp/normalize, 3 passes over nKeys) | 390,607 | 0.39 | 3.6% |
| RoPE (both `ropeAt` calls) | 53,447 | 0.05 | 0.5% |
| **sum** | **10,930,468** | **10.92** | — |

Sum matches the whole-function isolated measurement (11.10 ms) to within ~2% — the residual is
plausibly per-call gather/scatter bookkeeping (`copy` into `qh`, scatter out of `ch`) that these
isolated benchmarks skip. **QK and AV are within 4% of each other and together are 96% of
attention's cost; softmax and RoPE are exactly as minor as § "Why it costs this much" predicted.**
AV running very slightly slower than QK (5.34 vs 5.14 ms) is directionally consistent with this
doc's own item (c) — AV's element reads are the ones that are 1KB-strided (`vals` read with
element stride `kvDim`), while QK's keys read is row-strided but element-contiguous.

**Answer to what this decides: (b) (interleave independent outputs, latency-chain fix) and (c)
(AV's memory order) both matter, roughly equally, and both land on real cost — this is not a
"fix one and move on" split.** A1(a) (thread across heads) benefits both QK and AV equally since
it parallelizes the outer loop both live inside; (b) should be applied to both dot-product
kernels, not just one; (c) is AV-specific and should recover AV's small excess over QK plus
whatever the strided-read cache-line cost actually is at longer depths (untested at 130 keys,
which fits comfortably in L1 either way — the real test of (c) is at the depth-curve cells below,
where the strided/contiguous distinction should matter far more).

### Gate A0 item 3, answered (2026-08-23, same day) — the depth curve, and it is worse than the
### depth-130 numbers hinted

Same isolated `attendBatchedHeads` benchmark, real 1.5B shape, 28 layers/token, at nKeys =
128/512/2048/8192 (matmul+other held at their measured 43.6 ms/token — no reason for that to
move with depth, since decode's weight-matmul cost is depth-independent).

**Reproducibility check (same day, quiet-box lesson from the STREAM/swap incident applied):**
this box was under real, visible contention when the table below was first measured (other
sessions active; a naive spot-check attempt during that contention hung twice — 16 min and 2h38m
of wall time for ~15s of real CPU work each time, a benchmark-composition bug in the throwaway
recheck code itself, not a box problem, fixed by reverting to the same plain `b.Run`-subtest shape
as the original). Re-run clean on a quiet box (other sessions closed): **depth 128 → 10.93 ms
(original 10.84, +0.8%), depth 8192 → 858.8 ms (original 828.8, +3.6%)** — both within normal
run-to-run noise. The table's numbers hold.

| depth | attention ms/token | µs/key | ratio to depth-128 | est. total ms/token | est. tok/s | attn share of total |
|--:|--:|--:|--:|--:|--:|--:|
| 128 | 10.84 | 84.7 | 1.00x | 54.4 | 18.4 | 19.9% |
| 512 | 48.57 | 94.9 | 4.48x (4x depth) | 92.2 | 10.9 | 52.7% |
| 2048 | 200.39 | 97.9 | 18.5x (16x depth) | 244.0 | 4.1 | 82.1% |
| 8192 | 828.77 | 101.2 | 76.5x (64x depth) | 872.4 | **1.15** | **95.0%** |

**Two findings, both bad news for today and good news for this doc's payoff.** First, per-key
cost is not flat — it rises **~19.5%** from depth 128 to 8192 (84.7 → 101.2 µs/key), consistent
with the keys/vals arrays (8.39 MB each at depth 8192, 16.8 MB combined) outgrowing a cache tier
that held them comfortably at depth 128 (262 KB combined) — a cost ollama's flash-attention
tiling is specifically designed to avoid. **This drift is partially A1(c)'s problem, not purely a
future tiling question**: AV's 1KB-strided element reads are exactly worst at 17 MB/layer, well
past L2 — the keys-outer/dims-inner reorder (c) proposes turns that into contiguous row
streaming, so (c) should recover some (not necessarily all) of the 19.5% as a side effect of a
move already scoped for a different reason. Whatever residual remains after (c) ships and gets
measured is the genuine tiling question — kept a named non-goal below rather than pulled into
this doc's ladder now. Second, and more importantly: **attention's share of total decode
time is 20% at depth 128 and 95% at depth 8192** — by 32k (extrapolating the measured per-key
trend conservatively flat past 8192, almost certainly an underestimate given the rising trend) the
non-attention 43.6 ms/token is noise against an attention cost in the seconds. **This confirms, with
a number instead of an inference, that decode attention is very plausibly THE long-context
deficit** (§ "The measurement" above cited the GPU-side 5.54x-at-32k figure as circumstantial; this
is the CPU-side mechanism measured directly, and it is not subtle — an estimated 1.15 tok/s at
depth 8192 is not a usable server at that depth today).

**The depth curve also flips which parallel axis A1(a) should use, and this is load-bearing for
the design, not a footnote.** At depth 130 each QK/AV call is ~16.6K MACs (item 2's own number) —
too small for within-call splitting, heads-parallel only, as scoped. At depth 8192 each call is
~1M MACs — past the fork-join economics that made within-call parallelism the wrong move at short
depth, and now a real target: QK's per-key scores are independent outputs (split across keys, per
`task-decode-splitkv-attention.md:36`'s own principle — split the independent axis, never the
reduction) and AV's per-dim folds are equally independent (split across dims), neither touching
any output's reduction order. **A1(a) should be depth-aware — heads-parallel at short depth, plus
within-call independent-axis splits once a call's own work clears its own threshold — not
heads-only.** This matters concretely: heads-parallel alone caps near 6x on this box (this
session's own P/E-core findings), and the 95%-share regime at 8k+ is exactly where more than 6x is
worth having. (Same threshold-awareness item 2 already established — check M·N·K against a real
threshold before assuming a shape either parallelizes or doesn't.)

**This changes the acceptance framing for A1/A2 at long context: fixing the 20%-of-token cost at
depth 130 is worthwhile on its own terms, but fixing the SAME code path is worth vastly more at
long context, where it is not one lever among several — it is nearly the whole token.** Whether
A1's thread-across-heads win (a constant-factor speedup, independent of depth) is enough on its
own at 8k+, or whether the memory-order/cache-tier problem (c) plus eventually a tiling scheme
becomes load-bearing at that regime, is the natural next question once A1 ships and this depth
curve gets re-measured against it.

### Gate A0 item 2, answered (2026-08-23, same day) — existing infra doesn't compete with A1;
### it's a second instance of a known bug class

`MatmulBTAcc64Strided` (`linalg/matmul_strided.go:30` in aikit) DOES call `parallelCols(M*N*K, N,
...)` internally — it has real parallel capability. But at the shape `attendBatchedHeads` calls it
with (`decoder/forwardn.go`'s QKᵀ call: `M=K=1, N=nKeys=130, K=hd=128` ⇒ **M·N·K = 16,640 MACs**;
the scores·V call is the same magnitude), against aikit's package-default threshold
(`parThreshold = 1<<24 = 16,777,216`, `linalg/linalg.go:58` in aikit): **16,640 is ~1000x below
threshold.** Every one of these calls takes `parallelCols`'s serial fast path, every time, with
no exception.

**This is the SECOND confirmed instance of a named bug class — same pattern, different kernel.**
`decoder/weightmat.go`'s `int4ParThreshold` comment documents the FIRST instance precisely: on
Gemma-4's int4/W4A8 decode matmuls (M=1), every real shape (expert gate‖up 3.96M, down 1.98M,
dense ~5.9M, attention-proj ~11.5M MACs) fell under this SAME `parThreshold = 1<<24` default and
ran serial, capping 8-core scaling at 1.61x — fixed by introducing `int4ParThreshold = 1<<20` as a
lower, decode-appropriate threshold for that one call path. This finding is not "W4A8 again" — it
is the identical class (a kernel with real parallel capability, gated on aikit's one
prefill-tuned default, never firing for decode-sized calls) recurring in a completely different
kernel (`MatmulBTAcc64Strided`, attention's QK/AV, not any weight matmul). **Two confirmed
instances now; grep `parThreshold = 1<<24` (aikit) and `int4ParThreshold` (goinfer) before adding
a third decode-time matmul/kernel call anywhere in this codebase — check its own M·N·K against
the default before assuming it parallelizes.**

**But the fix is NOT "lower this threshold"** — parallelizing *inside* one 16,640-MAC call (attn
has 24 of these per layer: 12 heads × {QK, AV} × 28 layers = 672/token) would spawn a fork-join
around a sliver of work each time; the aikit int4 story's own lesson (thr=0 "over-parallelizes"
on the Ryzen box) says this shape is exactly wrong for that. **The correct fix is still A1(a) as
scoped: parallelize ACROSS the 12 independent per-head chains** (one goroutine per head/group,
each running its OWN unchanged serial QKᵀ→softmax→AV), not within any single call's N dimension.
That is a `decoder/attention.go`/`forwardn.go` restructuring, not an aikit threshold change.

**Answer to Gate A0 item 2's stated concern: existing infra does not already interleave in a way
that competes with or shrinks A1's ceiling.** If anything this strengthens the diagnosis — it
independently confirms WHY today's path is serial (threshold, not a hand-written serial loop
lacking any parallel primitive) without touching A1(a)'s available parallelism at all. The go/no-go
was waiting on this; **it resolves GO, ceiling unchanged.**

### Gate A0 closed (2026-08-23) — GO, all four items answered

| item | question | answer |
|---|---|---|
| 1 | QK vs softmax vs AV vs RoPE split | QK 47.1% / AV 48.9% / softmax 3.6% / RoPE 0.5% — (b) and (c) both matter, roughly equally |
| 2 | does existing infra already interleave (shrinks A1's ceiling)? | No — 1000x under aikit's default threshold, never fires; A1(a)'s ceiling is untouched, and this is the second confirmed instance of the same threshold-mismatch bug class `int4ParThreshold` first fixed for Gemma-4's W4A8 decode matmuls |
| 3 | depth curve, long-context baseline | 20% of token at depth 128 → 95% at depth 8192 (est. 1.15 tok/s); per-key cost itself rises ~19.5% over that range (cache-tier effects) — this is very plausibly the long-context deficit's mechanism, not just correlated with it |
| 4 | single-threaded? | Confirmed structurally (no goroutines anywhere) and by rate (~1.0 ns/MAC, serial-f64-chain speed) |

**Mechanism confirmed exactly as hypothesized: unoverlapped, single-threaded f64 chains in QK and
AV equally, not softmax/RoPE, not existing-but-unused parallel infrastructure.** A1's three moves
(thread across heads, interleave independent outputs in both QK and AV, fix AV's strided memory
order) are cleared to build against the acceptance table below, with the added, sharper framing
from item 3: the long-context payoff is not a bonus on top of the depth-130 win, it is where most
of the value actually is.

## A1 implementation — move (c) landed (2026-08-23, same day)

Per `docs/prompts/attention-a1-bit-identical-restructure.md`'s build order: (c) first.

**Built:** `aikit/linalg/matmul_av_acc64.go`'s `MatmulAVAcc64` — keys-outer/dims-inner, hd
independent f64 accumulators, replacing `MatmulBTAcc64Strided`'s dims-outer/keys-inner walk at
the AV call site only (`decoder/forwardn.go`'s `attendBatchedHeads`). QK is untouched (still
`MatmulBTAcc64Strided`) — that's move (b). Threaded `avAcc []float64` scratch through
`attendBatchedHeads`'s signature and all 6 call sites (3 decode in `attention.go`, 3 batched in
`forwardn.go`); goinfer's `decodeScratch` grew one new field (`avAccBuf`), forwardn.go's batched
scratch grew one new local. Developed against aikit via `go.work` (`../aikit` added to `use`) — a
local-dev convenience, not yet an aikit version bump; that's still owed before this fully lands,
per the brief's "Done looks like."

**Correctness — the bit-identity contract held exactly, not approximately:**
- `TestMatmulAVAcc64_exactMatchesStrided` (aikit `linalg`): exact `==` against
  `MatmulBTAcc64Strided` across 10 shapes (every nKeys mod-4 residue, hd ∈ {16,64,80,96,128}, M=1
  decode and M>1 batched, nKeys up to 8192). All pass.
- `TestForwardN_matchesSequential`, `TestSpeculativeGreedyParity`: **zero** logit difference
  (`worst max logit diff 0.00e+00`) — not "still under the cosine bar," genuinely unchanged.
- Full `go test ./decoder/...`: green except the pre-existing, unrelated `TestParityManifest_fresh`
  staleness flag (expected on any core-file touch) and `TestMellum2_logitParity`/
  `TestMellum2_windowParity`, confirmed to fail IDENTICALLY on the unmodified baseline (a missing
  safetensors shard on this box, `~/models/mellum2-unq/model-00002-of-00005.safetensors` does not
  exist) — reproduced before restoring the change, not assumed. T3 (`go run ./cmd/gate parity -v`):
  every gate that had its asset present passed (gemma4-E2B/E4B, gemma4-12B, mixtral, gpt2,
  qwen3_5_moe-tiny, deltanet, deepseek-tiny, kimi-tiny, phi3-tiny, llama4-tiny, nemotron-tiny,
  nemotron3nano-tiny); the other 28 registered assets are simply not on this Mac (documented,
  pre-existing, not this change's concern).

**Performance measured, isolated AV kernel first** (real 1.5B shape, order-alternated best-of-3):

| depth | before (`MatmulBTAcc64Strided`) | after (`MatmulAVAcc64`) | speedup |
|--:|--:|--:|--:|
| 130 | 16,157 ns | 8,929 ns | **1.81x** |
| 8192 | 1,353,207 ns | 565,038 ns | **2.39x** |

Then the full wired-in `attendBatchedHeads` depth curve (same shape as Gate A0's baseline table):

| depth | before (ms/token) | after move (c) (ms/token) | speedup | attn share was |
|--:|--:|--:|--:|--:|
| 128 | 10.84 | 8.74 | 1.24x | 19.9% |
| 512 | 48.57 | 35.12 | 1.38x | 52.7% |
| 2048 | 200.39 | 137.86 | 1.45x | 82.1% |
| 8192 | 828.77 | 558.62 | 1.48x | 95.0% |

**Speedup grows with depth, as predicted** — AV's strided reads are worst exactly where the
cache-tier drift lives, so (c) claws back part of the +19.5% per-key cost rise alongside its base
win. The full-call speedup (1.24-1.48x) is smaller than AV-alone's (1.81-2.39x) because QK
(unfixed until move (b)) is still ~47% of the cost — consistent with item 1's finding that QK and
AV are co-equal and neither move alone reaches the ≥3x acceptance bar. **(c) alone is not the
acceptance target; it's the first of three multiplying factors.**

**Disposition:** wired into production (not a side-by-side kept-for-comparison kernel like the
W4A8 doc's items 1+2 attempt) — this one is a clean, unconditional win with the bit-identity
gates green, so there was nothing to gate behind a flag. Uncommitted pending moves (b) and (a),
per the brief's per-move measurement discipline.

## A1 implementation — move (b) landed (2026-08-23, same day)

**Built:** `aikit/linalg/matmul_qk_acc64.go`'s `MatmulQKAcc64` — 8 keys' dot products run as 8
concurrent f64 accumulator chains in one pass over d, replacing `MatmulBTAcc64Strided` at the QK
call site only. Unlike move (c), QK's row reads were already contiguous (`bElemStride=1`) — the
lever here is pure FMA-latency hiding, not memory layout: `dotF32Acc64`'s single sequential
accumulator can't issue its next add until the previous one resolves, even though each key's dot
is fully independent of every other key's. **Measured 4 vs 8 wide before picking**, per the
brief's "4 (or 8) — measure": 4-wide gave 3.03-3.05x, 8-wide gave 4.41-4.42x (both depths,
isolated kernel bench) — 8-wide shipped, the 4-wide file removed rather than kept alongside.

**Correctness:** `TestMatmulQKAcc64_exactMatchesStrided` (aikit `linalg`), exact `==` across every
nKeys residue mod 8 (128-135), hd ∈ {16,64,80,128}, M=1 and M>1, nKeys up to 8192 — all pass.
Re-ran `TestForwardN_matchesSequential`/`TestSpeculativeGreedyParity`/the ring-parity suite with
both (c) and (b) wired in: **still zero logit difference.**

**Performance, isolated kernel** (real 1.5B shape, order-alternated best-of-3):

| depth | before (`MatmulBTAcc64Strided`) | after (`MatmulQKAcc64`, 8-wide) | speedup |
|--:|--:|--:|--:|
| 130 | 15,266 ns | 3,463 ns | **4.41x** |
| 8192 | 970,823 ns | 219,500 ns | **4.42x** |

**Depth-independent, unlike move (c)** — confirms the diagnosis: this is a pure latency fix (the
same win at every depth), where (c) was a memory-order fix (a bigger win at long depth, where the
strided-read penalty is worse).

**Cumulative depth curve, both moves (c)+(b) wired in** (same shape as Gate A0's baseline table):

| depth | before | after (c) | after (c)+(b) | cumulative speedup |
|--:|--:|--:|--:|--:|
| 128 | 10.84 ms | 8.74 ms | 4.54 ms | **2.39x** |
| 512 | 48.57 ms | 35.12 ms | 17.70 ms | **2.74x** |
| 2048 | 200.39 ms | 137.86 ms | 72.03 ms | **2.78x** |
| 8192 | 828.77 ms | 558.62 ms | 298.36 ms | **2.78x** |

**Already close to the ≥3x acceptance bar at depth 130 with one move still to go.** Move (a)
(thread across the 12 independent heads — up to ~6x more on this box, depth-aware per the earlier
design note) is the remaining multiplier; acceptance is measured after all three land, not per-move.

**Disposition:** wired into production, same as (c) — a clean, unconditional win, nothing to gate.

## A1 implementation — move (a) landed, A1 COMPLETE (2026-08-23, same day)

**Recalibration applied before building** (per feedback on the (c)+(b) results): (c)+(b) shrank
per-call work ~2.4x, so per-head chunks at depth 130 are ~13-14 µs, not the ~34 µs the brief
estimated — still comfortably above goroutine fan-out overhead, but with a thinner margin than
originally scoped. Built and measured against that, not the original estimate.

**Built:** a `headWorkerScratch` pool (`decoder/scratch.go`) — qh/scores/ch/avAcc per worker (kh/vt
included only for the untouched f32-fallback path, which stays serial and test-only), capped at
`maxAttnWorkers = 6` (P-core match). `attendBatchedHeads` (`decoder/forwardn.go`) now extracts a
closure, `attendOneHead`, and fans the nH independent query heads across `min(len(pool), nH)`
goroutines when `useAcc64` — the acc64 path only, since kh/vt (needed by the f32 fallback) aren't
touched there at all. Below `attnHeadsParThreshold` (or with nH≤1 or a 1-slot pool), it runs
serially through `pool[0]` — no fork-join for work too small to amortize it, the same discipline
`int4ParThreshold` already established one level down. **Threshold left at 0 (always parallelize
when possible)** — measured rather than guessed: even at depth 128 the fork-join paid for itself
clearly (see below), so no floor was needed in practice on this box/shape.

Batched (M=K>1) attention deliberately gets a 1-element pool (serial by construction) — threading
prefill/verify's heads is explicitly out of scope (the brief: "no M>1-specific work here"); batching
already amortizes the weight-matmul cost the way threading amortizes decode's per-token fork-join,
so there's no equivalent pressure to relieve there.

**The two named traps, both handled:**
- **Shared scratch** — solved by the pool itself; each worker's qh/scores/ch/avAcc are its own,
  never touched by another goroutine.
- **The local-ring path's deferred KV write** — untouched: `attendBatchedHeads` never writes KV
  (that's the caller's `cache.Append`/`commitBatch`, outside this function entirely), and this
  parallelizes an existing loop's iterations, not the surrounding call order — nothing about when
  the ring write happens relative to the read changed.

**Correctness:** `go test -race` on `TestForwardN_matchesSequential`, `TestSpeculativeGreedyParity`,
the ring-parity suite, `TestAttendBatchedHeads_vsNaive`, `TestAttendStrided_matchesGatherReference` —
**all green, no races detected, still zero logit difference.** The race detector is the load-bearing
check for a genuinely concurrent change; it ran clean, not just the bit-identity gates.

**Performance — the depth-aware shape held, and then some:**

| depth | before | after (c)+(b) | after (c)+(b)+(a) | cumulative speedup | (a)-only factor |
|--:|--:|--:|--:|--:|--:|
| 128 | 10.84 ms | 4.54 ms | 2.81 ms | **3.86x** | 1.62x |
| 512 | 48.57 ms | 17.70 ms | 7.80 ms | **6.23x** | 2.27x |
| 2048 | 200.39 ms | 72.03 ms | 19.65 ms | **10.20x** | 3.66x |
| 8192 | 828.77 ms | 298.36 ms | 85.23 ms | **9.72x** | 3.50x |

**Clears the ≥3x acceptance bar at depth 130 with room to spare (3.86x), and the long-context
payoff is exactly where the campaign said it would be** — 10.2x/9.72x at 2048/8192, not a "modest
factor," because per-head chunks at those depths (≈700 µs–~7 ms) sit far above any fan-out floor.
Even at depth 128, where the recalibrated ~13-14 µs chunks left a real question, threading still
delivered 1.62x cleanly — no threshold was needed to avoid a regression there.

**Real end-to-end confirmation (`bench_peer` method, decode-only, warm-up discarded, greedy, real
serve binary):**

| model | before (original diagnosis) | after A1 | speedup |
|---|--:|--:|--:|
| 0.5B, depth 128 | 34.5 tok/s | 40.71 tok/s | 1.18x |
| 1.5B, depth 128 | 17.0 tok/s | **21.52 tok/s** | **1.27x** |

**Clears the acceptance table's ≥21 tok/s target.** The depth-2048 end-to-end cell was abandoned
mid-run — not a stall, but a benchmark-design mistake: the harness independently re-prefills the
full 2048-token prompt for all 17 completions (1 warmup + 2×8), and batched prefill's O(L²)
attention is untouched by A1 by design, so that cost (not decode) dominated and made the cell take
tens of minutes for no additional information the isolated depth curve above didn't already give
more cheaply. Killed after ~40 minutes once the CPU-time/wall-time ratio made clear it was real,
unproductive work, not a hang.

**Cross-validation, not just two separate measurements agreeing by luck:** Amdahl's law on the
1.5B/depth-128 numbers — 43.6 ms (matmul, untouched by any A1 move) + 2.81 ms (new attention,
isolated) = 46.41 ms/token → **21.55 tok/s predicted**, against **21.52 tok/s measured**. The
isolated kernel benchmarks and the real server's end-to-end rate agree to within 0.03 tok/s — the
strongest evidence in this campaign that the isolated numbers mean what they claim to mean.

**A1 is complete: all three moves landed, wired into production, bit-identical (proven, not
argued — exact-equality kernel tests plus zero-diff whole-model gates under `-race`), and the
acceptance bar cleared at both the isolated-kernel and real-decode-rate level.**

**Resolved (2026-08-23): released as aikit v1.25.0.** `MatmulQKAcc64`/`MatmulAVAcc64` shipped
through the full `RELEASING.md` root-module ritual (releasegate 4/4, perfgate flat vs v1.24.0,
vulncheck 11 clean/0 vulnerable/4 unscanned-and-explained) as a single release event, tagged, gpu
backends bumped via `gpupins --fix`. goinfer's `go.mod` now requires the real tag; the `go.work`
override is gone. T3 (`go run ./cmd/gate parity`) re-ran against the tag: every tiny/synthetic
family the restructure touches (gemma4, mixtral, gpt2, qwen3_5_moe-tiny, deltanet, deepseek-tiny,
kimi-tiny, phi3-tiny, llama4-tiny, nemotron-tiny, nemotron3nano-tiny) passes with zero diff; the
only real failures are the pre-existing Mellum2 missing-asset issue, confirmed unrelated to this
change. Manifest hashes refreshed.

**`bench_peer` re-measured against the tag (2 runs, decode-only, warm-up discarded):**

| model | run 1 | run 2 | mean |
|---|--:|--:|--:|
| 0.5B, depth 128 | 38.73 tok/s | 39.16 tok/s | 38.9 tok/s |
| 1.5B, depth 128 | 19.54 tok/s | 20.74 tok/s | 20.1 tok/s |

This came in ~5-7% below the 40.71/21.52 tok/s figures measured earlier against the `go.work`
override — notably, the 1.5B mean (20.1) falls short of the ≥21 tok/s bar this same box cleared
hours earlier. The kernel code did not change between those two measurements (the tag sits on a
CHANGELOG-only commit past the feature commit the override pointed at), and `uptime` during this
re-measurement showed load averages around 5.2-5.9 — this machine had two Claude Code sessions and
VS Code active throughout the release process, unlike the quieter state the original number was
likely measured under. **Attributed to ambient system load, not a code regression**, but recorded
honestly rather than restated as a clean pass: the isolated depth-curve kernel benchmarks (the
3.86x/6.23x/10.20x/9.72x figures above) are the load-bearing acceptance evidence precisely because
they're a same-run before/after ratio, insensitive to ambient load in a way an absolute tok/s
figure is not.

**Not left indefinitely optional.** Re-running `bench_peer` on an otherwise-idle box isn't only
about re-certifying the ≥21 tok/s absolute number for A1 — these load-5.9 cells would otherwise
become the *before* baseline for the W4A8 item-3 campaign, and a contaminated baseline there would
repeat the swap-incident risk that nearly compromised this campaign's own Gate 0. **Do this as the
first act of the W4A8 item-3 work**, on an idle box: it re-certifies A1's absolute number and
establishes a clean before baseline for item-3 in the same run.

## Acceptance math — tied to the W4A8 doc's revised table (1.5B, ms/token)

| scenario | matmul | attn | other | total | tok/s | vs ollama |
|---|--:|--:|--:|--:|--:|--:|
| today | 43.2 | 15.2 | ~0.4 | 58.8 | 17.0 | 0.25x |
| A1 alone (attn ~3 ms) | 43.2 | 3.0 | 0.4 | 46.6 | 21.5 | 0.31x |
| A1 + W4A8 item-3 kernel at 2x | 21.6 | 3.0 | 0.4 | 25.0 | 40.0 | 0.59x |
| A1 + bandwidth-class kernel (~12.5 ms) | 12.5 | 3.0 | 0.4 | 15.9 | 62.9 | 0.92x |
| A2/A3 attn (~1 ms) + bandwidth-class kernel | 12.5 | 1.0 | 0.4 | 13.9 | 71.9 | ~1.05x |

**With this lever real, parity stops being out of reach** — the W4A8 doc's "0.84x is what
close-the-gap realistically means" was computed with a 5 ms non-matmul guess; A1's target beats
that. Acceptance for A1: attention component ≥3x faster at depth ~130, `bench_peer` decode
gain consistent with the table, **all parity/spec gates green with zero golden changes**, plus
one long-context cell (depth ≥8k) to size the depth-scaling claim. A2/A3 get their own
acceptance if funded, including the regold/cosine gate (A2) or the flag-off-by-default
divergence documentation (A3, matching the `--metal-fast-prefill` help-text convention).

## A2/A3 disposition

**Closed for now, revisit if long-context work post-W4A8 wants more.** At depth 128, A1 already
brought attention down to ~2.8 ms/token; the acceptance table above puts A2/A3's remaining upside
at roughly another ~2 ms (attn ~3.0 ms → ~1.0 ms in the "A2/A3 attn + bandwidth-class kernel" row).
That's real but modest next to the config-surface and golden-churn cost each would add — A2 needs
its own regold/cosine-tolerance gate (softmax numerics stop being order-preserving), A3 needs a
documented default-off divergence flag matching `--metal-fast-prefill`'s precedent. Neither is
funded by this doc. Long-context depths are where the remaining ~2ms would matter proportionally
more (attention is the dominant cost by depth 2048+ already), so the natural trigger to revisit is
long-context work landing after the W4A8 item-3 kernel, not this campaign continuing further on
its own.

## The prefill deferral has a MEASURED cost (added 2026-08-25, queue G16)

A1's scope-out — *"Prefill/verify (M=K) fast-pathing beyond what (b)/(c) give for free … no M>1-specific
work here"* — was the right call for A1 and is recorded here as the *cost* of that call, not as a
reversal. It is the discipline this doc already applies to its own negatives.

`runLayersFromEmbedN` implements the deferral literally: `newHeadWorkerPool(1, K, maxKeys, hd)`,
with the comment *"this stays serial by construction (pool len 1 always takes attendBatchedHeads's
serial branch)"*. Decode got A1's fan-out; prefill did not.

**What that costs, measured on an M1 Pro (dense qwen2.5-coder-1.5b, prefill + 1 token,
`docs/queue-performance.md` G15/G16):**

| prompt_tokens | `int8int8` prefill | effective |
|---|---|---|
| 170 | 3.3 s | 51.5 tok/s |
| 620 | 19.7 s | 31.5 tok/s |
| 1520 | 93.2 s | 16.3 tok/s |
| 3020 | 334.9 s | 9.0 tok/s |

CPU sampled every 2 s during a large prefill: `99.6 111.0 270.2 99.7 168.4 99.1 102.5` on a box with
**8 logical / 6 performance cores**. The ~100% floor is this deferral; the bursts are aikit's
`MatmulBTW8A8Batch`, which *does* fan out over the concatenated column space. So the weight matmuls
are threaded and attention is not — and because serial attention is O(K²) while the parallel matmuls
are O(K), attention's share **grows with prompt length**, which is why effective tok/s falls as the
prompt grows in both quants.

**The consequence, in one line:** an agent-shaped prompt (a 4 KB system prompt plus 25 tool schemas
≈ 8k tokens) could not be prefilled inside any harness's idle timeout on CPU. That is what surfaced
this — a real consumer session, not a benchmark.

**A1's own constraint does not forbid the fix.** *"Parallelism may only split independent outputs
across workers/registers — heads, layers' KV groups, individual QK scores, individual AV dims"* —
heads are named first. Threading prefill attention's heads is inside the bit-identity guarantee, not
a renegotiation of it. Estimated ceiling ~4–5×; tracked as G16, not opened here.

## Non-goals

- **Norms, sampler, KV-append.** Measured noise-level (probe 1); the KV-append number
  (4.29 µs/token) is already isolated and filed.
- **GPU/Metal attention.** `task-decode-splitkv-attention.md` owns that; this doc borrows its
  independent-axis argument, not its scope.
- **Softmax numerics, exp approximations.** Order-preserving restructuring only, until A2 is
  explicitly funded.
- **The W4A8 kernel items.** Other doc; the only coupling is the acceptance table above.
- **Changing what `acc64` guarantees.** A1 preserves it outright; A2/A3 renegotiate it only
  through their own gates, never silently.
- **Prefill/verify (M=K) fast-pathing.** Still out of scope *for A1*, as originally written — but
  see "The prefill deferral has a MEASURED cost" above before quoting this bullet as a standing
  decision. It deferred work that is now measured at ~4–5× on the CPU lane (G16).
- **Tiling the long-context cache-tier drift.** A1(c)'s memory-order fix should recover some of
  the measured ~19.5% per-key cost rise past L2, as a side effect. Whatever residual remains after
  (c) ships and gets re-measured is a genuine flash-attention-style tiling question — real, but
  explicitly out of this doc's ladder, not scope creep to absorb here.

## Done looks like

A0's split + kernel-read + depth curve recorded here with a go/no-go. If go: A1's three moves
in aikit/goinfer, gates green with zero golden churn, before/after `bench_peer` at depth ~130
and one ≥8k cell, and the acceptance table above re-filled with measured numbers. Then a
recorded decision on A2/A3 — funded with their own gates, or closed with the measured reason. A
negative at any step is filed at the same standard as the W4A8 doc's.

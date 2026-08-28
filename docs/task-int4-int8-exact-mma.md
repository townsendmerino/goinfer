# Investigation: an integer-exact int4×int8 GEMM on Metal's f16 MMA hardware

**Status: NEGATIVE RESULT at Phase 1. Stopped per the task's own instructions before any kernel
code was written. No files outside this doc were touched — decode path, prefill.go, and the
release path are all unmodified.**

## Task recap

Hypothesis under test: Metal's `simdgroup_matrix` MMA hardware (f16/f32 operands only — no int8
MMA exists on any Apple GPU) could still reproduce decode's bit-exact int4×int8 result, because
int8 activations and int4 weights are exactly representable in f16, their products are exact, and
f32 accumulation of integer sums is exact below 2^24. The proposal: feed raw quantized integers
into the MMA as f16, get exact per-K-block integer partial sums, then apply scales after each
block sum "in the same order and precision decode does" — closing prefill's known representational
divergence from decode (`docs/ollama-chase.md` §A2-Metal) while keeping simdgroup throughput.

Per the task's explicit gating: Phase 0 must establish decode's real numeric contract before any
kernel is written, and if Phase 0 or Phase 1 kills the approach, stopping there with a documented
reason is a complete, successful outcome — not a failure to push past.

## Phase 0 — decode's actual numeric contract

### Which kernels

Decode's dense per-token projections dispatch through the `gemv_w4a8_sa` family
(`metal/kernels.go:301-327`, wired at `metal/model.go:458`, called at `metal/model.go:1327,
1077, 1096, 1130, 1139, 1143, 1196, 1268, 1292, 1314, 1318` for QKV/O-proj/gate-up), plus
`gemv_w4a8_coal` (`metal/kernels.go:237-211`, wired at `metal/model.go:456`, called at
`metal/model.go:1356, 1387` for the down-projection). Both share the identical numeric structure
below — `gemv_w4a8_sa`'s `SA_BODY` macro (`metal/kernels.go:287-252`) and `gemv_w4a8_coal`'s
`W4A8_BODY` macro (`metal/kernels.go:220-186`) differ only in memory-access pattern (uint4-staged
vs per-word), not in arithmetic order or precision. This is decode's real, shipped contract for
every dense int4×int8 weight matrix in the model — the thing the new kernel needs to match.

### (a) Exact integer dot per block, but the scale is applied PER BLOCK, not once at the end

Per 32-element K-block, the kernel unpacks 32 int4 nibbles (zero-point offset −8, so range
[−8, 7]) and forms an exact int32 dot against 32 int8 activations (range [−127, 127]):

```c
int gi = UNP8(w.x,a) + UNP8(w.y,a+8) + UNP8(w.z,a+16) + UNP8(w.w,a+24);  // exact int32
acc += float(gi) * float(sr[g]);   // sr[g] = this block's half-precision weight scale
```

`gi` is exact (see Phase 1 overflow bound below), and `float(gi)` is an exact int→float
conversion. But `float(sr[g])` is a real-valued (non-integer) half-precision scale — the multiply
`float(gi) * float(sr[g])` is a **lossy, rounded** floating-point op, performed **inside the
K-loop, once per 32-element block**, not deferred to a single whole-row scale at the end. This
matches the task's own hypothesis phrasing ("apply scales after **the** block sum", singular per
block) — so decode's contract is not "int-dot-over-all-K then one scale," it's "int-dot-per-block
then scale-per-block," repeated across blocks.

### (b) Accumulator precision and cross-block/cross-lane order

`acc` is `float` (f32). Each of the 32 SIMD lanes in a simdgroup owns a **strided** subset of the
row's blocks (`for (uint g=lane; g<G; g+=32u)`) — lane 0 handles blocks {0, 32, 64, …}, lane 1
handles {1, 33, 65, …}, etc. Within a lane, its blocks are summed sequentially into `acc` in that
strided order (each addition already lossy per (a), so this partial sum is order-dependent).
The 32 lanes' partial sums are then combined via `simd_sum(acc)` — a hardware SIMD-group
reduction intrinsic whose internal combination tree Apple's MSL spec does not document (and which
is not guaranteed stable across GPU generations/driver versions).

### (c) Where each scale is applied, and in what precision

- **Weight block-scale** (`sr[g]`/`bsc`, half precision, one value per 32-element K-block per
  output row): applied **inside** the K-reduction, once per block, promoted to f32 for the
  multiply. This is the lossy step from (a).
- **Activation scale** (`asc[0]`, f32, one value per token — set once by `rmsnorm_quant` /
  `quant_vec`, `metal/kernels.go:51,70`): applied **exactly once**, after the full K-reduction
  and the cross-lane `simd_sum`, at the final store: `out[gid] = acc * asc[0]` (`gemv_w4a8_coal`,
  `metal/kernels.go:241`) / `out[row] = acc*asc[0]` (`gemv_w4a8_sa`, `metal/kernels.go:307`).

### (d) Quant block size and scale layout

Block size is **exactly 32** along K, confirmed from source both ways: `gemv_w4a8_sa`'s
`G = K>>5u` (`metal/kernels.go:290`) and `gemv_w4a8_coal`'s 4-word (32-nibble) groups indexed by
`wi>>2` (`metal/kernels.go:231`). Scale layout is one half-precision scale per (output row, 32-K
block) — row-major, `sct`/`bsc` indexed by `row*G + g` (`metal/kernels.go:293,292` /
`(K/32u)` stride at `metal/kernels.go:223`). This confirms the task's "presumably 32" guess.

### Phase 0 verdict

Decode's contract is: **exact-integer dot per 32-element block → immediate lossy scale multiply
per block (f32, weight scale) → order-dependent float accumulation across blocks within a lane →
opaque hardware `simd_sum` reduction across 32 lanes → single lossy scale multiply per token
(f32, activation scale) applied once at the very end.** This is compatible with the task's
hypothesis as stated (which anticipated per-block scaling), so per the task's gating this proceeds
to Phase 1 rather than stopping here.

## Phase 1 — feasibility

### Overflow: not the blocker

Per-block bound is independent of the model's total K (each block is always exactly 32 terms):
max `|nibble|` = 8, max `|activation|` = 127, max `|term|` = 1016, max `|gi|` over 32 terms =
32,512. This is exact in int32 and exact when converted to f32 (≪ 2^24 ≈ 16.7M). Every shipped
model's H/I dims are multiples of 32 (needed for the block layout to exist at all), so this bound
holds uniformly — overflow is a non-issue regardless of which model or K dimension is largest.

### The actual blocker: decode's cross-block combine, not MMA's internal order

The first cut at this reasoning (below, corrected) blamed `simdgroup_matrix`'s undocumented
internal K-reduction order. That is a **red herring** given the exact-integer premise, and it's
worth stating precisely why, because the correct root cause has different — and more useful —
corollaries.

**MMA's internal order doesn't matter for the per-block integer sum.** `gi` (the 32-term int32
dot for one block) is exact, and Phase 1's overflow bound (max 32,512) keeps it exact after
conversion to f32. Exact arithmetic is order-independent — reassociation only changes the result
once rounding has occurred. So however `simdgroup_matrix` internally sums the 8-wide K-chunks
that make up one block's contribution, the result is the same exact value decode's scalar loop
would produce for that block, regardless of order. Whether Apple ever documents that internal
order is irrelevant to this step.

**The genuine, unfixable-without-touching-decode blocker is one step later: decode's cross-block
combine.** The first lossy op, `float(gi) * float(sr[g])` (Phase 0(a)), is a plain IEEE-754
multiply — deterministic and exactly matchable by any kernel that computes the same `gi` against
the same scale, MMA-based or not. The problem is what happens to that already-lossy per-block
term next: decode accumulates it into a *lane-strided* running sum (lane *i* sequentially sums
blocks {i, i+32, i+64, …}), then combines the 32 lanes' partial sums via `simd_sum` — an
implementation-defined hardware reduction tree with no MSL-specified order and no API to pin,
query, or introspect it. Because these are sums of *already-rounded* terms, this combine step is
genuinely order-dependent (reassociation changes the bits here, unlike the exact block sum
above). A structurally different kernel — MMA-based or otherwise — cannot replicate an
opaque, hardware-owned combine order it has no way to observe or control. This is the actual
kill: it has nothing to do with MMA specifically, and everything to do with decode's own
`simd_sum` being unpinnable from outside decode's kernel.

A secondary, non-fatal-but-relevant point: even setting the order problem aside, applying a
scale *every 32 K-elements* (= every 4 of the natural 8-deep MMA K-steps) means the accumulator
tile must be flushed to a scalar per-row running sum after every block, before the next block's
raw integer contributions can begin (the task's own anticipated "round-trip through threadgroup
memory" option). That serializes the MMA pipeline back down to near-scalar throughput on the
critical path, eroding most of the reason to use MMA in the first place.

### Phase 1 verdict: kills the hypothesis — and pins the reason precisely

Bit-identity to decode cannot be established **without modifying decode's own reduction** — e.g.
replacing `simd_sum` with an explicitly-ordered shuffle tree that both decode and the new kernel
share. The task correctly forbade touching decode, and that path is almost certainly not worth
opening anyway even considered on its own: pinning decode's reduction order changes decode's
actual output bits on Metal, which means re-blessing every parity golden across every
Metal-eligible family — a cross-cutting cost for a kernel that's already been declined on other
grounds (the P10 Metal-leg ceiling was ~1.13x even assuming free/bit-identical drafting; see
`docs/spec/08-dspark-dflash.md`).

This precision matters for two reasons: first, "MMA order is undocumented" would wrongly suggest
the approach revives if Apple ever documents `simdgroup_matrix`'s internal order — it wouldn't;
the blocker is decode's `simd_sum`, not MMA, and Apple documenting MMA changes nothing about
that. Second, it clarifies what the blocker actually *forbids* and, symmetrically, what it does
**not** forbid: a kernel that keeps decode's exact reduction *shape* — lane-strided accumulation
into `simd_sum`, just with the token loop hoisted outward — inherits bit-identity for free. See
Corollary below.

## Outcome

**Stopping here per the task's explicit instruction** ("If Phase 0 or Phase 1 kills the approach,
stop there and write up why — a clean negative result recorded in the doc is a fully successful
outcome"). No kernel code was written (Phase 2/3 not attempted, correctly — building and
benchmarking a kernel that's already known not to be bit-identical would not answer the question
this investigation was scoped to answer). No changes were made to `metal/prefill.go`, any decode
kernel, or any file outside this doc.

This does not change the standing conclusion already recorded in `docs/ollama-chase.md` §A2-Metal
and `docs/spec/08-dspark-dflash.md` (Metal section): the representational gap between prefill's
f16-MMA activations and decode's int8 activations is not closeable by feeding decode's raw
quantities through the MMA unit. The precise reason (see Phase 1's corrected verdict above) is
not that the integer arithmetic is unmatchable — it's exact and order-independent — but that
decode's post-scale cross-block combine (lane-strided sum → `simd_sum`) is an opaque,
hardware-owned reduction with no way to observe or replicate its order from a structurally
different kernel. A real bit-identity fix, if ever wanted, would need to unify decode itself onto
a pinned, explicitly-ordered reduction shared with the batched path (i.e. change decode's
`simd_sum`, not add a matching batched kernel) — out of scope here per the task's constraints
("no changes to decode numerics"), and not worth opening on its own merits given the parity-golden
re-blessing cost against a kernel already declined on other grounds.

**Corollary — option 2 (a batched small-M decode kernel) survives fully, and is now provably the
only bit-identical path.** The blocker identified above is specific to a *structurally different*
kernel (MMA) trying to replicate decode's reduction from outside it. A kernel that instead hoists
an M-loop into the *existing* `gemv_w4a8_sa`/`gemv_w4a8_coal` structure — same lane-strided
accumulation, same `simd_sum`, same per-block scale placement, just looped over M tokens instead
of dispatched once per token — inherits decode's exact reduction shape by construction, not by
matching it after the fact. It gets bit-identity for free, which is the one property the MMA
route can never have. If P10-on-Metal (or any other Metal small-M batching use case) is ever
revisited, this is the path to take — not a variant of the MMA approach investigated here.

**Built and measured: `docs/task-metal-batched-verify-kernel.md`.** Bit-identity held exactly as
predicted (verified against real weights, two model families). It did not, however, clear P10's
performance bar — a structural finding (only 4 of ~12 per-layer dispatches got batched) that
doc explains in full; read it before proposing another attempt at Metal small-M batching.

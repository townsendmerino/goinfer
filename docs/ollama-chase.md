# Chasing Ollama — the consolidated performance backlog

**Status:** live working doc. Everything here is either measured, bounded, or explicitly
marked as unmeasured. Nothing in it is a plan of record until a campaign is opened.

**Purpose:** one place holding every lever that could close the gap to llama.cpp/Ollama,
what each is worth, what it costs, whether it survives the bit-identity contract, and what
must be measured before it is funded. It also records what has already been tried, so
nothing gets re-proposed.

---

## 1. Where we actually stand

Measured on the RTX 2070 SUPER box, qwen2.5-coder-1.5b, q4 both sides, against **Ollama
v0.32.5** (current) — not the 0.5.7 pin, which was ~18 months stale and produced a
competitive claim that did not survive re-measurement.

> **Sampling was not recorded for this table (2026-08-04).** The 2026-08-09 re-measurement in the
> README records it on both sides and broadly reproduces this curve (within 1–4% at every depth),
> so the shape stands; treat the README table as the citable one.

Decode tok/s by KV depth (best-of-3, interleaved, first run dropped; re-measured 2026-08-04 **after**
the explicit-FMA contraction fix, which changes instruction counts everywhere):

| decode @ depth | goinfer | Ollama v0.32.5 | verdict |
|---|---|---|---|
| 128 | **226.6** | 197.5 | **goinfer 1.15×** |
| 512 | **207.3** | 191.7 | **goinfer 1.08×** |
| 2048 | 160.1 (was 97; A1 + split-KV) | 186.6 | Ollama 1.17× |   ← re-measured 2026-08-09: 157.6 vs 179.2 (Ollama 1.14×)
| 3900 | 123.5 | 180.7 | Ollama 1.46× |
| prefill | 0.66 ms/tok | 0.14 ms/tok | **4.7× behind** |

**Parity across context lengths is NOT real** — goinfer is ahead ≤ ~512 ctx and behind at 2048+, the
gap widening with depth. Ollama's flash attention holds ~flat (197 → 181 over 128→3900) while goinfer
degrades (227 → 124). The decode crossover is ~1000 tokens. A1 coalescing + split-KV lifted 2048 from
97 → 160 (the old "1.94× behind" → 1.17×), but did not reach parity, and 3900 is 1.46× behind. The
remaining long-context lever is a bit-identical flash-attention-style decode (the §A2 split-KV is the
start; the arithmetic says parity is reachable but not yet built). **Execution method for this search:**
an autonomous edit→bench→keep/revert loop over the gates is scoped in `docs/task-autoresearch-loop.md`
(queue-engineering E9) — its primary target is this lever's fast-mode lane, since the exact-path
bit-identical levers are largely exhausted (Campaign A ceiling).

**Peer-independent claims that stand regardless:** CGO_ENABLED=0 with driver-only linkage,
portable, bit-identical decode with a goldens-gated parity discipline, and 2048-token TTFT
improved 13.1 s → 2.1 s by our own prior work.

**The honest one-line position:** *ahead* of current Ollama on short/mid-context decode (≤~512,
1.08–1.15×), behind at long context (2048+, widening to 1.46× at 3900) and on prefill; the decode
crossover is ~1000 tokens.

---

## 2. Ground rules (why this doc looks the way it does)

Carried from the prefill campaign, where five attributions were made and four were wrong:

- **Profile the unit before designing the fix.** "Far off peak" is not a diagnosis. Two
  kernels in this repo were both far off peak with *opposite* bounds — the GEMV was
  latency-bound with efficient loads, the prefill attention was throughput-bound with 4.5×
  wasted traffic. Carrying one diagnosis to the other would have built the wrong fix.
- **Total is the verdict line.** A residual bucket silently absorbs displaced cost; an
  optimised sub-bucket can look like a 3× win while the total moves 4%.
- **A synthetic must reproduce the *pressure*, not just the *configuration*.** Three probes
  in this program misled by reproducing buffer counts and dispatch shapes while omitting
  eviction, blocking, or scale.
- **Bit-identity is designed in, never discovered.** Every kernel that shipped stated its
  identity constraint before the first line was written.
- **On Metal, reduction WIDTH is part of the bit-identity contract, and it's the same number you
  tune for speed.** Any cross-thread float *sum* (rmsnorm sum-of-squares, softmax denominator,
  qk-norm) reduces N/T strided partials in a T-wide tree; float add is non-associative, so the result
  is wired to T = the threadgroup width — the exact knob you'd sweep for a 5% win. CUDA is immune (warp
  reduce is a fixed 32); Metal is not. **No existing gate catches a width change**: paged≡non-paged
  compares the same kernel at the same width (self-consistent), and GPU-vs-CPU is tolerance. It only
  bites when a *new* same-op kernel is gated byte-exact at a different width — and only past nKeys>width,
  below a short fixture (this is exactly how the split/staged attention rewrites diverged: 256-wide denom
  vs the shipped 128-wide tree). So pin the widths as named constants (`tgReduce*`, model.go), make any
  alternate same-op kernel inherit the width, and give a byte-exact fixture for such an op context >
  width. Max reductions and `simd_sum` (32, hardware-fixed) are exempt — associative, order-exact.
- **A self-consistent gate cannot detect a change that moves both arms together.** Width is one instance;
  the class is larger — a fused-kernel rewrite, a different accumulation order, a moved scale-application
  point all slip through `paged≡non-paged` (one kernel vs itself under different residency → both arms
  move identically) and through `GPU-vs-CPU` (tolerance by construction). Only an **absolute stored
  reference** catches them. CUDA has goldens; Metal-vs-CPU can only be tolerance (f16 scale gap), so this
  was a structural blind spot — **now closed by a self-referential snapshot golden**
  (`TestMetalSnapshotGolden`: fixed inputs, tiny models, depths past the widths, byte-compared; machine-
  pinned, `GOINFER_UPDATE_GOLDENS=1` to refresh). See §A2-Metal.
- **A float multiply-accumulate under a bit-identity contract MUST use an explicit intrinsic**
  (`__fmaf_rn` on CUDA, `fma()` on Metal). A bare `a*b + c` leaves the fma-vs-mul+add contraction to
  the compiler, and two kernels with identical source can compile to different numerics — ~1 ULP
  apart, DATA-DEPENDENT, invisible on uniform random fixtures but an 84% token-stream divergence on
  real weights (the batched-prefill-vs-decode split, `docs/completed/task-batched-prefill-bitidentity.md`). The
  gate is the PTX/AIR **instruction histogram**, not a numerical test — a data-driven test only catches
  it when the fixture has the dynamic range to expose it, which random data does not. Enforced by
  `cuda.TestKernelFMALint` (fails the build on any bare float MAC in a contracted kernel; extends to
  future kernels automatically). Same family as the Metal reduction-width rule above: **a numerical
  property silently coupled to a compiler or tuning decision.** aikit's `gemv_quant.cu` (the decode
  GEMV) carries the same rule in its own repo — the pair must agree, so both must follow it.
- **Thermal control on the Mac** (±700 ms drift): interleaved repeats, session-start run
  dropped. Single-run rankings are unreliable.
- **Peer version is part of the measurement.** The whole §B2 correction happened because a
  pinned peer outlived its evidence.
- **An instruction-mix histogram bounds throughput from above; it cannot establish you are
  at that bound.** Only stall and eligibility data can.
- **A demanded read rate exceeding DRAM proves reads are cache-served, not that the cache is
  saturated.**

---

## 3. The two deficits, decomposed

### 3a. Long-context decode — 1.94× → **1.41× behind** (A1 coalescing landed)

Decode costs ~**0.0028 ms per KV position** on a ~4.1 ms base (measured: 221 → 179 → 100 →
66 tok/s at 128 / 512 / 2048 / 3900 context).

For this model the KV read at a given position is roughly **28 KB across all layers**, which
is about **0.00006 ms** at the card's bandwidth. That is **~40× off the memory bound** on the
term that scales with context.

It was previously read as "dead-linear O(context), correct behaviour, not a cliff." Correct
in isolation, refuted comparatively: Ollama holds ~188 tok/s at 2048 where we fell to ~97.

**Profiled, fixed, and RE-profiled (2026-08-04, Campaign A0/A1).** ncu named the decode attention's
K read uncoalesced (21.96% bytes/sector); wiring decode to the float4-coalesced `attn_batched`
at M=1 (bit-identical) took 2048-ctx decode **99.5 → 133.5 tok/s (1.34×)**. The A1-reprofile (§A2)
then showed the bound **moved off memory-coalescing onto occupancy**: 12 blocks on 40 SMs, Waves/SM
0.04, 11.9% occupancy — the fix is **split-KV parallelism**, not a KV relayout, and it converges with
B1 (bit-identical tiled attention). Campaign A stays open on that build.

### 3b. Prefill — 4.7× behind

At 2048 tokens, after the prefill campaign: **61% GEMV, 39% attention**, ~1% glue.

- GEMV runs at **54% of dp4a peak**; dp4a is roughly **1/3 of IMMA** throughput on Turing.
- Attention is still **99.51% L1TEX-saturated** after the `float4` fix — the residual is
  O(M²) redundant K/V re-reads, which coalescing cannot touch.

So the prefill gap is part format-imposed (tensor cores unreachable — see §7) and part
unfinished work that bit-identity does not block at all.

---

## 4. Campaign A — long-context decode attention *(recommended next)*

**Why this one first.** It is the only deficit where the arithmetic says **parity is
reachable**. A 10× on the per-position term takes decode at 2048 from ~9.8 ms/token to
~4.7 ms ≈ **214 tok/s**, against Ollama's ~188. And it changes a *qualitative* claim — "at
parity at short context" becomes "at parity across context lengths," on the axis where users
spend their time — rather than improving a magnitude.

Prefill cannot reach parity (§3b, §7), so it cannot buy the same thing at any price.

### A0. Profile the decode attention kernel — **DONE (2026-08-04)**

`ncu` on the M=1 glue `attention` kernel at 2048 context (`TestDecodeAttn2048Probe`): Duration
**232.7 µs/launch** (~63% of the ~10.3 ms decode budget), **21.96% bytes/sector** on the K read
(uncoalesced, stride-`kvDim`), L1TEX 71%, No-Eligible 94.5%. The profile named it: the *same*
uncoalesced K-read signature the prefill attention had before its float4 fix — but this is a
distinct kernel and got its own profile (the caution held; the diagnosis was confirmed at the
hardware, not carried over).

### A1. Fix — **PARTIAL, landed (`1a1914b`, 2026-08-04)**

The float4-coalesced `attn_batched` at **M=1** is bit-identical to the audited glue `attention`
(`TestAttnBatched_bitIdentical`), so the decode attention was wired to it (guarded by
`prefillReady`; glue.ptx untouched, kept as fallback; `resident.go`). Same grid/block/shared/ctx
layout, `startPos=pos` ⇒ `nKeys=pos+1` ⇒ **decode stays byte-identical.**

Same-box A/B (`TestDecodeDepthThroughput`, git-stash resident.go): 2048-ctx decode **99.5 → 133.5
tok/s (1.34×)**, shallow unchanged. Narrows the gap to current Ollama from ~1.9× to **~1.41×**.

**Not yet parity.** The arithmetic above says a 10× on the per-position term reaches ~214 tok/s;
coalescing bought 1.34×, so the redundant-re-read / latency residual remains — that is what **A2
(KV layout)** and a query-tiled decode path would attack next. Campaign A is *open*, not closed.

Gates that held: decode byte-identical; parity manifest green; `TestRealE2EDecodeThroughput` / `TestBackendResidentWired`.
(`TestE2EDecode` was cited here as a gate, but it asserted nothing — synthetic weights, throughput
only. Renamed `TestE2EDecodeThroughput_synthetic`; `TestRealE2EDecodeThroughput` / `TestBackendResidentWired` now carries the token-identity
assertion vs the CPU reference. See audit G-01.)

### A1-Metal. Same fix on Metal — landed (`994539c`, 2026-08-04)

The Metal decode `attention` (and `attention_f32`, `attention_prefill`) carried the **same**
one-thread-per-key uncoalesced K-read. Profiled (A0-analogue microbench, `attention` @2048 ctx):
**~13 GB/s effective, ~3–6% of M1 peak** — the same signature, **more severe** than CUDA's ~22%.

**Path 1 (reuse the batched kernel, as CUDA did) was dead here:** Metal's `attention_prefill` is the
*same* uncoalesced code (only f16 I/O differs), **not** a coalesced kernel like CUDA's `attn_batched`
— nothing to borrow. So the fix is a **half4/float4 vectorized K-read** (8-byte loads, **same
sequential f32 accumulation ⇒ bit-identical**, verified 0 mismatches at nKeys 1/128/999/2048; guarded
on `hd%4==0` with a scalar tail). The AV read was already coalesced (adjacent lanes read adjacent `d`),
so the QK read is the whole fix.

Isolated-kernel A/B @2048: **1049 → 588 µs = 1.79×** (beats CUDA's 1.34×, consistent with the worse
starting coalescing). **End-to-end now MEASURED** (2026-08-04, real-model depth A/B, qwen2.5-coder-1.5b
int8, resident, best-of-40 warm; scalar `994539c^` vs coalesced HEAD):

| KV depth | scalar tok/s | coalesced tok/s | end-to-end |
|---:|---:|---:|:---:|
| 128 | 62.7 | 63.8 | 1.02× |
| 1024 | 37.5 | 39.8 | 1.06× |
| 2048 | 20.7 | 28.4 | **1.37×** |
| 4000 | 13.2 | 18.5 | **1.40×** |

The **1.37–1.40× @depth** lands squarely on the Amdahl estimate (kernel 1.79× diluted by attention's
~half share of the per-token cost). Shallow context is ~flat (1.02×) — attention is negligible against
the ~70 tok/s dispatch-bound Metal decode floor there; the win is strictly a **long-context** win, which
is the regime that matters. Applies to **every dense/GQA family** that decodes through
this kernel. Gates green: `TestAttention`, `TestPrefill`, dense/gemma3 resident parity, dense-scaled
geometry, paging bit-exact, shipped-kernel-shapes. **First Metal *decode* speed win** — prior Metal
work (26B paging) was at a hardware floor. Campaign A stays *open* on Metal too (same A2 residual).

### A1-reprofile — **DONE (2026-08-04); the bound moved to OCCUPANCY, not layout**

Re-`ncu`'d the *coalesced* decode kernel (`attn_batched` at M=1, the launch A1 wired in) at 2048
ctx, per the "profile the unit before designing the fix" rule — A1 changed the kernel, so the A0
profile is stale. First decode launch (Grid 12 = nH, Block 128, nKeys≈2049):

| metric | glue (A0) | coalesced (A1) |
|---|---|---|
| Duration | 232.7 µs | **134.2 µs** |
| L1/TEX throughput | 71% | **38.6%** |
| DRAM / L2 / Compute SoL | — | 9.5% / 12.3% / **5.4%** |
| Grid / **Waves per SM** | — | 12 blocks / **0.04** |
| Achieved occupancy | — | **11.9%** (theoretical 87.5%) |
| No-Eligible-Warp | 94.5% | **93.0%** |

**A1 killed the coalescing bound (L1TEX 71→38%); the new bound is OCCUPANCY STARVATION.** Decode
attention launches **12 blocks** (one per query head) on a 40-SM card — Waves/SM **0.04**, ~28 SMs
idle, and the 12 live blocks can't hide their own memory latency (scoreboard stalls = 78% of the
14-cycle stall average). It is **not** register/shared-limited (block limits 7–16) — purely too few
blocks. Nothing is throughput-saturated (DRAM 9.5%, Compute 5.4%).

### A2. Split-KV decode attention — **LANDED, opt-in, bit-identical (2026-08-04)**

The A1-reprofile refuted the KV-layout hypothesis (reads already coalesced) and named **occupancy**:
12 blocks on 40 SMs. The fix is parallelism over the *independent* axes, **not** a relayout and not
FA's non-bit-exact online rescale. Full design + the associativity argument (why a contiguous
key-split fails but a scores-over-keys / V-sum-over-dims split is byte-identical) in
`docs/task-decode-splitkv-attention.md`. D4 confirmed Ollama does exactly this via flash attention.

Built as a 3-kernel split (`decode_splitkv.cu`, own file): `splitkv_scores` (tile over keys),
`splitkv_softmax` (exact 128-wide partition+tree → byte-identical max/denominator), `splitkv_vsum`
(each thread the whole per-dim fold, tiled over dims). **Bit-identical** to attn_batched(M=1)
(`TestSplitKV_bitIdentical`: stream + 151936 logits byte-identical at depth 2048). **2048-ctx decode
133 → 160 tok/s (1.20×)**; long-ctx total now 99.5 → 160 = **1.61×** over glue, gap to Ollama
**1.41× → 1.17×**. **Default-on**, gated at runtime on `nKeys ≥ 256` (measured crossover: break-even
256, win from 384+, −3% only at 128 → shallow decode keeps `attn_batched`, byte-identical either
way); `GOINFER_SPLITKV_ATTN=0` force-disables. Bit-identity gated on GQA/hd=128/no-window (qwen2.5)
AND hd=256/windowed (gemma3, winStart>0).

> #### AMENDMENT (2026-08-09, P6a) — the `nKeys ≥ 256` gate above is REFUTED and replaced
>
> A 48-cell e2e sweep (4 geometries × 6 depths × ON/OFF, decode-only through `serve`) found the
> shipped constant fires **3–12× too early**, costing up to **18–25%**. **Two independent defects**,
> and it matters that they are separate:
>
> 1. **The original characterization overstated itself ~3–4× on its own geometry.** The 1.5B is the
>    model `TestSplitKVCrossover` measured, and it *loses* at 256 (0.941) and at 512 (0.939). Its real
>    crossover is in (512, 1024]. "Break-even at 256, clear win from 384+" was never true, even here.
> 2. **A one-geometry constant was generalized to every model.** qwen0.5b does not cross until ~2560;
>    phi3-mini (MHA, nH=32) **never crosses at any depth** — its ratio *declines monotonically*
>    (0.993 → 0.969 → 0.919 → 0.815 → 0.754 at 3900).
>
> **Why the microbenchmark disagreed with serving** (one line, not chased further): `TestSplitKVCrossover`
> times a tight in-process `ForwardArgmax` loop and takes **best-of-3 minimum**. Both choices flatter
> split-KV — the loop hides the per-token CPU dispatch a real request exposes, and best-of-min favours
> the higher-variance arm, which is consistently ON (spread 3.6–6.4 tok/s vs OFF's 0.1–0.6).
>
> **The replacement is a lookup, not a formula.** Split-KV buys occupancy and pays for it in DRAM:
> `attn_batched` keeps the score row in *shared* memory and launches nH blocks; split-KV materializes
> an nH×nWin f32 array in *global* memory and touches it three times. So
> `net(nWin) ≈ (A−B)·nWin − 2·L·T_launch`, and when A < B split-KV never wins and the deficit *widens*
> with depth — exactly phi3. That is why no formula works: a formula gate has the form
> "ON iff `nWin ≥ f(geometry)`", which always predicts ON wins eventually. **phi3 falsifies the form,
> not just the constants.** Two candidate laws (`nLayers/(nH·hd)`, `nLayers/(nKV·hd)`) also
> underpredicted the 0.5B crossover by ~2×. Shipped as a measured per-geometry table with a
> conservative default, pinned by `TestSplitKVGate_measuredGeometries`.
>
> Also fixed structurally: the gate now tests the **effective attended span `nWin`**, per layer, not
> the raw position. A sliding-window layer never attends more than `window` keys, so gating on
> position put gemma3's windowed layers on the split path at a 512-key span — its loss regime — at
> every depth past the window. Full table and method: `docs/benchmarks.md` §B6.

**Campaign A closed at the bit-identity ceiling (~160, 1.17× behind — not full parity).** ncu showed
`splitkv_vsum` the residual bottleneck (50 µs, 3.8% occupancy = the nH·hd parallelism ceiling). The
one bit-identical lever left — a **V-sum ILP unroll** — was **tried and REFUTED**: nvcc already
pipelines the scalar loop's independent loads, so a clean unroll landed 158 vs 160 (no gain) and a
naive-indexed one regressed to 111. Pushing past ~160 needs a non-bit-identical reduction (the §7
fork), which Campaign A does not take. The tiling primitive still stands to be shared with **B1**
(prefill query-tiling). See `docs/task-decode-splitkv-attention.md`.

### A2-old. KV cache layout (kept for the record) — *not indicated*

Layout changes are bit-identical by construction (they move where operands live, not the order
they accumulate) — the same reason `int2` coalescing and A-staging were safe. But the reprofile
says the decode kernel is occupancy-bound, not read-throughput-bound, so a relayout is not the
lever here. Revisit only if a future profile shows uncoalesced KV traffic.

### A2-Metal — split-KV BUILT + MEASURED: does NOT transfer to Metal (2026-08-04)

CUDA's split-KV won 1.20× because its decode attention was occupancy-starved (ncu: 12 blocks / 40 SMs,
11.9% occ). The Metal depth curve *looked* the same shape, so the split-KV was ported and measured. **It
is bit-identical but a consistent regression — the occupancy diagnosis does not hold on Metal.**

Built the exact CUDA 3-kernel structure (`splitkv_scores` / `splitkv_softmax` / `splitkv_vsum`), the
order-dependent softmax fold kept whole, the two O(nKeys·hd) passes split along their independent axes.
**Byte-identical: 0 / 151936 logit mismatches at depths 1 / 128 / 512 / 1024 / 2048** (the associativity
argument holds on Metal too). Best-of-40 depth A/B (interleaved, qwen2.5-coder-1.5b, resident):

| depth | shipped | split-KV | verdict |
|---:|---:|---:|:---:|
| 128 | 64.2 | 63.1 | 0.98× |
| 1024 | 40.5 | 38.1 | 0.94× |
| 2048 | 28.7 | 28.1 | 0.98× |
| 4000 | 18.5 | 17.9 | 0.97× |

**Why it fails where CUDA won — and it's not fixable by tuning:** the scores pass got a **32× threadgroup
increase (12 → 384)** and the vsum pass a **4× increase (12 → 48)**; *neither* moved the token time. So
Metal decode attention is **not occupancy-bound** — that diagnosis was card-specific (40 SMs starved by
12 blocks; the M1's ~16–20 cores are not). Meanwhile split-KV *structurally* needs the cross-threadgroup
sync that only a kernel-launch barrier gives, so it costs **+2 dispatches/layer × 28 = +56 dispatches/
token**, and Metal decode is **dispatch-count-bound** (the ~70 tok/s shallow floor). That tax is the
whole regression. The Metal decode lever points the **opposite** way — *fewer* dispatches (megakernel),
not more parallelism. **Reverted** (code removed; the throwaway A/B in the session transcript reproduces
it). CUDA's win is genuine and stays; it simply does not generalise to Metal.

### A2-Metal — profiled LOCALLY, then 4 dedup attempts, all lost (2026-08-04)

Rather than carry another CUDA diagnosis over (the split-KV mis-transfer above), profiled the Metal
decode attention **on this box** with cgo-free GPU timestamps + a collapse probe:

- **DRAM-latency-bound, not occupancy-bound.** Attention's depth-term runs at a flat **~17 GB/s = 8.6%
  of the M1's ~200 GB/s**, and its ALU is **~0.7% of peak** — so it's memory-latency-bound, ~98% GPU-side.
- **Collapse probe** (pin every K/V read to key 0: same loop/ALU/threadgroup traffic, zero distinct
  DRAM): all-28-layer attention **21.5 → 5.3 ms**, i.e. **75% of attention is the distinct per-key K/V
  DRAM reads**. The prize is real (~16 ms/tok @2048) and it's the **6× GQA redundant read** (12 query
  heads re-reading 2 KV heads).

**K-dedup is capturable; V-dedup is not; and neither structure captures a net win.** A grouped kernel
where one thread reads a key's K once and computes all G heads' dots got **scores 10 → 4.66 ms** — a real
win. But the V-fold is the other ~half, and sharing the V read forces either (a) a `(kvHead,dim)` mapping
→ 256 threads → **58 ms crater**, or (b) co-locating the group in one threadgroup. Every attempt:

| attempt | structure | bit-exact | @2048 |
|---|---|:--:|:--:|
| split-KV (3 dispatch) | key-split, per-head reads | ✓ | 0.95× |
| grouped 3-dispatch | K-dedup scores + parallel vsum | ✓ | 0.45×→0.92× |
| grouped 2-dispatch | K-dedup scores + fused finish | ✓ | 0.92× |
| **staged (1 dispatch)** | threadgroup-staged K/V, nKV tg | ✓ | **0.23×** |

The single-dispatch **threadgroup-staged** kernel is the mechanism the others lacked (one global K/V read
into threadgroup memory, G uses) — and it's the cleanest expression of the dedup. It cratered hardest
(**0.23×**) for the predicted reason: sharing the read across the group requires **one threadgroup per KV
head = 2 threadgroups on ~14 cores**, so the compute that shipped spreads over 12 cores runs on 2. Dedup
and occupancy are in direct opposition on Metal; you cannot have both.

(Bit-identity note: the staged kernel was only byte-identical once its softmax reduction ran at **128
threads to match the shipped 128-wide tree** — a 256-wide reduction sums the denominator in a different
order and diverged at nKeys>256. Reduction *width* is part of the bit-identity contract.)

**Conclusion — four independent confirmations that Metal decode attention is structurally
dispatch-/occupancy-bound.** The DRAM-dedup prize is real but uncapturable: any dedup needs group
co-location (→ few threadgroups → occupancy death) or extra dispatches (→ tax death). This is the same
wall as the whole decode path (`docs/completed/task-metal-cgofree-spike.md`: megakernel closed, dispatch-count the
ceiling). **A1-Metal (half4 coalescing, 1.37–1.40×) remains the one capturable decode-attention win on
this box.** Stop proposing dedup layouts; the lever is elsewhere (KV-quant §A3, or accept the floor).

**The floor is confirmed at the COMMIT level too, not only at dispatch count (census, 2026-08-12).**
An external Metal write-up (dmikey's A1111 fork) frames the dominant hidden cost as command-buffer
*submits* and CPU⇄GPU *syncs* — "the fastest kernel still loses if you submit the command buffer after
every call" — a lever separate from the dispatch/encode count our floor is framed in. Measured at the
Metal command-buffer boundary (qwen2.5-coder-1.5b int4, resident; counters on aikit branch
`metal-census-probe`, goinfer probe branch `metal-commit-census`): **commits/token = 1.000 and
waits/token = 1.000** at both shallow (pos 8) and depth 2048, on *both* the production pipelined
sampling path (`ForwardEmbPipe`) and the greedy on-device-argmax path (`ForwardArgmax`) — encoders
≈ 1/token (the +0.5% at shallow is the encode-ahead executor's one pre-encoded next-token spanning the
measurement window). This is the structural minimum: the next token needs its logits back, so one
commit + one blocking wait is unavoidable, and goinfer already encodes the whole trunk + LM head into a
single command buffer per token (`resident.encodeLogitsCB`), committing once. So submit/sync frequency
is **not** a hidden lever here — the coalescing win the external finding describes is already fully
realized, and the accepted-floor conclusion stands with one more axis explicitly ruled out.

**Where the token's dispatches actually go — the dispatch census (2026-08-12, same probe).** Since the
floor is dispatch-*count*-bound, the same probe counted dispatches/token and attributed them per
pipeline (qwen2.5-coder-1.5b int4, depth 2048, 28 layers): **338 dispatches/token** — 12/layer + the
final norm + the LM head. Per layer: `rms`×2, `rope`×2, and one each of `stageA_gemv`, `stageA_bias`,
`stageA_resid`, `kv_write`, `attn`, `gemv_resid`, `quant_vec`, `swiglu`. This turns the two queued
Metal dispatch-removals (P4/P5) from audit estimates into **measured dispatch shares**:

| item | pipeline | measured/token | removes | share of the 338 |
|---|---|---|---|---|
| **P4** RoPE grid-merge (2→1/layer) | `rope` | 56 (exactly 2/layer) | 28/token | **8.3%** (audit est. "a few %") |
| **P5** fuse `quant_vec` into o-proj | `quant_vec` | 28 (exactly 1/layer) | 28/token | **8.3%** (audit est. "~5–6%") |
| **combined** | | | 56/token | **16.6%** |

The census also *clarifies* P5: there is exactly **one** `quant_vec` dispatch per layer (the o-proj
input quant — the other GEMVs already fuse their quant), so P5 would remove the whole pipeline, not a
fraction.

**But the dispatch-count fraction does NOT convert to tok/s — the A/B says net-zero (2026-08-12).** The
count was the ceiling, and the ceiling turns out to be ~0. P4's grid-merge was already implemented
bit-identically on branch `metal-rope-merge` (snapshot-golden byte-exact) and re-A/B'd on the current
338-dispatch binary (`TestZZ_metalDepthBench`, qwen2.5-coder-1.5b W4A8, M1 Pro):

| depth | before (2 rope disp) | after (1, merged) |
|---|---|---|
| 128 | 59.7 | 61.0 |
| 512 | 49.1 | 46.5 |
| 2048 | 28.4 | 26.9 |
| 4000 | 18.4 | 18.4 |

Mixed-sign, within thermal noise — **8.3% fewer dispatches, 0% tok/s.** The reason is the floor's own
logic: "dispatch-count-bound" is set by the *sum* of ~338 dispatches gating the token, and removing one
small per-layer dispatch is lost in it (the same result the branch found at its then-476-dispatch
binary — the conclusion survived the binary shrinking). **P5 is the same magnitude and mechanism** (one
small per-layer dispatch, 28/token, 8.3%), so it is **predicted net-zero by direct analogy** and not
worth a standalone build — if built, A/B it, don't assume the estimate. This is the §11 note made
concrete: *microbenchmarks (dispatch count) nominate; full generations (tok/s) elect* — and they voted
no. The levers that actually move Metal decode remain the **megakernel** (collapse *most* dispatches at
once, not one) and **int4 unpack** (bandwidth). Recorded so P4/P5 are not re-proposed as tok/s wins.

**The Metal gate suite had a structural blind spot — now closed by a snapshot golden.** The
reduction-width finding (§2) is one instance of a general blind class: **a self-consistent gate cannot
detect a change that moves both arms together.** `paged ≡ non-paged` compares one kernel against itself
under different residency — anything that changes the *kernel* changes both arms identically and passes
(width, a fused-kernel rewrite, a different accumulation order, a moved scale-application point — all slip
through). `Metal-vs-CPU` is tolerance-based (unavoidable, given the f16 scale gap), so small movements
pass by construction. The only gate type that catches this is an **absolute stored reference** — one that
does not move when the code does. CUDA has goldens; Metal had nothing equivalent.

**Landed (`TestMetalSnapshotGolden`, `metal/snapshot_golden_test.go` + `testdata/metal_snapshot_golden.json`).**
Decodes a fixed token sequence through the Metal resident path on two tiny committed models to depths
**past both reduction widths (128, 256)** and byte-compares the logits (sha256) to a committed golden:
- **mixtral-tiny** (int8int8) — full-causal, so the attention softmax denominator reduces over >256 keys
  (the width coupling at multi-iteration depth); also `rmsnorm_quant`.
- **gemma4-dense-scaled** (int4) — `attention_f32` + `rmsnorm_f32` + `qk_norm` (the sandwich/f32 path).

Union = every pinned-width reduction kernel. Runs on **every `go test`** (tiny models, no heavy dep,
~2 s), and it is **self-referential** (detects that something moved, not which side is correct). Proven
both ways: byte-identical across two consecutive runs (it also validates run-to-run determinism, which
nothing else did), and it goes **red on a width change** (flipping `tgReduceNorm` 256→128 drifted the
gemma4 checkpoints — the H=1024 sum reorders; mixtral's H=64 norm is width-invariant, so only the truly
coupled arm fired). **Machine-pinned, with a guard against the refresh reflex.** Metal float results are
deterministic run-to-run and across code versions on a given GPU, but not across chip families or OS
toolchain versions — so the golden records the **GPU name + macOS version it was baked on** (`Apple M1
Pro / 26.5.2`), and a drift branches the failure message: *env differs* → "HARDWARE/OS DIFFERS,
EXPECTED, do NOT refresh unless intentionally re-baselining here"; *env same* → "SAME hardware, the bits
moved, INVESTIGATE before refreshing." Both branches proven (env-tamper + sha-tamper). This is what stops
the trained `GOINFER_UPDATE_GOLDENS` reflex from silently destroying the reference on a new Mac / OS bump.
Legitimate refresh (verified change, or re-baseline on this box): `GOINFER_UPDATE_GOLDENS=1 go test -run
TestMetalSnapshotGolden ./metal/`.

**Does aikit need the same enforcement? All three gaps now CLOSED (aikit `gpu` v0.25.2).** aikit already
makes the strict choice where parity matters (its ViT uses `CompileLibraryPrecise`) and correctly offers
both compile paths, so no default change was warranted upstream. The gap was that `CompileLibraryPrecise`
set `setFastMathEnabled:NO` but **never read it back** — while the *same function* asserts `languageVersion`
against exactly this landmine class — and that setter is on Apple's deprecation path (→ `MTLMathMode`), so
a future macOS could silently no-op it and hand back a fast-math library with aikit's own ViT parity gate
none the wiser. **Fixed (v0.25.1, hardened v0.25.2):** `CompileLibraryPrecise` prefers
`setMathMode:MTLMathModeSafe` (falling back to `setFastMathEnabled:NO` on older OSes) and **reads the state
back, erroring if it didn't take**. v0.25.2 closed the subtler hole — the fallback previously *sent* the
deprecated selectors without checking they respond, so a future OS dropping both APIs would send
unrecognized selectors and, if the getter returned 0, report a **silent pass** (an unverified "precise"
library — the exact failure the guard exists to prevent). Now a `respondsToSelector:`-gated switch
requires both setter and getter of the chosen path before touching either, and errors naming both
selectors it tried if neither responds. Also: a `RELEASING.md` ritual (the `gpu` module is separately
versioned and root CI doesn't exercise it, so a removed guard turns nothing red — before a `gpu/vX.Y.Z`
tag, run `gpu/` tests on a Mac+GPU and record machine+OS in the annotated tag), and aikit's ViT reduction
width is now pinned as a documented bit-identity contract (the `ViTBlock`/`LNBLOCK`=256 constants, five
f32-sum reductions tagged in-kernel). goinfer bumped its `aikit/gpu` require to **v0.25.2**. The remaining
lower-priority item, still aikit's call: its Metal ViT gate is tolerance-only — goinfer's snapshot-golden
technique transfers directly, but precise math already
lowers the drift risk there, so treat those as an optional separate hardening pass.

### A2-Metal — the bit-identity contract, the exposure, and how divergence is prevented

**What bit-identity means on Metal (stated explicitly — it never was).** Three axes move the bits:
reduction width/order, fast-math compiler discretion, and the GPU chip + OS Metal toolchain that
compiles the MSL. That splits the contract cleanly:
- **Within-machine (same binary, same OS): DELIVERABLE and gated.** Run-to-run stable (the snapshot
  golden proves it), `paged ≡ non-paged` byte-exact, decode byte-exact across code that doesn't touch
  the math.
- **Across-machine (any Mac, same bits): NOT deliverable, by construction.** MSL is compiled by the
  user's OS Metal toolchain, which varies by macOS version, onto a chip whose ALUs vary by family — so
  a different Mac *or a macOS update* can legally produce different bits. The snapshot golden is
  therefore machine+OS-pinned; that is a property of Metal, not a defect of the gate.

**The exposure audit (2026-08-04, report only).** goinfer's decode + prefill kernels compile via
`CompileLibrary` = **default `MTLCompileOptions`, fast-math ON** — which licenses contraction (`a*b+c`
→ fma), **reassociation**, reciprocal divides, and **low-precision transcendentals**, all at the
compiler's discretion (and that discretion can shift across OS toolchain versions — the mechanism
behind the across-OS fragility above). A `CompileLibraryPrecise` (fast-math OFF) already exists and the
ViT path uses it, so the strict lever is one call-swap away. Where it bites, worst first:
- **softmax `exp`** (`kernels.go` attention/attention_f32; `prefill.go` attention_prefill) — a
  low-precision `exp` is many ULP off, *and* it sits inside the denominator sum whose order we pin by
  hand; the approximation perturbs the very reduction §2 protects. Highest-value strict target.
- **`a/sum`** attention normalize → `a*rcp(sum)`, off a ULP (same class as the ViT quant-scale bug).
- **`rsqrt`** in every rmsnorm; **`exp`/`tanh`** in the SiLU/GELU activations.
- **pervasive contraction** — every float MAC (`a+=q*k`, `a+=sc*v`, `ss+=x*x`) contracts to fma.
- **reassociation of the per-thread accumulation loops** that feed the pinned reductions: the
  cross-barrier *tree* is safe (barriers block reassociation), but the per-thread partial sums are
  compiler-reorderable — so the "hand-preserved order" is only preserved *modulo whatever fast-math
  already did*. The real invariant today is "deterministic for a fixed width **and math-mode**."

**How divergence is prevented — for existing kernels and future ones.** Layered, because no single
mechanism reaches "never," and one axis can't be reached at all:
1. **Remove the invisible axis — compile precise. MEASURED, and NOT adopted as default (2026-08-04).**
   Fast-math OFF (`CompileLibraryPrecise`) makes the *source* fully determine the bits — robust to an OS
   toolchain update, which fast-math is not. But the interleaved A/B (idle, cold-run dropped) put the
   cost at **6.7% shallow / 3.7% @2048 decode tok/s, and it did NOT improve CPU parity** (21/24 vs
   22/24 — the gap is int8→int4 requant, not fast-math). Above the "cheap → take regardless" bar, and
   the argument that settles it: the snapshot golden already *detects* an OS-toolchain drift (and the
   env-branch guides the refresh), so precise buys *prevention* over *detection* — not worth a perpetual
   4–7% on the primary metric. **Decision: fast-math stays default; robustness is via golden-detection.**
   Kept as a documented opt-in (`GOINFER_PRECISE_MATH=1`, wired at the `BuildResident` compile call) for
   anyone who wants OS-robust bits at that cost. **Now ENFORCED (aikit `gpu` v0.25.2):** the earlier
   silent-no-op gap is closed — `CompileLibraryPrecise` prefers `setMathMode:MTLMathModeSafe` (falling
   back to `setFastMathEnabled:NO` on older OSes) and **reads the state back, erroring loudly if it didn't
   take, or if neither API responds** (v0.25.2 hardened the fallback so an OS that drops both selectors
   errors instead of silently passing). goinfer bumped its `aikit/gpu` require to v0.25.2, so the opt-in
   is a real guarantee, not a hope. If ever adopted as default, re-baseline the snapshot
   golden.
2. **Source determines order** *(enforcer: convention + the behavioral golden below).* Cross-thread
   float sums use the pinned `tgReduce*` widths (§2) and an explicit barrier-separated tree (never a
   compiler-reassociable free reduction). A cheap grep-lint can forbid a bare width literal at a
   reduction-kernel dispatch (must reference `tgReduce*`) — but that only polices the *known* shape.
3. **Catch movement** *(enforcer: the snapshot golden — behavioral, the primary gate).* Trips on any
   bit move in the paths it exercises, regardless of cause (width, order, a new fused kernel, a
   compiler change). Stronger than any lint because it checks the *result*, not the source text. Its
   coverage is the decode trunk of two architectures (dense-GQA + gemma4-sandwich); it does **not** yet
   cover other families' unique kernels (MLA, Mamba, cohere norm-parallel, gpt-oss) or the prefill
   path. Coverage-completeness is the gap.
4. **Force future kernels into the net** *(enforcer: a coverage lint + the onboarding checklist).* This
   is where linting earns its place — not policing math, but asserting **every compiled pipeline is
   exercised by at least one golden**; it fails when a new kernel ships un-snapshotted. New family /
   new kernel ⇒ add a tiny model + a golden entry past its reduction width, and (if it has a
   cross-thread float sum) a pinned width. A line in the family-onboarding checklist next to the
   parity_manifest step.

The honest ceiling: (1)+(2) make the bits a deterministic function of the source; (3)+(4) detect any
drift within the covered set; **across-chip identity remains out of reach** and the golden stays
machine-pinned. "Never diverge" is therefore *within-machine, robust-across-OS, drift-detected* — not
universal.

*Technique transfers across backends.* CUDA has real goldens so it needs the snapshot less; **WebGPU has
nothing — no absolute reference, no snapshot — and is the least-exercised backend**, so it carries the
same blind spot Metal just closed. A wgpu snapshot golden is the same pattern when that track reopens.

### A2-Metal — batched prefill has the CUDA divergence bug too (measured 2026-08-04)

CUDA's batched `PrefillLast` was 84% stream-divergent from sequential decode (§9). **Metal's has the same
defect class:** `TestMetalPrefillDivergenceRate` (50 seeded prompts, greedy, qwen2.5-coder-1.5b int8,
batched-prefill-then-decode vs sequential-prefill-then-decode) — **27/50 = 54% diverged, mean
first-divergence position 2.9** (i.e. it diverges almost immediately, as soon as the continuation attends
to the differently-written prompt KV).

Two corrections to the premise, both important:
- **Protected by a Metal-backend decline (not by the shared flag).** The shared `GOINFER_BATCHED_PREFILL`
  gate was flipped back to **default-ON** once the CUDA path was made fma-bit-identical — and it has no
  backend guard, so it *would* pull Metal's divergent path in by default. So `metal/backend.go`'s
  `PrefillLast` now **declines by default** (→ caller falls back to sequential = decode kernels =
  bit-identical); opt in only with `GOINFER_METAL_BATCHED_PREFILL=1`. This is the "default it off"
  response to a non-zero divergence, scoped to the backend that has it.
- **Metal's root cause is deeper than CUDA's, so the fix is harder.** CUDA's gap was fma-contraction
  (int-vs-int, ~1 ULP). Metal's `PrefillLast` runs **f16 activations** (for the MMA) while decode runs
  **int8 activations** — a fundamentally larger numerical gap — *plus* fast-math (contraction AND
  reassociation AND fast transcendentals). A bit-identical Metal batched prefill therefore needs the
  activation precision unified (an int8-activation batched path), not just contraction pinned. Real work;
  deferred (prefill is behind decode in §5 and improves an already-usable 2.1 s TTFT).

**The existing Metal prefill gates were blind to this by construction** — `TestPrefillParity` compares the
*last prompt logit's cosine* ("high-but-not-exact cosine" by design) and `TestPrefillGemmW4` is
cos≥0.99999/maxAbs tolerance. Neither decodes a continuation or checks the token stream, so a 54%
stream divergence passes green. Same lesson as the CUDA `TestPrefillLast_e2e` miss: a tolerance/last-token
gate cannot see a greedy-stream flip. If the Metal opt-in is ever hardened, its gate must be a
divergence-rate test (decode a continuation, compare streams), not a cosine.

### A3. KV cache quantization — **not bit-identical**

Ollama supports q8/q4 KV caches. Halving or quartering KV bytes directly attacks the term
that scales with context. **Changes decode numerics**, so it would need its own gate and a
parity refresh, and it should be an opt-in mode rather than a default. Listed for
completeness; ranked below A1/A2 because those are free of that cost.

---

## 5. Campaign B — prefill

Deferred behind A. It improves a number already past its usability threshold (2048-token
TTFT is 2.1 s, down from 13.1 s) and cannot reach parity.

### B1. Prefill attention query-tiling — **DESIGN-REVISED (2026-08-04); BANKED, not funded**

Design-first (before writing the kernel) found the clean ~1.3× is **not bit-identical-buildable on
Turing**: bit-identity pins the denom to the 128-strided tree ⇒ Bk=128, but a Bk=128 K-tile at hd=128 =
**64 KB maxes sm_75 shared alone** — K+V can't co-reside (128 KB). Three explicit paths now in
`task-prefill-attention.md`: (1) **bit-identical 2D key+dim tiling, ~1.15×** — intricate, multi-session,
byte-exact-critical (removes ~half the O(M²)); (2) **reduction-order re-baseline, ~1.3×** — one goldens
refresh (deterministic per-query denom, Bk-free), but cascades to decode + `attn_batched`/`splitkv_*`;
(3) **tolerance-gated flash, largest** — abandons bit-identity. **Banked:** prefill is past usability
(2.1 s) and can't reach parity (§7), so none is a release lever — fund path 1 or 2 as a focused campaign
only if prefill speed is wanted for its own sake.

### B2. The GEMV's remaining dp4a headroom — **PROFILED (2026-08-04); RETIRED**

ncu at the gate/up shape (N=8960/K=1536/M=512): occupancy **88.9%** (not starved), DRAM **2.2%** (not
bandwidth-bound), compute+memory **~54%**, dominant stall **scoreboard-on-data 43%** / 59%
no-eligible-warp. The 46% is **un-hidden dp4a→accumulate-chain latency at already-high occupancy** —
RN×MT already gives 32-way ILP, and more accumulators reorder the fold (breaks bit-identity). This is
§7's dp4a-vs-IMMA ceiling: perfect latency hiding buys ~1.85× *to the dp4a ceiling*, but the ceiling is
dp4a; the real win is tensor cores (the per-row-scales fork, §7). **Not a bit-identical lever — retired.**

### B3. Tensor cores via per-row scales — see §7

---

## 6. Campaign C — coverage (widens who benefits; makes nothing faster)

### C1. Coverage audit — **DONE (2026-08-04, `TestPrefillCoverageAudit`)**

`PrefillLast` declines: MoE, gemma4-moe, ~~sandwich norms~~ (**lifted**), ~~qk-norm~~ (**lifted**),
K=V, ~~int8~~ (**lifted 2026-08-07, C6**), non-uniform, over-cap.

**Two orthogonal axes — do not conflate them.** "Covers 7 of 23 families" is the *family-architecture*
axis: which resident decode paths have a batched-prefill counterpart. The *quant* axis is separate —
within a covered family, which weight kinds batch. As of C6 the quant axis is **fully open**: int4,
int4mix, int8, and int8int8 all batch; only native f32 (unquantized) stays sequential. Before C6 the
family figure was silently gated by a quant restriction — a covered family at `--quant int8int8` still
fell back — which is exactly the leak C6 fixed. The 7/23 below is unchanged by C6 (it removes a quant
restriction, not a family guard); the `int8` decline is struck from the list above.

**Result: batched CUDA prefill covers 7 of 23 validated families** — `llama`, `mistral`,
`phi3`, `qwen2`, `qwen2_5_vl`, **`qwen3`**, and **`gemma3`** (both guards extended 2026-08-04,
below) — **now at every quantized weight kind, not int4-only**. The other 16 fall back to sequential,
by binding guard:

| # | guard | families |
|---|---|---|
| 6 | not resident (family class) | gemma4, gpt-oss, gpt2, granitemoehybrid, llama4_text, qwen3_5_moe |
| 3 | not resident (MLA) | deepseek_v2, deepseek_v3, kimi_k2 |
| 2 | MoE | glm4_moe, mixtral |
| 1 | not resident (moe-gated-shared) | qwen2_moe |
| 1 | not resident (yarn-mscale) | mellum |
| 1 | not resident (non-gated-mlp + ssm) | nemotron_h |
| 1+1 | not resident (cohere features) | cohere, cohere2 |

**Guard extension #1 — qk-norm (`cuda.TestPrefillLast_qwen3`):** per-head Q/K RMSNorm is a per-row
op, so `qk_norm_batched` = the M=1 `qk_norm` + an M dimension, **bit-identical per token**. Validated
on real Qwen3-1.7B @int4 (KV bit-identical all 28 layers × 56 rows, logits bit-identical, 64-token
decode byte-identical). The batched glu/attn already covered GELU-tanh + sliding window.

**Guard extension #2 — sandwich norms (`cuda.TestPrefillLast_gemma3`):** Gemma norms the attn/MLP
sublayer output *before* the residual add. `rmsnorm_f32_batched` (per-row plain RMSNorm) + wiring the
o-proj/down GEMVs to write a temp → norm → residual-add (mirrors the decode `segB` path). With qk-norm
already batched and the batched pre-norm already applying `(1+w)` (addOne), sandwich was gemma3's LAST
binding guard — profiled the real Gemma-3-4B: kEqV=0/34, finalSoftcap=0, uniform geometry, so nothing
else surfaced. Validated @int4 (KV bit-identical all 34 layers × 56 rows, logits + 64-token decode).
*Caveat: gemma3 batches only when the gemma resident path is enabled (`GOINFER_GEMMA4_RESIDENT`); else
it stays on the staged sequential path.*

**Release-narrative consequence:** the batched-prefill / TTFT win (§9) is now **llama/mistral/phi3/
qwen2 + qwen3 + gemma3 + qwen2.5-VL** — the dense mainstream *plus the two most-used gated families*,
not just "dense." Still NOT "all prefill" (MoE + MLA + the not-resident classes remain sequential).

### C2. WebGPU `ForwardNoLogits` — **small, scoped, unbuilt**

WebGPU's resident decoder still runs full-logits M=1 prefill. `Run` computes the head inline,
so it needs a no-head variant. Blocked only by the absence of an in-package test harness for
`gpu/` — which is itself worth building, as it blocks more than this item.

### C3. Metal batched prefill for non-dense families

Metal's `PrefillLast` is dense-uniform only; it declines MoE and Gemma-4 per-layer geometry.

### C4. 26B non-expert half (CUDA) — **bounded, deferred**

Extend `PrefillLast` past the dense-only guard for gemma4moe: dense/attention projections
batched at M=len, experts sequential per token, per-token join. Mixed-M was verified
structurally sound (per-row norms, no cross-token reduction).

- **Bound:** 786 MB batchable / 714 MB sequential = Amdahl ceiling **2.10×**; realistic
  **1.73× at 128, 1.64× at 512, 1.43× at 2048** — shrinking with prompt length.
- **Verdict:** plumbing, not a TTFT fix. Days-level build.

### C5. Lever 4 — expert-major prefill batching

Gather all M tokens per expert, one GEMM per expert. Unsolved anywhere in the repo; Metal
declines it and has no reference. `moe.ptx`-constrained, and floored by per-token expert
streaming that does not batch.

### C6. int8 batched prefill — **DONE (2026-08-07, `gemv_w8a8_batched.cu`)**

**Built and gated.** `cuda/gemv_w8a8_batched.cu` (the M=1 `gemv_w8a8_fwd` plus an `MT`=32-wide
column loop), the `kind == "int8"` branch in `bGemvB`, and `nonInt4Kind` → `nonBatchableKind`
(admits any bundle whose seven projections are each int4 *or* int8; mixed int4/int8 — `int4mix` —
batches per projection). Bit-identity is **free by construction** as predicted and was proven at
three levels, on both int4 and int8int8:
- **kernel** (`TestGemvW8A8Batched_bitIdentical`): bit-for-bit vs per-row `gemv_w8a8_fwd`, M ∈
  {1, 8, 45, 100} (45 clamps the last MT tile; K/4=40 gives a partial lane-strided dot).
- **prefill** (`TestPrefillLast_e2e`, mistral-tiny-window): KV bit-identical across **all layers ×
  all 56 rows** + last-token logits bit-identical, prompt past the window (16) and not a multiple
  of MT.
- **e2e** (same test): 64-token greedy decode **byte-identical** to the sequential path.

The three checkable conditions all hold: no int32 overflow (K ≤ 8192 ≪ ~133k), the final f32
expression copies `gemv_w8a8_fwd` verbatim (`__fmaf_rn(__fmul_rn((float)acc, wScale[n]), aScale[m],
bias)`, no bare `a*b*c`) and is in `TestKernelFMALint`'s file list, and the per-row `aScale[m]` is
the same per-row quantizer the int4 lane already uses.

**Measured (qwen2.5-coder-1.5b, RTX 2070 SUPER, best-of-3, sequential → batched TTFT):**

| N | int8int8 seq | int8int8 batched | int8 speedup | int4 batched (ref) |
|---|---|---|---|---|
| 128 | 642 ms | 312 ms | **2.06×** | 82 ms (5.66×) |
| 512 | 2.73 s | 1.47 s | **1.86×** | 372 ms (5.34×) |
| 2048 | 12.3 s | 6.69 s | **1.84×** | 2.14 s (4.42×) |

The int8 win (~1.8–2.1×) is smaller than int4's (~4.4–5.7×) exactly as §C6 flagged: the batched
kernel is bandwidth-bound and int8 doubles the weight bytes per row, so batching amortizes fewer
weight re-reads. But the 9× sequential trap the C1 headline hid is gone — an int8int8 prompt is now
~2× faster at load-out-of-box lengths, and every quantized mode (int4/int4mix/int8/int8int8) batches.

---

<details><summary>Original C6 scoping (pre-build) — retained for the argument that made bit-identity free</summary>

**The gap.** C1 lists `int8` among the `PrefillLast` declines, but the coverage headline is
stated per *family*, and that framing leaked into the release notes. It hides a case that is
neither rare nor small: `--backend cuda --quant int8int8` on a family that IS in the covered
seven (llama, qwen2, …) still takes the sequential prefill. Measured on a 300-token prompt
(0.5B, RTX 2070 SUPER): **TTFT 1.73 s vs 0.19 s (9×), 4.56 vs 0.22 CPU-seconds (20×)**, no
compute hotspot — the CPU spin-waits through 300 sequential launches. Nothing reported it;
that half is now fixed (serve prints the resolved path, `--require-backend` refuses to start,
`docs/cuda-backend.md`). The visibility work is a **stopgap**, not the answer.

**Why it should be cheap — and why bit-identity is *free* here, not engineered.** int8 weights
are **per-row symmetric**-scaled: one `wScale[n]` per output row, one `aScale` per activation
row, no group axis (`cudaWQ.ws`, `gemv_w8a8.cu`). So the whole dot product accumulates in
**exact int32** (`__dp4a` → `int acc`, integer warp-reduce) and the scales apply **once at the
end**: `dst[n] = (float)acc * aScale * wScale[n]`.

Integer addition is associative and exact, so **the result does not depend on reduction
order**. A batched W8A8 GEMM may tile K, stage through shared memory, or re-associate the
reduce however it likes and still produce the identical `acc`. That is the opposite of the
int4 situation, which `gemv_w4a8_batched.cu` states in its own header: group scales force the
cross-group sum into **float**, float add is not associative, so bit-identity had to be
hand-built by copying the M=1 visit order verbatim — and that is precisely why the int4 lane
has no IMMA path.

**That objection does not transfer to int8.** Tensor-core IMMA accumulates in int32; reordering
an exact integer sum changes nothing. This does **not** reopen §7 — §7 was refuted for
converting *group-scaled int4* to per-row scales, which destroyed quality (ppl 108 vs 28.5).
int8 is already per-row symmetric as stored. No requantization, no quality question.

**Bit-identity holds subject to three checkable conditions**, none of them hard:
1. **No int32 overflow.** Worst case `K·127²`; K would have to exceed ~133k. Real K ≤ 8192 →
   ~16× margin.
2. **The final f32 expression is evaluated identically** — same left-to-right
   `(float)acc * aScale * wScale[n]`, no FMA contraction, no double promotion.
3. **Per-row activation scale.** Batched needs `aScale[m]`, not a scalar. The int4 batched path
   already quantizes per row (`aScB = r.af(M)`), so the quantizer exists.

**Scope.** A `gemv_w8a8_batched.cu` (the existing GEMV plus an MT-wide column loop —
*structurally simpler* than the int4 one, since there is no group-scale bookkeeping), the
`kind == "int8"` branch in `bGemvB` (currently a hard `int4-only` error), and relaxing
`nonInt4Kind` to admit a uniformly-int8 bundle. Everything else in `prefillCore` is already
weight-kind-independent: `rmsnorm_f32_batched` / `qk_norm_batched` / the attention kernels work
on f32 activations. Mixed bundles (`int4mix`) fall out for free, since dispatch is per
projection.

**Unmeasured.** The *speed* is not predicted here — the int4 batched kernel is
L1TEX-latency-bound, not DRAM-bound, and int8 doubles the weight bytes per row, so the win is
whatever profiling says and must be measured, not assumed. What is argued above is only that
the **bit-identity gate is free by construction**, which is the part that usually costs the
campaign. Profile the unit first (§11).

</details>

---

## 7. The bit-identity fork — the strategic decision

> **DECISION (2026-08-04): DEFERRED — tensor cores are not pursued.** Both halves of the fork are now
> measured, and neither justifies opening it: the CHEAP path (per-row scales via an MSE scale search,
> no rotation) is dead — Phase 0b showed the 1.24× weight-space error blows perplexity up ~4× at the
> output (108 vs 28.5); and the EXPENSIVE path (rotation + IMMA) buys a prefill-only tensor-core ~3× on
> a TTFT already past its usability threshold plus a decode-stream win that's partial (~mid-single-digit,
> not the naive 11%, capped by ~45% DRAM-bandwidth utilization). The format stays group-scaled int4;
> everything in §5 remains available. **Reopen only if a decode-BW profile shows the ~11% is largely
> realized AND someone funds the rotation campaign — both are now costed, neither assumed.** The Phase 0
> / 0b measurements and the payoff accounting below are retained as the decision record.

**Ollama has no bit-exactness contract at all.** Not between prefill and decode, not across
backends, not across versions. Its batched prefill runs FP16-accumulate GEMMs on tensor cores
while single-token decode runs a different path, so the KV cache prefill writes is not what
sequential decode would have written — and nothing checks, because nothing claims it should.

**But that is not the main reason they are faster at prefill.** cuBLAS is years of vendor
tuning on silicon built for the shape. A bit-exact tensor-core kernel would not match it out
of the gate either.

**And the constraint is more specific than "bit-exactness costs us tensor cores."** goinfer's
int4 carries an f16 scale per 32-element group, forcing a float accumulation every 8 values;
float addition is not associative, so no tensor-core tiling reproduces it. With **per-row**
scales, IMMA's int32 accumulation is exactly associative and a tensor-core GEMM would be
bit-identical **by construction**. Tensor cores and bit-identity are compatible in principle
— *group-scale granularity* is what makes them incompatible here.

### The two live options

**Keep the format.** Accept the prefill ceiling. Everything in §5 remains available; the
tensor-core 3× does not.

**Pay one parity refresh** and move to per-row scales, with rotation making the coarsening
quality-neutral. Scoped in `task-rotation-perrow-imma.md`, explicitly **not funded**. Note
that doc's motivating estimate still cites the stale ~23× weight-amortisation figure and must
be re-derived from the profile before anyone opens it.

**Phase 0 — MEASURED (2026-08-04, `TestPerRowScalePhase0`).** The cheap gate the doc called for is
now a number, not a guess. Real qwen3-1.7B weights (Q8→f32 proxy, real outlier distribution),
`sym` int4 reconstruction rel-error ‖W−Wq‖/‖W‖ across all 196 projections:

| scale granularity | rel-error | vs per-group |
|---|---|---|
| per-group `sym` (**shipped**, 32-elem groups) | 0.100 | 1.00× |
| per-row `sym` (naive maxabs, IMMA-associative) | 0.174 | **1.73×** |
| per-row `symmse` (MSE-optimal per-row scale, **no rotation**) | 0.125 | **1.24×** |

**The finding refutes the doc's own framing in BOTH directions.** Not "almost nothing" (naive
per-row is 1.73× — a real hit, down-proj worst at 1.95×), so per-row-as-a-free-swap is dead. A per-row
**scale search** (a cheap grid over the maxabs scale, no rotation) recovers most of the *weight-space*
gap to **1.24×** — which LOOKED like it might make the fork cheap. It does not.

**Phase 0b — MEASURED (2026-08-04, `TestPerRowScalePhase0b`); the cheap path is a MIRAGE.** Teacher-
forced forward quality on real qwen3-1.7B (bf16 safetensors — the GGUF loader row-quantizes directly
and bypasses the fakequant seam, so the probe uses the safetensors path), 87 tokens, each build vs the
f32 oracle:

| build | top-1 agree | meanKL(f32‖x) | perplexity |
|---|---|---|---|
| f32 oracle | — | — | 26.75 |
| per-group `sym` int4 (**shipped**) | 76.7% | 0.269 | 28.54 |
| per-row `symmse` int4 (§7 fork, no rotation) | 68.6% | **1.393** | **107.97** |

**The 1.24× weight-space error compounds CATASTROPHICALLY through the 28-layer stack: perplexity blows
up ~4× (28.5→108), KL 5×, agreement −8 pts.** This is the repo's own recurring lesson — a small
per-tensor weight error is not a small output error once it propagates (discrete argmax flips + residual
compounding, the same class as the MoE routing-sensitivity finding). **The weight-space proxy was
misleading; the forward gate caught it.** So per-row scales WITHOUT rotation are dead — the MSE scale
search is not enough. **Rotation goes back to PREREQUISITE, not last-resort** (correcting the optimistic
read from Phase 0 alone). Combined with the payoff analysis below (tensor-core 3× is prefill-only on a
past-threshold number; decode-stream win is partial), §7's realistic shape is now: **an expensive
rotation+IMMA campaign for a modest, mostly-prefill payoff — lean toward keeping it CLOSED.** The cheap
version measured out; what remains is the campaign the doc always feared, now with both halves priced.

### The PAYOFF half — per-row scales also shrink the decode weight stream (the axis that matters)

The tensor-core 3× is the WEAKER half of §7's payoff, and prefill-only. Worked through: at 2048,
prefill is ~2.1 s of which the GEMV is ~61% (~1.28 s); a full 3× on that lands total prefill at
~1.25 s — a 1.68× on a number already past its usability threshold, still ~4× behind Ollama's ~0.29 s
(not 7×). Decode is M=1 and gets nothing from tensor cores. On the prefill case alone, **close §7,
don't advance it.**

The stronger, previously-uncosted payoff is on the DECODE axis (where goinfer is at parity): **per-row
scales delete the group scales from the weight byte-stream.** MEASURED against the actual layout
(not assumed — the ~23×-estimate lesson): the resident GPU int4 carries an **f16** group scale per
32-value group (`cuda/resident.go:41` — deliberately f16 because "f32 would be 20%"), so scale bytes
are `2 / (16 + 2) = 11.1%` of the int4 weight stream. Per-row (one scale per row) deletes ~all of it
(≈0.2%). Decode reads every weight once per token → fewer weight bytes is the decode lever the dp4a
ceiling (B2) is NOT.

**But the byte→speedup conversion is PARTIAL, not the "unconditional 11%" it looks like — and this is
measured, not assumed.** Two data points: (1) B2 profiled the GEMV at **M=512** and found it
*latency*-bound (DRAM 2.2%, scoreboard-on-data 43%), not bandwidth-bound — boundedness is M-dependent.
(2) Decode is M=1 (no weight reuse, the textbook bandwidth-bound regime), but the measured decode rate
puts it at only **~45% of peak DRAM bandwidth** (~1.0 GB resident weights / token ÷ ~4.83 ms/tok @207
tok/s ≈ ~200 GB/s vs the 2070S's ~448 GB/s). Running at 45% of peak means decode is NOT
bandwidth-saturated — latency/occupancy/dispatch caps it — so removing 11% of the weight bytes buys
**less than 11%**, likely mid-single-digits. The honest figure is "**up to ~11% decode, realistically
partial; size it with an M=1 GEMV ncu DRAM SpeedOfLight before funding.**"

**Both sides of §7's ledger are now on the record, measured:** quality cost = per-row-symmse blows
perplexity up ~4× in the forward (Phase 0b — NOT benign; needs rotation to fix, expensive); payoff =
~11% smaller decode stream capped by ~45% bandwidth utilization (partial, ≈mid-single-digits realized;
size with an M=1 DRAM% profile before funding) + a prefill-only tensor-core 3× that alone doesn't
justify the fork. **Verdict: the cheap path (scale search, no rotation) is measured dead, and the
expensive path (rotation + IMMA) buys a modest, mostly-prefill payoff. Lean CLOSED.** Reopen only if a
decode-bandwidth profile shows the ~11% is largely realized AND someone funds the rotation campaign —
both now costed, neither assumed.

### The tempting middle path, and why it is a trap

Relax the invariant from "bit-identical KV" to "identical token stream" — let prefill use
FP16 accumulate and gate on the 64-token decode matching. Users observe tokens, not KV bits.

**This project already has the counterevidence.** Discrete decisions flip on ~0.001 margins —
the MoE routing-sensitivity finding measured exactly this — and argmax over a vocabulary is a
discrete decision. A tolerance-gated prefill would produce identical tokens *most* of the
time and diverge at near-ties, converting a clean gate into an intermittently-red one. Worse
than either strict alternative, and it would surface months later on a family nobody was
watching.

---

## 8. Campaign D — levers not yet considered anywhere

### D1. Speculative decoding — **SHIPPED on resident CUDA (2026-08-04); serve-wired + lossless-gated**

> **Status:** the CUDA-resident batched-verify spec-decode is built, measured, AND serve-wired.
> `serve --spec ngram` now engages it for GPU-resident models (the OpenAI handler routes a resident
> model through `Model.GenerateNgramSpeculativeAdaptive` when `--spec ngram` and the request isn't
> grammar-constrained; a validation error falls back to plain resident `Generate` before the KV is
> touched, so the fallback is exact). Measured lossless-vs-sequential on the real 1.5B
> (`TestSpecDecodeCurve`): **1.23× @128, 1.86× @512, 1.18× @2048** (compresses at long context — the
> M=k verify re-reads KV for all k positions; the win is on copy-heavy traffic — code edits / RAG /
> agent loops). All depths byte-identical to plain greedy.
>
> **Two changes made it ship-able** (both landed): (a) cuda `ForwardN` now routes the spec verify
> through the batched `prefillCore` (`PrefillLastN`) instead of a sequential per-token `step` loop —
> the amortization is the entire win; it falls back to the sequential loop for archs the batched path
> doesn't cover (MoE / K=V / non-int4 / non-uniform), distinguished by `errPrefillDeclined`. Bit-identical
> because the contraction fix made `prefillCore == decode`. (b) `genNgramInto`'s resident branch now
> claims the shared resident KV (`resBusy`, like `generateInto`) so two concurrent requests can't
> corrupt it — a loser falls back to the staged CPU path. Gates: `TestResidentSpecServe` (the serve
> entry byte-identical to plain `Generate`), `TestSpecDecodeCurve` (speedup + lossless), unchanged
> `TestPrefillDivergenceRate` 0/50. Grammar-constrained resident requests keep plain `Generate` (a
> resident grammar-fused drafter is a later step). The decoder-level `--spec ngram` on the STAGED/CPU
> path (via `forwardN` + f64 attention) is a separate, already-shipped implementation.

A drafter proposes k tokens; the target verifies them in one batched forward. Under greedy the
verify accepts only tokens the target would have produced, so **the emitted stream is identical by
construction** — compatible with the determinism contract in a way KV quant is not.

**Ceiling measured** (`TestSpecVerifyCeiling`, 1.5B @ depth 1024) — the whole win hinges on the
batched M=k verify being cheaper than k sequential decodes (decode is BW-bound; batched reads weights
once for k). It is:

| k | batched(M=k) | k×sequential | verify cheaper by |
|---|---|---|---|
| 4 | 8.2 ms | 20.8 ms | **2.53×** |
| 6 | 9.7 ms | 31.0 ms | **3.20×** |
| 8 | 11.6 ms | 41.6 ms | **3.59×** |

So D1 **can win**: real speedup ≈ (verify ceiling) × (accept rate). Free **n-gram** drafts on code
(repetitive → high accept) ⇒ a realistic **~1.6–2.5× decode**. Largest decode lever in the doc, now
measured not speculative. **Draft choice: n-gram (free), NOT a draft model** — the WebGPU draft-model
spec was draft-dominated (0.42×, `gpu-spec-decode-lever2.md`); n-gram has zero draft cost.

**Build scope (~1.5–2 days — over a one-day box, so funded-next, not now):**
1. *Batched all-position-logits verify* (the long pole): `PrefillLast` runs the batched stack but
   heads only the last row; verify needs logits at all M positions (LM head for all M rows +
   readback). `ForwardN` exists but is **sequential** (= k decodes, no win) — this is genuinely new.
2. *N-gram drafter*: reimplement minimal on main (the `spec-decode-ngram` branch is unmerged and
   conflict-heavy post-kernel-work — reimplement the context-lookup, don't merge).
3. *Speculative loop + lossless gate*: draft → batched verify → accept longest greedy-matching prefix
   + 1 bonus token → repeat. KV mgmt ≈ free (positional, overwritten). Gate: accepted stream ==
   sequential greedy, byte-identical.

**Status: BUILT & MEASURED — lossless + 1.53×; blocked on a bit-identity fork (2026-08-04).**

Built the full mechanism: `ngramDraft` (prompt-lookup, free), `PrefillLastN` (batched all-position
verify — `prefillCore` refactor of `PrefillLast`), the accept-longest-prefix + bonus loop, gates.
`TestSpecDecode` (1.5B, greedy, self-similar workload): **LOSSLESS** (200 spec tokens byte-identical to
the batched-forward greedy stream) and **1.53×** (158 → 242 tok/s; accept-rate 0.61, 2.04 tokens/round).
So D1 delivers.

**The fork (why it's not shipped yet):** the verify uses the **batched** forward at **startPos>0**
(appending to context) — a regime production never exercised (`PrefillLast` is only ever called at
startPos=0). There the batched forward diverges from the **decode-step `Forward`** by **~1e-6**
(byte-identical at startPos=0, which is all `TestPrefillLast_e2e` gated). So spec-decode is lossless
w.r.t. the *batched* forward but **not** bit-identical to the *decode* path — enabling it would change
output at rare near-ties, breaking the "flip a flag, same tokens" contract. rope_kv ≡ rope_kv_batched
(mathematically identical) and K-after-rope is byte-identical (KV gate), so the cause is in
attention-over-primed-KV; **unpinned** (`TestBatchedVsDecodeGap` is the committed repro).

**Three ways forward (a decision, not a default):**
1. **Unify the two forward paths** — find/fix the startPos>0 batched-vs-decode ULP gap; then the batched
   verify == decode and spec is a true bit-identical drop-in. Cleanest for the thesis; cause unpinned.
2. **Route decode through the batched path** (Forward → PrefillLast M=1) so spec and non-spec share one
   forward — consistent, but re-baselines the decode parity goldens.
3. **Ship spec as a documented mode** whose output is the batched-forward greedy stream — fastest to
   ship, but concedes the exact-match-with-non-spec property at rare ties.

The mechanism (drafter + batched verify + loop) is done and gated; only the path-unification decision
gates shipping.

### D2. Multi-token prediction / Medusa-style heads

Same family as D1, without a separate draft model. Requires model support; most checkpoints
do not have the heads.

### D3. Continuous batching / server-side concurrency

Improves throughput under concurrent load, not single-stream latency. Different axis from
everything else in this doc, and not what the current comparison measures — but it is what a
serving deployment actually cares about, and it is unexamined.

### D4. Confirm what Ollama actually does at long-context decode — **DONE (2026-08-04): flash attention**

Confirmed at the source. v0.32.5's engine is **llama.cpp** (`llama-server` + `libggml-base`), launched
with **`--flash-attn auto`**; the debug load log reports **`resolve_fused_ops: Flash Attention
enabled`** for the qwen2 1.5B on CUDA (`OLLAMA_FLASH_ATTENTION` default is `false`, but `auto` lets
llama.cpp turn it on per-model — and it does). KV is **f16 by default** (`K (f16) / V (f16)`).

So Ollama holds ~188 tok/s at 2048 because it runs the fused **`flash_attn_ext`** kernel — tiled,
parallel over the KV dimension, online-softmax (streaming max + rescale). **This is exactly the
split-KV shape the A1-reprofile named** (§A2): FA fills the SMs by parallelizing over keys, which is
why its rate stays flat where our 12-block-per-head kernel starves.

**Two implications for the build:**
1. Target validated — Campaign A's split-KV decode attention is chasing precisely what FA does.
2. **FA's online rescaling is NOT bit-exact** vs a serial pass (it reorders the softmax sum), so
   Ollama has no bit-exactness contract (confirms §7). To match FA's *parallelism* while keeping our
   bit-identity contract, we take the **two-pass** route (materialize tile scores → one global-max
   reduction → exp-weighted sum combined in fixed tile order), not FA's one-pass online rescale.

### D5. Hybrid GPU/CPU **layer split** — the right shape for an oversized model — **scoped, not built**

Ollama runs Gemma-4 **26B-A4B** at **24.5 tok/s** on the same 8 GB card that goinfer's expert
paging gets **16.98** (both measured, §B4). The difference is architectural, and it is worth
stating as a mechanism, not a number:

- **Layer split (Ollama):** partition *layers* between GPU and CPU. The only thing that crosses
  the PCIe boundary is the **activation vector** at the split point — `hidden × dtype ≈ 10–16 KB`
  per token, ~1–2 µs at ~12 GB/s. Negligible. The cost is that 58% of the layers run on **CPU
  compute**.
- **Expert paging (goinfer today):** keep every layer on the GPU, stream expert **weights**
  host→VRAM per token — **~380 MB/token**, ~31 ms of pure DMA at ~12 GB/s → a hard ceiling near
  **~31 tok/s** before any compute (measured 16.98 with overhead). Weights are ~10⁴× the size of
  the activations a layer split moves.

**So expert paging is very likely the wrong shape for an oversized model on a PCIe-attached GPU.**
It moves the big thing (weights) across the slow link every token; the layer split moves the small
thing (activations) once. This is consistent with the **Metal** track also hitting a floor on the
same model class. The durable value of the whole host↔VRAM paging line (`task-moe-streaming.md`) is
the **method record** — the LRU expert cache, the slot-id device-read trick, the mixed-M join, the
isolation-proves-the-primitive-never-the-composition lesson — **not the throughput.**

**goinfer already has both compute paths**: the pure-Go CPU decoder (**5.53 tok/s** full-model on
this 26B) and the resident CUDA runner. What is missing is *the split and the boundary*, not a new
kernel.

**The bound (why this is not a quick win, and what it would take to beat 24.5):**

- **Boundary:** fill VRAM with as many **contiguous** layers as fit (~42% here, the fraction
  Ollama achieves), rest on CPU. Contiguous ⇒ one activation hand-off GPU→CPU and one CPU→GPU per
  token. Transfer cost ≈ nil (see above). Bit-identical by construction — it moves *where* operands
  live, not the order they accumulate (same argument as int2-coalescing / A-staging / A2).
- **Ceiling is set by goinfer's CPU throughput, not by the mechanism.** Full-model pure-Go on this
  26B is **5.53 tok/s ≈ 181 ms/token**; the 58% that would live on CPU therefore costs **~105
  ms/token** on its own. Even with a free GPU half and a zero-cost boundary, the split **tops out
  around ~9–10 tok/s — below Ollama's 24.5.** Ollama wins here because its GGML CPU kernels
  (AVX2/AVX-512, threaded) are **~4× goinfer's pure-Go path**, not because its split is cleverer.
- **To close toward 24.5, two independent knobs, both outside this item:** (a) a **faster CPU
  kernel** — exactly the `cpubrrr` Q8_K integer-accumulation lane in
  `plan-cpubrrr-steal-and-bindings.md`. That work was **built, proved bit-exact, and declined** on a
  **Q6_K byte-ratio ceiling of 1.22×**, with the **Q4_K variant (ceiling 1.78×) explicitly left
  open** — declined at the time *because the CPU path did not matter*. **D5 is the reason it would
  matter.** (b) A **larger GPU fraction** (more VRAM — a card the model nearly fits).
- **But be honest about the ceiling: this revalidates the lever, it does not win the 26B.** Even at
  Q4_K's full **1.78×** CPU speedup the split lands around **~16–18 tok/s against Ollama's 24.5** —
  the CPU half is still the floor. So D5 is a *right-shape capability* item (single-stream on a card
  too small, at a usable rate) and a *reason to revive cpubrrr Q4_K*, **not a path to beating Ollama
  on this model.** Read it that way; do not let it read as a route to victory. (More VRAM is the only
  knob that actually wins, and that is buying hardware, not writing a kernel.)
- **Verdict:** right mechanism, real capability (single-stream on a card too small), but **not a
  throughput win until the CPU path closes to GGML-class.** Rank it **alongside D1** (both are
  decode-side, both reuse machinery goinfer already has) and **above anything remaining in
  §5/prefill** — prefill improves a number already past its usability threshold and cannot reach
  parity, whereas this changes what a small-VRAM box can run at a usable rate. Days-to-weeks build,
  gated on the CPU-kernel decision.

### D6. Sampling on the critical path — the `temperature==0` readback cliff — **SCOPED, evidence-attached**

Surfaced by outside-consumer testing of the released **v0.10.2** across a five-model sweep. Two
distinct sampling cliffs were measured; **one is now fixed, one remains and is the lever here.**

**Fixed this session (host-side, `decoder/sampler.go`).** The top-p/top-k selection path softmaxed
all V and then **full-sorted all V** (`sort.Slice`) on the host, single-threaded, every token — the
reporter's inferred "O(V×k) successive scans" was actually an O(V·log V) sort whose Go
reflection-per-comparison constant made it *present* as linear in V. Replaced with bounded
logit-space selection (k-bounded heap for top-k, adaptive nucleus for top-p; exp applied only to the
retained set). Measured on synthetic logits at the reporter's two vocab sizes, temp 0.8 + top_p 0.95:

| vocab | selection: full-sort | selection: bounded | full-path temp+top_p ÷ temp-only |
|---|---|---|---|
| 152 064 (qwen2.5) | 122.3 ms/tok | **1.80 ms/tok** (68×) | ~7× → **1.10×** |
| 262 144 (gemma3) | 233.0 ms/tok | **4.37 ms/tok** (53×) | → **1.48×** |

That closes the *top-p/top-k* portion of the reporter's ~15 tok/s cell (the ~361–418 ns/entry the
sweep attributed to "linear in V" was the sort). It does **not** touch the second cliff:

**The remaining lever — the `temperature==0` branch.** `Sampler.ArgmaxEquivalent()`
(`decoder/sampler.go`) is true only for greedy: `Temperature <= 0 && no bias && no penalties &&
!Logprobs`. When true, the resident CUDA/Metal backends return the token from an **on-device argmax
and never read the V-wide logits back to the host**. Any nonzero temperature — even `0.01` — flips
that branch, forcing a **full V-float logit readback per token + a host softmax over V**. The
reporter measured this as a **~2.9× cost for any nonzero temperature** (greedy → temperature-only),
before top-p is even applied. Reporter's sweep, per config (end-to-end, incl. model forward):

- greedy ≈ up to ~310 tok/s (qwen2.5-coder-0.5b); temperature-only ≈ ~100 tok/s; temp+top_p ≈ ~15
  tok/s (that last cell now recovers to ≈ the temperature-only figure via the host-side fix above).
- **gemma3-1b @ 262k carries the largest penalty** (widest vocab → largest per-token readback).
- The cost scales with V because it is a **per-token host-copy of the whole logit row**, not
  compute — so the design must move the selection to the device, not merely speed up the host.

**Design sketch (device-side, under the bit-identity contract).** Add a device-side **top-K
reduction** (K a small fixed bound, e.g. 256–1024) to the CUDA and Metal decode kernels: return only
the K highest logits + ids, not the V-wide row. The host runs the existing exact sampler over those
K. Correctness bound: the retained nucleus must fit inside K, so the host **verifies the returned K
carry mass ≥ top_p** and, on the rare short read, **falls back to a full V readback** for that token
(adaptive bound verified host-side — the same shape as the host fix's `topPCandidates`). This is
bit-identical to the current sampler on every token where K suffices, and exactly the full path
otherwise. The reduction itself is FMA-free (a max/compare tree), so it sits cleanly under the
existing kernel bit-identity lint. **Not attempted here** (scope): it needs new CUDA + Metal kernels
and a device box to validate, so it is a campaign, not a patch.

### The second, larger cost the same fix removes — full-vocabulary normalization

D6 was scoped as the `temperature==0` **branch cliff**. That is only half of it. The
temperature-only path (no `top_k`, no `top_p`) still **normalizes over the whole vocabulary**: a
softmax across all V, every token, on the host. This is a separate, additive cost from the readback
branch, and it is the larger one on wide-vocabulary models.

Measured (v0.10.3 verification, one box, one session, interleaved):

| model | vocab | temp-only added vs greedy | per vocabulary entry |
|---|---|---|---|
| phi3-mini | 32k | +1.19 ms | 37.2 ns |
| qwen2.5-coder-0.5b | 152k | +6.66 ms | 43.8 ns |
| gemma3-1b | 262k | +11.57 ms | 44.1 ns |

**~44 ns per vocabulary entry, flat across a 32k → 262k span** — i.e. genuinely linear in V, unlike
the "linear" top-p cell that turned out to be a sort. On a large-vocabulary model this is a **3.1×
throughput factor**, and it is why plain `temperature` is now the *slowest* sampled configuration:
adding `top_k=20` is faster than leaving it off, because `top_k` bounds the set that gets normalized.

**RESOLVED (2026-08-09) — P2b shipped the fix, and it was neither of the two approaches below.**
The full-vocabulary normalization is now done in PARALLEL over a fixed 64-chunk split with an
ascending-chunk reduction (machine-independent by construction), drawing against unnormalized
weights so the divide pass is gone. Host-only, no kernels. Measured: **3.06× at 152k, 4.72× at
262k**; end-to-end temperature-only **97.6 → 220.2 tok/s** (qwen2.5-0.5b) and **56.6 → 134.2**
(gemma3-1b). The top-p denominator got the same treatment in the same release (**+98% / +108%** on
the filtered path). Given-seed sampled output changed once, for both paths; distribution unchanged.

**Two candidate approaches — recorded so neither is re-proposed from intuition:**

1. ~~**Lazy Z (host-side, no new kernels).**~~ **BUILT AND REFUTED (2026-08-09) — do not re-propose
   from intuition.** The idea: top-K by *logit* with no `exp`, sum those exactly as `S_K`, bound the
   unseen remainder by `R = (V-K)·exp((x_K-m)/T)`, and skip the full pass whenever the draw resolves
   to the same token at BOTH ends of Z's interval `[S_K, S_K+R]`. (That endpoint-index condition is
   itself a correction: the original "lands inside the top-K prefix" shorthand pins the prefix, not
   the token — `t = r·Z` slides as Z moves and can cross a boundary inside K.) Implemented, proven
   correct against a slow exact reference — 432 matched draws across peaked / flat / tie-heavy plus
   boundary-straddling cases — and then measured **3.3× SLOWER at 152k, 4.4× at 262k**.
   The bound is structurally too loose here: `R < S_K` needs a gap of `ln((V-K)/S_K) ≈ 11.2 nats`,
   and on real qwen2.5-coder-0.5b decode logits the gap is 5.29 nats at K=32 (**R/S_K = 366**), 8.97
   at K=256 (8.65), reaching 11.6 only at K=2048 (0.60) — where the interval still spans 60% so most
   draws straddle a boundary and grow again, each growth costing a full O(V) pass. Reverted; table in
   `docs/plan-still-slow.md` P2.
2. **On-device sampling — BANKED, and no longer the finisher.** The host term is ~78% of the
   temperature-only penalty and the readback ~2% (measured 2026-08-09), and **P2b then took the host
   term down 3.06× at 152k / 4.72× at 262k** with deterministic parallel chunked normalization —
   host-only, no kernels. So on-device sampling now addresses the ~2% readback of an already-shrunk
   gap, at the cost of new kernels on two backends plus per-backend given-seed divergence (device
   `exp` ≠ host `math.Exp` at ULP level). Re-scope and the number to beat: `docs/plan-still-slow.md` P3.

### `top_k=1` should route to greedy — **DONE (P1, 2026-08-09)**

`top_k=1` with a positive temperature is **mathematically identical to greedy**: temperature scaling
is monotonic so it preserves ordering, and a distribution restricted to a single token is
deterministic regardless of its probability. It can therefore route to the `ArgmaxEquivalent` branch
and take the on-device argmax with no readback. Measured gap it would recover — **13–18%**:

- qwen2.5-coder-0.5b: `top_k=1` **272** vs greedy **312** tok/s
- gemma3-1b: `top_k=1` **148** vs greedy **180** tok/s

**SHIPPED and measured (2026-08-09, this box).** `top_k=1` now takes the on-device greedy fast path
via the `GreedyEquivalent` sampler predicate (`decoder/sampler.go`), so the readback is skipped.
Decode-only, prefill excluded; RTX 2070 SUPER / driver 595.58.03; goinfer at the P1 commit; q4_K_M
int4; 128-token prompt; 8 completions/run, 2 runs/cell, spread shown. "before" is the same binary
with `GOINFER_NO_GREEDY_FASTPATH=1` (the readback path), so both arms are one build:

| model | `top_k=1` before | `top_k=1` after | greedy (after) | recovered |
|---|---|---|---|---|
| qwen2.5-coder-0.5b | 271.4 ±1.8 | **319.2** ±5.5 | 315.2 ±3.3 | **+17.6%** |
| gemma3-1b | 144.3 ±0.8 | **175.9** ±1.0 | 175.9 ±3.0 | **+21.9%** |

The "before" figures independently reproduce the gap recorded above (271.4 vs 272; 144.3 vs 148),
and after routing `top_k=1` lands ON greedy — identically at 175.9 on gemma3-1b, and within spread
on qwen. Recovery is at or above the predicted 13–18%. Emitted tokens are unchanged by construction
and gated by `decoder.TestTopK1_MatchesGreedy`.

*Metal: the sampler-level predicate applies on every backend, so Metal inherits the routing; an e2e
A/B on the Mac is **pending** and not claimed here.*

*Banked, not built: `top_k=1` is now greedy-routed (ed81e13), so it could also join greedy
speculative eligibility — those paths gate on `temperature <= 0` directly, so it would be a
deliberate second change, not an automatic consequence.*

**Prerequisite — SATISFIED (2026-08-09).** Routing `top_k=1` to a device argmax requires that argmax
to break index ties the same way the host does (ascending token id, the amendment-1 contract
v0.10.3 established in `topFilterLogits`). It does, on both backends:

- **CUDA** — `argmax_reduce` returns the **lowest** index on an exact tie: audit **C-14**, fixed in
  `c6600fc` (2026-08-05) and split into its own `argmax.ptx`. Gate: `cuda.TestArgmaxTieBreak`,
  confirmed green on the box 2026-08-09 (3.91 s — it loads the 0.5B, so it ran rather than skipped).
- **Metal** — live greedy dispatches `gemv_w8a8_amax` + `argmax_finish`, both merging on
  `(v desc, i asc)` → lowest tied index. The `w4a8` fused amax is the N-09 variant and is unwired,
  so it does not touch the live path (confirmed on the Mac, `docs/plan-still-slow.md` P0.3).

*Earlier revisions of this section called the tie-break an "open audit critical" and used it as the
reason NOT to implement `top_k=1` routing. That text was written after the audit recorded the fix
and was simply stale — the fix predates it. Corrected here rather than left as a blocker that no
longer exists.*

**Rank:** decode-side, benefits every non-greedy serving user (most real chat traffic runs
temperature > 0), and reuses the adaptive-bound logic already written host-side. Above prefill;
alongside D1/D5. Lazy Z is host-only and needs no device box; the rest is gated on device-box kernel
work; `top_k=1` routing was gated on the argmax tie-break, which is now fixed and gated by test.

---

## 9. Landed — do not redo

| lever | result | notes |
|---|---|---|
| Batched prefill (`PrefillLast`), **CUDA** | 2048 TTFT 13.1→2.1s; crossover 128→320 vs 0.5.7 | **bit-identical, default-on** (restored). Was briefly default-off after an 84% divergence from fma-contraction; FIXED by explicit `__fmaf_rn` everywhere + `TestKernelFMALint`; `TestPrefillDivergenceRate` 0/50 (`task-batched-prefill-bitidentity.md`) |
| Batched prefill (`PrefillLast`), **Metal** | — | ⚠ **NOT fixed — 54% divergent** (`TestMetalPrefillDivergenceRate` 27/50, first-div ~2.9; f16-activation-MMA vs int8-activation decode + fast-math, §A2-Metal). The CUDA fix does NOT cover it (deeper root than contraction). The shared gate is now default-ON, which *would* have pulled this divergent path in by default — so the **Metal backend now DECLINES it by default** (`metal/backend.go`; falls back to sequential = bit-identical), opt-in `GOINFER_METAL_BATCHED_PREFILL=1`. Real fix needs the activation precision unified, deferred behind decode |
| KV-only prefill for `prompt[:-1]` | −4.39 ms/prompt-token on 26B | skips LM head |
| GEMV `MT=32` | ~6% | tile width, not an accumulation constraint |
| GEMV `int2` coalescing | 13% | bytes/sector 49.99 → 98.01% |
| GEMV `RN=2` register blocking | ~30% | scoreboard stall 17.8 → 7.5 cyc |
| Prefill attention `float4` | **3.1×**; 2048 TTFT 3.33 → 6.17× | bytes/sector 21.96 → 66.32% |
| Decode attention `float4` (A1, M=1 reuse) | 2048 decode **99.5 → 133.5 tok/s (1.34×)** | bit-identical to glue `attention`; `1a1914b` |

Cumulative on the GEMV ≈ **1.5×**. Total 2048 TTFT: **13.1 s → 2.1 s**. Long-ctx decode gap to
current Ollama: 1.94× → **1.41×**.

---

## 10. Refuted — do not re-propose

| lever | why it died |
|---|---|
| A-staging (shared-memory activations) | 8× traffic cut, **1.2×** time — kernel not traffic-bound. Kept unwired as the reproducible refutation |
| IMMA on the current format | GEMV at 7.9% of dp4a peak — raises a ceiling 92% unused |
| CUDA graphs as a speed lever | **1.01×** on the dense flagship; the ~19 ms dispatch figure was a 26B number. Shipped as a *safety* gate, not a speedup |
| Async miss-DMA | 26B-only (paged expert cache); dense models have no expert paging |
| Deeper unroll | wrecked the compiler's schedule — 4.41 → 25–33 ms, non-monotonic |
| `MT` > 32 | occupancy collapse (108 regs → 50%) |
| Metal `PrefillLast` as a CUDA reference | dense-uniform only; declines MoE and Gemma-4 |
| WILLNEED readahead (Metal) | faults/stage **rose** 92.5 → 147.1; cost changed bucket, total moved 4% |
| F_NOCACHE (Metal) | −1.4%, every predicted signature failed |
| Concurrent preads (Metal) | concurrency 1 already reached 3716 MB/s; queue depth was never the gap |
| Residency scoping, per-encoder (Metal) | re-validates per command buffer — phase-2 0.21 → 37.5 ms |
| N as a memory-pressure knob (Metal) | null under thermal control; the apparent effect was a thermal confound |
| Speculative expert prefetch (Metal) | exact-set match 0.1–0.3%; ceiling ≤27% of a bucket that isn't the bottleneck |

---

## 11. Method notes worth keeping

- **The five-attribution record.** Weight-amortisation ceiling → needs-IMMA →
  activation-L2-bandwidth → issue-bound → L1TEX latency. Four wrong, each reading like a
  measurement, all caught before shipping. The operational form: *profile the unit before
  designing the fix.*
- **Sub-bucket vs total.** Optimising a measured sub-bucket while a residual bucket absorbs
  the displaced cost produces a 3× that isn't there.
- **Synthetics without pressure.** Three probes reproduced configuration and omitted the
  condition that made the real thing slow.
- **Concurrency instruments that serialise.** `CUDA_LAUNCH_BLOCKING`, per-iteration `Sync` —
  they pass by removing the property under test.
- **Fixture representativeness has three axes:** depth, width, and *discreteness*. A fixture
  too shallow cannot compound; too narrow cannot route; without discrete decisions it cannot
  exhibit a flip.
- **Peer versions expire.** Re-pin and re-measure before any competitive claim ships.
- **Microbenchmarks nominate changes; full generations elect them.** A kernel/dispatch win measured in
  isolation is a candidate, not a result — the real path has costs (commit/sync cadence, readback,
  scheduling) a microbench omits, and several "wins" here reversed under a full decode (split-KV on
  Metal, the greedy on-device-argmax wiring). Our own form is the split-KV crossover lesson; stated
  more sharply, and independently derived, by dmikey's A1111 Metal write-up.

---

## 12. Suggested order

1. ~~**Campaign A**~~ **CLOSED** — A0 profile, A1 coalescing (1.34×), A1-reprofile (→occupancy), D4
   (Ollama=flash-attn, confirmed), split-KV decode attention (default-on, bit-identical, +1.20× @2048
   **on the 1.5B — its depth gate was re-characterized per geometry in P6a, see the §A2 amendment**).
   Long-ctx decode **99.5 → 160 tok/s = 1.61×**, gap to Ollama **1.94× → 1.17×**. Stopped at the
   bit-identity ceiling (V-sum ILP unroll refuted); full parity needs the §7 non-bit-identical fork,
   which A does not take. The tiling primitive is available to share with B1.
2. ~~**C1** — coverage audit~~ **DONE** (5/23 dense; the release must say "dense lane," not
   "prefill").
3. ~~**D1** — speculative decoding~~ **BUILT + MEASURED** (lossless vs sequential; 1.21× @128 / 1.80×
   @512 / 1.16× @2048, workload-dependent, `TestSpecDecodeCurve`). Unblocked by the contraction fix.
   Nice-to-haves only (grammar-fused drafter, cross-request stats).
4. **D5** — scope the hybrid GPU/CPU layer split. Right shape for the 26B-on-8GB case; gated on the
   CPU-kernel decision (`plan-cpubrrr-…`), above §5/prefill.
5. ~~**D4** — confirm Ollama's long-ctx mechanism~~ **DONE** (flash attention, §D4).
6. ~~**B1** prefill query-tiling / **B2** GEMV 46%~~ **BANKED** — B2 profiled (dp4a latency frontier,
   retired); B1 design-revised (64 KB wall → ~1.15× bit-identical or a re-baseline for ~1.3×; not a
   release lever, prefill past usability + can't reach parity). See §B1/§B2.
7. **§7 fork** (tensor cores via per-row scales) — the only path past the dp4a ceiling; Phase 0 (measure
   per-row-scale quality cost) still the cheap gate before funding.

Everything in §6 (C2–C5) is coverage work whose priority depends on C1's answer (now known: 5/23).

---

## 13. Per-token allocation & host-scratch audit (2026-08-10)

A fan-out decode audit (9 findings, adversarially verified vs the bit-identity contract) surfaced a
class the named frontiers (CUDA int4-unpack, Metal megakernel, WebGPU tiled GEMM) don't cover:
per-token host allocations and redundant per-token work. These are **GC/jitter-grade**, not throughput
levers — the GEMV dominates wall time — so the payoff is decode-loop tail-latency over long
generations, not steady-state tok/s. Recording them here so nothing gets re-proposed.

### Done (on branches, pending merge to main)
- **Sampler full-vocab scratch reuse** (audit #9), branch `sampler-scratch-reuse`. `sampleChunked`'s
  `e` and `chunkedZ`'s `tmp` allocated a vocab-wide `[]float64` (~1.2-2 MB) every sampled token; now a
  reused `Sampler.expScratch`. Bit-identical (a Sampler draws one token at a time; expChunked overwrites
  every element; `TestChunkedSoftmax_MachineIndependent` + full suite green under `-race`). Measured
  (`BenchmarkSample_*`, qwen 151936): temp-only **1.22 MB → 3.8 KB B/op (−99.7%) AND 525 → 392 µs/op
  (−25%)** — the per-token make+zero+GC was real sampler time, not just jitter; top-p **4.52 → 1.27 MB
  (−72%)** across two commits (chunkedZ scratch, then the candidate `[]indexedProb` via making
  topFilterLogits a Sampler method + `s.ipsBuf`; ns flat — the path is sort-dominated). Remaining
  1.27 MB is `cand` (`[]int` via topKByLogit's retry loop — higher surface, left as follow-up). The
  more-than-jitter temp-only speedup makes this the best of the small wins. Temperature paths only
  (greedy/argmax unaffected).
- **Metal Gemma final-logit softcap parallel-for** (audit #3), branch `metal-softcap-parallel`.
  `finalizeLogits` applied `sc·tanh(x/sc)` with a serial O(vocab) float64-tanh loop every sampling
  token; now fanned across GOMAXPROCS (bit-identical — element-independent, disjoint writes; gated by
  `TestMetalSoftcapParallel_bitIdentical` + snapshot golden). Measured `BenchmarkSoftcap_gemmaVocab`
  (262144): **3.40 → 0.86 ms/op (~3.96×, −2.54 ms per SAMPLING token on gemma-4)**. Greedy unaffected
  (skips the softcap — monotonic). gemma-3 removed the softcap (no-op there). The CUDA twin
  (`cuda/resident.go`, same host parallel-for) is a box follow-up — identical pattern, needs Linux to
  build/measure.
- **WebGPU logits host-buffer reuse.** `gpu.DecodeRunner.Run` did `make([]float32, vocab)` every token;
  now returns a reused `r.logitsHost` (mirrors CUDA pinned scratch / Metal `logitsHost`). Measured
  before/after (qwen3-class vocab 151936, `TestZZ_decodeRunAllocs`, `GOINFER_GPU_ALLOC_BENCH=1`):
  **615589 → 1182 B/op, 61 → 60 allocs/op**, wall time unchanged (garbage −99.8%, matches vocab·4).
  Bit-identical (parity tests green). `gpu/` is outside the parity-manifest freeze.

### Refuted — do not re-propose (measured negative)
- **Wiring Metal `ForwardArgmax` into `decoder.ResidentGreedy`** (the greedy on-device-argmax fast path
  CUDA takes). Built, bit-identical, gated — but a **~2-3% REGRESSION** on Metal (A/B on
  qwen2.5-coder-1.5b int4, B/A 0.977/0.984/0.971 @128/512/2048). Unified memory has **no PCIe D2H** to
  eliminate (unlike CUDA), so the only saving is a cheap host argmax scan, while the naive wiring drops
  the encode-ahead pipeline (~3%) and adds a dispatch on the dispatch-count-bound path (§A2-Metal). A
  *pipelined* on-device-argmax executor variant could recover ~1-2%, not worth it vs the megakernel.
  General rule: a win justified by one backend's bottleneck must be re-measured on the other (CUDA =
  PCIe/bandwidth, Metal = dispatch-count + unified memory). **NB:** an independent code-grounded audit
  (Cursor, 2026-08-10) listed this as its #1 "easy win / first cut" on the CUDA-analogy — measurement
  refutes it on Metal. Do not re-open without the pipelined-executor variant AND a fresh A/B.

### Freeze-blocked — deferred to the v1.0 unfreeze
These require editing parity-manifest "core"/hashed `decoder/` files (`model.go`, `forwardn.go`,
`attention.go`, `mlp.go`, `weightmat.go`), which re-stales every family's `deps_hash`. Held under the
pre-1.0 core-numerics freeze; **batch them when v1.0 lands so the manifest re-validates once.** All are
bit-identity-preserving (pure buffer/traffic reuse).
- **KV re-gather / V re-transpose every token (the big one).** `decoder/forwardn.go:378` (`attendBatchedHeads`)
  re-gathers the whole K history and re-transposes all of V into scratch each decode token (~2-3× the
  intrinsic KV traffic; ~10-15% of per-token traffic at 4k+ ctx, all mainstream CPU families). Needs a
  row-pitch arg on aikit `MatmulBTAcc64` + a per-layer persistent transposed-V cache layout. Highest
  lift, biggest broad CPU win — the headline unfreeze item.
- **embedResident host-scratch reuse.** `embedResident` (`decoder/residency.go:677`, itself freeze-safe) does
  `make([]float32, HiddenDim)` per token, then H2D. The decode-hot-path call sites (`decoder/model.go:979/977`)
  are frozen — can't reroute; and reusing in place breaks the batch caller `decoder/model.go:831`
  (`embs[i]=embedResident(id)` collection would alias). Small (~6-14 KB/token). Bigger follow-on: an
  **on-device embed table** (GPU looks the row up from the id — Metal's `loadEmbedRow` already does).
- **MoE `moeMLP` allocates MB/token.** `decoder/mlp.go:82` skips the `*decodeScratch` invariant the dense
  `gatedMLP` honors → ~7-8 MB garbage/token (Mixtral-class). Thread `*decodeScratch` through.
- **int4 W4A8 `Workspace` alloc/token.** `decoder/weightmat.go:202` int4 branch of `matmul()` uses a cap-0
  `linalg.Workspace` (the W8A8 sibling was fixed, int4 missed) → make() per projection per token. Add an
  int4 case to `matmulInto` with a persistent per-stream Workspace.

### Measured negative — not the win the audit claimed
- **PGO (profile-guided optimization) on the pure-Go CPU path.** Prototyped end-to-end (2026-08-11):
  collected a combined decode+prefill CPU profile on qwen2.5-coder-0.5b, rebuilt with it. **Net-zero
  throughput** — decode 70.5→69.9 tok/s, prefill 88-90→88-90 tok/s, both within noise. The profile
  shows why: 62.8% flat is `aikit.dotF32Acc64 (inline)` — already inlined — and the rest is
  hand-written assembly (dot/GEMV kernels), so PGO's inlining lever has no headroom. **AND
  bit-identical numerics**: a PGO vs non-PGO dump on the same arm64 box differs in 0/151,936 logits
  (PGO *was* applied — build info + binaries differ — but its codegen changes never reached the
  fusion-bearing hot loop). So the FMA-fusion-shift risk did not materialize either. Same root cause
  on both axes: the hot path's compilation is already at a fixed point. Not adopted; no `default.pgo`.
- **Metal 2× rope dispatch → one grid-merged dispatch (audit #4).** Built on branch `metal-rope-merge`
  (bit-identical: snapshot golden byte-exact, LayerB ropeParity green, `TestGeometryPortIsLive` seam
  live on the merged `uQKtotal`). **MEASURED NET-ZERO on decode** (`TestZZ_metalDepthBench`,
  qwen2.5-coder-1.5b W4A8, M1 Pro): 62.4/49.7/28.1/18.3 → 62.3/49.7/27.6/18.3 tok/s @ 128/512/2048/4000,
  flat. Removing ONE rope dispatch of ~17/layer (~6% of ~476 dispatches/token) is lost in noise.
  **Lesson:** "dispatch-count-bound" means the SUM — incremental single-dispatch removal (this, and by
  extension fuse-rope+kv_store alone) will NOT move Metal decode. Only the **megakernel** (collapse most
  dispatches) or **int4-unpack bandwidth** will. Correct + harmless; do not merge expecting a speedup.
- **aikit `q8Span` scalar→SIMD int8→f32 widen (audit #2).** Predicted "several ms/token" on large-vocab
  int4/int8 CPU decode; built on branch `aikit:q8span-simd-widen` (bit-identical, `dequantRowInt8` scale=1.0,
  gates green). **MEASURED NET-ZERO on M1 Pro arm64** (`BenchmarkQ8LMHeadDecode_fused`, M=1 K=1536 N=151936:
  6.79-6.85 → 6.83-6.87 ms/op, flat). The widen isn't the bottleneck at the LM-head shape — it overlaps the
  233 MB int8 head read (memory latency / dotF32 / parallel overhead dominate). **amd64 (AVX2 widen, different
  memory subsystem) UNMEASURED — the box should A/B before this is reconsidered for release.** Do not release
  for perf on the arm64 evidence alone.

### Outside the freeze — fundable now
- **CUDA Gemma softcap parallel-for** — the box twin of the Metal item above (Done). Same host
  parallel-for on `cuda/resident.go`'s serial tanh loop; bit-identical; needs Linux to build/measure.
- **CUDA g4x2 accumulator clear: H2D per MoE layer per token** (Cursor audit, verified). `cudaResident`
  clears the `g4x2` expert accumulator by uploading host zeros (`g4zero`, "no D2D helper" —
  cuda/resident.go:353,1182) every MoE layer. An on-stream memset/zero kernel removes an H2D (and its
  implicit null-stream sync) per MoE layer per token. cuda/ not frozen; bit-identical (a zero is a zero).

### Medium / larger — verify + measure before funding
- **MoE expert-cache host round-trip.** `loadRoutedExperts` (cuda/resident.go:596) does Sync → D2H routing
  indices → H2D expert misses; the Metal paged path is worse (submit/wait per layer, `metal/gemma4_moe.go`).
  A device-side gather or async overlap matters whenever experts are paged — see the standing verdict that
  synchronous MoE paging is dead and *speculative prefetch* is the path (memory: Metal MoE paging needs
  speculation). cuda/metal not frozen.
- **Parallel top-k expert GEMVs.** `moeMLPPost` (cuda/resident.go:1079) runs the selected experts
  sequentially. Concurrent launches need separate per-expert scratch + an ORDERED combine, or the FMA
  association changes and the bit-identity gate fails. Real but bit-identity-delicate; measure the win
  against the added scratch VRAM.
- **WebGPU on-device argmax.** Same shape as the Metal item above — today `Run` always `MapAsync`es full
  logits. CAVEAT from the Metal result: on unified memory this is a wash-to-negative (no D2H to remove);
  it only pays where the readback crosses PCIe (discrete-GPU WebGPU). Measure per target before funding.
- **Graph/fuse the rope→attention gap (CUDA).** Live launches still fire every layer after graph segA.
- **Constrained `Masker.Process` O(V).** Full-vocab mask each constrained-decode token, and setting a
  masker disables the greedy fast path. Scoped to grammar/JSON-constrained requests.
- **Metal batched prefill.** Default-off (f16-MMA vs int8-decode divergence, §A2-Metal) — a TTFT lever
  only if you accept a separate, non-bit-identical lane.

*Corroboration:* the fan-out audit and an independent code-grounded pass (Cursor, 2026-08-10) agreed on
the WebGPU logits reuse, the Gemma softcap, the embedResident scratch, and the Metal-argmax gap
(measurement then refuted the last as a Metal win). Cursor additionally surfaced the g4x2 H2D clear, the
rope+kv_store fuse, the MoE expert round-trip, and the parallel-expert-GEMV items above.

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

| | goinfer | Ollama v0.32.5 | verdict |
|---|---|---|---|
| decode, short context | ~200 tok/s | ~187 tok/s | **parity** (+7%) |
| decode, 2048 context | ~97 → ~133 (A1) → **~160 tok/s** (split-KV, default ≥256 ctx) | ~188 tok/s | 1.94× → **1.17× behind** |
| prefill | 0.66 ms/tok | 0.14 ms/tok | **4.7× behind** |
| total-request crossover | — | — | ~50 prompt tokens |

**Peer-independent claims that stand regardless:** CGO_ENABLED=0 with driver-only linkage,
portable, bit-identical decode with a goldens-gated parity discipline, and 2048-token TTFT
improved 13.1 s → 2.1 s by our own prior work.

**The honest one-line position:** at parity with llama.cpp on short-context decode; behind
at real context lengths and on prefill.

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
  real weights (the batched-prefill-vs-decode split, `docs/task-batched-prefill-bitidentity.md`). The
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

Gates that held: decode byte-identical; parity manifest green; `TestE2EDecode` / `TestRealE2EDecode`.

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
wall as the whole decode path (`docs/task-metal-cgofree-spike.md`: megakernel closed, dispatch-count the
ceiling). **A1-Metal (half4 coalescing, 1.37–1.40×) remains the one capturable decode-attention win on
this box.** Stop proposing dedup layouts; the lever is elsewhere (KV-quant §A3, or accept the floor).

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

**Does aikit need the same enforcement? Mostly no — one real gap.** aikit already makes the strict choice
where parity matters (its ViT uses `CompileLibraryPrecise`) and correctly offers both compile paths, so
no default change is warranted upstream. But `CompileLibraryPrecise` sets `setFastMathEnabled:NO` and
**never reads it back**, while the *same function* asserts `languageVersion` against exactly this landmine
class — and `setFastMathEnabled:` is the API Apple is deprecating (→ `MTLMathMode`), so a future macOS can
silently no-op it and turn "precise" back into fast-math with aikit's own ViT parity gate none the wiser.
**High-leverage aikit fix: read the fast-math option back after set and fail loudly if it didn't take**
(mirror the languageVersion assertion; ideally migrate to `setMathMode:`). Protects aikit + every
consumer. Lower priority (aikit's call): its Metal ViT gate is tolerance-only and its layernorm reductions
are `tgsz`-width-coupled — the snapshot-golden + width-pin transfer, but precise math already lowers the
drift risk there.

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
   anyone who wants OS-robust bits at that cost. If ever adopted, enforce with a read-back assertion (see
   the aikit note below) and re-baseline the snapshot golden.
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

### A3. KV cache quantization — **not bit-identical**

Ollama supports q8/q4 KV caches. Halving or quartering KV bytes directly attacks the term
that scales with context. **Changes decode numerics**, so it would need its own gate and a
parity refresh, and it should be an opt-in mode rather than a default. Listed for
completeness; ranked below A1/A2 because those are free of that cost.

---

## 5. Campaign B — prefill

Deferred behind A. It improves a number already past its usability threshold (2048-token
TTFT is 2.1 s, down from 13.1 s) and cannot reach parity.

### B1. Prefill attention query-tiling — **bit-identical, scoped, unbuilt**

Scoped in `task-prefill-attention.md`. Share a staged K/V tile across a query block to remove
the O(M²) redundant re-reads that coalescing cannot reach.

- **Worth:** attention 820 → perhaps 250–350 ms at 2048; total prefill 2.1 → ~1.6 s ≈ **1.3×**.
- **Cost:** substantial, delicate. The exact float-sum order fixes the blockDim=128 reduction
  tree and thread→key map, forcing Bk=128 key tiles that strain the 64 KB shared budget at
  hd=128/256. Hence the 2-pass-recompute design (no materialised scores).
- **Note:** may inherit machinery from Campaign A, since a tiled/coalesced KV read path is
  the same primitive.

### B2. The GEMV's remaining dp4a headroom — **unattributed**

The GEMV sits at 54% of dp4a peak *before* tensor cores enter the picture. That 46% has not
been attributed since the RN fix. It is not blocked by bit-identity.

**Measure before proposing anything.** This lane has a five-attribution record.

### B3. Tensor cores via per-row scales — see §7

---

## 6. Campaign C — coverage (widens who benefits; makes nothing faster)

### C1. Coverage audit — **DONE (2026-08-04, `TestPrefillCoverageAudit`)**

`PrefillLast` declines: MoE, gemma4-moe, ~~sandwich norms~~ (**lifted**), ~~qk-norm~~ (**lifted**),
K=V, int8, non-uniform, over-cap.

**Result: batched CUDA prefill now covers 7 of 23 validated families** — `llama`, `mistral`,
`phi3`, `qwen2`, `qwen2_5_vl`, **`qwen3`**, and **`gemma3`** (both guards extended 2026-08-04,
below). The other 16 fall back to sequential, by binding guard:

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

---

## 7. The bit-identity fork — the strategic decision

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

Its Phase 0 is the cheap test that could retire most of it: **measure how much per-row scales
alone cost in quality, with no rotation.** If the answer is "almost nothing," the candidate
collapses to a much cheaper change.

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

### D1. Speculative decoding — **GO signal measured (2026-08-04); token-identical; scoped, not built**

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

---

## 9. Landed — do not redo

| lever | result | notes |
|---|---|---|
| Batched prefill (`PrefillLast`) | 2048 TTFT 13.1→2.1s; crossover 128→320 vs 0.5.7 | **bit-identical, default-on** (restored). Was briefly default-off after an 84% divergence from fma-contraction; FIXED by explicit `__fmaf_rn` everywhere + `TestKernelFMALint`; `TestPrefillDivergenceRate` 0/50 (`task-batched-prefill-bitidentity.md`) |
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

---

## 12. Suggested order

1. ~~**Campaign A**~~ **CLOSED** — A0 profile, A1 coalescing (1.34×), A1-reprofile (→occupancy), D4
   (Ollama=flash-attn, confirmed), split-KV decode attention (default-on, bit-identical, +1.20× @2048).
   Long-ctx decode **99.5 → 160 tok/s = 1.61×**, gap to Ollama **1.94× → 1.17×**. Stopped at the
   bit-identity ceiling (V-sum ILP unroll refuted); full parity needs the §7 non-bit-identical fork,
   which A does not take. The tiling primitive is available to share with B1.
2. ~~**C1** — coverage audit~~ **DONE** (5/23 dense; the release must say "dense lane," not
   "prefill").
3. **D1** — speculative decoding: **GO signal measured** (verify ceiling 2.5–3.6× at k=4–8), scoped
   (~1.5–2 days: batched all-position verify + n-gram drafter + loop + lossless gate). The largest
   decode lever, token-identical — **the next decode campaign to build.**
4. **D5** — scope the hybrid GPU/CPU layer split (ranked *alongside* D1). The right shape for the
   26B-on-8GB case; gated on the CPU-kernel decision (`plan-cpubrrr-…`), above §5/prefill.
5. **D4** — confirm Ollama's long-context decode mechanism. Hours, sharpens A.
6. **B1** — prefill attention query-tiling, if A's primitive transfers.
7. **§7 fork** — only when B2 has been attributed and Phase 0 of the rotation doc has been
   run.

Everything in §6 (C2–C5) is coverage work whose priority depends on C1's answer (now known: 5/23).

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
| decode, 2048 context | ~97 tok/s | ~188 tok/s | **1.94× behind** |
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

### 3a. Long-context decode — 1.94× behind

Decode costs ~**0.0028 ms per KV position** on a ~4.1 ms base (measured: 221 → 179 → 100 →
66 tok/s at 128 / 512 / 2048 / 3900 context).

For this model the KV read at a given position is roughly **28 KB across all layers**, which
is about **0.00006 ms** at the card's bandwidth. That is **~40× off the memory bound** on the
term that scales with context.

It was previously read as "dead-linear O(context), correct behaviour, not a cliff." Correct
in isolation, refuted comparatively: Ollama holds ~188 tok/s at 2048 where we fall to ~97.

**The decode attention kernel has never been profiled.**

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

### A0. Profile the decode attention kernel — **prerequisite, unmeasured**

`ncu` at 128 / 512 / 2048 context: Scheduler Statistics, Warp State Statistics, Speed of
Light, sector utilisation, occupancy.

**Do not carry the prefill attention diagnosis across.** That kernel was L1TEX-saturated at
21.96% bytes/sector; the GEMV, superficially similar, was latency-bound with efficient
loads. This is a third kernel and gets its own profile.

### A1. Fix, against whatever the profile names

Bit-identity is a design constraint: decode stays byte-identical, existing reduction order
preserved exactly, no online rescaling, no tolerance gate.

Gates: decode byte-identical; parity manifest green; a context beyond the sliding window; a
context that is not a multiple of any tile size.

### A2. KV cache layout — *unmeasured, plausible*

If the profile says uncoalesced KV reads, the fix may be a layout change rather than a kernel
change. Layout changes are bit-identical by construction (they move where operands live, not
the order they accumulate) — the same reason `int2` coalescing and A-staging were safe.

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

### C1. Coverage audit — **cheap, unmeasured, gates the release narrative**

`PrefillLast` declines: MoE, gemma4-moe, sandwich norms, qk-norm, K=V, int8, non-uniform,
over-cap.

Enumerate every family in the parity manifest against those guards and report **gets batched
prefill / falls back to sequential, and which guard fires.** If coverage is narrow, extending
the guard is worth more than any remaining kernel work — and it must be known before a
release announces a prefill improvement.

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

### D1. Speculative decoding — **potentially large, token-identical**

A draft model proposes k tokens; the target model verifies them in one batched forward.
Under greedy sampling the verification step accepts only tokens the target model would have
produced anyway, so **the emitted token stream is identical by construction** — it is
compatible with the determinism contract in a way KV quantisation is not.

- **Worth:** commonly 2–3× on decode; the win scales with draft acceptance rate.
- **Why it fits here specifically:** verification is a **batched M=k forward** — exactly the
  path the prefill campaign just built and optimised. The machinery largely exists.
- **Cost:** a draft model per target, acceptance-rate tuning, and a correctness gate proving
  the accepted stream equals the sequential stream.
- **Status:** entirely unmeasured. Should be scoped before anything in §5.

### D2. Multi-token prediction / Medusa-style heads

Same family as D1, without a separate draft model. Requires model support; most checkpoints
do not have the heads.

### D3. Continuous batching / server-side concurrency

Improves throughput under concurrent load, not single-stream latency. Different axis from
everything else in this doc, and not what the current comparison measures — but it is what a
serving deployment actually cares about, and it is unexamined.

### D4. Confirm what Ollama actually does at long-context decode

We infer flash-attention-style KV handling from behaviour (it holds ~188 tok/s at 2048). It
would be cheap to confirm, and knowing the mechanism would sharpen Campaign A's target.

---

## 9. Landed — do not redo

| lever | result | notes |
|---|---|---|
| Batched prefill (`PrefillLast`) | crossover 128 → 320 vs 0.5.7 | mixed-M, bit-identical |
| KV-only prefill for `prompt[:-1]` | −4.39 ms/prompt-token on 26B | skips LM head |
| GEMV `MT=32` | ~6% | tile width, not an accumulation constraint |
| GEMV `int2` coalescing | 13% | bytes/sector 49.99 → 98.01% |
| GEMV `RN=2` register blocking | ~30% | scoreboard stall 17.8 → 7.5 cyc |
| Prefill attention `float4` | **3.1×**; 2048 TTFT 3.33 → 6.17× | bytes/sector 21.96 → 66.32% |

Cumulative on the GEMV ≈ **1.5×**. Total 2048 TTFT: **13.1 s → 2.1 s**.

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

1. **Campaign A** — profile decode attention, then fix. The only reachable parity.
2. **C1** — coverage audit. Cheap, and it gates what a release can honestly claim.
3. **D1** — scope speculative decoding. Potentially the largest decode lever in the doc,
   token-identical, and it reuses the batched forward already built.
4. **D4** — confirm Ollama's long-context decode mechanism. Hours, sharpens A.
5. **B1** — prefill attention query-tiling, if A's primitive transfers.
6. **§7 fork** — only when B2 has been attributed and Phase 0 of the rotation doc has been
   run.

Everything in §6 (C2–C5) is coverage work whose priority depends on C1's answer.

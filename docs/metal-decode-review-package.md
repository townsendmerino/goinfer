# Metal cgo-free decode — review package (for Claude app)

> A complete, honest accounting of the native-Metal GPU decode work in `goinfer`, written for
> a fresh expert reviewer. **Primary ask: sanity-check the bottleneck diagnosis and find the
> headroom I may have missed.** The diagnosis that "we're at the ceiling" rests on wall-clock
> inference, *not* GPU-counter profiling — that gap is the single biggest risk in this report
> and the first thing to attack. Companion docs in this repo: `task-metal-cgofree-spike.md`
> (running lab notebook), `metal-gemv-optimization-ask.md` + `metal-gemv-fable-response.md`
> (the earlier Fable exchange), `cuda-megakernel-spec.md` (the CUDA analog).

## 0. TL;DR

- **Goal:** a cgo-free (`CGO_ENABLED=0`) native-Metal GPU decode backend for Apple Silicon,
  in pure Go via `purego`/`objc` (dlopen Metal + `objc_msgSend`) — no Metal-cpp, no cgo.
- **Result:** **~70 tok/s** decode on qwen2.5-coder-1.5b (int8→W4A8), **0.98× the ~71 GO bar**,
  2.2× the WebGPU-on-Metal baseline (31.9), **3.5× the untuned spike (~20)**. Correct: 21/24
  argmax vs a CPU reference, logit cosine 0.989. **Now wired into the CLI** — real generation
  works (`--backend metal`).
- **Rig:** MacBook Pro, **Apple M1 Pro, 14-core GPU, 16 GB, ~200 GB/s**, Metal 4, MSL 3.1.
- **Diagnosis (INFERRED from wall-clock):** decode is **latency/occupancy-bound at ~27% of
  memory bandwidth**, not compute-, bandwidth-, or encode-bound. Five levers were tried; four
  are measured dead ends, and the structural one (megakernel) is **architecturally impossible
  on Metal** (no grid-sync).
- **The review ask:** (1) is that diagnosis right? (it's never been confirmed with Instruments
  GPU counters); (2) given 27% bandwidth utilization, what's the realistic ceiling and which
  unexplored lever (§6) is worth pulling; (3) is batch-1 single-stream just ceilinged here?

## 1. The bar and the constraints

- **GO bar ~71 tok/s** = 85% of the measured Ollama-Metal peer on the same machine/model
  (83.3 tok/s at int4/q4_k_m). Chosen as "competitive with the incumbent."
- **WebGPU-on-Metal baseline** (goinfer's existing `gpu/` backend, int8): **31.9 tok/s**.
- **Hard constraints:** `CGO_ENABLED=0` end to end; MSL compiled from source at runtime
  (`newLibraryWithSource:options:` at MSL 3.1); one Metal command buffer per token; Apple UMA
  (buffers are `newBufferWithBytes` shared — no host/device copy).
- **Model:** qwen2.5-coder-1.5b. H=1536, 28 layers, nH=12, nKV=2 (GQA), hd=128, I=8960,
  V=151936, RoPE full, qkv bias, untied LM head. Loaded int8, re-quantized to **W4A8** on
  device (int4 weights, group=32, f32 group scales; int8 per-vector activations).

## 2. Correctness (what "correct" means here)

- **21/24 argmax match** vs the CPU int8 reference decode, teacher-forced over 24 steps;
  last-logit **cosine 0.989**. The 3 mismatches are the **int8→int4 (W4A8) requant modeling
  gap**, not a bug — the GPU path is a lower-precision model than the int8 CPU reference, and
  low-confidence positions diverge. Cosine ~0.99 throughout.
- The **fused on-GPU argmax** (see §4) matches `argmax(full logits)` **24/24 exact** — so the
  argmax fusion is bit-exact; the 3/24 is purely the W4A8-vs-int8 gap.
- All individual kernels (GEMV W8A8/W4A8, rmsnorm+quant, rope, attention, swiglu) were
  validated **bit-exact vs CPU references** during bring-up.

## 3. The optimization arc (chronological, measured)

Each row is a committed step; tok/s is best-of-40 warm steady-state decode.

| # | change | isolated kernel effect | decode tok/s | verdict |
|---|---|---|---|---|
| 0 | untuned real-model decode (correct, cgo-free) | — | ~20 | ties WebGPU — NO-GO as-is |
| 1 | **profiled** → found attention was 68% (one-thread-per-head) | attn 1139→23 µs | 20→**56** | profiling overturned a wrong "GEMV-bound" guess |
| 2 | coalesced W4A8 GEMV (stride-4 → word-per-lane) | gate/up 254→220 µs | 56→**58.8** | |
| 3 | **Stage A** (Fable): `uint4` loads + int8 activation staged once into threadgroup `short` + 8 simdgroups/tg | gate/up 220→164 µs | 58.8→**68.9** | staging killed redundant *activation* cold-DRAM re-reads |
| 4 | **fused block-argmax lm head** (Fable) | — | ~neutral | bit-exact (24/24), UMA-neutral (readback is zero-copy) |
| 5 | **encode-tax trims**: batch buffer binds (setBuffers), skip redundant pipeline set | ~2600→337 msgSends/tok | 68.9→**~70** | *and* the decisive measurement (§5) |
| 6 | **Stage B** (Fable): tile-repack, lane-per-row, broadcast reads | gate/up 164→**118 µs** | **flat** (−1 parity) | reverted — the pivotal negative result |
| 7 | **megakernel** (redundant-recompute fusion, 12→9 dispatches/layer) | — | 70→**52** | reverted — net-negative (§5) |
| 8 | wire into CLI (`--backend metal`) | — | 69.9 via CLI | ships |

Net: **20 → 70 tok/s, 3.5×.**

## 4. Current architecture (what ships)

Package `metal/` (all `//go:build darwin`, its own Go module, ~2.4k LoC, 17 MSL kernels).

- **`metal.go`** — the cgo-free binding: `Device`, `CompileLibrary` (defuses the
  `LC_BUILD_VERSION`/MSL-2.4 landmine — `CGO_ENABLED=0` macOS binaries omit `LC_BUILD_VERSION`
  [golang/go#77917], so Metal silently defaults MSL to 2.4 and strips bfloat; fixed with an
  explicit `MTLCompileOptions` at 3.1 + a read-back version assertion + a bfloat canary
  kernel), `Encoder` (one command buffer/token; batched `setBuffers:offsets:withRange:`;
  skip-redundant-pipeline), `Buffer` (shared UMA, `At()` sub-views, `SetU32`/`U32`).
- **`kernels.go`** — the MSL kernel set. Decode-layer kernels: `rmsnorm_quant`, `quant_vec`,
  `gemv_w4a8_sa`/`_bias`/`_resid` (**Stage A** — the shipping GEMV: simdgroup-per-row, `uint4`
  loads = one 32-elem scale group, threadgroup-staged `short` activation), `gemv_w4a8_sa_amax`
  + `argmax_finish` (fused on-GPU greedy argmax), `rope` (NeoX half-split), `kv_store`,
  `attention` (threadgroup-per-head), `swiglu_quant`, `residual`. The older `_coal` W4A8 family
  is still in the tree and **down-proj still uses it at tg=32** (K=8960 > Stage A's `As[1536]`
  activation-staging cap). The Stage B `gemv_w4a8_t32` kernels + repacker are **NOT in the tree**
  — they were reverted (§5) and live only in git history + `metal-gemv-fable-response.md`.
- **`model.go`** — `Resident`: uploads W4A8 weights once (fused QKV, fused gate/up), runs the
  whole layer stack + LM head in ONE command buffer/token. `Forward(id,pos)→logits`,
  `ForwardEmb(emb,pos)→logits` (the decoder-interface shape), `ForwardArgmax(id,pos)→token`
  (fused greedy). LockOSThread discipline around the NSAutoreleasePool (Go goroutine migration
  vs per-thread pool = SIGSEGV otherwise).
- **`backend.go`** — `metalBackend` registers as `"metal"` (`decoder.RegisterBackend`),
  implements `decoder.Backend` + `ResidencyBackend`; `metalResident` adapts `*Resident` to the
  6-method `decoder.ResidentForward`. Declines gracefully (recover → CPU fallback) on any
  device/kernel/quant failure. Blank-imported from `demo/gemma` under `-tags metal`.

**Run it:** `go run -tags metal ./demo/gemma --backend metal --quant int8int8 --model <gguf>
--prompt "…" --max 128 --temp 0.7`. Verified: coherent output, greedy + temp/top-p both work,
**69.9 tok/s steady-state through the CLI** (two-point measurement cancelling load+prefill).

## 5. Where the time goes + what's RULED OUT (all measured)

**Per-kernel profile** (µs/dispatch, real dims, cache-resident best-of-20):

```
qkv gemv     (2048×K1536)      28.7      rmsnorm_quant (H1536)   11.3
gate/up SA   (17920×K1536)    163.4      swiglu_quant  (I8960)   22.9
down gemv    (1536×K8960)     116.4      attention (nH12,nKeys32) 23.0
o gemv       (1536×K1536)      24.2
lm head gemv (151936×K1536)  1733.0
---- per-token: 28×448µs + lm 1732µs = 14.3 ms/token ≈ 70 tok/s ----
```

**Key facts the diagnosis rests on:**
1. **Cache-resident kernel-sum (14.3 ms) ≈ measured token time (14.3–14.8 ms).** The profile
   reuses ONE weight buffer 200× (hot in L2); the real model streams 28 *distinct* cold weight
   matrices (~772 MB/token). That cold vs hot difference produces **almost no slowdown** →
   **the kernels are not bandwidth-bound** (cache-residency doesn't speed them).
2. **Aggregate: 772 MB/token ÷ 14.3 ms = ~54 GB/s = ~27% of the M1 Pro's ~200 GB/s.** Well
   below saturation → **latency/occupancy-bound** (not enough memory requests in flight), *if*
   the wall-clock reading is trustworthy.
3. A **46 µs/kernel isolated win bought ZERO end-to-end** (Stage B, #6) → decode is **not
   GEMV-compute-bound** in the real cold-weight regime.

**Ruled out, with the measurement that killed each:**

| lever | what it attacks | measured result | why it's dead |
|---|---|---|---|
| **Stage B repack** (lane-per-row, broadcast reads) | GEMV compute efficiency | gate/up 164→118 µs isolated, **decode flat**, −1 parity | not GEMV-compute-bound; the isolated win is a cache-resident artifact |
| **Fused argmax** | 608 KB logit readback | bit-exact, **throughput-neutral** | UMA readback is a zero-copy shared view (~30 µs), not the 0.2–0.4 ms Fable assumed for a discrete GPU |
| **Indirect Command Buffers (ICB)** | per-token encode overhead | setBuffers proxy recovered only **~0.5 ms** of the overhead | ICB removes only Go-side encode; the rest is **GPU-side** per-dispatch latency, which ICB doesn't touch |
| **Megakernel, redundant-recompute** (12→9 dispatches/layer) | dispatch count | **70 → 52 tok/s** | each of 2240 gate/up threadgroups redundantly recomputes the full 1536-elem rmsnorm reduction, which dwarfs its ~8-row GEMV slice |
| **Megakernel, true 1-launch/layer** | dispatch count | not attempted | **impossible on Metal**: needs grid-wide sync (`grid.sync()`/cooperative launch); Metal has neither |

## 6. HEADROOM — what's left (the review ask)

We're at 27% of memory bandwidth on a machine that isn't compute-bound → **in principle there
is 2–3× of bandwidth headroom.** The open question is what's *preventing* saturation and
whether it's reachable. Ranked by my guess at value × tractability; **all are unverified.**

### 6.1 Confirm the limiter with GPU counters (THE prerequisite — do this first)
Everything above is **wall-clock inference**. I never ran a Metal GPU capture / Instruments
"Compute" limiter counters. The whole "latency/occupancy-bound, ceiling reached" conclusion
could be wrong in a way that hides a big lever. **Ask:** run a frame capture on one token and
read the limiter counters (ALU vs LSU vs memory vs occupancy) for `gemv_w4a8_sa` and the lm
head. That single measurement re-grounds §5 and tells us whether 6.2/6.3/6.4 are worth it.
*Confidence the diagnosis holds: medium. Confidence it's worth confirming: high.*

### 6.2 Raise memory-level parallelism / occupancy (if LSU-latency-bound)
If we're latency-bound at 27% bandwidth, the fix is **more in-flight loads**: software-prefetch
the next `uint4` group while accumulating the current one; more simdgroups resident per core;
or restructure so each thread has more independent outstanding loads. This attacks the actual
suspected bottleneck (bandwidth *utilization*), unlike Stage A/B which only improved per-request
efficiency. *Uncertain; gated on 6.1.*

### 6.3 lm head as its own big kernel — Stage B at tg=128
The lm head is **1733 µs in ONE dispatch = 12% of the token** — and unlike the tiny per-layer
GEMVs, its cost is real single-kernel work, not launch overhead. It reads 116 MB int4 (floor
~583 µs at 200 GB/s → we're at ~3×). I kept it on Stage A because N=151936 isn't divisible by
256 — **but it IS divisible by 128**, so a Stage-B (broadcast-read) lm head at tg=128 is
buildable and could plausibly shave ~0.4–0.8 ms → ~73–75 tok/s. Cleanest *single* unexplored
kernel win. *Confidence: medium — but note Stage B's per-layer isolated win didn't translate;
the lm head may behave differently precisely because it's a big single kernel.*

### 6.4 Concurrent dispatch encoder
Metal's default compute encoder is **serially hazard-tracked** — it inserts a pipeline-flushing
barrier between dispatches with a read/write hazard, which is *most* of ours, but not all
(e.g. rope-q ∥ rope-k are independent). `MTLDispatchTypeConcurrent` + hand-placed barriers only
where a true dependency exists could overlap independent work and keep memory busier across the
337-dispatch chain — directly attacking the inter-dispatch idle that likely explains the 27%.
*Confidence: low-medium; needs 6.1 to know if inter-dispatch idle is real.*

### 6.5 f16 group scales
The W4A8 group scales are ~20% of both hot GEMVs' bytes. Halving them (f32→f16) is a pure
bandwidth trim — worth ~10% of traffic *if* bandwidth-utilization-bound. Parity-gated (check
whether the int8→int4 requant scales are already f16-exact — then it's free). *Low, gated.*

### 6.6 rope + kv-store fusion (the only zero-redundant-work dispatch trim)
rope-q, rope-k, kv-store are pure element-wise (no reduction) → fusing them adds **no**
redundant work (unlike 6-of-§5's megakernel), saving ~2 dispatches/layer = 56/token. If any of
the ~2.4 ms is genuinely dispatch-launch overhead, this is the safe way to claw a little back
(~0.3–0.4 ms → ~72 tok/s). Fiddly (three different thread counts). *Low but safe.*

### 6.7 The honest "maybe it's just ceilinged" option
Batch-1 single-stream decode may simply be at its architectural ceiling on this chip without a
grid-sync primitive. **The big structural wins (Stage B lane-per-row, megakernel-shape fusion)
all pay off in prefill/batch**, where each threadgroup's GEMV slice is fat enough to amortize
fused reductions and saturate bandwidth. If the reviewer agrees, the higher-value pivot is
**prefill/prompt-processing throughput** (and multi-sequence batching), not squeezing the last
~1 tok/s out of batch-1 decode. The Stage B kernels + repacker are already committed for this.

## 7. Specific questions for the reviewer

1. **Is the "latency/occupancy-bound at 27% bandwidth, not compute/bandwidth-bound" diagnosis
   correct?** What GPU-counter signature would confirm or refute it, and does the
   cache-hot-sum ≈ cold-real-time coincidence (§5.1) actually imply "not bandwidth-bound," or
   is there a subtler reading?
2. **At 27% bandwidth utilization, what's a realistic reachable ceiling** for batch-1 decode on
   a 14-core M1 Pro — 1.5×? 2×? Or is 27% just what serial dependent GEMVs get and the rest is
   unrecoverable without batching?
3. **Which of §6.2–6.6 would you pull first**, and is there a lever I've missed entirely
   (e.g. persistent-threads within a single core, `simdgroup_matrix` despite the int-accumulate
   parity concern, a smarter KV/attention layout, threadgroup-memory weight staging)?
4. **Is §6.7 the right call** — declare batch-1 decode done at ~70 and move the effort to
   prefill/batch, where the shelved Stage B/megakernel structure actually pays?

## 8. Appendix — commits (this arc, newest first)

```
a0704ae feat(metal): wire Metal decode backend into the generation loop (--backend metal)
abf40b4 docs(metal-spike): megakernel tested + CLOSED on Metal (no grid-sync; redundant-recompute net-negative)
54112a7 docs(metal-spike): rule out ICB (overhead is GPU-side), final verdict ~70 tok/s at bar
643541e perf(metal-spike): batch buffer binds + skip redundant pipeline set → ~70 tok/s
f6a9e72 feat(metal-spike): Fable fused block-argmax lm head — bit-exact, UMA-neutral
70dbe5a perf(metal-spike): Fable Stage A GEMV (uint4 + staged short activation) → 68.9 tok/s
97fa3cc perf(metal-spike): threadgroup-per-head attn + coalesced W4A8 → 58.8 tok/s
30392b9 spike(metal): real-model decode — CORRECT + cgo-free, but untuned perf ties WebGPU
77c4c31 spike(metal): full decode layer assembled in ONE command buffer — bit-exact
236eb4f spike(metal): Layer B — full decode-layer kernel set ported, all bit-exact
9deaaa9 spike(metal): Layer A COMPLETE — cgo-free compute dispatch, correct result
3cd9807 spike(metal): Layer A Phase 1 — reach Metal + MSL compiler cgo-free, landmine defused
0a3541e spike(metal): Step 0 complete — Ollama-Metal peer = 83.3 tok/s, GO bar ~71
```
(plus the intervening docs/measurement commits; full detail in `task-metal-cgofree-spike.md`.)

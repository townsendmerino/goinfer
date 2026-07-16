# Metal batch-1 decode — headroom deep-dive (Fable pass, companion to the review package)

> Companion to `metal-decode-review-package.md`. A deep/creative pass that **corrected the
> review's central number** and re-ranked the levers. Bottom line up front:
>
> **The "27% of 200 GB/s ⇒ 2–3× headroom" framing is wrong twice, and both matter:**
> 1. **Scale traffic was uncounted.** 772 MB/token is int4 nibbles only; the W4A8 layout
>    (`metal/pack.go`) also streams **f32 group scales ≈ 193 MB/token** (one f32 per 32
>    weights = +25%). True traffic ≈ **965 MB/token** → effective **67.5 GB/s = 34% of
>    nominal**, not 27%. Corollary: **f16 scales is a ~10% whole-token traffic cut, mis-
>    ranked "low" in the review — it should be lever #1.**
> 2. **200 GB/s is the SoC fabric peak, not GPU-reachable.** GPU-alone streaming on M1 Pro
>    is ~150–170 GB/s; and the repo's own peer, **Ollama at 83.3 tok/s on this rig**, streams
>    ~0.9–1.0 GB/token = **~75–83 GB/s effective**. Nobody demonstrably exceeds ~85 GB/s at
>    batch-1 on this rig for this model class. goinfer at 67.5 is at **~82–90% of the
>    incumbent's rate — the honest gap is ~1.2–1.5×, not 2–3×.**
>
> **Root cause is issue-bound (ALU/LSU ops per byte), not latency / occupancy / starvation.**
> Apple has no int8-dot/DP4A, so the nibble-unpack costs ~10–12 issue slots/weight. Everything
> is unverified-pending a GPU-counter capture (a cgo-free Step 0 exists — §Decision tree).

## 1. The re-derived budget (repo numbers, scales included)

Per-kernel effective bandwidth, hot profile, **bytes = nibbles + f32 scales**:

| kernel | shape | bytes | µs | GB/s | note |
|---|---|---|---|---|---|
| qkv (`_sa_bias`, tg=256) | 2048×1536 | 1.97 MB | 28.7 | 68 | small-N, ramp-dominated |
| gate/up (`_sa`, tg=256) | 17920×1536 | 17.2 MB | 163.4 | 105 | fits ~24 MB SLC when hot |
| down (`_resid` coal, **tg=32**) | 1536×8960 | 8.6 MB | 116.4 | 74 | K=8960 > Stage A `As[1536]` cap |
| o (`_sa_resid`) | 1536×1536 | 1.47 MB | 24.2 | 61 | small-N |
| lm head (`_sa`, tg=256) | 151936×1536 | 145.9 MB | 1733 | **84** | 146 MB ≫ SLC ⇒ this IS the cold number |
| Stage B gate/up (reverted) | same | 17.2 MB | 118 | 146 | SLC-assisted; proves ALU side can drop 40% |

**Token: GEMVs 11.05 ms (77%) at avg 87 GB/s; serial small-op + launch-floor chain ~3.2 ms
(23%); host bubble ~0.3–0.7 ms (3%).** Per-dispatch GPU floor ~3.8 µs (`tax_test.go`) × 337 ≈
1.3 ms/token irreducible at the current dispatch count.

## 2. Root cause — disentangled

- **Issue-bound (dominant), not latency/occupancy.** The lm head is the truth-teller: 19,000
  threadgroups (no parallelism/occupancy shortage, cold by construction) and still only **84
  GB/s**. Latency-bound would have ample outstanding requests; it doesn't help. What fits: no
  DP4A on Apple → ~10–12 issue slots/weight (extract+sub-8+widen+int-mad + one tgmem load per
  weight for the staged activation). ALU-issue ceiling ≈ 120–145 GB/s, right on the observed
  84–105. Stage B going 164→118 µs at identical bytes is the issue-bound signature.
- **Occupancy: not Stage A's staging** (3 KB tgmem → not the cap; check `maxTotalThreadsPer­
  Threadgroup` per PSO for register pressure), **but yes for down-proj at tg=32** (`model.go:264`,
  K=8960 > As cap → strands core thread slots → its 74 GB/s) and the 1–12 threadgroup micro-ops.
- **CPU-encode starvation: impossible within a token** — all 337 dispatches are pre-encoded
  into ONE command buffer before commit, so the GPU never waits on `objc_msgSend` mid-token.
  This vindicates the ICB dead-end (it correctly couldn't touch the GPU-side per-dispatch tax).
  **But** the token *boundary* bubbles ~0.3–0.7 ms (`wait(t) → embed → encode(t+1) → commit`);
  fixed by **encode-ahead**, not ICB.
- **Measurement artifacts:** the two in the header, plus the hot≈cold inference (§5.1 of the
  review) is low-resolution (both sides embed the launch floor; ±1 ms slop hides a 20% cold
  GEMV regression). Doc nit: the review's "Stage B t32 present but off-path" — those kernels
  are **not in the tree**, only git history + `metal-gemv-fable-response.md`.

## Decision tree (cgo-free Step 0 first)

- **Step 0 (~1 hr, no Instruments):** log `GPUStartTime/GPUEndTime/kernelStartTime/End` per
  command buffer (four msgSel, purego-objc; verify the float-return path once).
  `wall − (GPUEnd−GPUStart)` = the host bubble, per token, exactly. >1 ms → do encode-ahead first.
- **Step 1 — `xctrace record --template 'Metal System Trace'`:** inter-dispatch gaps + low-
  occupancy stretches >2 ms ⇒ chain share real (→ fusion/parallel small ops); dense timeline ⇒
  all value in-kernel (→ Stage C).
- **Step 2 — GPU Counters on gate/up + lm head:** ALU-Limiter high + BW 90–110 ⇒ **issue-bound
  (predicted)** → Stage C pays, 90+ live. Buffer-Read-Limiter high + BW ≥130 flat across
  variants ⇒ bandwidth-bound → only the traffic diet pays, ceiling ~85–90, pivot sooner.
  Neither + occupancy <40% ⇒ residency (drop staging, retune tg). Neither + occ high + BW low ⇒
  true latency-bound → MLP levers.
- **Step 3 — per-dispatch cold timestamps:** cold ≈ hot ⇒ SLC story dead, budget stands; cold ≫
  hot ⇒ real GEMV bandwidth worse than modeled, f16 scales + Stage C gain value.

## Levers (re-ranked; all unverified-pending-counters; each gated argmax 21/24 + fused 24/24 + best-of-40)

- **L1 — f16 group scales (do FIRST; review had it last).** −96.5 MB/token (10%) → at 87 GB/s
  −1.1 ms → **~75–76 tok/s (+5–6)**, and halves scale-load issue slots. Parity: f16 scale ≈
  2⁻¹¹ relative, ≪ W4 noise; gate anyway. Hours. Bonus: interleave scale words with the weight
  tile → one linear stream.
- **L2 — encode-ahead (token-boundary bubble).** Encode t+1 while t runs (bindings static;
  double-buffer `uPos/uNKeys`); after `wait(t)`, fill `r.x` (~2 µs), commit. **+2–4 tok/s.**
  Also try `commandBufferWithUnretainedReferences` (one selector; shaves driver retain/release).
- **L3 — fuse rope-q + rope-k + kv_store into the attention prologue.** 12→9 dispatches/layer,
  −84/token ≈ −0.5–0.8 ms → **~80–83 cumulative.** Strictly better than the review's separate
  fusion; low parity risk (one writer per kv-head).
- **L4 — lm head at tg=128, Stage-B/C shape.** 151936 = 128×1187 ✓. **Doubles as the Stage-C
  gate**: the lm head is cold even isolated (146 MB ≫ SLC), so an isolated win here is a genuine
  cold-stream win (unlike Stage B gate/up's SLC mirage). Issue-bound → 1733 → ~1250–1450 µs
  (**+1.5–2.5**); no improvement ⇒ DRAM-behavior ceiling is real and L5 is dead → pivot.
- **L5 — "Stage C" GEMV (gated hard on Step 2 / L4).** lane-per-row × tiled coalesced weights ×
  **uniform** activations (L1 broadcast, no tgmem gather, no barriers) × folded −8·S_g zero-point
  × 2-deep pipelined independent loads (the MLP depth, since MSL has no prefetch instruction).
  Replaces Stage A *and* `_coal` (drops the K≤1536 As cap → down-proj leaves tg=32). All N ÷128.
  Precompute per-group S_g in the rms/quant/swiglu epilogues (they already sweep the vector).

```metal
// tg=128: 4 simdgroups; simdgroup = 32 output rows; lane owns row (blk*32+lane).
// Unroll-by-2: two independent 16B weight loads + two f16 scale loads in flight,
// two accumulators to split the FMA chain. Activations UNIFORM across the simdgroup
// (same g for all lanes): uint4 loads served by L1 broadcast — no per-lane gather.
float acc0 = 0.f, acc1 = 0.f;
for (uint g = 0; g < G; g += 2) {
    uint4 w0 = wp[(g+0)*32];               // lane-coalesced 512B/simdgroup
    uint4 w1 = wp[(g+1)*32];               // independent: 2nd load in flight
    float s0 = float(sp[(g+0)*32]);        // f16 scale, tiled
    float s1 = float(sp[(g+1)*32]);
    device const uint4* a4 = (device const uint4*)(aq + (g<<5));
    uint4 a0 = a4[0], a1 = a4[1];          // uniform: activations group g
    uint4 a2 = a4[2], a3 = a4[3];          //          activations group g+1
    int d0 = DOT32(w0, a0, a1);            // Σ nib*a, nib in [0,15]  (no -8)
    int d1 = DOT32(w1, a2, a3);
    acc0 = fma(float(d0 - 8*Sg[g  ]), s0, acc0);   // zero-point folded once/group
    acc1 = fma(float(d1 - 8*Sg[g+1]), s1, acc1);
}
out[row] = (acc0 + acc1) * asc[0];         // coalesced 128B store per simdgroup
```

  Issue cost ~10–12 → ~3 slots/weight → ALU ceiling ~300+ GB/s (no longer the limiter); MLP =
  2–4 outstanding independent streams × 128 lanes × several tg/core. **If issue-bound confirmed:
  GEMV 9.3 → 7.4–8.2 ms → ~88–97 tok/s.** Small-N (o, down): optional split-K×2 w/ f32
  `atomic_fetch_add` (flag: order-nondeterminism vs the bit-exact gate — keep optional). Days.
- **L6 — chain shrinkers, counter-gated:** static (calibrated) A8 activation scales → deletes
  the amax reductions so rms/swiglu become multi-tg elementwise (−0.4–0.8 ms; medium parity
  risk); hazard-untracked `MTLHeap` + `MTLDispatchTypeConcurrent` with hand-placed barriers (the
  only way to shave the 3.8 µs/dispatch drain; Metal 4 makes this the forward-looking design);
  one execution-order weight arena (single buffer → one linear DRAM stream). **The review's
  §6.4 (overlap qkv with gate/up) is near-worthless — the layer graph is a chain**; the only
  independent dispatches are the rope/kv micro-ops (→ L3). ResidencySets/useResource: N/A.
  `simdgroup_matrix`: N/A at M=1 (prefill lever). KV f16: irrelevant at bench ctx, mandatory
  before long-context (KV read ~118 MB/token at 4096).

## Meta-verdict + the pivot

- **Realistic ceiling ~80–85 GB/s effective = ~92–95 tok/s at the f16-scale byte count, ~83 at
  today's bytes.** 2–3× was never on the table; ~1.15× near-certain, ~1.3× the stretch.
- **Near-certain stack (L1+L2+L3+L4) → ~80–86 tok/s.** Crossing 90 needs L5 to beat the
  incumbent's cold rate — plausible (Stage B's 146 GB/s hot; the issue arithmetic) but
  undemonstrated by anyone on this rig. If Step 2 reads Buffer-Read-limited, ceiling ~85–90, L5
  dead.
- **The higher-value pivot is real, and it is also the decode accelerator.** Build the
  prefill/batch-k forward (where Stage-B / `simdgroup_matrix` / fat-slice shapes genuinely pay),
  then point it back at decode as **self-speculative / n-gram-draft decoding**: verifying k
  drafted tokens per weight pass amortizes the stream k× — the *only* mechanism on this hardware
  that converts the unused bandwidth into single-stream tokens without grid-sync. For a coder
  model (repetitive completions), **prompt-lookup drafting plausibly yields 1.5–2.5× effective
  tok/s — more than everything in the lever list combined.** goinfer already ships the drafter
  (`--spec ngram`); the missing piece is the Metal batch-k verify forward.
- **Recommended sequence:** Step-0 timestamps + one xctrace capture (½ day) → L1/L2/L3 (+ L4 as
  the Stage-C gate) → pivot the structural effort to prefill/batch-k, carrying decode to ~90+ via
  **speculation** rather than the last 15 GB/s of GEMV tuning.

**Certain (measured/arithmetic):** the 77/20/3 budget; scales add 25% to weight traffic;
effective BW 67.5 not 54; incumbent ~75–83 GB/s on this rig; within-token starvation impossible;
per-dispatch floor 3.8 µs × 337; down-proj at tg=32; lm head streams cold at 84 GB/s with
unlimited parallelism (falsifies "too few requests in flight"). **Probable (needs capture):**
issue-bound GEMVs; 0.3–0.7 ms boundary bubble; Stage-B flat A/B was SLC. **Speculative:** cold
DRAM permits ≥100 GB/s/GEMV here; static-scale parity; untracked-hazard savings.

# CUDA cgo-free decode — headroom audit (Fable pass, RTX 2070 SUPER / sm_75)

> Companion to `task-cuda-cgofree-spike.md` and `cuda-megakernel-spec.md`. A "for grins"
> optimization pass that turned up a real reframe.
>
> **The correction it forced (worth stating plainly):** the earlier read — "GEMV is at
> 85% of peak, near-roofline, don't chase CUDA perf" — is right about the *GEMV kernel*
> and right that ALU/tensor-core tricks can't move a bandwidth-bound kernel. But it
> conflated GEMV-roofline with *token*-roofline. **The GEMV is at its roofline; the token
> is not.** End-to-end, a token streams ~1.10 GB in 4.57 ms = ~241 GB/s = **~54% of peak
> effective DRAM utilization**. The missing 31 points are the **~1.6 ms of glue** (single-
> block `rmsnorm_quant`/`quant_vec`/`swiglu_quant` at `GridX:1` = 1/40th of the GPU, a
> 12-block attention, ~310 small-kernel launch latencies) during which DRAM sits idle.
> That glue is recoverable — and it's the whole game on the **0.5B hero binary**, which is
> launch/glue-bound (fixed glue doesn't shrink with the model).
>
> **What it does NOT change:** tensor cores / `__dp4a` / IMMA are still a no for M=1 decode
> (bandwidth-bound). And the **cooperative single-launch megakernel is still NOT the build**
> — Fable independently agrees (grid.sync tax + occupancy risk); the win is the *non-
> cooperative* 3-kernel fusion already scoped in `cuda-megakernel-spec.md §5.2`.
>
> **Two checkable claims to verify before acting:** (1) production `cuda/resident.go` `step()`
> (~line 239) does a full 151936×4 B = 608 KB logits D2H + sync every token and never loads
> `argmax_reduce`, so the *shipped* path is ~1.5% (1.5B) / ~4–6% (fused 0.5B) slower than the
> 218.6 headline (which uses on-device argmax). (2) the 0.5B tok/s figures are all
> **unmeasured** — a 0.5B baseline run is the mandatory first step.

## Ground truth (verified in-tree)

- Per-token launches: **18/layer** + final rms + LM-head = **~506/token (1.5B, 28L)**,
  **~435 (0.5B, 24L)** (`resident.go step()`). The "~337" brief figure undercounts.
- Measured: W8A8 GEMV **85% of peak** isolated, W4A8 COAL3 **80%**, 141-GEMV chain **83%**
  (374 GB/s); GEMV→GEMV gap ~0.4–0.5 µs; per-dispatch CPU tax **~5 µs** (gocudrv channel-hop
  + `cuLaunchKernel`); outer executor hop **15.3 µs/token** (0.34%, once/token); e2e **218.6
  tok/s = 4.57 ms/token** = **~2.9 ms GEMV + ~1.6 ms glue/attention**.
- In-repo precedent for the 0.5B failure mode: on WebGPU the resident **0.5B ran 166 tok/s
  vs the 1.5B's ~87–111** — "only ~2× faster despite 3× smaller, because the fixed
  ~420-dispatch glue doesn't shrink" (`gpu-next-levers-assessment.md §7`).

## 1. Bytes levers (bandwidth-bound ⇒ only *fewer bytes* helps)

- **f16 group scales — the one real per-token byte cut. DO, gate-tested.** Scales are ~20%
  of the int4 stream; f16 (convert to f32 in-kernel, keep f32 math + int32 dp4a) saves
  ~7–8% of GEMV bytes → **+4–5% e2e (~228–230 tok/s), +6–8% on a fused token.** Caveat: the
  team moved f16→f32 for parity earlier, but the *committed* gate is the 3%-near-tie rule,
  not exact-count; f16 rounds at ~2⁻¹¹, far below 3%. High confidence on byte math, medium
  on parity outcome (it's a gate re-run, not a debate).
- **f16 KV + coalesced attention read — DO eventually, ~0% at bench context.** KV is f32;
  `attention` in `glue.cu` reads K fully uncoalesced (lanes ~1 KB apart → up to ~8× sector
  amplification). Material only at 2k–4k+ context; invisible in the current number.
- **Requant the q6_k-origin int8 tensors to int4 — DON'T by default** (~8% of bytes, but
  it's a model-quality decision; only behind parity + a real quality eval).
- **L2 weight residency — non-lever** (1.10 GB streamed once through 4 MB L2; nothing to pin).

## 2. Turing tensor cores / `__dp4a` / IMMA — no, for M=1 decode

The kernel is 80–85% of DRAM peak using plain `__dp4a`; W4A8 GEMV arithmetic intensity is
~2 int-ops/byte, orders below the IMMA roofline. An ALU trick cannot speed a bandwidth-bound
kernel. **Zero e2e gain, negative complexity.** Where they pay: M≥8 (prefill / batched
verify) — and note the CUDA backend's *prefill is currently the CPU staged path*, so a dp4a
prefill GEMM is the biggest **TTFT** lever, but that's a different metric and a separate build.

## 3. CUDA Graphs vs the executor

- goinfer executor (LockOSThread + channel): one hop/token, 15.3 µs = 0.34% — leave it.
- Per-launch ~5 µs CPU tax: **hidden on the 1.5B** (506×5 µs ≈ 2.5 ms < 4.57 ms GPU),
  **decisive on the 0.5B** (435×5 µs ≈ 2.2 ms vs ~2.0–2.3 ms GPU → async-hiding margin gone).
- **1.5B: Graphs buy ~+3–6%, not worth alone** (GEMV→GEMV gaps already ~0.5 µs; the 1.6 ms
  glue replays just as serialized). **0.5B: Graphs remove the CPU wall ~+15–25%, but fusion
  is strictly better.** gocudrv v0.2.0 has cooperative launch but no `cuGraph*` (~7 dlopen'd
  symbols to add; per-token pos/nKeys must be devirtualized to a device uniform — same pattern
  as the on-device `aScale`). **Cheaper pure-Go alternative:** the 5 µs is mostly gocudrv's
  per-call channel hop, not `cuLaunchKernel` (~1.5–2 µs native) — a batched-submit / direct-
  call gocudrv fork cuts 5→~2 µs/launch (0.5B CPU side 2.2→~0.9 ms, below the GPU timeline),
  zero PTX changes, and de-risks GC-pause jitter on the 0.5B critical path.

## 4. The 0.5B hero binary — where the win is real (do this)

Derived from measured 1.5B constants (qwen2.5-0.5b: H=896, 24L, nH=14, nKV=2, I=4864,
vocab=151936, tied embed) — **all 0.5B numbers unmeasured, flagged speculative:**

| quantity | estimate |
|---|---|
| weight bytes/token (LM head = 27–38% of bytes at this vocab) | ~330–385 MB |
| GEMV work @ 374 GB/s | ~0.9–1.0 ms |
| glue work+gaps (scales with launch count) | ~1.1–1.4 ms |
| GPU timeline (unfused) | ~2.0–2.4 ms |
| CPU launch side (435 × 5 µs) | ~2.2 ms |
| **current-architecture 0.5B estimate** | **~2.3–2.6 ms ≈ 390–440 tok/s** |
| **fused floor** (GEMV 0.95 + attn 0.1 + gaps/argmax 0.15) | **~1.2–1.4 ms ≈ 700–850 tok/s** |

Per-layer GEMVs on a 0.5B average ~4 µs of GPU work — the *same size* as the 5 µs launch tax
and the glue kernels. At this scale everything is fixed cost: "24 tiny layers + one big LM-head."

**Ranked 0.5B plan:**
1. **Measure first** — run `TestRealE2EDecode` against a 0.5B q4_k_m checkpoint. If it lands
   ~sub-450 while bytes predict ~900+, launch-boundedness is confirmed. ~half a day.
2. **Zero-risk launch diet (both models):** fuse `rope(q)+rope(k)+kv_store(k/v)` into one
   kernel (−3 launches/layer, same math/order); add a `+=`-accumulate flag to the GEMV
   epilogue to absorb the 2 `residual` launches (it already has a bias epilogue). 18→13/layer.
   **~+3–5% (1.5B), ~+10–15% (0.5B)**, parity-safe (gate re-run).
3. **The 3-super-kernel fusion (spec §5.2) — the headline. No cooperative launch needed.**
   The cross-block scale reductions dissolve with cheap redundant per-block recompute (tiny
   L2-resident vectors): K1 = QKV-GEMV blocks redundantly recompute rmsnorm+quant of x[H]
   then do their rows; K2 = attention → O-GEMV blocks redundantly quantize ctx; K3 = gate/up
   → swiglu elementwise → down as K3b off a globally-written maxabs. ~5–6 launches/layer ≈
   150/token. **Estimate: 0.5B ~1.2–1.5 ms ≈ 650–800 tok/s (+60–90%); 1.5B ~3.4–3.8 ms ≈
   265–295 tok/s (+20–35%).**
4. **Graphs (or the gocudrv de-hop) after fusion:** mops up residual CPU/gap slice, +5–10%,
   and buys jitter immunity for the shipped binary.
5. **Cooperative single-launch megakernel — DON'T.** Over fusion its only saving is ~150
   launch gaps ≈ 0.1–0.2 ms, bought with 5–7 `grid.sync()`/layer (~0.3–0.7 ms — wash or
   worse) plus co-residency caps that risk the measured 80% GEMV. The spec's own §5 order
   (start with the 3-kernel split) was correct; cooperative stays a documented option.

## 5. Other CUDA-specific findings in the source

- **Production full-logits D2H** (`resident.go:239`) vs the headline's on-device argmax —
  wire the existing `argmax_reduce` (in `glue.ptx`) into the greedy production path; optionally
  move embedding lookup on-device to kill the per-token H2D. **+~1.5% (1.5B), +4–6% (fused
  0.5B)**, cheap, parity-safe. (Same argmax-readback lesson as the Metal pass.)
- **Single-block glue kernels** are the 1.6 ms — don't multi-block them (the global scale
  reduction is why they're one block); fusion (§4.3) is the fix.
- **Streams/overlap:** nothing real (bandwidth-bound; tokens sequentially dependent).
- **Beyond-bytes footnote:** the only way past bytes/token is amortizing the weight stream
  over >1 token — a true M=K+1 dp4a batched-verify + the zero-cost n-gram draft (the CUDA
  analogue of the deferred WebGPU "Stage B"; acceptance 0.78–0.89 measured on code). Large
  build, prior kill-gate on file. Park behind fusion; revisit with fused numbers.

## Ranked levers (honest e2e)

**1.5B (218.6 today):** (1) 3-super-kernel fusion **+20–35% → ~265–295**; (2) f16 scales
+4–5%; (3) cheap fusion subset (rope/kv + residual-epilogue + greedy argmax D2H) +4–7%;
(4) Graphs alone +3–6% (skip for 1.5B); (5) tensor cores / 80→85% chase / L2 / streams **~0**.

**0.5B (unmeasured; est. ~390–440 baseline):** (1) 3-super-kernel fusion (+ cheap subset)
**+60–90% → ~650–800**; (2) gocudrv de-hop or Graphs +15–25% standalone / +5–10% stacked;
(3) cooperative megakernel — not worth it over (1).

## Certain vs speculative

**Certain (measured in-repo):** GEMV 80/85% isolated, 83% chained; 5 µs/dispatch; 15 µs/token
executor; 4.57 ms = 2.9 GEMV + 1.6 glue; 18 launches/layer; the headline-vs-production
argmax/logits asymmetry; the WebGPU 0.5B glue-domination precedent. **Derived-but-solid:**
e2e ≈54% of peak; the 0.5B CPU/GPU crossover. **Speculative until the box runs:** every 0.5B
tok/s figure; the fusion floor; f16-scale parity outcome; graph per-node savings.

**Verdict:** the GEMV is at its roofline and nothing ALU-flavored moves it — but the *token*
runs at ~54% of the memory ceiling, and the recoverable gap is the glue: modest on the 1.5B
(+20–35%, medium effort, parity churn), and the difference between **~420 and ~750 tok/s on
the 0.5B hero binary**, where the already-spec'd **non-cooperative 3-kernel fusion** — not the
cooperative megakernel, not tensor cores — is the right build, with a 0.5B baseline
measurement as the mandatory first step.

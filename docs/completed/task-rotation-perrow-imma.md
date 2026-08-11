# Exploration — rotation + per-row scales + IMMA (one candidate, not three)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


**Status:** ✅ **DECIDED — DEFERRED (2026-08-04).** Both halves of the fork are now
measured and neither justifies opening it: the cheap path (per-row scale search, no
rotation) is **dead** (Phase 0b, below), and the expensive path (rotation + IMMA) buys a
prefill-only ~3× on an already-past-threshold TTFT plus a capped mid-single-digit decode
win. The format **stays group-scaled int4**; tensor cores are deferred. This doc is
retained as the decision record — the coupling analysis and the measured kill are why the
candidate is closed, not skipped. The one-line decision lives in `ollama-chase.md` §7
(`a276201`); the full accounting is here. Reopen only under the trigger in §11.

**Why it exists:** the parked "int32-per-group accumulation" note in
`gemv_w4a8_batched.cu` records that goinfer's int4 scale granularity is what blocks
tensor-core IMMA. Separately, rotation-based quantization ([ConvRot,
arXiv:2512.03673](https://arxiv.org/abs/2512.03673); QuaRot / SpinQuant in the LLM
lineage) attacks exactly the property that makes fine-grained scales necessary. These
read as two unrelated ideas and are in fact one change. This doc records the coupling
so that whoever opens the candidate opens all of it, and so nobody opens half of it and
takes a quality regression.

---

## 1. The coupling

Three facts that only matter together:

1. **goinfer's int4 is group-scaled.** `int4GroupSize = 32`, symmetric, codes `[1,15]`
   with `8` = zero (15 levels, no zero-point). One scale per 32-element group along K —
   `f32` in the `.giw` / CPU path, `f16` on the GPU side.

2. **Group scales force float accumulation, which forecloses IMMA.** `gemv_w4a8_fwd`'s
   inner loop takes an `int32` `__dp4a` partial over **one word — 8 values** — then does
   `facc += (float)p * groupScale`. Float addition is not associative, so no tensor-core
   K-tiling reproduces that accumulation order bit-for-bit. This is why milestone 1's
   batched GEMV is dp4a weight-stationary rather than an IMMA GEMM: it was the only
   design that held the byte-identical gate. Measured cost of that choice: dp4a is
   ~1/3 of IMMA int8 throughput on Turing — **~23× at M=512 against a ~72× theoretical
   ceiling** (recorded in the kernel's doc comment).

3. **Rotation flattens the distributions that make fine granularity necessary.**
   Rotating weights (and, in the general form, activations) redistributes outliers
   across channels so that coarser quantization granularity costs less accuracy. If
   rotation buys enough, **per-row** scales become viable — and per-row scales mean
   `int32` accumulation across the entire K dimension, applied once per output element,
   which is exactly the condition IMMA needs.

So: rotation is not a quality nicety bolted onto a kernel change. It is the thing that
makes the kernel change quality-neutral. Scope them together or not at all.

---

## 2. Why this is not funded now

Four reasons, all evidence-backed from this project:

- **There is no demonstrated int4 quality problem.** int4 has been blamed for a quality
  deficit twice — `625303e` ("int4 is too aggressive") and `bcadd44` ("the deficit is
  the int4 weights → build affine") — and **both were overturned and retracted**. The
  26B Metal "divergence" that looked like a quantization collapse turned out to be
  discrete top-8 routing flips at ~0.001 router margins. The standing record on
  "blame int4" is 0-for-2.

- **The blast radius is a full parity refresh.** Any change to scale granularity or
  weight values changes decode bits for **every int4 model**. That means the goldens
  actually run across all validated families, `deps_hash` refreshed through
  `scripts/refresh_parity_hashes.sh`, and `validated_at` bumped. Never refresh a
  goldens-gated hash without running the goldens.

- **Rotation's headline wins are in activation quantization, and goinfer is W4A8.**
  QuaRot-class results are strongest for W4A4, where 4-bit *activations* make outlier
  channels catastrophic. goinfer already runs int8 activations, where outliers are far
  better tolerated. The claim available here is narrower: *can rotation flatten the
  weight distribution enough that per-row scales match per-group quality?* That is
  plausible but it is a smaller claim than the papers' headline, and it must be measured
  on goinfer's own scheme, not inherited from theirs.

- **Nothing currently on the critical path is quantization-bound.** The live levers are
  staging I/O (Metal), sequential M=1 prefill (CUDA — crossover against Ollama measured
  at ~128 prompt tokens), and per-layer submit coordination. Prefill already gets ~23×
  from the dp4a batched GEMV without touching quantization at all.

---

## 3. What would open it

Any one of these makes the candidate worth funding:

- The dp4a prefill path lands and **compute becomes the measured bottleneck** at real
  prompt lengths — i.e. the remaining ~3× is on the critical path rather than theoretical.
- A quality deficit against a peer or reference is **demonstrated and attributed to
  quantization**, by measurement, not by elimination.
- A parity refresh is being paid **anyway** for an unrelated reason, making the blast
  radius marginal rather than the dominant cost.

Absent all three, the correct action is to leave this doc alone.

---

## 4. Phase 0 — bounded-first, before any kernel or quantizer work

All three are cheap. Two need no GPU. **Do not proceed past Phase 0 unless all three
clear.**

### 0a. How much does per-row alone cost? (the load-bearing measurement)

Re-quantize existing validated checkpoints with **per-row** scales instead of per-group,
**with no rotation**, and measure the calibrated envelope: `CPU-int4-perrow vs CPU-f32`
against today's `CPU-int4-group vs CPU-f32`, on the same prompts and the same
instrument.

- **If per-row alone is nearly as good** → rotation is unnecessary and the candidate
  collapses to a much cheaper "coarsen the scales, build the IMMA kernel" change. Best
  possible outcome; retire most of this doc.
- **If per-row alone is materially worse** → that gap is exactly what rotation has to
  buy, and 0b decides whether it can.

Offline, CPU-only, no kernel work. This is the measurement that sizes everything else.

### 0b. Are the weights actually outlier-concentrated?

Rotation's value is proportional to how concentrated the outliers are. Characterise the
per-channel weight distribution of the real target models (Gemma-4 26B-A4B, and at least
one dense flagship). Report kurtosis / max-to-median per channel per layer.

**If the distributions are already flat, rotation buys nothing** and the candidate ends
here regardless of 0a. Do not assume the papers' distributions transfer — they were
measured on other models with other training recipes.

### 0c. Is the IMMA ceiling real on this card?

Microbenchmark IMMA against dp4a **on the actual GEMM shapes goinfer would run**, on the
2070 SUPER, not on a spec-sheet ratio. The ~3× is currently an arithmetic estimate.

If it comes in materially under 3× on real shapes, the entire payoff shrinks and the
parity-refresh cost stops being worth it.

---

## 5. Phase 1 — rotation offline, quality only, no kernels

Only if Phase 0 clears.

Apply the rotation at quantization time and measure quality **on the CPU path alone**.
No GPU kernels, no new PTX, nothing shipping.

Design questions to answer here, not later:

- **What is foldable?** Hadamard-style rotations are free where they can be absorbed
  into adjacent linear layers along the residual stream. Enumerate which of Gemma-4's
  matmuls are foldable and which need an online rotation at runtime — the latter is
  per-token compute in the decode path and must be costed, not waved through.
- **The learned norm weights are in the way.** Gemma-4's RMSNorms carry learned
  per-channel scales, and an orthogonal rotation does not commute with them. The standard
  treatment absorbs the norm scale into the following weight matrix. That is real model
  surgery, and it interacts with the sandwich-norm structure (§A1's seven norms and
  `layer_scalar`). Spec it against the CPU `runLayersGemma4`, which is backend-agnostic
  and where the risk lives.
- **MoE routing will move.** Rotation changes numerics, so top-8 selections will flip at
  the ~0.001 margins already characterised. That is expected, not a defect — but the
  re-validation must use the **margin-gated routing instrument** (threshold 0.01, see
  `cuda/gemma4_moe_resident_test.go`), not unconditional index agreement, which the 26B
  finding established as invalid at real width.

**Gate:** the calibrated envelope on the CPU path, at depth. Use the scaled-dense
fixture family at 12/24/48/64 layers — a fixture too shallow to compound cannot detect a
per-layer quantization floor, and a fixture without routing cannot detect a routing
consequence. Representativeness has three axes here: **depth, width, and discreteness**.

---

## 6. Phase 2 — the IMMA kernel

Only if Phase 1's quality holds.

Per-row scales permit `int32` accumulation across the whole K dimension with the scale
applied once per output element, so an IMMA GEMM **can** be bit-identical to a
correspondingly-restructured GEMV. That identity must be designed in from the start, the
way milestone 1's dp4a kernel was — not discovered afterwards.

- New kernel in its own file. `moe.ptx` and the audited PTX artifacts stay untouched.
- The M=1 case must reproduce the reference GEMV bit-for-bit. That is the first test that
  passes; nothing downstream is worth measuring until it does.
- Both the decode GEMV and the batched prefill GEMV move to per-row together, or neither
  does. A split would leave two accumulation contracts in the tree.

---

## 7. The parity refresh, in full

This is the real cost and it should be planned, not absorbed.

- Goldens **run**, not skipped, across every validated family.
- `scripts/refresh_parity_hashes.sh` — the sanctioned, goldens-gated mechanism. Not a raw
  hash bump.
- `validated_at` bumped; `deps_hash` refreshed from the run's actual output.
- Commit message carries the `Parity-Deps-Refresh:` trailer with pass/skip/fail counts.
- Families whose fixtures skip locally are covered by **inference**, not evidence — say
  so explicitly rather than letting the row read as proven.
- Cross-backend MoE comparisons use the margin-gated instrument. Any gate still asserting
  unconditional expert-index agreement will false-red and must be migrated first.

---

## 8. Risks and open questions

- **W4A8 vs W4A4.** The literature's gains are largely activation-quantization gains.
  Transferring the headline number to a weight-only scheme is exactly the class of error
  this project has made repeatedly (the 81.6% locality figure carried across hardware;
  Metal's `PrefillLast` assumed to cover MoE). Measure on goinfer's scheme.
- **Online rotation cost in decode.** Decode is bandwidth-bound at M=1; extra per-token
  compute is cheap there. Prefill is compute-bound above M≈45, where it is not. Cost both
  regimes separately.
- **The Metal f16 group-scale floor.** Metal uses f16 group scales against the `.giw`'s
  f32, which floors Metal-vs-CPU cosine near ~0.98. Moving to per-row changes that
  relationship, possibly for the better (fewer, larger scales are more representable in
  f16). Worth measuring, but do not let it become a justification on its own.
- **Doing half the change is the failure mode.** Coarsening scales without rotation is a
  quality regression; rotation without coarsening buys nothing goinfer needs. If the
  candidate is opened, it is opened whole.

---

## 9. Related docs

- `docs/task-giw-f16-scales.md` — the f16 scale representation in the `.giw` / GPU path.
  Any granularity change lands on top of this.
- `docs/task-gpu-batched-prefill.md` and `docs/task-prefill-optimization-campaign.md` —
  where the dp4a batched GEMV and its ~23×/~72× ceiling live.
- `docs/task-moe-streaming.md` — the decomposition this candidate would have to beat to
  be worth funding.
- `docs/parity-coverage-policy.md` and `docs/parity-hunt-playbook.md` — the refresh
  discipline §7 is bound by.

---

## 10. Prior art

- ConvRot — rotation-based plug-and-play 4-bit quantization for diffusion transformers.
  [arXiv:2512.03673](https://arxiv.org/abs/2512.03673). The DiT-side result that prompted
  this doc; the LLM analogues (QuaRot, SpinQuant) predate it and are the relevant
  references for goinfer.
- Practical note: INT8 + row-wise scaling + ConvRot as shipped in ComfyUI v0.27.0 —
  [explainer](https://note.com/hirorohi03/n/n047a8c5f7f8b?hl=en). Benchmarks there are
  image-generation start-up seconds, **not** LLM decode, and carry no weight for goinfer's
  purposes. Its one directly relevant observation is that INT8 receives hardware
  acceleration on RTX 20/30 where FP8 does not — which is a point in favour of the lane
  goinfer already occupies, and requires no action.

---

## 11. Decision record

| date | decision | basis |
|---|---|---|
| 2026-08-03 | Scoped, **not funded** | No demonstrated int4 quality problem (0-for-2 on prior attributions); full parity refresh blast radius; nothing on the critical path is quantization-bound; rotation's headline gains are W4A4, goinfer is W4A8 |
| 2026-08-04 | Phase 0 run (weight-space) — **optimistic read** | Per-row symmse int4 weight error 1.24× per-group (naive 1.73×). Read as "cheap path might be viable"; corrected the next day (weight-space proxy misleading) |
| 2026-08-04 | Phase 0b (forward) — **cheap path DEAD** (`663fdf6`) | `TestPerRowScalePhase0b`, teacher-forced qwen3-1.7B: per-row ppl **107.97** vs per-group **28.54** vs oracle 26.75 (KL 1.39 vs 0.27, agree 68.6% vs 76.7%). The 1.24× weight-space error compounds ~4× at the output through 28 layers — **not benign**. Scale search alone is dead; rotation reverts to prerequisite |
| 2026-08-04 | **DEFER tensor cores; format stays group-scaled int4** (`a276201`, `ollama-chase.md` §7) | Cheap path dead (above); expensive path (rotation + IMMA) buys prefill-only ~3× on a past-threshold TTFT + a capped decode win (~mid-single-digit, bounded by 45% BW util, not the 11% scale-stream saving). **Reopen only if** a decode-BW profile realizes the ~11% AND rotation is funded |

# Prefill optimization campaign

---

## CAMPAIGN OUTCOME — CUDA resident batched prefill (2026-08-03, DONE & BANKED)

> This section is the retrospective; the plan that follows it (from 2026-07) is kept as history.
> Lever 1 ("batched CUDA prefill") was executed on the `cuda/` resident path. It is **done and
> deliberately stopped** — see "Why it stopped" and "Do not re-propose IMMA" below. All commits
> pushed to `main` through `1d90394`.

**The product question is answered for the DENSE lane.** A 2048-token prompt's TTFT went from
**13.1 s (sequential M=1) to 2.1 s (batched)** on real qwen2.5-coder-1.5b — the unusable
long-context regime is gone; RAG / code-context / long-chat are no longer blocked *on dense models*.
That absolute 6.17× is a **peer-independent engineering result** and it stands.

> **⚠ The "crossover ~128 → ~320 tokens" below was measured against Ollama 0.5.7 (Jan 2025), which
> was re-discovered on 2026-08-04 to be ~18 months stale. Against CURRENT Ollama (v0.32.5), the
> crossover collapses to ~50 tokens — goinfer's decode edge is now ~1.19× at 1.5B short-context and
> it is *behind* at long context and on prefill. Treat every "crossover"/"wins short prompts"
> statement in this doc as an ENGINEERING result vs a fixed historical peer, NOT a current
> competitive claim. The current-peer numbers and the honest framing are in `docs/benchmarks.md`
> §B2's re-anchor box. This does not change the engineering (bit-identical, the levers, the
> attribution lessons) — only the competitive framing.**

The **Ollama crossover moved ~128 → ~320 tokens** *(vs 0.5.7 — historical; see the warning above)*:
goinfer wins total request time up to ~320-token prompts against that peer. Beyond that it stays
behind on raw prefill throughput — a competitiveness gap, not a usability one, and structural.

> **SCOPE CAVEAT — this does NOT cover the 26B (or any gemma4moe/MoE that declines to sequential).**
> `PrefillLast` guards out gemma4moe, so gemma-4-26b-a4b still falls back to the per-token loop and
> remains at **~61 s TTFT at 2048** (measured, KV-only prefill already on) — squarely in the unusable
> regime. And it **cannot be rescued by the levers on this campaign's list**: the batched dense/attn
> plumbing is bounded at ~1.43× (61 → ~43 s, `docs/task-26b-prefill-bound.md`), and even a full
> expert-major batching (Lever 3) on top might reach 3–4× → ~15–20 s. The real cause is a **hardware
> mismatch** — an 8 GB card paging a 13 GB model (and, on Metal, a 16 GB laptop holding a 13 GB mmap
> at 0.73 tok/s with phase-1 immovable). No amount of kernel quality closes that. The engineering
> here is durable and the lessons transfer; the 26B numbers were never going to be good on this
> hardware, and it stays **disclosed-with-caveats** whatever is done to it. The strategically higher
> move is to point the now-much-better prefill at models that **fit** the hardware (a resident dense
> flagship on the 2070 SUPER), where the decode advantage is real — publishable rows with honest
> provenance, not another factor on a model that will remain caveated regardless.

### The five landed levers (each bit-identical to the sequential M=1 GEMV; decode byte-identical)

| # | lever | what | result |
|---|---|---|---|
| 0 | **Batched mixed-M forward** (`PrefillLast`, `0b289b1`) | whole prompt in one weight-stationary pass; causal + per-row sliding-window masks; experts still sequential | the plumbing; e2e KV bit-identical all-layers×rows + 64-tok decode byte-identical |
| 1 | **MT tile sweep** (`a41eb74`) | activation-columns-per-weight-load 8→32 | ~6% (the weight-amortization lever, and *only* ~6% — see below) |
| 2 | **Coalesced GEMV load** (`int2`, `9070c2e`) | activation pair as one 64-bit load | bytes/sector 49.99→98%, GEMV 4.98→4.41 ms |
| 3 | **Register-blocked GEMV** (RN=2 rows/warp, `7c6d935`) | reuse each load across RN output rows → RN× fewer L1TEX loads | GEMV 4.41→3.38 ms; scoreboard stall 17.8→7.5 cyc |
| 4 | **Coalesced attention** (`float4` QK, `55c850a`) | vectorize the K/Q read, adds kept separate | attn (M=2048) 92→29.9 ms (3.1×); 2048 TTFT 3.33→**6.17×** |

Cumulative GEMV: MT ~6% + coalesce ~13% + RN ~30% ≈ **1.5×** (gate/up 4.98→3.38 ms). TTFT vs
sequential: **128 5.78× / 512 5.80× / 2048 6.17×**. Attention's share of prefill at 2048 fell
66.9% → 38.9%; the GEMV is now ~61% at 2048.

**Coverage (`TestPrefillCoverageAudit`, all 23 validated families).** Batched prefill runs for the
**5 dense-lane families that are cuda-resident: llama, mistral, phi3, qwen2, qwen2_5_vl** — complete
for the lane it targets. The other 18 fall back to the sequential loop: 4 are cuda-resident but
declined by a `PrefillLast` guard (MoE — mixtral, glm4_moe; sandwich — gemma3; qk-norm — qwen3), and
14 are not cuda-resident at all (mla, ssm, yarn-mscale, gated-shared, layer-norm, or family-class),
so they never reach `PrefillLast`. Extending the guard reaches at most 4 more; the single cheapest is
**qwen3 (qk-norm** — the decode `qk_norm` kernel already exists), which would add a current flagship.
The 14 need residency features first — a separate, larger effort. So batched prefill covers exactly
{cuda-resident} ∩ {dense}, which is the intended scope.

### The attribution record — the most transferable output

The GEMV gap was attributed **five times; four were wrong**, and the discipline that caught them is
the point. Each wrong attribution read like a measurement:

1. *"~23× weight-amortization ceiling"* — refuted by the MT sweep (~6%). **This is the number the
   `task-rotation-perrow-imma.md` candidate still cites; it is stale.**
2. *"needs IMMA / fundamental to bit-identity"* — refuted: 7.9% of dp4a peak, the compute ceiling
   was 92% unused.
3. *"activation-L2-bandwidth-bound"* — this one **became code**: the shared-staging kernel
   (`gemv_w4a8_staged`) cut global traffic 8× and moved wall time **1.2×**. It is kept, gated
   bit-identical, **unwired**, as the reproducible refutation. The specific error: a *demanded* read
   rate exceeding DRAM proves the reads are cache-served, **not** that the cache is saturated.
4. *"issue-bound on a fat SASS instruction mix"* — refuted by ncu (22.78% issue slots). The distinct
   lesson: an instruction-mix histogram bounds throughput **from above**; it cannot establish you are
   *at* that bound — only stall/eligibility data can.
5. **L1TEX latency from uncoalesced loads** — the hardware stated it directly, once `ncu` was
   installed. That is what levers 2 and 3 fixed.

**A second standing lesson — the cross-context transfer error.** Both decode "levers" that were
queued after this campaign (CUDA graphs ~1.4–1.7×, async miss-DMA ~4.3 ms) turned out to be **26B
numbers applied to the dense lane**: graphs help only dispatch-bound models (measured 1.01× on the
1.5B), and the miss-DMA is a paged-expert-cache cost a fitting model does not have. Same class as
"a tiny-fixture cosine cannot detect a per-layer floor" and "the 81.6% locality figure did not
transfer across hardware" — a measurement taken in one context (the 26B, hardware-mismatched) read
as a property of another (the dense lane goinfer wins). The lesson is more durable than either
lever: **name the model and the regime a number was measured in, before you carry it.**

The operational lesson, now standing: **profile the unit before designing the fix, now that the
profiler exists.** The attention lever (lever 4) followed it — `attn_batched` was profiled *first*
(L1TEX-throughput-saturated, 21.96% bytes/sector, genuinely traffic-bound — the opposite of the
GEMV), which is why a one-line `float4` change bought 3.1×.

### Why it stopped (and why query-tiling is not next)

The obvious next lever — query-tiled attention with shared-K/V staging, to kill the O(M²) K/V
re-reads that still leave L1TEX at 99.5% — was **declined on the arithmetic**. It optimizes a number
past the threshold that mattered: attention 820 → maybe 250–350 ms takes total prefill 2.1 → ~1.6 s,
against Ollama's ~0.22 s at 2048 — i.e. 9.4× behind becomes 7.2× behind. A days-level delicate
kernel (its exact-float-sum order forces a blockDim=128 reduction tree and Bk=128 tiles that strain
the 64 KB shared budget → a 2-pass-recompute design), to move a number already past usability and
still nowhere near the peer. Scoped, ready, **not funded**: `docs/task-prefill-attention.md`.

### The ceiling — DO NOT RE-PROPOSE IMMA without reading this

At 2048 the GEMV is ~61% of prefill and **at its bit-identical knee** (RN=2, 100% occupancy,
Compute 54% of the *dp4a* peak). The residual is the **tensor-core gap**: dp4a is ~1/3 of Turing
IMMA, and closing it needs an IMMA GEMM — which reorders the group-scaled cross-group **float** sum
and therefore **cannot be bit-identical** to the decode GEMV. That is an architectural consequence
of the cgo-free, bit-identical thesis: a **stated trade, not a deficiency**.

IMMA is the recurring "obvious" proposal and it is **scoped-not-funded on purpose**
(`docs/task-rotation-perrow-imma.md`): it requires coarsening int4 scale granularity (per-row, via
rotation) to permit int32 accumulation, which is a full parity refresh across every validated family
for a model with **no demonstrated int4 quality problem** (the "blame int4" record is 0-for-2). Note
that that doc's motivation still cites the stale *~23×* number from lever 1 above; the real dp4a-path
GEMV is L1TEX-bound, not weight-amortization-bound, so the IMMA payoff estimate there should be
re-derived from the profile before the candidate is ever opened. Absent a *measured* quantization
bottleneck on the critical path, the correct action is to leave both docs alone.

### Deferred, bounds recorded (cheap to resume, do not re-derive)

- **26B non-expert half** — batch the gemma4moe dense/attention branch, experts sequential.
  Bounded at **786 MB batchable / 714 MB sequential → ~1.43× at 2048, shrinking with M** — plumbing,
  not a TTFT fix. `docs/task-26b-prefill-bound.md`.
- **Query-tiled attention** — the O(M²) redundancy fix above. `docs/task-prefill-attention.md`.

### Next, in order, when the box is next worked (NOT another prefill kernel)

The remaining wins are on the **decode** lane goinfer actually wins (the basis of every §B2 claim):

1. **CUDA graphs, safe-gated — DONE (`102b902`), and MEASURED to not be the win.** The safe-gate
   shipped (`cuda/graphs_safe.go` `admitGraphs`: admit only `CU_COMPUTEMODE_EXCLUSIVE_PROCESS` or
   active MPS, startup bit-exactness self-test, decline under DEFAULT — "byte-identical or decline").
   But the measured decode speedup on a **fitting** model is **1.01×** (real 1.5B, 220.9 → 223.3
   tok/s): the ~1.4–1.7× was a *tiny-model* number that did not transfer — at real model size CPU
   dispatch **overlaps** GPU compute, off the critical path. The ~19 ms-of-~29 ms dispatch figure was
   the **26B** (MoE launch explosion), the hardware-mismatch model. So graphs are **safe now but not a
   speed win for the models that fit** — another "measure, don't assume" catch. Do not flip the
   default. (`docs/cuda-graphs-investigation.md`.)
2. **Async miss-DMA (~4.3 ms) — RETIRED, no measurement needed.** That ~4.3 ms was the 26B's PAGED
   expert-cache miss (host→VRAM DMA per token); a dense flagship that fits VRAM has no paged expert
   cache and no miss to hide. Like graphs, it was a 26B number that does not apply to the dense lane.
3. **Then step back from the 26B.** Point the now-much-better prefill at models that **fit** the
   hardware — a resident dense flagship on the 2070 SUPER — for publishable §B2 rows with honest
   provenance. A **release consolidating the prefill campaign** would surface the work; the §B2 rows
   are stronger now than when they were written. This beats another factor on a model that stays
   caveated regardless.

The Mac track continues independently on the staging gap; the **destination-memory hypothesis** there
(is *pinning the slots* what makes the writes slow?) is worth running regardless of the 26B's low
ceiling — if true, it is a real tension inside a shipped default and worth knowing.

---

## Original plan (2026-07, historical)

> **BLUF.** Prefill **correctness is solved** — the non-deterministic NaN that blocked v0.9.0 was
> root-fixed (`59804f3`, the LM head was on the int4 GEMM path instead of its int8-pinned path) and
> is regression-gated (`TestPrefillNoNaN`, `mustFinite` across the Metal parity gates). What remains
> is **speed, and it is uneven**: Metal has a real batched f16-MMA prefill (~4× TTFT on dense);
> CPU is batched-decent; **CUDA and WebGPU prefill are still linear row-by-row** (one decode Run per
> prompt token). This campaign closes the linear-prefill gap on the GPU residency path and finishes
> MoE prefill batching — sequenced around exactly one hard upstream gate that turns out to bind only
> WebGPU, not CUDA.
>
> **Governing discipline:** every prefill path is parity-gated (first-token logits ≡ the decode /
> CPU reference — that identity is *exactly* the bug just fixed), break-it-first, per-geometry
> (M=8 / padded / long — the NaN and `Kwords%32` bugs were both geometry-specific). Measure the
> mechanism, don't assume it (`docs/parity-hunt-playbook.md`).

## Baseline — grounded current state (2026-07)

Correctness: **green on every backend.** Speed, per backend:

| backend | prefill implementation | measured | gap |
|---|---|---|---|
| **Metal** | **batched f16 `simdgroup_matrix` MMA** — whole prompt in one command buffer; head via int8 `gemv_w8a8` | **~3.7–4.0× TTFT** on dense Qwen (140 tok: 2034→543 ms) | **dense-only** — MoE, Gemma 3/4, sliding-window **decline to the per-token loop** |
| **CPU** | batched attention (`forwardn.go`); sparse-MoE batched | ~3.4× dense / ~1.7× Gemma; **MoE 2.4×** on Mellum2 (3.36→8.11 tok/s @1k) | **MoE FFN still per-row** (`forwardn.go:14,228`) — router picks different experts/token |
| **CUDA** | **row-by-row** — `DecodeRunner.Run` once per prompt token to fill KV | short-prompt TTFT 0.66 s @1.5B, 1.3 s @7B; peer wall-clock 2.04×/1.41× vs Ollama (prefill *inside*, not isolated) | **not batched** — linear in prompt length |
| **WebGPU** | **row-by-row** (same option-(a) loop) | 1k-token prompt ≈ **~18 s** to first token | **not batched**; naive tiled WGSL GEMM is *below* the M=1 GEMV until DP4A |

Instrumentation gap worth naming: `benchmarks.md` has **no isolated prefill-vs-peer number** — the
peer ratios are all-in wall-clock (prefill + decode + sampling + HTTP). You can't manage what you
don't isolate; that's Lever 0.

## Levers, ranked

### Lever 0 — isolate a prefill (TTFT) benchmark. *Prerequisite instrument, cheap.*
A prefill-only harness: TTFT / prompt-eval tok/s at fixed prompt lengths (short / 1k / long) per
backend × model, and the peer's prompt-eval on the *same* box. Without it every win below is an
anecdote. Ties to `task-benchmark-refresh.md` B11 ("batched GPU prefill benchmark"). **Do this
first** — it's also how each subsequent lever is proven non-vacuous.

### Lever 1 — batched CUDA prefill. *Biggest actionable win, NOT upstream-gated.*
The single largest structural gap is GPU long-prompt prefill (linear row-by-row). The existing
`task-gpu-batched-prefill.md` marks this "do not build yet" — but that gate is **WebGPU-specific**:
it waits on `dot4I8Packed` (DP4A) in WGSL because a naive tiled WGSL GEMM is *slower* than the GEMV
until packed int8 dot lands. **CUDA has `__dp4a` natively** (already used in resident decode), and
the tiled M>1 building blocks exist (`gpu/gemm.go` `BatchTiled` / `MatmulW8A8Tiled`) — they are just
**not wired into a prefill runner**. So batched CUDA prefill is buildable now, independent of any
upstream binding.
- **Work:** an M-sized `PrefillRunner` (batched causal-attention kernel + tiled int8 GEMM per layer,
  KV written in one pass, head on the last row) wired into `decoder.Generate`; the `cuda/resident.go`
  `UploadKV` bridge is the kept hook.
- **Prove the mechanism first (playbook):** confirm the tiled int8 GEMM via `__dp4a` actually beats
  the row-by-row GEMV *before* committing — a "suspiciously-small or negative win" means the kernel
  is bandwidth-bound and the premise is wrong. Measure on two prompt lengths (short + long).
- **Parity:** batched prefill first-token logits ≡ the row-by-row path ≡ CPU, argmax + cosine +
  ΔNLL; break-it-first.

### Lever 2 — extend Metal batched prefill beyond dense. *No upstream gate; flagship families.*
Metal's batched f16-MMA path exists but `prefillOK` declines **MoE, Gemma 3/4, sliding-window** to
the slow per-token loop — i.e. two flagship families (Gemma, and any MoE) get *no* prefill
acceleration on Metal today. Extend the batched path to (a) sliding-window causal attention, (b)
Gemma's dual-norm / K=V layers, (c) MoE (needs Lever 3's expert-major batching to be worthwhile).
Sequence sliding-window + Gemma first (pure attention/norm work); MoE after Lever 3.

### Lever 3 — expert-major MoE prefill batching. *Cross-backend; the "missing half."*
Today MoE prefill FFN is per-row on CPU (`forwardn.go`) and declines entirely on Metal. Group rows
by selected expert within a prefill chunk (**gather → per-expert GEMM → scatter**) so fetches drop
from `rows × k` toward `distinct experts` — `task-gemma4-moe.md` §B4 calls this "the missing half of
the 2.4× batched-MoE-prefill result on Mellum2" (Phases 5–8, **not started**). Serves CPU, Metal-MoE
(Lever 2c), and batched-GPU-MoE prefill at once. Also generalizes to always-on shared-expert archs
(Qwen2-MoE, GLM, DeepSeek) where the shared expert plays the dense branch's role (§B3).

### Lever 4 — batched WebGPU prefill. *Gated on the upstream binding — verify, don't assume.*
Same M-sized runner as Lever 1 but on WGSL, blocked because the naive tiled GEMM is ~0.91× the GEMV
until `dot4I8Packed` is available. **Status to confirm before scheduling:** the WGSL builtin is
implemented upstream in `wgpu` (gfx-rs), but whether goinfer's binding **`cogentcore/webgpu` exposes
it** needs a direct check of that repo — the internal "do not build yet" note predates the upstream
merge, so the gate may be closer to cleared than assumed. **Do not benchmark batched WebGPU prefill
before the packed-int8 dot is wired** — measuring before is pointless (`task-benchmark-refresh.md`
B11). If the binding lags, the cheapest unblock may be contributing `dot4I8Packed` upstream.

### Lever 5 — vision prefill (SigLIP) tiled GEMM. *Adjacent, real multiple.*
The vision tower's attention matmuls are still naive f32 (the WebGPU path already gives ~9×,
171→18.8 s/image); a tiled GEMM is the noted next lever for another multiple. Aikit-side (ties to
`aikit/docs/task-native-gpu.md` vision phase); listed here for campaign completeness, owned there.

## Sequencing

1. **Lever 0** (isolate prefill benchmark) — instrument before optimizing.
2. **Lever 1** (batched CUDA prefill) — biggest gap, no upstream gate, building blocks exist.
   Mechanism-verify (`__dp4a` GEMM > GEMV) before committing.
3. **Lever 2a/2b** (Metal sliding-window + Gemma prefill) — parallelizable with Lever 1 (different box).
4. **Lever 3** (expert-major MoE prefill) — unlocks Lever 2c and the CPU MoE-FFN gap together.
5. **Lever 4** (batched WebGPU prefill) — **only after** confirming `cogentcore/webgpu` exposes
   `dot4I8Packed`; until then, track the binding, don't build.
6. **Lever 5** (vision tiled GEMM) — owned in aikit; coordinate.

The non-upstream-gated wins (1, 2, 3) are the campaign's core and can start immediately; only Lever 4
waits on an external dependency — and that dependency may already be satisfiable, which is the first
thing to check.

## Cross-cutting gates (non-negotiable)

- **First-token parity is the invariant.** Every batched prefill path must produce first-token logits
  **identical to the sequential / CPU path** — argmax + logit-cosine + ΔNLL on a real prompt. This is
  the exact class the NaN bug lived in (prefill head diverging from decode head); it is the campaign's
  primary gate, per backend, break-it-first.
- **No-NaN / `mustFinite` stays standing infra.** Extend `TestPrefillNoNaN`'s geometry set to each new
  batched path (M=8 single-tile, padded M, long M); `mustFinite` before every cosine/error floor so a
  NaN can't sail through (`NaN < x` is false in Go).
- **Per-geometry before ship.** Run a *second* prompt length/shape before claiming a lever — the two
  worst prefill bugs this cycle (`Kwords%32`, the padded-140→144 NaN) were geometry-specific and
  invisible on the tested size.
- **Measured, not assumed.** A tiled GEMM that doesn't beat the GEMV is a failed lever, not a shipped
  one; a suspiciously-large win is a symptom to chase (the pageable-D2H lesson).

## What this campaign does NOT touch

Correctness (done, gated); decode/token-generation speed (separate); and MLA-on-CUDA/Metal residency
(the DeepSeek/Kimi "make it fast" work — a different task). Prefill here means prompt-processing /
TTFT only.

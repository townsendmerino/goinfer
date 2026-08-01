# Prefill optimization campaign

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

# Task: find the actual bug behind WebGPU's no-GQA resident-decode divergence

> **Status: OPEN, filed 2026-09-18.** Interim safety fix shipped (gpu/residency.go declines the
> broken case rather than serving it); the actual kernel bug is NOT found. This doc exists so the
> next pass does not repeat the eliminations already done here.

## What triggered this

Benchmarking phi3-mini on WebGPU (`docs/benchmarks.md`'s peer-matrix re-run) found it running at
CPU-equivalent speed (~8.4 tok/s) despite the `webgpu` backend flag. Its `/health` endpoint showed
why: `BuildResident declined ... residency KV alloc (layer 13): ... Not enough memory left`,
falling back to the staged (host-prefill) path.

Root cause of *that*: `gpu/residency.go`'s `ctxCap` used a fixed per-precision ceiling
(`decoder.WebGPUCtxCeiling`, 16384 positions at f32) with no clamp to the model's own
`max_position_embeddings`. Every WebGPU-resident model tested before this used real GQA
(`head_count_kv < head_count`), so its real KV-cache bytes/position were small enough that 16384
positions always fit an 8 GB card — the ceiling's own comment calls this "the proven 8 GB fit".
Phi-3-mini has **no GQA at all** (`head_count_kv == head_count == 32`, confirmed directly from the
GGUF header, not assumed) — its real per-position KV cost is ~6x a typical GQA model's, and 16384
positions is 4x its own 4096-token native window besides. The allocation loop ran out of VRAM
partway through (layer 13 of 32) trying to reserve KV for positions the model could never even
serve.

**Fixed** (`gpu/residency.go`): clamp `ctxCap` to `m.Config().MaxPositions` before anything else,
mirroring `cuda/resident.go`'s `resolveCtxCap`, which already does exactly this. Verified: at its
own 4096-position ceiling phi3-mini's real KV need is ~3.1 GB, which fits an 8 GB card easily
beside its ~2.3 GB of int4 weights — the model now builds resident (`webgpu:vulkan-resident`)
instead of declining on VRAM.

## The bug this uncovered

Letting phi3-mini actually reach WebGPU residency exposed a **second, unrelated, and more
serious** problem: the resident decode path's output does not match CPU.

Direct per-position logit comparison (`rf.Forward` vs `mcpu.ForwardForTest`, both int4, fixed
arbitrary token prompt `[1 7 42 100 5 200 13 88]`):

- Prompt-position cosine similarity: **0.94–0.99** across all 8 positions. This repo's own bar
  for "the resident kernel is correct" is 0.999+ (`gpu/gemma3_resident_parity_test.go`'s own
  gate). Every other WebGPU-resident model this session re-checked (Gemma3, gpt-oss, Nemotron
  MoE, LoRA) landed at 0.9998–1.0 on the same style of test.
- Greedy continuation (each side fed its own argmax back in) diverges at generation step 0-1 and
  goes **cosine-negative** by step 2 (-0.02) and again at steps 4, 6, 9 (-0.28, -0.43, -0.12) —
  the two logit vectors point in nearly opposite directions. That is not quantization noise
  (which stays close to 1.0); it is a structural computation difference.

## What was ruled out (do not re-check these first)

- **Not the ctxCap fix itself.** The divergence is present with the fix applied and absent only
  in the sense that the model can no longer reach residency at all without it — the fix is
  necessary to even observe the bug, not the cause of it.
- **Not the `qkvFinalize` fused dispatch.** Forcing the separate `rope`/`ropeStore`/`vStore`
  kernels instead of the fused one (`gpu/decoderunner.go`'s `if m.kvF16 || m.kvI8 { ... } else {
  qkvFinalize(...) }` — temporarily changed to always take the separate-kernel branch) reproduced
  **bit-for-bit identical** wrong output (same cosines, same tokens, to the decimal). Whatever is
  wrong is either upstream of both (Q/K/V projection) or in something both paths share (the
  attention kernel, the KV cache itself, or the RoPE frequency table).
- **Not solely an int4-quantization artifact.** Re-run with `Quant: ""` (full f32, `decode_path:
  "webgpu:vulkan-resident (native)"` — a genuinely different code path from int4) still diverges,
  but with a *different* pattern: worse at position 0 (cosine 0.358, vs int4's 0.988) but greedy
  generation stays correct for 4 steps before diverging (vs int4's divergence at step 0-1). Two
  different wrong patterns across two different precision code paths suggests either two
  compounding bugs, or one bug whose visibility depends on precision-path timing/ordering — not a
  single simple off-by-one that a naive "make it match" patch would fix blind.
- **Not the fused-QKV weight split.** `decoder/weights.go`'s `buildPhi3Weights` splits
  `self_attn.qkv_proj.weight` into Q/K/V by fixed, index-based row ranges (`qkv[0:qDim*hidden]`,
  `qkv[qDim*hidden:(qDim+kvDim)*hidden]`, `qkv[(qDim+kvDim)*hidden:(qDim+2*kvDim)*hidden]`) with no
  size-dependent branching. This is **shared** CPU/GPU code, and CPU is correct — if this split
  were wrong, CPU would be wrong too.
- **Not the GQA head-mapping arithmetic.** `group := nH / nKV` and `kvh := qh / p.group`
  (`gpu/decodelayer.go:146`, `gpu/attention.go:118` and its three other copies) both evaluate
  correctly for phi3's `group = 32/32 = 1` (every query head maps to its own KV head — the
  identity case, not a divide-by-zero or off-by-one).
- **Not partial-rotary tail handling.** `qkvFinalizeShaderWGSL`'s `ktail` pass-through block (for
  GLM/some-Phi partial rotary) is a documented no-op for full rotary; phi3-mini-4k's own GGUF
  metadata confirms `rope.dimension_count: 96 == head_count * embedding_length/head_count` — full
  rotary, `ktail = 0`. This code path never fires for this checkpoint.
- **Softmax/multi-key aggregation is not implicated by position 0.** At position 0 there is
  exactly one cached key, so softmax over one element is 1.0 regardless of whether the attention
  weights are computed correctly (`CLAUDE.md`'s own documented minimal-repro trap, in reverse
  here: position 0 does NOT mask this bug — real divergence is present even where softmax is
  inert). Whatever's wrong runs before or independent of the multi-key softmax math.
- **RoPE rotation is inert at position 0 too** (`theta = pos * invFreq = 0`, so `cos=1, sin=0`,
  an identity transform) — yet position-0 divergence is still present in the f32/native path
  (cosine 0.358). So the bug is not purely "RoPE angle is wrong at pos > 0" either, though a
  RoPE-table or `AttnScale`/`RotaryDimResident` mismatch specific to hd=96 has not been fully
  ruled out for the int4 path specifically (only shown to be non-exclusive as the position-0
  cause in f32).

## What's still open — where to look next

Not found: the actual line. The likely remaining suspects, roughly in order of how cheap they'd
be to check with proper instrumentation:

1. **The int4 GEMV/dequant kernel for phi3's specific weight shapes.** `hidden=3072`, Q/K/V/O all
   `[3072, 3072]` (unlike every other tested model, where Q is strictly wider than K/V) — check
   whether the int4 quantization group layout or the dequant kernel has an assumption keyed on
   relative Q/K/V sizes.
2. **The attention kernel (`attnShaderWGSL`) at `group=1` specifically**, even though the
   arithmetic reads correctly on paper — build a *pure* kernel-level unit test (no full model)
   with `nH == nKV` and a known Go-computed reference, the way `attnbatched_test.go` already does
   for `(nH,nKV,hd) = (8,4,64)` and `(4,1,50)` but never `(n,n,h)`.
3. **O-projection / MLP**, downstream of attention — not yet isolated at all.
4. **Build the per-layer hidden-state capture this class of bug needs.** This repo's own
   CLAUDE.md is explicit that this class of divergence should be chased "per layer, not from
   final logits" — that tooling does not exist for the WebGPU backend today (it does, in a
   test-only form, for CUDA's MLA work — `cuda/mla_resident_test.go`'s per-position cosine check
   is the shape to copy). Building it properly (dump Q/K/V/attn-output/o-proj-output per layer,
   not just the final 32-layer-deep logits) is very likely the fastest real path to the actual
   line, not more WGSL reading.

## The interim fix, and its blast radius

`gpu/residency.go`, two independent changes:

1. **`ctxCap` clamps to `m.Config().MaxPositions`** before the ceiling/explicit-request clamps.
   Unconditionally beneficial — stops wasting VRAM reserving positions no model could ever serve.
   Verified inert for models whose native window already exceeds the ceiling (Qwen2.5-1.5B:
   32768-token window, ctxCap stays at the 16384 ceiling as before).
2. **`BuildResident` declines when `nKV == nH` (no GQA at all)**, SCOPED to exclude MLA
   (DeepSeek/Kimi), Qwen3.5/DeltaNet and Nemotron — those take entirely separate attention code
   paths further down this function and their own `nKV`/`nH` can coincide for reasons unrelated
   to GQA grouping (`deepseek-tiny` reports `nKV==nH==4`; an unscoped first version of this guard
   broke `TestMLAResidency_matchesCPU`, caught by the full `gpu` test suite before it shipped).

Verified on real hardware (RTX 2070 SUPER): phi3-mini now declines cleanly with an honest reason
(`no-GQA attention (kv heads == query heads, 32 == 32) hits an unresolved WebGPU resident decode
divergence`) instead of either the old misleading OOM message or silently-wrong fast output.
Qwen2.5-1.5B (real GQA) still goes resident and generates correctly, unaffected. Full `gpu` module
test suite (`go test -tags 'gpu goinfer_testhooks' .`, non-heavy): 123 pass, 0 fail — includes
Gemma3/gpt-oss/Nemotron-MoE/LoRA/QKNorm resident-parity gates at their existing 0.999+ floors and
the MLA parity test.

**What this does NOT fix**: phi3-mini and Phi-4 (and any other future no-GQA checkpoint) stay on
the CPU-staged WebGPU path — slow, but correct — until whoever picks this up finds the real bug
and removes the `nKV == nH` guard. `docs/hardware-matrix.md`'s "✅ resident" for Phi-3/Phi-4 on
WebGPU is unaffected by this change (it reflects feature *admission*, checked with no device
present, not a real `BuildResident` attempt — the same admission/runtime-fit distinction this
repo's MLA nGroup/topkGroup trap already documents at `decoder/features.go`).

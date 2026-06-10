# Task (goinfer): batched on-device GPU prefill (long-prompt TTFT)

> **For:** Claude Code, in `~/tmcode/goinfer` (GPU work → the 64 GB RTX box;
> `-tags gpu`). Deferred follow-on from `docs/roadmap.md`. Increments ordered and
> independently shippable. **Bit-exact greedy parity is the non-negotiable gate —
> none of this touches the CPU forward; it must match the sequential GPU prefill
> token-for-token.** Pure-Go core CI job stays untouched.
>
> **⚠️ GATED — do not build yet (measured 2026-06-09, RTX 2070 SUPER + M1 Pro).**
> The whole premise — amortize the 91% VRAM weight-read with a compute-bound tiled
> GEMM — fails on current hardware because the WGSL tiled GEMM has no `dot4I8Packed`
> and tops out at 680 GFLOP/s (RTX), *below* the bandwidth-bound M=1 GEMV's
> 748 GFLOP/s-equiv → batched prefill ≈ 0.91× (RTX), ≈1.2× (Metal): a wash. It's
> kernel-limited, not silicon-limited. **Prerequisite: `dot4I8Packed` unblocks in
> `cogentcore/webgpu`** (TU104 has the DP4A hardware) — only then does the tiled GEMM
> clear the bandwidth wall and these increments pay off. See `docs/roadmap.md`
> (Backlog → GPU long-context, and the dp4a item).

## Problem

The full-residency GPU path (`gpu.DecodeRunner`, weights resident, one fence per
token — 7B int4 51.7 tok/s, 1.5B 102 tok/s) has **no batched prefill**. Prefill is
**option (a)**: loop `DecodeRunner.Run(x, pos)` once per prompt token to fill the
GPU KV cache (also warms the pipelines), last token's logits seed decode. That's
**O(prompt-len) one-fence Runs** — fast for short prompts (7B TTFT 1.3 s, 1.5B
0.66 s) but **linear**: a 1 k-token prompt ≈ 1000 × ~18 ms ≈ **~18 s** before the
first output token (RAG context, large system prompts, pasted docs).

Fix: process all M = len(prompt) positions in **one (or a few) on-device passes**,
streaming each resident weight once and reusing it across the M rows — sub-linear
TTFT. The CPU analog already ships (`prefillLogits`/`forwardLayersN`, ~1.7–2×).

## What already exists (building blocks)

- **Tiled M>1 GEMM** — `gpu/gemm.go`: `BatchTiled` (batched, shared activation →
  one submit) and `MatmulW8A8Tiled`; compute/traffic-bound at M>1 (each weight
  streamed once across the M rows). W8A8 **and** W4A8 both have it. This is the
  projection matmul; the decode path's `gemv` is the M=1 sibling.
- **The plan-builder pattern** — `newDecodeRunner` (`gpu/decoderunner.go`) records
  a fixed dispatch graph against persistent buffers. Mirror it at M.
- **Per-row kernels** (M=1 today, generalize to M rows): `rmsQuant`, `swigluQuant`,
  `quant`, `rope`, `ropeStore` (rotate-K-into-cache), `vStore`, `gemvAdd` (residual
  epilogue). Each currently dispatches `(1,1)` / per-element for one row.
- **Residency eligibility** — `decoder/residency.go` `DecodeRunnerEligible` (dense
  Qwen2/Llama; not MoE / Gemma4 / qwen3_5). The new path inherits this gate.

## The one new kernel: batched causal attention

Today's attention (`gpu/attention.go` `attnShaderWGSL`, `c.attnPipeline`) is
**single-query**: one query position, `nH` workgroups, attends to all cached keys
`[0, pos]`. Prefill needs **M queries**, each query i (abs pos `startPos+i`)
attending causally to keys `[0, startPos+i]` — a **plain causal mask, no sliding
window.** `DecodeRunnerEligible` requires `SlidingWindow == 0` (verified,
`residency.go:54`), so the residency path is **full-attention-only** by
construction — there are no local/windowed layers to handle. K/V for all M are
written into the resident cache by the batched `ropeStore`/`vStore` *before* the
attention reads it, so it's a self-contained pass. This kernel doesn't exist yet,
but with no per-row window it's a straight causal mask, not M distinct windowed
masks.

## Increments

### Increment 1 — batched causal attention kernel (do first)
WGSL: `[M, nH, hd]` queries × the resident `[nKeys, nKV*hd]` K/V cache, **plain
causal mask** per row (query i attends to keys `[0, startPos+i]`), GQA broadcast
(`nH/nKV`). Workgroup-per-(query, head) or tiled. Add `ensureAttnBatched` + the
pipeline.
- [ ] **Gate:** bit-exact vs M sequential single-query `attn` dispatches over the
      same cache (clone `TestAttnBlock_parity`, software-adapter-skipped, run on
      real HW). Both oracles read a **fully pre-populated cache** — write all M K/V
      first, then attend; the sequential oracle must read the *same complete* cache
      (causal-masked per query), not rebuild it incrementally, or it isn't
      apples-to-apples. (No sliding-window case — the path is full-attention-only;
      the off-by-one to watch is the causal bound `j ≤ startPos+i`.)

### Increment 2 — the prefill runner (the M-sized plan)
A `PrefillRunner` (or a `DecodeRunner` M-mode), **built per-prompt** at M (prefill
is once per request, so the per-call buffer alloc amortizes over the big pass —
unlike per-token decode, which is why `DecodeRunner` is persistent). It **shares
the resident model's weights + KV caches** with the decode runner, so after
prefill the decode `Run` continues from `pos = M`. Per layer, mirror
`newDecodeRunner` but at M:
- `xd` → `[M*hidden]`; per-row activation scale `[M]`.
- `rmsQuant`/`swigluQuant`/`quant` dispatch **M workgroups** (one row each).
- projections via the **tiled GEMM** (`BatchTiled`) instead of `gemv`.
- `ropeStore`/`vStore` write **all M positions** into the cache (`pos = startPos+i`).
- the **Increment-1 batched attention**.
- **LM head on the LAST row only** (`h[M-1]` → norm → head) — the other rows'
  logits aren't needed (matches `prefillLogits`); avoids the M×vocab matmul.
- one Submit (or a few; see chunking).
- [ ] **Gate:** bit-exact vs M sequential `DecodeRunner.Run` calls — same KV cache
      contents and same `h[M-1]` logits. Run on real HW.

### Increment 3 — wire into `decoder.Generate`
In the residency prefill (`decoder/model.go` ~338 / `residency.go`), when
`len(prompt)` exceeds a small threshold use the `PrefillRunner`; else keep the
per-token `Run` loop (and as the fallback). Continue decode from `pos = M`.
- [ ] **Gate:** `TestDecodeParity`-class greedy continuation **unchanged**;
      long-prompt TTFT measurement shows sub-linear scaling vs option (a)
      (e.g. 1 k-token prompt: ~18 s → target a few × the one-pass cost, not 1000×).

## Memory & chunking (note, don't over-build)

The M-sized scratch is transient (freed after prefill) but real: `gate`/`up` are
`[M*inter]` (7B: 1 k × ~18 k × 4 B ≈ 72 MB each). Fine for typical prompts. For
very long prompts, **chunk** the prefill into blocks of e.g. 256 — each block is a
batched pass attending to the growing cache — keeping scratch bounded while still
amortizing the weight stream. Start unchunked; add chunking only if a real prompt
length needs it. **When chunking lands: a query in block 2 attends causally over
ALL keys written so far (blocks 1–2), not a block-relative range** — the causal
bound is the query's absolute position, so each block attends over the whole
growing cache.

## Scope / constraints

- **Residency-eligible archs only** (dense Qwen2/Llama, `SlidingWindow == 0` ⇒
  **full-attention-only**). Staged/CPU path, Gemma 4, qwen3_5_moe keep their
  existing prefill (Gemma 4 / qwen3_5 are CPU-orchestrated; the CPU `prefillLogits`
  already batches the dense ones).
- **W8A8 and W4A8** (both have tiled GEMMs).
- **Bit-exact** with the sequential GPU prefill — greedy decode parity must not
  move. The batched attention is the place a subtle **causal-mask off-by-one**
  hides (there's no window to get wrong); the Increment-1 gate is the guard.

## Why deferred / when to pick up

Typical single-shot prompts already prefill in ~0.7–1.3 s, so there's no felt pain
*today*, and this is the **least urgent** open GPU follow-on — keep it behind
release-gating work (e.g. the qwen3.6 GGUF loader) until the trigger below fires.

**The concrete trigger — and it's sharper than "long prompts":** the residency
path is **stateless in v1** (no `Session` / prefix-reuse — that was the documented
W4A8 limit; `UploadKV` is the kept bridge toward fixing it). So a **multi-turn or
RAG user on the residency path re-prefills the *entire growing history every
turn*** at O(len) — there is **no warm-KV escape hatch** there like the staged
path's `sessionLRU` has. That makes long-prompt prefill bite *exactly* the
7B-on-8GB residency users the W4A8 work targeted, harder than "no felt pain"
implies. **Pick-up signal: someone runs multi-turn chat or RAG on the GPU
residency path** — at that point this jumps the queue (and pairs with either
residency prefix-reuse via `UploadKV`, or the f16-KV item for the long context
those workloads imply).

## Definition of done

- [ ] Increments 1–3 landed, each with its bit-exact gate on real hardware.
- [ ] Long-prompt TTFT measurement recorded (option (a) vs batched) in the GPU
      campaign doc / CHANGELOG.
- [ ] `TestDecodeParity` + the GPU parity gates green; software-adapter CI still
      skips the hardware-sensitive ones (no CI regression).

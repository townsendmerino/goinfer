# Per-group activation scales for W4A8 / W8A8 (2026-09)

> **Status 2026-09-25: planned, not started.** Owner decision 2026-09-25: this is the fix for
> `queue-engineering.md` H2. The guard shipped first (`dcbbaa91`); this lifts it.

## Why

Every quantized projection in goinfer (int4 = W4A8, int8int8 = W8A8, int4mix, and every resident GPU
projection) quantizes its f32 input to int8 with **one max/127 scale per activation row**. A family whose
projection inputs carry massive outliers loses nearly the whole row to rounding:

- Phi-3-mini's `down_proj` input has max/rms ≈ 80–90 in most layers at a filler position (layer 4:
  max 563). One per-row scale rounds 99.9% of the row to zero.
- goinfer's own f32 matches HF exactly (cosine 1.000000 at all 129 positions). int4 and int8int8 fall
  to cosine ≈ 0 by position 16.
- Simulated on HF, per-32 activation scales restore cosine 0.9998 (median) / 0.980 (min) with f32
  weights. The full record is H2.

llama.cpp quantizes activations per block (Q8_0 / Q8_K), which is why its Q4 output stays coherent on
the same weights.

**Blast radius beyond Phi-3: activation damage is general, and Phi-3 is the extreme case.** H2 step 1
(2026-09-25, raw data and tool in [`measurements/actquant-sweep-2026-09-25/`](../measurements/actquant-sweep-2026-09-25/)).

**Method.** goinfer on CPU, each quant against goinfer's own f32 on the same checkpoint, per position, on
two chat-templated prompts of ~130 tokens: the essay-v2 filler, and a natural prose paragraph. `int8`
(weight-only) against `int8int8` differs ONLY in activation quantization. Cells are the 10th-percentile
logit cosine over positions, and the first position where cosine drops below 0.9 (−1 = never).

| family | int8 (w-only) p10 | int8int8 p10 filler / prose | int4 p10 filler / prose | first < 0.9 (int4) |
|---|---|---|---|---|
| phi3-mini | ≥ 0.97 | **0.322 / 0.958** | **0.236 / 0.944** | 0 |
| qwen2.5-7b | ≥ 0.99 | **0.833** / 0.972 | **0.152** / 0.942 | 2 |
| olmo3-7b | ≥ 0.98 | 0.976 / 0.968 | **0.798** / 0.936 | 2 |
| llama3.2-1b | ≥ 0.98 | 0.965 / 0.974 | **0.704 / 0.729** | 1 |
| smollm3-3b | ≥ 0.98 | 0.993 / 0.989 | 0.965 / **0.847** | 3–4 |
| qwen3-1.7b | ≥ 0.97 | 0.992 / 0.980 | 0.969 / **0.890** | 0 |
| granite-4.2-3b | ≥ 0.99 | 0.991 / 0.987 | 0.976 / 0.950 | 0 |
| qwen3.5-0.8b | ≥ 0.99 | 0.995 / 0.994 | 0.962 / 0.954 | 0 |
| gemma3-1b | ≥ 0.99 | 0.997 / 0.995 | 0.983 / 0.977 | 1 |
| tinyllama-1.1b | ≥ 0.99 | 0.981 / 0.998 | 0.968 / 0.990 | 0 |
| qwen2.5-coder-1.5b | ≥ 0.99 | 0.999 / 0.998 | 0.994 / 0.991 | 0 |
| mistral-7b | 1.000 | 0.999 / 0.999 | 0.995 / 0.997 | never |

Not measured: lfm2.5 (its own-forward layers have no logit-capture seam). olmo3-7b's int8int8 minimum
(−0.14) sits at position 2; int4 widens it on filler. Three groups:

- **Damage confined to the first token or two** (p10 ≥ 0.95; the low minimum sits at position 0, the
  "attention sink" token whose activations are the classic massive outliers): most families here.
- **Widespread damage from activation quantization** (int8int8 itself falls): **phi3-mini** and
  **qwen2.5-7b** (int8int8 p10 0.83 on filler, and int4 p10 0.15). This is what per-group activation
  scales fix, and qwen2.5-7b is one of the peer-benchmark models.
- **Widespread damage from the int4 WEIGHTS, not the activations** (int8int8 fine, int4 not):
  **llama3.2-1b**, and on prose smollm3-3b and qwen3-1.7b. Per-group activation scales will not fix
  these. goinfer's int4 is symmetric g32, re-quantized from an already-Q4 GGUF (a double quantization),
  so it is a separate lever: int4 weight quality, with mins or k-quant style scales. It is out of scope
  here and should get its own queue entry.

Weight-only int8 holds everywhere, which is why the Phi-3 guard is safe. Mistral-7b is clean at every
precision.

## The change

- **Scale granularity:** one scale per 32 input elements, matching `int4GroupSize`. The scale buffer
  grows from M to M×K/32 floats.
- **W4A8:** every kernel already turns each 32-element group (or each 8-element word) into f32 using the
  weight's group scale. The multiply becomes `wScale[g]·aScale[g]`, and the final `·aScale` goes away.
  This is cheap.
- **W8A8:** every kernel accumulates the whole row in one int32. It needs per-group int32 partials,
  each scaled, which restructures the inner loop.
- **Keep a per-row API** for callers outside goinfer's forward: aikit `ann` FlatI8, anncuda/annmetal,
  and the ViT kernels.

## Where (survey of 2026-09-25; paths as of that date)

| area | quantizers | W8A8 (restructure) | W4A8 (cheap) | difficulty |
|---|---|---|---|---|
| aikit CPU amd64 | Go (`linalg/quant.go`) | dotI8 AVX2/VNNI, Tile4x1 asm | fold AVX2/VNNI, tile, splithalf asm | hard (asm) |
| aikit CPU arm64 | Go + NEON asm | dotI8 NEON/SDOT, Tile4x4 asm | ~12 SDOT kernels + tiles | hard (asm) |
| aikit CUDA (`gpu/gemv_quant.cu`) | none | gemv_w8a8_fwd | gemv_w4a8_fwd | medium (PTX regen) |
| goinfer CUDA | 10 kernels + 4 fused | gemv_w8a8_batched | rn, mma, moe×3, fused×4 | medium–hard (frozen PTX) |
| Metal | 8 kernels | 3 | ~13 | easy–medium |
| WebGPU | host + 6 shaders | 9 | 3 | easy–medium |

## Traps named up front

1. **The zero-scale fast path.** The CPU W8A8/W4A8 spans skip a whole row when `aScales[i] == 0`. A
   zero *group* is not a zero row. The backends also disagree on the all-zero scale today: 0 on CPU and
   CUDA, 1 on Metal and WebGPU (`QuantizeRowsInt8`).
2. **Pairs must stay bit-identical.** Decode and batched kernels are pinned bit-identical in pairs:
   aikit `gemv_w4a8_fwd` ↔ goinfer `gemv_w4a8_rn`, and `gemv_w8a8_fwd` ↔ `gemv_w8a8_batched`. Each
   pair must change together, with every new FMA written explicitly (`TestKernelFMALint`).
3. **Frozen PTX.**
   - `glue.ptx` and `moe.ptx` are audited 12.6.85 artifacts: regenerate them with the pinned NVRTC venv.
   - The rest were built with 12.9.86.
   - Gates: `TestPTX_matchesSourcesAndBindings`, aikit `TestPTXReproducible`.
4. **The 4-row asm tiles** (W4A8, both arches) share one weight-scale broadcast across four activation
   rows. They need four activation-scale broadcasts per group, in the hottest loops.
5. **`gemm_w4a8_mma`** applies the activation scale in its epilogue. It needs a shared-memory scale tile
   of one scale per row per group, for each KSTEP.
6. **Every Go caller that sizes `aSc` / `aScales`** as 1 or M must be resized: CUDA bindings, the Metal
   and WebGPU callers, the LoRA hooks, and the host dequant in `cuda/prefill.go`.
7. **Every precision golden re-baselines:**
   - `testdata/int4_forward_goldens.json` (+ `scripts/refresh_parity_hashes.sh`);
   - `testdata/metal_snapshot_golden.json`;
   - CUDA's CPU-oracle tests;
   - aikit's quant/tile tests.

   Re-baselining is only legitimate with the mechanism written down, which this doc is.

## Order of work

1. **Measure first:** the H2 step-1 sweep sizes the problem across families. A W8A8 perf baseline per
   backend (bench_compare.sh for goinfer-vs-goinfer) is taken before any kernel change.
2. **aikit CPU, Go reference first:** per-group quantizer + Go W4A8/W8A8 spans, behind a flag. Measure
   quality (the sweep, re-run) and CPU speed. Kill threshold: pre-register it before building the asm.
3. **aikit asm, amd64 then arm64:** W4A8 first (cheap), then the W8A8 per-group partial kernels.
4. **CUDA:** the W4A8 family (rn, mma, moe, fused), then the W8A8 batched kernel; PTX regenerated; the
   aikit/goinfer pairs in lockstep.
5. **Metal, then WebGPU.**
6. **Re-gate every family**, re-run the peer sweep, and lift the Phi-3 guard only after a long-prompt
   quality gate passes at int4.

<!-- doc-reviewed: 2026-09-25 -->

# Task: int4 weight quality: stop re-quantizing Q4_K (2026-09-26)

**Owner decision 2026-09-26:** this goes ahead of R6 phase 3, the f16 resident KV cache, which is parked
(`docs/prompts/cuda-r6-flash-decode.md`). The reasons are in `docs/measurements/f16kv-baseline-2026-09-26.md`:
Phi-3 on CUDA is 0.70× the peers at depth 128 and 0.68× at 2048. That gap comes from its **int8
weights**, the only safe setting after H2, not from the KV cache.

## Why

H2's per-32 activation scales fixed activation quantization
(`docs/tasks/task-actquant-pergroup-2026-09.md`). At int8int8 + per-32, phi3-mini scores p10 0.973 /
0.998 and qwen2.5-7b 0.991 / 0.998 against f32. At int4 + per-32 the same models score 0.766 / 0.970
and 0.307 / 0.958. The difference is int4 **weight** error, and Track B showed that neither symmetric
rule recovers it (both candidates FAIL).

The mechanism is double quantization. `buildWeightsFromGGUF` (`decoder/gguf.go`) dequantizes every GGUF
tensor to f32 and then re-quantizes it to goinfer's symmetric int4, which decodes as
`(nibble−8)·scale` with `scale = max|w|/7`. A Q4_K tensor is already 4-bit, and **asymmetric**: per
32-element sub-block, `w = d·sc·q − dmin·m` with `q ∈ [0, 15]`. Its values sit on a grid that
symmetric int4 cannot represent, so the transcode adds a second rounding on top of the file's own.
The quality reference in every sweep here is goinfer's f32 from the same GGUF, which means the
dequantized Q4_K values themselves. **A layout that stores Q4_K's own per-sub-block scale and
minimum reproduces those weights exactly.**

## What the files contain (layer matmuls, from the GGUF headers)

| model | Q4_K | Q5_K | Q6_K | other |
|---|---|---|---|---|
| phi3-mini | 64% | 25% | 11% | |
| qwen2.5-7b, qwen2.5-coder-1.5b, llama3.2-1b, mistral-7b | ~85% | | ~15% | |
| gemma3-1b | 19% | | 15% | 65% Q5_0, <1% Q8_0 |

Only the Q4_K share can be carried exactly in 4 bits. The rest is the design question: Q5_K/Q6_K kept
near-exact at int8 (8 bits a weight) or re-quantized to 4 bits (lossy).

## Bytes per decoded token, layer matmuls (why the split matters)

At the measured ~330 GB/s, decode is weight-byte-bound at shallow depth, so bytes ≈ speed:

| model | today's int4 (4.5 bpw) | Q4_K exact + rest at int8 | everything 4-bit at 4.5 bpw | today's int8int8 |
|---|---|---|---|---|
| phi3-mini | 2.04 GB | 2.61 GB (+28%) | 2.04 GB | 3.62 GB |
| qwen2.5-7b | 3.67 GB | 4.10 GB (+12%) | 3.67 GB | 6.53 GB |

"Q4_K exact" at 4.5 bpw means keeping Q4_K's own super-block layout (f16 `d`/`dmin`, 6-bit sub-block
scales and minimums), which is the bytes goinfer's int4 already spends. A plain affine layout with an
f16 scale and f16 offset per 32 costs 5.0 bpw.

## Phase 0: the quality experiment (no kernels), PRE-REGISTERED 2026-09-26 before any run

**Instrument.** The H2 sweep tool (`docs/measurements/actquant-sweep-g32-2026-09-25/actsweep.go.txt`),
unchanged in its metric: p10 logit cosine against goinfer's f32 on the filler and prose prompts, with
per-32 activations on. A test-hook seam in the GGUF loader fake-quantizes each layer weight row by its
source GGUF type. Each arm is then loaded at **int8int8** with per-32 activations, so weights ride
the int8 carrier the int8int8 cells already use. The carrier adds the same small int8 error to every
arm, including the reference arm A0. Embeddings and the LM head are untouched in every arm.

**Families:** the six GGUF sources: phi3-mini, qwen2.5-coder-1.5b, qwen2.5-7b, gemma3-1b, llama3.2-1b,
mistral-7b. bf16-sourced families have no Q4_K to preserve and are out of scope for Phase 0.

**Arms, all same session:**
- **C0**: today's int4 with per-32, the regression baseline;
- **A0**: carrier only, no fake-quant. Represents "Q4_K exact, every other tensor near-exact";
- **A2**: Q4_K rows unchanged; Q5_K/Q6_K/Q5_0/Q8_0 rows → affine int4, per-32 min/max
  (`s = (max−min)/15`, codes 0..15);
- **A3**: Q4_K rows unchanged; the other rows → goinfer's symmetric int4 (`max/7`, per 32);
- **A4**: every row → affine int4 per-32 min/max, Q4_K included. Tests whether *exactness* matters or
  a generic asymmetric quantizer is enough. It is also the only candidate for bf16 sources.

**Bar, Track B's, reused unchanged:** PASS = phi3-mini and qwen2.5-7b p10 ≥ 0.90 on both prompts, AND no
family's p10 more than 0.005 below its own C0. AMBIGUOUS → parked = the lower of those two models'
filler p10 in [0.80, 0.90) with no such regression. Everything else FAILs.

**Decision rule, fixed now:**
1. If A3 passes, build native Q4_K plus symmetric int4 for the rest: **4.5 bpw, today's bytes**.
2. Else if A2 passes, build native Q4_K plus affine int4 for the rest (4.5 to 5.0 bpw).
3. Else if A0 passes, build native Q4_K plus int8 for the rest, and state the byte cost from the table
   above in the build doc.
4. If A4 also passes whichever arm decided, and its minimum over (phi3-mini, qwen2.5-7b) × (filler,
   prose) is within 0.01 of that arm's, **prefer a generic affine quantizer over a native Q4_K layout**.
   It is simpler, and it serves bf16 sources too.
5. If nothing passes, kill: int4 stays guarded for Phi-3, as today.

A arm that crashes or errors is reported void, not guessed.

## Phase 1+ (after Phase 0's record; not yet scoped)

Format and kernels per the decision: aikit `WeightMat`, the CPU W4A8 kernels (amd64/arm64), CUDA
`gemv_w4a8`, then Metal/WebGPU. Then the quality gate on the real path, and the speed gate against
peers on the Phi-3 and 7B cells.

<!-- doc-reviewed: 2026-09-26 -->

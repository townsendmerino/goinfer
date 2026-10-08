# `--quant q4k` on Metal: Phi-3 mini resident (2026-10)

**Status: REGISTERED 2026-10-08, before any code.** Owner decision 2026-10-08: "for Phi-3, Mac Metal lets do the code fix
now".

## Why

The goinfer.dev/models Phi-3 cell on Metal is blocked (`docs/measurements/peer-vetted-2026-10-07-macbook.md`, M1).

- **The decline:** at `--quant q4k`, Metal declines residency (`decoder/residency.go`: "--quant q4k (native Q4_K) has no
  resident kernel on metal yet"), and only CUDA has a Q4_K kernel.
- **No workaround through other quants:** at int8int8, Phi-3 needs per-32 activation scales, which Metal does not have
  either, so it declines there too (`peer-refresh-2026-10-06.md`).
- **Why the activations matter:** per-row int8 activations round Phi-3's activation outliers to zero
  (`queue-engineering.md` H2). So a Metal Phi-3 path has to keep its activations in float, or add per-32 activation
  quantization.

## What q4k loads for Phi-3 (the vetted `Phi-3-mini-4k-instruct-q4.gguf`)

| tensor | GGUF type | goinfer at `--quant q4k` |
|---|---|---|
| attn_output, ffn_up (gate‖up), token_embd | Q4_K | native Q4_K, the super-blocks verbatim |
| ffn_down, 16 of 32 layers | Q4_K | native Q4_K |
| ffn_down, the other 16 | Q6_K | int8 |
| attn_qkv | Q5_K | int8 |
| output (LM head) | Q6_K | int8 |

## The design: a float-activation lane

The experimental W4F16 lane (`GOINFER_METAL_DECODE_LANE=w4f16`, R1) is the shape: an f16 activation from the norm,
no quantization, into each projection.

- **The q4k lane extends it to every projection,** for any model loaded at q4k on Metal:
  - **New kernels:**
    - a Q4_K GEMV over f16 activations, which decodes the 6-bit scales and mins and the 4-bit quants per super-block of
      256, with f32 accumulation; plus its bias and residual variants;
    - an int8-weight GEMV over f16 activations (per-row weight scale) for the int8 tensors, with the same variants;
    - SwiGLU writing f16 for the down projection;
    - the LM head on the int8 kernel.
  - **The down projection** dispatches per layer on its kind.
  - **Prefill:** the batched kernels do not take these kinds, so prefill runs decode's rows, as Phi-3's q4k prefill does
    on CUDA (sequential, disclosed in the 2026-09-29 sweep). A decode rate is unaffected.
  - **Arithmetic:** float activations against the CPU's per-32 int8 activations, so it is closer to f32 than the CPU path
    is, and not bit-identical to it.

## Gates, registered before any code

- **G-Q1, the kernels:** each new GEMV against a float64 reference, which dequantizes the same blocks on the host
  (`linalg`'s Q4_K and int8 dequant) and dots with the same f16-rounded activations.
  - **Shapes:** the real Phi-3 tensors (3072x3072, 9216x3072, 16384x3072, 3072x8192, 32064x3072) and synthetic ones
    (one super-block; K a multiple of 256; random scales and mins at the 6-bit range ends).
  - **The bar:** max relative error 1e-5 of each row's norm (f32 accumulation order only).
  - **Planted defects, each red:**
    1. the min term dropped;
    2. the 6-bit scale and min unpack swapped;
    3. a super-block offset of one;
    4. the int8 row scale dropped.
- **G-Q2, the model on the real file (Mac, Metal):** Phi-3 resident at q4k against the CPU at q4k on the same file, in
  S1's G3 shape: teacher-forced argmax agreement over G3's prompts, against the validated Qwen2.5-Coder-1.5B's Metal
  agreement measured in the same process.
  - **PASS:** at least the reference's agreement minus 2.0 points, with free-run passes at least the reference's minus 1.
  - **Ambiguous (parked):** 2.0-4.0 points below.
  - **FAIL:** worse.
  - **Also reported, not graded:** last-row logits cosine per prompt.
  - **The decode path must read `metal-resident (q4k)`.** Anything else voids the reading.
- **G-Q3, served:** one greedy chat request (the peer harness's depth-128 prompt), `--backend metal --quant q4k` against
  `--backend cpu --quant q4k`. Identical reply, or a first divergence at a near-tie (the reference's p(other) at least
  half its top).
- **Then the peer cell M1**, at night, as the Mac record registered it (q4k, depth 128, greedy, 3 runs, against Ollama
  `p3m-local`), by an amendment to that record.

## Not in scope

- Per-32 activation quantization on Metal: the CUDA design. Float activations make it unnecessary for decode.
- A batched Q4_K prefill kernel.
- WebGPU.

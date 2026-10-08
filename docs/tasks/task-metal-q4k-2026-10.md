# `--quant q4k` on Metal: Phi-3 mini resident (2026-10)

**Status: the decode lane is BUILT 2026-10-08; G-Q1 PASS; G-Q2 and G-Q3 owed** (blocked by day on the Mac's memory,
below). Registered 2026-10-08, before any code. Owner decision 2026-10-08: "for Phi-3, Mac Metal lets do the code fix
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

## Amendment 2026-10-08, during implementation: two departures from the design, neither touching a bar

- **f32 activations, not f16.** Every lane projection reads the f32 normed residual directly (`rmsnorm_f32_out`), and
  SwiGLU writes f32 (`swiglu_f32`).
  - The reason: f16 would round each activation to 11 bits and saturate at 65504, and the lane exists because Phi-3's
    activation outliers do not survive rounding. f32 removes the question.
  - The cost is small on a decode GEMV: the activation vector is one row of K floats, read once per output row from
    cache, against a weight matrix N times larger.
  - G-Q1 ran with f32 activations, which makes its float64 reference stricter, not looser (no f16 rounding to absorb).
    Its bar is unchanged.
- **A projection runs as segments of one weight kind.** A q4_k_m file mixes kinds inside a fused group: Qwen2.5's
  attn_v is Q6_K (int8) in half the layers, beside Q4_K attn_q and attn_k.
  - The lane uploads each run of consecutive same-kind parts as its own buffer and dispatches it into its rows of the
    output (`q4kLaneProj`, `q4kGemv` in `metal/model.go`).
  - Phi-3's projections are each a single kind, so it runs one segment per projection, as designed.
  - The bias variants the design lists (`gemv_q4k_f32_bias`, `gemv_w8_f32_bias`) take the combined q/k/v bias, which
    is zeros for Phi-3.
- **What the lane covers, as built:** dense pre-norm attention (q/k/v bias allowed) with a SwiGLU FFN.
  - It declines every family variant (sandwich and post-only norms, parallel blocks, LayerNorm, MoE, DeltaNet, QK-norm,
    attention sinks, per-layer geometry, int8 KV).
  - Prefill runs decode's kernels a token at a time (the f16 MMA prefill is off on the lane).
  - The batched step is off, and LoRA adapters are refused.

### G-Q1 with the bias variants, 2026-10-08

- **Coverage:** `TestQ4KLaneKernels` now checks the plain, residual and bias variants of both GEMVs.
- **Worst error:** 3.74e-7 (Q4_K) and 1.26e-7 (int8) of Σ|w·a|, bar 1e-5: **PASS**.
- **Planted defects:** all four are red, at 6.3e-2, 2.1, 1.5 and 5.1e2.

### By-day wiring smoke (not G-Q2), 2026-10-08

`TestQ4KLaneResidentQwen` (`metal/q4k_resident_test.go`): a Qwen2.5-Coder q4_k_m resident on Metal at q4k, against the
CPU at q4k on the same file, teacher-forced over 32 positions. The bar, set before the first run: every position's
logits cosine at least 0.99 and no NaN, on a decode path reading `metal-resident (q4k)`.

| file | what it exercises | worst cosine | argmax agree |
|---|---|---:|---:|
| 0.5B | Q4_K and int8 ffn_down by layer (Phi-3's pattern), int8 elsewhere, q/k/v bias, int8 head | 0.998985 | 31/32 |
| 1.5B (`GOINFER_HEAVY_TESTS=1`) | also Q4_K QKV, gate\|up and o-proj, and Q4_K + int8 segments inside the fused QKV | 0.997885 | 31/32 |

- **The bar can go red:** with the q/k/v bias dropped (a throwaway edit, reverted), the 0.5B read cosine -0.769 and
  0/32 argmax agreement.
- **What it does not show:** this is CPU parity on a model the lane does not exist for. Phi-3 itself, and the grading
  against the validated reference, are G-Q2.

### G-Q2 and G-Q3: blocked by day on memory

- **The refusal:** the fit guard refuses Phi-3 at q4k on this 16 GB Mac by day. On 2026-10-08 it needed 7.0 GB (about
  3.9 GB of weights, KV at the default context, and the 2.4 GB GGUF mapped during the load) against a 4.1 GB budget.
- **The guard has no `.giw` form to page from at q4k.** `decoder/serialize.go` refuses a q4k sidecar.
- **The two ways forward, which are the owner's call:**
  - run G-Q2/G-Q3 at night with a pinned small context, when the Mac's free memory is the budget;
  - a guarded bypass with the +1 GB swap kill-watch.


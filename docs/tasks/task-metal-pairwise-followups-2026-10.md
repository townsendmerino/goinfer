# Task — the two limits left by Metal pairwise RoPE (2026-10)

**Status:** Parts A and B BUILT 2026-10-09 (results below). Part C not started. Gates were registered before any code
(owner: "lets fix these 2 limits").

The limits, from `docs/tasks/task-metal-pairwise-rope-2026-10.md`:

1. **Metal's batched f16 prefill runs neither family.** Cohere prompts prefill one token at a time; a GLM-OCR image turn
   prefills on the CPU and uploads.
2. **Command-R7B on Metal reads mean cosine 0.989 against the CPU at int4.** CUDA's resident reads 0.997. Metal's f16
   KV cache is a hypothesis, not a finding.

## What is missing, from the code (`metal/model.go`'s `prefillFeatures`, `prefillOK`)

- **GLM-OCR** lacks only `FeatPairwiseRoPE` and `FeatPairwiseMRoPE` in the prefill feature set.
  - The pairwise prefill kernels (`rope_f16_pw`, `rope_mrope_f16_pw`) exist, are bound for a pairwise model, and
    passed G-PR1.
  - Its sandwich norms are already in the pass.
  - Admitting it also builds the m-RoPE axis table (built only when `prefillOK`), so S16's resident image prefill
    reaches it too.
- **Cohere and Cohere2** lack `FeatLayerNorm` (bias-free), `FeatParallelBlock` and `FeatLogitScale` in the pass, and the
  pairwise features.
  - The decode path has all three: `layernorm_quant`, the shared norm, and a host multiply in `finalizeLogits`.
  - The f16 pass has none.

## Part A — GLM-OCR's batched prefill (and S16's image prefill)

- **Change:** add the two pairwise features to `prefillFeatures`.
- **G-A1, tiny** (`TestGlmOcrResidentParityMetal`; glm-ocr-tiny unpeaked, int8int8, the G-PR3 amendment). The paths it
  logged as declined become graded:
  - the text batched prefill, last token, against the CPU per-32;
  - the image block through the resident m-RoPE prefill, against the CPU per-32;
  - decode past the image after it, against the CPU per-row, teacher-forced.

  Bars: cosine >= 0.995 and relL2 <= 0.15 on each path. The NeoX rope rebound into the same resident must read red on
  every path.
- **G-A2, real invoice:** the 1,656-token test invoice through `metal/cmd/chat`, Metal (batched image prefill) against
  `--backend cpu`.
  - The JSON reply is byte-identical, or each difference is a teacher-forced near-tie (3% rule).
  - TTFT is reported, exploratory.
- **G-A3:** `gate quick` green.

### Part A results, 2026-10-09 (by day, the M1 Pro)

- **Found by G-A1's first run, and fixed: a latent bug in the batched pass's admission.**
  - The f16 GEMMs need every reduction length (hidden size, attention output width, FFN width) a multiple of 32.
  - An int4 resident cannot violate that (`int4Buf` refuses at build), but a native int8 one can, and `prefillOK`
    never checked.
  - glm-ocr-tiny (hidden 48, FFN 144) was admitted and read cosine 0.218 (text) and 0.312 (image block) against the
    CPU: garbage, with no error.
  - `resident.prefillUnaligned` now declines such a model to the sequential path with the reason named. Real
    checkpoints are 32-aligned, so none changes path.
- **G-A1 amendment (mechanism above):** glm-ocr-tiny cannot take the pass by its dimensions, so the tiny test asserts
  the decline names the 32-alignment. The batched paths are graded on the real checkpoint instead.
- **G-A1 PASS, real** (`TestPairwiseRoPERealMetal/glm-ocr`, 64 README tokens and a 16-token CPU continuation):
  - batched-prefill last token: cosine 0.996932 against the CPU int4;
  - decode after it: mean 0.997001, worst 0.994574, 0 non-tie flips.

  An exploratory 256-token probe read batched against the CPU per-32 at 0.999841, closer than Metal's sequential path
  (0.997716).
- **G-A2 PASS:** the test invoice's reply through `metal/cmd/chat` is byte-identical to the CPU decoder's. The image
  prefill now runs on the GPU (S16's resident m-RoPE pass).
  - Exploratory timing, two runs: prefill 1.2-1.3 s against about 5.7 s before; decode 95 tok/s; whole invoice
    14-16 s (70 s that morning; 23 s after the GPU tower; about 20 s with the resident decoder).

## Part B — Cohere's batched prefill

- **Change:** three things in the f16 pass:
  - `layernorm_f16`, bias-free and mean-centred, the f16 twin of `layernorm_quant`'s math;
  - the parallel block: one norm feeding both attention and the MLP, with `x += attn + mlp`;
  - the logit scale on the last-token logits.

  Then add the three features to `prefillFeatures`.
- **G-B1, kernel:** `layernorm_f16` against a float64 reference, max abs diff <= 4e-3 on inputs in [-2, 2] with a
  non-zero mean. Control: the same input through `rmsnorm_f16` (no centring) must miss by at least 10x.
- **G-B2, tiny** (`TestPairwiseRoPEResidentParityMetal`; cohere-tiny and cohere2-tiny, peaked, int4). The two paths it
  logged as declined become graded:
  - the batched prefill's last token, against the CPU per-32;
  - decode at [32, 40) after a 32-row batched prefill, against the CPU per-row.

  Bars as G-PR2. NeoX rebound red. A planted defect (the parallel block's MLP fed from the post-attention residual)
  must read red.
- **G-B3, real Command-R7B** (`TestPairwiseRoPERealMetal`, extended):
  - batched-prefill last-token cosine >= 0.98 against the CPU int4 (CUDA's real tier);
  - 16 teacher-forced decode steps after a batched prefill of the 64-token README prompt, with no non-tie flip.
- **G-B4, speed, exploratory by day:** Command-R7B TTFT on about 512 tokens, batched against sequential. It ships if
  batched >= 1.02x; anything below is reported. A served night gate is optional, at the owner's call.
- **G-B5:** `gate quick` green.

### Part B results, 2026-10-09 (by day, the M1 Pro)

- **Change:** `layernorm_f16` and `layernorm_quant_f16` (bias-free); the pre-attention, pre-MLP and final norms swap to
  them for a LayerNorm family; a parallel block's MLP reads the attention's shared input norm; the logit scale is applied
  to the batched pass's logits. FeatLayerNorm, FeatParallelBlock and FeatLogitScale joined `prefillFeatures`. A biased
  LayerNorm (GPT-2), or LayerNorm with QK-norm, is declined by name.
- **G-B1 PASS:** `layernorm_f16` max |diff| 9.45e-4 (tolerance 4e-3); the `rmsnorm_f16` control missed by 1.17. The
  quantized final norm (`layernorm_quant_f16`) landed within half a step (0.00998 against 0.01).
- **G-B2 PASS** (peaked tiny fixtures, int4):

  | | cohere-tiny | cohere2-tiny |
  |---|---|---|
  | batched prefill, last token | cos 0.999021, relL2 0.0443 | 0.999393, 0.0352 |
  | decode after a 32-row batched prefill | worst 0.998191, relL2 0.0609 | worst 0.998660, 0.0519 |
  | NeoX control | batched 0.856, decode 0.437 | batched 0.924, decode 0.833 |
  | planted defect (parallel MLP from the post-attention residual) | batched -0.132 | -0.076 |

- **G-B3 PASS as registered** (real Command-R7B, `gb3-real-r7b.log`; the CPU reference under the owner-approved
  guard bypass, the Metal arm through its sidecar):
  - batched-prefill last token: cosine 0.995699 against the CPU int4 (bar >= 0.98);
  - 16 teacher-forced decode steps after it: 0 non-tie flips.

  The decode after it reads mean 0.985772, worst 0.944647: the same Metal decode gap as the plain path (mean 0.989242,
  owner-accepted). It is logged, not graded, and Part C owns it.
  - The test first also asserted a 0.99 decode mean on the batched path, which was never registered; that assertion
    was removed.
  - The plain path's mean bar for Command-R7B now records the owner's 0.989 acceptance by name and date instead of
    failing every run.
- **G-B4, exploratory** (one run each): Command-R7B prefill of 512 README tokens took 2.52 s batched against 23.40 s
  sequential, 9.3x. Ships (bar 1.02x).

## Part C — the Command-R7B gap (localize first; no fix registered yet)

- **C0, per-layer differencing** at the worst position (13 on the README prompt): the Metal resident's per-layer
  residual against the CPU int4's, to name the first layer and op where the gap opens (CLAUDE.md: difference per layer).
- **C1, the KV hypothesis, measured.** A decoder test seam rounds the CPU's K/V through f16 on store. Then:
  - if that CPU arm's per-position cosine against the CPU at f32 KV falls by about as much as Metal's does, the KV
    precision is the mechanism;
  - if not, it is not.
- **Decision rule:**
  - if C1 shows f16 KV explains at least half the gap (mean 0.997 - 0.989), an f32-KV option for Metal is scoped, with
    its own speed and memory gate (it doubles KV bytes);
  - otherwise C0's layer and op name the next step, written up for the owner before any code.

## Cost

- Part A: an hour, with its gates.
- Part B: a day (three kernels' worth of wiring and their gates).
- Part C: C0 and C1 are about an hour each, by day.

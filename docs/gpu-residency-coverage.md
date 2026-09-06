# GPU resident-decode coverage — what's covered, and the path to the rest

How far the GPU **resident decode runner** reaches across the model zoo, and a
grounded, per-family scoping of what putting each *remaining* family resident
would take. Companion to [`capability-matrix.md`](capability-matrix.md) (the
per-arch `gpu_residency_eligible` column) — this doc is the *why* and the *how*.

Grounded in the live `gpu/` + `decoder/` seam as of the resident **Mamba-2 SSM engine** +
Nemotron-H default-on (commit `d2ba970`, 2026-06-19; the C-lever ladder through C7 is
`6553d72`). No "would be easy" without a code reference.

## What "resident decode" is

The resident runner (`gpu/decoderunner.go`) keeps the whole model on-device and
runs `Forward(embedding, pos) → logits` as one command buffer per token — one
submit, one poll, no CPU interleave. The decoder routes a plain `Generate`
through it whenever `DecodeRunnerEligible()` holds and the webgpu backend's
`BuildResident` accepts the model (`decoder/residency.go`). Anything ineligible
falls back to the staged per-matmul path, gracefully.

The runner expresses ONE uniform per-layer block, in int8 W8A8:

```
input_RMSNorm → q/k/v/o GQA attention (RoPE) → +residual
             → post_RMSNorm → SwiGLU(silu) MLP  OR  sparse MoE → +residual
```

A family is resident-eligible only if every layer collapses onto that block (or
a variant the levers below added). The eligibility predicate
(`Architecture.decodeRunnerEligible`) is the authoritative list of what's
allowed; `BuildResident` is where the per-layer weights are quantized/uploaded.

## The coverage ladder (shipped)

Each lever taught the runner one more shape. All are parity-gated (a GPU forward
vs a CPU int8 oracle at cosine ~1.0, often bit-identical), and several are
additionally validated end-to-end on a real tiny checkpoint loaded through the
normal path.

| Lever | Added | Unlocked | Gate |
|------|-------|----------|------|
| C1 | per-head QK-norm before RoPE | Qwen3 | `4d26637` |
| C3a–c | MoE router top-k + indexed stacked-expert GEMV + combine | Mixtral-class MoE | `4af1db2`/`c20b851`/`ad03c37` |
| C3d | always-on shared expert (sigmoid-gated qwen2_moe / ungated GLM-DeepSeek) | qwen2_moe | `ae7e1fa` |
| C4a–d | **MLA latent attention** (q-LoRA, compressed-KV latent cache, decoupled interleaved RoPE, rank-space absorb attend, W_UK/W_UV lift, group-limited routing) | **DeepSeek-V2/V3, Kimi K2** | `4df70f4`/`88abd21`/`b130f27`/`f96b386` |
| C5 | partial RoPE (rotary_dim < head_dim) | **GLM-4.5/4.6** | `f1bde3b` |
| C6 | sliding-window (local) attention — per-layer windowed start | **Mistral** | `793252b` |
| C7 | per-layer-type RoPE (different invFreq + mscale per global/local layer) | **Mellum** | `6553d72` |
| **SSM** | **resident Mamba-2 decode engine** (`mambaConv`/`mambaSSM`/`mambaGatedNorm` kernels + build-once {conv-ring, ssm} state) + **single-op-per-block** wiring + **squared-ReLU MLP** (`relu2Quant`) + NoPE-via-zeroed-invFreq | **Granite-4.0-H** (opt-in), **Nemotron-H** (DEFAULT-on int4) | `ede17ae`/`f912a25`/`64aa9cc`/`0928611` |

Resident coverage today: **dense Llama / Qwen2 / Qwen2.5 / Phi-3·Phi-4 / Qwen3 ·
Mixtral · Qwen2-MoE · GLM-4.5/4.6 · DeepSeek-V2/V3 · Kimi K2 · Mistral · Mellum ·
Granite-4.0-H (opt-in) · Nemotron-H (default-on int4)** — i.e. the first recurrent/hybrid
SSM families, not just the attention transformers. The SSM engine came from the reframe that
*decode is a bounded per-token recurrence, not the prefill scan*. Nemotron-H is default-on
because, with no MoE router, it quantizes near-losslessly; Granite stays opt-in because its
64-expert router makes int8 a fundamental cliff (`ssm-int8-quality.md`,
`nemotron-resident.md`, `decode-residency-campaign.md`).

Two design notes that recur:

- **Many "blocking" features were already expressible.** C5 (partial RoPE) and
  C6 (sliding window) needed *no new kernel* — the rope kernel already takes a
  `half` independent of `head_dim`, and the attention kernel already honors a
  `start` index. The work was the eligibility relaxation + a posUni wired per
  layer. C7 was similar: a per-distinct-scale rope-uniform cache.
- **The big lever was MLA** (C4), the one genuinely new attention primitive — a
  single-query attend over a shared compressed latent where the score width
  (rank+rope) exceeds the value width (rank). See [`gpu/mla.go`](../gpu/mla.go).

### Hardware reality

The newly-covered families' *real* checkpoints are large: DeepSeek-V2-Lite /
Moonlight ≈ 16B, Mellum2 ≈ 12B. At int8 those exceed the dev box's 8 GB (RTX 2070
SUPER), so the real-model runs are HW-bound here and fall back to staged
gracefully — eligibility is on and the path is parity-proven on tiny checkpoints
(`testdata/deepseek-tiny`, `glm-tiny`). A bigger-VRAM box (or int4 residency for
the experts) is the gate to a full real run.

## The remaining families — per-family scoping

The capability matrix still shows these `gpu_residency_eligible: no`. Grouped by
lift. "New kernel" means a primitive the runner cannot express today; "plumbing"
means new variants of existing ops + eligibility/bridge wiring.

### Small lifts — no new kernel, just feature plumbing

**`gpt2`** (generic forward + GPT-2 specifics). All primitives already exist
elsewhere in the codebase; only the *combination* is new to the resident path:
- **LayerNorm** (mean-centered, with bias) instead of RMSNorm — a norm variant.
- **Learned position embeddings** (`wpe[pos]` added to the input) instead of RoPE.
- **Non-gated GELU MLP** (up → gelu → down, with biases) — non-gated activation.
- Fused QKV / fused gate-up projections + QKV bias + attn-output bias + tied
  embeddings + Conv1D weight layout (all handled at load).
- *Verdict:* small, but low value (GPT-2 is legacy). Would mostly exercise the
  "non-RMSNorm + learned-pos + non-gated-gelu" plumbing.

**`gemma3` / `gemma3_text`** (generic forward + gemma specifics,
`registry.go` `gemma3Architecture`). Sliding window and per-layer dual-base RoPE
are now handled (C6/C7); the remaining deltas are all expressible primitives:
- **`NormSandwich4`** — post-attention and post-FFN norms in addition to the two
  pre-norms (`decoder/registry.go:268`). The runner does Pre2 only; this needs 2
  extra RMSNorm dispatches/weights per layer and a norm-placement branch.
- **GeGLU** (gelu activation in the gated MLP) instead of SwiGLU(silu) — a
  one-line activation variant of the existing `swigluQuant` fuse.
- **Embedding scale** (×√hidden on the input embedding) — the bridge's
  `embedResident` would apply it (eligible archs currently have no scale).
- **query_pre_attn_scalar** attention scale — already representable via the
  runner's `scale` param (like MLA's override).
- *Verdict:* small–medium. No new kernel. The honest blocker count is 3 (sandwich
  norm, GeGLU, embed scale), each individually cheap. Gemma3 is partly superseded
  by Gemma4, so weigh ROI.

### Medium lifts — a few new op variants, no recurrence

**`llama4_text`** (`decoder/forward_llama4.go`). The text decoder shipped on the
staged path (see `llama4-family.md`); residency needs:
- **iRoPE** — per-layer RoPE/NoPE interleave: RoPE layers use *interleaved
  (complex-pair) RoPE* + a parameter-free **L2 QK-norm** (RMS-over-head-dim, no
  weight) applied AFTER rope; NoPE layers skip rope and apply an
  **attention-temperature** tweak to the query (`log1p(floor((pos+1)/floor_scale))
  · attn_scale + 1`).
- **Top-1 sigmoid, input-scaled MoE** — route to one expert by raw sigmoid logit
  (no softmax/group-limit/norm), and scale the expert *input* by the gate
  (`sigmoid(logit)·h`) rather than the output. A variant of the C3 MoE path
  (`llama4MoE`), not the current router kernel.
- Dense/MoE per-layer interleave (already expressible: per-layer `isMoE`).
- *New kernels:* interleaved-complex RoPE, parameter-free L2 norm, input-scaled
  top-1 MoE routing. All small individually; medium in aggregate.

**`gemma4`** (`decoder/forward_gemma4.go`). Per-layer attention geometry is the
theme:
- **Per-layer head_dim and KV-head count** (global 512 / local 256;
  `forward_gemma4.go`), **per-attention-type partial RoPE** (only global layers
  rotate, `GlobalRotaryDim`), **cross-layer KV sharing** (last N layers reuse an
  earlier layer's KV cache), **scale-less V-norm** (RMSNorm without weight),
  **per-layer output scalar**, and on E2B/E4B a **Per-Layer-Embedding** branch
  (gate→gelu→×PLE→proj→norm→+residual) plus variable per-layer FFN width.
- The runner assumes one `hd`/`nKV`/cache shape for all layers; per-layer KV
  widths (see `gemma4-per-layer-kv-widths.md`) break the single-`kvDim` cache
  allocation and attention dispatch.
- *Verdict:* medium — mostly per-layer-geometry bookkeeping + a v-norm variant +
  the PLE branch; no recurrence. The per-layer KV-width generalization is the
  riskiest piece.

### Large lifts — entirely new sequence mixers (state-space / linear attention)

These interleave **non-attention mixers** whose state is a fixed-size recurrent
matrix, not a growing KV cache. The resident runner's whole attention path
(rope-store → windowed softmax attend) does not apply; they need a new on-device
*scan* primitive that is inherently sequential per token.

**`granitemoehybrid` / Granite-4.0-H** (`decoder/forward_granite.go`,
`decoder/mamba2.go`): alternates **Mamba-2 selective-scan** layers with GQA
attention layers, MoE on every layer, plus 3 scalar multipliers
(embedding/attention/residual) and a logits scale. A resident Mamba-2 layer
needs: a depthwise causal conv over `xBC`, the SSM recurrence over an
`[nHeads × headDim × dState]` state (cached, not KV), and a **grouped** gated
RMSNorm. New kernels: Mamba-2 scan + grouped gated norm.

**`nemotron_h`** (`decoder/forward_nemotron.go`): **single-op-per-block** hybrid —
each layer is exactly one of {Mamba-2, NoPE GQA attention, relu² MLP}. Reuses the
Mamba-2 scan (shared with Granite); adds a NoPE attention path (no rope) and a
non-gated relu² activation. Medium-on-top-of-Mamba-2 once the scan exists.

**`qwen3_5` / `qwen3_5_moe`** (`decoder/forward_qwen35.go`, `decoder/deltanet.go`):
most layers are **Gated DeltaNet linear attention** — a depthwise causal conv over
`[q;k;v]`, a recurrent matrix state `S [head_k_dim × head_v_dim]` per value head
updated by the delta rule, and a gated output RMSNorm; softmax layers interleave
with a double-width (query‖gate) projection + sigmoid output gate + partial RoPE,
and the FFN is MoE+shared-expert. New kernels: the Gated DeltaNet scan
(linear-attention recurrence) + depthwise conv + gated norm.

The common blocker for all three: **a recurrent state cache and a sequential scan
kernel** (Mamba-2 SSM or DeltaNet). That is a different residency substrate than
the positional KV cache the runner is built around — the biggest single piece of
work remaining, and the reason these stay staged.

## If the program resumes

Recommended order by (value × tractability):

1. **`llama4_text`** — medium, no recurrence, and a current frontier model. The
   iRoPE / L2-norm / input-scaled-MoE variants are self-contained.
2. **`gemma4`** — medium; unblocks the Gemma frontier but the per-layer KV-width
   generalization touches the cache allocation broadly.
3. **`gemma3` / `gpt2`** — small, but lower value (superseded / legacy). Good
   "warm-up" tasks that exercise sandwich-norm + non-gated-gelu + learned-pos.
4. **Mamba-2 / DeltaNet substrate** (`granite`, `nemotron_h`, `qwen3_5`) — large;
   only worth it as a deliberate "state-space residency" project, since one new
   scan kernel + recurrent-state cache unlocks all three at once.

Orthogonal to coverage: the **perf** levers (DP4A/`dot4I8Packed`, GPU speculative
decode) are tracked separately in
[`gpu-next-levers-assessment.md`](gpu-next-levers-assessment.md), and the real
big-model runs need a bigger-VRAM box or int4-expert residency.

## Which architectures run resident — summary table

Moved here from the README (2026-08-27), unchanged. This is a CAPABILITY table, not a measured
one, which is why it landed here beside the per-family scoping rather than in the
provenance-gated `benchmarks.md`.

### What runs on the GPU

GPU-resident decode covers a subset of architectures. **Everything else runs on the
pure-Go CPU path automatically.** A shared feature taxonomy checks each model's required
features against what the backend implements; an unsupported architecture is *declined at
load and falls back to CPU* rather than run with a feature quietly dropped.

| Family | CUDA | Metal |
|---|---|---|
| Qwen2 · Qwen3 · Llama | ✅ resident | ✅ resident |
| Mistral · Phi-3-mini-4k | ✅ resident | ✅ resident¹ |
| Gemma 3 | ✅ resident³ | ✅ resident³ |
| MoE — Mixtral · Qwen2-MoE · Qwen3-MoE · GLM-MoE | ✅ resident⁴ | ✅ resident² |
| Gemma 4 (dense + MoE) | ✅ resident⁵ | ✅ resident⁵ |
| MLA · DeltaNet/YaRN | CPU fallback | CPU fallback |

The full per-family × 4-backend (CPU · WebGPU · CUDA · Metal) table is **generated** from the
residency predicate (`decoder.ResidentEligible`) and freshness-gated in CI, so it can never drift
from what a backend actually admits: [docs/hardware-matrix.md](hardware-matrix.md).

¹ Metal Mistral-7B needs > 16 GB unified memory (int8 + int4). Both backends implement
qk-norm + sliding-window; Metal also does partial rotary, so a partial-rotary Phi
variant is resident on Metal but falls back on CUDA.

³ Gemma 3 (both backends) covers the sandwich-norm block, GeGLU, the (1+w) RMS offset, the
√hidden embedding scale, and Gemma's dual RoPE base — validated on a real gemma-3-4b-it against
the CPU path. Metal parity was gated on a GELU-tanh overflow fix (the `<bos>` massive-activation
gate drove `tanh`'s argument past its internal `exp` range → NaN; clamped).

⁵ Gemma 4 (both the dense variants and the `enable_moe_block` MoE) runs resident on CUDA and
Metal. It was opt-in behind `GOINFER_GEMMA4_RESIDENT` through the bring-up and is now
unconditional; the variable is inert and can be removed from any script that sets it. WebGPU
still declines — it lacks the four Gemma kernels — and **E-models (E2B/E4B, per-layer
embeddings) decline on every backend**, since none implements the PLE branch and admitting one
would silently skip it. Both are the feature gate's answer, not a hardcoded row, and both are
asserted (`TestGemma4Admission_unconditional`, `TestGemma4EModel_realDeclinesResident`).

⁴ CUDA MoE runs Mixtral and GLM-MoE resident (on-GPU router, row-stacked int4 experts, ungated
shared expert). Qwen2-MoE / Qwen3-MoE decline to CPU on CUDA — their gated shared expert
(sigmoid-scaled) isn't built yet.

² Metal MoE (router + stacked experts + shared expert) is validated by assembly
equivalence (identical experts ≡ the dense FFN, cosine 1.0) + per-kernel parity vs CPU;
a real MoE checkpoint needs a Mac with enough unified memory (Qwen1.5-MoE-A2.7B is 14.3B
≈ 14 GB at int8 load), so the real-model e2e cross-check runs on the CUDA box. The
DeltaNet/Llama-4/Gemma hybrids stay on CPU (declined before residency).

An unlisted or unsupported model still runs — in pure Go on the CPU. The portable
WebGPU backend (`-tags gpu`) covers a broader resident set (MoE, MLA, SSM, YaRN); see
[docs/capability-matrix.md](capability-matrix.md) for the full map.

> **Note:** in `cmd/serve`, a GPU-resident model skips prompt-prefix KV reuse and
> speculative decoding — the resident decode path is fast enough that the per-request
> session optimization isn't worth it. The OpenAI API is stateless (clients resend the
> whole conversation), so this is a throughput trade, not a correctness change.

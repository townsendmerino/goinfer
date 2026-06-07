# Qwen3.5/3.6 MoE (`qwen3_5_moe`) — bring-up spec & plan

Qwen 3.6 (HF `model_type: qwen3_5_moe`, `Qwen3_5MoeForConditionalGeneration`;
the 3.5 and 3.6 checkpoints share this architecture) is a **hybrid
linear/softmax-attention MoE**. It is the largest forward-pass addition in
goinfer to date: a brand-new sequence-mixing primitive (Gated DeltaNet linear
attention) with a **recurrent matrix state**, alongside the existing KV-cached
softmax attention.

Reference: `transformers==5.10.2`
`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (line refs below). The
torch fallbacks (no `fla` / `causal_conv1d` installed) are the parity targets:
`torch_recurrent_gated_delta_rule` (decode) and `torch_chunk_gated_delta_rule`
(prefill) — mathematically equivalent.

## Config (Qwen3.6-35B-A3B)

```
num_hidden_layers 40   hidden_size 2048   rms_norm_eps 1e-6
# softmax ("gated attention") layers:
head_dim 256   num_attention_heads 16   num_key_value_heads 2 (GQA)
partial_rotary_factor 0.25  (rotary_dim = 64)   rope_theta 1e7   qk-norm: yes
# layer pattern:
layer_types: 3 linear_attention : 1 full_attention   (full_attention_interval 4)
             → 30 Gated-DeltaNet + 10 Gated-Attention layers
# Gated DeltaNet (linear) layers:
linear_conv_kernel_dim 4
linear_key_head_dim 128   linear_value_head_dim 128
linear_num_key_heads 16   linear_num_value_heads 32   (GVA: v/k = 2)
# MoE (every layer):
num_experts 256   num_experts_per_tok 8   moe_intermediate_size 512
shared_expert_intermediate_size 512   (norm_topk_prob absent → false)
```

## Gated DeltaNet layer (the new primitive)

`Qwen3_5MoeGatedDeltaNet` (modeling §368–540). Per-layer weights (all Linear
bias=False unless noted):

| tensor | shape | note |
|---|---|---|
| `in_proj_qkv` | `[2·key_dim + value_dim, hidden]` | key_dim=`head_k·num_k`, value_dim=`head_v·num_v` |
| `in_proj_z` | `[value_dim, hidden]` | output gate z |
| `in_proj_b` | `[num_v_heads, hidden]` | write-gate logits β |
| `in_proj_a` | `[num_v_heads, hidden]` | decay-gate input a |
| `conv1d.weight` | `[conv_dim, 1, K=4]` | depthwise causal, groups=conv_dim; **bias=False**; conv_dim = 2·key_dim+value_dim |
| `dt_bias` | `[num_v_heads]` | |
| `A_log` | `[num_v_heads]` | decay magnitude |
| `norm.weight` | `[head_v_dim]` | gated RMSNorm |
| `out_proj` | `[hidden, value_dim]` | |

Forward (per token at decode; chunked at prefill but equivalent):

```
mixed = in_proj_qkv(h)                     # [conv_dim]
mixed = silu(depthwise_causal_conv1d(mixed, K=4))   # bias-free, left-pad K-1
q, k, v = split(mixed, [key_dim, key_dim, value_dim])
q,k → [num_k_heads, head_k_dim];  v → [num_v_heads, head_v_dim]
beta = sigmoid(in_proj_b(h))               # [num_v_heads]
g    = -exp(A_log) * softplus(in_proj_a(h) + dt_bias)   # [num_v_heads], log-decay (<0)
# GVA: repeat_interleave q,k by num_v_heads/num_k_heads (=2) → 32 heads
# per head h, recurrent state S_h is [head_k_dim, head_v_dim]:
q,k = l2norm(q,dim=-1), l2norm(k,dim=-1)   # eps 1e-6
q  *= 1/sqrt(head_k_dim)                    # query scale (after l2norm)
S  *= exp(g)                               # decay
kv  = sum_k(S * k[:,None])                 # [head_v_dim]  = Sᵀk  (predicted v)
δ   = (v - kv) * beta                      # [head_v_dim]
S  += k[:,None] * δ[None,:]                # outer product k⊗δ
o   = sum_k(S * q[:,None])                 # [head_v_dim]  = Sᵀq
o   = rmsnorm(o, head_v_dim) * silu(z)     # gated RMSNorm (norm BEFORE gate)
y   = out_proj(o)                          # [hidden]
```

`torch_recurrent_gated_delta_rule` §324–365; `RMSNormGated` §184–199 (norm then
`* silu(gate)`). **Recurrent state = `[num_v_heads, head_k_dim, head_v_dim]`** per
linear layer — fixed size, O(1)/token, *not* a growing KV cache and *not*
position-truncatable. There is also a small **conv state** (last K-1 inputs) per
linear layer for streaming decode.

## Gated Attention layer (the softmax 1-in-4)

`Qwen3_5MoeAttention` (modeling §, pinned): standard causal softmax + GQA +
per-head QK-norm + **partial RoPE** (rotary_dim = 0.25·head_dim) — which goinfer's
`causalAttention` mostly does — with two twists:

- **`q_proj` is double-width** `[num_heads · head_dim · 2, hidden]`: per head it
  emits `[query ‖ gate]`. Split on the last dim → `query` (head_dim) and `gate`
  (head_dim). q_norm is applied to the **query half only**; the gate is raw.
- After attention + reshape, **`attn_out *= sigmoid(gate)`** before `o_proj` (the
  output gate). `gate` is reshaped to `[num_heads·head_dim]`, not RoPE'd, not normed.

So a qwen3_5_moe forward can't reuse the plain attention path verbatim — it needs
the 2×-width q_proj split + the output gate. k_proj/v_proj/o_proj are normal
(bias = attention_bias, false here). scaling = head_dim^-0.5.

## MoE

256 experts, top-8, expert width 512, plus a shared expert (width 512) — the
existing `MoEConfig` + `moeMLP` (routed + shared + sigmoid gate) machinery. On
every layer. **Gotcha:** `Qwen3_5MoeTopKRouter` ALWAYS renormalizes the top-k
probabilities (`router_top_value /= sum`), even though `norm_topk_prob` is absent
from config — so `NormTopKProb` must be set true (verified: it's the difference
between cosine 0.9985 and 1.0).

**Norm gotcha:** `Qwen3_5MoeRMSNorm` is `x · (1 + weight)` with weight zero-init
(Gemma-style), so `RMSAddOne = true` for the block/final/QK norms. The DeltaNet's
*gated* RMSNorm (`Qwen3_5MoeRMSNormGated`) is separately the standard `x · weight`
(weight ones-init) and is handled inside the primitive.

## Hybrid cache (decision: correctness-first)

A sequence's state is **hybrid**: KV cache for the 10 full layers + a recurrent
`DeltaState{S, convState}` for the 30 linear layers. Because the recurrent state
is not position-truncatable, for `qwen3_5_moe`:
- cross-call prefix KV reuse (`Session`, v0.3.0) **falls back to full recompute**;
- speculative decoding's `TruncateTo` is **disabled**.
Optimizing those for hybrid models (state checkpoints) is a later track.

## Plan (parity-first, matches the Gemma 4 bring-up)

1. **1a ✅** Confirm HF oracle (`transformers 5.10.2` has `qwen3_5_moe`) + pin the
   math (this doc).
2. **1b** `qwen3_5_moe` config + validator + `Architecture` descriptor: per-layer
   linear/full dispatch (extend the `layerIsGlobal`-style mechanism to a 3-way
   layer kind), DeltaNet params, output-gate flag, partial RoPE, MoE-256.
3. **1c** Tiny-random `qwen3_5_moe` HF golden (`scripts/pin_qwen35_forward.py` →
   `testdata/qwen35_forward_golden.json`) + a decode golden; parity test scaffold.
4. **2** Gated DeltaNet forward (conv + gated delta recurrence + gated RMSNorm),
   sequential, gated to argmax-exact + logit-cosine on the tiny golden.
5. **3** Hybrid cache wired into `Generate`/prefill/`runLayers`; reuse/spec fall
   back for hybrid models.
6. **4** Gated-Attention output gate + tensor schema (safetensors + GGUF) +
   full-model smoke on a real checkpoint (Mellum2-style).
7. **5 (later)** perf (chunked/parallel scan), reuse/spec for hybrid models.

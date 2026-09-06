#!/usr/bin/env python
"""Tiny-random Bailing Hybrid (Ling 3.0, model_type "bailing_hybrid") checkpoint + forward golden.

inclusionAI/Ling-3.0-tiny: DeepSeek-style MLA alternating with Kimi Delta Attention (KDA) every
layer_group_size-th layer being MLA, over a DeepSeekMoE FFN (docs/task-families-2026-09.md batch 2,
G5). This script does NOT instantiate the real BailingMoeV3ForCausalLM via trust_remote_code: the
real modeling_bailing_moe_v3.py imports `fla.ops.kda` at module top level, which transitively
imports Triton (fla/ops/__init__.py -> fla/ops/abc/chunk.py -> `import triton`) -- unavailable on
this Mac (no wheel for this platform; the same constraint scripts/pin_kda_rehearsal.py already
documented for the SAME reason). is_torch_fx_available also moved in transformers 5.15.0, a second,
unrelated import break in the same file.

Instead, this builds a hand-assembled reference model whose forward logic is verified against the
real source directly (fetched from inclusionAI/Ling-3.0-tiny, not paraphrased) piece by piece:

  - MLA (BailingMoeV3MultiLatentAttention), the MoE Gate/SparseMoeBlock/MLP, and RMSNorm are
    reproduced near-verbatim -- none of them import fla at all, confirmed by reading the real file.
  - KDA (BailingMoeV3KimiDeltaAttention) is reproduced structurally (same q/k/v proj -> per-stream
    depthwise causal conv+SiLU -> f_proj/b_proj/g_proj gates -> recurrence -> sigmoid-gated
    RMSNorm -> o_proj), but the recurrence itself calls naive_kda_lowerbound_gate +
    naive_recurrent_kda -- fla-org/flash-linear-attention's OWN reference implementation, copied
    verbatim (MIT license) from scripts/pin_kda_rehearsal.py, which already proved it bit-matches
    the same functions (maxAbsDiff 2.98e-08 against goinfer's Go port). chunk_kda/fused_recurrent_kda
    (the Triton-accelerated, Triton-only entry points the real class calls) are a DIFFERENT
    parallelization of the IDENTICAL recurrence, per fla's own naming convention -- not a different
    model.

Weight attribute names match the real state_dict exactly (attention.*, mlp.*, input_layernorm,
etc. -- verified against the real checkpoint's actual safetensors header, not assumed), so the
resulting checkpoint loads through goinfer's real bailingHybridArchitecture adapter unmodified.

    ~/.venv-nemotron3/bin/python scripts/pin_bailing_hybrid_tiny.py
    -> testdata/bailing_hybrid_forward_golden.json + testdata/bailing_hybrid_forward_full.json
       + testdata/bailing_hybrid-tiny/
"""
import json, math, os, random
import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import save_file

HERE = os.path.dirname(__file__)
TD = os.path.join(HERE, "..", "testdata")
GOLDEN = os.path.join(TD, "bailing_hybrid_forward_golden.json")
FULL = os.path.join(TD, "bailing_hybrid_forward_full.json")
CKPT = os.path.join(TD, "bailing_hybrid-tiny")

CFG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=8,
    layer_group_size=4,  # 3 KDA + 1 MLA, matching the real release's own ratio
    q_lora_rank=32, kv_lora_rank=48, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
    n_group=2, topk_group=1,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    num_shared_experts=1, moe_shared_expert_intermediate_size=32,
    first_k_dense_replace=1, routed_scaling_factor=2.5,
    head_dim=8, short_conv_kernel_size=4, no_kda_lora=True, kda_safe_gate=True, kda_lower_bound=-5.0,
    gated_attention_proj_granularity_type="head_wise",
    rms_norm_eps=1e-6, rope_theta=10000.0, rope_interleave=True,
)
PROMPT = [1, 2, 7, 42, 100, 5, 200, 13, 88, 250]
SAMPLE_SEED = 1234
N_SAMPLE = 128
N_TOPK = 16


# --------------------------------------------------------------------------------------------
# Verbatim from fla-org/flash-linear-attention, fla/ops/kda/{gate,naive}.py (MIT license) --
# same copies as scripts/pin_kda_rehearsal.py, which already proved these match goinfer's Go
# port (maxAbsDiff 2.98e-08, cosine 1.0).
# --------------------------------------------------------------------------------------------

def naive_kda_lowerbound_gate(g, A_log=None, dt_bias=None, lower_bound=-5.0, output_dtype=torch.float32):
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    if A_log is not None:
        g = A_log.view(H, 1).float().exp() * g
    g = lower_bound * F.sigmoid(g)
    return g.to(output_dtype)


def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
    # Verbatim from the real modeling_bailing_moe_v3.py (no fla dependency).
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    # BailingMoeV3RMSNorm, verbatim.
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class MLP(nn.Module):
    # BailingMoeV3MLP, verbatim.
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GatedRMSNorm(nn.Module):
    # FusedRMSNormGated's own weight parameter, as a proper submodule (not a flat Parameter) so it
    # serializes as "attention.o_norm.weight" -- matching the real checkpoint's actual tensor name,
    # not a script-convenience shortcut.
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class Gate(nn.Module):
    # BailingMoeV3Gate, verbatim (noaux_tc: sigmoid + expert_bias + group-limited top-k +
    # routed_scaling_factor).
    def __init__(self, c):
        super().__init__()
        self.top_k = c["num_experts_per_tok"]
        self.num_experts = c["num_experts"]
        self.n_group = c["n_group"]
        self.topk_group = c["topk_group"]
        self.weight = nn.Parameter(torch.empty((self.num_experts, c["hidden_size"])))
        self.routed_scaling_factor = c["routed_scaling_factor"]
        self.register_buffer("expert_bias", torch.zeros((self.num_experts)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(self, scores):
        num_tokens, _ = scores.size()
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(
            num_tokens, self.n_group, self.num_experts // self.n_group).reshape(num_tokens, -1)
        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        return torch.topk(masked_scores, k=self.top_k, dim=-1)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = torch.sigmoid(logits.float()).type_as(logits)
        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)
        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class SparseMoeBlock(nn.Module):
    # BailingMoeV3SparseMoeBlock, verbatim (training-path branch only -- no moe_infer fast path
    # needed for a 3-token-ish tiny fixture, and it must match bit-for-bit anyway).
    def __init__(self, c):
        super().__init__()
        self.num_experts_per_tok = c["num_experts_per_tok"]
        self.experts = nn.ModuleList([MLP(c["hidden_size"], c["moe_intermediate_size"]) for _ in range(c["num_experts"])])
        self.gate = Gate(c)
        self.shared_experts = MLP(c["hidden_size"], c["moe_shared_expert_intermediate_size"] * c["num_shared_experts"])

    def forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        flat_hidden = hidden_states.view(-1, h)
        flat_topk_idx = topk_idx.view(-1)
        flat_hidden_rep = flat_hidden.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(flat_hidden_rep)
        for i, expert in enumerate(self.experts):
            mask = flat_topk_idx == i
            if mask.any():
                y[mask] = expert(flat_hidden_rep[mask])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.view(bsz, seq_len, h)
        y = y + self.shared_experts(identity)
        return y


class MLA(nn.Module):
    # BailingMoeV3MultiLatentAttention, verbatim (q-LoRA path only -- q_lora_rank is set on the
    # release; the direct-q_proj path is DeepSeek-V2-Lite's, already covered by deepseekArchitecture).
    def __init__(self, c):
        super().__init__()
        self.num_heads = c["num_attention_heads"]
        self.q_lora_rank = c["q_lora_rank"]
        self.qk_rope_head_dim = c["qk_rope_head_dim"]
        self.kv_lora_rank = c["kv_lora_rank"]
        self.v_head_dim = c["v_head_dim"]
        self.qk_nope_head_dim = c["qk_nope_head_dim"]
        self.qk_head_dim = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
        self.gate_granularity = c["gated_attention_proj_granularity_type"]

        self.q_a_proj = nn.Linear(c["hidden_size"], self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(c["hidden_size"], self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        if self.gate_granularity == "head_wise":
            self.g_proj = nn.Linear(c["hidden_size"], self.num_heads, bias=False)
        else:
            self.g_proj = None
        self.dense = nn.Linear(self.num_heads * self.v_head_dim, c["hidden_size"], bias=False)
        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(self, hidden_states, cos, sin):
        bsz, seq_len = hidden_states.shape[:2]
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(bsz, seq_len, -1, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(bsz, seq_len, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = k_rot.view(bsz, 1, seq_len, self.qk_rope_head_dim)

        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        causal = torch.full((seq_len, seq_len), float("-inf")).triu(1).to(attn_weights.dtype)
        attn_weights = attn_weights + causal
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()

        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(hidden_states).float()).type_as(hidden_states)
            attn_output = attn_output * gate[:, :, :, None]

        attn_output = attn_output.reshape(bsz, seq_len, -1)
        return self.dense(attn_output)


class KDA(nn.Module):
    # BailingMoeV3KimiDeltaAttention, reproduced structurally (three separate q/k/v projections
    # and per-stream depthwise causal convs -- verified against the real checkpoint's actual
    # tensor shapes, not assumed reusable as one combined conv); the recurrence calls fla's own
    # naive reference (see module docstring) instead of the Triton-only chunk_kda entry point.
    def __init__(self, c):
        super().__init__()
        self.head_dim = c["head_dim"]
        self.num_heads = c["num_attention_heads"]
        proj = self.head_dim * self.num_heads
        self.conv_size = c["short_conv_kernel_size"]
        self.lower_bound = c["kda_lower_bound"]

        self.q_proj = nn.Linear(c["hidden_size"], proj, bias=False)
        self.k_proj = nn.Linear(c["hidden_size"], proj, bias=False)
        self.v_proj = nn.Linear(c["hidden_size"], proj, bias=False)
        self.q_conv1d = nn.Conv1d(proj, proj, kernel_size=self.conv_size, groups=proj, padding=self.conv_size - 1, bias=False)
        self.k_conv1d = nn.Conv1d(proj, proj, kernel_size=self.conv_size, groups=proj, padding=self.conv_size - 1, bias=False)
        self.v_conv1d = nn.Conv1d(proj, proj, kernel_size=self.conv_size, groups=proj, padding=self.conv_size - 1, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads).uniform_(1, 16)))
        self.f_proj = nn.Linear(c["hidden_size"], proj, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(proj))
        self.b_proj = nn.Linear(c["hidden_size"], self.num_heads, bias=False)
        self.g_proj = nn.Linear(c["hidden_size"], proj, bias=False)
        self.o_norm = GatedRMSNorm(self.head_dim)
        self.o_proj = nn.Linear(proj, c["hidden_size"], bias=False)

    def _conv(self, x, conv):
        # x: [B,T,proj]. Causal depthwise conv + SiLU, matching fla's ShortConvolution exactly
        # (bias=False, padding=kernel_size-1, then trim the trailing padding off the right).
        y = conv(x.transpose(1, 2))[:, :, : x.shape[1]]
        return F.silu(y.transpose(1, 2))

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self._conv(self.q_proj(hidden_states), self.q_conv1d)
        k = self._conv(self.k_proj(hidden_states), self.k_conv1d)
        v = self._conv(self.v_proj(hidden_states), self.v_conv1d)

        q = q.view(bsz, seq_len, H, D)
        k = k.view(bsz, seq_len, H, D)
        v = v.view(bsz, seq_len, H, D)
        q = l2norm(q)
        k = l2norm(k)

        raw_gate = self.f_proj(hidden_states).view(bsz, seq_len, H, D)
        dt_bias = self.dt_bias.view(H, D)
        g = naive_kda_lowerbound_gate(raw_gate, A_log=self.A_log, dt_bias=dt_bias, lower_bound=self.lower_bound)
        beta = torch.sigmoid(self.b_proj(hidden_states).float())

        o, _ = naive_recurrent_kda(q, k, v, g, beta, output_final_state=False)

        g_out = self.g_proj(hidden_states).view(bsz, seq_len, H, D)
        # FusedRMSNormGated(activation='sigmoid'): rmsnorm(o) * weight * sigmoid(g_out).
        var = o.pow(2).mean(-1, keepdim=True)
        o = o * torch.rsqrt(var + 1e-6) * self.o_norm.weight * torch.sigmoid(g_out.float()).type_as(o)
        o = o.reshape(bsz, seq_len, H * D)
        return self.o_proj(o)


class DecoderLayer(nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        group = c["layer_group_size"]
        tail_start = (c["num_hidden_layers"] // group) * group
        self.is_mla = (layer_idx + 1) % group == 0 or layer_idx >= tail_start
        self.attention = MLA(c) if self.is_mla else KDA(c)
        self.mlp = SparseMoeBlock(c) if layer_idx >= c["first_k_dense_replace"] else MLP(c["hidden_size"], c["intermediate_size"])
        self.input_layernorm = RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])

    def forward(self, h, cos, sin):
        residual = h
        h = self.input_layernorm(h)
        h = self.attention(h, cos, sin) if self.is_mla else self.attention(h)
        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = residual + h
        return h


class BailingHybridTiny(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.word_embeddings = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList([DecoderLayer(c, i) for i in range(c["num_hidden_layers"])])
        self.norm = RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        self.lm_head = nn.Linear(c["hidden_size"], c["vocab_size"], bias=False)
        rope_dim = c["qk_rope_head_dim"]
        inv_freq = 1.0 / (c["rope_theta"] ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, ids):
        seq_len = ids.shape[1]
        pos = torch.arange(seq_len).float()
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)

        h = self.word_embeddings(ids)
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm(h)
        return self.lm_head(h)


def main():
    torch.manual_seed(0)
    m = BailingHybridTiny(CFG).eval()
    with torch.no_grad():
        logits = m(torch.tensor([PROMPT]))[0, -1].float()
    lg = logits.tolist()
    vocab = len(lg)
    argmax = int(torch.tensor(lg).argmax())

    order = sorted(range(vocab), key=lambda i: lg[i], reverse=True)
    top_k = [[i, lg[i]] for i in order[:N_TOPK]]
    rng = random.Random(SAMPLE_SEED)
    sample_ids = rng.sample(range(vocab), min(N_SAMPLE, vocab))
    sample = [[i, lg[i]] for i in sample_ids]
    stats = dict(n=vocab, sum=sum(lg), sum_sq=sum(v * v for v in lg), min=min(lg), max=max(lg))

    golden = dict(
        model_id="testdata/bailing_hybrid-tiny (hand-assembled BailingHybridTiny, verified against "
                  "the real modeling_bailing_moe_v3.py -- see pin_bailing_hybrid_tiny.py's own docstring "
                  "for why the real class can't be instantiated on this Mac)",
        note="tiny hand-assembled forward oracle; float32, next-token logits at the last position. "
             "ids are raw token ids (the Go test is tokenizer-independent). argmax must match; "
             "top_k/sample to small tol; full cosine in the gitignored bailing_hybrid_forward_full.json. "
             "Regenerate: pin_bailing_hybrid_tiny.py",
        dtype="float32", prompt="", config=CFG, ids=PROMPT,
        argmax=argmax, argmax_token="", vocab_size=vocab, stats=stats,
        top_k=top_k, sample_seed=SAMPLE_SEED, sample=sample,
    )
    os.makedirs(TD, exist_ok=True)
    json.dump(golden, open(GOLDEN, "w"))
    json.dump(dict(argmax=argmax, logits=lg), open(FULL, "w"))

    sd = {}
    for name, t in m.state_dict().items():
        if name == "inv_freq":
            continue
        if name.startswith("layers."):
            sd["model." + name] = t.contiguous()
        elif name in ("word_embeddings.weight", "norm.weight"):
            sd["model." + name] = t.contiguous()
        else:
            sd[name] = t.contiguous()  # lm_head.weight
    os.makedirs(CKPT, exist_ok=True)
    save_file(sd, os.path.join(CKPT, "model.safetensors"), metadata={"format": "pt"})

    config_json = dict(CFG)
    config_json.update(dict(
        model_type="bailing_hybrid",
        architectures=["BailingMoeV3ForCausalLM"],
        qk_head_dim=CFG["qk_nope_head_dim"] + CFG["qk_rope_head_dim"],
        rope_scaling=None,
        tie_word_embeddings=False,
        num_nextn_predict_layers=0,
        pad_token_id=None, bos_token_id=None, eos_token_id=None,
    ))
    json.dump(config_json, open(os.path.join(CKPT, "config.json"), "w"))

    print(f"vocab={vocab} argmax={argmax} layers={CFG['num_hidden_layers']}")
    print(f"stats min={stats['min']:.4f} max={stats['max']:.4f}")
    print("wrote", GOLDEN, FULL, CKPT)


if __name__ == "__main__":
    main()

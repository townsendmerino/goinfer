#!/usr/bin/env python3
"""Tiny-random Gemma 1 (GemmaForCausalLM) and Gemma 2 (Gemma2ForCausalLM) checkpoints + forward goldens.

    python3 scripts/pin_gemma_tiny.py
    -> testdata/{gemma1,gemma2,gemma2-hd}-tiny/ + testdata/{gemma1,gemma2,gemma2-hd}_forward_golden.json + ..._forward_full.json

Built so each family-specific feature changes the output, and the script checks that it does:
  - both: add-one RMSNorm with random norm weights (a zero weight is a scale of 1 and would hide a missing +1), the
    sqrt(hidden) embedding scale, GeGLU (gelu_pytorch_tanh), a tied head, head_dim != hidden/heads;
  - Gemma 1: MQA (one KV head), pre-norm only;
  - Gemma 2: sandwich norms (pre and post, attention and FFN), query_pre_attn_scalar != head_dim, alternating
    sliding/full attention with a window of 4 under a 12-token prompt, and both softcaps (attention 5, final 3) small
    enough, with q/k and the embedding scaled up, that tanh saturates. The script recomputes the logits with each softcap
    off and refuses to write a golden where turning one off does not move them.
"""
import json, os, random
import torch
from transformers import GemmaConfig, GemmaForCausalLM, Gemma2Config, Gemma2ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "testdata")
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88, 250, 9, 300, 17]  # 12 ids, 3x the Gemma 2 window
SAMPLE_SEED, N_SAMPLE, N_TOPK = 1234, 256, 32

FAMILIES = {
    "gemma1": (GemmaConfig, GemmaForCausalLM, dict(  # Gemma 1 (model_type gemma); "gemma_*" names belong to gemma3-270m
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=1, head_dim=32, hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True)),
    "gemma2": (Gemma2Config, Gemma2ForCausalLM, dict(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, hidden_activation="gelu_pytorch_tanh",
        query_pre_attn_scalar=24, sliding_window=4, attn_logit_softcapping=5.0, final_logit_softcapping=3.0,
        max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True)),
    # The GGUF gate's Gemma 2: query_pre_attn_scalar == head_dim, the rule llama.cpp encodes (a GGUF does not store the
    # scalar; every release satisfies it: 256 = head_dim on the 2B/9B, hidden/heads = 144 on the 27B).
    "gemma2-hd": (Gemma2Config, Gemma2ForCausalLM, dict(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, hidden_activation="gelu_pytorch_tanh",
        query_pre_attn_scalar=32, sliding_window=4, attn_logit_softcapping=5.0, final_logit_softcapping=3.0,
        max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True)),
}


def logits_of(m):
    with torch.no_grad():
        return m(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0, -1].float()


def main():
    for name, (C, M, cfg) in FAMILIES.items():
        torch.manual_seed(0)
        c = C(**cfg)
        c._attn_implementation = "eager"  # HF applies the attention softcap on the eager path
        m = M(c).eval().to(torch.float32)
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for pn, p in m.named_parameters():
                if pn.endswith("norm.weight"):
                    p.copy_(torch.randn(p.shape, generator=g) * 0.3)
                elif any(k in pn for k in ("q_proj", "k_proj")):
                    p.mul_(8.0)  # scores of several units, so the attention softcap is not the identity
                elif pn.endswith("embed_tokens.weight"):
                    p.mul_(6.0)  # tied head: logits of several units, so the final softcap is not the identity
        lg = logits_of(m)
        if name.startswith("gemma2"):
            # Gemma2Attention caches attn_logit_softcapping at construction; the final one is read from the config.
            for key in ("attn_logit_softcapping", "final_logit_softcapping"):
                owners = [l.self_attn for l in m.model.layers] if key.startswith("attn") else [m.config]
                keep = [getattr(o, key) for o in owners]
                for o in owners:
                    setattr(o, key, None)
                off = logits_of(m)
                for o, k in zip(owners, keep):
                    setattr(o, key, k)
                d = float((lg - off).abs().max())
                print(f"{name}: {key} off moves the logits by up to {d:.4f}")
                assert d > 0.05, f"{key} barely matters in this fixture ({d})"
        lg = lg.tolist()
        order = sorted(range(len(lg)), key=lambda i: lg[i], reverse=True)
        rng = random.Random(SAMPLE_SEED)
        sample_ids = rng.sample(range(len(lg)), min(N_SAMPLE, len(lg)))
        golden = dict(
            model_id=f"testdata/{name}-tiny (seeded {M.__name__})",
            note=f"tiny {M.__name__} forward oracle; HF float32 eager, next-token logits at the last position. "
                 "Regenerate: scripts/pin_gemma_tiny.py",
            dtype="float32", prompt="", config=cfg, ids=PROMPT, argmax=order[0], argmax_token="", vocab_size=len(lg),
            stats=dict(n=len(lg), sum=sum(lg), sum_sq=sum(v * v for v in lg), min=min(lg), max=max(lg)),
            top_k=[[i, lg[i]] for i in order[:N_TOPK]], sample_seed=SAMPLE_SEED, sample=[[i, lg[i]] for i in sample_ids])
        json.dump(golden, open(os.path.join(TD, f"{name}_forward_golden.json"), "w"))
        json.dump(dict(argmax=order[0], logits=lg), open(os.path.join(TD, f"{name}_forward_full.json"), "w"))
        m.save_pretrained(os.path.join(TD, f"{name}-tiny"), safe_serialization=True)
        print(f"{name}: argmax {order[0]}, logits {min(lg):.3f} .. {max(lg):.3f}")


if __name__ == "__main__":
    main()

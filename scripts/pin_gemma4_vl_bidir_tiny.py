#!/usr/bin/env python
"""Build a tiny-random Gemma 4 vision-language checkpoint with
use_bidirectional_attention="vision" — the 26B-A4B/31B-class fixture for
decoder/forward_gemma4_batched.go's batched forward (mirrors
pin_gemma4_vl_tiny.py's approach, which is the E2B/E4B-class, causal-only twin
of this file — that one must stay untouched as the P0 regression pin).

Saves a small Gemma4ForConditionalGeneration and pins a TEXT-ONLY forward
golden (no pixel_values). Even with use_bidirectional_attention="vision", a
text-only forward degenerates to plain causal: Gemma4Model.forward always
builds block_sequence_ids when this config is set (even with no images, as an
all -1 tensor), and blockwise_overlay's predicate (q_group==kv_group AND
q_group>=0) is unconditionally false when every group is -1 — so this pins
that BOTH the new batched path (with imgLen=0) AND the existing sequential
path still agree with HF and with each other on text-only input, the
regression check the plan calls for.

layer_types mixes BOTH sliding_attention and full_attention (the masking
change must be proven correct on both — real HF source confirmed this session
that both layer types get the identical OR(causal, blockwise) treatment, not
just sliding as an earlier session's doc note wrongly claimed).
num_kv_shared_layers=0 and hidden_size_per_layer_input=0 match the real
26B-A4B checkpoint exactly (confirmed via ~/models/gemma-4-26b-a4b-it/config.json
this session) and isolate the masking change from the separately-tested
KV-sharing/PLE paths. enable_moe_block=False (dense-only): the masking change
is orthogonal to FFN dispatch, and MoE FFN correctness is already covered by
TestGemma4MoE_forwardParity — real 26B-A4B (deferred, not this pass) is where
MoE+bidirectional first need joint verification.

Tiny -> sub-second; no download.

    ~/.venv-vl/bin/python scripts/pin_gemma4_vl_bidir_tiny.py
    -> testdata/gemma4_vl_bidir_tiny_text_golden.json
    -> testdata/gemma4-vl-bidir-tiny/   (HF safetensors checkpoint: vision + text)
"""
import json
import os

import torch
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "gemma4_vl_bidir_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-vl-bidir-tiny")

TEXT = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    global_head_dim=512, num_global_key_value_heads=2, attention_k_eq_v=True,
    layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    # Deliberately narrower than the image block the companion image-golden script
    # uses (block length 9) — the discriminator: a query late in the block has a
    # causal window that no longer reaches the block's own start, so only a real
    # blockwise-OR mask (not a causal-only or windowed-only approximation) can be
    # correct there. See gemma4AttendRange's own doc comment for the proof.
    sliding_window=3,
    max_position_embeddings=64, rms_norm_eps=1e-6,
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    hidden_size_per_layer_input=0,  # PLE-free — matches real 26B-A4B exactly
    enable_moe_block=False,         # dense-only — masking is orthogonal to FFN dispatch
    num_kv_shared_layers=0,         # matches real 26B-A4B exactly
    use_bidirectional_attention="vision",
)
VISION = dict(
    hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=4,
    head_dim=8, patch_size=4, pooling_kernel_size=3, position_embedding_size=16,
    rms_norm_eps=1e-6, use_clipped_linears=True,
    rope_parameters={"rope_theta": 100.0, "rope_type": "default"}, standardize=False,
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]  # text-only, no image-soft-token
N_NEW = 6


def strengthen_text(model):
    """Same degeneracy guard as pin_gemma4_vl_tiny.py's strengthen_text: HF init
    leaves norms/layer_scalar at identity, so a bug applying them (x1) would not
    move the golden. Separate generator => the linear weights + input stay
    bit-identical."""
    g = torch.Generator().manual_seed(1234)
    n_norm = n_ls = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "vision_tower" in name or "embed_vision" in name:
                continue
            if name.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or name.endswith("model.norm.weight"):
                p.normal_(1.0, 0.1, generator=g)
                n_norm += 1
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.normal_(1.0, 0.1, generator=g)
                n_ls += 1
    return dict(norm=n_norm, layer_scalar=n_ls)


def main():
    torch.manual_seed(0)
    cfg = Gemma4Config(
        text_config=Gemma4TextConfig(**TEXT),
        vision_config=Gemma4VisionConfig(**VISION),
    )
    model = Gemma4ForConditionalGeneration(cfg)
    model.eval().to(torch.float32)
    strengthen_counts = strengthen_text(model)

    gen = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "embed_vision" in name:
                p.normal_(0.0, 0.02, generator=gen)
        for name, buf in model.named_buffers():
            if name.endswith("_min"):
                buf.copy_(torch.tensor(-2.0 + 0.1 * torch.randn(1, generator=gen).item()))
            elif name.endswith("_max"):
                buf.copy_(torch.tensor(2.0 + 0.1 * torch.randn(1, generator=gen).item()))

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        out = model(input_ids=ids, use_cache=False)  # text-only: no pixel_values
        last_logits = out.logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = {
        "note": "tiny-random Gemma4ForConditionalGeneration, use_bidirectional_attention='vision', "
                "TEXT-ONLY forward (degenerates to plain causal — see this script's own docstring); "
                "CPU fp32. num_kv_shared_layers=0 and hidden_size_per_layer_input=0 match the real "
                "26B-A4B checkpoint. Norms/layer_scalar strengthened with a seeded separate RNG.",
        "strengthened": strengthen_counts,
        "text_config": TEXT,
        "vision_config": VISION,
        "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,
        "n_new": N_NEW,
        "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}\n  argmax={golden['argmax']}  continuation={cont}")

    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}  (model_type={cfg.model_type!r})")


if __name__ == "__main__":
    main()

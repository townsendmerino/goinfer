#!/usr/bin/env python
"""Build a tiny-random Gemma 4 vision-language checkpoint (mirrors
pin_gemma3_vl_tiny.py's approach) — the P7 serving-integration fixture.

Saves a small Gemma4ForConditionalGeneration (tiny Gemma4VisionConfig + tiny
Gemma4TextConfig) as a real HF checkpoint, and pins a TEXT-ONLY forward golden
(no pixel_values). goinfer loads the text decoder from this VL checkpoint
(ignoring vision_tower/embed_vision) and must reproduce the golden bit-for-bit
— the P0 invariant, same as Gemma 3's.

DELIBERATELY sets num_kv_shared_layers=2 (of 4 layers) — this checkpoint shape
is what caught two real, pre-existing bugs in decoder/weights.go's safetensors
gemma4 loader this session (missing cross-layer KV-sharing skip; missing
per-layer FFN width discovery, both found only by loading a REAL E2B
checkpoint, which also has this shape). Exercising it here means the fix has
CI-repeatable coverage, not just a one-off real-checkpoint observation.
hidden_size_per_layer_input=0 (PLE-free, like the existing dense-twogeom
fixture) — safetensors PLE loading is a separate, undone gap (Phase 4); this
checkpoint deliberately stays outside it so the P7 gate isn't blocked by it.

Tiny -> sub-second; no download.

    ~/.venv-vl/bin/python scripts/pin_gemma4_vl_tiny.py
    -> testdata/gemma4_vl_tiny_text_golden.json
    -> testdata/gemma4-vl-tiny/   (HF safetensors checkpoint: vision + text)
"""
import json
import os

import torch
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "gemma4_vl_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-vl-tiny")

TEXT = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    # Same alternating sliding/full geometry as the PROVEN pin_gemma4_dense_twogeom.py
    # fixture (global_head_dim=512, attention_k_eq_v=True on the global layers),
    # extended from 2 to 4 layers so the last 2 (one of each type) can be the
    # cross-layer-KV-shared tail — layer 2 (sliding) reuses layer 0's KV, layer 3
    # (full) reuses layer 1's KV, exercising the TYPE-AWARE kvSrc selection
    # (forward_gemma4.go's kvSrc) a uniform-geometry fixture cannot.
    global_head_dim=512, num_global_key_value_heads=2, attention_k_eq_v=True,
    layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    sliding_window=4, max_position_embeddings=64, rms_norm_eps=1e-6,
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    hidden_size_per_layer_input=0,  # PLE-free — safetensors PLE loading is Phase 4, not this
    enable_moe_block=False,
    num_kv_shared_layers=2,  # layers 2,3 carry NO k_proj/v_proj — the real bug this exercises
    use_bidirectional_attention=None,  # E2B/E4B-class causal image-block attention
)
VISION = dict(
    hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=4,  # no GQA in this tower (aikit's Gemma4Encoder validate() requires nH*hd==hidden)
    head_dim=8, patch_size=4, pooling_kernel_size=3, position_embedding_size=16,
    rms_norm_eps=1e-6, use_clipped_linears=True,
    rope_parameters={"rope_theta": 100.0, "rope_type": "default"}, standardize=False,
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]  # text-only, no image-soft-token
N_NEW = 6


def strengthen_text(model):
    """Same degeneracy guard as pin_gemma4_dense_twogeom.py: HF init leaves
    norms/layer_scalar at identity, so a bug applying them (x1) would not move
    the golden. Separate generator => the linear weights + input stay
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

    # Give the vision->text projector real values (HF may init near-zero) and give
    # use_clipped_linears real finite bounds (matching the real E2B checkpoint's
    # own non-default bounds) so a later image-path gate actually exercises them,
    # not a degenerate all-zero/no-op case.
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
        "note": "tiny-random Gemma4ForConditionalGeneration, TEXT-ONLY forward; CPU fp32. "
                "num_kv_shared_layers=2/4 (the real bug this pins); hidden_size_per_layer_input=0 "
                "(PLE-free, safetensors PLE loading is a separate undone gap). Norms/layer_scalar "
                "strengthened with a seeded separate RNG so the golden pins those paths too.",
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

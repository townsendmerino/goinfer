#!/usr/bin/env python
"""Tiny Qwen3.5 image checkpoint + golden — P8a gate G1 (docs/measurements/p8a-qwen35-vl-2026-09/
preregistration.md). The real structure at toy size, from `Qwen3_5ForConditionalGeneration`:

  vision  biased Conv3d patch embed, learned 8x8 pos table (bilinear/align_corners), 2D rotary,
          full attention, LayerNorm, non-gated gelu-tanh MLP, erf merger, deepstack_visual_indexes [].
          Weights re-randomised with NON-ZERO biases / non-unit norms and the merger fc1 scaled x8
          (see aikit scripts/oracle/pin_qwen35_vision.py: cosine alone cannot see erf vs tanh).
  text    the 3:1 Gated-DeltaNet/softmax hybrid, head_dim 64 with partial_rotary_factor 0.25 (rotary
          dim 16, so 8 frequencies), mrope_section [3,3,2] summing to those 8, mrope_interleaved true.

Golden: image_features (tower output that replaces the placeholders), position_ids (m-RoPE, [3][seq]),
rope_delta, last_logits, argmax, and a greedy continuation of N_NEW tokens driven by HF's own generate.
The top-1/top-2 gap of every continuation step is printed and asserted >= 1e-3, so an exact-argmax gate
is not a coin flip on a near-tie.

    ~/g4venv/bin/python scripts/pin_qwen35_vl_tiny.py
    -> testdata/qwen35vl-tiny/  testdata/qwen35vl_tiny_image_golden.json
"""
import json
import os

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "testdata")
CKPT = os.path.join(TD, "qwen35vl-tiny")
OUT = os.path.join(TD, "qwen35vl_tiny_image_golden.json")

IMG, VSTART, VEND, VIDEO = 251, 250, 252, 253
TEXT = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=64,           # nH*hd = 256 != hidden
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,
    rms_norm_eps=1e-6, max_position_embeddings=512, tie_word_embeddings=False,
    hidden_act="silu", attention_bias=False, attn_output_gate=True,
    # rope_theta is 10, not the released 1e7, ON PURPOSE: at 1e7 every frequency but the first is so
    # slow that positions 3..9 barely rotate the query/key, and a contiguous-vs-interleaved m-RoPE
    # layout error or a dropped decode delta changes the logits by less than the cosine bar (measured
    # 2026-09-30: both mutants passed the gate at theta 1e7). The layout logic is theta-independent.
    rope_parameters={"rope_type": "default", "rope_theta": 10.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [3, 3, 2], "mrope_interleaved": True},
)
VISION = dict(depth=2, hidden_size=64, intermediate_size=96, num_heads=4, in_channels=3, patch_size=4,
              spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,   # == text hidden
              num_position_embeddings=64, hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[])
GRID = [1, 4, 6]                      # (t, h, w) in patch units: 6 merged tokens, non-square
PREFIX, SUFFIX = [2, 7, 42], [13, 88, 5, 100]
N_NEW = 8


def randomise_visual(v, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in v.named_parameters():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
        v.merger.linear_fc1.weight.mul_(8.0)


def main():
    torch.manual_seed(0)
    cfg = Qwen3_5Config(text_config=TEXT, vision_config=VISION, image_token_id=IMG, video_token_id=VIDEO,
                        vision_start_token_id=VSTART, vision_end_token_id=VEND, tie_word_embeddings=False)
    m = Qwen3_5ForConditionalGeneration(cfg).eval()
    randomise_visual(m.model.visual, 7)
    with torch.no_grad():
        m.lm_head.weight.mul_(12.0)   # widen logit margins so exact-argmax is not a near-tie coin flip
    assert len(m.model.visual.config.deepstack_visual_indexes or []) == 0
    assert torch.isfinite(m.model.visual.rotary_pos_emb.inv_freq).all()

    v = m.model.visual.config
    patch_dim = v.in_channels * v.temporal_patch_size * v.patch_size ** 2
    n_patches = GRID[0] * GRID[1] * GRID[2]
    n_img = n_patches // v.spatial_merge_size ** 2
    g = torch.Generator().manual_seed(1)
    pixel_values = torch.randn(n_patches, patch_dim, generator=g)
    grid = torch.tensor([GRID])
    ids = PREFIX + [VSTART] + [IMG] * n_img + [VEND] + SUFFIX
    input_ids = torch.tensor([ids])
    mm = torch.tensor([[1 if t == IMG else 0 for t in ids]], dtype=torch.int32)
    img_start = len(PREFIX) + 1

    with torch.no_grad():
        feats = m.model.get_image_features(pixel_values, grid, return_dict=True).pooler_output[0]
        out = m(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), pixel_values=pixel_values,
                image_grid_thw=grid, mm_token_type_ids=mm)
        pos, delta = m.model.get_rope_index(input_ids, mm, image_grid_thw=grid,
                                            attention_mask=torch.ones_like(input_ids))
        last = out.logits[0, -1]
        gen = m.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                         pixel_values=pixel_values, image_grid_thw=grid, mm_token_type_ids=mm,
                         max_new_tokens=N_NEW, do_sample=False, min_new_tokens=N_NEW,
                         output_scores=True, return_dict_in_generate=True)
    cont = gen.sequences[0, len(ids):].tolist()
    gaps = []
    for sc in gen.scores:
        t2 = sc[0].topk(2).values
        gaps.append(float(t2[0] - t2[1]))
    top2 = last.topk(2).values
    print("continuation", cont)
    print("top1-top2 gaps:", ["%.3f" % x for x in gaps], "last-position gap %.3f" % float(top2[0] - top2[1]))
    assert min(gaps) >= 1e-3 and float(top2[0] - top2[1]) >= 1e-3, "near-tie: widen the logit margin"
    assert int(last.argmax()) == cont[0], "generate and forward disagree on the first token"

    os.makedirs(CKPT, exist_ok=True)
    m.save_pretrained(CKPT, safe_serialization=True)
    json.dump(dict(
        input_ids=ids, image_token_id=IMG, vision_start_token_id=VSTART, image_token_start=img_start,
        n_image_tokens=n_img, grid_thw=[GRID], pixel_values=pixel_values.reshape(-1).tolist(),
        image_features=feats.reshape(-1).tolist(), position_ids=pos[:, 0].tolist(),
        rope_delta=int(delta[0, 0]), argmax=int(last.argmax()), last_logits=last.tolist(),
        continuation_ids=cont, n_new=N_NEW,
    ), open(OUT, "w"))
    print(f"wrote {OUT} and {CKPT}: {len(ids)} tokens, {n_img} image tokens, rope_delta {int(delta[0,0])}")


if __name__ == "__main__":
    main()

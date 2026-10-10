#!/usr/bin/env python
"""Tiny random Qwen3-VL MoE checkpoint + golden for goinfer's qwen3_vl_moe decoder (S10, G-S10q-a;
docs/tasks/task-multimodal-support-2026-10.md, "S10, Qwen3-VL MoE").

A small Qwen3VLMoeForConditionalGeneration (transformers' own, float32, eager attention): a 4-layer MoE text decoder
(8 experts, top 2, interleaved m-RoPE section [4, 2, 2]) and a tiny vision config the golden never runs. Every norm
randomised. The experts are written to disk in transformers 4.57's layout, as the released Qwen3-VL-30B-A3B stores them
(gate_up_proj [E, hidden, 2*inter], down_proj [E, inter, hidden]): transformers 5.12's module holds the transpose.

Golden:
  text   a text-only prompt, every position's logits (input_ids through the whole model);
  image  a prompt with one 4x6-patch image (6 merged rows): random merged rows and three random DeepStack sets spliced
         by hand into HF's own text model (inputs_embeds, visual_pos_masks, deepstack_visual_embeds), the m-RoPE
         positions from HF's get_rope_index; every position's logits.

    ~/.venv-vl/bin/python scripts/pin_qwen3vlmoe_tiny.py <out dir>
    -> <out>/qwen3vlmoe-tiny/{config.json,model.safetensors}  <out>/qwen3vlmoe_tiny_golden.json
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

OUT = sys.argv[1]
IMG, VSTART, VEND = 500, 498, 499
TEXT = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, moe_intermediate_size=24, num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
            decoder_sparse_step=1, mlp_only_layers=[], vocab_size=512, rms_norm_eps=1e-6, rope_theta=10000.0,
            rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
            max_position_embeddings=256, hidden_act="silu", attention_bias=False, tie_word_embeddings=False)
VISION = dict(depth=3, hidden_size=32, intermediate_size=48, num_heads=2, in_channels=3, patch_size=4, spatial_merge_size=2,
              temporal_patch_size=2, out_hidden_size=64, num_position_embeddings=16, deepstack_visual_indexes=[0, 1, 2])


def main():
    torch.manual_seed(0)
    cfg = Qwen3VLMoeConfig(text_config=TEXT, vision_config=VISION, image_token_id=IMG, vision_start_token_id=VSTART,
                           vision_end_token_id=VEND, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    m = Qwen3VLMoeForConditionalGeneration(cfg).eval()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    lm = m.model.language_model
    gold = {}
    # text
    ids = torch.randint(0, 490, (1, 14), generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        gold["text"] = dict(ids=ids[0].tolist(), logits=m(input_ids=ids).logits[0].flatten().tolist())
    # image: 4x6 patches -> 6 merged rows
    pre, post = [5, 17, 33], [41, 52, 66, 70]
    ids = torch.tensor([pre + [VSTART] + [IMG] * 6 + [VEND] + post])
    grid = torch.tensor([[1, 4, 6]])
    mm = (ids == IMG).int()
    pos, _ = m.model.get_rope_index(ids, mm, image_grid_thw=grid)
    r = torch.Generator().manual_seed(13)
    merged = 0.5 * torch.randn(6, 64, generator=r)
    deep = [0.3 * torch.randn(6, 64, generator=r) for _ in range(3)]
    with torch.no_grad():
        emb = lm.embed_tokens(ids).clone()
        mask = ids == IMG
        emb[mask] = merged
        h = lm(inputs_embeds=emb, position_ids=pos, visual_pos_masks=mask, deepstack_visual_embeds=deep).last_hidden_state
        logits = m.lm_head(h)[0]
    gold["image"] = dict(ids=ids[0].tolist(), grid=grid[0].tolist(), positions=pos[:, 0, :].T.tolist(),
                         merged=merged.flatten().tolist(), deepstack=[d.flatten().tolist() for d in deep],
                         logits=logits.flatten().tolist())
    # save: the experts in the released (4.57) layout
    sd = {}
    for k, v in m.state_dict().items():
        if k.endswith("mlp.experts.gate_up_proj") or k.endswith("mlp.experts.down_proj"):
            v = v.transpose(1, 2)  # 5.12 [E, 2I, H] / [E, H, I] -> 4.57 [E, H, 2I] / [E, I, H]
        sd[k] = v.contiguous()
    gu = [v.shape for k, v in sd.items() if k.endswith("layers.0.mlp.experts.gate_up_proj")][0]
    assert list(gu) == [8, 64, 48], gu  # [E, hidden, 2*inter], not square: the shape decides the layout, as on the 30B
    ck = os.path.join(OUT, "qwen3vlmoe-tiny")
    os.makedirs(ck, exist_ok=True)
    save_file(sd, os.path.join(ck, "model.safetensors"))
    cfg.save_pretrained(ck)
    json.dump(dict(note="tiny Qwen3VLMoeForConditionalGeneration, transformers " + __import__("transformers").__version__
                   + ", float32, eager; experts saved in the 4.57 layout", vocab=512, hidden=64, image_token=IMG, merge=2, **gold),
              open(os.path.join(OUT, "qwen3vlmoe_tiny_golden.json"), "w"))
    print("wrote", ck, "keys", [k for k in sd if "layers.0.mlp" in k], "gate_up", list(gu))


if __name__ == "__main__":
    main()

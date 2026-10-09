#!/usr/bin/env python
"""Tiny Pixtral vision tower + golden for aikit's PixtralVisionEncoder (S10, Ministral 3; G-S10m-b's tiny half,
docs/tasks/task-multimodal-support-2026-10.md).

A random PixtralVisionModel (transformers' own, float32, eager attention) of the real structure at toy size: patch conv
without bias, RMSNorm ln_pre, pre-norm blocks with separate q/k/v/o and a SiLU-gated MLP, the 2-D RoPE whose rows take
the even-indexed frequencies and columns the odd-indexed ones. Every norm weight is randomised (a 1.0 norm hides a
swapped or skipped one). Two non-square images of different sizes go through in ONE call, so the block-diagonal mask is
exercised, and each image also alone.

Golden stages, in order, for the two-image call: s_conv (the patch conv, each image's patches row-major), s_lnpre, then
each block's output (s_layer0, s_layer1, ...; the last is last_hidden_state). Plus last_hidden_state for each image alone.

    ~/g4venv/bin/python scripts/pin_pixtral_vision.py <out dir>
    -> <out>/pixtral-vision-tiny/{config.json,model.safetensors}  <out>/pixtral_vision_golden.json
    (aikit: testdata/pixtral-vision-tiny/ and testdata/pixtral_vision_golden.json)
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import PixtralVisionConfig, PixtralVisionModel

OUT = sys.argv[1]
CFG = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, head_dim=16,
           num_channels=3, patch_size=4, image_size=64, hidden_act="silu", rope_theta=10000.0,
           rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
SIZES = [(12, 20), (16, 8)]  # (h, w) pixels: 3x5 and 4x2 patches


def main():
    torch.manual_seed(0)
    cfg = PixtralVisionConfig(**CFG)
    cfg._attn_implementation = "eager"
    m = PixtralVisionModel(cfg).eval()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("norm.weight") or name.endswith("ln_pre.weight"):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    imgs = [torch.randn(3, h, w, generator=torch.Generator().manual_seed(10 + i)) for i, (h, w) in enumerate(SIZES)]
    H, W = max(h for h, _ in SIZES), max(w for _, w in SIZES)
    pv = torch.zeros(len(imgs), 3, H, W)
    for i, im in enumerate(imgs):
        pv[i, :, :im.shape[1], :im.shape[2]] = im
    sizes = torch.tensor([[h, w] for h, w in SIZES])
    with torch.no_grad():
        out = m(pixel_values=pv, image_sizes=sizes, output_hidden_states=True)
        conv = []
        for im in imgs:  # the patch conv per image, cropped and flattened row-major, as the model does
            e = m.patch_conv(im.unsqueeze(0))[0]  # [hidden, rows, cols]
            conv.append(e.flatten(1).T)
        s_conv = torch.cat(conv)
        hs = out.hidden_states  # (post-ln_pre input, each layer)
        alone = [m(pixel_values=im.unsqueeze(0), image_sizes=torch.tensor([[im.shape[1], im.shape[2]]])).last_hidden_state[0]
                 for im in imgs]
    stages = {"s_conv": s_conv, "s_lnpre": hs[0][0]}
    for i in range(cfg.num_hidden_layers):
        stages[f"s_layer{i}"] = hs[i + 1][0]
    assert torch.allclose(hs[-1][0], out.last_hidden_state[0])

    ck = os.path.join(OUT, "pixtral-vision-tiny")
    os.makedirs(ck, exist_ok=True)
    save_file({"vision_tower." + k: v.contiguous() for k, v in m.state_dict().items()}, os.path.join(ck, "model.safetensors"))
    json.dump({"model_type": "mistral3", "vision_config": {**CFG, "model_type": "pixtral"}}, open(os.path.join(ck, "config.json"), "w"), indent=1)
    json.dump(dict(
        note="tiny Pixtral tower, transformers " + __import__("transformers").__version__ + ", float32, eager attention",
        sizes=SIZES, grids=[[h // CFG["patch_size"], w // CFG["patch_size"]] for h, w in SIZES],
        images=[im.flatten().tolist() for im in imgs], hidden=CFG["hidden_size"],
        stages={k: v.flatten().tolist() for k, v in stages.items()},
        alone=[a.flatten().tolist() for a in alone],
    ), open(os.path.join(OUT, "pixtral_vision_golden.json"), "w"))
    print("wrote", ck, "and the golden; stages", list(stages))


if __name__ == "__main__":
    main()

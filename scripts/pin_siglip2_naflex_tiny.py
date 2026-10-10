#!/usr/bin/env python
"""Tiny SigLIP2 NaFlex tower + golden for aikit's Siglip2NaFlexEncoder, ResizeBilinearAA and ResizeBilinearAAFloat (S10,
LFM2.5-VL; G-S10l-b's tiny half, docs/tasks/task-multimodal-support-2026-10.md).

A random Siglip2VisionModel (transformers' own, float32, eager attention) at toy size, every LayerNorm weight and bias
randomised (a 1.0 norm hides a swapped or skipped one), a 4x4 position table. Three tiles, each run alone and also all
three batched with HF's padding and key mask (the two must agree; the script asserts it):
  grow   6x5 patches  (the table upscaled: antialias changes nothing),
  shrink 2x3 patches  (the table downscaled: antialias matters),
  same   4x4 patches  (the table as it is).
Each tile's image is random, patchified by HF's own convert_image_to_patches (LFM2-VL's processor helper), so the
patch order is checked too. Golden stages per tile: embeddings (patch embedding plus positions), each block's output,
last_hidden_state (post-layernorm).

Also the two resizes: torchvision's uint8 bilinear with antialias on a random RGB image to four sizes (two shrinking,
two growing, mixed), and torch's F.interpolate(bilinear, align_corners=False, antialias=True) on a random float grid.

    ~/.venv-vl/bin/python scripts/pin_siglip2_naflex_tiny.py <out dir>
    -> <out>/siglip2-naflex-tiny/{config.json,model.safetensors}  <out>/siglip2_naflex_golden.json
    (aikit: testdata/siglip2-naflex-tiny/ and testdata/siglip2_naflex_golden.json)
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF
from safetensors.torch import save_file
from transformers import Siglip2VisionConfig, Siglip2VisionModel
from transformers.models.lfm2_vl.image_processing_lfm2_vl import convert_image_to_patches

OUT = sys.argv[1]
PATCH = 4
CFG = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2, num_channels=3,
           patch_size=PATCH, num_patches=16, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6, vision_use_head=False)
TILES = {"grow": (6, 5), "shrink": (2, 3), "same": (4, 4)}


def main():
    torch.manual_seed(0)
    cfg = Siglip2VisionConfig(**CFG)
    cfg._attn_implementation = "eager"
    m = Siglip2VisionModel(cfg).eval()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "layer_norm" in name or "layernorm" in name:
                p.copy_((1.0 + 0.2 * torch.randn(p.shape, generator=g)) if name.endswith("weight") else 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    vm = m  # transformers 5.12: the vision model itself (embeddings, encoder, post_layernorm)
    tiles, alone = {}, {}
    for k, (name, (gh, gw)) in enumerate(TILES.items()):
        img = torch.randn(1, 3, gh * PATCH, gw * PATCH, generator=torch.Generator().manual_seed(10 + k))
        patches = convert_image_to_patches(img, PATCH)[0]  # [gh*gw, P*P*C]
        st = {}
        hooks = [vm.embeddings.register_forward_hook(lambda mod, i, o: st.__setitem__("embeddings", o[0].clone()))]
        for L, blk in enumerate(vm.encoder.layers):
            hooks.append(blk.register_forward_hook(
                lambda mod, i, o, L=L: st.__setitem__(f"layer{L}", (o[0] if isinstance(o, tuple) else o)[0].clone())))
        with torch.no_grad():
            out = m(pixel_values=patches[None], pixel_attention_mask=torch.ones(1, gh * gw, dtype=torch.int32),
                    spatial_shapes=torch.tensor([[gh, gw]]))
        for h in hooks:
            h.remove()
        st["last"] = out.last_hidden_state[0]
        tiles[name] = dict(grid=[gh, gw], image=img[0].flatten().tolist(), patches=patches.flatten().tolist(),
                           stages={s: v.flatten().tolist() for s, v in st.items()})
        alone[name] = out.last_hidden_state[0]
    # The three batched with padding and the key mask, as LFM2-VL runs them: each tile's rows must equal its own run.
    n = max(gh * gw for gh, gw in TILES.values())
    pv = torch.zeros(len(TILES), n, 3 * PATCH * PATCH)
    mask = torch.zeros(len(TILES), n, dtype=torch.int32)
    for i, (name, (gh, gw)) in enumerate(TILES.items()):
        pv[i, :gh * gw] = torch.tensor(tiles[name]["patches"]).reshape(gh * gw, -1)
        mask[i, :gh * gw] = 1
    with torch.no_grad():
        b = m(pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=torch.tensor(list(TILES.values()))).last_hidden_state
    batched_diff = 0.0
    for i, (name, (gh, gw)) in enumerate(TILES.items()):
        batched_diff = max(batched_diff, (b[i, :gh * gw] - alone[name]).abs().max().item())
    assert batched_diff < 1e-5, batched_diff

    # The resizes.
    src = torch.randint(0, 256, (3, 37, 53), dtype=torch.uint8, generator=torch.Generator().manual_seed(21))
    resizes = []
    for th, tw in [(24, 70), (64, 96), (20, 20), (37, 107)]:
        r = tvF.resize(src, [th, tw], interpolation=tvF.InterpolationMode.BILINEAR, antialias=True)
        resizes.append(dict(size=[th, tw], out=r.permute(1, 2, 0).flatten().tolist()))
    grid = torch.randn(1, 5, 4, 4, generator=torch.Generator().manual_seed(22))  # (1, C, H, W)
    floats = []
    for th, tw in [(2, 3), (6, 5), (3, 7)]:
        r = F.interpolate(grid, size=(th, tw), mode="bilinear", align_corners=False, antialias=True)
        floats.append(dict(size=[th, tw], out=r[0].permute(1, 2, 0).flatten().tolist()))

    ck = os.path.join(OUT, "siglip2-naflex-tiny")
    os.makedirs(ck, exist_ok=True)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, os.path.join(ck, "model.safetensors"))
    json.dump({**CFG, "model_type": "siglip2_vision_model"}, open(os.path.join(ck, "config.json"), "w"), indent=1)
    json.dump(dict(
        note="tiny Siglip2VisionModel, transformers " + __import__("transformers").__version__ + ", float32, eager; torchvision "
             + __import__("torchvision").__version__,
        hidden=CFG["hidden_size"], patch=PATCH, tiles=tiles, batched_max_abs_diff=batched_diff,
        resize_src=src.permute(1, 2, 0).flatten().tolist(), resize_src_size=[37, 53], resizes=resizes,
        float_src=grid[0].permute(1, 2, 0).flatten().tolist(), float_src_size=[4, 4, 5], floats=floats,
    ), open(os.path.join(OUT, "siglip2_naflex_golden.json"), "w"))
    print("wrote", ck, "batched-vs-alone max|diff|", batched_diff, "tensors", list(m.state_dict())[:4])


if __name__ == "__main__":
    main()

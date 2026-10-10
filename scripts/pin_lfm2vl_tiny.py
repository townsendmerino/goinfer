#!/usr/bin/env python
"""Goldens for goinfer's LFM2-VL preprocessing layout and projector (S10, LFM2.5-VL; docs/tasks/task-multimodal-support-2026-10.md,
G-S10l-a's layout half and G-S10l-b's tiny projector).

1. The layout: HF's Lfm2VlProcessor (the real checkpoint's processor_config.json and tokenizer) on blank images of many sizes,
   recording what decides the prompt: tiled or not, the grid, each tile's size, and the image's token counts in the ids.
   The sizes include an exact aspect tie (1250x1000: 1x2 and 2x1 are equally far), strips both ways, the smart-resize floor
   and ceiling, and the tile budget's edge.
2. The projector: HF's Lfm2VlMultiModalProjector (float32, random weights and biases) on two tiles' tower output of
   different grids, the weights in the golden so the Go test runs without a checkpoint (CI).

    ~/.venv-vl/bin/python scripts/pin_lfm2vl_tiny.py <checkpoint dir> <out dir>
    -> <out>/lfm2vl_layout_golden.json  <out>/lfm2vl_projector_golden.json
"""
import json
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Lfm2VlConfig
from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector

CKPT, OUT = sys.argv[1], sys.argv[2]
SIZES = [(56, 84), (20, 1000), (1000, 20), (300, 300), (400, 400), (512, 512), (600, 600), (700, 500), (896, 896),
         (900, 1200), (1200, 1000), (1000, 1250), (768, 1024), (2000, 3000), (64, 4000), (4000, 64), (511, 1025),
         (360, 640), (1080, 1920), (100, 3000), (720, 720)]  # (h, w)


def layout():
    proc = AutoProcessor.from_pretrained(CKPT)
    tok = proc.tokenizer
    img_id = tok.convert_tokens_to_ids("<image>")
    rows = []
    for h, w in SIZES:
        im = Image.new("RGB", (w, h), (120, 30, 200))
        enc = proc(text="<image>", images=[im], return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        # the <image> runs, in order
        runs, n = [], 0
        for t in ids + [-1]:
            if t == img_id:
                n += 1
            elif n:
                runs.append(n)
                n = 0
        ie = proc.image_processor(images=[im], return_tensors="pt")  # the grid and tiles (the processor drops image_rows)
        shapes = ie["spatial_shapes"].tolist()
        if [a * b // 4 for a, b in shapes] != runs:
            sys.exit(f"{h}x{w}: tile token counts {[a * b // 4 for a, b in shapes]} disagree with the ids' runs {runs}")
        rows.append(dict(size=[h, w], rows=int(ie["image_rows"][0]), cols=int(ie["image_cols"][0]),
                         tile_patches=shapes, runs=runs, ids=len(ids),
                         markers=[tok.convert_ids_to_tokens(t) for t in ids if t != img_id]))
    return dict(note="Lfm2VlProcessor from " + os.path.basename(CKPT.rstrip("/")) + ", transformers " + __import__("transformers").__version__,
                cases=rows)


def projector():
    torch.manual_seed(0)
    cfg = Lfm2VlConfig(vision_config={"model_type": "siglip2_vision_model", "hidden_size": 12, "intermediate_size": 16,
                                      "num_hidden_layers": 1, "num_attention_heads": 2, "patch_size": 4, "num_patches": 16},
                       text_config={"model_type": "lfm2", "hidden_size": 10, "intermediate_size": 16, "num_hidden_layers": 1,
                                    "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 32, "layer_types": ["full_attention"]},
                       downsample_factor=2, projector_hidden_size=14, projector_hidden_act="gelu", projector_bias=True,
                       projector_use_layernorm=False)
    p = Lfm2VlMultiModalProjector(cfg).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name, w in p.named_parameters():
            w.copy_(0.3 * torch.randn(w.shape, generator=g))
    tiles = []
    for k, (gh, gw) in enumerate([(4, 6), (2, 2)]):
        x = torch.randn(gh * gw, 12, generator=torch.Generator().manual_seed(7 + k))
        with torch.no_grad():
            y = p(x.reshape(1, gh, gw, 12)).reshape(-1, 10)
        tiles.append(dict(grid=[gh, gw], input=x.flatten().tolist(), output=y.flatten().tolist()))
    return dict(note="tiny Lfm2VlMultiModalProjector, transformers " + __import__("transformers").__version__,
                vision_hidden=12, text_hidden=10, mid=14, factor=2, tiles=tiles,
                weights={k: v.flatten().tolist() for k, v in p.state_dict().items()})


def main():
    os.makedirs(OUT, exist_ok=True)
    json.dump(layout(), open(os.path.join(OUT, "lfm2vl_layout_golden.json"), "w"), indent=0)
    json.dump(projector(), open(os.path.join(OUT, "lfm2vl_projector_golden.json"), "w"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

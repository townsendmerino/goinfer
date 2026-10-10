#!/usr/bin/env python
"""Tiny Mistral 3 multimodal projector + golden for goinfer's multimodal.PixtralProjector (S10, Ministral 3;
docs/tasks/task-multimodal-support-2026-10.md).

HF's own Mistral3MultiModalProjector (transformers' modeling_mistral3, float32) at toy size, random weights and a
randomised norm, over two images of different patch grids in one call (the merger splits per image by image_sizes).
Golden: the tower-side input [Σ rows·cols, vision_hidden], the grids, and the projector's output.

    ~/g4venv/bin/python scripts/pin_pixtral_projector.py <out dir>
    -> <out>/pixtral-projector-tiny/{config.json,model.safetensors}  <out>/pixtral_projector_golden.json
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

OUT = sys.argv[1]
VH, TH, PATCH, MERGE = 16, 24, 4, 2
GRIDS = [(4, 6), (2, 4)]  # patch rows, cols


def main():
    torch.manual_seed(0)
    cfg = Mistral3Config(
        vision_config={"model_type": "pixtral", "hidden_size": VH, "intermediate_size": 32, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "head_dim": 8, "patch_size": PATCH, "image_size": 64},
        text_config={"model_type": "ministral3", "hidden_size": TH, "intermediate_size": 48, "num_hidden_layers": 1,
                     "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 12, "vocab_size": 64,
                     "rms_norm_eps": 1e-5},
        spatial_merge_size=MERGE, multimodal_projector_bias=False, projector_hidden_act="gelu", vision_feature_layer=-1)
    p = Mistral3MultiModalProjector(cfg).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name, w in p.named_parameters():
            w.copy_((1.0 + 0.2 * torch.randn(w.shape, generator=g)) if name.endswith("norm.weight") else 0.2 * torch.randn(w.shape, generator=g))
    n = sum(r * c for r, c in GRIDS)
    x = torch.randn(n, VH, generator=torch.Generator().manual_seed(7))
    sizes = torch.tensor([[r * PATCH, c * PATCH] for r, c in GRIDS])
    with torch.no_grad():
        y = p(x, sizes)
    ck = os.path.join(OUT, "pixtral-projector-tiny")
    os.makedirs(ck, exist_ok=True)
    save_file({"multi_modal_projector." + k: v.contiguous() for k, v in p.state_dict().items()}, os.path.join(ck, "model.safetensors"))
    d = cfg.to_dict()
    json.dump({"model_type": "mistral3", "spatial_merge_size": MERGE, "projector_hidden_act": "gelu",
               "multimodal_projector_bias": False, "text_config": {"hidden_size": TH, "rms_norm_eps": 1e-5},
               "vision_config": {"hidden_size": VH, "patch_size": PATCH}}, open(os.path.join(ck, "config.json"), "w"), indent=1)
    json.dump(dict(note="tiny Mistral3MultiModalProjector, transformers " + __import__("transformers").__version__,
                   grids=GRIDS, vision_hidden=VH, text_hidden=TH, input=x.flatten().tolist(), output=y.flatten().tolist(),
                   # the weights too, so the Go test runs where the gitignored safetensors are absent (CI)
                   weights={k: v.flatten().tolist() for k, v in p.state_dict().items()}),
              open(os.path.join(OUT, "pixtral_projector_golden.json"), "w"))
    print("wrote", ck, "output", tuple(y.shape))


if __name__ == "__main__":
    main()

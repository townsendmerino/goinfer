#!/usr/bin/env python
"""Generate a REAL-SIZED random SigLIP vision tower for the ViT throughput benchmark
(docs/BENCH-gpu.md). Unlike the tiny parity fixture (pin_siglip_vision.py, hidden 32), this
is a ViT-B/16-ish tower whose forward is real compute — so the GPU-vs-CPU crossover is
meaningful. Weights are random: throughput does not depend on weight VALUES, and the
benchmark gates parity as GPU-vs-CPU on the SAME tower (not against a Python golden), so a
random tower is exactly right and needs no download.

    ~/tmcode/aikit/.venv/bin/python scripts/oracle/gen_siglip_bench.py
    -> testdata/siglip-bench/   (HF safetensors checkpoint; gitignored)
"""
import os
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

# Three parents — see the note in pin_siglip_vision.py; "..", "testdata" lands in
# scripts/testdata, which no Go test reads.
TD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "testdata")

# Two real sizes, so the ViT slice shows the GPU win GROWING with tower size (bigger ops
# amortize the per-op dispatch floor): a ViT-B/16-ish tower (196 patches) and a larger one
# (~576 patches at higher hidden), both gelu-tanh like Gemma-3's SigLIP.
CONFIGS = {
    "siglip-bench": dict(hidden_size=512, intermediate_size=2048, num_hidden_layers=12,
                         num_attention_heads=8, image_size=224, patch_size=16),
    "siglip-bench-l": dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                           num_attention_heads=12, image_size=384, patch_size=16),
}
COMMON = dict(num_channels=3, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6, attention_dropout=0.0)


def main():
    for name, cfg in CONFIGS.items():
        torch.manual_seed(0)
        model = SiglipVisionModel(SiglipVisionConfig(**cfg, **COMMON)).eval().to(torch.float32)
        out = os.path.join(TD, name)
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out, safe_serialization=True)
        grid = cfg["image_size"] // cfg["patch_size"]
        print(f"saved {out}  hidden={cfg['hidden_size']} layers={cfg['num_hidden_layers']} patches={grid * grid}")


if __name__ == "__main__":
    main()

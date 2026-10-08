#!/usr/bin/env python3
"""G-S10b's tiny half (docs/tasks/task-multimodal-support-2026-10.md, S10): a tiny Qwen3-VL vision tower WITH DeepStack
and its reference, for CI. transformers' Qwen3VLVisionModel at the tiny Qwen3-VL fixture's vision_config (hidden 32,
2 heads, 3 blocks, DeepStack at blocks 0 and 1), seeded, float32, on random pre-patchified pixel values for two images
(grids 1x4x6 and 1x6x4). Sharpened so planted defects cannot hide in a degenerate fixture (S3's lesson: zero biases and
init-scale attention hid 5 of its 7 defects):
  - every LayerNorm's weight and bias randomised (an all-ones norm hides a dropped or misplaced norm: the DeepStack
    merger's post-shuffle norm is a planted defect this must catch);
  - every other bias randomised, nonzero (re-pinned 2026-10-07 after the Cowork audit found them all exactly 0), at the
    weights' own init scale, 0.02. Larger drowns the signal in the merger: the post-shuffle-norm-dropped defect read
    0.998 at a 0.3 bias scale, 0.95 at 0.05, 0.90 at 0.02 (it was 0.89 / 0.82 with zero biases);
  - the attention qkv projection scaled up 6x, so attention is not near-uniform;
  - a third block, so DeepStack's taps taken one block late ([1, 2]) stay inside the tower and are measured as numbers
    rather than refused (G-S10e's planted defect).
Writes:

  <aikit>/testdata/qwen3vl-vision-tiny/config.json, model.safetensors (committed in aikit, which tests it:
      vision/qwen3_deepstack_test.go): the tower under the real checkpoint's names (model.visual.*), so
      LoadQwen3VisionEncoder reads it as it reads Qwen3-VL-2B;
  <aikit>/testdata/qwen3vl-vision-tiny/golden.json.gz: the pixel values, the grids, the main merged rows and each
      DeepStack set.

Run: python3 scripts/pin_qwen3vl_vision_tiny.py [--out <dir>]   (default: the sibling aikit checkout)
"""
import gzip
import json
import os

import torch
from safetensors.torch import save_file
import transformers
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "aikit", "testdata", "qwen3vl-vision-tiny")
VC = {"depth": 3, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "in_channels": 3, "patch_size": 14,
      "spatial_merge_size": 2, "temporal_patch_size": 2, "out_hidden_size": 64, "num_position_embeddings": 16,
      "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": [0, 1]}
GRIDS = [[1, 4, 6], [1, 6, 4]]


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    OUT = ap.parse_args().out
    torch.manual_seed(20261007)
    cfg = Qwen3VLVisionConfig(**VC)
    cfg._attn_implementation = "sdpa"
    m = Qwen3VLVisionModel(cfg).float().eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if ".norm" in name or name.startswith("norm"):
                p.copy_((1.0 if name.endswith("weight") else 0.0) + 0.3 * torch.randn(p.shape, generator=g))
            elif name.endswith(".bias"):
                p.copy_(0.02 * torch.randn(p.shape, generator=g))
            elif name.endswith("attn.qkv.weight"):
                p.mul_(6.0)
        zero = [n for n, p in m.named_parameters() if n.endswith(".bias") and not p.abs().gt(0).all()]
        assert not zero, f"biases with a zero entry: {zero}"
        assert any(n.endswith("attn.qkv.weight") for n, _ in m.named_parameters()), "no attn.qkv.weight: the scaling hit nothing"
    patch_dim = VC["in_channels"] * VC["temporal_patch_size"] * VC["patch_size"] ** 2
    n = sum(t * h * w for t, h, w in GRIDS)
    px = torch.randn(n, patch_dim, generator=g)
    with torch.no_grad():
        out = m(px, torch.tensor(GRIDS))
    deep = out.deepstack_features
    assert len(deep) == len(VC["deepstack_visual_indexes"])
    os.makedirs(OUT, exist_ok=True)
    save_file({"model.visual." + k: v.contiguous() for k, v in m.state_dict().items() if "inv_freq" not in k},
              os.path.join(OUT, "model.safetensors"))
    json.dump({"model_type": "qwen3_vl", "vision_config": VC}, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    golden = {"transformers": transformers.__version__, "torch": torch.__version__, "grids": GRIDS,
              "pixel_values": px.flatten().tolist(), "merged": out.pooler_output.flatten().tolist(),
              "deepstack": [d.flatten().tolist() for d in deep], "out_hidden": VC["out_hidden_size"]}
    with gzip.open(os.path.join(OUT, "golden.json.gz"), "wt") as f:
        json.dump(golden, f)
    print(f"{n} patches, {out.pooler_output.shape[0]} merged rows, {len(deep)} DeepStack sets -> {OUT}")


if __name__ == "__main__":
    main()

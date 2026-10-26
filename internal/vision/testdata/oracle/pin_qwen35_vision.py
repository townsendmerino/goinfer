#!/usr/bin/env python
"""Pin the Qwen3.5+ vision tower (Qwen3_5VisionModel — the Qwen3-VL tower with DeepStack
removed) as an aikit parity golden. The third ViT family after SigLIP and Qwen2.5-VL.

Two modes, both CPU float32:

  tiny (default; sub-second, no download; the checkpoint is COMMITTED with its golden, since each golden is valid only against its exact weights: regenerating rewrites both, so commit both together)
      A small random Qwen3_5VisionModel with REAL structure — biased Conv3d patch embed, learned
      square pos table resampled by bilinear/align_corners interpolation, 2D rotary, full
      attention, LayerNorm with bias, non-gated gelu-tanh MLP, erf merger, deepstack []. Weights
      are re-randomised with NON-ZERO biases and non-unit norm weights so a dropped bias or a wrong
      norm cannot hide behind a default init.
          python scripts/oracle/pin_qwen35_vision.py
          -> testdata/qwen35vl_vision_golden.json
          -> testdata/qwen35vl-vision-tiny/     (config.json + model.safetensors, "model.visual." keys)

  --real DIR  (the pre-registered G0 gate: docs/measurements/p8a-qwen35-vl-2026-09/preregistration.md)
      The real Qwen3.5-0.8B tower weights, random pixel_values, four grids. The golden is large
      (~1.4M floats) so it is written OUTSIDE the repo, gzip-compressed:
          python scripts/oracle/pin_qwen35_vision.py --real ~/models/qwen3.5-0.8b \
              --out ~/models/qwen35vl_tower_golden/qwen35vl_tower_golden.json.gz

Each golden carries three stages, so a failure names its stage:
  s1  patch_embed + interpolated pos_embed            [n_patches, hidden]
  s2  last block output (HF last_hidden_state)        [n_patches, hidden]
  s3  merged features (HF pooler_output)              [n_merged, out_hidden]
plus pixel_values and grid_thw, and the bilinear tap indices/weights and rotary position ids that
the Go side must reproduce exactly.
"""
import argparse
import gzip
import json
import os

import torch
from safetensors.torch import save_file

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.vision_utils import (get_vision_interpolation_indices_and_weights,
                                       get_vision_position_ids)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTDATA = os.path.join(HERE, "..", "testdata")

TINY = dict(depth=2, hidden_size=64, intermediate_size=96, num_heads=4, in_channels=3,
            patch_size=4, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=48,
            num_position_embeddings=64, hidden_act="gelu_pytorch_tanh",
            deepstack_visual_indexes=[])
TINY_GRIDS = [[1, 4, 6], [1, 2, 8]]                       # two images, non-square, one packed call
REAL_GRIDS = [[1, 16, 16], [1, 12, 20], [1, 32, 8], [1, 2, 4]]


def randomise(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))


def stages(v, pv, grid):
    """Run the tower once, capturing S1 by recomputing it from the same helpers forward() uses."""
    cfg = v.config
    with torch.no_grad():
        idx, wts = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=v.num_grid_per_side, mode=v.interpolation_mode,
            align_corners=v.interpolation_align_corners, spatial_merge_size=cfg.spatial_merge_size)
        s1 = v.patch_embed(pv)
        s1 = s1 + (v.pos_embed(idx) * wts[:, :, None]).sum(1).to(s1.dtype)
        out = v(pv, grid_thw=grid, return_dict=True)
        pos = get_vision_position_ids(grid, cfg.spatial_merge_size)
    return s1, out.last_hidden_state, out.pooler_output, idx, wts, pos


def dump(v, grids, seed, path, extra):
    cfg = v.config
    patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2
    g = torch.Generator().manual_seed(seed)
    grid = torch.tensor(grids, dtype=torch.long)
    n = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    pv = torch.randn(n, patch_dim, generator=g)
    s1, s2, s3, idx, wts, pos = stages(v, pv, grid)
    f = lambda t: t.reshape(-1).tolist()
    gold = dict(grid_thw=grids, n_patches=n, n_merged=n // cfg.spatial_merge_size ** 2,
                hidden=cfg.hidden_size, out_hidden=cfg.out_hidden_size,
                pixel_values=f(pv), s1=f(s1), s2=f(s2), s3=f(s3),
                interp_indices=idx.reshape(-1).tolist(), interp_weights=f(wts),
                pos_ids=pos.reshape(-1).tolist(), **extra)
    opener = gzip.open if path.endswith(".gz") else open
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with opener(path, "wt") as fh:
        json.dump(gold, fh)
    print(f"wrote {path}: {n} patches, {gold['n_merged']} merged, s1/s2 {tuple(s1.shape)}, s3 {tuple(s3.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="a Qwen3.5+ checkpoint dir (uses its real tower weights)")
    ap.add_argument("--out", help="golden path (default: testdata/qwen35vl_vision_golden.json)")
    a = ap.parse_args()

    if a.real:
        from transformers import Qwen3_5ForConditionalGeneration
        m = Qwen3_5ForConditionalGeneration.from_pretrained(os.path.expanduser(a.real), dtype=torch.float32).eval()
        v = m.model.visual
        # CLAUDE.md: a persistent=False buffer computed in __init__ can come back as uninitialised
        # memory from from_pretrained's fast-init path. Check before trusting any disagreement.
        inv = v.rotary_pos_emb.inv_freq
        exp = 1.0 / (10000.0 ** (torch.arange(0, inv.numel() * 2, 2, dtype=torch.float) / (inv.numel() * 2)))
        assert torch.isfinite(inv).all() and torch.allclose(inv, exp, rtol=1e-6), "rotary inv_freq is not the expected table"
        assert len(v.config.deepstack_visual_indexes or []) == 0
        out = a.out or os.path.join(os.path.expanduser("~/models/qwen35vl_tower_golden"), "qwen35vl_tower_golden.json.gz")
        dump(v, REAL_GRIDS, 1, out, dict(source=os.path.basename(os.path.expanduser(a.real).rstrip("/"))))
        return

    torch.manual_seed(0)
    cfg = Qwen3_5VisionConfig(**TINY)
    v = Qwen3_5VisionModel(cfg).eval()
    randomise(v, 7)
    # Amplify the merger's fc1 so its pre-activations reach |x| ~ 3, where GELU-erf and GELU-tanh
    # differ by ~1e-3. At default scale the two agree to ~1e-6 after fc2 and a merger that used the
    # wrong one PASSES a cosine gate (measured 2026-09-30 by mutating the Go merger: llama.cpp makes
    # exactly this substitution). Tiny fixture only; the real tower is never touched.
    with torch.no_grad():
        v.merger.linear_fc1.weight.mul_(8.0)
    inv = v.rotary_pos_emb.inv_freq
    assert torch.isfinite(inv).all()
    ckpt = os.path.join(TESTDATA, "qwen35vl-vision-tiny")
    os.makedirs(ckpt, exist_ok=True)
    sd = {"model.visual." + k: t.detach().contiguous().clone() for k, t in v.state_dict().items()}
    save_file(sd, os.path.join(ckpt, "model.safetensors"))
    with open(os.path.join(ckpt, "config.json"), "w") as fh:
        json.dump({"model_type": "qwen3_5", "vision_config": {**TINY, "model_type": "qwen3_5"}}, fh, indent=1)
    dump(v, TINY_GRIDS, 1, a.out or os.path.join(TESTDATA, "qwen35vl_vision_golden.json"), {})


if __name__ == "__main__":
    main()

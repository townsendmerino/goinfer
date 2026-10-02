#!/usr/bin/env python3
"""Pin the GLM-OCR VISION TOWER (transformers' GlmOcrVisionModel, f32, CPU) as an aikit parity golden.

Gate O2 of docs/tasks/task-glm-ocr-2026-10.md. The Go side is aikit/vision/glm_ocr_encoder.go
(GlmOcrVisionEncoder). Tower-only: pixel_values + grid_thw in, merged features (HF pooler_output) out.
No decoder is involved.

Two modes (run with ~/.venv-vl/bin/python — transformers 5.12.0, torch CPU, PIL, safetensors):

  tiny   (default; seconds, no download)
      A small random-weight GlmOcrVisionModel with the REAL structure: biased Conv3d patch embed,
      axial 2D rotary over head_dim 64, per-head q_norm/k_norm RMSNorm before the rotary, full
      attention, RMSNorm (weight only), SiLU-gated biased MLP, post_layernorm, biased Conv2d
      `downsample`, and the unbiased `merger` (proj -> LayerNorm -> erf-GELU -> SiLU-gated MLP).
      Ratios follow the real tower: out_hidden = 1.5 x hidden, merger inner = out_hidden x
      in_channels (HF's context_dim). Two images of different non-square grids in ONE call.
      The weights are rounded to bf16 first (so the committed checkpoint is bf16, like the real one,
      and f32 HF math sees exactly what the Go loader reads).
          python scripts/pin_glm_ocr_vision.py
          -> <aikit>/testdata/glm_ocr_vision_golden.json
          -> <aikit>/testdata/glm-ocr-vision-tiny/   (config.json + model.safetensors, "model.visual." keys)

  real   (the pre-registered O2 gate; NOT committed — the files are tens of MB)
      The REAL checkpoint's tower weights (~/models/glm-ocr), fed the output of the REAL
      Glm46VImageProcessor on a procedurally rendered document page (PIL; deterministic) at two sizes.
      pixel_values are dumped to disk so the Go and HF sides see the SAME bytes.
          python scripts/pin_glm_ocr_vision.py real --ckpt ~/models/glm-ocr --out ~/models/glm-ocr-tower-golden
          -> <out>/manifest.json, <out>/<case>.pv.f32, <out>/<case>.pooler.f32, <out>/<case>.last.f32
         (raw little-endian float32; last = HF last_hidden_state = the downsample output)

Degeneracy guards (goinfer scripts/pin_cohere_tiny.py's lesson): HF's default init leaves every norm
weight at 1 and every bias at 0, so a Go bug that dropped a norm weight or a bias would not move a
golden. The tiny tower reseeds EVERY norm weight (norm1/norm2/q_norm/k_norm/post_layernorm/merger LN)
to non-trivial per-element values from a SEPARATE generator, gives every bias a non-zero value, and
scales qkv and the merger so that (a) skipping q/k norm, (b) rotary-before-norm, (c) the downsample
flattening order, and (d) GELU-tanh in place of erf each move the golden far beyond the 5e-6 bar. The
mutation results are in the O2 report; they were measured, not assumed.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig
from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrVisionModel
from transformers.vision_utils import get_vision_position_ids

TINY = dict(depth=2, hidden_size=128, intermediate_size=256, num_heads=2, in_channels=3,
            patch_size=4, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=192,
            rms_norm_eps=1e-5, attention_bias=True, hidden_act="silu")
TINY_GRIDS = [[1, 4, 6], [1, 2, 8]]  # two images, non-square, one packed call
DEFAULT_AIKIT = os.path.expanduser("~/mycode/aikit/aikit")


def tiny_config():
    cfg = GlmOcrVisionConfig(**TINY)
    cfg._attn_implementation = "eager"
    return cfg


def randomise(model, seed):
    """Non-trivial everything, from a generator separate from the one that draws pixels."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                # per-element, centred on 1 but nowhere near identity: q_norm/k_norm must be
                # NON-UNIFORM across head_dim or "rotary before norm" would commute with it.
                p.copy_(1.0 + 0.4 * torch.randn(p.shape, generator=g))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
        for b in model.blocks:
            b.attn.qkv.weight.mul_(2.5)  # raw q/k rms >> 1, so skipping q/k norm is loud
        # post_projection_norm output is ~N(0,1)*w+b; amplify so GELU-erf vs tanh differ visibly.
        model.merger.post_projection_norm.weight.mul_(2.0)
        # round to bf16: the committed checkpoint is bf16 like the real one
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))


class Capture:
    """Forward hooks for the intermediate stages (patch_embed out, post_layernorm out)."""

    def __init__(self, model):
        self.s1 = self.s2 = None
        model.patch_embed.register_forward_hook(lambda m, i, o: setattr(self, "s1", o.detach().clone()))
        model.post_layernorm.register_forward_hook(lambda m, i, o: setattr(self, "s2", o.detach().clone()))


def run(model, pv, grid):
    cap = Capture(model)
    with torch.no_grad():
        out = model(pv, grid_thw=grid, return_dict=True)
        pos = get_vision_position_ids(grid, model.spatial_merge_size)
    return cap.s1, cap.s2, out.last_hidden_state, out.pooler_output, pos


def check_rotary(model):
    """CLAUDE.md trap: a persistent=False buffer computed in __init__ can be uninitialised memory
    after from_pretrained's fast init. This script constructs the module directly and loads a state
    dict, so inv_freq is the real table; assert it anyway before trusting any disagreement."""
    inv = model.rotary_pos_emb.inv_freq
    n = inv.numel()
    exp = 1.0 / (10000.0 ** (torch.arange(0, n * 2, 2, dtype=torch.float) / (n * 2)))
    assert torch.isfinite(inv).all() and torch.allclose(inv, exp, rtol=1e-6), "rotary inv_freq is not the expected table"


def tiny_mode(aikit):
    torch.manual_seed(0)
    cfg = tiny_config()
    v = GlmOcrVisionModel(cfg).eval().to(torch.float32)
    randomise(v, 7)
    check_rotary(v)
    testdata = os.path.join(aikit, "testdata")
    ckpt = os.path.join(testdata, "glm-ocr-vision-tiny")
    os.makedirs(ckpt, exist_ok=True)
    sd = {"model.visual." + k: t.detach().to(torch.bfloat16).contiguous().clone() for k, t in v.state_dict().items()}
    save_file(sd, os.path.join(ckpt, "model.safetensors"))
    cfgd = {k: val for k, val in TINY.items()}
    cfgd.update(model_type="glm_ocr_vision", rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"})
    with open(os.path.join(ckpt, "config.json"), "w") as fh:
        json.dump({"model_type": "glm_ocr", "vision_config": cfgd}, fh, indent=1)

    g = torch.Generator().manual_seed(1)
    grid = torch.tensor(TINY_GRIDS, dtype=torch.long)
    n = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2
    pv = torch.randn(n, patch_dim, generator=g)
    s1, s2, s3, s4, pos = run(v, pv, grid)
    f = lambda t: t.reshape(-1).tolist()
    gold = dict(grid_thw=TINY_GRIDS, n_patches=n, n_merged=n // cfg.spatial_merge_size ** 2,
                hidden=cfg.hidden_size, out_hidden=cfg.out_hidden_size,
                pixel_values=f(pv), s1=f(s1), s2=f(s2), s3=f(s3), s4=f(s4), pos_ids=pos.reshape(-1).tolist())
    out = os.path.join(testdata, "glm_ocr_vision_golden.json")
    with open(out, "w") as fh:
        json.dump(gold, fh)
    print(f"wrote {out}: {n} patches, {gold['n_merged']} merged; s1/s2 {tuple(s1.shape)}, s3/s4 {tuple(s3.shape)}")
    print(f"wrote {ckpt}/")


# --- real mode ---------------------------------------------------------------------------------

def render_page(w, h, seed):
    """A deterministic document-like page: ruled text lines, a table grid, a stamp. PIL default font."""
    from PIL import Image, ImageDraw, ImageFont
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (w, h), (250, 249, 245))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(10, h // 70))
    except TypeError:
        font = ImageFont.load_default()
    words = "invoice total amount due payment terms net thirty days quantity unit price tax customer ship order".split()
    lh = max(14, h // 55)
    y = lh
    d.text((w // 12, y), "INVOICE No. %05d" % rng.randint(0, 99999), fill=(10, 10, 10), font=font)
    y += 2 * lh
    while y < h * 0.55:
        line = " ".join(rng.choice(words, rng.randint(6, 14)))
        d.text((w // 12, y), line, fill=(20, 20, 25), font=font)
        y += lh
    top, left, cw, ch = int(y + lh), w // 12, (w - 2 * (w // 12)) // 4, lh * 2
    for r in range(6):
        for c in range(4):
            x0, y0 = left + c * cw, top + r * ch
            d.rectangle([x0, y0, x0 + cw, y0 + ch], outline=(40, 40, 40))
            d.text((x0 + 4, y0 + 4), "%s %d" % (rng.choice(words), rng.randint(0, 9999)), fill=(0, 0, 0), font=font)
    d.ellipse([w * 0.7, h * 0.85, w * 0.7 + w // 7, h * 0.85 + w // 7], outline=(180, 30, 30), width=3)
    return img


def real_mode(ckpt, out, seed):
    from safetensors import safe_open
    from transformers import AutoImageProcessor
    ckpt = os.path.expanduser(ckpt)
    out = os.path.expanduser(out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(ckpt, "config.json")) as fh:
        vc = json.load(fh)["vision_config"]
    cfg = GlmOcrVisionConfig(**{k: val for k, val in vc.items() if k in GlmOcrVisionConfig.__dataclass_fields__})
    cfg._attn_implementation = "eager"
    v = GlmOcrVisionModel(cfg).eval().to(torch.float32)
    sd = {}
    with safe_open(os.path.join(ckpt, "model.safetensors"), "pt") as sf:
        for k in sf.keys():
            if k.startswith("model.visual."):
                sd[k[len("model.visual."):]] = sf.get_tensor(k).to(torch.float32)
    # inv_freq is persistent=False (not in the checkpoint): everything else must match exactly.
    res = v.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys and all("inv_freq" in k for k in res.missing_keys), res
    check_rotary(v)
    proc = AutoImageProcessor.from_pretrained(ckpt)

    # (name, source page w x h). The processor's smart_resize (factor 28, bounds HALVED for the 2-frame
    # still image: 6,272..4,816,896 px) decides the real grid.
    cases = [("small", 230, 220, seed), ("page", 1100, 1350, seed + 1)]
    manifest = dict(checkpoint=os.path.basename(ckpt.rstrip("/")), torch=torch.__version__,
                    processor=type(proc).__name__, cases=[])
    for name, w, h, s in cases:
        img = render_page(w, h, s)
        img.save(os.path.join(out, f"{name}.png"))
        pp = proc(images=[img], return_tensors="pt")
        pv = pp["pixel_values"].to(torch.float32)
        grid = pp["image_grid_thw"].to(torch.long)
        n = pv.shape[0]
        print(f"[{name}] source {w}x{h} -> grid {grid.tolist()} patches {n} merged {n // 4} "
              f"(pixels ~{n * 196 / 1e6:.2f} MP)", file=sys.stderr, flush=True)
        import time
        t0 = time.time()
        _, _, last, pooler, _ = run(v, pv, grid)
        print(f"[{name}] HF tower forward {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
        for tag, t in (("pv", pv), ("pooler", pooler), ("last", last)):
            t.detach().contiguous().numpy().astype("<f4").tofile(os.path.join(out, f"{name}.{tag}.f32"))
        manifest["cases"].append(dict(name=name, source_wh=[w, h], grid_thw=grid.tolist(), n_patches=n,
                                      n_merged=n // 4, patch_dim=int(pv.shape[1]), out_hidden=int(pooler.shape[1]),
                                      hidden=int(last.shape[1])))
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {out}/manifest.json", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="tiny", choices=["tiny", "real"])
    ap.add_argument("--aikit", default=DEFAULT_AIKIT, help="aikit checkout (tiny mode writes its testdata/)")
    ap.add_argument("--ckpt", default="~/models/glm-ocr")
    ap.add_argument("--out", default="~/models/glm-ocr-tower-golden")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    if a.mode == "tiny":
        tiny_mode(a.aikit)
    else:
        real_mode(a.ckpt, a.out, a.seed)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase A gate A2t (docs/tasks/task-embeddinggemma2.md): a tiny random gemma4_audio tower and its reference, for CI.

transformers' Gemma4AudioModel at a tiny width (hidden 64, 4 heads, 2 layers, the real subsampler's [128, 32]
channels, the real window) with a Linear embedder to a text width of 40, in float32 with sdpa attention (eager inverts
the audio mask). Random init leaves the ClippableLinear bounds at ±inf and every norm weight at 1, which hides a
dropped clamp or a misapplied norm (the fixture trap of Phase VM), so this sets finite asymmetric bounds (each at about
the 98th percentile of its own input or output on the golden's clip, so they bind), random norm weights and a random
per_dim_scale. Writes:

  testdata/gemma4-audio-tiny/config.json and model.safetensors (committed): an audio_config, a text_config.hidden_size,
      and the tower under the real checkpoint's names (audio_tower.*, embed_audio.embedding_projection.weight);
  testdata/gemma4-audio-tiny/golden.json.gz (committed): a seeded 1.3 s waveform, the extractor's log-mel for it, and
      the tower's stages on that log-mel (after the subsampler, each block, the tower, the embedder), valid rows only.

Run: python3 scripts/pin_gemma4_audio_tiny.py --out testdata/gemma4-audio-tiny
"""
import argparse
import gzip
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
import transformers
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4ClippableLinear
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

TEXT_HIDDEN = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(20261006)
    cfg = Gemma4AudioConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, output_proj_dims=48,
                            use_clipped_linears=True)
    cfg._attn_implementation = "sdpa"
    m = Gemma4AudioModel(cfg).eval()
    if m.config._attn_implementation != "sdpa":
        raise SystemExit("not sdpa")
    emb = torch.nn.Linear(cfg.output_proj_dims, TEXT_HIDDEN, bias=False)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("norm.weight") or name.endswith("layer_norm.weight") or "norm" in name.split(".")[-2]:
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            if name.endswith("per_dim_scale"):
                p.copy_(torch.randn(p.shape, generator=g))
    # The clip: seeded chirp, bursts and noise, 1.3 s (32 soft tokens, so the 12-key window binds).
    sr = 16000
    n = int(1.3 * sr) + 21
    t = np.arange(n) / sr
    r = np.random.default_rng(3)
    wav = (0.4 * np.sin(2 * np.pi * (200 + 900 * t) * t) + 0.2 * np.sin(2 * np.pi * 700 * t) * (t % 0.25 < 0.1)
           + 0.05 * r.standard_normal(n)).astype(np.float32)
    fe = Gemma4AudioFeatureExtractor()
    f = fe([wav], return_tensors="pt")
    feats, mask = f["input_features"].float(), f["input_features_mask"]
    tv = int(mask[0].sum())
    # Bounds at about the 98th percentile of what each ClippableLinear sees on this clip, asymmetric.
    seen = {}
    hooks = []
    for name, mod in m.named_modules():
        if isinstance(mod, Gemma4ClippableLinear):
            def h(mod, inp, out, name=name):
                seen[name] = (inp[0].detach().flatten(), out.detach().flatten())
            hooks.append(mod.register_forward_hook(h))
    with torch.no_grad():
        m(feats, mask)
    for h_ in hooks:
        h_.remove()
    with torch.no_grad():
        for name, mod in m.named_modules():
            if isinstance(mod, Gemma4ClippableLinear):
                i, o = seen[name]
                mod.input_min.fill_(float(torch.quantile(i, 0.01)))
                mod.input_max.fill_(float(torch.quantile(i, 0.985)))
                mod.output_min.fill_(float(torch.quantile(o, 0.02)))
                mod.output_max.fill_(float(torch.quantile(o, 0.98)))
    stages = {}
    hooks = [m.subsample_conv_projection.register_forward_hook(lambda mod, i, o: stages.__setitem__("sub", o[0]))]
    for L, layer in enumerate(m.layers):
        hooks.append(layer.register_forward_hook(lambda mod, i, o, L=L: stages.__setitem__(f"block{L}", o[0] if isinstance(o, tuple) else o)))
    hits = []
    for name, mod in m.named_modules():
        if isinstance(mod, Gemma4ClippableLinear):
            def hc(mod, inp, out, name=name):
                x = inp[0]
                hits.append(int(((x < mod.input_min) | (x > mod.input_max)).sum()))
            hooks.append(mod.register_forward_hook(hc))
    with torch.no_grad():
        out = m(feats, mask)
        # The embedder: the unweighted Gemma4RMSNorm, then the projection.
        x = out.last_hidden_state
        normed = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps, -0.5)
        e = emb(normed)
    for h_ in hooks:
        h_.remove()
    valid = out.attention_mask[0].bool()
    print(f"{n} samples, {tv} valid frames, {int(valid.sum())} soft tokens; clamp inputs bound in {sum(1 for x in hits if x)} of {len(hits)} projections", flush=True)
    os.makedirs(a.out, exist_ok=True)
    sd = {"audio_tower." + k: v.detach().contiguous() for k, v in m.state_dict().items() if "inv_timescales" not in k and "softcap" not in k}
    sd["embed_audio.embedding_projection.weight"] = emb.weight.detach().contiguous()
    save_file(sd, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    conf = {"model_type": "embedding_gemma2", "audio_config": cfg.to_dict(), "text_config": {"hidden_size": TEXT_HIDDEN}}
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(conf, fh, indent=1, default=str)

    def rows(x):
        return [float(v) for v in x[0][valid].flatten()]

    golden = {"transformers": transformers.__version__, "torch": torch.__version__, "attn_implementation": "sdpa",
              "samples": [float(v) for v in wav], "t_valid": tv, "features": [float(v) for v in feats[0, :tv].flatten()],
              "n_soft": int(valid.sum()), "sub": rows(stages["sub"]),
              "blocks": [rows(stages[f"block{L}"]) for L in range(len(m.layers))],
              "tower": rows(out.last_hidden_state), "embed": rows(e)}
    with gzip.open(os.path.join(a.out, "golden.json.gz"), "wt") as fh:
        json.dump(golden, fh)


if __name__ == "__main__":
    main()

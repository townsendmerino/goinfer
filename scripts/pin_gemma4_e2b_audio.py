#!/usr/bin/env python3
"""G-S5a's reference (docs/tasks/task-multimodal-support-2026-10.md, S5): Gemma 4 E2B's audio tower and embed_audio, from
E2B's own checkpoint, on the three committed EmbeddingGemma 2 clips (testdata/embeddinggemma2-audio/*.wav).

Only the tower and the embedder are built (transformers' Gemma4AudioModel and Gemma4MultimodalEmbedder from E2B's
config, float32, sdpa: eager inverts the audio mask), with E2B's weights loaded strictly: the whole E2B model in float32
is about 20 GB and does not fit the Mac, the tower is about 1.2 GB. The features come from Gemma4AudioFeatureExtractor's
defaults, which this script checks equal E2B's upstream processor_config.json (feature_extractor block, saved beside the
golden). The same load checks as pin_embeddinggemma2_audio.py: sdpa, the rel-pos inverse timescales against their
formula, the softcap, three clip scalars and one per_dim_scale against the checkpoint's header.

Writes <artifacts>/<clip>.{features,sub,block<L>,tower,embed}.f32 (valid rows only, float32 little-endian) and a
golden.json with the clip hashes, the shapes and the versions.

Run: python3 scripts/pin_gemma4_e2b_audio.py --model ~/models/gemma-4-E2B-unq --wavs testdata/embeddinggemma2-audio \
         --processor-config <E2B's processor_config.json> --artifacts ~/goinfer-logs/gemma4-e2b-audio
"""
import argparse
import hashlib
import json
import math
import os
import sys
import wave

import numpy as np
import torch
from safetensors import safe_open
import transformers
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder

CLIPS = ["short", "mid", "long"]


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != 16000:
            sys.exit(f"{path}: want 16 kHz mono 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--processor-config", required=True)
    ap.add_argument("--artifacts", required=True)
    a = ap.parse_args()
    model_dir = os.path.expanduser(a.model)
    if model_dir.startswith("/Volumes/") or model_dir.startswith("/srv/models"):
        sys.exit(f"{model_dir} is on the archive, not the bench set (CLAUDE.md)")
    art = os.path.expanduser(a.artifacts)
    os.makedirs(art, exist_ok=True)

    pc = json.load(open(os.path.expanduser(a.processor_config)))
    fe_cfg = pc.get("feature_extractor", pc)
    fe = Gemma4AudioFeatureExtractor()
    d = fe.to_dict()
    for k, v in fe_cfg.items():
        if k in ("feature_extractor_type", "processor_class"):
            continue
        dv = d.get(k)
        dv = float(dv) if isinstance(dv, np.ndarray) else dv
        if dv != v:
            sys.exit(f"feature extractor default {k}={dv!r}, E2B's processor_config says {v!r}")

    cfg = Gemma4Config.from_pretrained(model_dir)
    cfg.audio_config._attn_implementation = "sdpa"
    tower = Gemma4AudioModel(cfg.audio_config).float().eval()
    embed = Gemma4MultimodalEmbedder(cfg.audio_config, cfg.text_config).float().eval()
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as st:
        keys = list(st.keys())
        pfx = "model." if any(k.startswith("model.audio_tower.") for k in keys) else ""
        for mod, sub in [(tower, "audio_tower."), (embed, "embed_audio.")]:
            sd = {k[len(pfx + sub):]: st.get_tensor(k).float() for k in keys if k.startswith(pfx + sub)}
            missing, unexpected = mod.load_state_dict(sd, strict=False)
            # non-persistent buffers (inv_timescales) are computed, not loaded
            missing = [m for m in missing if "inv_timescales" not in m]
            if missing or unexpected:
                sys.exit(f"{sub}: missing {missing[:5]}, unexpected {unexpected[:5]}")
        impl = tower.config._attn_implementation
        if impl != "sdpa":
            sys.exit(f"audio tower attention is {impl!r}, not sdpa")
        inv = tower.rel_pos_enc.inv_timescales.flatten().double()
        want = torch.exp(-torch.arange(inv.numel(), dtype=torch.float64) * math.log(10000.0) / (inv.numel() - 1))
        if not torch.allclose(inv, want, rtol=1e-5):
            sys.exit("rel_pos_enc.inv_timescales does not match its formula")
        cap = float(tower.layers[0].self_attn.softcap)
        if cap != 50.0:
            sys.exit(f"softcap {cap}, want 50")
        last = len(tower.layers) - 1
        for name, val in [
            ("audio_tower.layers.0.feed_forward1.ffw_layer_1.input_max", tower.layers[0].feed_forward1.ffw_layer_1.input_max),
            ("audio_tower.layers.0.self_attn.post.output_max", tower.layers[0].self_attn.post.output_max),
            (f"audio_tower.layers.{last}.lconv1d.linear_end.input_min", tower.layers[last].lconv1d.linear_end.input_min),
        ]:
            hv = float(st.get_tensor(pfx + name).float())
            if abs(hv - float(val)) > 1e-6:
                sys.exit(f"{name}: loaded {float(val)}, header {hv}")
        pds = st.get_tensor(pfx + "audio_tower.layers.0.self_attn.per_dim_scale").float()
        if not torch.allclose(pds, tower.layers[0].self_attn.per_dim_scale.detach().float()):
            sys.exit("per_dim_scale did not load from the checkpoint")

    golden = {"model": os.path.basename(model_dir.rstrip("/")), "transformers": transformers.__version__,
              "torch": torch.__version__, "attn_implementation": impl, "text_hidden": cfg.text_config.hidden_size,
              "audio_hidden": cfg.audio_config.hidden_size, "layers": len(tower.layers), "clips": []}
    for name in CLIPS:
        path = os.path.join(a.wavs, name + ".wav")
        raw = open(path, "rb").read()
        wav = read_wav(path)
        f = fe([wav], return_tensors="pt")
        feats, mask = f["input_features"].float(), f["input_features_mask"]
        tv = int(mask[0].sum())
        stages = {}
        hooks = [tower.subsample_conv_projection.register_forward_hook(lambda mod, i, o: stages.__setitem__("sub", o[0]))]
        for L, layer in enumerate(tower.layers):
            hooks.append(layer.register_forward_hook(lambda mod, i, o, L=L: stages.__setitem__(f"block{L}", o[0] if isinstance(o, tuple) else o)))
        with torch.no_grad():
            out = tower(feats, mask, return_dict=True)
            emb = embed(inputs_embeds=out.last_hidden_state)
        for h in hooks:
            h.remove()
        valid = out.attention_mask[0].bool()
        n = int(valid.sum())
        feats[0, :tv].numpy().astype("<f4").tofile(os.path.join(art, name + ".features.f32"))
        stages["sub"][0][valid].numpy().astype("<f4").tofile(os.path.join(art, name + ".sub.f32"))
        for L in range(len(tower.layers)):
            stages[f"block{L}"][0][valid].numpy().astype("<f4").tofile(os.path.join(art, f"{name}.block{L}.f32"))
        out.last_hidden_state[0][valid].numpy().astype("<f4").tofile(os.path.join(art, name + ".tower.f32"))
        emb[0][valid].numpy().astype("<f4").tofile(os.path.join(art, name + ".embed.f32"))
        golden["clips"].append({"name": name, "path": path, "samples": int(wav.size), "sha256": hashlib.sha256(raw).hexdigest(),
                                "t_valid": tv, "n_soft": n})
        print(f"{name:6s} {wav.size} samples, {tv} valid frames, {n} soft tokens", file=sys.stderr)
    with open(os.path.join(art, "golden.json"), "w") as fh:
        json.dump(golden, fh, indent=1)


if __name__ == "__main__":
    main()

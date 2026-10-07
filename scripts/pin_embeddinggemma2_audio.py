#!/usr/bin/env python3
"""Phase A of docs/tasks/task-embeddinggemma2.md: the reference for EmbeddingGemma 2 audio embeddings.

Writes three seeded 16-bit 16 kHz mono WAVs (chirp, tone bursts and noise; 0.37 s, 2.37 s and 7.83 s, the last not
aligned to 128 or 160 samples) into --wavs, then, with sentence-transformers loading google/embeddinggemma-2 in float32
on the CPU with sdpa attention (eager inverts the audio mask: the Gate 0 spec, section 5.2), records:

  --golden (committed): each case's clip, prompt, text, ids, soft-token count and normalised 768-wide embedding, for
      each clip as audio alone, with the `query` prompt, and followed by text (9 cases);
  --artifacts (local, not committed): per clip, HF's log-mel features for the valid frames (<clip>.features.f32,
      [T_valid, 128]) and the tower's stages on them, valid rows only: after the subsampler (<clip>.sub.f32,
      [n, 1024]), each block's output (<clip>.block<L>.f32), the tower's output (<clip>.tower.f32, [n, 1536]) and the
      embedder's (<clip>.embed.f32, [n, 512]), all float32 little-endian.

Before recording it checks what the spec says can silently go wrong: sdpa, the rel-pos inverse timescales against
their formula, the softcap, and three clip scalars and one per_dim_scale against the checkpoint's header.

Run: python3 scripts/pin_embeddinggemma2_audio.py --model ~/models/embeddinggemma-2 \
         --wavs testdata/embeddinggemma2-audio --golden testdata/embeddinggemma2-audio/golden.json \
         --artifacts ~/goinfer-logs/embeddinggemma2-audio
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
from sentence_transformers import SentenceTransformer
import sentence_transformers
import transformers

SR = 16000
CLIPS = [("short", 0.37), ("mid", 2.37), ("long", 7.83)]
TEXT = "a recording of tones and a sweep"


def make_clip(name, secs, seed):
    n = int(round(SR * secs))
    if name == "long":
        n += 37  # not a multiple of 128 or 160
    t = np.arange(n) / SR
    r = np.random.default_rng(seed)
    f0, f1 = 150.0, 3500.0
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * max(secs, 1e-3)) * t * t))
    bursts = np.zeros(n)
    for k in range(int(secs * 4) + 1):
        a = int(k * SR / 4)
        b = min(n, a + SR // 10)
        f = 300 + 200 * k
        bursts[a:b] += np.sin(2 * np.pi * f * t[a:b])
    x = 0.35 * chirp + 0.25 * bursts + 0.05 * r.standard_normal(n)
    x = np.clip(x, -1, 1)
    return np.round(x * 32767).astype(np.int16)


def write_wav(path, pcm):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def header_scalar(st, name):
    return float(st.get_tensor(name).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wavs", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--artifacts", required=True)
    a = ap.parse_args()
    model_dir = os.path.expanduser(a.model)
    if model_dir.startswith("/Volumes/") or model_dir.startswith("/srv/models"):
        sys.exit(f"{model_dir} is on the archive, not the bench set (CLAUDE.md)")
    art = os.path.expanduser(a.artifacts)
    os.makedirs(art, exist_ok=True)
    os.makedirs(a.wavs, exist_ok=True)

    m = SentenceTransformer(model_dir, model_kwargs={"dtype": torch.float32}, device="cpu")
    inner = m[0].model
    tower = inner.audio_tower
    impl = tower.config._attn_implementation
    if impl != "sdpa":
        sys.exit(f"audio tower attention is {impl!r}, not sdpa: eager inverts the audio mask (Gate 0 spec 5.2)")
    # Non-persistent buffers and loaded scalars, against the formula and the header.
    inv = tower.rel_pos_enc.inv_timescales.flatten().double()
    want = torch.exp(-torch.arange(inv.numel(), dtype=torch.float64) * math.log(10000.0) / (inv.numel() - 1))
    if not torch.allclose(inv, want, rtol=1e-5):
        sys.exit("rel_pos_enc.inv_timescales does not match exp(-i ln(10000)/511)")
    cap = float(tower.layers[0].self_attn.softcap)
    if cap != 50.0:
        sys.exit(f"softcap {cap}, want 50")
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as st:
        for name, mod in [
            ("audio_tower.layers.0.feed_forward1.ffw_layer_1.input_max", tower.layers[0].feed_forward1.ffw_layer_1.input_max),
            ("audio_tower.layers.0.self_attn.post.output_max", tower.layers[0].self_attn.post.output_max),
            ("audio_tower.layers.11.lconv1d.linear_end.input_min", tower.layers[11].lconv1d.linear_end.input_min),
        ]:
            if abs(header_scalar(st, name) - float(mod)) > 1e-6:
                sys.exit(f"{name}: loaded {float(mod)}, header {header_scalar(st, name)}: the clip scalars fell back to init")
        pds = st.get_tensor("audio_tower.layers.0.self_attn.per_dim_scale").float()
        if not torch.allclose(pds, tower.layers[0].self_attn.per_dim_scale.detach().float()):
            sys.exit("per_dim_scale did not load from the checkpoint")

    capd = {}
    orig = inner.forward

    def hook(*args, **kw):
        capd.update(ids=kw.get("input_ids"))
        return orig(*args, **kw)

    inner.forward = hook
    rev_file = os.path.join(model_dir, "REVISION")
    golden = {"model_revision": open(rev_file).read().strip() if os.path.exists(rev_file) else "unknown",
              "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__,
              "torch": torch.__version__, "attn_implementation": impl, "text": TEXT, "clips": [], "items": []}
    proc = m[0].processor
    for ci, (name, secs) in enumerate(CLIPS):
        pcm = make_clip(name, secs, 20261006 + ci)
        path = os.path.join(a.wavs, name + ".wav")
        write_wav(path, pcm)
        raw = open(path, "rb").read()
        wav = pcm.astype(np.float32) / 32768.0
        golden["clips"].append({"name": name, "path": path, "samples": int(pcm.size), "sha256": hashlib.sha256(raw).hexdigest()})
        for prompt, text in [("", ""), ("query", ""), ("", TEXT)]:
            item = {"audio": {"array": wav, "sampling_rate": SR}}
            if text:
                item["text"] = text
            emb = m.encode([item], prompt_name=prompt or None, convert_to_numpy=True)[0]
            ids = capd["ids"][0].tolist()
            n_soft = ids.count(inner.config.audio_token_id)
            golden["items"].append({"clip": name, "prompt": prompt, "text": text, "ids": ids, "n_soft": n_soft,
                                    "embedding": [float(x) for x in emb]})
            print(f"{name:6s} prompt={prompt or '-':6s} text={'yes' if text else 'no ':3s} {len(ids)} ids, {n_soft} soft tokens", file=sys.stderr)
        # Stages, from HF's own features.
        fe = proc.feature_extractor([wav], return_tensors="pt")
        feats, mask = fe["input_features"].float(), fe["input_features_mask"]
        tv = int(mask[0].sum())
        stages = {}
        hooks = [tower.subsample_conv_projection.register_forward_hook(lambda mod, i, o: stages.__setitem__("sub", o[0]))]
        for L, layer in enumerate(tower.layers):
            hooks.append(layer.register_forward_hook(lambda mod, i, o, L=L: stages.__setitem__(f"block{L}", o[0] if isinstance(o, tuple) else o)))
        with torch.no_grad():
            out = inner.get_audio_features(feats, mask, return_dict=True)
        for h in hooks:
            h.remove()
        valid = out.attention_mask[0].bool()
        n = int(valid.sum())
        if n != golden["items"][-1]["n_soft"]:
            sys.exit(f"{name}: {n} valid tower rows, {golden['items'][-1]['n_soft']} soft tokens in the ids")
        feats[0, :tv].numpy().astype("<f4").tofile(os.path.join(art, name + ".features.f32"))
        stages["sub"][0][valid].numpy().astype("<f4").tofile(os.path.join(art, name + ".sub.f32"))
        for L in range(len(tower.layers)):
            stages[f"block{L}"][0][valid].numpy().astype("<f4").tofile(os.path.join(art, f"{name}.block{L}.f32"))
        out.last_hidden_state[0][valid].numpy().astype("<f4").tofile(os.path.join(art, name + ".tower.f32"))
        out.pooler_output[0][valid].numpy().astype("<f4").tofile(os.path.join(art, name + ".embed.f32"))
        golden["clips"][-1].update(t_valid=tv, n_soft=n)
        print(f"  {name}: {pcm.size} samples, {tv} valid frames, {n} soft tokens", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(a.golden)), exist_ok=True)
    with open(a.golden, "w") as f:
        json.dump(golden, f)


if __name__ == "__main__":
    main()

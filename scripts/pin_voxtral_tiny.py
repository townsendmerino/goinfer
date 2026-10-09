#!/usr/bin/env python3
"""G-S14e2's fixture (docs/tasks/task-multimodal-support-2026-10.md, "S14.4b ... G-S14e"): a tiny random-weight Voxtral built by transformers' own VoxtralForConditionalGeneration, saved the ordinary
way in the REAL checkpoint's layout (audio_tower.*, multi_modal_projector.*, language_model.model.*, language_model.lm_head.weight), plus goldens for goinfer's composition.

It keeps the properties the real model has and the planted defects need: the audio intermediate_size is 4 x its hidden size (the projector reads four stacked frames), the text head_dim (16) is
NOT hidden/heads (48/4 = 12), rope_theta is 1e8 as in the real config, the head is untied, audio_token_id 24 is inside the vocabulary. Weights are sized so the defects can be SEEN.
Goldens, per clip: the samples as 16-bit PCM (the features are computed by goinfer's own front end, gated bit-exact in G-S14a, from the same samples; the reference's are
the ones VoxtralProcessor makes: padded to a multiple of 480,000 samples, split into 3000-frame windows), the prompt ids, the projector
output rows, the last-position logits after the prefill, and an 8-token greedy continuation; plus a text-only prompt's logits and continuation.

Run (repo root): ~/g4venv/bin/python -I scripts/pin_voxtral_tiny.py testdata/voxtral-tiny"""
import json, os, sys, zipfile
import numpy as np, torch, transformers
from transformers import VoxtralConfig, VoxtralForConditionalGeneration, WhisperFeatureExtractor

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
torch.manual_seed(14)
AUDIO_TOKEN = 24
cfg = VoxtralConfig(
    audio_config=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=16, intermediate_size=128, num_mel_bins=128,
                      max_source_positions=1500, scale_embedding=False, vocab_size=160),
    text_config=dict(model_type="llama", hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=160,
                     rms_norm_eps=1e-5, rope_theta=1e8, max_position_embeddings=4096, rope_scaling=None, tie_word_embeddings=False, attention_bias=False, mlp_bias=False),
    audio_token_id=AUDIO_TOKEN, projector_hidden_act="gelu")
cfg._attn_implementation = "eager"
cfg.audio_config._attn_implementation = "eager"
cfg.text_config._attn_implementation = "eager"
model = VoxtralForConditionalGeneration(cfg).float().eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if "audio_tower.embed_positions" in n: p.normal_(0, 0.5)
        elif "audio_tower" in n and (p.dim() >= 3 or (p.dim() == 2 and "embed" not in n)):
            fan_in = p.numel() // p.shape[0]
            gain = 5.0 if ("q_proj" in n or "k_proj" in n) else (3.0 if ("fc1" in n or "conv" in n) else 1.5)
            p.normal_(0, gain / fan_in ** 0.5)
        elif "audio_tower" in n and "layer_norm" in n and n.endswith("weight"): p.copy_(1 + 0.2 * torch.randn_like(p))
        elif "audio_tower" in n: p.normal_(0, 0.3)
        elif "multi_modal_projector" in n:
            fan_in = p.shape[1]; p.normal_(0, 2.5 / fan_in ** 0.5)
        elif "embed_tokens" in n: p.normal_(0, 0.8)
        elif "lm_head" in n: p.normal_(0, 1.2 / p.shape[1] ** 0.5)
        elif p.dim() == 2:
            fan_in = p.shape[1]
            gain = 4.0 if ("q_proj" in n or "k_proj" in n) else 1.5
            p.normal_(0, gain / fan_in ** 0.5)
        elif "norm" in n and n.endswith("weight"): p.copy_(1 + 0.2 * torch.randn_like(p))
model.save_pretrained(out)
# transformers 5.15 writes the text tensors as language_model.model.model.*; the hub checkpoint (and every file goinfer will meet) has language_model.model.*. Rename to the REAL layout; the reload
# through from_pretrained below then proves transformers reads that layout and gets the same model.
from safetensors.torch import load_file, save_file
_st = load_file(f"{out}/model.safetensors")
save_file({k.replace("language_model.model.model.", "language_model.model.", 1): v.contiguous() for k, v in _st.items()}, f"{out}/model.safetensors", metadata={"format": "pt"})
fe = WhisperFeatureExtractor(feature_size=128)


def clip(n, f0):
    """Deterministic signal stored as 16-bit PCM so the golden is small and EXACT: the float32 samples are int16/32767 (a correctly rounded float64 division, then a cast)."""
    i = np.arange(n, dtype=np.float64); noise = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0 - 0.5
    x = 0.3 * np.sin(2 * np.pi * f0 * i / 16000) + 0.15 * np.sin(2 * np.pi * (f0 * 2.7) * i / 16000) * (1 + np.sin(i / 3000.0)) + 0.1 * noise
    return np.clip(np.round(x * 32767), -32767, 32767).astype("<i2")


def features(x):
    """VoxtralProcessor._retrieve_input_features: pad to a multiple of 480,000 samples, then split the mel along time into 3000-frame windows (batch dimension)."""
    f = fe(x, sampling_rate=16000, padding=True, truncation=False, pad_to_multiple_of=480000, return_tensors="pt")["input_features"]
    return f.reshape(128, -1, 3000).transpose(0, 1).contiguous()  # [windows][128][3000]


def prompt(n_audio):
    return [1, 3, 25] + [AUDIO_TOKEN] * n_audio + [4, 100, 101, 102, 34]


meta = {"transformers": transformers.__version__, "torch": torch.__version__, "audio_token": AUDIO_TOKEN, "cases": []}
f32 = lambda t: t.detach().float().numpy().astype("<f4").tobytes()


def greedy(m, **kw):
    with torch.no_grad():
        o = m.generate(max_new_tokens=8, do_sample=False, **kw)
    return [int(i) for i in o[0, kw["input_ids"].shape[1]:]]


with zipfile.ZipFile(f"{out}/golden.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    # the text-only path (loader, explicit head_dim, rope_theta, untied head)
    tids = [1, 5, 9, 33, 70, 12, 99, 41, 2, 57, 140, 8]
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([tids])).logits[0, -1]
    z.writestr("text.ids.json", json.dumps(tids)); z.writestr("text.logits.f32", f32(lg))
    z.writestr("text.cont.json", json.dumps(greedy(model, input_ids=torch.tensor([tids]))))
    for name, secs, f0 in (("clip6s", 6.0, 330.0), ("clip35s", 35.0, 180.0)):
        pcm = clip(int(secs * 16000), f0)
        x = (pcm.astype(np.float64) / 32767.0).astype(np.float32)
        feats = features(x)
        n_audio = feats.shape[0] * 375
        ids = prompt(n_audio)
        with torch.no_grad():
            emb = model.get_audio_features(feats).pooler_output  # [windows*375][48]
            lg = model(input_ids=torch.tensor([ids]), input_features=feats).logits[0, -1]
        assert emb.shape == (n_audio, 48), emb.shape
        cont = greedy(model, input_ids=torch.tensor([ids]), input_features=feats)
        z.writestr(f"{name}.pcm16", pcm.tobytes()); z.writestr(f"{name}.prompt.json", json.dumps(ids))
        z.writestr(f"{name}.embeds.f32", f32(emb)); z.writestr(f"{name}.logits.f32", f32(lg)); z.writestr(f"{name}.cont.json", json.dumps(cont))
        meta["cases"].append({"name": name, "samples": int(len(x)), "windows": int(feats.shape[0]), "audio_tokens": int(n_audio), "embed_rms": float(emb.pow(2).mean().sqrt())})
        print(f"{name}: {len(x)} samples -> {feats.shape[0]} window(s), {n_audio} audio tokens, embed rms {emb.pow(2).mean().sqrt():.3f}, continuation {cont}")
    z.writestr("meta.json", json.dumps(meta, indent=1))
re = VoxtralForConditionalGeneration.from_pretrained(out, dtype=torch.float32).eval()
with torch.no_grad():
    f = features((clip(96000, 330.0).astype(np.float64) / 32767.0).astype(np.float32))
    d = float((re.get_audio_features(f).pooler_output - model.get_audio_features(f).pooler_output).abs().max())
    d2 = float((re(input_ids=torch.tensor([tids])).logits - model(input_ids=torch.tensor([tids])).logits).abs().max())
print("reload through from_pretrained: audio max |diff|", d, " text max |diff|", d2)
assert d < 1e-4 and d2 < 1e-4, "the reloaded model differs from the saved one"

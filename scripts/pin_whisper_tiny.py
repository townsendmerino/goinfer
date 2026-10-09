#!/usr/bin/env python3
"""G-S14d1's fixture (docs/tasks/task-multimodal-support-2026-10.md, "G-S14d"): a tiny random-weight Whisper (encoder AND decoder, so S14.4c can reuse it) built from transformers' own classes
and saved the ordinary way (transformers 5.15.0 reloads it: the round trip is checked below), plus the encoder's output on two synthetic clips whose features come from transformers' own
WhisperFeatureExtractor. Weights are sized so the planted defects can be SEEN (pre-activations of order 1-3, sharp attention, a loud position table).
Run (repo root): ~/g4venv/bin/python -I scripts/pin_whisper_tiny.py testdata/whisper-tiny-rand"""
import json, os, sys, zipfile
import numpy as np, torch, transformers
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
torch.manual_seed(11)
cfg = WhisperConfig(vocab_size=160, num_mel_bins=80, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
                    max_source_positions=1500, max_target_positions=64, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=3)
cfg._attn_implementation = "eager"
model = WhisperForConditionalGeneration(cfg).float().eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if "embed_positions" in n and "encoder" in n: p.normal_(0, 0.5)
        elif p.dim() >= 3 or (p.dim() == 2 and "embed" not in n):
            fan_in = p.numel() // p.shape[0]
            gain = 5.0 if ("q_proj" in n or "k_proj" in n) else (3.0 if ("fc1" in n or "conv" in n) else 1.5)
            p.normal_(0, gain / fan_in ** 0.5)
        elif p.dim() == 2: p.normal_(0, 0.3)
        elif "layer_norm" in n and n.endswith("weight"): p.copy_(1 + 0.2 * torch.randn_like(p))
        else: p.normal_(0, 0.3)
model.save_pretrained(out)
fe = WhisperFeatureExtractor(feature_size=80)
def clip(n, f0):
    i = np.arange(n, dtype=np.float64); noise = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0 - 0.5
    return (0.3 * np.sin(2 * np.pi * f0 * i / 16000) + 0.15 * np.sin(2 * np.pi * (f0 * 2.7) * i / 16000) * (1 + np.sin(i / 3000.0)) + 0.1 * noise).astype(np.float32)
meta = {"transformers": transformers.__version__, "torch": torch.__version__, "cases": []}
with zipfile.ZipFile(f"{out}/golden.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name, x in (("clip6s5", clip(104000, 330.0)), ("clip25s", clip(400000, 180.0))):
        feats = fe(x, sampling_rate=16000, return_tensors="pt")["input_features"]
        with torch.no_grad():
            enc = model.model.encoder(input_features=feats).last_hidden_state[0]
        z.writestr(f"{name}.samples.f32", x.astype("<f4").tobytes()); z.writestr(f"{name}.feats.f32", feats[0].numpy().astype("<f4").tobytes()); z.writestr(f"{name}.enc.f32", enc.numpy().astype("<f4").tobytes())
        meta["cases"].append({"name": name, "samples": int(len(x)), "frames": int(enc.shape[0]), "d_model": 32})
        print(f"{name}: {len(x)} samples -> encoder output {tuple(enc.shape)}, rms {enc.pow(2).mean().sqrt():.3f}")
    z.writestr("meta.json", json.dumps(meta, indent=1))
re = WhisperForConditionalGeneration.from_pretrained(out, torch_dtype=torch.float32).eval()
with torch.no_grad():
    f = fe(clip(104000, 330.0), sampling_rate=16000, return_tensors="pt")["input_features"]
    d = float((re.model.encoder(input_features=f).last_hidden_state - model.model.encoder(input_features=f).last_hidden_state).abs().max())
print("reload through from_pretrained: encoder max |diff|", d)
assert d < 1e-4, "the reloaded model differs from the saved one (every tensor was checked equal when this was written; 4e-06 is threading noise in the forward)"

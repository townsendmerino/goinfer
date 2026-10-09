#!/usr/bin/env python3
"""G-S14b1's golden (docs/tasks/task-multimodal-support-2026-10.md, "G-S14b"): transformers' Qwen3ASRFeatureExtractor called as Qwen3ASRProcessor calls it (padding=True, truncation=False,
return_attention_mask=True, n_window=50) on seven clips: the features, the mask and the input samples, as a zip of raw little-endian float32 / uint8 and a meta.json.
Run (repo root): ~/g4venv/bin/python -I scripts/pin_qwen3asr_features.py testdata/qwen3asr_features_golden.zip"""
import json, sys, wave, zipfile
import numpy as np, torch, transformers
from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor

out = sys.argv[1]
def wav(p):
    w = wave.open(p); assert w.getframerate() == 16000 and w.getnchannels() == 1
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
def synth(n, f0):
    i = np.arange(n, dtype=np.float64); noise = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0 - 0.5
    return (0.3 * np.sin(2 * np.pi * f0 * i / 16000) + 0.2 * np.sin(2 * np.pi * (f0 * 3.1) * i / 16000) + 0.1 * noise).astype(np.float32)
libri = wav("testdata/speech/librispeech-1272-128104-0000.wav")
tiled = np.tile(libri, 6)[:488000].copy()   # 30.5 s: past the 30 s a Whisper extractor would cut at; stored as a recipe, not as samples
cases = [("libri", libri, True), ("short4800", synth(4800, 440.0), True), ("exact300", synth(48000, 330.0), True), ("plus1", synth(48160, 330.0), True),
         ("synth6s", synth(96000, 220.0), True), ("long30s5", tiled, False), ("silence2s", np.zeros(32000, np.float32), True)]
fe = Qwen3ASRFeatureExtractor()
meta = {"transformers": transformers.__version__, "torch": torch.__version__, "cases": [], "recipes": {"long30s5": "np.tile(libri, 6)[:488000]"}}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name, x, store in cases:
        b = fe(x, sampling_rate=16000, return_tensors="np", return_attention_mask=True, padding=True, truncation=False, n_window=50)
        f, m = b["input_features"][0].astype("<f4"), b["attention_mask"][0].astype("u1")
        assert f.shape[0] == 128 and f.shape[1] % 100 == 0 and m.shape[0] == f.shape[1], (f.shape, m.shape)
        if store: z.writestr(f"{name}.in.f32", x.astype("<f4").tobytes())
        z.writestr(f"{name}.feats.f32", f.tobytes()); z.writestr(f"{name}.mask.u8", m.tobytes())
        meta["cases"].append({"name": name, "samples": int(len(x)), "T": int(f.shape[1]), "valid": int(m.sum()), "min": float(f.min()), "max": float(f.max()), "stored": store})
        print(f"{name:10s} {len(x):7d} samples -> T {f.shape[1]:5d}, valid {int(m.sum()):5d}, range [{f.min():.4f}, {f.max():.4f}]")
    z.writestr("meta.json", json.dumps(meta, indent=1))

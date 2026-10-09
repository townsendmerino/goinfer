#!/usr/bin/env python3
"""G-S14a's golden (docs/tasks/task-multimodal-support-2026-10.md, "G-S14a"): transformers' WhisperFeatureExtractor, BOTH of its paths (the torch float32 one __call__ uses and the NumPy
float64 one), on eight inputs, plus the valid-frame count its attention mask gives for a range of lengths. Written as a zip of raw little-endian float32 (inputs and features) and a
meta.json, because the features are [mels, 3000] each and mostly constant padding for a short clip, which deflate removes.
Run: ~/g4venv/bin/python -I scripts/pin_whisper_features.py testdata/whisper_features_golden.zip   (from the repo root)"""
import json, sys, wave, zipfile
import numpy as np
import torch, transformers
from transformers import WhisperFeatureExtractor

out = sys.argv[1]
S = "testdata/speech/"
def wav(p):
    w = wave.open(p); assert w.getframerate() == 16000 and w.getnchannels() == 1
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
def f32(p): return np.fromfile(p, dtype="<f4")
def synth(n):  # deterministic, no RNG: two tones and a multiplicative-hash noise
    i = np.arange(n, dtype=np.float64)
    noise = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0 - 0.5
    return (0.3 * np.sin(2 * np.pi * 440.0 * i / 16000) + 0.2 * np.sin(2 * np.pi * 1234.5 * i / 16000) + 0.1 * noise).astype(np.float32)
libri = wav(S + "librispeech-1272-128104-0000.wav")
cases = [("libri", libri, 128), ("libri80", libri, 80),
         ("ref44k", f32(S + "librispeech-1272-128104-0000-44k1.ref16k.f32"), 128), ("ref48k", f32(S + "librispeech-1272-128104-0000-48k-stereo.ref16k.f32"), 128),
         ("synth10", synth(160000), 128), ("short800", synth(800), 128), ("silence2s", np.zeros(32000, np.float32), 128)]
meta = {"transformers": transformers.__version__, "torch": torch.__version__, "numpy": np.__version__, "cases": [], "counts": []}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name, x, mels in cases:
        fe = WhisperFeatureExtractor(feature_size=mels)
        pad = np.zeros(480000, np.float32); pad[: min(len(x), 480000)] = x[:480000]
        a = np.asarray(fe._np_extract_fbank_features(pad[None, :], "cpu"))[0].astype("<f4")
        b = np.asarray(fe._torch_extract_fbank_features(pad, "cpu")).astype("<f4")
        call = fe(x, sampling_rate=16000, return_tensors="np")["input_features"][0].astype("<f4")
        assert a.shape == b.shape == call.shape == (mels, 3000), (a.shape, b.shape, call.shape)
        assert np.array_equal(call, b), "__call__ is expected to be the torch path"
        d = np.abs(a - b)
        meta["cases"].append({"name": name, "mels": mels, "samples": int(len(x)), "np_vs_torch_max": float(d.max()), "np_vs_torch_mean": float(d.mean()), "min": float(a.min()), "max": float(a.max())})
        z.writestr(f"{name}.in.f32", x.astype("<f4").tobytes()); z.writestr(f"{name}.torch.f32", b.tobytes())
        if name in ("libri", "libri80", "synth10"):  # the NumPy path too, for three cases (the file stays small); the other cases are held to the torch path alone
            z.writestr(f"{name}.np.f32", a.tobytes())
        print(f"{name:10s} mels {mels:3d} samples {len(x):7d}  np-vs-torch max {d.max():.3e} mean {d.mean():.3e}  range [{a.min():.4f}, {a.max():.4f}]")
    fe = WhisperFeatureExtractor(feature_size=128)
    for n in (1, 159, 160, 161, 800, 16000, 93680, 480000, 500000):
        r = fe(np.zeros(n, np.float32) + 0.01, sampling_rate=16000, return_tensors="np", return_attention_mask=True)
        m = r["attention_mask"][0]
        meta["counts"].append({"samples": n, "valid_frames": int(m.sum()), "mask_len": int(len(m))})
        print(f"samples {n:7d}: attention-mask sum {int(m.sum())} of {len(m)}")
    z.writestr("meta.json", json.dumps(meta, indent=1))

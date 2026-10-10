#!/usr/bin/env python3
"""G-S14g1's reference (docs/tasks/task-multimodal-support-2026-10.md, "G-S14g"): transformers 5.15.0's WhisperFeatureExtractor as long-form generate calls it (truncation=False, padding="longest",
return_attention_mask=True) on three clips: the registered 5.9 s LibriSpeech clip, the clip six times with half-second gaps (36 s), and a 70 s composite of the first LibriSpeech dummy clips with half-second gaps.
Writes <out>/<case>.in.f32, <case>.feats.f32 ([mels][T] row-major), <case>.mask (frames the attention mask counts) and meta.json.
Run (repo root): ~/g4venv/bin/python -I scripts/pin_whisper_longest.py --model ~/models/whisper-small --data ~/goinfer-bench/librispeech-dummy --out <dir>"""
import argparse, glob, json, os, wave
import numpy as np, transformers
from transformers import WhisperFeatureExtractor

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
def wav(p):
    w = wave.open(p); return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
gap = np.zeros(8000, dtype=np.float32)
libri = wav("testdata/speech/librispeech-1272-128104-0000.wav")
composite, total = [], 0
for p in sorted(glob.glob(f"{a.data}/*.wav")):
    x = wav(p); composite.append(np.concatenate([x, gap])); total += len(x) + len(gap)
    if total >= 70 * 16000: break
cases = {"libri": libri, "libri6x": np.concatenate([np.concatenate([libri, gap]) for _ in range(6)]), "long70": np.concatenate(composite)}
fe = WhisperFeatureExtractor.from_pretrained(a.model)
meta = {"transformers": transformers.__version__, "mels": fe.feature_size, "cases": {}}
for name, x in cases.items():
    r = fe(x, sampling_rate=16000, truncation=False, padding="longest", return_attention_mask=True, return_tensors="np")
    f = r["input_features"][0]; m = r["attention_mask"][0]
    np.asarray(x, "<f4").tofile(f"{a.out}/{name}.in.f32"); np.asarray(f, "<f4").tofile(f"{a.out}/{name}.feats.f32")
    meta["cases"][name] = {"samples": int(len(x)), "frames": int(f.shape[-1]), "mask": int(m.sum()), "mask_len": int(len(m))}
    print(name, len(x), "samples ->", f.shape, "mask sum", int(m.sum()), "of", len(m), "range", float(f.min()), float(f.max()))
json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)

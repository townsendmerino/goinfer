#!/usr/bin/env python3
"""G-S14d2/3's reference (docs/tasks/task-multimodal-support-2026-10.md, "G-S14d"): a real Whisper checkpoint's ENCODER in transformers 5.15.0 float32 on the LibriSpeech clip and a 25 s synthetic signal, with the
PROOF that the reference is the real model and not a silently random-initialised one: the same model transcribes the LibriSpeech clip. Writes <out>/{libri,synth25s}.{in,feats,enc}.f32 and meta.json.
Run: ~/g4venv/bin/python -I scripts/pin_whisper_real.py --model ~/models/whisper-small --out <dir>   (from the repo root)"""
import argparse, json, os, wave
import numpy as np, torch, transformers
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
model, info = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float32, output_loading_info=True); model.eval()
print("loading info: missing", sorted(info["missing_keys"]), "unexpected", sorted(info["unexpected_keys"]))
assert not info["missing_keys"], "weights missing: the reference would be randomly initialised"
mels = model.config.num_mel_bins
fe = WhisperFeatureExtractor.from_pretrained(a.model); tok = WhisperTokenizer.from_pretrained(a.model)
w = wave.open("testdata/speech/librispeech-1272-128104-0000.wav"); libri = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
f = fe(libri, sampling_rate=16000, return_tensors="pt")["input_features"]
with torch.no_grad():
    gen = model.generate(input_features=f, max_new_tokens=64, language="en", task="transcribe")
text = tok.decode(gen[0], skip_special_tokens=True)
print("TRANSCRIPTION:", repr(text)); assert "quilter" in text.lower() and "gospel" in text.lower(), "the reference does not transcribe: it is not the real model"
def synth(n, f0):
    i = np.arange(n, dtype=np.float64); noise = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0 - 0.5
    return (0.3 * np.sin(2 * np.pi * f0 * i / 16000) + 0.15 * np.sin(2 * np.pi * (f0 * 2.7) * i / 16000) * (1 + np.sin(i / 3000.0)) + 0.1 * noise).astype(np.float32)
meta = {"transformers": transformers.__version__, "torch": torch.__version__, "mels": mels, "d_model": model.config.d_model, "transcription": text, "cases": []}
for name, x in (("libri", libri), ("synth25s", synth(400000, 260.0))):
    feats = fe(x, sampling_rate=16000, return_tensors="pt")["input_features"]
    with torch.no_grad():
        enc = model.model.encoder(input_features=feats).last_hidden_state[0]
    np.asarray(x, "<f4").tofile(f"{a.out}/{name}.in.f32"); feats[0].numpy().astype("<f4").tofile(f"{a.out}/{name}.feats.f32"); enc.numpy().astype("<f4").tofile(f"{a.out}/{name}.enc.f32")
    meta["cases"].append({"name": name, "samples": int(len(x)), "frames": int(enc.shape[0])})
    print(f"{name}: encoder output {tuple(enc.shape)}, rms {enc.pow(2).mean().sqrt():.3f}")
json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)

#!/usr/bin/env python3
"""G-S14g3's reference (docs/tasks/task-multimodal-support-2026-10.md, "G-S14g"): openai/whisper-small in transformers 5.15.0 float32, generate(return_timestamps=True, return_segments=True, task="transcribe") on the
registered 5.9 s LibriSpeech clip (short form, the extractor's default 30 s padding) and on the 70 s composite of scripts/pin_whisper_longest.py (long form: truncation=False, padding="longest", the attention mask
passed). Per case: the segments (start, end, tokens), the concatenated sequence, the decoded text (skip_special_tokens=True), the detected language id. The reference is PROVED real: zero missing keys, and the
short clip's text asserted against the corpus. Writes <out>/golden.json.
Run (repo root): ~/g4venv/bin/python -I scripts/pin_whisper_ts_real.py --model ~/models/whisper-small --data ~/goinfer-bench/librispeech-dummy --out <dir>"""
import argparse, glob, json, os, re, wave
import numpy as np, torch, transformers
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
model, info = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32, output_loading_info=True, attn_implementation="eager"); model.eval()
assert not info["missing_keys"], "weights missing: the reference would be randomly initialised"
fe, tok = WhisperFeatureExtractor.from_pretrained(a.model), WhisperTokenizer.from_pretrained(a.model)
def wav(p):
    w = wave.open(p); return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
gap = np.zeros(8000, dtype=np.float32)
libri = wav("testdata/speech/librispeech-1272-128104-0000.wav")
comp, total = [], 0
for p in sorted(glob.glob(f"{a.data}/*.wav")):
    x = wav(p); comp.append(np.concatenate([x, gap])); total += len(x) + len(gap)
    if total >= 70 * 16000: break
golden = {"transformers": transformers.__version__, "torch": torch.__version__, "cases": {}}
for name, x in (("libri", libri), ("long70", np.concatenate(comp))):
    long = len(x) > 480000
    r = fe(x, sampling_rate=16000, truncation=not long, padding="longest" if long else "max_length", return_attention_mask=long, return_tensors="pt")
    kw = {"attention_mask": r["attention_mask"]} if long else {}
    with torch.no_grad():
        lang_id = int(model.detect_language(input_features=r["input_features"][:, :, :3000]).reshape(-1)[0])
        o = model.generate(input_features=r["input_features"], task="transcribe", return_timestamps=True, return_segments=True, **kw)
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "tokens": [int(t) for t in s["tokens"].tolist()]} for s in o["segments"][0]]
    seq = [int(t) for t in o["sequences"][0].tolist()]
    text = tok.decode(seq, skip_special_tokens=True)
    np.asarray(x, "<f4").tofile(f"{a.out}/{name}.in.f32")
    golden["cases"][name] = {"samples": int(len(x)), "language_id": lang_id, "segments": segs, "sequence": seq, "text": text}
    print(name, len(x), "samples,", len(segs), "segments, lang", lang_id, "|", repr(text[:100]))
    if name == "libri":
        norm = lambda s: " ".join(re.sub(r"[^a-z ]", " ", re.sub(r"\bmr\.?\b", "mister", s.lower())).split())
        assert norm(text) == "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel", "the reference does not transcribe: it is not the real model"
json.dump(golden, open(f"{a.out}/golden.json", "w"))

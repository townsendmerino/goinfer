#!/usr/bin/env python3
"""G-S14h2's reference (docs/tasks/task-multimodal-support-2026-10.md, "G-S14h"): openai/whisper-small in transformers 5.15.0 float32, generate(return_timestamps=True, return_segments=True, task="transcribe", temperature=(0.0,),
condition_on_prev_tokens=True, logprob_threshold=-1.0, compression_ratio_threshold=2.4, no_speech_threshold=0.6) (OpenAI's defaults, without the sampled fallback) on three configurations on two long clips (the last uses logprob_threshold=-0.4 on the silent composite): the 70 s composite of
scripts/pin_whisper_longest.py, and a "silent" composite: 35 s of zeros, the registered 5.9 s clip, 35 s of zeros, the first two LibriSpeech dummy clips (long form, a silent first window and a silent third).
Per case: the segments (start, end, tokens), the sequence, the decoded text. Writes <out>/golden.json and <case>.in.f32.
Run (repo root): ~/g4venv/bin/python -I scripts/pin_whisper_policy_real.py --model ~/models/whisper-small --data ~/goinfer-bench/librispeech-dummy --out <dir>"""
import argparse, glob, json, os, wave
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
files = sorted(glob.glob(f"{a.data}/*.wav"))
comp, total = [], 0
for p in files:
    x = wav(p); comp.append(np.concatenate([x, gap])); total += len(x) + len(gap)
    if total >= 70 * 16000: break
sil = np.zeros(35 * 16000, dtype=np.float32)
silent = np.concatenate([sil, libri, sil, wav(files[0]), gap, wav(files[1])])
golden = {"transformers": transformers.__version__, "torch": torch.__version__, "cases": {}}
CASES = (("long70", np.concatenate(comp), -1.0), ("silent", silent, -1.0), ("silent_strict", silent, -0.4))  # the last: a logprob threshold the silent window falls under, so its no-speech skip fires
for name, x, lpt in CASES:
    r = fe(x, sampling_rate=16000, truncation=False, padding="longest", return_attention_mask=True, return_tensors="pt")
    with torch.no_grad():
        o = model.generate(input_features=r["input_features"], attention_mask=r["attention_mask"], task="transcribe", return_timestamps=True, return_segments=True, temperature=(0.0,),
                           condition_on_prev_tokens=True, logprob_threshold=lpt, compression_ratio_threshold=2.4, no_speech_threshold=0.6)
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "tokens": [int(t) for t in s["tokens"].tolist()]} for s in o["segments"][0]]
    seq = [int(t) for t in o["sequence" if "sequence" in o else "sequences"][0].tolist()]
    text = tok.decode(seq, skip_special_tokens=True)
    np.asarray(x, "<f4").tofile(f"{a.out}/{name}.in.f32")
    golden["cases"][name] = {"samples": int(len(x)), "logprob_threshold": lpt, "segments": segs, "sequence": seq, "text": text}
    print(name, len(x), "samples,", len(segs), "segments, starts", [round(s["start"], 1) for s in segs], "|", repr(text[:90]))
json.dump(golden, open(f"{a.out}/golden.json", "w"))

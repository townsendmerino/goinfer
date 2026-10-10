#!/usr/bin/env python3
"""G-S14e4a(ii) of docs/tasks/task-multimodal-support-2026-10.md: the processor's own transcription request ids with NO language (apply_transcription_request(language=None)), for three clip lengths,
so goinfer's multimodal.VoxtralPrompt(tk, n, "") can be compared id for id. Tokenizer and processor only: no model is loaded. Writes testdata/voxtral_nolang_ids.json.
Run (repo root): PYTHONPATH=~/goinfer-bench/s14e/mc ~/g4venv/bin/python scripts/pin_voxtral_nolang.py ~/models/voxtral-mini-3b-2507"""
import json, sys, wave
import numpy as np, transformers
from transformers import AutoProcessor
model = sys.argv[1]
w = wave.open("testdata/speech/librispeech-1272-128104-0000.wav")
libri = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
gap = np.zeros(8000, dtype=np.float32)
cases = {"zeros3s": np.zeros(48000, dtype=np.float32), "libri": libri, "libri6x": np.concatenate([np.concatenate([libri, gap]) for _ in range(6)])}
proc = AutoProcessor.from_pretrained(model)
out = {"transformers": transformers.__version__, "cases": {}}
for name, x in cases.items():
    r = proc.apply_transcription_request(language=None, audio=x, model_id="mistralai/Voxtral-Mini-3B-2507", sampling_rate=16000, format="wav", return_tensors="pt")
    out["cases"][name] = {"samples": int(len(x)), "ids": r["input_ids"][0].tolist()}
json.dump(out, open("testdata/voxtral_nolang_ids.json", "w"))
print({k: (v["samples"], len(v["ids"])) for k, v in out["cases"].items()})

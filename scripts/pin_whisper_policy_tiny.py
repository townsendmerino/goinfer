#!/usr/bin/env python3
"""G-S14h1's reference (docs/tasks/task-multimodal-support-2026-10.md, "G-S14h"): transformers' generate on the tiny random-weight Whisper of scripts/pin_whisper_ts_tiny.py (testdata/whisper-tiny-ts) with the decode
policy switched on, temperature=(0.0,) so no sampled attempt is reachable: per clip and per configuration the segments (start, end, tokens) and the sequence. Writes testdata/whisper-tiny-ts/policy_golden.json.
Configurations: cond (condition_on_prev_tokens), thr (a logprob and a compression-ratio threshold that "need a fallback" on some windows, with nothing to fall back to), skip (logprob and no-speech thresholds that skip
windows), skip2 (a lower no-speech threshold: more windows skipped), all (everything together).
Run (repo root): ~/g4venv/bin/python -I scripts/pin_whisper_policy_tiny.py testdata/whisper-tiny-ts"""
import json, sys
import numpy as np, torch, transformers
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

d = sys.argv[1]
model = WhisperForConditionalGeneration.from_pretrained(d, dtype=torch.float32, attn_implementation="eager").eval()
fe = WhisperFeatureExtractor(feature_size=80)
def clip(n, a, b):
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        saw1 = ((i * a) % 1000) / 1000 - 0.5; saw2 = ((i * b) % 1733) / 1733 - 0.5; env = ((i // 4000) % 3) / 2.0; noise = ((i * 2654435761) & 0xFFFFFFFF) / 4294967296.0 - 0.5
        out[i] = np.float32(0.25 * saw1 + 0.2 * saw2 * env + 0.1 * noise * (1 + ((i // 9000) % 2)))
    return out
CONFIGS = {"cond": dict(condition_on_prev_tokens=True),
           "thr": dict(logprob_threshold=-0.5, compression_ratio_threshold=1.2),
           "skip": dict(logprob_threshold=-0.5, no_speech_threshold=1e-4),
           "skip2": dict(logprob_threshold=-0.5, no_speech_threshold=1e-5),
           "all": dict(condition_on_prev_tokens=True, logprob_threshold=-0.5, compression_ratio_threshold=1.2, no_speech_threshold=1e-4)}
out = {"transformers": transformers.__version__, "configs": {k: {kk: vv for kk, vv in v.items()} for k, v in CONFIGS.items()}, "cases": {}}
for name, (n, a, b) in {"short8s": (130000, 5, 2), "long68s": (1088000, 5, 11)}.items():
    x = clip(n, a, b); long = n > 480000
    r = fe(x, sampling_rate=16000, truncation=not long, padding="longest" if long else "max_length", return_attention_mask=long, return_tensors="pt")
    kw = {"attention_mask": r["attention_mask"]} if long else {}
    res = {}
    for cname, cfg in CONFIGS.items():
        with torch.no_grad():
            try:
                o = model.generate(input_features=r["input_features"], task="transcribe", return_timestamps=True, return_segments=True, temperature=(0.0,), **cfg, **kw)
            except Exception as e:
                print(name, cname, "ERR", type(e).__name__, str(e)[:100]); continue
        segs = [{"start": float(s["start"]), "end": float(s["end"]), "tokens": [int(t) for t in s["tokens"].tolist()]} for s in o["segments"][0]]
        res[cname] = {"segments": segs, "sequence": [int(t) for t in o["sequences"][0].tolist()]}
        print(name, cname, len(segs), "segments", [round(s["start"], 1) for s in segs])
    out["cases"][name] = {"samples": n, "a": a, "b": b, "results": res}
json.dump(out, open(f"{d}/policy_golden.json", "w"))

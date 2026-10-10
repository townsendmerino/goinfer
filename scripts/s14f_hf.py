#!/usr/bin/env python3
"""G-S14f3's reference arm (docs/tasks/task-multimodal-support-2026-10.md, "G-S14f"): transformers 5.15.0 float32 Whisper, short-form greedy with the language detected, on every WAV in <data dir> (scripts/s14c4_data.py's);
writes <out>.json {id: {"text": batch_decode(skip_special_tokens=True), "lang": the language token}}.
Run: ~/g4venv/bin/python -I scripts/s14f_hf.py <whisper dir> <data dir> <out.json> [max clips]"""
import json, sys, time, wave
import numpy as np, torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
model_dir, data, out = sys.argv[1:4]; lim = int(sys.argv[4]) if len(sys.argv) > 4 else 0
model, info = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32, output_loading_info=True, attn_implementation="eager"); model.eval()
assert not info["missing_keys"], "weights missing: the reference would be randomly initialised"
fe, tok = WhisperFeatureExtractor.from_pretrained(model_dir), WhisperTokenizer.from_pretrained(model_dir)
ids = sorted(json.load(open(f"{data}/refs.json")))[: lim or None]
res, t0 = {}, time.time()
for k, cid in enumerate(ids):
    w = wave.open(f"{data}/{cid}.wav"); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    f = fe(x, sampling_rate=16000, return_tensors="pt")["input_features"]
    with torch.no_grad():
        g = model.generate(input_features=f, task="transcribe", return_dict_in_generate=True).sequences[0].tolist()  # with the dict the prompt is kept: g[1] is the language token
    res[cid] = {"text": tok.decode(g, skip_special_tokens=True), "lang": tok.convert_ids_to_tokens([g[1]])[0]}
    print(f"[s14f hf {time.time() - t0:5.0f}s] {k + 1}/{len(ids)} {cid}: {res[cid]['lang']} {res[cid]['text'][:80]!r}", flush=True)
json.dump(res, open(out, "w"), indent=0)

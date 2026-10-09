#!/usr/bin/env python3
"""G-S14c4's reference arm R: transformers 5.15.0 float32 on the converted Qwen3-ASR-0.6B (built by scripts/pin_qwen3asr_real.py), greedy, on every WAV in <data dir>; writes <out>.json {id: raw text}.
Run: ~/g4venv/bin/python -I scripts/s14c4_hf.py <model dir> <converted hf dir> <data dir> <out.json> [max clips]"""
import json, sys, time, wave
import numpy as np, torch
from transformers import AutoTokenizer, Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor
from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor
model_dir, hf_dir, data, out = sys.argv[1:5]; lim = int(sys.argv[5]) if len(sys.argv) > 5 else 0
model = Qwen3ASRForConditionalGeneration.from_pretrained(hf_dir, torch_dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(model_dir)
proc = Qwen3ASRProcessor(feature_extractor=Qwen3ASRFeatureExtractor(), tokenizer=tok, chat_template=json.load(open(f"{model_dir}/chat_template.json"))["chat_template"])
prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "audio", "audio": "x"}]}], add_generation_prompt=True, tokenize=False)
ids = sorted(json.load(open(f"{data}/refs.json")))[: lim or None]
res, t0 = {}, time.time()
for k, cid in enumerate(ids):
    w = wave.open(f"{data}/{cid}.wav"); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    inp = proc(text=[prompt], audio=[x], return_tensors="pt")
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=256, do_sample=False)
    res[cid] = tok.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=False)
    print(f"[s14c4 hf {time.time() - t0:5.0f}s] {k + 1}/{len(ids)} {cid}: {res[cid][:90]!r}", flush=True)
json.dump(res, open(out, "w"), indent=0)

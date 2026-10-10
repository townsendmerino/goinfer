#!/usr/bin/env python3
"""G-S14e3's reference (docs/tasks/task-multimodal-support-2026-10.md, "S14.4b ... G-S14e"): the REAL Voxtral Mini 3B (mistralai/Voxtral-Mini-3B-2507) in transformers 5.15.0 float32 on the CPU.

The reference is built through the official path and PROVED real the way G-S14d2's was: the model loads with zero missing keys (a silently random-initialised reference is the Qwen3-ASR trap) and
transcribes the LibriSpeech clip, asserted (normalised) equal to the corpus text. The request is the processor's own `apply_transcription_request` (mistral_common builds the ids).

Cases: `libri` (the 5.9 s clip, one 30 s window, 375 audio tokens: the registered gate) and `libri6x` (the clip six times with half a second of silence between, 36 s, two windows, 750 audio tokens:
RECORD ONLY, it exercises the chunking on real weights). Per case it writes <out>/<case>.{in.f32,ids.json,feats.f32,enc.f32,embeds.f32,logits.f32,gen.json,tf_logits.f32}: the processor's ids, the
features, the tower output [windows*1500][1280], the projector rows [windows*375][3072], the prompt's last-position logits, the greedy generation, and the logits at every generated position with the
reference's own tokens forced. meta.json carries the versions, the weights' sha256 and the texts.

Run (repo root; mistral_common's dependencies live in a directory of their own so g4venv is untouched):
  PYTHONPATH=~/goinfer-bench/s14e/mc ~/g4venv/bin/python scripts/pin_voxtral_real.py --model ~/models/voxtral-mini-3b-2507 --out <dir>"""
import argparse, hashlib, json, os, re, time, wave
import numpy as np, torch, transformers
from transformers import AutoProcessor, VoxtralForConditionalGeneration

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--language", default="en", help="the request language; 'none' sends language=None (G-S14e4)"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
CORPUS = "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"


def norm(s):  # lowercase, letters and spaces only, "mr" spelled out as the corpus does
    s = re.sub(r"\bmr\.?\b", "mister", s.lower())
    return " ".join(re.sub(r"[^a-z ]", " ", s).split())


w = wave.open("testdata/speech/librispeech-1272-128104-0000.wav")
libri = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
gap = np.zeros(8000, dtype=np.float32)
libri6x = np.concatenate([np.concatenate([libri, gap]) for _ in range(6)])
t0 = time.time()
model, info = VoxtralForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32, output_loading_info=True)
model.eval()
print(f"loaded in {time.time() - t0:.0f}s; missing {sorted(info['missing_keys'])} unexpected {sorted(info['unexpected_keys'])}")
assert not info["missing_keys"], "weights missing: the reference would be randomly initialised"
proc = AutoProcessor.from_pretrained(a.model)
shards = {}
for f in sorted(os.listdir(a.model)):
    if f.endswith(".safetensors"):
        h = hashlib.sha256()
        with open(os.path.join(a.model, f), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        shards[f] = h.hexdigest()
meta = {"transformers": transformers.__version__, "torch": torch.__version__, "shard_sha256": shards, "model_dir": os.path.basename(a.model.rstrip("/")), "cases": {}}
f32 = lambda t: t.detach().float().numpy().astype("<f4").tofile
for name, x, max_new in (("libri", libri, 96), ("libri6x", libri6x, 220)):
    if a.smoke and name != "libri":
        continue
    inputs = proc.apply_transcription_request(language=None if a.language == "none" else a.language, audio=x, model_id=a.model, sampling_rate=16000, format="wav", return_tensors="pt")
    ids = inputs["input_ids"]; feats = inputs["input_features"]
    n_prompt = ids.shape[1]
    t1 = time.time()
    with torch.no_grad():
        ao = model.get_audio_features(feats)
        enc, emb = ao.last_hidden_state.reshape(-1, ao.last_hidden_state.shape[-1]), ao.pooler_output
        logits = model(input_ids=ids, input_features=feats).logits[0, -1]
        gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)[0, n_prompt:]
        full = torch.cat([ids[0], gen])[None]
        tf = model(input_ids=full, input_features=feats).logits[0, n_prompt - 1: n_prompt - 1 + len(gen)]
    text = proc.batch_decode(gen[None], skip_special_tokens=True)[0]
    print(f"{name}: {n_prompt} prompt ids, {feats.shape[0]} window(s), {emb.shape[0]} audio rows; generated {len(gen)} tokens in {time.time() - t1:.0f}s: {text!r}")
    if name == "libri":
        assert norm(text) == CORPUS, f"the reference does not transcribe the clip ({norm(text)!r}): it is not the real model"
    np.asarray(x, "<f4").tofile(f"{a.out}/{name}.in.f32")
    json.dump([int(i) for i in ids[0]], open(f"{a.out}/{name}.ids.json", "w")); json.dump([int(i) for i in gen], open(f"{a.out}/{name}.gen.json", "w"))
    for tag, t in (("feats", feats), ("enc", enc), ("embeds", emb), ("logits", logits), ("tf_logits", tf)):
        f32(t)(f"{a.out}/{name}.{tag}.f32")
    meta["cases"][name] = {"prompt_ids": n_prompt, "windows": int(feats.shape[0]), "audio_rows": int(emb.shape[0]), "generated": len(gen), "text": text, "vocab": int(logits.shape[0]), "enc_rows": int(enc.shape[0])}
json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
print("wrote", a.out)

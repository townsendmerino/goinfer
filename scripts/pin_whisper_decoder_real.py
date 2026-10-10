#!/usr/bin/env python3
"""G-S14f2's reference (docs/tasks/task-multimodal-support-2026-10.md, "G-S14f"): a real Whisper checkpoint's generation in transformers float32 on the LibriSpeech clip: the detected language, the greedy
transcription with the default logits processors, the logits at the last prompt position, and the logits at every generated position teacher-forced on the reference's own tokens.
The reference is PROVED real as G-S14d2's was: zero missing keys, and the transcription asserted against the corpus text. Writes <out>/libri.{in.f32,prompt.json,gen.json,lang.json,plogits.f32,tf.f32} and meta.json.
Run (repo root): ~/g4venv/bin/python -I scripts/pin_whisper_decoder_real.py --model ~/models/whisper-small --out <dir>"""
import argparse, json, os, re, wave
import numpy as np, torch, transformers
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
model, info = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32, output_loading_info=True, attn_implementation="eager"); model.eval()
print("loading info: missing", sorted(info["missing_keys"]), "unexpected", sorted(info["unexpected_keys"]))
assert not info["missing_keys"], "weights missing: the reference would be randomly initialised"
fe = WhisperFeatureExtractor.from_pretrained(a.model); tok = WhisperTokenizer.from_pretrained(a.model)
w = wave.open("testdata/speech/librispeech-1272-128104-0000.wav"); libri = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
f = fe(libri, sampling_rate=16000, return_tensors="pt")["input_features"]
with torch.no_grad():
    lang = model.detect_language(input_features=f)
    lang_id = int(lang.reshape(-1)[0])
    out = model.generate(input_features=f, task="transcribe", return_dict_in_generate=True)
seq = out.sequences[0].tolist()
text = tok.decode(seq, skip_special_tokens=True)
print("detected language id", lang_id, tok.convert_ids_to_tokens([lang_id]), "| sequence head", seq[:6], "| TRANSCRIPTION:", repr(text))
norm = lambda s: " ".join(re.sub(r"[^a-z ]", " ", re.sub(r"\bmr\.?\b", "mister", s.lower())).split())
assert norm(text) == "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel", "the reference does not transcribe: it is not the real model"
# the prompt is the leading special tokens up to and including <|notimestamps|>; what follows is generated
nots = model.generation_config.no_timestamps_token_id
k = seq.index(nots) + 1
prompt, gen = seq[:k], seq[k:]
print("prompt", prompt, "generated", len(gen), "tokens, last", gen[-1], "(eos", model.generation_config.eos_token_id, ")")
with torch.no_grad():
    plog = model(input_features=f, decoder_input_ids=torch.tensor([prompt])).logits[0, -1]
    tf = model(input_features=f, decoder_input_ids=torch.tensor([prompt + gen[:-1]])).logits[0, len(prompt) - 1:]
assert tf.shape[0] == len(gen)
np.asarray(libri, "<f4").tofile(f"{a.out}/libri.in.f32"); plog.numpy().astype("<f4").tofile(f"{a.out}/libri.plogits.f32"); tf.numpy().astype("<f4").tofile(f"{a.out}/libri.tf.f32")
json.dump(prompt, open(f"{a.out}/libri.prompt.json", "w")); json.dump(gen, open(f"{a.out}/libri.gen.json", "w")); json.dump(lang_id, open(f"{a.out}/libri.lang.json", "w"))
json.dump({"transformers": transformers.__version__, "torch": torch.__version__, "text": text, "model": os.path.basename(a.model.rstrip("/"))}, open(f"{a.out}/meta.json", "w"), indent=1)
print("wrote", a.out)

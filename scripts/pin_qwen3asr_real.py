#!/usr/bin/env python3
"""G-S14b3's reference (docs/tasks/task-multimodal-support-2026-10.md, "G-S14b"): Qwen3-ASR-0.6B's audio encoder and projector in transformers 5.15.0, float32, on three clips.
transformers 5.15.0 CANNOT read this checkpoint as shipped (it keeps the nested thinker_config as defaults, builds the 1.7B's dimensions and reports every weight MISSING, randomly initialised, with no error),
so this converts it by a pure rename into a flat-config HF directory, loads that with zero missing and zero unexpected keys, and PROVES the conversion by transcribing the LibriSpeech clip.
Writes <out>/hf (the converted directory), and per clip <out>/<name>.{feats,mask,tower,out}.f32 + <out>/meta.json.
Run: ~/g4venv/bin/python -I scripts/pin_qwen3asr_real.py --model ~/models/qwen3-asr-0.6b --out <dir>   (from the repo root)"""
import argparse, json, os, sys, wave
import numpy as np, torch, transformers
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor
from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(f"{a.out}/hf", exist_ok=True)
real = json.load(open(f"{a.model}/config.json"))["thinker_config"]
ac, tc = dict(real["audio_config"]), dict(real["text_config"])
ac["model_type"] = "qwen3_asr_encoder"
cfg = Qwen3ASRConfig(audio_config=ac, text_config=tc, audio_token_id=real["audio_token_id"])
cfg.save_pretrained(f"{a.out}/hf")
sd = load_file(f"{a.model}/model.safetensors")
def rename(k):
    if k.startswith("thinker.audio_tower.proj1."): return "model.multi_modal_projector.linear_1." + k.split(".")[-1]
    if k.startswith("thinker.audio_tower.proj2."): return "model.multi_modal_projector.linear_2." + k.split(".")[-1]
    if k.startswith("thinker.audio_tower."): return "model." + k[len("thinker."):]
    if k.startswith("thinker.model."): return "model.language_model." + k[len("thinker.model."):]
    if k == "thinker.lm_head.weight": return "lm_head.weight"
    raise KeyError(k)
save_file({rename(k): v.contiguous() for k, v in sd.items()}, f"{a.out}/hf/model.safetensors")
model, info = Qwen3ASRForConditionalGeneration.from_pretrained(f"{a.out}/hf", torch_dtype=torch.float32, output_loading_info=True)
model.eval()
print("loading info: missing", sorted(info["missing_keys"]), "unexpected", sorted(info["unexpected_keys"]), "mismatched", info.get("mismatched_keys"))
assert not info["missing_keys"] and not info["unexpected_keys"], "the conversion is not a pure rename"
print("audio d_model", model.config.audio_config.d_model, "layers", model.config.audio_config.encoder_layers, "text hidden", model.config.text_config.hidden_size)

def wav(p):
    w = wave.open(p); return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
libri = wav("testdata/speech/librispeech-1272-128104-0000.wav")
tok = AutoTokenizer.from_pretrained(a.model)
fe = Qwen3ASRFeatureExtractor()
tmpl = json.load(open(f"{a.model}/chat_template.json"))["chat_template"]
proc = Qwen3ASRProcessor(feature_extractor=fe, tokenizer=tok, chat_template=tmpl)
# PROOF OF THE CONVERSION: a real transcription of the LibriSpeech clip ("MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL").
prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "audio", "audio": "x"}]}], add_generation_prompt=True, tokenize=False)
print("PROMPT:", repr(prompt))
inp = proc(text=[prompt], audio=[libri], return_tensors="pt")
with torch.no_grad():
    gen = model.generate(**inp, max_new_tokens=64, do_sample=False)
text = tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=False)
print("TRANSCRIPTION:", repr(text))
assert "quilter" in text.lower() and "gospel" in text.lower(), "the converted reference does not transcribe: the conversion or the processor is wrong"

def synth(n, f0):
    i = np.arange(n, dtype=np.float64); noise = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0 - 0.5
    return (0.3 * np.sin(2 * np.pi * f0 * i / 16000) + 0.15 * np.sin(2 * np.pi * (f0 * 2.7) * i / 16000) * (1 + np.sin(i / 3000.0)) + 0.1 * noise).astype(np.float32)
clips = [("libri", libri), ("synth12s", synth(192000, 260.0)), ("long46s8", np.tile(libri, 8))]  # the last: 46.8 s, past one 30 s window, many attention windows
meta = {"transformers": transformers.__version__, "torch": torch.__version__, "transcription": text, "cases": []}
for name, x in clips:
    b = fe(x, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding=True, truncation=False, n_window=50)
    feats, mask = b["input_features"], b["attention_mask"]
    with torch.no_grad():
        tower = model.model.audio_tower(input_features=feats, input_features_mask=mask).last_hidden_state
        out = model.model.multi_modal_projector(tower)
    np.asarray(x, "<f4").tofile(f"{a.out}/{name}.in.f32"); feats[0].numpy().astype("<f4").tofile(f"{a.out}/{name}.feats.f32")
    mask[0].numpy().astype("u1").tofile(f"{a.out}/{name}.mask.u8"); tower.numpy().astype("<f4").tofile(f"{a.out}/{name}.tower.f32"); out.numpy().astype("<f4").tofile(f"{a.out}/{name}.out.f32")
    meta["cases"].append({"name": name, "samples": int(len(x)), "T": int(feats.shape[-1]), "valid": int(mask.sum()), "tokens": int(tower.shape[0])})
    print(f"{name}: {len(x)} samples, T {feats.shape[-1]}, valid {int(mask.sum())}, tokens {tower.shape[0]}, tower rms {tower.pow(2).mean().sqrt():.3f}, out rms {out.pow(2).mean().sqrt():.3f}")

# G-S14c1 (real, text path) and G-S14c3 (real, composition): the decoder half, with HF's own prompt ids and greedy transcription, and HF's logits at every generated position
# (its own path forced), for goinfer to be held to token by token.
if not os.path.exists(f"{a.model}/tokenizer.json"):  # the repo ships vocab.json + merges.txt only; goinfer's tokenizer reads tokenizer.json, so write the one HF builds from those
    tok.backend_tokenizer.save(f"{a.model}/tokenizer.json"); print("wrote", f"{a.model}/tokenizer.json", "(generated by transformers from vocab.json + merges.txt)")
cap = tok("The capital of France is", return_tensors="pt")["input_ids"]
with torch.no_grad():
    tl = model(input_ids=cap).logits[0, -1]
    cont = model.generate(input_ids=cap, attention_mask=torch.ones_like(cap), max_new_tokens=6, min_new_tokens=6, do_sample=False)[0, cap.shape[1]:].tolist()
np.asarray(tl.numpy(), "<f4").tofile(f"{a.out}/text.logits.f32")
meta["text"] = {"ids": cap[0].tolist(), "continuation": cont}
inp = proc(text=[prompt], audio=[libri], return_tensors="pt")
with torch.no_grad():
    gen = model.generate(**inp, max_new_tokens=64, do_sample=False)
P = inp["input_ids"].shape[1]; gen_ids = gen[0, P:].tolist()
full = gen[:, :]
with torch.no_grad():
    lg = model(input_ids=full, input_features=inp["input_features"], input_features_mask=inp["input_features_mask"]).logits[0].float()
tf = lg[P - 1: P - 1 + len(gen_ids)]  # the logits that predicted each generated token
np.asarray(tf.numpy(), "<f4").tofile(f"{a.out}/libri.tf_logits.f32")
meta["transcribe"] = {"prompt_ids": inp["input_ids"][0].tolist(), "generated_ids": gen_ids, "text": tok.decode(gen_ids, skip_special_tokens=False), "vocab": int(tf.shape[1])}
print("generated", len(gen_ids), "tokens:", meta["transcribe"]["text"])
json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)

#!/usr/bin/env python3
"""G-S14g2's fixture and reference (docs/tasks/task-multimodal-support-2026-10.md, "G-S14g"): a tiny random-weight Whisper whose vocabulary has the real layout in miniature (3,110 ids: text 0-1,599, stop 1,600, start
1,601, <|en|> <|fr|> <|de|> 1,602-1,604, translate 1,605, transcribe 1,606, <|notimestamps|> 1,607, 1,501 timestamp tokens from 1,608), built from transformers' own classes and saved the ordinary way, with a generation_config.json
carrying what generate needs; and transformers' generate(return_timestamps=True, return_segments=True) on a 7 s clip and an 8 s clip (short form; the 8 s one ends a window on a single timestamp) and a 68 s clip (long form, three windows) as the reference: per clip the segments (start, end,
tokens), the final sequence, and the detected language token. Random weights make every branch of the timestamp rules fire. Writes <out>/{config.json,generation_config.json,model.safetensors,golden.json}.
Run (repo root): ~/g4venv/bin/python -I scripts/pin_whisper_ts_tiny.py testdata/whisper-tiny-ts"""
import json, os, sys
import numpy as np, torch, transformers
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
torch.manual_seed(23)
V, EOS, SOT, LANG, TRANSL, TRANSC, NOTS, TS0 = 3110, 1600, 1601, 1602, 1605, 1606, 1607, 1608
cfg = WhisperConfig(vocab_size=V, num_mel_bins=80, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
                    max_source_positions=1500, max_target_positions=64, pad_token_id=EOS, bos_token_id=EOS, eos_token_id=EOS, decoder_start_token_id=SOT)
cfg._attn_implementation = "eager"
model = WhisperForConditionalGeneration(cfg).float().eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if "embed_positions" in n and "encoder" in n: p.normal_(0, 0.5)
        elif p.dim() >= 3 or (p.dim() == 2 and "embed" not in n):
            fan_in = p.numel() // p.shape[0]
            gain = 5.0 if ("q_proj" in n or "k_proj" in n) else (3.0 if ("fc1" in n or "conv" in n) else 1.5)
            p.normal_(0, gain / fan_in ** 0.5)
        elif p.dim() == 2: p.normal_(0, 0.5)
        elif "layer_norm" in n and n.endswith("weight"): p.copy_(1 + 0.2 * torch.randn_like(p))
        else: p.normal_(0, 0.3)
model.save_pretrained(out)
gc = {"decoder_start_token_id": SOT, "eos_token_id": EOS, "pad_token_id": EOS, "bos_token_id": EOS, "no_timestamps_token_id": NOTS, "max_initial_timestamp_index": 50, "max_length": 64, "is_multilingual": True,
      "lang_to_id": {"<|en|>": LANG, "<|fr|>": LANG + 1, "<|de|>": LANG + 2}, "task_to_id": {"translate": TRANSL, "transcribe": TRANSC},
      "suppress_tokens": [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931], "begin_suppress_tokens": [220, EOS], "return_timestamps": False}
json.dump(gc, open(f"{out}/generation_config.json", "w"), indent=1)
model = WhisperForConditionalGeneration.from_pretrained(out, dtype=torch.float32, attn_implementation="eager").eval()
fe = WhisperFeatureExtractor(feature_size=80)
def clip(n, a, b):
    """Integer-driven synthesis (sawtooth, sawtooth, hash noise): every step is exact in float64, so goinfer's test rebuilds the same float32 samples with no sin() ulp difference."""
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        saw1 = ((i * a) % 1000) / 1000 - 0.5
        saw2 = ((i * b) % 1733) / 1733 - 0.5
        env = ((i // 4000) % 3) / 2.0
        noise = ((i * 2654435761) & 0xFFFFFFFF) / 4294967296.0 - 0.5
        out[i] = np.float32(0.25 * saw1 + 0.2 * saw2 * env + 0.1 * noise * (1 + ((i // 9000) % 2)))
    return out
golden = {"transformers": transformers.__version__, "torch": torch.__version__, "cases": {}}
for name, x in (("short7s", clip(112000, 7, 3)), ("short8s", clip(130000, 5, 2)), ("long68s", clip(1088000, 5, 11))):
    long = len(x) > 480000
    r = fe(x, sampling_rate=16000, truncation=not long, padding="longest" if long else "max_length", return_attention_mask=long, return_tensors="pt")
    kw = {"attention_mask": r["attention_mask"]} if long else {}
    with torch.no_grad():
        lang_id = int(model.detect_language(input_features=r["input_features"][:, :, :3000]).reshape(-1)[0])
        o = model.generate(input_features=r["input_features"], task="transcribe", return_timestamps=True, return_segments=True, **kw)
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "tokens": [int(t) for t in s["tokens"].tolist()]} for s in o["segments"][0]]
    seq = [int(t) for t in o["sequences"][0].tolist()]
    golden["cases"][name] = {"language_id": lang_id, "samples": int(len(x)), "a": {"short7s": 7, "short8s": 5, "long68s": 5}[name], "b": {"short7s": 3, "short8s": 2, "long68s": 11}[name], "segments": segs, "sequence": seq, "windows": int(np.ceil((len(x) // 160) / 3000)) if long else 1}
    print(name, "language id", lang_id, "segments", len(segs), "first", segs[0]["start"], segs[0]["end"], "last end", segs[-1]["end"], "| sequence head", seq[:5], "len", len(seq))
json.dump(golden, open(f"{out}/golden.json", "w"))

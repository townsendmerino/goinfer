#!/usr/bin/env python3
"""G-S14f1's golden (docs/tasks/task-multimodal-support-2026-10.md, "G-S14f"): transformers' own WhisperDecoder on the tiny random-weight Whisper of scripts/pin_whisper_tiny.py (testdata/whisper-tiny-rand, which
holds the decoder's weights), given the fixture's encoder output for each of its two clips (golden.zip's <clip>.enc.f32): the teacher-forced logits at every position of a fixed 12-token input, and a 12-token
greedy continuation from the decoder start token with no logits processors. Writes testdata/whisper-tiny-rand/dec_golden.zip.
Run (repo root): ~/g4venv/bin/python -I scripts/pin_whisper_decoder_tiny.py testdata/whisper-tiny-rand"""
import json, sys, zipfile
import numpy as np, torch, transformers
from transformers import WhisperForConditionalGeneration

d = sys.argv[1]
model = WhisperForConditionalGeneration.from_pretrained(d, dtype=torch.float32, attn_implementation="eager").eval()
src = zipfile.ZipFile(f"{d}/golden.zip")
IDS = [3, 17, 99, 5, 42, 150, 8, 77, 120, 33, 66, 9]
meta = {"transformers": transformers.__version__, "torch": torch.__version__, "ids": IDS, "start": model.config.decoder_start_token_id, "cases": []}
with zipfile.ZipFile(f"{d}/dec_golden.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name in ("clip6s5", "clip25s"):
        enc = torch.from_numpy(np.frombuffer(src.read(f"{name}.enc.f32"), dtype="<f4").copy()).reshape(1, 1500, 32)
        with torch.no_grad():
            h = model.model.decoder(input_ids=torch.tensor([IDS]), encoder_hidden_states=enc).last_hidden_state
            logits = model.proj_out(h)[0]
            ids = [model.config.decoder_start_token_id]
            for _ in range(12):
                out = model.proj_out(model.model.decoder(input_ids=torch.tensor([ids]), encoder_hidden_states=enc).last_hidden_state)[0, -1]
                ids.append(int(out.argmax()))
        z.writestr(f"{name}.logits.f32", logits.numpy().astype("<f4").tobytes())
        z.writestr(f"{name}.greedy.json", json.dumps(ids[1:]))
        meta["cases"].append({"name": name, "logit_rms": float(logits.pow(2).mean().sqrt()), "logit_absmax": float(logits.abs().max())})
        print(name, "logits", tuple(logits.shape), "rms %.3f max %.3f" % (logits.pow(2).mean().sqrt(), logits.abs().max()), "greedy", ids[1:])
    z.writestr("meta.json", json.dumps(meta, indent=1))

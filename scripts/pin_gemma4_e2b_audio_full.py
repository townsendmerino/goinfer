#!/usr/bin/env python3
"""G-S5b's reference (docs/tasks/task-multimodal-support-2026-10.md, S5): transformers' Gemma4ForConditionalGeneration,
float32, on goinfer's own prompt ids for the mid clip (decoder/gemma4_audio_e2b_real_test.go, step "ids", writes
<out>/ids.json) and the clip's log-mel from Gemma4AudioFeatureExtractor (its defaults equal E2B's processor config,
checked by pin_gemma4_e2b_audio.py). Writes <out>/hf_last_logits.f32 (float32 little-endian, the vocabulary) and
<out>/hf_argmax.json (the argmax at every position). About 20 GB in float32: run where that fits (nobara).

Checks before it records: the attention implementation, the audio tower's valid rows equal the ids' audio run, and the
model's audio token id is the one in the ids.

Run: python3 scripts/pin_gemma4_e2b_audio_full.py --model ~/models/gemma-4-E2B-unq \
         --wav testdata/embeddinggemma2-audio/mid.wav --out <dir with ids.json>
"""
import argparse
import json
import os
import sys
import wave

import numpy as np
import torch
import transformers
from transformers import Gemma4ForConditionalGeneration
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_dir = os.path.expanduser(a.model)
    if model_dir.startswith("/Volumes/") or model_dir.startswith("/srv/models"):
        sys.exit(f"{model_dir} is on the archive, not the bench set (CLAUDE.md)")
    ids_in = json.load(open(os.path.join(a.out, "ids.json")))
    ids = ids_in["ids"]
    with wave.open(a.wav, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != 16000:
            sys.exit("want 16 kHz mono 16-bit")
        wav = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    f = Gemma4AudioFeatureExtractor()([wav], return_tensors="pt")
    feats, mask = f["input_features"].float(), f["input_features_mask"]
    if int(mask[0].sum()) != ids_in["frames"]:
        sys.exit(f"{int(mask[0].sum())} valid frames, the ids were built for {ids_in['frames']}")

    torch.set_num_threads(max(1, os.cpu_count() - 2))
    model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    atok = model.config.audio_token_id
    if ids.count(atok) != ids_in["audio_len"]:
        sys.exit(f"{ids.count(atok)} audio tokens (id {atok}) in the ids, want {ids_in['audio_len']}")
    with torch.no_grad():
        af = model.model.get_audio_features(feats, mask, return_dict=True)
        n = int(af.attention_mask[0].sum())
        if n != ids_in["audio_len"]:
            sys.exit(f"the tower gives {n} valid rows, the ids hold {ids_in['audio_len']}")
        out = model(input_ids=torch.tensor([ids]), input_features=feats, input_features_mask=mask)
    logits = out.logits[0].float()
    logits[-1].numpy().astype("<f4").tofile(os.path.join(a.out, "hf_last_logits.f32"))
    json.dump([int(x) for x in logits.argmax(-1)], open(os.path.join(a.out, "hf_argmax.json"), "w"))
    json.dump({"transformers": transformers.__version__, "torch": torch.__version__, "ids": len(ids),
               "audio_run": [ids_in["audio_pos"], ids_in["audio_pos"] + ids_in["audio_len"]],
               "last_argmax": int(logits[-1].argmax())}, open(os.path.join(a.out, "hf_provenance.json"), "w"), indent=1)
    print(f"{len(ids)} ids, last argmax {int(logits[-1].argmax())}", file=sys.stderr)


if __name__ == "__main__":
    main()

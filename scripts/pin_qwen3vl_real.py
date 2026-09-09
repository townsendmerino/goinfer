#!/usr/bin/env python
"""Real-model parity golden for Qwen3-VL-2B-Instruct (qwen3_vl, TEXT-ONLY — P8 Phase 0,
docs/multimodal.md): the real HF reference for goinfer's qwen3_vlArchitecture on actual
released weights. No pixel_values, no vision tower, no DeepStack — proves the text decoder
(Qwen3's attention shape: per-head q/k RMSNorm, GQA, no q/k/v bias, wired here under
Qwen3-VL's nested text_config) loads and runs correctly on a real checkpoint, mirroring
scripts/pin_phi3_mini.py's shape. Dumps last-token logits + argmax + greedy continuation;
decoder/qwen3vl_real_test.go (build tag realckpt) loads the same safetensors and matches.

Needs GOINFER_QWEN3VL_2B (testdata/assets.json) — pull with:
    models-pull qwen3vl-2b-instruct   (or download Qwen/Qwen3-VL-2B-Instruct directly)

    ~/.venv-vl/bin/python scripts/pin_qwen3vl_real.py
    -> testdata/qwen3vl_real_golden.json   (committed; weights are NOT)
"""
import json
import os

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

CKPT = os.environ.get("GOINFER_QWEN3VL_2B", os.path.expanduser("~/models/qwen3vl-2b-instruct"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "qwen3vl_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 6


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist())
    m = Qwen3VLForConditionalGeneration.from_pretrained(
        CKPT, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    with torch.no_grad():
        last = m(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur = ids.tolist()[0]
        cont = []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])
    g = dict(
        note="Qwen3-VL-2B-Instruct TEXT-ONLY f32 reference (no pixel_values)",
        prompt=PROMPT, prompt_ids=ids[0].tolist(),
        argmax=int(torch.tensor(last).argmax()),
        last_logits=last, n_new=N_NEW, continuation_ids=cont,
        continuation_text=tok.decode(cont),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont!r} -> {g['continuation_text']!r}")
    print("saved", OUT)


if __name__ == "__main__":
    main()

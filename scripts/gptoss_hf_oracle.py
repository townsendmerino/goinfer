#!/usr/bin/env python3
"""HF reference oracle for the gpt-oss:20b logit-parity gate (docs/task-mxfp4-gptoss.md
§6.2). Loads openai/gpt-oss-20b with transformers on CPU (eager attention, so the
per-head sink softmax runs), runs a fixed prompt, and emits a golden JSON with the
prompt token ids + the next-token argmax + the full last-position logit vector.

goinfer (decoder/gptoss_real_test.go, the -tags realckpt logit gate) feeds the SAME
token ids and compares: argmax-exact + logit cosine. goinfer loads int8-resident while
HF is bf16, so expect argmax-exact with cosine ~0.99+ (the deepseek/llama4 real-model
bar), not bit-exact.

  ~/.venv-vl/bin/python3 scripts/gptoss_hf_oracle.py \
      ~/models/gpt-oss-20b-hf decoder/testdata/gptoss_20b_golden.json
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is"


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/francis/models/gpt-oss-20b-hf"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "decoder/testdata/gptoss_20b_golden.json"

    tok = AutoTokenizer.from_pretrained(model_dir)
    sys.stderr.write("loading gpt-oss-20b (CPU, bf16, eager attn — dequantizes MXFP4)…\n")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager",
    ).eval()

    ids = tok(PROMPT, return_tensors="pt").input_ids
    sys.stderr.write(f"prompt ids: {ids[0].tolist()}\n")
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float().numpy()
    argmax = int(logits.argmax())
    sys.stderr.write(f"argmax={argmax} decode={tok.decode([argmax])!r}\n")

    golden = {
        "prompt": PROMPT,
        "prompt_ids": ids[0].tolist(),
        "argmax": argmax,
        "last_logits": logits.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(golden, f)
    sys.stderr.write(f"wrote {out_path} ({len(logits)} logits)\n")


if __name__ == "__main__":
    main()

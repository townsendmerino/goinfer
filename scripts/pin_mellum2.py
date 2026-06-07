#!/usr/bin/env python3
"""Pin Mellum2 (JetBrains/Mellum2-12B-A2.5B-Instruct) HF bf16 logit goldens for
goinfer's parity tests — the only family in the README list previously without a
parity golden. 12B MoE bf16 ≈ 24 GB, fits the 64 GB box; one forward per golden.

Writes two forwardGolden records (the format decoder/forward_test.go consumes):

  mellum2_forward_golden.json  — chat-templated prompt; argmax is the first
      answer token, so the parity test doubles as a coherence gate.
  mellum2_window_golden.json   — a long (> sliding_window=1024) prompt; the
      next-token logits at a past-window position pin the 3:1 sliding/full
      eviction + YaRN-on-full RoPE interaction on the real checkpoint (Inc3).

    ~/g4venv/bin/python scripts/pin_mellum2.py
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = os.path.expanduser("~/models/mellum2-unq")
TD = os.path.expanduser("~/mycode/goinfer/testdata")


def record(model, tok, ids, prompt_note):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]))
    logits = out.logits[0, -1].float().numpy()
    order = np.argsort(-logits)
    rng = np.random.default_rng(1234)
    samp = rng.choice(len(logits), 256, replace=False)
    lf = logits.astype(np.float64)
    return {
        "model_id": "JetBrains/Mellum2-12B-A2.5B-Instruct",
        "note": prompt_note,
        "dtype": "bfloat16",
        "prompt": "",
        "ids": [int(x) for x in ids],
        "argmax": int(order[0]),
        "argmax_token": tok.decode([int(order[0])]),
        "vocab_size": int(len(logits)),
        "stats": {
            "n": int(len(logits)), "sum": float(lf.sum()), "sum_sq": float((lf * lf).sum()),
            "min": float(logits.min()), "max": float(logits.max()),
        },
        "top_k": [[int(i), float(logits[i])] for i in order[:16]],
        "sample": [[int(i), float(logits[i])] for i in samp],
    }


def main():
    tok = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()

    # 1. Chat-templated prompt → answer token (Inc2). Mellum2's template is ChatML.
    # Render to string then tokenize to a plain id list (this tokenizers version's
    # apply_chat_template return_tensors path yields an Encoding, not a tensor).
    chat_text = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        add_generation_prompt=True, tokenize=False,
    )
    chat_ids = tok(chat_text, add_special_tokens=False)["input_ids"]
    g1 = record(model, tok, chat_ids, "chat-templated (HF bf16, CPU); first answer token. Inc2 parity gate.")
    json.dump(g1, open(os.path.join(TD, "mellum2_forward_golden.json"), "w"), indent=1)
    print(f"chat golden: {len(chat_ids)} ids, argmax {g1['argmax']} = {g1['argmax_token']!r}")

    # 2. Long prompt past the 1024 sliding window (Inc3) — pins eviction + YaRN.
    long_text = (
        "The following is a long technical passage about distributed systems. "
        "Consistency, availability, and partition tolerance form the CAP theorem. "
    ) * 60
    long_ids = tok(long_text, add_special_tokens=False)["input_ids"]
    assert len(long_ids) > 1100, f"long prompt only {len(long_ids)} tokens; need > window 1024"
    g2 = record(model, tok, long_ids, f"long prompt ({len(long_ids)} tok > window 1024); pins sliding eviction + YaRN-on-full. Inc3.")
    json.dump(g2, open(os.path.join(TD, "mellum2_window_golden.json"), "w"), indent=1)
    print(f"window golden: {len(long_ids)} ids (> 1024), argmax {g2['argmax']} = {g2['argmax_token']!r}")


if __name__ == "__main__":
    main()

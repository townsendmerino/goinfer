#!/usr/bin/env python3
"""Real-model parity golden for Aya-Expanse-8B (model_type "cohere", CohereForCausalLM) —
the full bf16/f32 reference for goinfer's Cohere / Command-R Phase 1 on ACTUAL released
weights (docs/task-model-family-cohere.md). Aya-Expanse-8B is the ideal T3 target: it is
the smallest cohere1 checkpoint, and it is use_qk_norm=FALSE — exactly the Phase-1 scope
(the LayerNorm QK-norm of Command-R+ is deferred). It exercises the real bias-free
LayerNorm + parallel attn/MLP block + GPT-J interleaved RoPE + logit_scale multiplier +
tied 256k embeddings on genuine weights, at full head_dim=128 / GQA (32 heads, 8 kv).

Dumps the last-token logits + argmax + a short greedy continuation (token IDs) for a fixed
prompt; the goinfer side (cohere_aya_real_test.go, build tag realckpt) loads the same
safetensors at f32 and matches argmax + continuation + cosine ≥ 0.9999.

    ~/.venv-vl/bin/python scripts/pin_cohere_aya.py
    -> testdata/cohere_aya_golden.json   (committed; the 8B weights are NOT)

Put the checkpoint at ~/models/aya-expanse-8b (CohereForAI/aya-expanse-8b), or set
GOINFER_COHERE_AYA to its path.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ.get("GOINFER_COHERE_AYA", os.path.expanduser("~/models/aya-expanse-8b"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "cohere_aya_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32).eval()
    arch = model.config.architectures[0] if model.config.architectures else "?"
    print(f"loaded {arch} model_type={model.config.model_type} "
          f"use_qk_norm={getattr(model.config, 'use_qk_norm', None)} "
          f"logit_scale={getattr(model.config, 'logit_scale', None)}")

    ids = tok(PROMPT, return_tensors="pt").input_ids
    prompt_ids = ids[0].tolist()
    with torch.no_grad():
        last = model(ids).logits[0, -1].to(torch.float64)  # already ×logit_scale
        argmax = int(torch.argmax(last).item())
        cont = []
        cur = ids
        for _ in range(N_NEW):
            nxt = int(torch.argmax(model(cur).logits[0, -1]).item())
            cont.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    golden = dict(
        prompt=PROMPT,
        prompt_ids=prompt_ids,
        argmax=argmax,
        vocab_size=model.config.vocab_size,
        last_logits=[float(x) for x in last.tolist()],
        n_new=N_NEW,
        continuation_ids=cont,
    )
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print("wrote", os.path.relpath(OUT))
    print("prompt_ids", prompt_ids, "argmax", argmax, "cont", cont)
    print("continuation:", repr(tok.decode(cont)))


if __name__ == "__main__":
    main()

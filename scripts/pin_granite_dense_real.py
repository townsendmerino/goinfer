#!/usr/bin/env python3
"""Real-model parity golden for dense Granite 4.2 (model_type "granite", GraniteForCausalLM) —
the T3 promotion of goinfer's DENSE granite family from tiny-golden to a released checkpoint
(docs/parity-coverage-policy.md).

NAME COLLISION WARNING, READ BEFORE TOUCHING: "granite" (this family, GraniteForCausalLM, a
plain llama skeleton + four scalar multipliers) is NOT "granitemoehybrid" (the Mamba-2 +
attention + MoE hybrid, GraniteMoeHybridForCausalLM, already real-oracle 100.0%/0.99566 via
GOINFER_GRANITE_HF / decoder/granite_real_test.go's TestGraniteReal_oracle, which asserts
arch.Name == "granitemoehybrid"). This script and its gate target the OTHER family — do not
reuse that env var or extend that test.

ibm-granite/granite-4.2-3b is the smallest released dense size. Real config: attention_multiplier
0.015625 (the only one of Granite's four scalars that deviates from its identity default on any
released size), embedding_multiplier/residual_multiplier/logits_scaling all 1.0
(docs/task-families-2026-09.md F3).

Dumps the last-token logits + argmax + a short greedy continuation (token IDs) for a fixed
prompt; the goinfer side (granite_dense_real_test.go, build tag realckpt) loads the same
safetensors at f32 and matches argmax + continuation + cosine >= 0.9999.

    ~/.venv-vl/bin/python scripts/pin_granite_dense_real.py
    -> testdata/granite_dense_real_golden.json   (committed; the ~6 GB weights are NOT)

Put the checkpoint at ~/models/granite-4.2-3b (ibm-granite/granite-4.2-3b), or set
GOINFER_GRANITE_DENSE_3B to its path.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ.get("GOINFER_GRANITE_DENSE_3B", os.path.expanduser("~/models/granite-4.2-3b"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "granite_dense_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32).eval()
    cfg = model.config
    arch = cfg.architectures[0] if cfg.architectures else "?"
    print(f"loaded {arch} model_type={cfg.model_type} num_hidden_layers={cfg.num_hidden_layers} "
          f"embedding_multiplier={getattr(cfg,'embedding_multiplier',None)} "
          f"attention_multiplier={getattr(cfg,'attention_multiplier',None)} "
          f"residual_multiplier={getattr(cfg,'residual_multiplier',None)} "
          f"logits_scaling={getattr(cfg,'logits_scaling',None)}")

    ids = tok(PROMPT, return_tensors="pt").input_ids
    prompt_ids = ids[0].tolist()
    with torch.no_grad():
        last = model(ids).logits[0, -1].to(torch.float64)
        argmax = int(torch.argmax(last).item())
        cont, cur = [], ids
        for _ in range(N_NEW):
            nxt = int(torch.argmax(model(cur).logits[0, -1]).item())
            cont.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    golden = dict(
        prompt=PROMPT,
        prompt_ids=prompt_ids,
        argmax=argmax,
        vocab_size=cfg.vocab_size,
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

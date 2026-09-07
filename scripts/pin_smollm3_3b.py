#!/usr/bin/env python3
"""Real-model parity golden for SmolLM3-3B (model_type "smollm3", SmolLM3ForCausalLM) —
the T3 promotion of goinfer's SmolLM3 family from tiny-golden to a released checkpoint
(docs/parity-coverage-policy.md). SmolLM3-3B is the smallest released SmolLM3 checkpoint
and is the only size in the family, so it is both the smallest AND the real model. It
exercises the family's one non-llama primitive on real weights: per-layer NoPE via
`no_rope_layers` (a field whose VALUES are the opposite of what its name suggests — 1
means "has RoPE", 0 means NoPE — already pinned on the tiny fixture in
scripts/pin_smollm3_tiny.py).

Dumps the last-token logits + argmax + a short greedy continuation (token IDs) for a
fixed prompt; the goinfer side (smollm3_real_test.go, build tag realckpt) loads the same
safetensors at f32 and matches argmax + continuation + cosine >= 0.9999.

    ~/.venv-vl/bin/python scripts/pin_smollm3_3b.py
    -> testdata/smollm3_3b_golden.json   (committed; the ~6 GB weights are NOT)

Put the checkpoint at ~/models/smollm3-3b (HuggingFaceTB/SmolLM3-3B), or set
GOINFER_SMOLLM3_3B to its path.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ.get("GOINFER_SMOLLM3_3B", os.path.expanduser("~/models/smollm3-3b"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "smollm3_3b_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32).eval()
    cfg = model.config
    arch = cfg.architectures[0] if cfg.architectures else "?"
    no_rope_layers = getattr(cfg, "no_rope_layers", None)
    print(f"loaded {arch} model_type={cfg.model_type} num_hidden_layers={cfg.num_hidden_layers} "
          f"no_rope_layer_interval={getattr(cfg, 'no_rope_layer_interval', None)}")
    if no_rope_layers is not None:
        print(f"no_rope_layers: {no_rope_layers}")

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
        no_rope_layers=no_rope_layers,
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

#!/usr/bin/env python3
"""Real-model parity golden for Command-R7B (model_type "cohere2",
Cohere2ForCausalLM) — the T3 proof of goinfer's Cohere2 / Command-R7B on ACTUAL
released weights (docs/task-model-family-cohere.md, Phase 2). Command-R7B is the
smallest cohere2 checkpoint and exercises everything cohere1 does PLUS the two
Phase-2 primitives on real weights: interleaved sliding-window / full attention
(sliding_window 4096, every 4th layer global) and NoPE on the global layers
(only the sliding layers carry RoPE). No QK-norm.

Dumps the last-token logits + argmax + a short greedy continuation (token IDs)
for a fixed prompt; the goinfer side (cohere2_r7b_real_test.go, build tag
realckpt) loads the same safetensors at f32 and matches argmax + continuation +
cosine ≥ 0.9999.

    ~/.venv-vl/bin/python scripts/pin_cohere2_r7b.py
    -> testdata/cohere2_r7b_golden.json   (committed; the 7B weights are NOT)

Put the checkpoint at ~/models/command-r7b (CohereForAI/c4ai-command-r7b-12-2024),
or set GOINFER_COHERE2_R7B to its path.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ.get("GOINFER_COHERE2_R7B", os.path.expanduser("~/models/command-r7b"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "cohere2_r7b_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32).eval()
    cfg = model.config
    arch = cfg.architectures[0] if cfg.architectures else "?"
    layer_types = getattr(cfg, "layer_types", None)
    print(f"loaded {arch} model_type={cfg.model_type} sliding_window={getattr(cfg,'sliding_window',None)} "
          f"logit_scale={getattr(cfg,'logit_scale',None)} use_qk_norm={getattr(cfg,'use_qk_norm',None)}")
    if layer_types is not None:
        print(f"layer_types (first 8): {layer_types[:8]}")

    ids = tok(PROMPT, return_tensors="pt").input_ids
    prompt_ids = ids[0].tolist()
    with torch.no_grad():
        last = model(ids).logits[0, -1].to(torch.float64)  # already ×logit_scale
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
        sliding_window=getattr(cfg, "sliding_window", None),
        layer_types=layer_types,
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

#!/usr/bin/env python3
"""Real-model parity golden for Olmo 3 (model_type "olmo3", Olmo3ForCausalLM) — the T3
promotion of goinfer's olmo3 family from tiny-golden to a released checkpoint
(docs/parity-coverage-policy.md). allenai/Olmo-3-7B-Think is the smallest released size
(32B also exists but was scoped out of the T3 target, docs/task-families-2026-09.md G2).

Exercises Olmo 3's two real departures on real weights: NormPostOnly (no pre-norm at all,
only the sublayer OUTPUT is normalized before the residual add) and QKNormWhole (QK-norm over
the FULL projected q/k vector, not per head) — plus the local/global RoPE split, since the real
release is sliding/full 3:1 (sliding_window=4096) with YaRN applying to full_attention layers
only.

Dumps the last-token logits + argmax + a short greedy continuation (token IDs) for a fixed
prompt; the goinfer side (olmo3_real_test.go, build tag realckpt) loads the same safetensors at
f32 and matches argmax + continuation + cosine >= 0.9999.

    ~/.venv-vl/bin/python scripts/pin_olmo3_real.py
    -> testdata/olmo3_real_golden.json   (committed; the ~14 GB weights are NOT)

Put the checkpoint at ~/models/olmo3-7b-think (allenai/Olmo-3-7B-Think), or set
GOINFER_OLMO3_7B to its path.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ.get("GOINFER_OLMO3_7B", os.path.expanduser("~/models/olmo3-7b-think"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "olmo3_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32).eval()
    cfg = model.config
    arch = cfg.architectures[0] if cfg.architectures else "?"
    layer_types = getattr(cfg, "layer_types", None)
    n_sliding = sum(1 for t in (layer_types or []) if t == "sliding_attention")
    n_full = sum(1 for t in (layer_types or []) if t == "full_attention")
    print(f"loaded {arch} model_type={cfg.model_type} num_hidden_layers={cfg.num_hidden_layers} "
          f"sliding_window={getattr(cfg,'sliding_window',None)} layers: {n_sliding} sliding / {n_full} full "
          f"rope_scaling={getattr(cfg,'rope_scaling',None)}")

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

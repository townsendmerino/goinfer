#!/usr/bin/env python3
"""Real-model parity golden for Olmo Hybrid (model_type "olmo_hybrid", OlmoHybridForCausalLM) —
the T3 promotion of goinfer's olmo_hybrid family from tiny-golden to a released checkpoint
(docs/parity-coverage-policy.md). allenai/Olmo-Hybrid-7B is the only released size.

Exercises the family's real departures on real weights: per-layer NormPlacement (full-attention
layers use olmo3's NormPostOnly, DeltaNet layers use plain NormPre2 — NOT a straight composition,
the reason this family needed Architecture.NormPlacementLinear), whole-vector QK-norm on the
full-attention layers, and linear_allow_neg_eigval on the DeltaNet layers. The release ships
rope_parameters: {"rope_theta": null} — NO RoPE anywhere, on any layer (unlike olmo3, which
carries YaRN on its full-attention layers; do not assume the two families' RoPE handling is the
same failure surface).

Dumps the last-token logits + argmax + a short greedy continuation (token IDs) for a fixed
prompt; the goinfer side (olmo_hybrid_real_test.go, build tag realckpt) loads the same
safetensors at f32 and matches argmax + continuation + cosine >= 0.9999.

    ~/.venv-vl/bin/python scripts/pin_olmo_hybrid_real.py
    -> testdata/olmo_hybrid_real_golden.json   (committed; the ~14 GB weights are NOT)

Put the checkpoint at ~/models/olmo-hybrid-7b (allenai/Olmo-Hybrid-7B), or set
GOINFER_OLMO_HYBRID_7B to its path.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ.get("GOINFER_OLMO_HYBRID_7B", os.path.expanduser("~/models/olmo-hybrid-7b"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "olmo_hybrid_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32).eval()
    cfg = model.config
    arch = cfg.architectures[0] if cfg.architectures else "?"
    layer_types = getattr(cfg, "layer_types", None)
    n_linear = sum(1 for t in (layer_types or []) if t == "linear_attention")
    n_full = sum(1 for t in (layer_types or []) if t == "full_attention")
    print(f"loaded {arch} model_type={cfg.model_type} num_hidden_layers={cfg.num_hidden_layers} "
          f"layers: {n_linear} linear / {n_full} full "
          f"rope_theta={getattr(cfg,'rope_theta',None)} linear_allow_neg_eigval={getattr(cfg,'linear_allow_neg_eigval',None)}")

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

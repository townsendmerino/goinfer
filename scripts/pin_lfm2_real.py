#!/usr/bin/env python3
"""Real-model parity golden for LFM2.5-2.6B (model_type "lfm2", Lfm2ForCausalLM) — the T3
promotion of goinfer's LFM2 family from tiny-golden to a released checkpoint
(docs/parity-coverage-policy.md). LFM2.5-2.6B is the smallest released LFM2.5 checkpoint and
the same one the 2026-08-31 real-checkpoint debugging session already diffed against by hand
(decoder/lfm2_test.go's own comment: that pass found NormEps read from the wrong config key
and a zeroed AttnScale, both fixed) — this script/gate is the permanent, re-runnable version
of that ad-hoc diff, not a new investigation.

Exercises the hybrid layer_types split on real weights: LFM2.5-2.6B is 30 layers, 22 short-conv
+ 8 full-attention (docs/scoping-lfm2.md).

Dumps the last-token logits + argmax + a short greedy continuation (token IDs) for a fixed
prompt; the goinfer side (lfm2_real_test.go, build tag realckpt) loads the same safetensors at
f32 and matches argmax + continuation + cosine >= 0.9999.

    ~/.venv-vl/bin/python scripts/pin_lfm2_real.py
    -> testdata/lfm2_real_golden.json   (committed; the ~5.1 GB weights are NOT)

Put the checkpoint at ~/models/lfm25-2.6b (LiquidAI/LFM2.5-2.6B), or set GOINFER_LFM2_2_6B to
its path.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ.get("GOINFER_LFM2_2_6B", os.path.expanduser("~/models/lfm25-2.6b"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "lfm2_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32).eval()
    cfg = model.config
    arch = cfg.architectures[0] if cfg.architectures else "?"
    layer_types = getattr(cfg, "layer_types", None)
    n_conv = sum(1 for t in (layer_types or []) if t == "conv")
    n_attn = sum(1 for t in (layer_types or []) if t == "full_attention")
    print(f"loaded {arch} model_type={cfg.model_type} num_hidden_layers={cfg.num_hidden_layers} "
          f"layers: {n_conv} conv / {n_attn} full_attention")

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

#!/usr/bin/env python3
"""Pin a tiny-random Cohere2 / Command-R7B (model_type "cohere2") forward + greedy
decode as a goinfer parity golden — the independent HF oracle for the cohere2 family
(docs/task-model-family-cohere.md, Phase 2).

Cohere2 is cohere1's stack (bias-free LayerNorm, parallel attn+MLP block, gated SiLU,
tied embeddings, GPT-J interleaved RoPE, logit_scale) PLUS the two things this gate
must pin:
  * interleaved SLIDING-WINDOW / full attention (every sliding_window_pattern-th
    layer is global/full, the rest window at sliding_window);
  * NoPE on the GLOBAL layers — only the sliding layers carry RoPE.
And NO QK-norm (cohere2 dropped it).

Config exercises all of it: 4 layers, pattern 4 → layers 0-2 sliding(+RoPE), layer 3
global(+NoPE). The prompt is LONGER than the window (window=8, 24 tokens) so the
sliding layers actually drop older tokens — a wrong window boundary moves the golden.
Positions span 0-23 so interleaved-vs-NeoX RoPE and RoPE-vs-NoPE both bite.

Degeneracy guard (as pin_cohere_tiny.py): reseed every LayerNorm weight to non-trivial
values via a separate torch.Generator so a dropped norm-weight (×1) can't hide.

Run in a transformers venv:
    python3 scripts/pin_cohere2_tiny.py
    -> testdata/cohere2_tiny_golden.json  (+ testdata/cohere2-tiny/ checkpoint)
"""
import json
import os

import torch
from transformers import Cohere2Config
from transformers.models.cohere2.modeling_cohere2 import Cohere2ForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "cohere2_tiny_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "cohere2-tiny")

CFG = dict(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,          # GQA
    intermediate_size=128,
    layer_norm_eps=1e-5,
    rope_theta=10000.0,
    logit_scale=0.25,               # ≠ 1: pins the reciprocal-divide
    hidden_act="silu",
    tie_word_embeddings=True,
    max_position_embeddings=128,
    attention_bias=False,
    sliding_window=8,               # < prompt length so windowing is exercised
    # transformers ≥5 drops sliding_window_pattern for explicit layer_types; this is
    # what a pattern-4 group looks like: 3 sliding(+RoPE) then 1 full/global(+NoPE).
    # goinfer's Config.IsGlobalLayer reads layer_types authoritatively.
    layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
)

# 24 tokens → positions 0-23. Longer than the window (8) and long enough that
# interleaved RoPE (sliding layers) and NoPE (global layer) both move the logits.
PROMPT_IDS = [
    3, 141, 7, 88, 219, 14, 66, 190, 42, 5, 133, 201,
    17, 250, 99, 60, 31, 178, 4, 122, 233, 9, 71, 164,
]
N_NEW = 6


def main():
    torch.manual_seed(0)
    cfg = Cohere2Config(**CFG)
    model = Cohere2ForCausalLM(cfg).eval().to(torch.float32)

    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                p.copy_(1.0 + 0.5 * torch.randn(p.shape, generator=g))

    ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
    with torch.no_grad():
        last = model(ids).logits[0, -1].to(torch.float64)
        argmax = int(torch.argmax(last).item())
        cont, cur = [], ids
        for _ in range(N_NEW):
            nxt = int(torch.argmax(model(cur).logits[0, -1]).item())
            cont.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    # Record the resolved layer classification so a goinfer/HF interleave mismatch
    # is visible in the golden itself, not just as a logit diff.
    layer_types = getattr(cfg, "layer_types", None)

    golden = dict(
        prompt="",
        prompt_ids=PROMPT_IDS,
        argmax=argmax,
        vocab_size=cfg.vocab_size,
        sliding_window=cfg.sliding_window,
        layer_types=layer_types,
        last_logits=[float(x) for x in last.tolist()],
        n_new=N_NEW,
        continuation_ids=cont,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f, indent=1)
    print("wrote", os.path.relpath(OUT), "argmax", argmax, "cont", cont, "layer_types", layer_types)

    model.save_pretrained(CKPT, safe_serialization=True)
    print("wrote", os.path.relpath(CKPT))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pin a tiny-random Cohere / Command-R (model_type "cohere") forward + greedy
decode as a goinfer parity golden — the independent HF oracle for the cohere1
family (docs/task-model-family-cohere.md, Phase 1).

Builds a SMALL random CohereForCausalLM and dumps:
  - testdata/cohere-tiny/         a safetensors checkpoint + config.json goinfer loads
  - testdata/cohere_tiny_golden.json  input ids + last-token logits + greedy continuation

What this gate PINS (the two new primitives + the borrowed ones):
  * bias-free LayerNorm (mean-subtract) — NOT RMSNorm;
  * the PARALLEL block: one shared input_layernorm feeds attn AND mlp, summed
    into one residual add;
  * GPT-J interleaved RoPE (rope_gptj), not NeoX rotate_half;
  * logit_scale MULTIPLIER on the output logits (goinfer stores its reciprocal);
  * tied 256k-style embeddings, GQA, gated SiLU MLP.

Degeneracy guard (same lesson as the gemma4-moe / mamba goldens): HF's default
init leaves every LayerNorm weight at 1.0, so a bug that DROPS the norm weight
(×1) would not move the golden. We reseed every norm weight (input_layernorm +
final model.norm) to NON-TRIVIAL values via a SEPARATE torch.Generator — the
global RNG that drew the linear weights + input ids is untouched, so the HF
forward stays the oracle while the norm-weight application is genuinely pinned.
logit_scale is set to 0.125 (≠1) so the reciprocal-divide is exercised too.

Run in a transformers venv (any python with torch+transformers+safetensors):
    python3 scripts/pin_cohere_tiny.py
    -> testdata/cohere_tiny_golden.json  (+ testdata/cohere-tiny/ checkpoint)
"""
import json
import os

import torch
from transformers import CohereConfig
from transformers.models.cohere.modeling_cohere import CohereForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "cohere_tiny_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "cohere-tiny")

# Tiny dims that still exercise GQA (kv<heads), the parallel block, interleaved
# RoPE over the full head_dim, and the logit_scale multiply. head_dim*heads == hidden.
CFG = dict(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=4,            # 4 layers so a rope-style error compounds
    num_attention_heads=4,
    num_key_value_heads=2,          # GQA
    intermediate_size=128,
    layer_norm_eps=1e-5,
    rope_theta=10000.0,
    logit_scale=0.125,              # ≠ 1: pins goinfer's reciprocal-divide
    hidden_act="silu",
    tie_word_embeddings=True,
    use_qk_norm=False,              # cohere1 Phase-1 scope (Aya/Command-R)
    max_position_embeddings=128,
    attention_bias=False,
)

# A LONG, varied prompt on purpose: interleaved (GPT-J) vs NeoX rope only diverge
# meaningfully once positions are large enough for the rotation angles to differ,
# so a short prompt (positions 0-7) leaves cosine ~0.99999 and the gate can't tell
# the two apart (break-it-first caught exactly this). 40 tokens → positions 0-39.
PROMPT_IDS = [
    3, 141, 7, 88, 219, 14, 66, 190, 42, 5, 133, 201, 17, 250, 99, 60,
    31, 178, 4, 122, 233, 9, 71, 164, 28, 199, 12, 84, 245, 53, 108, 2,
    147, 36, 211, 77, 19, 156, 91, 240,
]
N_NEW = 6                                        # greedy continuation length


def main():
    torch.manual_seed(0)
    cfg = CohereConfig(**CFG)
    model = CohereForCausalLM(cfg).eval().to(torch.float32)

    # --- degeneracy guard: reseed every LayerNorm weight to non-trivial values ---
    g = torch.Generator().manual_seed(1234)  # separate stream; global RNG untouched
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                # centered around 1.0 so the norm still behaves, but ≠ identity
                p.copy_(1.0 + 0.5 * torch.randn(p.shape, generator=g))

    ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
    with torch.no_grad():
        out = model(ids)
        last = out.logits[0, -1].to(torch.float64)  # already ×logit_scale in HF
        argmax = int(torch.argmax(last).item())

        # greedy continuation (argmax), pinned so a subtle positional/rope bug that
        # still gets token-0 right is caught downstream.
        cont = []
        cur = ids
        for _ in range(N_NEW):
            lg = model(cur).logits[0, -1]
            nxt = int(torch.argmax(lg).item())
            cont.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    golden = dict(
        prompt="",  # ids pinned directly (no tokenizer dependency for a random model)
        prompt_ids=PROMPT_IDS,
        argmax=argmax,
        vocab_size=cfg.vocab_size,
        last_logits=[float(x) for x in last.tolist()],
        n_new=N_NEW,
        continuation_ids=cont,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f, indent=1)
    print("wrote", os.path.relpath(OUT), "argmax", argmax, "cont", cont)

    # Save the checkpoint goinfer loads (safetensors + config.json).
    model.save_pretrained(CKPT, safe_serialization=True)
    print("wrote", os.path.relpath(CKPT))


if __name__ == "__main__":
    main()

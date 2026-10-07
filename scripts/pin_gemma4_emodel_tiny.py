#!/usr/bin/env python3
"""Pin a tiny-random Gemma 4 E-MODEL (the E2B/E4B shape) — S1.1 of
docs/tasks/task-multimodal-support-2026-10.md.

Every other Gemma 4 tiny fixture is PLE-free (hidden_size_per_layer_input=0) and has no
cross-layer KV sharing, because until S1.1 the safetensors loader refused a PLE checkpoint.
This one carries all three E-model features the GPU backends do not implement yet:

  - Per-Layer Embeddings: hidden_size_per_layer_input=32 (the model-level token table,
    projection and norm, plus each layer's input gate / projection / post norm),
  - cross-layer KV sharing: num_kv_shared_layers=2 over layer_types [s, s, f, s, s, f], so
    shared layer 4 (sliding) reads layer 3 and shared layer 5 (full) reads layer 2 — the full
    layer's source is NOT the layer right before it,
  - use_double_wide_mlp: the two shared layers' FFNs are twice intermediate_size,

plus the E2B attention shape: one KV head, two head_dims (local 32 / global 64),
attention_k_eq_v=False (so v_norm runs on every K/V-owning layer), final softcap 30, tied
embeddings. The prompt (12) is longer than the sliding window (4).

Degeneracy guard, as in pin_gemma4_dense_twogeom.py: HF init leaves every norm and
layer_scalar at identity, so a bug that skips one would not move the golden. strengthen()
overrides them from a SEPARATE seeded generator (the linear weights and inputs stay
bit-identical). The PLE norms are included; v_norm has no weight.

The golden holds the logits at EVERY prompt position (teacher-forced, one HF forward) and a
greedy continuation. CPU fp32; HF is the oracle.

    python3 scripts/pin_gemma4_emodel_tiny.py
    -> testdata/gemma4_emodel_tiny_golden.json  (+ testdata/gemma4-emodel-tiny/)
"""
import json
import os

import torch
import transformers
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "gemma4_emodel_tiny_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-emodel-tiny")

LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention",
               "sliding_attention", "sliding_attention", "full_attention"]
CFG = dict(
    # hidden 256 for the same reason as the two-geometry fixture: the GPU gates that will reuse
    # this checkpoint quantize activations to int8, which is degenerate at hidden=64.
    vocab_size=256, hidden_size=256, num_hidden_layers=len(LAYER_TYPES),
    num_attention_heads=4, num_key_value_heads=1, head_dim=32,
    intermediate_size=256, rms_norm_eps=1e-6, tie_word_embeddings=True,
    max_position_embeddings=128, sliding_window=4,
    layer_types=LAYER_TYPES,
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
    enable_moe_block=False,
    global_head_dim=64, num_global_key_value_heads=1, attention_k_eq_v=False,
    # The E-model features.
    hidden_size_per_layer_input=32, vocab_size_per_layer_input=256,
    num_kv_shared_layers=2, use_double_wide_mlp=True,
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88, 9, 151, 64, 30]
N_NEW = 6


def strengthen(model):
    """Override identity scaling params with seeded non-trivial values (separate generator)."""
    g = torch.Generator().manual_seed(1234)
    n_norm = n_ls = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.normal_(1.0, 0.1, generator=g)
                n_norm += 1
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.normal_(1.0, 0.1, generator=g)
                n_ls += 1
    return dict(norm=n_norm, layer_scalar=n_ls)


def main():
    torch.manual_seed(0)
    config = Gemma4TextConfig(**CFG)
    model = Gemma4ForCausalLM(config).eval().to(torch.float32)
    counts = strengthen(model)

    # The fixture must actually have the shapes it claims; a config key HF ignores would
    # silently pin a plain model.
    lm = model.model
    widths = [layer.mlp.gate_proj.weight.shape[0] for layer in lm.layers]
    shared = [layer.self_attn.is_kv_shared_layer for layer in lm.layers]
    assert widths == [256, 256, 256, 256, 512, 512], widths
    assert shared == [False] * 4 + [True] * 2, shared
    assert lm.embed_tokens_per_layer.weight.shape == (256, 6 * 32)

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        all_logits = model(ids, use_cache=False).logits[0].float()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            nxt = int(model(torch.tensor([cur], dtype=torch.long), use_cache=False).logits[0, -1].argmax())
            cont.append(nxt)
            cur.append(nxt)

    golden = {
        "note": "tiny-random gemma4 E-model: PLE (P=32), KV sharing (last 2 of 6 layers), "
                "double-wide shared FFNs, two head_dims, k_eq_v off; CPU fp32. Norms and "
                "layer_scalar strengthened from a separate seeded RNG; HF forward is the oracle.",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "strengthened": counts,
        "ffn_widths": widths,
        "prompt_ids": PROMPT,
        "argmax": [int(r.argmax()) for r in all_logits],
        "logits": [r.tolist() for r in all_logits],  # [len(PROMPT), vocab]
        "n_new": N_NEW,
        "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}")
    print(f"  strengthened={counts} widths={widths} argmax={golden['argmax']} continuation={cont}")

    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}")


if __name__ == "__main__":
    main()

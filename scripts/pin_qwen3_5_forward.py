#!/usr/bin/env python
"""Tiny-random Qwen3.8 (model_type qwen3_5) checkpoint + text golden — the dense sibling of
pin_qwen35_forward.py.

Pinned oracle: transformers 5.12.0 (the first release carrying models/qwen3_5). The parity
targets are the torch fallbacks (torch_recurrent_gated_delta_rule / torch_chunk_...), which is
what runs when the `fla` kernels are absent — as they are here.

The fixture keeps the released model's SHAPE CHARACTER, not its size:
  * head_dim (32) is NOT hidden/num_heads — the real model is head_dim 256 at hidden 5120 with
    24 heads, so nH·hd != hidden. A fixture that let them coincide would hide exactly the
    class of bug this family is prone to.
  * 3:1 layer_types (linear, linear, linear, full) so BOTH mixer kinds run.
  * GVA: linear_num_value_heads a multiple (>1x) of linear_num_key_heads.
  * partial_rotary_factor 0.25, mrope_section + mrope_interleaved present — the text path must
    reduce to plain partial RoPE despite them.

    ~/.venv-vl/bin/python scripts/pin_qwen3_5_forward.py
    -> decoder/testdata/qwen3_5_tiny_text_golden.json + decoder/testdata/qwen3_5-tiny/
"""
import json
import os
import sys

import torch
from transformers import (Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM,
                          Qwen3_5MoeTextConfig, Qwen3_5TextConfig)

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "decoder", "testdata")
OUT = os.path.join(TD, "qwen3_5_tiny_text_golden.json")
CKPT = os.path.join(TD, "qwen3_5-tiny")

# --moe writes a SECOND fixture, the MoE sibling — Qwen3_5MoeForCausalLM, its own model_type
# (goinfer's dense adapter rejects a num_experts config outright, and correctly so). It exists for one reason: the WebGPU resident
# bridge composes a DeltaNet mixer with a sparse-MoE FFN in the same layer, and nothing gated
# that pairing. The mixer alone is gated by the dense fixture, and mixer+MoE is gated for
# Mamba-2 (Granite) — but "two proven halves" is the argument that has been wrong here before
# (the A' zero-copy post-mortem: isolation proves the primitive, never the composition).
MOE_OUT = os.path.join(TD, "qwen3_5_moe_tiny_text_golden.json")
MOE_CKPT = os.path.join(TD, "qwen3_5_moe-tiny")
# --scaled writes a THIRD fixture: 4 layers at FULLY REAL width. Not a correctness fixture — the
# tiny ones gate that — but the only way to get a defensible tok/s number for a 27B model that
# does not fit 8 GB of VRAM. Everything the decode cost depends on is the released value
# (hidden 5120, intermediate 17408, head_dim 256, 24/4 heads, DeltaNet 128/128 with 16/48 heads,
# partial rotary 0.25) and the layer mix is the real 3:1, so ms/LAYER extrapolates. Only the
# layer count and the vocab shrink; the released vocab is 248320, whose LM head is ~5% of the
# real model's per-token MACs and must be added back by hand, not assumed away.
SCALED_CKPT = os.path.join(os.path.expanduser("~"), "models", "qwen3_5-scaled4")
# The note lives WITH the checkpoint, not in testdata/: every JSON in testdata/ is a parity
# golden, and a timing artifact filed among them would be read as one.
SCALED_OUT = os.path.join(SCALED_CKPT, "goinfer_timing_note.json")
SCALED_CFG = dict(
    vocab_size=4096, hidden_size=5120, intermediate_size=17408, num_hidden_layers=4,
    num_attention_heads=24, num_key_value_heads=4, head_dim=256,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_value_head_dim=128,
    linear_num_key_heads=16, linear_num_value_heads=48,
    rms_norm_eps=1e-6, max_position_embeddings=4096, tie_word_embeddings=False,
    hidden_act="silu", attention_bias=False, attn_output_gate=True,
    rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                     "partial_rotary_factor": 0.25,
                     "mrope_section": [11, 11, 10], "mrope_interleaved": True},
)

MOE_CFG = dict(
    num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
    shared_expert_intermediate_size=64, norm_topk_prob=True, decoder_sparse_step=1,
    mlp_only_layers=[],
)

CFG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,   # nH*hd = 128 != hidden 64
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,            # GVA rep = 2
    rms_norm_eps=1e-6, max_position_embeddings=512, tie_word_embeddings=False,
    hidden_act="silu", attention_bias=False, attn_output_gate=True,
    rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                     "partial_rotary_factor": 0.25,
                     "mrope_section": [3, 3, 2], "mrope_interleaved": True},
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6


def main():
    if "--scaled" in sys.argv:
        # No golden: this fixture exists for TIMING, and running an HF forward at this width
        # would cost minutes to produce a number nothing reads. Correctness is the tiny
        # fixtures' job; conflating the two would make this look like a parity artifact.
        torch.manual_seed(0)
        m = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**SCALED_CFG)).eval().to(torch.bfloat16)
        m.save_pretrained(SCALED_CKPT, safe_serialization=True)
        n = sum(p.numel() for p in m.parameters())
        json.dump({"note": "TIMING fixture, NOT a parity golden — 4 layers at real Qwen3.8 width",
                   "config": SCALED_CFG, "params": n}, open(SCALED_OUT, "w"), indent=1)
        print(f"{n/1e9:.2f}B params -> {SCALED_CKPT}")
        return
    moe = "--moe" in sys.argv
    cfg = dict(CFG, **MOE_CFG) if moe else CFG
    out, ckpt = (MOE_OUT, MOE_CKPT) if moe else (OUT, CKPT)
    torch.manual_seed(0)
    if moe:
        model = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**cfg))
        # AMPLIFY the shared-expert gate, deliberately. Default init leaves shared_expert_gate·h
        # small, so sigmoid(gl) sits near 0.5 — and at 0.5 the GATED and UNGATED combines differ
        # by a factor of two on a contribution that is already small next to the routed experts
        # and the residual. Measured: a resident backend wired to combine UNGATED still scores
        # cosine 0.983 against this fixture, comfortably inside any sane floor. A fixture that
        # cannot distinguish the feature it is the only coverage of is not coverage.
        #
        # x20 drives |gl| large enough that sigmoid saturates per token, so the gated and ungated
        # paths differ unmistakably. Nothing else about the model changes, and the golden below is
        # recomputed from THIS model, so HF remains the reference either way.
        with torch.no_grad():
            for layer in model.model.layers:
                blk = getattr(layer, "mlp", None)
                if hasattr(blk, "shared_expert_gate"):
                    blk.shared_expert_gate.weight.mul_(20.0)
    else:
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**cfg))
    model = model.eval().to(torch.float32)
    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last = model(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    json.dump({
        "note": "tiny-random Qwen3_5ForCausalLM (dense DeltaNet/softmax hybrid), CPU fp32, "
                "transformers 5.12.0 torch fallback kernels",
        "config": {k: v for k, v in cfg.items()},
        "prompt_ids": PROMPT, "n_new": N_NEW,
        "argmax": int(torch.tensor(last).argmax()), "last_logits": last,
        "continuation_ids": cont,
    }, open(out, "w"))
    model.save_pretrained(ckpt, safe_serialization=True)
    print(f"argmax={int(torch.tensor(last).argmax())} cont={cont}")
    print(f" -> {out}\n -> {ckpt}")


if __name__ == "__main__":
    main()

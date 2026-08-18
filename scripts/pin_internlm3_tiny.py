#!/usr/bin/env python
"""Tiny-random InternLM3 checkpoint + text golden.

WHY THIS EXISTS WHEN internlm3 IS A LLAMA ALIAS. The alias rides llamaArchitecture and
llama's tensor schema, both already covered by llama's own oracle — so the SHAPE needs no
new gate. What llama's oracle does NOT exercise is the one config departure that makes
internlm3 its own model_type: `rope_scaling: {"rope_type": "dynamic", ...}`, which goinfer
accepts as in-window identity. That claim needs a reference, not an argument.

So the fixture is a real LlamaForCausalLM (HF implements dynamic NTK natively, so the
reference genuinely applies it) written out with model_type "internlm3". If goinfer's
"dynamic is identity below max_position_embeddings" reading were wrong, this diverges.

    ~/.venv-vl/bin/python scripts/pin_internlm3_tiny.py
    -> decoder/testdata/internlm3_tiny_text_golden.json + decoder/testdata/internlm3-tiny/
"""
import json
import os

import torch
from transformers import LlamaConfig, LlamaForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "decoder", "testdata")
OUT = os.path.join(TD, "internlm3_tiny_text_golden.json")
CKPT = os.path.join(TD, "internlm3-tiny")

CFG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=128, rms_norm_eps=1e-5, rope_theta=50000000.0,
    attention_bias=False, mlp_bias=False, tie_word_embeddings=False, hidden_act="silu",
    # THE POINT OF THE FIXTURE: dynamic NTK. HF rescales only past max_position_embeddings,
    # so at these lengths it is identity — which is exactly the claim under test.
    rope_scaling={"rope_type": "dynamic", "factor": 6.0},
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6


def main():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(**CFG)).eval().to(torch.float32)
    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last = model(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])
    golden = {
        "note": "tiny-random LlamaForCausalLM with rope_scaling dynamic, saved as model_type "
                "internlm3; CPU fp32. Gates the dynamic-NTK-is-in-window-identity reading.",
        "config": CFG, "prompt_ids": PROMPT, "n_new": N_NEW,
        "argmax": int(torch.tensor(last).argmax()), "last_logits": last, "continuation_ids": cont,
    }
    os.makedirs(TD, exist_ok=True)
    json.dump(golden, open(OUT, "w"))
    model.save_pretrained(CKPT, safe_serialization=True)
    # Rewrite model_type so goinfer resolves it through the internlm3 registration.
    cfg_path = os.path.join(CKPT, "config.json")
    c = json.load(open(cfg_path))
    c["model_type"] = "internlm3"
    c["architectures"] = ["InternLM3ForCausalLM"]
    json.dump(c, open(cfg_path, "w"), indent=1)
    print(f"argmax={golden['argmax']} cont={cont}\n -> {OUT}\n -> {CKPT} (model_type=internlm3)")


if __name__ == "__main__":
    main()

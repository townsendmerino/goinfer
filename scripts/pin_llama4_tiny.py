#!/usr/bin/env python
"""Tiny-random Llama 4 text decoder (Llama4ForCausalLM) checkpoint + text golden — the
Llama 4 parity fixture. Llama 4 is the iRoPE decoder: a per-layer NoPE interleave
(no_rope_layers: every 4th layer skips RoPE), parameter-free L2 (RMS-over-head-dim) QK-norm
on the RoPE layers, attention-temperature tuning on the NoPE layers
(q *= log1p(floor((pos+1)/floor_scale))*attn_scale + 1), interleaved (complex) RoPE, GQA,
and a dense/MoE interleave (moe_layers) where MoE = top-1 SIGMOID routing + an always-on
ungated shared expert (experts are fused gate_up + down, batched [nE, in, out]).

This tiny config exercises all of it across 4 layers: L0 dense+RoPE, L1 MoE+RoPE,
L2 dense+NoPE (attn-temp fires via small floor_scale=4), L3 MoE+RoPE. QK-norm on the RoPE
layers only.

    ~/.venv-vl/bin/python scripts/pin_llama4_tiny.py
    -> testdata/llama4_tiny_text_golden.json + testdata/llama4-tiny/
"""
import json, os, torch
from transformers import Llama4TextConfig, Llama4ForCausalLM
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "llama4_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "llama4-tiny")
CFG = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, intermediate_size_mlp=128,
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4, head_dim=8,
    num_local_experts=4, num_experts_per_tok=1, interleave_moe_layer_step=2,
    no_rope_layers=[1, 1, 0, 1], use_qk_norm=True, attn_temperature_tuning=True,
    floor_scale=4, attn_scale=0.1, rope_theta=10000.0, rms_norm_eps=1e-5,
    max_position_embeddings=64, tie_word_embeddings=False, pad_token_id=0,
    bos_token_id=1, eos_token_id=2,
)
PROMPT = [1, 7, 42, 100, 5, 88, 13, 19]
N_NEW = 6


def main():
    torch.manual_seed(0)
    c = Llama4TextConfig(**CFG)
    print("moe_layers =", c.moe_layers, "| no_rope_layers =", c.no_rope_layers)
    m = Llama4ForCausalLM(c).eval().to(torch.float32)
    with torch.no_grad():
        last = m(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax())); cur.append(cont[-1])
    g = dict(note="tiny Llama4 text decoder fwd fp32", config=CFG, prompt_ids=PROMPT,
             argmax=int(torch.tensor(last).argmax()), last_logits=last,
             n_new=N_NEW, continuation_ids=cont)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont}")
    m.save_pretrained(CKPT, safe_serialization=True); print("saved", CKPT)


if __name__ == "__main__":
    main()

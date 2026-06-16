#!/usr/bin/env python
"""Tiny-random Phi-3 (Phi3ForCausalLM) checkpoint + text golden — the Phi-3/Phi-4 parity
fixture. Phi-3 is the llama skeleton (RMSNorm no-offset, Pre2, SwiGLU, NeoX RoPE, no QK-norm,
no bias, untied head) with TWO fused tensors that goinfer's loader splits at load:
self_attn.qkv_proj (q ‖ k ‖ v, split at num_heads*head_dim then +kv) and mlp.gate_up_proj
(gate ‖ up, chunked in half). Full rotary, no rope scaling (the 4k / Phi-4 path; LongRoPE
on the 128k variants is deferred). GQA (kv<heads) exercised.

    ~/.venv-vl/bin/python scripts/pin_phi3_tiny.py
    -> testdata/phi3_tiny_text_golden.json + testdata/phi3-tiny/
"""
import json, os, torch
from transformers import Phi3Config, Phi3ForCausalLM
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "phi3_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "phi3-tiny")
CFG = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
           num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
           rope_theta=10000.0, rms_norm_eps=1e-5, tie_word_embeddings=False,
           pad_token_id=0, eos_token_id=1, bos_token_id=2)
PROMPT = [2, 7, 42, 100, 5, 88, 13, 19]
N_NEW = 6


def main():
    torch.manual_seed(0)
    c = Phi3Config(**CFG)
    print("=== config of interest ===")
    cd = c.to_dict()
    for k in ['partial_rotary_factor', 'rope_scaling', 'attention_bias', 'hidden_act', 'head_dim']:
        print(f"  {k} = {cd.get(k, '<absent>')}")
    m = Phi3ForCausalLM(c).eval().to(torch.float32)
    with torch.no_grad():
        last = m(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax())); cur.append(cont[-1])
    g = dict(note="tiny Phi-3 text fwd fp32", config=CFG, prompt_ids=PROMPT,
             argmax=int(torch.tensor(last).argmax()), last_logits=last,
             n_new=N_NEW, continuation_ids=cont)
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont}")
    m.save_pretrained(CKPT, safe_serialization=True); print("saved", CKPT)


if __name__ == "__main__":
    main()

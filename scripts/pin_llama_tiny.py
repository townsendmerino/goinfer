#!/usr/bin/env python
"""Build a tiny-random Llama (LlamaForCausalLM) checkpoint + text golden — the plain `llama`
arch fixture (mirrors pin_lfm2_tiny / pin_mixtral_tiny).

WHY IT DID NOT EXIST, AND WHY THAT MATTERED. `llama` is a REQUIRED parity gate
(parityGates: {"llama", "TestLlama_forwardParity"}) and the single most common architecture in
the ecosystem, and until 2026-09-02 there was NO tiny fixture for it anywhere in the tree — only
goldens. The only llama checkpoints on any machine here are the Linux box's gitignored
llama3.2-1b (2.4 GB), tinyllama-awq (731 MB) and tinyllama-gptq (733 MB), so a .giw round-trip of
the arch could not be censused on any other machine, and could not be censused CHEAPLY on that
one. TestSerializeCensus_noSilentFieldDrop therefore round-tripped 21 families without ever
touching llama. Found by the census's completeness gate reporting the box's untracked fixtures
(audit-2026-09-02 C-03 follow-on).

    ~/.venv-vl/bin/python scripts/pin_llama_tiny.py
    -> testdata/llama_tiny_text_golden.json
    -> testdata/llama-tiny/   (HF safetensors checkpoint)

Deliberately plain: GQA, SwiGLU, RMSNorm, untied embeddings, rope_theta at Llama-3's 500000.0
rather than a default 10000.0, so a loader that hardcodes theta fails parity. Nothing exotic —
the point is the BASELINE arch, which every other family's descriptor is a deviation from.
"""
import json
import os

import torch
from transformers import LlamaConfig, LlamaForCausalLM

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "llama_tiny_text_golden.json")
CKPT = os.path.join(HERE, "..", "testdata", "llama-tiny")

CFG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2,  # GQA 2:1; head_dim = 64/4 = 16
    max_position_embeddings=128, rms_norm_eps=1e-5,
    # Llama-3's rope base, not the 1e4 default: a loader that hardcodes theta fails parity.
    rope_theta=500000.0,
    # UNTIED, unlike the lfm2/gemma fixtures — the llama family ships both, and the untied head is
    # a separate LMHead tensor the serializer must carry.
    tie_word_embeddings=False,
    attention_bias=False, mlp_bias=False,
)
PROMPT = [2, 7, 42, 100, 5, 200, 13, 88]
N_NEW = 6


def main():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(**CFG)).eval().to(torch.float32)
    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last_logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = {
        "note": "tiny-random Llama (GQA + SwiGLU + RMSNorm, untied head), text forward; CPU fp32",
        "config": CFG, "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits, "n_new": N_NEW, "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}\n  argmax={golden['argmax']}  continuation={cont}")
    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}")


if __name__ == "__main__":
    main()

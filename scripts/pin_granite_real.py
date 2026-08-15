#!/usr/bin/env python
"""Real-model parity golden for Granite-4.0-H-Tiny (granitemoehybrid, 7B-A1B) — the bf16
reference behind the family's T3 row. Granite is the repo's FIRST Mamba-2 + attention
hybrid: 40 layers of which 4 are attention and 36 are Mamba-2 (layer_types), MoE on every
layer (64 experts top-6 + an ungated shared_mlp), NoPE on the attention layers, and the
four Granite multipliers (embedding 12, attention 1/128, residual 0.22, logits 6) that a
forward which silently drops one still "runs".

The family's existing real gate (TestGraniteReal_gate) is coherence-only — a distinct-token
floor on a greedy continuation — which is not a T3 method. This produces the numeric oracle
it was missing. Dumps last-token logits + argmax + a short greedy continuation;
granite_real_test.go (build tag realckpt) loads the SAME safetensors at int8 and matches
argmax + continuation + cosine.

    huggingface-cli download ibm-granite/granite-4.0-h-tiny --local-dir ~/models/granite-hf
    ~/.venv-vl/bin/python scripts/pin_granite_real.py
    -> testdata/granite_real_golden.json   (committed; weights are NOT)
"""
import json, os, torch
from transformers import AutoTokenizer, GraniteMoeHybridForCausalLM

CKPT = os.path.expanduser("~/models/granite-hf")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "granite_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 6


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist())
    m = GraniteMoeHybridForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    with torch.no_grad():
        last = m(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur = ids.tolist()[0]
        cont = []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])
    g = dict(note="Granite-4.0-H-Tiny (granitemoehybrid, Mamba-2 + attention hybrid) bf16 reference",
             prompt=PROMPT, prompt_ids=ids[0].tolist(), argmax=int(torch.tensor(last).argmax()),
             last_logits=last, n_new=N_NEW, continuation_ids=cont,
             continuation_text=tok.decode(cont))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont!r} -> {g['continuation_text']!r}")
    print("saved", OUT)


if __name__ == "__main__":
    main()

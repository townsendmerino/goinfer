#!/usr/bin/env python
"""Real-model parity golden for Phi-3-mini-4k-instruct (phi3, 3.8B dense) — the full f32
reference for goinfer's Phi-3/Phi-4 loader on actual released weights. 3.8B fits an f32
forward in RAM, so this is a TIGHT cosine gate (not the int8-vs-bf16 of the bigger
families). Exercises the fused qkv_proj / gate_up_proj split, full rotary (no LongRoPE on
the 4k variant), MHA (kv=heads), untied head. Dumps last-token logits + argmax + greedy
continuation; phi3_real_test.go (build tag realckpt) loads the same safetensors and matches.

    ~/.venv-vl/bin/python scripts/pin_phi3_mini.py
    -> testdata/phi3_mini_golden.json   (committed; weights are NOT)
"""
import json, os, torch
from transformers import Phi3Config, Phi3ForCausalLM, AutoTokenizer

CKPT = os.path.expanduser("~/models/phi3-mini-4k")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "phi3_mini_golden.json")
PROMPT = "The capital of France is"
N_NEW = 6


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist())
    cfg = Phi3Config.from_pretrained(CKPT)
    m = Phi3ForCausalLM.from_pretrained(CKPT, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    with torch.no_grad():
        last = m(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur = ids.tolist()[0]
        cont = []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax())); cur.append(cont[-1])
    g = dict(note="Phi-3-mini-4k-instruct f32 reference", prompt=PROMPT,
             prompt_ids=ids[0].tolist(), argmax=int(torch.tensor(last).argmax()),
             last_logits=last, n_new=N_NEW, continuation_ids=cont,
             continuation_text=tok.decode(cont))
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont!r} -> {g['continuation_text']!r}")
    print("saved", OUT)


if __name__ == "__main__":
    main()

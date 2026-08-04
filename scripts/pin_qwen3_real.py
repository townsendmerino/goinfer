#!/usr/bin/env python
"""Real-model parity golden for Qwen3-1.7B (qwen3, dense) — the full f32 reference for
goinfer's qwen3 loader on actual released weights. 1.7B fits an f32 forward in RAM, so this
is a TIGHT cosine gate (bf16 weights upcast to f32 on both sides). Exercises the qwen3 axis:
per-head QK-RMSNorm before RoPE, GQA, full rotary, untied vs tied head per config. Dumps
last-token logits + argmax + greedy continuation; qwen3_real_test.go (build tag realckpt)
loads the same safetensors and matches.

    ~/.venv-vl/bin/python scripts/pin_qwen3_real.py
    -> testdata/qwen3_real_golden.json   (committed; weights are NOT)
"""
import json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = os.path.expanduser("~/models/qwen3-1.7b-bf16")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "qwen3_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 6


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist())
    m = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    with torch.no_grad():
        last = m(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur = ids.tolist()[0]
        cont = []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax())); cur.append(cont[-1])
    g = dict(note="Qwen3-1.7B f32 reference", prompt=PROMPT,
             prompt_ids=ids[0].tolist(), argmax=int(torch.tensor(last).argmax()),
             last_logits=last, n_new=N_NEW, continuation_ids=cont,
             continuation_text=tok.decode(cont))
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont!r} -> {g['continuation_text']!r}")
    print("saved", OUT)


if __name__ == "__main__":
    main()

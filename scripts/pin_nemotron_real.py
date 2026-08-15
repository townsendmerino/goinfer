#!/usr/bin/env python
"""Real-model parity golden for NVIDIA-Nemotron-Nano-9B-v2 (nemotron_h, 9B) — the bf16
reference behind the family's T3 row. Nemotron-H is the repo's SINGLE-OP-per-block hybrid:
56 blocks that are each exactly one of Mamba-2 / NoPE-attention / relu² MLP
(hybrid_override_pattern), grouped-gated Mamba with n_groups=8, and no MoE at all — the
control case against granite's per-layer mamba+MoE stack.

The family's existing real gate (TestNemotronReal_gate) is coherence-only — a distinct-token
floor on a greedy continuation — which is not a T3 method. This produces the numeric oracle
it was missing. Dumps last-token logits + argmax + a short greedy continuation;
nemotron_real_test.go (build tag realckpt) loads the SAME safetensors at int8 and matches
argmax + continuation + cosine.

    huggingface-cli download nvidia/NVIDIA-Nemotron-Nano-9B-v2 --local-dir ~/models/nemotron-hf
    ~/.venv-vl/bin/python scripts/pin_nemotron_real.py
    -> testdata/nemotron_real_golden.json   (committed; weights are NOT)
"""
import json, os, torch
# Native transformers NemotronH (NOT trust_remote_code — the repo's auto_map points at a
# modeling_nemotron_h.py written against transformers 4.51).
from transformers import AutoTokenizer, NemotronHForCausalLM

CKPT = os.path.expanduser("~/models/nemotron-hf")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "nemotron_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 6


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist())
    m = NemotronHForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    with torch.no_grad():
        last = m(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur = ids.tolist()[0]
        cont = []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])
    g = dict(note="NVIDIA-Nemotron-Nano-9B-v2 (nemotron_h, single-op hybrid) bf16 reference",
             prompt=PROMPT, prompt_ids=ids[0].tolist(), argmax=int(torch.tensor(last).argmax()),
             last_logits=last, n_new=N_NEW, continuation_ids=cont,
             continuation_text=tok.decode(cont))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont!r} -> {g['continuation_text']!r}")
    print("saved", OUT)


if __name__ == "__main__":
    main()

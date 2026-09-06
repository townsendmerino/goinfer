"""Real-model parity golden for Qwen/Qwen3-30B-A3B (qwen3_moe, 30B-A3B) — the family's T3 oracle.

Phase 0/1 and a tiny-scale HF oracle (T1, cosine 0.9999999999999462) landed on the Mac
(docs/task-families-2026-09.md F1); this is the real-scale proof that the adapter holds on
released weights — a tiny fixture cannot catch a wrong tensor name, a transposed expert stack,
or a GGUF fused-expert stride bug, every one of which produces correct shapes and plausible
values.

WHY A PINNED GOLDEN RATHER THAN A LIVE COMPARISON. The bf16 checkpoint is ~61 GB; pinning the
oracle to a file lets goinfer's own int8 load run alone, the same reason every other real-oracle
golden here is a file.

    ~/.venv-vl/bin/python scripts/pin_qwen3moe_real.py
    -> testdata/qwen3moe_real_golden.json   (committed; weights are NOT)
"""

import json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = os.path.expanduser("~/models/qwen3moe-30b-a3b-bf16")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "qwen3moe_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 6


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist())
    m = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    print("config:", type(m).__name__, m.config.model_type,
          "layers", m.config.num_hidden_layers)
    with torch.no_grad():
        last = m(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur = ids.tolist()[0]
        cont = []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])
    g = {
        "prompt_ids": ids[0].tolist(),
        "argmax": int(torch.tensor(last).argmax()),
        "last_logits": last,
        "n_new": N_NEW,
        "continuation_ids": cont,
    }
    with open(OUT, "w") as f:
        json.dump(g, f)
    print("argmax", g["argmax"], tok.decode([g["argmax"]]))
    print("continuation", cont, tok.decode(cont))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

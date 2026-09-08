"""Real-model parity golden for Qwen/Qwen1.5-MoE-A2.7B (qwen2_moe, 14.3B total / 2.7B active) —
the family's T3 oracle. The existing tiny-golden gate (TestQwen2Moe_forwardParity) already uses
a real-but-tiny-random checkpoint (katuni4ka/tiny-random-qwen1.5-moe) for structural coverage;
this is the real-scale proof that the adapter holds on released weights and full expert counts —
a tiny fixture cannot catch a wrong tensor name, a transposed expert stack, or a shared-expert
sigmoid-gate wiring bug that only bites at the real number of experts.

WHY A PINNED GOLDEN RATHER THAN A LIVE COMPARISON, and WHY bf16 NOT f32: the checkpoint is
~28.6 GB bf16; an f32 upcast (params x 4) would need ~57 GB for weights alone on this 62 GB box,
too tight alongside goinfer's own later int8 load and everything else resident. Loaded at bf16
directly (same convention as pin_qwen3moe_real.py) instead.

    ~/.venv-vl/bin/python scripts/pin_qwen2moe_real.py
    -> testdata/qwen2moe_real_golden.json   (committed; the ~28.6 GB weights are NOT)

Put the checkpoint at ~/models/qwen15-moe-a27b (Qwen/Qwen1.5-MoE-A2.7B), or set
GOINFER_QWEN2MOE_HF to its path.
"""

import json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = os.environ.get("GOINFER_QWEN2MOE_HF", os.path.expanduser("~/models/qwen15-moe-a27b"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "qwen2moe_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist())
    m = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    print("config:", type(m).__name__, m.config.model_type,
          "layers", m.config.num_hidden_layers,
          "experts", m.config.num_experts, "shared_expert_intermediate_size",
          getattr(m.config, "shared_expert_intermediate_size", None))
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

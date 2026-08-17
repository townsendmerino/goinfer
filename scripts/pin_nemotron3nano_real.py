"""Real-model parity golden for NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (nemotron_h MoE, 30B-A3B)
— the bf16 reference behind G4's T3 row.

This is the MoE variant of the family already validated at T3 (nemotron_h dense, 9B): the same
single-op-per-block hybrid, but 52 blocks patterned MEMEM*... — 23 Mamba-2, 23 MoE-FFN, 6 NoPE
attention — with a DeepSeek-V3-shaped noaux_tc router (sigmoid + e_score_correction_bias +
group-limited top-k over 128 experts, 6 active) and NON-GATED relu² experts (up/down only, no
gate_proj). Phase 0/1 and a tiny-scale HF oracle (T1, cosine 1.000000) landed on the Mac; this
is the real-scale proof that the adapter holds on released weights.

WHY A PINNED GOLDEN RATHER THAN A LIVE COMPARISON. The bf16 checkpoint is 63 GB and this box has
62 GB of RAM, so the HF reference and goinfer's int8 load cannot coexist. Pinning the oracle to
a file lets each run alone — the same reason every other real-oracle golden here is a file.

    ~/.venv-vl/bin/python scripts/pin_nemotron3nano_real.py
    -> testdata/nemotron3nano_real_golden.json   (committed; weights are NOT)
"""

import json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = os.path.expanduser("~/models/nemotron3nano-30b-bf16")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "nemotron3nano_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 6


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist())
    # AutoModelForCausalLM, not NemotronHForCausalLM by name: the MoE variant registers under
    # the same model_type and transformers dispatches on config. Naming the dense class would
    # silently build the wrong module if the release ever splits them.
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

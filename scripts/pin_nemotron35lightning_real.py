"""Real-model parity golden for NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (nemotron_h MoE).

Phase 0 found this checkpoint's config.json is IDENTICAL to Nemotron 3 Nano's (already T3'd —
docs/completed/queue-correctness.md G4) in every architecturally meaningful field: same
hidden_size/heads/mamba geometry/MoE geometry/routing params, and the SAME 52-block layer pattern
(23 mamba / 23 moe / 6 attention, in the same order — decoded from Nano's hybrid_override_pattern
and diffed position-by-position against Lightning's layers_block_type array). The only differences
are presentational (hybrid_override_pattern string vs an explicit layers_block_type array — both
already handled by decoder/config.go's normalizeNemotronBlocks) and vestigial/inactive fields
(moe_latent_size: null — confirmed against transformers' modeling_nemotron_h.py that this only
activates a bottleneck projection when non-null; moe_shared_expert_overlap — confirmed absent
from modeling_nemotron_h.py entirely, an inference-engine scheduling hint with no HF-reference
effect; num_nextn_predict_layers/mtp_layers_block_type — an MTP head transformers' mainline class
doesn't implement either, dropped the same way every other family's MTP head is).

So this is NOT a new family or a registry change — it is the SAME nemotron_h adapter, same tensor
schema, same tiny golden shape. This script exists to check the one thing Phase 0 cannot: whether
the ACTUALLY TRAINED weights behave the way the architecture predicts (a tiny fixture cannot catch
a wrong tensor name, a transposed expert stack, or a router bias read from the wrong key — every
one of which produces correct shapes and plausible values on different training data).

    ~/.venv-vl/bin/python scripts/pin_nemotron35lightning_real.py
    -> testdata/nemotron35lightning_real_golden.json   (committed; weights are NOT)
"""

import json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = os.path.expanduser("~/models/nemotron35lightning-30b-bf16")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "nemotron35lightning_real_golden.json")
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

#!/usr/bin/env python
"""Real-model parity golden for Moonlight-16B-A3B-Instruct (deepseek_v3, Moonshot) — the
full bf16 reference for goinfer's MLA on the V3 routing flavor: SIGMOID noaux_tc scoring
with a real e_score_correction_bias + routed_scaling_factor 2.446 (V2-Lite is softmax, so
this is the complementary real-weights gate). Also direct-q (q_lora_rank=None), 512-wide
latent, rope_theta 50000, no YaRN. Dumps last-token logits + argmax + a short greedy
continuation; deepseek_real_test.go (build tag realckpt) loads the same safetensors at int8
and matches argmax + continuation + cosine.

    ~/.venv-vl/bin/python scripts/pin_deepseek_moonlight.py
    -> testdata/deepseek_moonlight_golden.json   (committed; weights are NOT)
"""
import json, os, torch
# Native transformers DeepseekV3 (NOT trust_remote_code — the repo's auto_map points at an
# old custom modeling_deepseek.py incompatible with transformers 5.12's rope machinery).
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM, AutoTokenizer

CKPT = os.path.expanduser("~/models/moonlight-16b")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "deepseek_moonlight_golden.json")
PROMPT = "The capital of France is"
N_NEW = 6


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist())
    cfg = DeepseekV3Config.from_pretrained(CKPT)
    m = DeepseekV3ForCausalLM.from_pretrained(
        CKPT, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    with torch.no_grad():
        last = m(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur = ids.tolist()[0]
        cont = []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])
    g = dict(note="Moonlight-16B-A3B (deepseek_v3 MLA) bf16 reference", prompt=PROMPT,
             prompt_ids=ids[0].tolist(), argmax=int(torch.tensor(last).argmax()),
             last_logits=last, n_new=N_NEW, continuation_ids=cont,
             continuation_text=tok.decode(cont))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont!r} -> {g['continuation_text']!r}")
    print("saved", OUT)


if __name__ == "__main__":
    main()

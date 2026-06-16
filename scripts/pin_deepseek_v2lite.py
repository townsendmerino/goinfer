#!/usr/bin/env python
"""Real-model parity golden for DeepSeek-V2-Lite (deepseek_v2, 15.7B-A2.4B MLA) — the
full bf16 reference for goinfer's MLA + DeepSeekMoE on actual released weights. V2-Lite
exercises the direct-q path (q_lora_rank=None, no q-LoRA bottleneck), softmax/greedy
routing (n_group=1, no e_score_correction_bias), the 512-wide latent KV, and live YaRN
(factor 40, mscale==mscale_all_dim ⇒ attention_factor 1.0). Dumps the last-token logits +
argmax + a short greedy continuation (as token IDs) for a fixed prompt; the goinfer side
(deepseek_v2lite_real_test.go, build tag realckpt) loads the same safetensors at int8 and
matches argmax + continuation + cosine.

    ~/.venv-vl/bin/python scripts/pin_deepseek_v2lite.py
    -> testdata/deepseek_v2lite_golden.json   (committed; weights are NOT)
"""
import json, os, torch
# Native transformers DeepseekV2 (NOT trust_remote_code — the repo's auto_map points at an
# old custom modeling_deepseek.py incompatible with transformers 5.12's rope machinery).
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM, AutoTokenizer

CKPT = os.path.expanduser("~/models/deepseek-v2-lite")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "deepseek_v2lite_golden.json")
PROMPT = "The capital of France is"
N_NEW = 6


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist())
    cfg = DeepseekV2Config.from_pretrained(CKPT)
    m = DeepseekV2ForCausalLM.from_pretrained(
        CKPT, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    with torch.no_grad():
        last = m(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur = ids.tolist()[0]
        cont = []
        for _ in range(N_NEW):
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])
    g = dict(note="DeepSeek-V2-Lite (MLA) bf16 reference", prompt=PROMPT,
             prompt_ids=ids[0].tolist(), argmax=int(torch.tensor(last).argmax()),
             last_logits=last, n_new=N_NEW, continuation_ids=cont,
             continuation_text=tok.decode(cont))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont!r} -> {g['continuation_text']!r}")
    print("saved", OUT)


if __name__ == "__main__":
    main()

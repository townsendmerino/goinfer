#!/usr/bin/env python
"""Pin gemma-3-4b-it TEXT-ONLY logit parity as a goinfer parity golden — the P0
gate that the gemma3 descriptor scales 270m → 4B before vision is layered on.

The committed gemma3 gate today is gemma-3-270m (a text-only checkpoint); this
confirms the SAME descriptor reproduces the real 4B text decoder (catching any 4B
rope / sliding-window / config quirk) on a no-image prompt. goinfer loads the 4B
checkpoint's text decoder (ignoring vision_tower / multi_modal_projector, the way
the qwen3.6-VL text side already loads) and must match these logits.

REAL MODEL — not a tiny synthetic. Needs the gemma-3-4b-it checkpoint (~8 GB) on
disk and a single 4B CPU forward (a few minutes); run it when the box is idle (NOT
alongside the 8h fuzz soak — fuzzing pegs all cores). Set GEMMA3_4B to the path.

    GEMMA3_4B=~/models/gemma-3-4b-it ~/g4venv/bin/python scripts/pin_gemma3_4b_text.py
    -> testdata/gemma3_4b_text_golden.json
"""
import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "gemma3_4b_text_golden.json")
PATH = os.environ.get("GEMMA3_4B", os.path.expanduser("~/models/gemma-3-4b-it"))

# Fixed text-only prompt ids (no <image> tokens). Short — this is a decoder-scale
# check, not a generation benchmark.
PROMPT = [2, 651, 6996, 576, 6081, 603, 235248]  # "<bos>The capital of France is " region
N_NEW = 6


def main():
    if not os.path.isdir(PATH):
        raise SystemExit(f"gemma-3-4b checkpoint not found at {PATH} — set GEMMA3_4B "
                         f"(this is the deferred real-model pin; run when the box is idle)")
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(PATH)
    # gemma-3-4b-it is a Gemma3ForConditionalGeneration; take the text decoder so
    # the golden is text-only (the VL wrapper would route through the vision path).
    model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        out = model(ids, use_cache=False)
        last_logits = out.logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(torch.tensor([cur], dtype=torch.long), use_cache=False)
            nxt = int(o.logits[0, -1].argmax())
            cont.append(nxt)
            cur.append(nxt)

    text_cfg = getattr(cfg, "text_config", cfg)
    golden = {
        "note": "gemma-3-4b-it text-only decoder; CPU fp32 (P0 descriptor-scale gate)",
        "model_path": PATH,
        "config": {
            "vocab_size": getattr(text_cfg, "vocab_size", None),
            "hidden_size": getattr(text_cfg, "hidden_size", None),
            "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", None),
            "num_attention_heads": getattr(text_cfg, "num_attention_heads", None),
            "num_key_value_heads": getattr(text_cfg, "num_key_value_heads", None),
            "head_dim": getattr(text_cfg, "head_dim", None),
            "sliding_window": getattr(text_cfg, "sliding_window", None),
        },
        "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,
        "n_new": N_NEW,
        "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}\n  argmax={golden['argmax']}  continuation={cont}")


if __name__ == "__main__":
    main()

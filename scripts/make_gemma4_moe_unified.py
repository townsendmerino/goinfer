#!/usr/bin/env python
"""Build a SYNTHETIC 'unified' (Gemma4ForConditionalGeneration-shaped) variant of the
tiny gemma4-MoE checkpoint, to verify goinfer's loader handles the real 26B-A4B's
layout WITHOUT the 51 GB download. The real checkpoint differs from the tiny golden
ONLY in packaging:

  - the text decoder lives under `model.language_model.*` (not `model.*`);
  - the top config is model_type "gemma4" (architectures Gemma4ForConditionalGeneration)
    with the text arch nested under `text_config`;
  - there is a vision tower (`model.vision_tower.*`, `model.embed_vision.*`) the
    text-only load must ignore.

So this re-keys the tiny checkpoint's tensors to the language_model prefix, wraps the
config, and adds one dummy `model.vision_tower.*` tensor. The WEIGHTS are byte-identical
to gemma4-moe-tiny, so the SAME golden (gemma4_moe_forward_golden.json) must reproduce —
proving the prefix + text_config flatten + vision-skip path, not new numerics.

    ~/.venv-vl/bin/python scripts/make_gemma4_moe_unified.py
    -> testdata/gemma4-moe-unified-tiny/  (reuses gemma4_moe_forward_golden.json)
"""
import json
import os

import numpy as np
from safetensors.numpy import load_file, save_file

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "testdata", "gemma4-moe-tiny")
DST = os.path.join(HERE, "..", "testdata", "gemma4-moe-unified-tiny")


def main():
    tensors = load_file(os.path.join(SRC, "model.safetensors"))
    # Re-key model.* -> model.language_model.* (the text decoder's real placement).
    out = {}
    for k, v in tensors.items():
        assert k.startswith("model."), k
        out["model.language_model." + k[len("model."):]] = v
    # A dummy vision tensor the text-only loader must never request.
    out["model.vision_tower.std_scale"] = np.zeros(4, dtype=np.float32)
    os.makedirs(DST, exist_ok=True)
    save_file(out, os.path.join(DST, "model.safetensors"), metadata={"format": "pt"})

    # Wrap the config: top model_type "gemma4" (Gemma4ForConditionalGeneration) with the
    # tiny text arch nested under text_config, plus a token vision_config.
    src_cfg = json.load(open(os.path.join(SRC, "config.json")))
    unified = {
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": src_cfg,
        "vision_config": {"model_type": "gemma4_vision", "hidden_size": 8},
    }
    json.dump(unified, open(os.path.join(DST, "config.json"), "w"), indent=1)
    # carry the generation config if present (harmless).
    gc = os.path.join(SRC, "generation_config.json")
    if os.path.exists(gc):
        json.dump(json.load(open(gc)), open(os.path.join(DST, "generation_config.json"), "w"))
    print(f"wrote {DST}: {len(out)} tensors (text re-keyed to model.language_model.*, +1 dummy vision), "
          f"top model_type=gemma4 / text_config=gemma4_text")


if __name__ == "__main__":
    main()

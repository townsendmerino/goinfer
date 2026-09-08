#!/usr/bin/env python3
"""Real-model parity golden for Ministral 3 (outer model_type "mistral3",
Mistral3ForConditionalGeneration; goinfer's registry key is "mistral3", per
decoder/ministral3_test.go's own note) — the T3 promotion of the mistral3 family from
tiny-golden to a released checkpoint (docs/parity-coverage-policy.md).

The released checkpoints are a vision-language WRAPPER (Mistral3ForConditionalGeneration,
text_config.model_type "ministral3") even though goinfer only implements the text decoder —
the safetensors loader is pull-based and never queries the vision-tower tensor prefixes, the
same mechanism gemma4_unified_text/qwen2_5_vl already rely on. So this script loads the FULL
wrapper via AutoModelForImageTextToText and calls it with pixel_values=None: HF's own
text-only path through the wrapper, not a hand-built weight extraction, which is exactly the
forward goinfer's text-only load reproduces.

target the -BF16 sibling repo. mistralai/Ministral-3-3b-Instruct-2512 (no suffix) ships
FP8-quantized by default (quant_method: "fp8") -- not f32-castable the way this gate's tight
cosine bar needs; mistralai/Ministral-3-3b-Instruct-2512-BF16 is the unquantized sibling
(docs/task-families-2026-09.md:653).

Exercises the two real new primitives on real weights: attn-temp (AttnTempBeta/
AttnTempOrigMaxPos, Llama4's own formula generalized to run alongside RoPE) and YaRN with the
DeepSeek-style mscale/mscale_all_dim override (both 1.0 on the real release, per that doc).

Dumps the last-token logits + argmax + a short greedy continuation (token IDs) for a fixed
prompt; the goinfer side (ministral3_real_test.go, build tag realckpt) loads the same
safetensors at f32 and matches argmax + continuation + cosine >= 0.9999.

    ~/.venv-vl/bin/python scripts/pin_ministral3_real.py
    -> testdata/ministral3_real_golden.json   (committed; the ~15 GB weights are NOT)

Put the checkpoint at ~/models/ministral3-3b-bf16 (mistralai/Ministral-3-3b-Instruct-2512-BF16),
or set GOINFER_MINISTRAL3_3B to its path.
"""
import json
import os

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

CKPT = os.environ.get("GOINFER_MINISTRAL3_3B", os.path.expanduser("~/models/ministral3-3b-bf16"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "ministral3_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForImageTextToText.from_pretrained(CKPT, torch_dtype=torch.float32).eval()
    cfg = model.config
    text_cfg = cfg.text_config
    arch = cfg.architectures[0] if cfg.architectures else "?"
    print(f"loaded {arch} model_type={cfg.model_type} text_config.model_type={text_cfg.model_type} "
          f"num_hidden_layers={text_cfg.num_hidden_layers} sliding_window={getattr(text_cfg, 'sliding_window', None)} "
          f"rope_type={getattr(getattr(text_cfg, 'rope_parameters', None), 'get', lambda k, d=None: None)('rope_type')}")

    ids = tok(PROMPT, return_tensors="pt").input_ids
    prompt_ids = ids[0].tolist()
    with torch.no_grad():
        # pixel_values=None: text-only forward through the wrapper, the same path goinfer's
        # text-only loader implements (it never reads the vision-tower tensors at all).
        last = model(input_ids=ids, pixel_values=None).logits[0, -1].to(torch.float64)
        argmax = int(torch.argmax(last).item())
        cont, cur = [], ids
        for _ in range(N_NEW):
            nxt = int(torch.argmax(model(input_ids=cur, pixel_values=None).logits[0, -1]).item())
            cont.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    golden = dict(
        prompt=PROMPT,
        prompt_ids=prompt_ids,
        argmax=argmax,
        vocab_size=text_cfg.vocab_size,
        last_logits=[float(x) for x in last.tolist()],
        n_new=N_NEW,
        continuation_ids=cont,
    )
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print("wrote", os.path.relpath(OUT))
    print("prompt_ids", prompt_ids, "argmax", argmax, "cont", cont)
    print("continuation:", repr(tok.decode(cont)))


if __name__ == "__main__":
    main()

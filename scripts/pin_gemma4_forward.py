#!/usr/bin/env python3
"""Pin the Gemma 4 (E2B) HF forward oracle: last-position next-token logits for a
fixed BOS-prefixed prompt, in the testdata/*_forward_golden.json format goinfer's
parity tests consume. Run in bfloat16 (a 5B model in float32 won't fit 16 GB);
the GGUF parity test compares goinfer's quantized forward to this via argmax +
cosine. Resolves the open softcap question empirically (does HF cap to ±30?).

    /tmp/g4venv/bin/python scripts/pin_gemma4_forward.py
"""
import json, os, numpy as np, torch

PATH = os.path.expanduser("~/models/gemma-4-E2B-unq")
OUT = os.path.expanduser("~/tmcode/goinfer/testdata/gemma4_forward_golden.json")
PROMPT = "The capital of France is"

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(PATH)
ids = tok(PROMPT, return_tensors="pt").input_ids

# The checkpoint is the FULL multimodal model (model.{language_model,vision_tower,
# audio_tower}.*), so it must load as Gemma4ForConditionalGeneration — loading the
# text-only Gemma4ForCausalLM leaves every weight randomly initialized (key
# mismatch). Text-only input (no pixel/audio) still routes through the language
# model, so out.logits are the correct next-token logits.
from transformers import Gemma4ForConditionalGeneration

model = Gemma4ForConditionalGeneration.from_pretrained(
    PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True
).eval()

with torch.no_grad():
    out = model(input_ids=ids)
logits = out.logits[0, -1].float().numpy()

order = np.argsort(-logits)
argmax = int(order[0])
rng = np.random.default_rng(1234)
samp = rng.choice(len(logits), 256, replace=False)
lf64 = logits.astype(np.float64)
golden = {
    "model_id": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
    "note": "E2B forward oracle (HF bf16, CPU); last-position next-token logits.",
    "dtype": "bfloat16",
    "prompt": PROMPT,
    "ids": [int(x) for x in ids[0].tolist()],
    "argmax": argmax,
    "argmax_token": tok.decode([argmax]),
    "vocab_size": int(len(logits)),
    "stats": {
        "n": int(len(logits)),
        "sum": float(lf64.sum()),
        "sum_sq": float((lf64 * lf64).sum()),
        "min": float(logits.min()),
        "max": float(logits.max()),
    },
    "top_k": [[int(i), float(logits[i])] for i in order[:32]],
    "sample": [[int(i), float(logits[i])] for i in samp],
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(golden, open(OUT, "w"), indent=1)
print("ids:", golden["ids"])
print("argmax:", argmax, repr(golden["argmax_token"]))
print("logit min/max:", golden["stats"]["min"], golden["stats"]["max"],
      "(|max|>30 => HF does NOT softcap; ~30 => softcap applied)")
print("wrote", OUT)

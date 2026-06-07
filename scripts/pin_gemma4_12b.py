#!/usr/bin/env python3
"""Pin the Gemma 4 12B HF parity golden (BOS-prefixed) + a short greedy
continuation for a coherence check. bf16 on CPU; writes the forward_golden format
goinfer's parity test consumes. The 12B is K=V (attention_k_eq_v) on its global
layers — this golden gates that path.

    ~/g4venv/bin/python scripts/pin_gemma4_12b.py
"""
import json, os, numpy as np, torch

PATH = os.path.expanduser("~/models/gemma-4-12b-unq")
OUT = os.path.expanduser("~/mycode/goinfer/testdata/gemma4_12b_forward_golden.json")
PROMPT = "The capital of France is"

from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(PATH)
# This is an instruction-tuned model: a raw completion prompt yields degenerate
# next-tokens ('0','1',...). Use the chat template so the golden's argmax is the
# meaningful first answer token — the parity test then doubles as a coherence gate.
MSG = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
ids = tok.apply_chat_template(MSG, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
print("templated ids:", ids[0].tolist())

model = AutoModelForCausalLM.from_pretrained(
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
    "model_id": "google/gemma-4-12B-it-qat-q4_0-unquantized",
    "note": "12B forward oracle (HF bf16, CPU, chat-templated); last-pos next-token logits. Gates K=V; argmax is the first answer token (coherence).",
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
    "top_k": [[int(i), float(logits[i])] for i in order[:16]],
    "sample": [[int(i), float(logits[i])] for i in samp],
}
json.dump(golden, open(OUT, "w"), indent=1)
print("argmax:", argmax, repr(golden["argmax_token"]), "logit", round(golden["stats"]["max"], 3))

# Coherence: greedy 8-token continuation.
with torch.no_grad():
    gen = model.generate(ids, max_new_tokens=8, do_sample=False)
cont = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
print("greedy continuation:", repr(cont))
print("wrote", OUT)

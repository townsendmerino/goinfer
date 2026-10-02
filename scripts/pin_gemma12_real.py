#!/usr/bin/env python3
"""Real-checkpoint goldens for Gemma 1 (gemma-2b-it), CodeGemma (codegemma-2b) and Gemma 2 (gemma-2-2b-it): HF f32,
eager attention (Gemma 2's attention softcap is skipped by SDPA), next-token logits at the last prompt position and an
8-token greedy continuation.

    ~/d0venv/bin/python scripts/pin_gemma12_real.py   ->  testdata/gemma12_real_golden.json.gz

Checkpoints from ~/models (ungated mirrors: unsloth/gemma-2b-it, unsloth/codegemma-2b, unsloth/gemma-2-2b-it), or set
GEMMA_2B / CODEGEMMA_2B / GEMMA2_2B. The prompt is tokenized by each checkpoint's own tokenizer, BOS included; CodeGemma
gets a code prompt.
"""
import gzip, json, os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "testdata", "gemma12_real_golden.json.gz")
N_NEW = 8
MODELS = [
    ("gemma-2b-it", "GEMMA_2B", "The capital of France is"),
    ("codegemma-2b", "CODEGEMMA_2B", "def fibonacci(n):\n    "),
    ("gemma-2-2b-it", "GEMMA2_2B", "The capital of France is"),
]

torch.set_grad_enabled(False)
out = {"transformers": transformers.__version__, "torch": torch.__version__, "models": []}
for name, env, text in MODELS:
    path = os.environ.get(env, os.path.expanduser(f"~/models/{name}"))
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(text, add_special_tokens=True)["input_ids"]
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()
    last = m(torch.tensor([ids]), use_cache=False).logits[0, -1].float()
    cur, cont = list(ids), []
    for _ in range(N_NEW):
        nxt = int(m(torch.tensor([cur]), use_cache=False).logits[0, -1].argmax())
        cont.append(nxt)
        cur.append(nxt)
    out["models"].append({"name": name, "model_type": m.config.model_type, "prompt": text, "prompt_ids": ids,
                          "argmax": int(last.argmax()), "last_logits": last.tolist(), "n_new": N_NEW,
                          "continuation_ids": cont, "continuation": tok.decode(cont)})
    print(f"{name} ({m.config.model_type}): {len(ids)} prompt tokens, argmax {int(last.argmax())}, continuation {tok.decode(cont)!r}", flush=True)
    del m
with gzip.open(OUT, "wt") as f:
    json.dump(out, f)
print("wrote", OUT)

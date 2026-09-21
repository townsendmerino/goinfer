#!/usr/bin/env python3
"""Real-model parity golden for Spark-X2.5-1.7B (model_type "spark2_5", Spark2_5ForCausalLM) —
Gate 2 (docs/tasks/task-spark-x2-5.md): the T3 promotion from Gate 1's tiny-random synthetic
fixture to a released checkpoint. 1.7B is the smaller of the two released sizes (task doc's own
"start with the 1.7B... then the 4B").

Needs transformers==4.57.1 EXACTLY, same reason as scripts/pin_spark2_5_tiny.py: the vendored
modeling_spark.py breaks against newer transformers (tied-weight-key expansion assumes a dict,
not this module's old-style list; create_causal_mask()'s signature changed). Unlike the tiny
fixture, tie_word_embeddings=True on the REAL checkpoint cannot be worked around by leaving it
False — this run needs the exact pinned version, not a workaround.

    ~/.venv-spark25/bin/python scripts/pin_spark2_5_real.py
    -> testdata/spark2_5_real_golden.json   (committed; the ~3.2 GB weights are NOT)

Put the checkpoint at ~/models/spark25-1.7b (XHToken/Spark-X2.5-1.7B), or set
GOINFER_SPARK25_1_7B to its path.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ.get("GOINFER_SPARK25_1_7B", os.path.expanduser("~/models/spark25-1.7b"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "spark2_5_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32, trust_remote_code=True).eval()
    cfg = model.config
    arch = cfg.architectures[0] if cfg.architectures else "?"
    print(f"loaded {arch} model_type={cfg.model_type} num_hidden_layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size} heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"head_dim={cfg.head_dim} tie_word_embeddings={cfg.tie_word_embeddings}")

    ids = tok(PROMPT, return_tensors="pt").input_ids
    prompt_ids = ids[0].tolist()
    with torch.no_grad():
        last = model(ids).logits[0, -1].to(torch.float64)
        argmax = int(torch.argmax(last).item())
        cont, cur = [], ids
        for _ in range(N_NEW):
            nxt = int(torch.argmax(model(cur).logits[0, -1]).item())
            cont.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    golden = dict(
        prompt=PROMPT,
        prompt_ids=prompt_ids,
        argmax=argmax,
        vocab_size=cfg.vocab_size,
        layer_types=cfg.layer_types,
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

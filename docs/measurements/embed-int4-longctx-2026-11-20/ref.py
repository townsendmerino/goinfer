#!/usr/bin/env python3
"""The HF f32 reference for #1160's int4-head check (README.md beside this file).

For one GGUF checkpoint, loads it with transformers in f32 (the GGUF's own weights, dequantized), tokenizes each document (its text at
a pinned commit, `git show <rev>:<path>`), keeps the first N_TOKENS ids, runs one causal forward, and writes for each scored position
p (the 64 positions ending at each K) the full log-softmax over the vocabulary as float16 (<name>.lp.f16), with the ids and positions
in <name>.json. Position p's distribution predicts ids[p+1] after ids[0..p].

  ~/.venv-vl/bin/python -I ref.py --gguf-dir ~/models/qwen05 --gguf-file qwen2.5-coder-0.5b-instruct-q4_k_m.gguf --repo ~/mycode/reanchor-wt \
      --rev 9d7485a29 --out <dir> internal/app/serveapp/openai.go docs/server.md ...
"""
import argparse, hashlib, json, os, subprocess, sys, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

KS = [256, 1024, 2048, 4096, 8192]
WINDOW = 64
N_TOKENS = 8256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--gguf-file", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("docs", nargs="+")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.gguf_dir, gguf_file=a.gguf_file)
    model = AutoModelForCausalLM.from_pretrained(a.gguf_dir, gguf_file=a.gguf_file, torch_dtype=torch.float32)
    model.eval()
    print(f"loaded {a.gguf_file} in {time.time() - t0:.0f}s: {sum(p.numel() for p in model.parameters())} params, "
          f"torch {torch.__version__}, attn {model.config._attn_implementation}", flush=True)
    scored, k_of = [], []
    for k in KS:
        for p in range(k - WINDOW, k):
            scored.append(p)
            k_of.append(k)
    for path in a.docs:
        name = path.replace("/", "_")
        text = subprocess.run(["git", "-C", a.repo, "show", f"{a.rev}:{path}"], capture_output=True, text=True, check=True).stdout
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) < N_TOKENS:
            sys.exit(f"{path}: {len(ids)} tokens, need {N_TOKENS}")
        ids = ids[:N_TOKENS]
        t = time.time()
        with torch.no_grad():
            hidden = model.model(torch.tensor([ids[: max(scored) + 1]])).last_hidden_state[0]
            logits = model.lm_head(hidden[scored]).float()
            lp = torch.log_softmax(logits, dim=-1).numpy().astype(np.float16)
        vocab = lp.shape[1]
        lp.tofile(os.path.join(a.out, name + ".lp.f16"))
        with open(os.path.join(a.out, name + ".json"), "w") as f:
            json.dump({"doc": name, "path": path, "rev": a.rev, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "gguf": a.gguf_file, "ids": ids, "scored": scored, "k_of": k_of, "vocab": vocab}, f)
        print(f"{name}: {len(ids)} ids, forward over {max(scored) + 1} positions in {time.time() - t:.0f}s, vocab {vocab}, "
              f"elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

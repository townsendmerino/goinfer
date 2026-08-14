#!/usr/bin/env python3
"""shard_checkpoint.py — split a single-file safetensors checkpoint into N
shards + a model.safetensors.index.json, to exercise the decoder's sharded
loader (multi-model-plan G1) on a real model without downloading a genuinely
sharded (multi-GB) checkpoint.

Writes <out>/model-0000i-of-0000N.safetensors + the index, and copies
config.json so decoder.LoadWeights(<out>) works. The result mirrors the HF
sharded layout byte-for-byte (the loader can't tell it from a real shard set).

Usage (from repo root):
    .venv/bin/python scripts/shard_checkpoint.py \
        [SRC_DIR=testdata/gemma-3-270m] [OUT_DIR=testdata/gemma-3-270m-sharded] [N=3]
"""
from __future__ import annotations

import json
import os
import shutil
import sys


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "testdata/gemma-3-270m"
    out = sys.argv[2] if len(sys.argv) > 2 else "testdata/gemma-3-270m-sharded"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    weights = os.path.join(src, "model.safetensors")
    if not os.path.exists(weights):
        sys.stderr.write(f"[shard] missing {weights}\n")
        return 1

    from safetensors.torch import load_file, save_file

    sd = load_file(weights)
    names = sorted(sd.keys())
    groups = [names[i::n] for i in range(n)]  # round-robin spreads layers across shards

    os.makedirs(out, exist_ok=True)
    weight_map: dict[str, str] = {}
    total = 0
    for i, g in enumerate(groups):
        fn = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        save_file({k: sd[k].contiguous() for k in g}, os.path.join(out, fn))
        for k in g:
            weight_map[k] = fn
        total += os.path.getsize(os.path.join(out, fn))

    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)

    # Copy the non-weight files so <out> is a complete, loadable checkpoint
    # (config for the loader; tokenizer files for end-to-end generation).
    for name in ("config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                 "special_tokens_map.json", "generation_config.json", "added_tokens.json"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, name))

    sys.stderr.write(f"[shard] wrote {n} shards + index to {out} ({total / 1e6:.0f} MB), "
                     f"{len(weight_map)} tensors\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

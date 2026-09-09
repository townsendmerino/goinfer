#!/usr/bin/env python
"""Real-model parity golden for a family whose checkpoint is too big to co-reside with anything
else in 62 GB RAM at bf16 — laguna (33B-A3B, ~63 GB) and qwen3_5 (Qwen3.8-27B, ~55.6 GB). Both
already had a real checkpoint locally and a REQUIRED gate, but both gates were coherence-only by
design: TestLagunaReal_gate has no emitParityRow call, and qwen3_5's own manifest text says
plainly that no bf16 reference forward was ever run. This is that reference forward, using the
SAME technique scripts/pin_qwen3next_real.py already proved on an 80B model (163 GB bf16) — the
reference and goinfer never need to be resident at the same instant, so the reference is loaded
via accelerate's DISK OFFLOAD (roughly one layer resident, streamed from NVMe) and pinned to a
JSON file; the goinfer-side gate reads that file back later, in a separate process.

ONE SCRIPT, --family, BECAUSE THE OFFLOAD MACHINERY IS IDENTICAL AND ONLY THE MODEL CLASS
DIFFERS. Two real divergences between the families, both isolated below:
  - laguna ships custom remote code (LagunaForCausalLM via auto_map) — needs
    trust_remote_code=True and AutoModelForCausalLM.
  - qwen3_5's released checkpoint is a VISION-LANGUAGE WRAPPER
    (Qwen3_5ForConditionalGeneration, mainline transformers, no trust_remote_code needed) even
    though only the text path is used — same shape as mistral3's own real-checkpoint gate:
    AutoModelForImageTextToText with pixel_values=None for the text-only forward.
Neither divergence touches the offload/pinning logic itself, which is why this is one script
with a branch, not two independent copies of the boilerplate.

GPU IS DELIBERATELY HIDDEN, same reasoning as pin_qwen3next_real.py: the card is 8 GB against a
50-60 GB model, so a device_map that notices it wins a layer or two and adds a second placement
regime to reason about. CPU + disk is one story.

    ~/.venv-vl/bin/python scripts/pin_sequential_oracle.py --family laguna
    ~/.venv-vl/bin/python scripts/pin_sequential_oracle.py --family qwen3_5
    -> testdata/{laguna,qwen3_5}_real_golden.json   (committed; weights are NOT)

NEVER point CKPT at /srv/models: the archive is a 5400 rpm SMR disk and is a bench surface for
neither machine. See docs/benchmarks.md, "Model storage".
"""
import argparse
import json
import os
import sys
import time

# Hide the GPU before torch initialises it. See the docstring.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

HERE = os.path.dirname(__file__)
PROMPT = "The capital of France is"
N_NEW = 8
ARCHIVE_ROOTS = ("/srv/models", "/Volumes/")

FAMILIES = {
    "laguna": dict(
        ckpt_env="LAGUNA_CKPT", ckpt_default="~/models/laguna-xs2",
        out="laguna_real_golden.json", trust_remote_code=True, wrapper=False,
        max_cpu_env="LAGUNA_MAX_CPU", max_cpu_default="40GiB",
    ),
    "qwen3_5": dict(
        ckpt_env="QWEN35_CKPT", ckpt_default="~/models/qwen3.8-27b",
        out="qwen3_5_real_golden.json", trust_remote_code=False, wrapper=True,
        max_cpu_env="QWEN35_MAX_CPU", max_cpu_default="40GiB",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    args = ap.parse_args()
    spec = FAMILIES[args.family]

    ckpt = os.path.expanduser(os.environ.get(spec["ckpt_env"], spec["ckpt_default"]))
    real = os.path.realpath(ckpt)
    if any(real.startswith(r) for r in ARCHIVE_ROOTS):
        sys.exit(f"REFUSED: {ckpt} resolves to {real}, on the ARCHIVE. Validate from the local "
                 f"bench set (~/models); see docs/benchmarks.md 'Model storage'.")

    idx_path = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        shards = [f for f in os.listdir(ckpt) if f.endswith(".safetensors")]
        idx = json.load(open(idx_path))
        want = len(set(idx["weight_map"].values()))
        if len(shards) != want:
            sys.exit(f"REFUSED: {len(shards)}/{want} shards present in {ckpt}. A reference "
                      f"forward over a PARTIAL checkpoint silently reads uninitialised weights "
                      f"for the missing layers -- it does not error, it produces a plausible "
                      f"wrong golden. Finish the download and re-run.")

    offload = os.path.expanduser(f"~/{args.family}-t3-offload")
    os.makedirs(offload, exist_ok=True)
    out = os.path.join(HERE, "..", "testdata", spec["out"])
    max_cpu = os.environ.get(spec["max_cpu_env"], spec["max_cpu_default"])

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=spec["trust_remote_code"])
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist(), flush=True)

    load_kwargs = dict(
        dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory={"cpu": max_cpu}, offload_folder=offload,
        trust_remote_code=spec["trust_remote_code"],
    )
    model_cls = AutoModelForImageTextToText if spec["wrapper"] else AutoModelForCausalLM

    t0 = time.time()
    m = model_cls.from_pretrained(ckpt, **load_kwargs).eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    dev = getattr(m, "hf_device_map", None)
    if dev:
        kinds = {}
        for v in dev.values():
            kinds[str(v)] = kinds.get(str(v), 0) + 1
        print("placement:", kinds, flush=True)

    fwd_kwargs = dict(pixel_values=None) if spec["wrapper"] else {}

    with torch.no_grad():
        t0 = time.time()
        last = m(input_ids=ids, use_cache=False, **fwd_kwargs).logits[0, -1].float().tolist()
        print(f"prompt forward {time.time()-t0:.0f}s", flush=True)
        cur = ids.tolist()[0]
        cont = []
        for i in range(N_NEW):
            t0 = time.time()
            o = m(input_ids=torch.tensor([cur]), use_cache=False, **fwd_kwargs)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])
            print(f"  step {i+1}/{N_NEW} -> {cont[-1]} ({time.time()-t0:.0f}s)", flush=True)

    g = dict(
        note=f"{args.family} bf16 reference, full model via accelerate disk offload",
        prompt=PROMPT, prompt_ids=ids[0].tolist(), argmax=int(torch.tensor(last).argmax()),
        last_logits=last, n_new=N_NEW, continuation_ids=cont,
        continuation_text=tok.decode(cont),
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(g, open(out, "w"))
    print(f"argmax={g['argmax']} cont={cont!r} -> {g['continuation_text']!r}")
    print("wrote", out)


if __name__ == "__main__":
    main()

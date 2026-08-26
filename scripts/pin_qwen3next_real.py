#!/usr/bin/env python
"""Real-model parity golden for Qwen/Qwen3-Next-80B-A3B-Instruct (qwen3_next) — the bf16
reference behind the family's T3 row.

WHY THIS EXISTS, AND WHY IT IS NOT A SLICE. The family's T3 row was blocked on a recorded
constraint: "no full reference forward of a 163GB bf16 model fits 62GB". That is true of a
RESIDENT reference and only of that. The reference and goinfer never need to be alive at the
same instant — this script pins the reference's logits to a JSON file, and
qwen3next_real_test.go reads that file back later, in a separate process. Pinning offline is
already how every tiny golden in this repo is made; the existing slice route generalised the
SLICE when what needed generalising was the PINNING.

So the model is loaded with accelerate's DISK OFFLOAD: resident footprint is roughly one
layer, not the model, and a forward streams the checkpoint from NVMe. It is slow and it is
correct, which is the right trade for a gate that runs per release rather than per commit.

GPU IS DELIBERATELY HIDDEN. The card is 8 GB against a 163 GB model, so a device_map that
notices it wins two layers and adds a second placement regime to reason about. CPU + disk is
one story.

    models-pull / hf download Qwen/Qwen3-Next-80B-A3B-Instruct --local-dir ~/models/...
    ~/.venv-vl/bin/python scripts/pin_qwen3next_real.py
    -> testdata/qwen3next_real_golden.json   (committed; weights are NOT)

NEVER point CKPT at /srv/models: the archive is a 5400 rpm SMR disk and is a bench surface for
neither machine. See docs/benchmarks.md, "Model storage".
"""
import json, os, sys, time

# Hide the GPU before torch initialises it. See the docstring.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from transformers import AutoTokenizer, Qwen3NextForCausalLM

CKPT = os.path.expanduser(os.environ.get("QWEN3NEXT_CKPT", "~/models/qwen3next-80b-partial"))
OFFLOAD = os.path.expanduser(os.environ.get("QWEN3NEXT_OFFLOAD", "~/qwen3next-t3/offload"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "qwen3next_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 6
# Headroom below the box's 62 GB so the offloader has somewhere to work and the machine stays
# usable. Raising this does not make the run correct-er, only more likely to be OOM-killed.
MAX_CPU = os.environ.get("QWEN3NEXT_MAX_CPU", "36GiB")

ARCHIVE_ROOTS = ("/srv/models", "/Volumes/")


def main():
    real = os.path.realpath(CKPT)
    if any(real.startswith(r) for r in ARCHIVE_ROOTS):
        sys.exit(f"REFUSED: {CKPT} resolves to {real}, on the ARCHIVE. Validate from the local "
                 f"bench set (~/models); see docs/benchmarks.md 'Model storage'.")

    shards = [f for f in os.listdir(CKPT) if f.endswith(".safetensors")]
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
    want = len(set(idx["weight_map"].values()))
    if len(shards) != want:
        sys.exit(f"REFUSED: {len(shards)}/{want} shards present in {CKPT}. A reference forward "
                 f"over a PARTIAL checkpoint silently reads uninitialised weights for the missing "
                 f"layers -- it does not error, it produces a plausible wrong golden. Finish the "
                 f"download and re-run.")

    os.makedirs(OFFLOAD, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print("prompt_ids =", ids[0].tolist(), flush=True)

    t0 = time.time()
    m = Qwen3NextForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory={"cpu": MAX_CPU}, offload_folder=OFFLOAD,
    ).eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    dev = getattr(m, "hf_device_map", None)
    if dev:
        kinds = {}
        for v in dev.values():
            kinds[str(v)] = kinds.get(str(v), 0) + 1
        print("placement:", kinds, flush=True)

    with torch.no_grad():
        t0 = time.time()
        last = m(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        print(f"prompt forward {time.time()-t0:.0f}s", flush=True)
        cur = ids.tolist()[0]
        cont = []
        for i in range(N_NEW):
            t0 = time.time()
            o = m(input_ids=torch.tensor([cur]), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])
            print(f"  step {i+1}/{N_NEW} -> {cont[-1]} ({time.time()-t0:.0f}s)", flush=True)

    g = dict(note="Qwen/Qwen3-Next-80B-A3B-Instruct (qwen3_next, 80B-A3B hybrid) bf16 reference, "
                  "full model via accelerate disk offload",
             prompt=PROMPT, prompt_ids=ids[0].tolist(), argmax=int(torch.tensor(last).argmax()),
             last_logits=last, n_new=N_NEW, continuation_ids=cont,
             continuation_text=tok.decode(cont))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(g, open(OUT, "w"))
    print(f"argmax={g['argmax']} cont={cont!r} -> {g['continuation_text']!r}")


if __name__ == "__main__":
    main()

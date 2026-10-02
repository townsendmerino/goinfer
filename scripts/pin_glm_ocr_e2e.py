#!/usr/bin/env python3
"""Pin the GLM-OCR END-TO-END reference (gate O3 of docs/tasks/task-glm-ocr-2026-10.md): the real
checkpoint, transformers f32 on CPU, greedy, the checkpoint's OWN processor and chat template, on the
three procedurally rendered documents of scripts/gen_glm_ocr_doc_images.py (NOT real scans).

    ~/.venv-vl/bin/python scripts/pin_glm_ocr_e2e.py [--ckpt ~/models/glm-ocr] [--images invoice table formula]
                                                     [--pv-dir ~/models/glm-ocr-e2e] [--n-new 64]

Per image it writes
    testdata/glm_ocr/golden_<name>.json.gz   (committed; gzip per CLAUDE.md, read via decoder.ReadGoldenJSONForTest)
    <pv-dir>/<name>.pv.f32                   (NOT committed: HF's own pixel_values, raw little-endian f32,
                                              [n_patches,1176]; the O3 gate feeds them to the Go tower to separate
                                              a preprocessing divergence from a model-path one)
The golden holds: image sha256, the prompt, the processor grid, input_ids (HF's rendering of the template with the
image placeholder expanded), the n_new greedy token ids, their decoded text, the top-1/top-2 logit gap at every
step (the repo's near-tie rule needs it), and the LAST-PROMPT-TOKEN logits as base64 little-endian f32.

Attention is eager (the O2 tower reference used it; sdpa on CPU would also work but eager is what the O2 numbers
were taken against). A heartbeat line is printed every 30 s: the tower alone took ~70 s on a 1.47 MP page and the
whole image takes minutes, so silence would look like a hang.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DOCS = os.path.join(REPO, "testdata", "glm_ocr")
# (image, task prompt). The card's three task prompts, one per document kind.
PROMPTS = {
    "invoice": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


def heartbeat(label, t0, stop):
    while not stop.wait(30):
        print(f"[{label}] still running, {time.time() - t0:.0f}s elapsed", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="~/models/glm-ocr")
    ap.add_argument("--images", nargs="*", default=list(PROMPTS))
    ap.add_argument("--pv-dir", default="~/models/glm-ocr-e2e")
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--prompt", default=None, help="override the task prompt for every image (also tags the golden file name with --tag)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ckpt, pvdir = os.path.expanduser(a.ckpt), os.path.expanduser(a.pv_dir)
    os.makedirs(pvdir, exist_ok=True)

    t0 = time.time()
    print(f"loading {ckpt} (f32, eager attention)...", file=sys.stderr, flush=True)
    proc = AutoProcessor.from_pretrained(ckpt)
    model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.float32, attn_implementation="eager").eval()
    print(f"loaded in {time.time() - t0:.0f}s; transformers loaded model {type(model).__name__}", file=sys.stderr, flush=True)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)

    for name in a.images:
        prompt = a.prompt or PROMPTS[name]
        path = os.path.join(DOCS, name + ".png")
        raw = open(path, "rb").read()
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        ids = inp["input_ids"][0].tolist()
        grid = inp["image_grid_thw"].tolist()
        pv = inp["pixel_values"].to(torch.float32)
        n_img = ids.count(59280)
        label = f"{name}:{prompt!r}"
        print(f"[{label}] {img.width}x{img.height}, grid {grid}, {pv.shape[0]} patches, {len(ids)} prompt tokens "
              f"({n_img} image); generating {a.n_new} greedy tokens", file=sys.stderr, flush=True)
        t1 = time.time()
        stop = threading.Event()
        threading.Thread(target=heartbeat, args=(label, t1, stop), daemon=True).start()
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=a.n_new, do_sample=False, output_logits=True, return_dict_in_generate=True)
        stop.set()
        toks = gen.sequences[0, len(ids):].tolist()
        # gen.logits[i] is the raw logit row that produced token i; row 0 is the LAST PROMPT TOKEN's logits.
        rows = [lg[0].float() for lg in gen.logits]
        gaps = [float(r.topk(2).values.diff().abs()) for r in rows]
        last = rows[0].numpy().astype("<f4")
        first_stop = next((i for i, t in enumerate(toks) if t in eos), None)
        text = proc.tokenizer.decode(toks)
        golden = dict(
            note="HF f32 CPU eager greedy reference for the GLM-OCR O3 gate. The image is procedurally rendered, NOT a real scan.",
            transformers=__import__("transformers").__version__, torch=torch.__version__,
            checkpoint=os.path.basename(ckpt.rstrip("/")),
            image=name + ".png", image_sha256=hashlib.sha256(raw).hexdigest(), image_wh=[img.width, img.height],
            prompt=prompt, grid_thw=grid, n_patches=int(pv.shape[0]), n_image_tokens=n_img,
            pixel_values_sha256=hashlib.sha256(pv.numpy().astype("<f4").tobytes()).hexdigest(),
            input_ids=ids, image_token_start=ids.index(59280), eos_ids=eos,
            n_new=a.n_new, hf_tokens=toks, hf_text=text, hf_gaps=gaps, hf_stopped_at=first_stop,
            last_logits_f32_b64=base64.b64encode(last.tobytes()).decode(), vocab=int(last.shape[0]),
            hf_seconds=round(time.time() - t1, 1))
        out = os.path.join(DOCS, f"golden_{name}{a.tag}.json.gz")
        with gzip.open(out, "wt") as f:
            json.dump(golden, f)
        pv.numpy().astype("<f4").tofile(os.path.join(pvdir, name + ".pv.f32"))
        print(f"[{label}] done in {time.time() - t1:.0f}s: stopped_at={first_stop} min-gap={min(gaps):.4f}\n{text}\n-> {out}",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

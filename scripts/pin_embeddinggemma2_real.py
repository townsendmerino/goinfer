#!/usr/bin/env python3
"""Gate 2 of docs/tasks/task-embeddinggemma2.md: the real google/embeddinggemma-2 against sentence-transformers.

Reads a real corpus from this repository's own top-level docs at a pinned revision (`git show REV:docs/<file>.md`),
split into sections at `## ` headings: the heading is a query, the body a document. Then, with sentence-transformers
(>= 6.1.0, the version the checkpoint requires) loading the checkpoint in float32 on the CPU:

  parity   48 texts, each with the prompt sentence-transformers applies (query, document, none, and two other named
           prompts; four long documents of 1,200+ tokens, past the sliding layers' 1,025-wide window): the input ids
           sentence-transformers feeds the model and its normalised 768-wide embedding. Written to --golden.
  truncate every qualifying section (up to 400): the heading embedded under "query", the body under "document", at 768
           and truncated (then renormalised) to 512, 256 and 128. For each width, recall@1 and recall@10 of a heading
           retrieving its own section by cosine. Written to --truncation and printed.

Run: python3 scripts/pin_embeddinggemma2_real.py --model ~/models/embeddinggemma-2 --rev <sha> \
         --golden testdata/embeddinggemma2-real/golden.json --truncation <out.json>
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import sentence_transformers
import transformers

WIDTHS = [768, 512, 256, 128]


def corpus(rev):
    files = subprocess.run(["git", "ls-tree", "--name-only", rev, "docs/"], capture_output=True, text=True, check=True).stdout.split()
    files = sorted(f for f in files if f.endswith(".md"))
    sections = []
    for f in files:
        text = subprocess.run(["git", "show", f"{rev}:{f}"], capture_output=True, text=True, check=True).stdout
        parts = re.split(r"(?m)^## +", text)
        for p in parts[1:]:
            head, _, body = p.partition("\n")
            head = re.sub(r"[`*_\[\]()]", "", head).strip()
            body = body.strip()
            if len(head) < 8 or len(body) < 300:
                continue
            sections.append({"file": f, "heading": head, "body": body[:2400]})
    return files, sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--truncation", required=True)
    a = ap.parse_args()
    model_dir = os.path.expanduser(a.model)
    if model_dir.startswith("/Volumes/") or model_dir.startswith("/srv/models"):
        sys.exit(f"{model_dir} is on the archive, not the bench set (CLAUDE.md)")
    t0 = time.time()
    files, secs = corpus(a.rev)
    print(f"[eg2 {time.time()-t0:6.1f}s] corpus: {len(secs)} sections from {len(files)} docs at {a.rev}", file=sys.stderr)
    m = SentenceTransformer(model_dir, model_kwargs={"dtype": torch.float32}, device="cpu")
    assert next(m.parameters()).dtype == torch.float32
    rev_file = os.path.join(model_dir, "REVISION")
    model_rev = open(rev_file).read().strip() if os.path.exists(rev_file) else "unknown"

    def ids_of(text):
        return m.tokenize([text])["input_ids"][0].tolist()

    # Parity set: deterministic picks from the corpus.
    items = []
    for s in secs[0:16]:
        items.append(("query", s["heading"]))
    for s in secs[16:32]:
        items.append(("document", s["body"][:1500]))
    for s in secs[32:40]:
        items.append(("", s["heading"] + ". " + s["body"][:400]))
    for s in secs[40:42]:
        items.append(("STS", s["heading"]))
    for s in secs[42:44]:
        items.append(("Classification", s["body"][:600]))
    longs, buf = [], ""
    for s in secs[44:]:
        buf += s["heading"] + "\n" + s["body"] + "\n\n"
        if len(ids_of(m.prompts["document"] + buf)) > 1200:
            longs.append(buf)
            buf = ""
            if len(longs) == 4:
                break
    for t in longs:
        items.append(("document", t))
    golden = {"model_revision": model_rev, "corpus_rev": a.rev, "transformers": transformers.__version__,
              "sentence_transformers": sentence_transformers.__version__, "torch": torch.__version__, "items": []}
    for prompt, text in items:
        full = (m.prompts[prompt] if prompt else "") + text
        emb = m.encode([text], prompt_name=prompt or None, prompt=None if prompt else "", convert_to_numpy=True)[0]
        golden["items"].append({"prompt": prompt, "text": text, "ids": ids_of(full), "embedding": [float(x) for x in emb]})
    os.makedirs(os.path.dirname(os.path.abspath(a.golden)), exist_ok=True)
    with open(a.golden, "w") as f:
        json.dump(golden, f)
    lens = [len(it["ids"]) for it in golden["items"]]
    print(f"[eg2 {time.time()-t0:6.1f}s] parity golden: {len(items)} texts, {min(lens)}-{max(lens)} tokens, {sum(l > 1025 for l in lens)} over 1025 -> {a.golden}", file=sys.stderr)

    # Truncation: heading -> own section retrieval at each width.
    secs = secs[:400]
    q = m.encode([s["heading"] for s in secs], prompt_name="query", convert_to_numpy=True, batch_size=16)
    d = m.encode([s["body"] for s in secs], prompt_name="document", convert_to_numpy=True, batch_size=16)
    res = {"model_revision": model_rev, "corpus_rev": a.rev, "sections": len(secs), "widths": {}}
    for w in WIDTHS:
        qw = q[:, :w] / np.linalg.norm(q[:, :w], axis=1, keepdims=True)
        dw = d[:, :w] / np.linalg.norm(d[:, :w], axis=1, keepdims=True)
        sim = qw @ dw.T
        rank = (sim > sim[np.arange(len(secs)), np.arange(len(secs))][:, None]).sum(axis=1)
        r1, r10 = float((rank < 1).mean()), float((rank < 10).mean())
        res["widths"][str(w)] = {"recall@1": r1, "recall@10": r10}
        print(f"[eg2 {time.time()-t0:6.1f}s] RESULT width {w:3d}: recall@1 {r1:.3f}, recall@10 {r10:.3f} ({len(secs)} sections)", file=sys.stderr)
    with open(a.truncation, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

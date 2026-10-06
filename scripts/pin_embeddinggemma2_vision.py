#!/usr/bin/env python3
"""Phase V of docs/tasks/task-embeddinggemma2.md: the reference for EmbeddingGemma 2 image embeddings.

With sentence-transformers (>= 6.1.0) loading google/embeddinggemma-2 in float32 on the CPU, for four repo images of
different aspect ratios, each as the image alone, the image with the `query` prompt, and the image followed by text
(12 cases), this records:

  --golden (committed): each case's image path, prompt, text, the input ids the model was fed, its soft-token count
      and its normalised 768-wide embedding;
  --artifacts (a local directory, not committed; several MB an image): per image, the processor's valid patches
      (<name>.patches.f32, [n_patches, 768] float32 little-endian) and their (x, y) positions (<name>.pos.i32,
      [n_patches, 2] int32), and HF's pooled image features from get_image_features (<name>.feats.f32,
      [n_soft, 512]), which Gate V2 compares aikit's tower with on HF's own pixels.

Run: python3 scripts/pin_embeddinggemma2_vision.py --model ~/models/embeddinggemma-2 \
         --golden testdata/embeddinggemma2-vision/golden.json --artifacts ~/goinfer-logs/embeddinggemma2-vision
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import sentence_transformers
import transformers

IMAGES = [
    "testdata/gemma3_preprocess_image.png",
    "testdata/qwen25vl_preprocess_image.png",
    "testdata/glm_ocr/formula.png",
    "testdata/glm_ocr/table.png",
]
TEXT = "a picture of a document"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--artifacts", required=True)
    a = ap.parse_args()
    model_dir = os.path.expanduser(a.model)
    if model_dir.startswith("/Volumes/") or model_dir.startswith("/srv/models"):
        sys.exit(f"{model_dir} is on the archive, not the bench set (CLAUDE.md)")
    art = os.path.expanduser(a.artifacts)
    os.makedirs(art, exist_ok=True)
    m = SentenceTransformer(model_dir, model_kwargs={"dtype": torch.float32}, device="cpu")
    inner = m[0].model
    cap = {}
    orig = inner.forward

    def hook(*args, **kw):
        cap.update(ids=kw.get("input_ids"), pv=kw.get("pixel_values"), pos=kw.get("image_position_ids"))
        return orig(*args, **kw)

    inner.forward = hook
    rev_file = os.path.join(model_dir, "REVISION")
    golden = {"model_revision": open(rev_file).read().strip() if os.path.exists(rev_file) else "unknown",
              "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__,
              "torch": torch.__version__, "text": TEXT, "items": []}
    for path in IMAGES:
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        for prompt, text in [("", ""), ("query", ""), ("", TEXT)]:
            item = {"image": img} if not text else {"image": img, "text": text}
            emb = m.encode([item], prompt_name=prompt or None, convert_to_numpy=True)[0]
            ids = cap["ids"][0].tolist()
            n_soft = ids.count(inner.config.image_token_id)
            golden["items"].append({"image": path, "prompt": prompt, "text": text, "ids": ids, "n_soft": n_soft,
                                    "embedding": [float(x) for x in emb]})
            print(f"{name:28s} prompt={prompt or '-':6s} text={'yes' if text else 'no ':3s} {len(ids)} ids, {n_soft} soft tokens",
                  file=sys.stderr)
            if prompt == "" and text == "":
                pv, pos = cap["pv"][0], cap["pos"][0]
                valid = (pos[:, 0] >= 0)
                pv_v, pos_v = pv[valid].float().numpy(), pos[valid].numpy().astype(np.int32)
                with torch.no_grad():
                    feats = inner.get_image_features(cap["pv"], cap["pos"], return_dict=True).pooler_output[0]
                feats = feats.float().numpy()
                pv_v.astype("<f4").tofile(os.path.join(art, name + ".patches.f32"))
                pos_v.astype("<i4").tofile(os.path.join(art, name + ".pos.i32"))
                feats.astype("<f4").tofile(os.path.join(art, name + ".feats.f32"))
                print(f"  {name}: {pv_v.shape[0]} patches, features {feats.shape}", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(a.golden)), exist_ok=True)
    with open(a.golden, "w") as f:
        json.dump(golden, f)


if __name__ == "__main__":
    main()

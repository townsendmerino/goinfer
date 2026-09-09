#!/usr/bin/env python
"""Real-model parity golden for Qwen2.5-VL (Qwen/Qwen2.5-VL-3B-Instruct) — the T3 promotion
of the qwen2_5_vl family from tiny-golden to a released checkpoint. This is the first family
in the finishing pass whose ORACLE SHAPE differs: the existing tiny golden
(pin_qwen25vl_image.py) is already e2e encoder->decoder, on SYNTHETIC pixel_values (no real
image, no real processor). A real oracle needs a real image run through the real
AutoImageProcessor, not just real decoder weights — mirroring what
scripts/pin_qwen25vl_preprocess.py already pins for preprocessing ALONE, now carried all the
way to logits on real weights.

REUSES THE EXISTING PRE-SIZED TEST IMAGE (testdata/qwen25vl_preprocess_image.png, 84x56 —
already grid-aligned so the processor's smart_resize is a no-op) rather than a new photo:
resize/bicubic parity is a SEPARATE, already-pinned concern
(qwen25vl_preprocess_golden.json); this gate isolates the decoder-on-real-weights question,
the same discipline mistral3/ministral3's own gates use to keep one gate testing one thing.

    ~/.venv-vl/bin/python scripts/pin_qwen25vl_real.py
    -> testdata/qwen25vl_real_golden.json   (committed; the ~7 GB weights are NOT)

Put the checkpoint at ~/models/qwen25vl-3b-instruct (Qwen/Qwen2.5-VL-3B-Instruct), or set
GOINFER_QWEN25VL_3B to its path.
"""
import io
import json
import os

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

CKPT = os.environ.get("GOINFER_QWEN25VL_3B", os.path.expanduser("~/models/qwen25vl-3b-instruct"))
HERE = os.path.dirname(__file__)
IMG = os.path.join(HERE, "..", "testdata", "qwen25vl_preprocess_image.png")
OUT = os.path.join(HERE, "..", "testdata", "qwen25vl_real_golden.json")
N_NEW = 8


def main():
    from PIL import Image

    proc = AutoProcessor.from_pretrained(CKPT)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(CKPT, dtype=torch.float32).eval()
    cfg = model.config
    image_token = cfg.image_token_id
    vision_start = cfg.vision_start_token_id
    merge = cfg.vision_config.spatial_merge_size
    print(f"loaded {type(model).__name__} model_type={cfg.model_type} "
          f"image_token_id={image_token} vision_start_token_id={vision_start} merge={merge}")

    with open(IMG, "rb") as f:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    img_out = proc.image_processor(images=img, return_tensors="pt")
    pixel_values = img_out["pixel_values"]
    grid_thw = img_out["image_grid_thw"]
    t, h, w = [int(x) for x in grid_thw[0].tolist()]
    n_img_tok = (t * h * w) // (merge * merge)
    print(f"image {img.size} -> grid_thw={[t, h, w]} n_img_tok={n_img_tok} "
          f"pixel_values_shape={list(pixel_values.shape)}")

    tok = proc.tokenizer
    prefix = tok.encode("Describe this image:", add_special_tokens=False)
    suffix = tok.encode(" The image shows", add_special_tokens=False)
    input_ids = prefix + [vision_start] + [image_token] * n_img_tok + suffix
    img_start = len(prefix) + 1
    ids = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        last = model(input_ids=ids, pixel_values=pixel_values, image_grid_thw=grid_thw,
                      use_cache=False).logits[0, -1].float().tolist()
        cur = list(input_ids)
        cont = []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur]), pixel_values=pixel_values,
                       image_grid_thw=grid_thw, use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = dict(
        note="Qwen2.5-VL-3B-Instruct real image->logits (T3), pre-sized test PNG (resize no-op)",
        input_ids=input_ids,
        image_token_id=image_token, vision_start_token_id=vision_start,
        image_token_start=img_start, n_image_tokens=n_img_tok,
        grid_thw=[[t, h, w]],
        pixel_values_shape=list(pixel_values.shape),
        pixel_values=pixel_values.reshape(-1).float().tolist(),
        argmax=int(torch.tensor(last).argmax()),
        last_logits=last,
        n_new=N_NEW,
        continuation_ids=cont,
    )
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print("argmax", golden["argmax"], "cont", cont, "->", repr(tok.decode(cont)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

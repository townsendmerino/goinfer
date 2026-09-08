#!/usr/bin/env python
"""Real-model parity golden for Gemma 3 (google/gemma-3-4b-it) — the gap-0 real-checkpoint gate
GenerateVL's resident decode path currently lacks (docs/multimodal.md gap 0's own note: "no
real-image end-to-end gate yet"). Mirrors pin_qwen25vl_real.py's shape for the other VL family.

REUSES A PRE-SIZED TEST IMAGE (testdata/gemma3_preprocess_image.png, 896x896 exactly — SigLIP's
own input size for Gemma 3, vision.Gemma3() in aikit) so neither HF's processor nor goinfer's own
vision.Preprocess needs to resize it: goinfer's resize is documented bilinear (NOT pixel-exact
vs PIL's bicubic — aikit/vision/preprocess.go's own doc comment), so an already-896x896 source
makes that mismatch a non-issue for this gate, the same discipline the Qwen2.5-VL pin already
uses. Generated once via:
    python3 -c "from PIL import Image; Image.open('testdata/qwen25vl_preprocess_image.png') \
      .convert('RGB').resize((896,896), Image.BICUBIC).save('testdata/gemma3_preprocess_image.png')"

    ~/.venv-vl/bin/python scripts/pin_gemma3_real.py
    -> testdata/gemma3_real_golden.json   (committed; the weights are NOT)

Put the checkpoint at ~/models/gemma-3-4b-it (google/gemma-3-4b-it), or set GEMMA3_4B.
"""
import io
import json
import os

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

CKPT = os.environ.get("GEMMA3_4B", os.path.expanduser("~/models/gemma-3-4b-it"))
HERE = os.path.dirname(__file__)
IMG = os.path.join(HERE, "..", "testdata", "gemma3_preprocess_image.png")
OUT = os.path.join(HERE, "..", "testdata", "gemma3_real_golden.json")
N_NEW = 8


def main():
    from PIL import Image

    proc = AutoProcessor.from_pretrained(CKPT)
    model = Gemma3ForConditionalGeneration.from_pretrained(CKPT, dtype=torch.float32).eval()
    cfg = model.config
    image_token = cfg.image_token_id
    mm_tokens = cfg.mm_tokens_per_image
    print(f"loaded {type(model).__name__} model_type={cfg.model_type} "
          f"image_token_id={image_token} mm_tokens_per_image={mm_tokens}")

    with open(IMG, "rb") as f:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    assert img.size == (896, 896), f"expected a pre-sized 896x896 image, got {img.size}"
    img_out = proc.image_processor(images=img, return_tensors="pt")
    pixel_values = img_out["pixel_values"]
    print(f"image {img.size} -> pixel_values_shape={list(pixel_values.shape)}")

    tok = proc.tokenizer
    boi = tok.convert_tokens_to_ids("<start_of_image>")
    eoi = tok.convert_tokens_to_ids("<end_of_image>")
    prefix = tok.encode("Describe this image:", add_special_tokens=True)
    suffix = tok.encode(" The image shows", add_special_tokens=False)
    input_ids = prefix + [boi] + [image_token] * mm_tokens + [eoi] + suffix
    img_start = len(prefix) + 1
    ids = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        last = model(input_ids=ids, pixel_values=pixel_values, use_cache=False).logits[0, -1].float().tolist()
        cur = list(input_ids)
        cont = []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur]), pixel_values=pixel_values, use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = dict(
        note="gemma-3-4b-it real image->logits (gap 0), pre-sized 896x896 test PNG (resize no-op)",
        input_ids=input_ids,
        image_token_id=image_token,
        image_token_start=img_start,
        mm_tokens_per_image=mm_tokens,
        pixel_values=pixel_values.flatten().tolist(),
        pixel_values_shape=list(pixel_values.shape),
        argmax=int(torch.tensor(last).argmax()),
        last_logits=last,
        continuation_ids=cont,
    )
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}  argmax={golden['argmax']}  continuation={cont}")


if __name__ == "__main__":
    main()

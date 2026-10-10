#!/usr/bin/env python3
"""S10's real-checkpoint references for Ministral 3 (Pixtral), G-S10m-a/b/c (docs/tasks/task-multimodal-support-2026-10.md,
"S10, Ministral 3 (Pixtral)"): transformers' Mistral3ForConditionalGeneration in float32 (sdpa), loaded once.

  a. The processor (PixtralImageProcessorFast, the checkpoint's processor_config.json) on the four F2a images and two
     made-up layout cases written here (a 1000x20 strip, one merged row; a 3000x2000 image, downscaled): the pixel
     values cropped to each image's size, the size, and the [IMG]/[IMG_BREAK]/[IMG_END] counts the full processor
     puts in the ids.
  b. On the four images, from the processor's own pixels: the tower's stages (patch conv, ln_pre, each block) and the
     projector's (norm, patch merger, linear_1, linear_2), per image; and two images in one call, each image's
     projector output.
  c. The full model on goinfer's own ids (<out>/ids.json, written first by decoder/pixtral_real_test.go's "ids" step):
     the last position's logits and every position's argmax.

Everything goes to <out> as float32 little-endian files, named in <out>/golden.json; the Go side is
multimodal/pixtral_real_test.go (a, b) and decoder/pixtral_real_test.go (c). About 17 GB in float32: nobara.

Run (nobara, ~/.venv-vl): python3 scripts/pin_pixtral_real.py --model ~/models/ministral3-3b-bf16 --images testdata \
         --out ~/goinfer-logs/pixtral-real
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

IMAGES = ["gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"]
IMG, BRK, END = 10, 12, 13


def made_up(out):
    """The two layout cases, deterministic, saved as PNG so both sides read the same bytes."""
    r = np.random.default_rng(5)
    strip = (np.linspace(0, 255, 1000)[None, :, None] * np.ones((20, 1, 3)) + r.normal(0, 20, (20, 1000, 3)))
    big = np.zeros((2000, 3000, 3))
    big[..., 0] = np.linspace(0, 255, 3000)[None, :]
    big[..., 1] = np.linspace(0, 255, 2000)[:, None]
    big[..., 2] = 128 + r.normal(0, 40, (2000, 3000))
    names = []
    for name, a in (("strip_1000x20.png", strip), ("big_3000x2000.png", big)):
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB").save(os.path.join(out, name))
        names.append(name)
    return names


def processed(proc, images):
    """The full PixtralProcessor on [IMG] per image: its pixel values and sizes are the model's (it passes the image
    processor patch_size x spatial_merge_size; the image processor called alone rounds to the patch, 14, and the merged
    grid then disagrees with the ids)."""
    enc = proc(text="[IMG]" * len(images), images=images, return_tensors="pt")
    f = proc.patch_size * proc.spatial_merge_size
    for h, w in enc["image_sizes"].tolist():
        if h % f or w % f:
            sys.exit(f"processor size {h}x{w} is not a multiple of {f}")
    return enc


def save(out, name, t):
    t.detach().float().contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".f32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_dir = os.path.expanduser(a.model)
    if model_dir.startswith("/Volumes/") or model_dir.startswith("/srv/models"):
        sys.exit(f"{model_dir} is on the archive, not the bench set (CLAUDE.md)")
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", file=sys.stderr, flush=True)

    torch.set_num_threads(max(1, os.cpu_count() - 2))
    proc = AutoProcessor.from_pretrained(model_dir)
    model = Mistral3ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    log(f"loaded ({type(proc.image_processor).__name__}, transformers {transformers.__version__})")
    cfg = model.config
    if cfg.image_token_index != IMG or cfg.vision_feature_layer != -1:
        sys.exit(f"image token {cfg.image_token_index}, feature layer {cfg.vision_feature_layer}: not what the Go side assumes")
    golden = {"transformers": transformers.__version__, "torch": torch.__version__,
              "processor": type(proc.image_processor).__name__, "attn": "sdpa", "a": [], "b": [], "two": None, "c": None}
    tower, mp = model.model.vision_tower, model.model.multi_modal_projector

    # a: the processor.
    files = [(img, os.path.join(a.images, img)) for img in IMAGES] + [(n, os.path.join(out, n)) for n in made_up(out)]
    pil = {}
    for img, path in files:
        name = img.replace("/", "_").removesuffix(".png")
        im = Image.open(path).convert("RGB")
        pil[img] = im
        enc = processed(proc, [im])
        h, w = (int(x) for x in enc["image_sizes"][0])
        save(out, name + ".pixels", enc["pixel_values"][0, :, :h, :w])
        ids = enc["input_ids"][0].tolist()
        golden["a"].append({"image": img, "name": name, "src": [im.height, im.width], "size": [h, w],
                            "img": ids.count(IMG), "brk": ids.count(BRK), "end": ids.count(END)})
        log(f"a: {img} {im.width}x{im.height} -> {w}x{h}, {ids.count(IMG)} [IMG]")

    # b: every stage of the tower and the projector, from the processor's own pixels.
    for img in IMAGES:
        name = img.replace("/", "_").removesuffix(".png")
        enc = processed(proc, [pil[img]])
        st = {}
        hooks = [tower.patch_conv.register_forward_hook(lambda m, i, o: st.__setitem__("conv", o[0].flatten(1).T.clone())),
                 tower.ln_pre.register_forward_hook(lambda m, i, o: st.__setitem__("lnpre", o[0].clone()))]
        for L, blk in enumerate(tower.transformer.layers):
            hooks.append(blk.register_forward_hook(
                lambda m, i, o, L=L: st.__setitem__(f"block{L}", (o[0] if isinstance(o, tuple) else o)[0].clone())))
        for k in ("norm", "patch_merger", "linear_1", "linear_2"):
            hooks.append(getattr(mp, k).register_forward_hook(lambda m, i, o, k=k: st.__setitem__("proj_" + k, o.clone())))
        with torch.no_grad():
            feats = model.model.get_image_features(pixel_values=enc["pixel_values"], image_sizes=enc["image_sizes"],
                                                   vision_feature_layer=-1).pooler_output[0]
        for hk in hooks:
            hk.remove()
        if not torch.equal(feats, st["proj_linear_2"].reshape(feats.shape)):
            sys.exit("the projector's linear_2 hook is not the image features")
        for k, v in st.items():
            save(out, f"{name}.{k}", v)
        h, w = (int(x) for x in enc["image_sizes"][0])
        golden["b"].append({"image": img, "name": name, "size": [h, w], "stages": list(st), "units": feats.shape[0]})
        log(f"b: {img} {h // 14}x{w // 14} patches, {feats.shape[0]} units, {len(st)} stages")

    # b: two images in one call (block-diagonal mask), each image's projector output.
    pair = [IMAGES[3], IMAGES[0]]
    enc = processed(proc, [pil[p] for p in pair])
    with torch.no_grad():
        fs = model.model.get_image_features(pixel_values=enc["pixel_values"], image_sizes=enc["image_sizes"],
                                            vision_feature_layer=-1).pooler_output
    for k, f in enumerate(fs):
        save(out, f"two.{k}", f)
    golden["two"] = {"images": pair, "sizes": enc["image_sizes"].tolist()}
    log("b: two images in one call")

    # c: the full model on goinfer's ids.
    idsf = os.path.join(out, "ids.json")
    if not os.path.exists(idsf):
        log("c: no ids.json (run decoder/pixtral_real_test.go's ids step first); skipped")
    else:
        ids_in = json.load(open(idsf))
        ids = ids_in["ids"]
        enc = processed(proc, [pil[ids_in["image"]]])
        with torch.no_grad():
            n = model.model.get_image_features(pixel_values=enc["pixel_values"], image_sizes=enc["image_sizes"],
                                               vision_feature_layer=-1).pooler_output[0].shape[0]
            if ids.count(IMG) != n:
                sys.exit(f"{ids.count(IMG)} [IMG] in goinfer's ids, the tower gives {n} rows")
            o = model(input_ids=torch.tensor([ids]), pixel_values=enc["pixel_values"], image_sizes=enc["image_sizes"])
        logits = o.logits[0].float()
        save(out, "hf_last_logits", logits[-1])
        golden["c"] = {"image": ids_in["image"], "ids": len(ids), "img": n, "argmax": [int(x) for x in logits.argmax(-1)]}
        log(f"c: {len(ids)} ids, {n} image rows, last argmax {int(logits[-1].argmax())}")
    json.dump(golden, open(os.path.join(out, "golden.json"), "w"))
    log("wrote " + os.path.join(out, "golden.json"))


if __name__ == "__main__":
    main()

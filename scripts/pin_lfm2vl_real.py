#!/usr/bin/env python3
"""S10's real-checkpoint references for LFM2.5-VL, G-S10l-a/b/c (docs/tasks/task-multimodal-support-2026-10.md, "S10,
LFM2.5-VL"): transformers' Lfm2VlForConditionalGeneration in float32 (sdpa), loaded once.

  a. The full Lfm2VlProcessor on the four F2a images and three layout cases written here (a 1000x20 strip, a 3000x2000
     image, an 84x56 image): each tile's patch rows (the processor's own pixel_values, unpadded), its patch grid, and the
     image's <image> runs in the ids.
  b. On the four images, from the processor's own patches: each tile (and thumbnail) through the tower alone, every
     stage (embeddings, each block, post_layernorm), then the projector's (the pixel unshuffle, linear_1 before the GELU,
     linear_2). HF batches tiles with a key mask; alone is the same rows (the tiny pin asserts it to 7e-7).
  c. The full model on goinfer's own ids (<out>/ids.json, from decoder/lfm2vl_real_test.go's "ids" step): the last
     position's logits and every position's argmax.

Everything goes to <out> as float32 little-endian files named in <out>/golden.json; the Go side is
multimodal/lfm2vl_real_test.go (a, b) and decoder/lfm2vl_real_test.go (c).

Run (nobara, ~/.venv-vl): python3 scripts/pin_lfm2vl_real.py --model ~/models/lfm25-vl-1.6b --images testdata \
         --out ~/goinfer-logs/lfm2vl-real
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
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration

IMAGES = ["gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"]


def made_up(out):
    """The three layout cases, deterministic, saved as PNG so both sides read the same bytes."""
    r = np.random.default_rng(5)
    strip = np.linspace(0, 255, 1000)[None, :, None] * np.ones((20, 1, 3)) + r.normal(0, 20, (20, 1000, 3))
    big = np.zeros((2000, 3000, 3))
    big[..., 0] = np.linspace(0, 255, 3000)[None, :]
    big[..., 1] = np.linspace(0, 255, 2000)[:, None]
    big[..., 2] = 128 + r.normal(0, 40, (2000, 3000))
    small = r.integers(0, 256, (56, 84, 3)).astype(float)
    names = []
    for name, a in (("strip_1000x20.png", strip), ("big_3000x2000.png", big), ("small_84x56.png", small)):
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB").save(os.path.join(out, name))
        names.append(name)
    return names


def save(out, name, t):
    t.detach().float().contiguous().numpy().astype("<f4").tofile(os.path.join(out, name + ".f32"))


def runs_of(ids, tok):
    runs, n = [], 0
    for t in list(ids) + [-1]:
        if t == tok:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    return runs


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
    model = Lfm2VlForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    img_tok = model.config.image_token_id
    log(f"loaded ({type(proc.image_processor).__name__}, transformers {transformers.__version__}, image token {img_tok})")
    golden = {"transformers": transformers.__version__, "torch": torch.__version__,
              "processor": type(proc.image_processor).__name__, "attn": "sdpa", "a": [], "b": [], "c": None}
    tower, mp = model.model.vision_tower, model.model.multi_modal_projector

    def processed(images):
        return proc(text="<image>" * len(images), images=images, return_tensors="pt")

    # a: the processor.
    files = [(img, os.path.join(a.images, img)) for img in IMAGES] + [(n, os.path.join(out, n)) for n in made_up(out)]
    pil = {}
    for img, path in files:
        name = img.replace("/", "_").removesuffix(".png")
        im = Image.open(path).convert("RGB")
        pil[img] = im
        enc = processed([im])
        shapes = enc["spatial_shapes"].tolist()
        for k, (gh, gw) in enumerate(shapes):
            save(out, f"{name}.tile{k}.patches", enc["pixel_values"][k, :gh * gw])
        golden["a"].append({"image": img, "name": name, "src": [im.height, im.width], "tiles": shapes,
                            "runs": runs_of(enc["input_ids"][0].tolist(), img_tok)})
        log(f"a: {img} {im.width}x{im.height} -> {len(shapes)} tile(s) {shapes[-1]}, runs {golden['a'][-1]['runs'][:3]}...")

    # b: every stage of the tower and the projector, tile by tile, from the processor's own patches.
    for img in IMAGES:
        name = img.replace("/", "_").removesuffix(".png")
        enc = processed([pil[img]])
        shapes = enc["spatial_shapes"].tolist()
        stage_names = None
        for k, (gh, gw) in enumerate(shapes):
            n = gh * gw
            st = {}
            hooks = [tower.embeddings.register_forward_hook(lambda m, i, o: st.__setitem__("embeddings", o[0].clone()))]
            for L, blk in enumerate(tower.encoder.layers):
                hooks.append(blk.register_forward_hook(
                    lambda m, i, o, L=L: st.__setitem__(f"block{L}", (o[0] if isinstance(o, tuple) else o)[0].clone())))
            with torch.no_grad():
                hs = tower(pixel_values=enc["pixel_values"][k:k + 1, :n], pixel_attention_mask=torch.ones(1, n, dtype=torch.int32),
                           spatial_shapes=torch.tensor([[gh, gw]])).last_hidden_state[0]
                for hk in hooks:
                    hk.remove()
                st["post_layernorm"] = hs
                un = mp.pixel_unshuffle(hs.reshape(1, gh, gw, -1))
                st["proj_unshuffle"] = un.reshape(-1, un.shape[-1])
                l1 = mp.linear_1(un)
                st["proj_linear_1"] = l1.reshape(-1, l1.shape[-1])
                l2 = mp.linear_2(mp.act(l1))
                st["proj_linear_2"] = l2.reshape(-1, l2.shape[-1])
                # the model's own feature path for this tile must be the stage chain's end
                feat = model.model.get_image_features(pixel_values=enc["pixel_values"][k:k + 1, :n],
                                                      spatial_shapes=torch.tensor([[gh, gw]]),
                                                      pixel_attention_mask=torch.ones(1, n, dtype=torch.int32)).pooler_output[0]
                if not torch.allclose(feat, st["proj_linear_2"], atol=1e-5):
                    sys.exit(f"{img} tile {k}: the stage chain is not get_image_features")
            for s, v in st.items():
                save(out, f"{name}.tile{k}.{s}", v)
            stage_names = list(st)
        golden["b"].append({"image": img, "name": name, "tiles": shapes, "stages": stage_names})
        log(f"b: {img} {len(shapes)} tile(s), {len(stage_names)} stages each")

    # c: the full model on goinfer's ids.
    idsf = os.path.join(out, "ids.json")
    if not os.path.exists(idsf):
        log("c: no ids.json (run decoder/lfm2vl_real_test.go's ids step first); skipped")
    else:
        ids_in = json.load(open(idsf))
        ids = ids_in["ids"]
        enc = processed([pil[ids_in["image"]]])
        hf_runs = runs_of(enc["input_ids"][0].tolist(), img_tok)
        if runs_of(ids, img_tok) != hf_runs:
            sys.exit(f"goinfer's <image> runs {runs_of(ids, img_tok)} are not the processor's {hf_runs}")
        with torch.no_grad():
            o = model(input_ids=torch.tensor([ids]), pixel_values=enc["pixel_values"],
                      pixel_attention_mask=enc["pixel_attention_mask"], spatial_shapes=enc["spatial_shapes"])
        logits = o.logits[0].float()
        save(out, "hf_last_logits", logits[-1])
        hf_prompt = proc.apply_chat_template([{"role": "system", "content": ids_in["system"]},
                                              {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ids_in["text"]}]}],
                                             add_generation_prompt=True, tokenize=False)
        hf_ids = proc(text=hf_prompt, images=[pil[ids_in["image"]]], return_tensors="pt")["input_ids"][0].tolist()
        golden["c"] = {"image": ids_in["image"], "ids": len(ids), "runs": hf_runs, "argmax": [int(x) for x in logits.argmax(-1)],
                       "hf_template_ids_equal": hf_ids == ids, "hf_template_ids": len(hf_ids)}
        log(f"c: {len(ids)} ids (HF's own template: {len(hf_ids)}, equal {hf_ids == ids}), last argmax {int(logits[-1].argmax())}")
    json.dump(golden, open(os.path.join(out, "golden.json"), "w"))
    log("wrote " + os.path.join(out, "golden.json"))


if __name__ == "__main__":
    main()

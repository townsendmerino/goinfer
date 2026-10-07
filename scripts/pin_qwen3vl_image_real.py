#!/usr/bin/env python3
"""G-S10c's reference (docs/tasks/task-multimodal-support-2026-10.md, S10): transformers' Qwen3VLForConditionalGeneration,
float32, on one image prompt (testdata/glm_ocr/table.png, "What does this image show? Answer briefly."), its own
processor and chat template making the inputs. Writes, to <out>: inputs.json (input_ids, image_grid_thw, the image token
id), pixels.f32 (HF's pixel values, float32 little-endian), hf_last_logits.f32, and hf_argmax.json (the argmax at every
position). decoder/qwen3vl_image_real_test.go reads them, so both sides see identical inputs. About 8 GB in float32.

Run (nobara, ~/.venv-vl): python3 scripts/pin_qwen3vl_image_real.py --model ~/models/qwen3-vl-2b-instruct --out <dir>
"""
import argparse
import json
import os
import sys

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default="testdata/glm_ocr/table.png")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_dir = os.path.expanduser(a.model)
    if model_dir.startswith("/Volumes/") or model_dir.startswith("/srv/models"):
        sys.exit(f"{model_dir} is on the archive (CLAUDE.md)")
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    proc = AutoProcessor.from_pretrained(model_dir)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What does this image show? Answer briefly."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], images=[Image.open(a.image).convert("RGB")], return_tensors="pt")
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    with torch.no_grad():
        o = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], pixel_values=enc["pixel_values"].float(),
                  image_grid_thw=enc["image_grid_thw"])
    logits = o.logits[0].float()
    ids = enc["input_ids"][0].tolist()
    itok = model.config.image_token_id
    json.dump({"input_ids": ids, "image_grid_thw": enc["image_grid_thw"][0].tolist(), "image_token_id": itok,
               "spatial_merge_size": model.config.vision_config.spatial_merge_size, "transformers": transformers.__version__,
               "torch": torch.__version__}, open(os.path.join(out, "inputs.json"), "w"))
    enc["pixel_values"].float().numpy().astype("<f4").tofile(os.path.join(out, "pixels.f32"))
    logits[-1].numpy().astype("<f4").tofile(os.path.join(out, "hf_last_logits.f32"))
    json.dump([int(x) for x in logits.argmax(-1)], open(os.path.join(out, "hf_argmax.json"), "w"))
    print(f"{len(ids)} ids ({ids.count(itok)} image tokens), grid {enc['image_grid_thw'][0].tolist()}, last argmax {int(logits[-1].argmax())}", file=sys.stderr)


if __name__ == "__main__":
    main()

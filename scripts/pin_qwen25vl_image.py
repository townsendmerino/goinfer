#!/usr/bin/env python
"""Pin the IMAGE->logits path of the tiny Qwen2.5-VL checkpoint — the P5 end-to-end
gate (ViT + patch merger + embedding interleaving + m-RoPE 3D positions).

Loads the SAME tiny checkpoint pin_qwen25vl_tiny.py saved (testdata/qwen25vl-tiny),
builds an input with a <vision_start> + <image>*N run + pixel_values + grid_thw, and
dumps stage-isolated goldens:
  - image_features: the visual model output (ViT -> windowed attn -> merger), the
    [n_img_tokens, text_hidden] block that replaces the placeholders,
  - position_ids: the m-RoPE 3D positions (the new structural piece vs Gemma 3) —
    so goinfer's get_rope_index port is gated in isolation, and
  - last_logits: the end-to-end logits after interleaving + m-RoPE.
goinfer gates the ViT/merger on image_features, the position computation on
position_ids, and the full path on last_logits.

Unlike Gemma 3 (fixed grid, bidirectional image-block mask, scalar RoPE),
Qwen2.5-VL uses dynamic patch grids + m-RoPE; the placeholder run length is
grid_thw-derived (n_patches / spatial_merge^2), not a fixed constant.

Tiny -> sub-second; no download.

    ~/.venv-vl/bin/python scripts/pin_qwen25vl_image.py
    -> testdata/qwen25vl_tiny_image_golden.json
"""
import json
import os

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

HERE = os.path.dirname(__file__)
CKPT = os.path.join(HERE, "..", "testdata", "qwen25vl-tiny")
OUT = os.path.join(HERE, "..", "testdata", "qwen25vl_tiny_image_golden.json")


def main():
    if not os.path.isdir(CKPT):
        raise SystemExit(f"{CKPT} missing — run scripts/pin_qwen25vl_tiny.py first")
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(CKPT, torch_dtype=torch.float32).eval()
    vcfg = model.config.vision_config
    image_token = model.config.image_token_id
    vision_start = model.config.vision_start_token_id
    merge = vcfg.spatial_merge_size

    # One image, grid (t,h,w) in patch units (h,w multiples of merge for the merger).
    t, h, w = 1, 4, 6
    n_patches = t * h * w
    patch_dim = vcfg.in_chans * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
    n_img_tok = n_patches // (merge * merge)

    gen = torch.Generator().manual_seed(1)
    pixel_values = torch.randn(n_patches, patch_dim, generator=gen, dtype=torch.float32)
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long)

    prefix = [2, 7, 42]
    suffix = [13, 88, 5]
    input_ids = prefix + [vision_start] + [image_token] * n_img_tok + suffix
    img_start = len(prefix) + 1  # position of the first image token (after vision_start)
    ids = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        # Stage 1: the visual model. get_image_features returns BOTH stages —
        # last_hidden_state [n_patches, vision_hidden] (ViT pre-merge, gates the
        # encoder) and pooler_output[0] [n_img_tok, out_hidden] (after the patch
        # merger — the embeds that replace the placeholders, gates the merger+scatter).
        vis_out = model.model.get_image_features(pixel_values, grid_thw)
        vit_hidden = vis_out.last_hidden_state           # [n_patches, vision_hidden]
        image_features = vis_out.pooler_output[0]        # [n_img_tok, out_hidden]
        # Stage 2: the m-RoPE 3D position ids the decoder will use. mm_token_type_ids
        # marks image-token positions (1) vs text (0) — the processor normally supplies
        # it; build it here from the placeholder run.
        mm_token_type_ids = (ids == image_token).int()
        position_ids, mrope_delta = model.model.get_rope_index(ids, mm_token_type_ids, image_grid_thw=grid_thw)
        # Stage 3: end-to-end logits (interleave + m-RoPE attention).
        out = model(input_ids=ids, pixel_values=pixel_values, image_grid_thw=grid_thw, use_cache=False)
        last_logits = out.logits[0, -1].float().tolist()
        # Stage 4: greedy continuation past the image prompt — gates DECODE-time
        # m-RoPE (text positions resume at max+1, i.e. seqPos + mrope_delta). Recompute
        # the full forward each step (no cache); pixel_values stay at the image run.
        N_CONT = 4
        cur, cont = list(input_ids), []
        for _ in range(N_CONT):
            o = model(input_ids=torch.tensor([cur]), pixel_values=pixel_values,
                      image_grid_thw=grid_thw, use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = {
        "note": "tiny Qwen2.5-VL image->logits; CPU fp32 (P5 end-to-end gate)",
        "input_ids": input_ids,
        "image_token_id": image_token, "vision_start_token_id": vision_start,
        "image_token_start": img_start, "n_image_tokens": n_img_tok,
        "grid_thw": [[t, h, w]],
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values": pixel_values.flatten().tolist(),
        "vit_hidden_shape": list(vit_hidden.shape),                     # ViT pre-merge (encoder gate)
        "vit_hidden": vit_hidden.reshape(-1).float().tolist(),
        "image_features_shape": list(image_features.shape),
        "image_features": image_features.reshape(-1).float().tolist(),  # merged embeds (merger+scatter gate)
        "position_ids_shape": list(position_ids.shape),                  # [3, 1, seq]
        "position_ids": position_ids.reshape(3, -1).long().tolist(),     # m-RoPE 3D positions (isolation gate)
        "mrope_position_delta": int(mrope_delta.reshape(-1)[0]),
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,                                      # end-to-end (final gate)
        "continuation_ids": cont,                                        # greedy decode past the image (decode m-RoPE gate)
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}")
    print(f"  image_features_shape={golden['image_features_shape']}  n_img_tok={n_img_tok}")
    print(f"  position_ids_shape={golden['position_ids_shape']}  argmax={golden['argmax']}")


if __name__ == "__main__":
    main()

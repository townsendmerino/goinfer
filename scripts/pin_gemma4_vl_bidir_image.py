#!/usr/bin/env python
"""Pin the IMAGE->logits path of the tiny bidirectional Gemma 4 VL checkpoint
(scripts/pin_gemma4_vl_bidir_tiny.py) — the real gate for
decoder/forward_gemma4_batched.go's batched, blockwise-masked forward.

Loads the SAME checkpoint pin_gemma4_vl_bidir_tiny.py saved
(testdata/gemma4-vl-bidir-tiny) and builds an image block DELIBERATELY LONGER
than sliding_window=3 and positioned so the block's OWN start falls outside
the causal window of the block's own last position:

    prefix=[2,7,42] (3 tokens) -> block [3,12) (9 tokens, grid=9/pool_k=3) -> suffix.
    Last block position pos=11: windowStart(11) = 11-3+1 = 9 > block start (3).

This is what makes the gate non-vacuous: under a causal-only forward, an early
block token (say position 4) can NEVER be seen by a later block token (11) —
no amount of window tuning fixes that, since causal masks never grant future
visibility at all. And under the generic family's own existing
attendHi-style single-range approximation (decoder/kvcache.go, NOT reused here
— see gemma4AttendRange's doc comment for why), the window would never get
widened back to the block's start, so positions 3..8 would still be wrongly
excluded from position 11's attention on the two sliding_attention layers.
Both distinguishing behaviors are exercised, on both layer types (full and
sliding are both present in this fixture's layer_types).

mm_token_type_ids must be constructed and passed explicitly — confirmed this
session by reading modeling_gemma4.py: unlike pixel_values, it is NEVER
auto-derived from input_ids/image_token_id anywhere in Gemma4Model.forward,
it is purely a caller-supplied tensor (None by default, which silently
degrades block_sequence_ids to all -1, i.e. plain causal even with a real
image) — passing it is required for this checkpoint's real mechanism to fire.

Tiny -> sub-second; no download.

    ~/.venv-vl/bin/python scripts/pin_gemma4_vl_bidir_image.py
    -> testdata/gemma4_vl_bidir_tiny_image_golden.json
"""
import json
import os

import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

HERE = os.path.dirname(__file__)
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-vl-bidir-tiny")
OUT = os.path.join(HERE, "..", "testdata", "gemma4_vl_bidir_tiny_image_golden.json")
IMAGE_TOKEN_ID = 250  # within the tiny vocab (256)


def main():
    if not os.path.isdir(CKPT):
        raise SystemExit(f"{CKPT} missing — run scripts/pin_gemma4_vl_bidir_tiny.py first")
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration.from_pretrained(CKPT, dtype=torch.float32)
    model.eval()
    model.config.image_token_id = IMAGE_TOKEN_ID
    model.config.text_config.image_token_id = IMAGE_TOKEN_ID

    vcfg = model.config.vision_config
    patch_size, pool_k = vcfg.patch_size, vcfg.pooling_kernel_size
    grid = 9  # patches per side (multiple of pool_k=3) -> pools to 3x3=9 soft tokens
    n_image_tokens = (grid // pool_k) * (grid // pool_k)

    prefix = [2, 7, 42]
    suffix = [13, 88, 5]
    input_ids = prefix + [IMAGE_TOKEN_ID] * n_image_tokens + suffix
    img_start = len(prefix)
    seq_len = len(input_ids)

    mm_token_type_ids = torch.zeros(1, seq_len, dtype=torch.long)
    mm_token_type_ids[0, img_start:img_start + n_image_tokens] = 1

    gen = torch.Generator().manual_seed(1)
    num_patches = grid * grid
    patch_dim = 3 * patch_size * patch_size
    pixel_values = torch.rand(1, num_patches, patch_dim, generator=gen, dtype=torch.float32)
    xs, ys = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="xy")
    position_ids = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).unsqueeze(0).long()

    with torch.no_grad():
        # Stage 1: the tower's own pooled+projected output (isolation gate).
        image_features = model.model.get_image_features(pixel_values, position_ids, return_dict=True).pooler_output
        # Stage 2: end-to-end logits (masked_scatter splice + the REAL blockwise mask).
        out = model(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            pixel_values=pixel_values,
            image_position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
        )
        last_logits = out.logits[0, -1].float().tolist()

        # Non-vacuousness proof, computed HERE (not left to the Go side to assert
        # blind): the SAME forward with mm_token_type_ids omitted (block_sequence_ids
        # degrades to all -1, i.e. plain causal) must give a DIFFERENT last_logits —
        # otherwise this fixture's block placement doesn't actually distinguish the
        # two mechanisms and the gate would be vacuous.
        out_causal = model(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            pixel_values=pixel_values,
            image_position_ids=position_ids,
            use_cache=False,
        )
        causal_logits = out_causal.logits[0, -1].float().tolist()

    diff = max(abs(a - b) for a, b in zip(last_logits, causal_logits))
    if diff < 1e-3:
        raise SystemExit(f"fixture is VACUOUS: bidirectional vs causal-only last_logits differ by only {diff} "
                          "— the block placement doesn't distinguish the two mechanisms, widen it")
    print(f"non-vacuousness check: bidirectional vs causal-only last_logits max-abs-diff={diff:.4f} (must be large)")

    img_feat = image_features.reshape(-1).float().tolist()
    golden = {
        "note": "tiny Gemma4 VL image->logits, use_bidirectional_attention='vision'; CPU fp32. "
                "Block [img_start,img_start+n_image_tokens) is longer than sliding_window=3 and "
                "positioned so the block's own start falls outside the last block position's causal "
                "window — the discriminator (see this script's docstring). causal_only_last_logits is "
                "the SAME forward with mm_token_type_ids omitted, included so the Go test can assert "
                "the old sequential (strictly causal) path provably disagrees with this golden.",
        "input_ids": input_ids,
        "image_token_id": IMAGE_TOKEN_ID,
        "image_token_start": img_start,
        "n_image_tokens": n_image_tokens,
        "pixel_values_shape": list(pixel_values.shape),
        "position_ids_shape": list(position_ids.shape),
        "image_features_shape": list(image_features.shape),
        "image_features": img_feat,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,
        "causal_only_argmax": int(torch.tensor(causal_logits).argmax()),
        "causal_only_last_logits": causal_logits,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"wrote {OUT}")
    print(f"  n_image_tokens={n_image_tokens}  image_features_shape={golden['image_features_shape']}  "
          f"argmax={golden['argmax']} (causal-only would be {golden['causal_only_argmax']})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Pin a SCALED bidirectional-block Gemma 4 VL checkpoint — realistic text
geometry, still no download — to exercise the GPU-resident decode bridge
(decoder/generate_gemma4_vl.go's residentUploadPrefill wiring) at a scale the
tiny bidir fixture (scripts/pin_gemma4_vl_bidir_tiny.py, head_dim=16) can't.

This is pin_gemma4_dense_scaled.py's exact TEXT geometry (hidden 1024, 12
layers in the real 5:1 sliding/full interleave, REAL head dims 256 local /
512 global, K=V on the two global/full layers — the geometry
cuda/gemma4_dense_scaled_test.go's own history says a tiny fixture can't
exercise) combined with pin_gemma4_vl_bidir_tiny.py's small VISION config and
use_bidirectional_attention="vision", built as a real
Gemma4ForConditionalGeneration instead of dense_scaled's text-only
Gemma4ForCausalLM.

Two goldens:
  - TEXT (regression, mirrors gemma4_dense_scaled_golden.json's shape):
    proves the VL wrapper's text-only forward still matches dense_scaled's
    own reference geometry once a vision tower is attached but unused.
  - IMAGE (the real gate): a block LONGER than sliding_window=8 and
    positioned so the block's own start falls outside the last block
    position's causal window — pin_gemma4_vl_bidir_image.py's exact
    discriminator, rescaled: prefix=[2,7,42] -> block [3,28) (25 tokens,
    grid=15/pool_k=3) -> suffix. Last block position pos=27:
    windowStart(27) = 27-8+1 = 20 > block start (3).

Tiny -> sub-second; no download.

    ~/.venv-vl/bin/python scripts/pin_gemma4_vl_bidir_scaled.py
    -> testdata/gemma4_vl_bidir_scaled_text_golden.json
    -> testdata/gemma4_vl_bidir_scaled_image_golden.json
    -> testdata/gemma4-vl-bidir-scaled/   (HF safetensors checkpoint: vision + text)
"""
import json
import os

import torch
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

HERE = os.path.dirname(__file__)
CKPT = os.path.join(HERE, "..", "testdata", "gemma4-vl-bidir-scaled")
TEXT_OUT = os.path.join(HERE, "..", "testdata", "gemma4_vl_bidir_scaled_text_golden.json")
IMAGE_OUT = os.path.join(HERE, "..", "testdata", "gemma4_vl_bidir_scaled_image_golden.json")
IMAGE_TOKEN_ID = 250  # within the vocab (256), matching pin_gemma4_vl_bidir_image.py

# 5:1 sliding/full interleave over 12 layers -> two global (full) layers, at indices 5 and 11 —
# IDENTICAL to pin_gemma4_dense_scaled.py's own geometry (the fixture this bridge must agree with
# for plain text once a vision tower is attached).
LAYER_TYPES = (["sliding_attention"] * 5 + ["full_attention"]) * 2

TEXT = dict(
    vocab_size=256, hidden_size=1024, num_hidden_layers=12,
    num_attention_heads=4, num_key_value_heads=2, head_dim=256,  # local q_dim=1024, kv_dim=512
    intermediate_size=2048, rms_norm_eps=1e-6,
    max_position_embeddings=256, sliding_window=8,
    layer_types=LAYER_TYPES,
    hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=30.0,
    hidden_size_per_layer_input=0,  # PLE-free, matches real 26B-A4B
    rope_local_base_freq=10000.0, rope_theta=1000000.0,
    enable_moe_block=False,  # dense-only — masking/resident-bridge is orthogonal to FFN dispatch
    num_kv_shared_layers=0,  # matches real 26B-A4B exactly — required for resident admission
    global_head_dim=512, num_global_key_value_heads=2, attention_k_eq_v=True,  # REAL geometry, K=V
    use_bidirectional_attention="vision",
)
VISION = dict(
    hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=4,
    head_dim=8, patch_size=4, pooling_kernel_size=3, position_embedding_size=16,
    rms_norm_eps=1e-6, use_clipped_linears=True,
    rope_parameters={"rope_theta": 100.0, "rope_type": "default"}, standardize=False,
)
PROMPT = [1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200]  # dense_scaled's own, len 16 > window 8
N_NEW = 6


def strengthen_text(model):
    """Same degeneracy guard as pin_gemma4_dense_scaled.py / pin_gemma4_vl_bidir_tiny.py: HF init
    leaves norms/layer_scalar at identity, so a bug applying them (x1) would not move the golden.
    Separate generator => the linear weights + input stay bit-identical. Vision tower excluded —
    this fixture's gate is about the TEXT-side resident KV bridge, not vision numerics."""
    g = torch.Generator().manual_seed(1234)
    n_norm = n_ls = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "vision_tower" in name or "embed_vision" in name:
                continue
            if name.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or name.endswith("model.norm.weight"):
                p.normal_(1.0, 0.1, generator=g)
                n_norm += 1
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.normal_(1.0, 0.1, generator=g)
                n_ls += 1
    return dict(norm=n_norm, layer_scalar=n_ls)


def build_model():
    torch.manual_seed(0)
    cfg = Gemma4Config(
        text_config=Gemma4TextConfig(**TEXT),
        vision_config=Gemma4VisionConfig(**VISION),
    )
    model = Gemma4ForConditionalGeneration(cfg)
    model.eval().to(torch.float32)
    counts = strengthen_text(model)

    gen = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "embed_vision" in name:
                p.normal_(0.0, 0.02, generator=gen)
        for name, buf in model.named_buffers():
            if name.endswith("_min"):
                buf.copy_(torch.tensor(-2.0 + 0.1 * torch.randn(1, generator=gen).item()))
            elif name.endswith("_max"):
                buf.copy_(torch.tensor(2.0 + 0.1 * torch.randn(1, generator=gen).item()))
    return model, cfg, counts


def main():
    model, cfg, counts = build_model()
    model.config.image_token_id = IMAGE_TOKEN_ID
    model.config.text_config.image_token_id = IMAGE_TOKEN_ID

    # --- TEXT golden (regression: matches dense_scaled's own reference geometry) ---
    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        out = model(input_ids=ids, use_cache=False)  # text-only: no pixel_values
        last_logits = out.logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    text_golden = {
        "note": "SCALED bidirectional-block gemma4 VL: hidden 1024, 12 layers (5:1 sliding/full), "
                "REAL head dims (local 256 / global 512) + attention_k_eq_v on the two global "
                "layers, use_bidirectional_attention='vision'. TEXT-ONLY forward (degenerates to "
                "plain causal). CPU fp32.",
        "strengthened": counts,
        "text_config": TEXT,
        "vision_config": VISION,
        "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits,
        "n_new": N_NEW,
        "continuation_ids": cont,
    }
    os.makedirs(os.path.dirname(TEXT_OUT), exist_ok=True)
    with open(TEXT_OUT, "w") as f:
        json.dump(text_golden, f)
    print(f"wrote {TEXT_OUT}\n  argmax={text_golden['argmax']}  continuation={cont}")

    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}  (model_type={cfg.model_type!r})")

    # --- IMAGE golden (the real gate) ---
    vcfg = model.config.vision_config
    patch_size, pool_k = vcfg.patch_size, vcfg.pooling_kernel_size
    grid = 15  # patches per side (multiple of pool_k=3) -> pools to 5x5=25 soft tokens
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
        image_features = model.model.get_image_features(pixel_values, position_ids, return_dict=True).pooler_output
        out = model(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            pixel_values=pixel_values,
            image_position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
        )
        last_logits = out.logits[0, -1].float().tolist()

        out_causal = model(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            pixel_values=pixel_values,
            image_position_ids=position_ids,
            use_cache=False,
        )
        causal_logits = out_causal.logits[0, -1].float().tolist()

        # A few resident-decode continuation tokens off the bidirectional prefill — the
        # cuda-tagged parity test's own CPU reference (decode always runs sequentially
        # regardless of which prefill produced the seed, same convention as the tiny fixture).
        cur_ids = list(input_ids)
        decode_cont = []
        cur_logits = last_logits
        for _ in range(6):
            nxt = int(torch.tensor(cur_logits).argmax())
            decode_cont.append(nxt)
            cur_ids.append(nxt)
            o = model(input_ids=torch.tensor([cur_ids], dtype=torch.long), use_cache=False)
            cur_logits = o.logits[0, -1].float().tolist()

    diff = max(abs(a - b) for a, b in zip(last_logits, causal_logits))
    if diff < 1e-3:
        raise SystemExit(f"fixture is VACUOUS: bidirectional vs causal-only last_logits differ by only {diff} "
                          "— the block placement doesn't distinguish the two mechanisms, widen it")
    print(f"non-vacuousness check: bidirectional vs causal-only last_logits max-abs-diff={diff:.4f} (must be large)")

    img_feat = image_features.reshape(-1).float().tolist()
    image_golden = {
        "note": "SCALED gemma4 VL image->logits, use_bidirectional_attention='vision', REAL head "
                "dims (256 local / 512 global) + K=V on global layers — the geometry the GPU-resident "
                "decode bridge (decoder/generate_gemma4_vl.go) needs exercised at real width. Block "
                "[img_start,img_start+n_image_tokens) is longer than sliding_window=8 and positioned "
                "so the block's own start falls outside the last block position's causal window. "
                "causal_only_last_logits is the SAME forward with mm_token_type_ids omitted. "
                "decode_continuation_ids are N greedy decode steps off the bidirectional prefill's "
                "last_logits, via HF's own per-step forward (the CPU reference the cuda-tagged "
                "resident-upload parity test compares against).",
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
        "decode_continuation_ids": decode_cont,
    }
    with open(IMAGE_OUT, "w") as f:
        json.dump(image_golden, f)
    print(f"wrote {IMAGE_OUT}")
    print(f"  n_image_tokens={n_image_tokens}  image_features_shape={image_golden['image_features_shape']}  "
          f"argmax={image_golden['argmax']} (causal-only would be {image_golden['causal_only_argmax']})  "
          f"decode_continuation={decode_cont}")


if __name__ == "__main__":
    main()

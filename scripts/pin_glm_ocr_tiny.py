#!/usr/bin/env python3
"""Pin a tiny-random GLM-OCR text decoder (model_type "glm_ocr" / "glm_ocr_text") forward +
greedy decode as a goinfer parity golden, and pin the pairwise m-RoPE rotation directly against
HF's own cos/sin construction. The independent HF oracle for the glm_ocr family
(docs/tasks/task-glm-ocr-2026-10.md, O1).

Builds a SMALL random GlmOcrForConditionalGeneration (transformers >= 5.12, the model class the
real zai-org/GLM-OCR checkpoint uses) and writes:
  - testdata/glm-ocr-tiny/            model.safetensors + config.json goinfer loads. The tensors are
                                      the real LAYOUT: model.language_model.{embed_tokens,layers.N.*,norm}
                                      and a top-level lm_head.weight. The vision tower is NOT written
                                      (text only), and a fake multi-token-prediction layer is ADDED at
                                      layers.<num_hidden_layers>, exactly where the real checkpoint's
                                      layer 16 sits, so the loader's skip is exercised.
  - testdata/glm_ocr_tiny_golden.json prompt ids + last-token logits + greedy continuation (text only, f32)
  - testdata/glm_ocr_tiny_mrope_golden.json  the WHOLE tiny decoder under m-RoPE: a prompt with an
                                      image-placeholder run whose embeddings are random "image
                                      features", HF's 3-D positions (t = base, h = base + row, w = base
                                      + col, text resuming at base + max(h, w)), last-token logits and
                                      a greedy continuation at the resumed scalar positions. This is the
                                      pairwise m-RoPE exercised THROUGH the forward, not just the function
  - testdata/glm_ocr_rope_golden.json HF's GlmOcrTextRotaryEmbedding cos/sin + apply_rotary_pos_emb
                                      applied to a random q at non-trivial 3-component positions

What this PINS (the traps O0 found):
  * head_dim EXPLICIT and != hidden/heads (32 vs 48/4 = 12), so a loader or adapter that derives it
    reads the wrong q_proj width and fails to load at all;
  * GQA (kv < heads), no attention bias, fused mlp.gate_up_proj split GATE FIRST;
  * the four norms per layer under GLM's names: post_self_attn_layernorm is the norm on the attention
    OUTPUT and post_attention_layernorm is the PRE-MLP norm (the reverse of Gemma's);
  * PAIRWISE rotation (rotate_half_llm), full rotary width, and the m-RoPE section config
    (the real [16, 24, 24] scaled to the tiny half-width: [4, 6, 6]);
  * the untied lm_head and the ignored MTP layer.

Degeneracy guards (same lesson as cohere-tiny / gemma4-moe):
  * HF's default init leaves every RMSNorm weight at 1.0, so swapping two norms (or dropping a weight)
    would not move the golden. Every norm weight is reseeded to a DISTINCT non-trivial value from a
    SEPARATE torch.Generator (the global RNG that drew the linear weights is untouched);
  * the prompt is 48 tokens long so the pairwise-vs-NeoX rotation angles differ materially (a short
    prompt leaves them within noise).

Run with the VL venv (transformers 5.12.0, torch, safetensors):
    ~/.venv-vl/bin/python scripts/pin_glm_ocr_tiny.py
"""
import json
import os

import torch
from safetensors.torch import save_file
from transformers import GlmOcrConfig
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrForConditionalGeneration,
    GlmOcrTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "testdata")
CKPT = os.path.join(TESTDATA, "glm-ocr-tiny")
OUT_LOGITS = os.path.join(TESTDATA, "glm_ocr_tiny_golden.json")
OUT_ROPE = os.path.join(TESTDATA, "glm_ocr_rope_golden.json")
OUT_MROPE = os.path.join(TESTDATA, "glm_ocr_tiny_mrope_golden.json")

# Real text shape ratios, tiny dims: hidden/heads = 12 but head_dim = 32; intermediate = 3*hidden;
# mrope_section [16,24,24] over 64 frequencies -> [4,6,6] over 16.
TEXT = dict(
    vocab_size=160,
    hidden_size=48,
    intermediate_size=144,
    num_hidden_layers=3,  # >= 3 so an error compounds; the MTP layer is index 3
    num_attention_heads=4,
    num_key_value_heads=2,  # GQA
    head_dim=32,  # EXPLICIT, != hidden/heads
    hidden_act="silu",
    max_position_embeddings=512,
    rms_norm_eps=1e-5,
    attention_bias=False,
    tie_word_embeddings=False,
    rope_parameters=dict(
        rope_type="default",
        rope_theta=10000.0,
        partial_rotary_factor=1.0,
        mrope_section=[4, 6, 6],
    ),
    num_nextn_predict_layers=1,
    eos_token_id=[3, 5],
    pad_token_id=3,
)
VISION = dict(  # the smallest valid tower; never written to the fixture
    depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=48,
    patch_size=14, temporal_patch_size=2, spatial_merge_size=2, image_size=28,
)

# 48 tokens, varied, in [8, 160): positions 0..47 so the rotation angles at the high-frequency
# end differ between pairwise and half-split pairing.
PROMPT_IDS = [
    9, 141, 17, 88, 119, 14, 66, 150, 42, 25, 133, 101, 37, 120, 99, 60,
    31, 78, 24, 122, 133, 19, 71, 154, 28, 99, 12, 84, 145, 53, 108, 22,
    147, 36, 111, 77, 19, 156, 91, 140, 8, 63, 95, 12, 128, 44, 70, 159,
]
N_NEW = 8


def reseed_norms(model):
    """Distinct non-trivial value for EVERY norm weight, from a separate generator."""
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name.endswith("language_model.norm.weight"):
                p.copy_(1.0 + 0.5 * torch.randn(p.shape, generator=g))


def sharpen_weights(model):
    """HF's init (std 0.02) leaves a tiny model's logits nearly flat and its attention nearly
    uniform, so a wrong rotation barely moves them. Redraw every language-model matrix at std 0.25
    (a separate generator again) so attention is peaked and the logits have a real spread."""
    g = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("model.visual."):
                continue
            if p.dim() == 2:
                p.copy_(0.25 * torch.randn(p.shape, generator=g))


def fake_mtp_tensors(h, inter, vocab, nH, nKV, hd, layer):
    """Tensors shaped like the real layer-16 MTP block (eh_proj, enorm, hnorm, its own
    embed_tokens, shared_head, plus a full decoder block). Values are loud and unrelated to
    anything else, so a loader that read them would visibly corrupt the forward."""
    g = torch.Generator().manual_seed(777)
    p = f"model.language_model.layers.{layer}."
    r = lambda *s: (torch.randn(*s, generator=g) * 3.0).contiguous()
    return {
        p + "eh_proj.weight": r(h, 2 * h),
        p + "enorm.weight": r(h),
        p + "hnorm.weight": r(h),
        p + "embed_tokens.weight": r(vocab, h),
        p + "shared_head.head.weight": r(vocab, h),
        p + "shared_head.norm.weight": r(h),
        p + "input_layernorm.weight": r(h),
        p + "post_attention_layernorm.weight": r(h),
        p + "post_self_attn_layernorm.weight": r(h),
        p + "post_mlp_layernorm.weight": r(h),
        p + "self_attn.q_proj.weight": r(nH * hd, h),
        p + "self_attn.k_proj.weight": r(nKV * hd, h),
        p + "self_attn.v_proj.weight": r(nKV * hd, h),
        p + "self_attn.o_proj.weight": r(h, nH * hd),
        p + "mlp.gate_up_proj.weight": r(2 * inter, h),
        p + "mlp.down_proj.weight": r(h, inter),
    }


def pin_model():
    torch.manual_seed(0)
    cfg = GlmOcrConfig(text_config=TEXT, vision_config=VISION)
    model = GlmOcrForConditionalGeneration(cfg).eval().to(torch.float32)
    assert model.config.text_config.head_dim == 32 != TEXT["hidden_size"] // TEXT["num_attention_heads"]
    sharpen_weights(model)
    reseed_norms(model)

    ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
    with torch.no_grad():
        last = model(input_ids=ids).logits[0, -1].to(torch.float64)
        argmax = int(torch.argmax(last).item())
        cont, cur = [], ids
        for _ in range(N_NEW):
            lg = model(input_ids=cur).logits[0, -1]
            nxt = int(torch.argmax(lg).item())
            cont.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    golden = dict(
        prompt="",  # ids pinned directly (no tokenizer dependency for a random model)
        prompt_ids=PROMPT_IDS,
        argmax=argmax,
        vocab_size=TEXT["vocab_size"],
        last_logits=[float(x) for x in last.tolist()],
        n_new=N_NEW,
        continuation_ids=cont,
    )
    with open(OUT_LOGITS, "w") as f:
        json.dump(golden, f, indent=1)
    print("wrote", os.path.relpath(OUT_LOGITS), "argmax", argmax, "cont", cont)

    # Checkpoint: language model + lm_head only, plus the fake MTP block.
    tensors = {}
    for k, v in model.state_dict().items():
        if k.startswith("model.visual."):
            continue
        tensors[k] = v.detach().to(torch.float32).contiguous()
    tensors.update(fake_mtp_tensors(
        TEXT["hidden_size"], TEXT["intermediate_size"], TEXT["vocab_size"],
        TEXT["num_attention_heads"], TEXT["num_key_value_heads"], TEXT["head_dim"],
        TEXT["num_hidden_layers"]))
    assert "lm_head.weight" in tensors and "model.language_model.embed_tokens.weight" in tensors
    os.makedirs(CKPT, exist_ok=True)
    save_file(tensors, os.path.join(CKPT, "model.safetensors"), metadata={"format": "pt"})

    # config.json in the real checkpoint's shape: wrapper model_type, nested text_config, and the
    # vision_config (which goinfer's text loader ignores).
    cfg_json = json.loads(cfg.to_json_string())
    keep = {
        "architectures": ["GlmOcrForConditionalGeneration"],
        "model_type": "glm_ocr",
        "tie_word_embeddings": False,
        "image_token_id": 150, "image_start_token_id": 151, "image_end_token_id": 152,
        "text_config": {k: v for k, v in cfg_json["text_config"].items()
                        if k not in ("transformers_version", "_name_or_path", "output_hidden_states")},
        "vision_config": cfg_json["vision_config"],
    }
    keep["text_config"]["model_type"] = "glm_ocr_text"
    with open(os.path.join(CKPT, "config.json"), "w") as f:
        json.dump(keep, f, indent=1)
    print("wrote", os.path.relpath(CKPT), "tensors:", len(tensors))
    pin_mrope(model)


def pin_mrope(model):
    """The tiny decoder under m-RoPE positions, via the text model's own inputs_embeds + 3-D
    position_ids interface (what GlmOcrModel.forward hands it for an image turn). The placeholder
    run stands for a 6x8-patch grid after the 2x2 merge: 3 rows x 4 cols = 12 tokens."""
    hidden = TEXT["hidden_size"]
    image_id = 150
    rows, cols = 3, 4
    pre = [9, 141, 17, 88, 119, 14, 66, 150 - 1]  # 8 text tokens (last one is "begin of image")
    post = [152, 42, 25, 133, 101, 37]            # "end of image" + 5 text tokens
    ids = pre + [image_id] * (rows * cols) + post
    img_pos = len(pre)
    S = len(ids)

    g = torch.Generator().manual_seed(2024)
    feats = torch.randn(rows * cols, hidden, generator=g)

    # HF get_rope_index for one image: text positions 0..img_pos-1; the image block is
    # (t = base, h = base + row, w = base + col) with base = img_pos; then the scalar resumes at
    # base + max(rows, cols).
    pos = [[i, i, i] for i in range(img_pos)]
    for r in range(rows):
        for c in range(cols):
            pos.append([img_pos, img_pos + r, img_pos + c])
    nxt = img_pos + max(rows, cols)
    pos += [[nxt + i] * 3 for i in range(len(post))]
    assert len(pos) == S

    lm = model.model.language_model
    embed = lm.embed_tokens
    n_new = 6

    def logits_for(emb, position_ids):
        h = lm(inputs_embeds=emb, position_ids=position_ids).last_hidden_state
        return model.lm_head(h)

    with torch.no_grad():
        emb = embed(torch.tensor([ids]))
        emb[0, img_pos:img_pos + rows * cols] = feats
        pid = torch.tensor(pos, dtype=torch.long).t().unsqueeze(1)
        last = logits_for(emb, pid)[0, -1].to(torch.float64)
        argmax = int(torch.argmax(last).item())
        # text-only twin (same embeddings, plain scalar positions) to show the golden is not
        # insensitive to the m-RoPE positions
        pid_plain = torch.arange(S).view(1, 1, -1).expand(3, 1, -1)
        plain = logits_for(emb, pid_plain)[0, -1].to(torch.float64)
        cont, cur_emb, cur_pos = [], emb, pos
        for i in range(n_new):
            lg = logits_for(cur_emb, torch.tensor(cur_pos, dtype=torch.long).t().unsqueeze(1))[0, -1]
            t = int(torch.argmax(lg).item())
            cont.append(t)
            cur_emb = torch.cat([cur_emb, embed(torch.tensor([[t]]))], dim=1)
            cur_pos = cur_pos + [[nxt + len(post) + i] * 3]
    cos_plain = float(torch.nn.functional.cosine_similarity(last, plain, dim=0))
    print("m-RoPE vs scalar-position last-logit cosine (must be visibly < 1):", cos_plain)

    golden = dict(
        prompt_ids=ids, image_token_id=image_id, image_start=img_pos, n_image_tokens=rows * cols,
        grid_thw=[1, rows * 2, cols * 2], merge=2, positions=pos,
        image_features=[[float("%.9g" % x) for x in row] for row in feats.tolist()],
        argmax=argmax, vocab_size=TEXT["vocab_size"], last_logits=[float(x) for x in last.tolist()],
        scalar_position_logits_cosine=cos_plain, n_new=n_new, continuation_ids=cont,
    )
    with open(OUT_MROPE, "w") as f:
        json.dump(golden, f, indent=1)
    print("wrote", os.path.relpath(OUT_MROPE), "argmax", argmax, "cont", cont)


def pin_rope():
    """HF's cos/sin construction + apply_rotary_pos_emb at the REAL geometry: head_dim 128, full
    rotary, theta 1e4, mrope_section [16, 24, 24]. Positions mix plain text (equal components) with
    an image block's (t = base, h = base + row, w = base + col) and the text that resumes after it.
    HF builds cos/sin in float32 (inv_freq float32, position matmul float32); positions stay below
    ~300 so that rounding stays far under the Go test's tolerance."""
    hd, heads = 128, 2
    section = [16, 24, 24]
    cfg = GlmOcrConfig(
        text_config=dict(
            vocab_size=160, hidden_size=1536, intermediate_size=4608, num_hidden_layers=1,
            num_attention_heads=16, num_key_value_heads=8, head_dim=hd, max_position_embeddings=131072,
            rope_parameters=dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=1.0,
                                 mrope_section=section)),
        vision_config=VISION).text_config
    rot = GlmOcrTextRotaryEmbedding(cfg)

    # 3-component positions [S][3]: 5 text tokens at 0..4, an image block of 3 rows x 4 cols whose
    # base is 5 (t=5, h=5+row, w=5+col), then text resuming at base + max(rows, cols) = 9.. plus a
    # far text position and a second image-style block with a LARGE base, so t, h and w all differ
    # and rotate at every frequency band.
    pos = [[i, i, i] for i in range(5)]
    for r in range(3):
        for c in range(4):
            pos.append([5, 5 + r, 5 + c])
    pos += [[9 + i, 9 + i, 9 + i] for i in range(3)]
    for r in range(2):
        for c in range(5):
            pos.append([200, 200 + r, 200 + c])
    pos += [[205, 205, 205], [251, 251, 251]]
    S = len(pos)

    g = torch.Generator().manual_seed(42)
    q = torch.randn(1, heads, S, hd, generator=g)
    k = torch.randn(1, heads, S, hd, generator=g)
    q, k = q.to(torch.float32), k.to(torch.float32)
    position_ids = torch.tensor(pos, dtype=torch.long).t().unsqueeze(1)  # [3, 1, S]
    with torch.no_grad():
        cos, sin = rot(q, position_ids)  # [1, S, hd]
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

    def per_pos(t):  # [1, heads, S, hd] -> [S][heads][hd], float32 values at 9 significant digits
        return [[[float("%.9g" % x) for x in head] for head in row] for row in t[0].permute(1, 0, 2).tolist()]

    # q only: HF applies the identical function to k, and the Go side calls the same function for both.
    golden = dict(
        head_dim=hd, heads=heads, mrope_section=section, rope_theta=10000.0, positions=pos,
        q_in=per_pos(q), q_out=per_pos(q_out),
    )
    with open(OUT_ROPE, "w") as f:
        json.dump(golden, f, separators=(",", ":"))
    print("wrote", os.path.relpath(OUT_ROPE), "positions", S)


if __name__ == "__main__":
    pin_model()
    pin_rope()

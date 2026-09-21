#!/usr/bin/env python
"""Build a tiny-random Spark-X2.5 checkpoint + text golden (Gate 1, task-spark-x2-5.md).

Reference module is the REAL modeling_spark.py / configuration_spark.py, fetched directly from
XHToken/Spark-X2.5-4B (trust_remote_code) and vendored alongside the checkpoint so the fixture
reproduces offline — same discipline as pin_laguna_tiny.py.

Deliberately exercises every real departure from a plain Llama, at a size small enough to be
fast and byte-exact, per docs/tasks/task-spark-x2-5.md and its own corrections to the L-06 audit
(docs/audit-2026-09-10.md) found while scoping this: the decoder layer is the STANDARD sequential
residual (NOT Cohere's parallel block, despite the audit's claim), and the MLP is GATED (SwiGLU-
shaped) with exact-erf GELU as its activation (NOT "non-gated GELU", the audit's other wrong
claim) — see decoder/registry.go's spark25Architecture doc comment for the full correction.

  - Fused q_k_v_proj, split Q‖K‖V by output rows.
  - Head-wise SIGMOID attention-output gate (g_proj), applied before out_proj.
  - 1:3 sliding:full interleave (3 sliding, 1 full, repeating) with a window SMALLER than the
    prompt so it actually clips (Laguna/gemma4-twogeom convention).
  - Per-layer-type partial RoPE: full_attention rotates 1/4 of head_dim at theta 5e6; sliding
    rotates the FULL head_dim at theta 1e4 — the exact real-config ratio, scaled down.
  - Decoupled head_dim (hidden_size 64 != num_heads*head_dim 128) — the real checkpoint has this
    too (hidden 2560 != 16*256=4096); a loader that assumes head_dim=hidden/heads fails silently.
  - tie_word_embeddings is LEFT FALSE here, deliberately: the installed transformers (5.16.1)
    crashes in PreTrainedModel.post_init's get_expanded_tied_weights_keys on this module's
    old-style `_tied_weights_keys = ["lm_head.weight"]` (a list, where 5.x's tie-key expansion
    expects a dict) — a library-version mismatch against the model's own pinned
    transformers_version (4.57.1), unrelated to anything Spark-specific. Tying is already a
    generic, separately-tested goinfer mechanism (arch.TiedLMHead, finalized from lm_head.weight
    PRESENCE at load — see buildSpark25Weights/buildPhi3Weights), not something this fixture needs
    to re-prove; the real tie_word_embeddings=True config is exercised at Gate 2 (real-oracle,
    the actual released checkpoint, run in whatever environment matches its pinned transformers).

    ~/.venv-spark25/bin/python scripts/pin_spark2_5_tiny.py
    -> decoder/testdata/spark2_5_tiny_text_golden.json   (tracked)
    -> decoder/testdata/spark2-5-tiny/                   (gitignored)

Needs transformers==4.57.1 exactly (the model's own pinned transformers_version) in a dedicated
venv, NOT the system/global python: the vendored modeling_spark.py hits two real breaking API
changes against a newer transformers (5.16.1, tried first) — PreTrainedModel.post_init's
get_expanded_tied_weights_keys chokes on the module's old-style `_tied_weights_keys` list (worked
around below by leaving tie_word_embeddings=False, since tying is a separately-tested generic
goinfer mechanism, not something this fixture needs to re-prove), and create_causal_mask()'s
signature no longer accepts input_embeds= at all (no workaround available — this one needs the
matching library version). `python3 -m venv ~/.venv-spark25 && ~/.venv-spark25/bin/pip install
"torch>=2.2,<2.9" "transformers==4.57.1"`.
"""
import json
import os
import shutil
import urllib.request

import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "decoder", "testdata")
SRC_REPO = "XHToken/Spark-X2.5-4B"
CKPT = os.path.join(TESTDATA, "spark2-5-tiny")
CODE = os.path.join(TESTDATA, ".spark2-5-code")
CODE_FILES = ("configuration_spark.py", "modeling_spark.py")

PROMPT = [1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 9, 17, 60, 33, 2, 88]  # len 16 > sliding_window 8
N_NEW = 6

CFG = dict(
    model_type="spark2_5",
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,  # decoupled: hidden(64) != heads*head_dim(128)
    rms_norm_eps=1e-6, hidden_act="gelu",
    attention_bias=False, mlp_bias=False, tie_word_embeddings=False,
    max_position_embeddings=512,
    headwise_attn_output_gate=True, gate_attn_act_mode="sigmoid",
    sliding_window=8,
    layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    rope_parameters={
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
        "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
    },
)


def fetch_code():
    os.makedirs(CODE, exist_ok=True)
    for fn in CODE_FILES:
        out = os.path.join(CODE, fn)
        url = f"https://huggingface.co/{SRC_REPO}/resolve/main/{fn}"
        with urllib.request.urlopen(url, timeout=60) as r, open(out, "wb") as f:
            f.write(r.read())


def main():
    os.makedirs(TESTDATA, exist_ok=True)
    fetch_code()

    cfg_dict = dict(CFG)
    cfg_dict["auto_map"] = {
        "AutoConfig": "configuration_spark.Spark2_5Config",
        "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
    }
    cfg_dict["architectures"] = ["Spark2_5ForCausalLM"]
    with open(os.path.join(CODE, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=1)

    cfg = AutoConfig.from_pretrained(CODE, trust_remote_code=True)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval().to(torch.float32)

    with torch.no_grad():
        ids = torch.tensor([PROMPT], dtype=torch.long)
        last_logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().tolist()
        cur, cont = list(PROMPT), []
        for _ in range(N_NEW):
            o = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
            cont.append(int(o.logits[0, -1].argmax()))
            cur.append(cont[-1])

    golden = {
        "note": "tiny-random Spark2_5ForCausalLM, text forward; CPU fp32; reference module from " + SRC_REPO,
        "config": cfg_dict, "prompt_ids": PROMPT,
        "argmax": int(torch.tensor(last_logits).argmax()),
        "last_logits": last_logits, "n_new": N_NEW, "continuation_ids": cont,
    }
    out = os.path.join(TESTDATA, "spark2_5_tiny_text_golden.json")
    with open(out, "w") as f:
        json.dump(golden, f)

    model.save_pretrained(CKPT, safe_serialization=True)
    for fn in CODE_FILES:
        shutil.copyfile(os.path.join(CODE, fn), os.path.join(CKPT, fn))

    import glob
    st_files = glob.glob(os.path.join(CKPT, "*.safetensors"))
    has_lm_head = False
    try:
        from safetensors import safe_open
        for stf in st_files:
            with safe_open(stf, framework="pt") as sf:
                if "lm_head.weight" in sf.keys():
                    has_lm_head = True
    except Exception as e:
        print(f"  (could not inspect safetensors keys: {e})")

    print(f"argmax={golden['argmax']} cont={cont}")
    print(f"lm_head.weight written to checkpoint: {has_lm_head} (expect True — untied, see the tie_word_embeddings note above)")
    print(f"-> {out}\n-> {CKPT}")


if __name__ == "__main__":
    main()

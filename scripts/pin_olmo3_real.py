#!/usr/bin/env python3
"""Real-model parity golden for Olmo 3 (model_type "olmo3", Olmo3ForCausalLM) — the T3
promotion of goinfer's olmo3 family from tiny-golden to a released checkpoint
(docs/parity-coverage-policy.md). allenai/Olmo-3-7B-Think is the smallest released size
(32B also exists but was scoped out of the T3 target, docs/task-families-2026-09.md G2).

Exercises Olmo 3's two real departures on real weights: NormPostOnly (no pre-norm at all,
only the sublayer OUTPUT is normalized before the residual add) and QKNormWhole (QK-norm over
the FULL projected q/k vector, not per head) — plus the local/global RoPE split, since the real
release is sliding/full 3:1 (sliding_window=4096) with YaRN applying to full_attention layers
only.

Dumps the last-token logits + argmax + a short greedy continuation (token IDs) for a fixed
prompt; the goinfer side (olmo3_real_test.go, build tag realckpt) loads the same safetensors at
f32 and matches argmax + continuation + cosine >= 0.9999.

    ~/.venv-triton-check/bin/python scripts/pin_olmo3_real.py      # nobara-pc: transformers 5.15.0
    -> testdata/olmo3_real_golden.json.gz   (committed; the ~14 GB weights are NOT)

NEEDS transformers >= 5.15.0, and refuses older. transformers 5.12's Olmo3 builds ONE rotary table
and applies the checkpoint's flat YaRN rope_scaling to every layer; 5.15 splits it per layer type
(YaRN on full_attention only), which is what the Olmo 3 paper (arXiv 2512.13961) says the model was
trained with. The 2026-09-07 golden was pinned under 5.12 from ~/.venv-vl, so it baked YaRN into
the sliding layers and goinfer's correct split read cosine 0.992789 against it. The script now
checks the reference's per-layer-type RoPE before trusting it, and records the versions and the
RoPE layout in the golden.

Put the checkpoint at ~/models/olmo3-7b-think (allenai/Olmo-3-7B-Think), or set
GOINFER_OLMO3_7B to its path.
"""
import gzip
import json
import os

import math

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ.get("GOINFER_OLMO3_7B", os.path.expanduser("~/models/olmo3-7b-think"))
MIN_TRANSFORMERS = (5, 15, 0)


def _version_tuple(v):
    parts = []
    for p in v.split(".")[:3]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def check_rope(model, cfg):
    """The reference's own per-layer-type RoPE, checked before any logit is trusted (CLAUDE.md's
    internlm2 lesson: a reference buffer can be wrong). Returns the layout recorded in the golden."""
    rot = model.model.rotary_emb
    want_full = (cfg.rope_parameters["full_attention"].get("attention_factor")
                 or (0.1 * math.log(cfg.rope_parameters["full_attention"]["factor"]) + 1.0))
    layout = {}
    for lt, want in (("full_attention", want_full), ("sliding_attention", 1.0)):
        inv = getattr(rot, f"{lt}_inv_freq", None)
        scale = getattr(rot, f"{lt}_attention_scaling", None)
        if inv is None or scale is None:
            raise SystemExit(f"rotary_emb has no {lt} table: this transformers does not split RoPE by layer type")
        if not torch.isfinite(inv).all():
            raise SystemExit(f"{lt}_inv_freq has non-finite values: the reference is broken, refusing to pin")
        if abs(float(scale) - want) > 1e-4 * max(1.0, want):
            raise SystemExit(f"{lt} attention scaling {float(scale)} != expected {want}")
        layout[lt] = dict(rope_type=rot.rope_type[lt], attention_scaling=float(scale))
    return layout
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "olmo3_real_golden.json.gz")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    if _version_tuple(transformers.__version__) < MIN_TRANSFORMERS:
        raise SystemExit(f"transformers {transformers.__version__} is older than "
                         f"{'.'.join(map(str, MIN_TRANSFORMERS))}: its Olmo3 applies YaRN to every layer (see above)")
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.float32).eval()
    cfg = model.config
    rope_layout = check_rope(model, cfg)
    print("rope per layer type:", rope_layout)
    arch = cfg.architectures[0] if cfg.architectures else "?"
    layer_types = getattr(cfg, "layer_types", None)
    n_sliding = sum(1 for t in (layer_types or []) if t == "sliding_attention")
    n_full = sum(1 for t in (layer_types or []) if t == "full_attention")
    print(f"loaded {arch} model_type={cfg.model_type} num_hidden_layers={cfg.num_hidden_layers} "
          f"sliding_window={getattr(cfg,'sliding_window',None)} layers: {n_sliding} sliding / {n_full} full "
          f"rope_scaling={getattr(cfg,'rope_scaling',None)}")

    ids = tok(PROMPT, return_tensors="pt").input_ids
    prompt_ids = ids[0].tolist()
    with torch.no_grad():
        last = model(ids).logits[0, -1].to(torch.float64)
        argmax = int(torch.argmax(last).item())
        cont, cur = [], ids
        for _ in range(N_NEW):
            nxt = int(torch.argmax(model(cur).logits[0, -1]).item())
            cont.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    golden = dict(
        prompt=PROMPT,
        prompt_ids=prompt_ids,
        argmax=argmax,
        vocab_size=cfg.vocab_size,
        sliding_window=getattr(cfg, "sliding_window", None),
        layer_types=layer_types,
        last_logits=[float(x) for x in last.tolist()],
        n_new=N_NEW,
        continuation_ids=cont,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        rope=rope_layout,
    )
    with gzip.open(OUT, "wt") as f:
        json.dump(golden, f)
    print("wrote", os.path.relpath(OUT))
    print("prompt_ids", prompt_ids, "argmax", argmax, "cont", cont)
    print("continuation:", repr(tok.decode(cont)))


if __name__ == "__main__":
    main()

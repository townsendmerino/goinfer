#!/usr/bin/env python3
"""Real-model parity golden for InternLM2.5-1.8B-Chat (model_type "internlm2",
InternLM2ForCausalLM) — the T3 promotion of goinfer's InternLM2 family from
tiny-golden to a released checkpoint. The tiny fixture (scripts/pin_internlm2_tiny.py)
pins ONLY the grouped wqkv de-interleave in isolation (a reference llama's separate
q/k/v repacked by hand into InternLM2's fused per-KV-head [Q..Q K V] layout); this
script proves the same de-interleave against the model's OWN native fused wqkv
weights, on real trained values, end to end.

Dumps the last-token logits + argmax + a short greedy continuation (token IDs) for a
fixed prompt; the goinfer side (internlm2_real_test.go, build tag realckpt) loads the
same safetensors at f32 and matches argmax + continuation + cosine >= 0.9999.

    ~/.venv-nemotron3/bin/python scripts/pin_internlm2_real.py
    -> testdata/internlm2_real_golden.json   (committed; the ~3.6 GB weights are NOT)

Put the checkpoint at ~/models/internlm2-1_8b (internlm/internlm2_5-1_8b-chat), or
set GOINFER_INTERNLM2_1_8B to its path. Needs trust_remote_code=True — InternLM2 is
not a built-in transformers architecture, it ships its own modeling_internlm2.py.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = os.environ.get("GOINFER_INTERNLM2_1_8B", os.path.expanduser("~/models/internlm2-1_8b"))
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "testdata", "internlm2_real_golden.json")
PROMPT = "The capital of France is"
N_NEW = 8


def main():
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, torch_dtype=torch.float32, trust_remote_code=True
    ).eval()

    # WORKAROUND (confirmed 2026-09-07): this transformers version's from_pretrained never
    # re-runs InternLM2RotaryEmbedding.__init__'s inv_freq formula for buffers absent from
    # the checkpoint's state dict. inv_freq is registered `persistent=False` (correctly
    # excluded from the checkpoint, since it's derived, not learned) but its fast-init path
    # leaves it as UNINITIALIZED MEMORY (denormals/zeros, sometimes literal NaN) instead of
    # 1/base^(2d/dim) — a class of bug HF's newer ROPE_INIT_FUNCTIONS registry exists to
    # prevent; InternLM2's older-style remote code predates that registry, so it's exposed.
    # Without this patch the "reference" this script produces is itself garbage: bisecting
    # the loaded model's real forward down to the exact buffer showed the corruption, and
    # the patched model's output matches goinfer's Go forward (and two independent
    # from-scratch reimplementations) to 7 significant figures, while the unpatched one does
    # not. Harmless no-op if a future transformers fixes the underlying bug (reassigning an
    # already-correct value) — printed rather than asserted, since that's not worth halting
    # golden generation over.
    already_correct = 0
    for layer in model.model.layers:
        r = layer.attention.rotary_emb
        correct = 1.0 / (float(r.base) ** (torch.arange(0, r.dim, 2, dtype=torch.int64).float() / r.dim))
        if torch.allclose(r.inv_freq, correct, atol=1e-12):
            already_correct += 1
        r.inv_freq.data.copy_(correct)
    if already_correct:
        print(f"NOTE: inv_freq was already correct on {already_correct} layer(s) — "
              f"the from_pretrained bug this patches may be fixed upstream")

    cfg = model.config
    arch = cfg.architectures[0] if cfg.architectures else "?"
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nh)
    groups = nh // nkv
    print(f"loaded {arch} model_type={cfg.model_type} num_hidden_layers={cfg.num_hidden_layers} "
          f"num_attention_heads={nh} num_key_value_heads={nkv} groups={groups}")
    if groups < 2:
        print(f"WARNING: groups={groups} < 2 — the grouped wqkv layout and a plain concat "
              f"coincide on this checkpoint, so this golden would not discriminate a broken "
              f"de-interleave. (The tiny fixture already covers the discriminating case.)")

    ids = tok(PROMPT, return_tensors="pt").input_ids
    prompt_ids = ids[0].tolist()
    with torch.no_grad():
        # use_cache=False: this checkpoint's config.json defaults use_cache=true, and its
        # remote modeling_internlm2.py unconditionally calls the removed
        # DynamicCache.from_legacy_cache on transformers 5.15 when a cache would be built.
        # Explicit False sidesteps that (a version-compat issue in the reference code, not
        # a numerics path) without touching any modeling file; every forward here already
        # recomputes the full growing sequence rather than reusing a cache.
        last = model(ids, use_cache=False).logits[0, -1].to(torch.float64)
        argmax = int(torch.argmax(last).item())
        cont, cur = [], ids
        for _ in range(N_NEW):
            nxt = int(torch.argmax(model(cur, use_cache=False).logits[0, -1]).item())
            cont.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    golden = dict(
        prompt=PROMPT,
        prompt_ids=prompt_ids,
        argmax=argmax,
        vocab_size=cfg.vocab_size,
        groups=groups,
        last_logits=[float(x) for x in last.tolist()],
        n_new=N_NEW,
        continuation_ids=cont,
    )
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print("wrote", os.path.relpath(OUT))
    print("prompt_ids", prompt_ids, "argmax", argmax, "cont", cont)
    print("continuation:", repr(tok.decode(cont)))


if __name__ == "__main__":
    main()

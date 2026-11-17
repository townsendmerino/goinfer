#!/usr/bin/env python3
"""KV10-a (README.md here): which checkpoints under a models directory have no fused CUDA prefill attention kernel.

  python3 fused_census.py [models-dir, default ~/models] > fused-census.md

The rule is backend/cuda/prefill.go:cudaResident.useAttnFused: the fused kernel serves a layer only for head dim 64 or 128 (attnFusedFor), with no attention
softcap, in a pass of enough rows above the fast-prefill floor. Everything else runs the exact attn_batched. This reads each config.json's head dim (head_dim, else
hidden_size / num_attention_heads) and softcap, and groups by (model_type, head_dim, softcap). Gemma 3's config carries no head_dim: its 256 is the released value.
"""
import collections, glob, json, os, sys

FUSED_DIMS = (64, 128)
KNOWN_HD = {"gemma3_text": 256}  # configs that omit head_dim; the released checkpoints' value
SOFTCAP_KEYS = ("attn_logit_softcapping", "attention_logit_softcapping", "attn_logit_softcap")
# Families the rule does not read, with why: the layer path it governs is not theirs.
NOT_THIS_RULE = {
    "deepseek_v2": "MLA: latent attention, not the dense layer path the rule governs",
    "deepseek_v3": "MLA: latent attention, not the dense layer path the rule governs",
    "kimi_k2": "MLA: latent attention, not the dense layer path the rule governs",
    "nemotron_h": "Mamba-2 mixer: the prompt takes the sequential path, which has no batched attention (prefill.go: nonBatchableKind)",
    "granitemoehybrid": "Mamba-2 mixer: the prompt takes the sequential path, which has no batched attention (prefill.go: nonBatchableKind)",
    "lfm2": "not CUDA-resident (a resident build is declined, fit check 2026-11-16)",
}


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/models")
    groups = collections.defaultdict(list)
    n = 0
    for p in sorted(glob.glob(os.path.join(root, "*", "config.json"))):
        try:
            c = json.load(open(p))
        except (OSError, ValueError):
            continue
        t = c.get("text_config", c)
        nh = t.get("num_attention_heads") or 0
        hd = t.get("head_dim") or KNOWN_HD.get(t.get("model_type"), 0) or ((t.get("hidden_size", 0) // nh) if nh else 0)
        if not hd:
            continue  # not an attention decoder (an encoder, a speech model)
        cap = any(t.get(k) for k in SOFTCAP_KEYS)
        n += 1
        groups[(t.get("model_type", "?"), hd, cap)].append(os.path.basename(os.path.dirname(p)))
    print(f"# Fused CUDA prefill attention: which checkpoints get it ({n} attention-decoder configs under {root})\n")
    print("The rule (`backend/cuda/prefill.go:cudaResident.useAttnFused`): the fused kernel serves a layer only at head dim 64 or 128, with no attention softcap; every other layer runs the exact `attn_batched`.\n")
    for title, keep in (("No fused prefill kernel", lambda mt, hd, cap: mt not in NOT_THIS_RULE and (hd not in FUSED_DIMS or cap)),
                        ("Fused prefill kernel", lambda mt, hd, cap: mt not in NOT_THIS_RULE and hd in FUSED_DIMS and not cap)):
        print(f"## {title}\n")
        print("| model type | head dim | softcap | checkpoints here |\n|---|---|---|---|")
        for (mt, hd, cap), names in sorted(groups.items()):
            if keep(mt, hd, cap):
                print(f"| `{mt}` | {hd} | {'yes' if cap else 'no'} | {', '.join(names)} |")
        print()
    print("## The rule does not read these\n")
    print("| model type | head dim | why | checkpoints here |\n|---|---|---|---|")
    for (mt, hd, cap), names in sorted(groups.items()):
        if mt in NOT_THIS_RULE:
            print(f"| `{mt}` | {hd} | {NOT_THIS_RULE[mt]} | {', '.join(names)} |")
    print()


if __name__ == "__main__":
    main()

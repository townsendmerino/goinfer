#!/usr/bin/env python3
"""I1 of docs/tasks/task-mellum21-2026-10.md ("The third crater, investigated"): does HF's own top-8 router choice differ from the exact one?

    ~/g4venv/bin/python scripts/hf_router_flips.py run <model-dir> <window_golden.json> <A3 layers_cos.json> <out-dir>
    ~/g4venv/bin/python scripts/hf_router_flips.py selftest

HF's MellumTopKRouter computes its logits with a bf16 matmul (F.linear on the bf16 hidden state and bf16 weight), rounds them to bf16, and takes the top 8 of 64
from them; goinfer's router is f32. `run` loads the HF bf16 checkpoint, runs the golden's ids once, hooks every layer's router, and for every (layer, position)
recomputes the router in f32 from the SAME input (hidden.float() @ weight.float().T, softmax, top 8) and records the experts HF chose that the exact router did not
(flips), the bf16 logit error, and the exact boundary margin (logit of the 8th minus the 9th). It then reads, from the goinfer-vs-HF cosine table given, each
position's ONSET layer (the first layer whose cosine falls more than 0.001 below the previous layer's) and prints the rule's counts.

Rule (registered before the run): over the positions with an onset at or after layer 1 in that table, H-flip is SUPPORTED if at least 80% of the onset cells have a flip
in HF's own selection (HF chose an expert the exact router did not) and fewer than 20% of the control cells (the same positions, every layer before the onset) do;
REFUTED if at most 40% of the onset cells have one; PARTIAL otherwise.
"""
import json
import os
import sys

import numpy as np

DROP = 0.001


def onsets(C, positions):
    """C[l][pi]. Returns {position: onset layer}: the first layer whose cosine is more than DROP below the previous layer's."""
    out = {}
    for i, p in enumerate(positions):
        prev = 1.0
        for l in range(len(C)):
            if C[l][i] < prev - DROP:
                out[p] = l
                break
            prev = C[l][i]
    return out


def verdict(flip_at, onset, nl):
    """flip_at(l, p) -> bool. Returns (counts line, verdict)."""
    cells = [(p, l) for p, l in onset.items() if l >= 1]
    ctl = [(p, l) for p, lo in onset.items() for l in range(lo)]
    on_f = sum(1 for p, l in cells if flip_at(l, p))
    ct_f = sum(1 for p, l in ctl if flip_at(l, p))
    fo = on_f / max(1, len(cells))
    fc = ct_f / max(1, len(ctl))
    v = "SUPPORTED" if fo >= 0.8 and fc < 0.2 else ("REFUTED" if fo <= 0.4 else "PARTIAL")
    return f"onset cells with a flip: {on_f}/{len(cells)} ({fo:.0%}); control cells with a flip: {ct_f}/{len(ctl)} ({fc:.0%})", v


def run(model_dir, golden, cos_json, out_dir):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    os.makedirs(out_dir, exist_ok=True)
    ids = json.load(open(golden))["ids"]
    cj = json.load(open(cos_json))
    positions, C = cj["positions"], cj["cos"]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    nl, T = model.config.num_hidden_layers, len(ids)
    k = model.config.num_experts_per_tok
    flips = np.zeros((nl, T), dtype=np.int8)       # experts HF chose that the exact (f32) router did not
    margin = np.zeros((nl, T), dtype=np.float32)   # exact logit gap between the 8th and 9th expert
    err = np.zeros((nl, T), dtype=np.float32)      # max |bf16 logit - exact logit| over the experts
    hooks = []
    for l, layer in enumerate(model.model.layers):
        def mk(l):
            def hook(mod, args, output):
                h = args[0].reshape(-1, mod.hidden_dim)
                exact = F.linear(h.float(), mod.weight.float())
                _, idx = output[0], output[2]
                top_e = torch.topk(exact, k + 1, dim=-1)
                sel_e = torch.topk(exact, k, dim=-1).indices
                a = torch.zeros_like(exact, dtype=torch.bool).scatter_(1, idx, True)
                b = torch.zeros_like(exact, dtype=torch.bool).scatter_(1, sel_e, True)
                flips[l] = (a & ~b).sum(dim=1).numpy()
                margin[l] = (top_e.values[:, k - 1] - top_e.values[:, k]).numpy()
                err[l] = (output[0].float() - exact).abs().max(dim=1).values.numpy()
            return hook
        hooks.append(layer.mlp.gate.register_forward_hook(mk(l)))
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]))
    for h in hooks:
        h.remove()
    np.savez(os.path.join(out_dir, "router_flips.npz"), flips=flips, margin=margin, err=err, positions=np.array(positions))
    lines = [f"HF bf16 router against the exact f32 router from the same input, {T} tokens, top-{k} of {model.config.num_experts}",
             f"fraction of (layer, token) cells where HF's top-{k} differs from the exact one: {(flips > 0).mean():.4f}; mean experts differing per cell {flips.mean():.4f}",
             f"max |bf16 logit - exact logit| over the cells: median {np.median(err):.4f}, 99th percentile {np.percentile(err, 99):.4f}; exact boundary margin: median {np.median(margin):.4f}, 5th percentile {np.percentile(margin, 5):.4f}",
             "per layer: fraction of tokens with a flip: " + " ".join(f"L{l}={(flips[l] > 0).mean():.3f}" for l in range(nl))]
    pi = {p: i for i, p in enumerate(positions)}
    ons = onsets(C, positions)
    lines.append("onset layer per position (first layer whose goinfer-vs-HF cosine falls > 0.001 below the previous): " + ", ".join(f"p{p}: L{l}" for p, l in sorted(ons.items())))
    for p, l in sorted(ons.items()):
        lines.append(f"  p{p} onset L{l}: flips at that cell {int(flips[l, p])}, margin {margin[l, p]:.4f}, bf16 logit err {err[l, p]:.4f}; flips earlier at this position: "
                     + (",".join(f"L{j}={int(flips[j, p])}" for j in range(l) if flips[j, p]) or "none"))
    cnt, v = verdict(lambda l, p: flips[l, p] > 0, ons, nl)
    lines += [cnt, f"I1 H-flip: {v}"]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(out_dir, "result.txt"), "w").write(text + "\n")


def selftest():
    C = [[1.0] * 3 for _ in range(6)]
    for l in range(3, 6):
        C[l][1] = 0.99          # position index 1 drops at layer 3
    for l in range(5, 6):
        C[l][2] = 0.98          # position index 2 drops at layer 5
    pos = [10, 600, 1300]
    ons = onsets(C, pos)
    assert ons == {600: 3, 1300: 5}, ons
    always = lambda l, p: (p, l) in {(600, 3), (1300, 5)}
    c, v = verdict(always, ons, 6)
    assert v == "SUPPORTED", (c, v)
    never = lambda l, p: False
    assert verdict(never, ons, 6)[1] == "REFUTED"
    everywhere = lambda l, p: True
    assert verdict(everywhere, ons, 6)[1] == "PARTIAL"   # onsets all flip, but so do the controls: the flips do not single the onsets out
    half = lambda l, p: (p, l) == (600, 3)
    assert verdict(half, ons, 6)[1] == "PARTIAL"
    print("selftest ok: onsets found; the verdict is SUPPORTED, REFUTED and PARTIAL on the planted cases")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "selftest":
        selftest()
    elif len(sys.argv) == 6 and sys.argv[1] == "run":
        run(*sys.argv[2:])
    else:
        sys.exit(__doc__)

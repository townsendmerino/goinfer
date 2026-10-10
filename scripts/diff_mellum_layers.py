#!/usr/bin/env python3
"""Per-layer residual comparison, goinfer vs Hugging Face, for the Mellum window golden (follow-up A of docs/tasks/task-mellum21-2026-10.md).

    ~/g4venv/bin/python scripts/diff_mellum_layers.py run  <model-dir> <window_golden.json> <goinfer_dump.json.gz> <out-dir>
    ~/g4venv/bin/python scripts/diff_mellum_layers.py selftest

`run` loads the HF bf16 checkpoint, runs the golden's ids once with a forward hook on every decoder layer (the residual AFTER the layer, before
the final norm: the same tensor goinfer's ForwardCapture returns; HF's own hidden_states[-1] is post-norm and would not compare), reads the
goinfer dump TestMellum2_layerDump wrote, and prints per-layer cosine and relative L2 at each dumped position, then the pre-registered
classification. It also checks the persistent=False rotary buffers (the internlm2 lesson, CLAUDE.md): inv_freq finite, and of the magnitude the
config implies.

Classification (registered before the run; positions <= 1023 are "short": the 1024 window cannot have changed them; >= 1024 are "long"):
  d_short(l) = 1 - median over short positions of cos(l, p);  d_long(l) = 1 - median over long positions.
  An EXCESS layer is one with d_long(l) > d_short(l) + 0.005. The first excess layer, and its type, name the finding:
    W  first excess layer is a SLIDING layer -> the window path is suspect
    F  first excess layer is a FULL-attention layer -> long-context full attention / YaRN is suspect
    N  no excess layer -> the logit-level drop is not localised by layer here; reported with the final-layer d_long and d_short
"""
import gzip
import json
import math
import os
import sys

import numpy as np

SHORT_MAX = 1023
EXCESS = 0.005


def cos(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def rel(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def classify(C, positions, layer_types):
    """C[l][pi] cosines. Returns (class, first_excess_layer or None, d_short, d_long)."""
    short = [i for i, p in enumerate(positions) if p <= SHORT_MAX]
    long_ = [i for i, p in enumerate(positions) if p > SHORT_MAX]
    d_s = [1 - float(np.median([C[l][i] for i in short])) for l in range(len(C))]
    d_l = [1 - float(np.median([C[l][i] for i in long_])) for l in range(len(C))]
    first = next((l for l in range(len(C)) if d_l[l] > d_s[l] + EXCESS), None)
    if first is None:
        return "N", None, d_s, d_l
    return ("W" if layer_types[first].startswith("sliding") else "F"), first, d_s, d_l


def report(C, R, positions, layer_types, out):
    lines = ["layer type       " + "".join(f"{('p' + str(p)):>9}" for p in positions) + "   (cosine goinfer vs HF)"]
    for l in range(len(C)):
        lines.append(f"{l:>5} {layer_types[l][:7]:<8} " + "".join(f"{C[l][i]:9.5f}" for i in range(len(positions))))
    lines.append("")
    lines.append("relative L2 error (goinfer vs HF) at the LAST layer: " + " ".join(f"p{p}={R[-1][i]:.4f}" for i, p in enumerate(positions)))
    cls, first, d_s, d_l = classify(C, positions, layer_types)
    lines.append(f"d_short(L{len(C)-1})={d_s[-1]:.5f}  d_long(L{len(C)-1})={d_l[-1]:.5f}")
    if first is None:
        lines.append(f"CLASS N: no layer exceeds d_short + {EXCESS}")
    else:
        lines.append(f"CLASS {cls}: first excess layer {first} ({layer_types[first]}): d_long={d_l[first]:.5f} vs d_short={d_s[first]:.5f}")
    text = "\n".join(lines)
    print(text)
    if out:
        open(os.path.join(out, "layers_report.txt"), "w").write(text + "\n")
    return cls, first


def hf_dump(model, ids, positions, nl):
    import torch
    got = {}
    hooks = []
    for l, layer in enumerate(model.model.layers):
        def mk(l):
            def hook(mod, args, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                got[l] = h[0, positions, :].float().numpy().copy()
            return hook
        hooks.append(layer.register_forward_hook(mk(l)))
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]))
    for h in hooks:
        h.remove()
    assert len(got) == nl, f"hooks fired for {len(got)} of {nl} layers"
    return [got[l] for l in range(nl)]  # [layer][posIdx][hidden]


def rotary_check(model, cfg):
    import torch
    out = []
    for name, buf in model.named_buffers():
        if "inv_freq" in name:
            b = buf.float()
            ok = bool(torch.isfinite(b).all())
            out.append(f"  {name}: shape {tuple(b.shape)} finite={ok} min={b.min().item():.3e} max={b.max().item():.3e}")
            if not ok:
                out.append("  !!! NON-FINITE inv_freq: the reference is broken (persistent=False buffer trap), do not blame goinfer")
    return out


def run(model_dir, golden, dump, out_dir):
    import torch
    from transformers import AutoModelForCausalLM
    os.makedirs(out_dir, exist_ok=True)
    ids = json.load(open(golden))["ids"]
    d = json.load(gzip.open(dump, "rt"))
    positions = d["positions"]
    assert d["n_ids"] == len(ids), f"dump is of {d['n_ids']} ids, golden has {len(ids)}"
    if d["quant"] == "int8int8":
        assert d["argmax"] == d["golden_argmax"], "the goinfer dump is not of the run the window gate measured"
    elif d["argmax"] != d["golden_argmax"]:
        print(f"note: quant {d['quant']} argmax {d['argmax']} differs from the int8int8 golden's {d['golden_argmax']} (reported, not an error)")
    print(f"goinfer dump: quant {d['quant']}, argmax {d['argmax']}, sample-256 logit cosine {d['sample256_logit_cosine']:.5f}")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    cfg = model.config
    print("rotary buffers (persistent=False check):")
    print("\n".join(rotary_check(model, cfg)) or "  (none found)")
    nl = cfg.num_hidden_layers
    ref = hf_dump(model, ids, positions, nl)
    types = list(cfg.layer_types)
    C = [[cos(np.array(d["resid"][l][i]), ref[l][i]) for i in range(len(positions))] for l in range(nl)]
    R = [[rel(np.array(d["resid"][l][i]), ref[l][i]) for i in range(len(positions))] for l in range(nl)]
    report(C, R, positions, types, out_dir)
    json.dump({"cos": C, "rel_l2": R, "positions": positions, "layer_types": types}, open(os.path.join(out_dir, "layers_cos.json"), "w"))


def selftest():
    """Plant a defect: noise added to layer K's residual at long positions only; the classifier must name layer K and the right type."""
    rng = np.random.default_rng(0)
    nl, H = 12, 64
    types = ["sliding_attention" if (l % 4) != 3 else "full_attention" for l in range(nl)]
    positions = [10, 300, 900, 1000, 1030, 1100, 1440]
    base = [[rng.standard_normal(H) for _ in positions] for _ in range(nl)]
    for plant_layer in (5, 7):  # 5 is sliding, 7 is full
        for noise_scale, expect_first in ((0.0, None), (0.5, plant_layer)):
            C = []
            for l in range(nl):
                row = []
                for i, p in enumerate(positions):
                    g = base[l][i].copy()
                    if l >= plant_layer and p > SHORT_MAX:  # a defect at the plant layer persists in the residual
                        g = g + noise_scale * rng.standard_normal(H)
                    row.append(cos(g, base[l][i]))
                C.append(row)
            cls, first, _, _ = classify(C, positions, types)
            want_cls = "N" if expect_first is None else ("W" if types[plant_layer].startswith("sliding") else "F")
            assert first == expect_first and cls == want_cls, f"plant {plant_layer} noise {noise_scale}: got {cls} {first}, want {want_cls} {expect_first}"
            print(f"selftest ok: plant layer {plant_layer} ({types[plant_layer]}) noise {noise_scale} -> class {cls}, first excess {first}")
    # and the hook plumbing on a tiny random HF model, if this transformers has Mellum
    try:
        import torch
        from transformers import MellumConfig, MellumForCausalLM
    except Exception as e:
        print("selftest: Mellum not importable here, skipped the hook check:", e)
        return
    cfg = MellumConfig(vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=4, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2, max_position_embeddings=256, sliding_window=8,
                       layer_types=["sliding_attention"] * 3 + ["full_attention"], mlp_layer_types=["sparse"] * 4)
    m = MellumForCausalLM(cfg).eval()
    ids = list(range(1, 40))
    pos = [3, 20, 38]
    a = hf_dump(m, ids, pos, 4)
    b = hf_dump(m, ids, pos, 4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "two hooked runs of one model differ"
    assert a[0].shape == (3, 64), a[0].shape
    print("selftest ok: hooks fire on all 4 layers of a tiny Mellum, residual shape", a[0].shape)


# Follow-up A2 (docs/tasks/task-mellum21-2026-10.md): the three craters follow-up A found in the int8int8 dump, as (position, layer):
# position 1030 at layer 5, position 1300 at layers 10 and 15. r = (1 - cos in the new arm) / (1 - cos in the int8int8 arm).
CRATERS = [(1030, 5), (1300, 10), (1300, 15)]


def compare(base_json, arm_json):
    """Registered rule: Q if every crater has r <= 0.25 (the craters are quantization sensitivity); P if every r >= 0.75 (the craters
    survive without that quantization: the window path, or a difference present in both arms); M otherwise. Control: at the short
    positions 300 and 600 the arm's last-layer cosine must be no worse than the int8int8 arm's, or the arm is suspect (class X)."""
    b, a = json.load(open(base_json)), json.load(open(arm_json))
    pos = b["positions"]
    assert pos == a["positions"], "the two dumps are not of the same positions"
    rs = []
    for p, l in CRATERS:
        i = pos.index(p)
        cb, ca = b["cos"][l][i], a["cos"][l][i]
        r = (1 - ca) / (1 - cb)
        rs.append(r)
        print(f"crater position {p} layer {l}: int8int8 cos {cb:.5f}, arm cos {ca:.5f}, r = {r:.3f}")
    last = len(b["cos"]) - 1
    ctl = [(p, b["cos"][last][pos.index(p)], a["cos"][last][pos.index(p)]) for p in (300, 600)]
    for p, cb, ca in ctl:
        print(f"control position {p} last layer: int8int8 {cb:.5f}, arm {ca:.5f}")
    if any(ca < cb - 1e-4 for _, cb, ca in ctl):
        print("CLASS X: the arm is worse than int8int8 at the short control positions: suspect, no reading")
    elif all(r <= 0.25 for r in rs):
        print("CLASS Q: every crater is gone without int8 activations/weights: quantization sensitivity, not the window path")
    elif all(r >= 0.75 for r in rs):
        print("CLASS P: every crater survives the arm: the window path (or a difference common to both arms) is the cause")
    else:
        print("CLASS M: mixed: " + ", ".join(f"{r:.2f}" for r in rs) + " (parked)")


def selftest_compare():
    import tempfile
    pos = [300, 600, 1030, 1300]
    def mk(c):
        C = [[0.9999] * 4 for _ in range(28)]
        for (p, l), v in c.items():
            C[l][pos.index(p)] = v
        return {"cos": C, "positions": pos}
    base = mk({(1030, 5): 0.9940, (1300, 10): 0.9700, (1300, 15): 0.9250})
    cases = {"CLASS Q": mk({(1030, 5): 0.9999, (1300, 10): 0.9990, (1300, 15): 0.9990}),
             "CLASS P": mk({(1030, 5): 0.9941, (1300, 10): 0.9705, (1300, 15): 0.9260}),
             "CLASS M": mk({(1030, 5): 0.9999, (1300, 10): 0.9850, (1300, 15): 0.9500}),
             "CLASS X": mk({(1030, 5): 0.9999, (1300, 10): 0.9990, (1300, 15): 0.9990})}
    cases["CLASS X"]["cos"][27][0] = 0.9900
    import io, contextlib
    for want, arm in cases.items():
        with tempfile.TemporaryDirectory() as d:
            json.dump(base, open(d + "/b.json", "w")); json.dump(arm, open(d + "/a.json", "w"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                compare(d + "/b.json", d + "/a.json")
            assert want in buf.getvalue(), (want, buf.getvalue())
    print("selftest ok: compare returns Q, P, M and X on the four synthetic arms")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        selftest()
    elif len(sys.argv) == 6 and sys.argv[1] == "run":
        run(*sys.argv[2:])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "selftest-compare":
        selftest_compare()
    else:
        print(__doc__)
        sys.exit(2)

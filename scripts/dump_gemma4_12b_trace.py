#!/usr/bin/env python3
"""Dump a layer-wise activation trace of the Gemma 4 12B HF reference (bf16, CPU)
for the K=V debug. Captures: per-layer residual stream (last token), final logits,
and the q/k/v projection+norm outputs at layers 4 (last local) and 5 (first global)
so the goinfer forward can be diffed layer-by-layer.

    ~/g4venv/bin/python scripts/dump_gemma4_12b_trace.py
"""
import json, os, numpy as np, torch

PATH = os.path.expanduser("~/models/gemma-4-12b-unq")
OUT = os.path.expanduser("~/mycode/goinfer/testdata/gemma4_12b_trace.json")
PROMPT = "The capital of France is"
PROBE_LAYERS = [4, 5]  # last sliding, first global

from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(PATH)
ids = tok(PROMPT, return_tensors="pt").input_ids
print("ids:", ids.tolist())

model = AutoModelForCausalLM.from_pretrained(
    PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True
).eval()

# locate the decoder layer stack (unified multimodal -> language_model)
def find_layers(m):
    for attr in ("model", "language_model"):
        m2 = getattr(m, attr, None)
        if m2 is not None:
            r = find_layers(m2)
            if r is not None:
                return r
    if hasattr(m, "layers"):
        return m.layers
    return None

layers = find_layers(model)
print("found", len(layers), "decoder layers")

# hook q/k/v proj + norm outputs at the probe layers
probe = {}
def mk(name):
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        probe[name] = t.detach()[0, -1].float().numpy().copy()
    return hook
for li in PROBE_LAYERS:
    sa = layers[li].self_attn
    for sub in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "v_norm"):
        m = getattr(sa, sub, None)
        if m is not None:
            m.register_forward_hook(mk(f"L{li}.{sub}"))

with torch.no_grad():
    out = model(input_ids=ids, output_hidden_states=True)

logits = out.logits[0, -1].float().numpy()
hs = out.hidden_states  # tuple len = nlayers+1; [0]=embed output, [i]=after layer i-1
order = np.argsort(-logits)
print("final argmax:", int(order[0]), repr(tok.decode([int(order[0])])),
      "logit", float(logits[order[0]]))

trace = {
    "model_id": "google/gemma-4-12B-it-qat-q4_0-unquantized",
    "prompt": PROMPT,
    "ids": ids[0].tolist(),
    "argmax": int(order[0]),
    "argmax_token": tok.decode([int(order[0])]),
    "top_k": [[int(i), float(logits[i])] for i in order[:16]],
    # residual stream at last position, per layer index (0 = post-embedding)
    "hidden_last": {str(i): h[0, -1].float().numpy().tolist() for i, h in enumerate(hs)},
    "hidden_norms": {str(i): float(np.linalg.norm(h[0, -1].float().numpy())) for i, h in enumerate(hs)},
    "probe": {k: v.tolist() for k, v in probe.items()},
    "probe_shapes": {k: list(v.shape) for k, v in probe.items()},
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(trace, open(OUT, "w"))
print("hidden_norms:", {k: round(v, 2) for k, v in trace["hidden_norms"].items()})
print("probe keys:", list(probe.keys()), "shapes:", trace["probe_shapes"])
print("wrote", OUT, f"({os.path.getsize(OUT)/1e6:.1f} MB)")

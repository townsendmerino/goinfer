#!/usr/bin/env python
"""Gate-1 reference: HF f32 forward of the REAL Qwen3.6-35B-A3B (qwen3_5_moe)
TEXT decoder, truncated to embed + layers 0-3, dumping the last-position residual
stream AFTER layer 3 (pre-final-norm) — the exact point goinfer's runLayers
returns.

Cheap by construction: a 4-layer Qwen3_5MoeForCausalLM (built from the real
text_config with num_hidden_layers=4) loaded from only the 4 shards that hold the
slice — no offload, no 35B in RAM, ~17 GB f32. f32 on BOTH sides so the 1-1e-5
bar is honest (see docs/task-qwen35-realckpt.md, "Dtype honesty").

    ~/g4venv/bin/python scripts/pin_qwen35_slice.py
    -> ~/models/qwen35_real_golden/slice4_f32.json   {prompt_ids, hidden}
"""
import json
import os
import re

import torch
from safetensors import safe_open
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

MODEL = os.path.expanduser(os.environ.get("GOINFER_QWEN35_REAL",
                                          "~/models/qwen3.6-35b-a3b"))
OUT = os.path.expanduser(os.environ.get("GOINFER_QWEN35_SLICE_REF",
                                        "~/models/qwen35_real_golden/slice4_f32.json"))
N_LAYER = 4
# Gate-2 golden prompt00 ("The capital of France is"); same ids goinfer prefills.
PROMPT_IDS = [760, 6511, 314, 9338, 369]


def main():
    cfg_full = json.load(open(os.path.join(MODEL, "config.json")))
    tc = dict(cfg_full["text_config"])
    tc["num_hidden_layers"] = N_LAYER
    tc["layer_types"] = tc["layer_types"][:N_LAYER]
    config = Qwen3_5MoeTextConfig(**tc)

    # Build eagerly (not on meta) so computed buffers — the mrope rotary_emb
    # inv_freq, absent from the checkpoint — are materialized; ~17 GB f32, fits.
    torch.set_default_dtype(torch.float32)
    model = Qwen3_5MoeForCausalLM(config)
    print(f"built 4-layer text model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # Map each parameter of the truncated text model to its real-checkpoint name:
    # the VL checkpoint nests the decoder under model.language_model.* (lm_head
    # stays top-level). Load only those tensors, from the shards that hold them.
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    want = set(model.state_dict().keys())

    def real_name(p):  # text-model param name -> real checkpoint tensor name
        if p == "lm_head.weight":
            return p
        if p.startswith("model."):
            return "model.language_model." + p[len("model."):]
        return p

    # Group the tensors we need by shard so each shard opens once.
    by_shard = {}
    for p in want:
        rn = real_name(p)
        if rn not in idx:
            continue  # tie or otherwise absent; load_state_dict(strict=False) tolerates
        by_shard.setdefault(idx[rn], {})[p] = rn

    state = {}
    for shard, params in by_shard.items():
        with safe_open(os.path.join(MODEL, shard), framework="pt") as f:
            for p, rn in params.items():
                state[p] = f.get_tensor(rn).to(torch.float32)
    print(f"loaded {len(state)} tensors from {len(by_shard)} shards")

    missing, unexpected = model.load_state_dict(state, strict=False)
    # The slice path (embed + 4 layers + final norm) must be fully real; lm_head
    # may stay random (untied, unused for hidden_states[4]).
    need = re.compile(r"^model\.(embed_tokens|norm)\.|^model\.layers\.[0-3]\.")
    bad = [n for n in missing if need.match(n)]
    if bad:
        raise SystemExit(f"slice params not loaded from checkpoint: {bad[:6]} ...")
    model.eval()

    ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
    # goinfer's runLayers returns the residual stream AFTER the last layer but
    # BEFORE the final model.norm. HF's output_hidden_states applies model.norm to
    # the LAST entry (the standard decoder loop appends post-norm), so comparing
    # against hidden_states[N] is wrong — it folds in a ||1+w||≈119 RMSNorm gain.
    # Capture the pre-norm input to model.norm directly instead.
    pre_norm = {}
    h_pre = model.model.norm.register_forward_pre_hook(
        lambda mod, inp: pre_norm.__setitem__("x", inp[0].detach()))
    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True, use_cache=False)
    h_pre.remove()
    hidden = pre_norm["x"][0, -1, :].to(torch.float32).cpu().numpy()
    print(f"hidden after layer {N_LAYER-1} (pre-final-norm): shape {hidden.shape}  ||h||={float((hidden**2).sum()**0.5):.4f}  h[:4]={hidden[:4]}")

    # Intermediate boundaries hidden_states[0..N-1] are pre-norm already (the final
    # norm only touches the last entry); replace that last entry with the captured
    # pre-norm hidden so hidden_all[k] is uniformly the pre-norm state after layer
    # k-1 (k=0 → embeddings) — what goinfer's N-layer slice returns.
    hidden_all = [[float(x) for x in hs[0, -1, :].to(torch.float32).cpu().numpy()]
                  for hs in out.hidden_states]
    hidden_all[N_LAYER] = [float(x) for x in hidden]
    for k, hs in enumerate(hidden_all):
        n = sum(x * x for x in hs) ** 0.5
        print(f"  boundary {k} (||h||={n:.4f}) {'embeddings' if k==0 else f'after layer {k-1}'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"prompt_ids": PROMPT_IDS, "hidden": [float(x) for x in hidden],
               "hidden_all": hidden_all},
              open(OUT, "w"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

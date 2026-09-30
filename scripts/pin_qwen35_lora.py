#!/usr/bin/env python3
"""HF + PEFT reference for LoRA merge-at-load on Qwen3.5 (D3, docs/tasks/task-constrained-confidence.md).

    ~/.venv-peft/bin/python scripts/pin_qwen35_lora.py   ->  decoder/testdata/qwen3_5-tiny-lora/  (the adapter)
                                                           decoder/testdata/qwen35_lora_golden.json

peft is not in the default environment; the venv is `python3 -m venv --system-site-packages ~/.venv-peft &&
~/.venv-peft/bin/pip install peft` (pinned below by what the golden records).

The adapter is written by PEFT itself, from the EXISTING tiny checkpoint (not regenerated), so its tensor names are
PEFT's, not a guess at them. It targets every module autotrust's JEV recipe targets: the DeltaNet in_proj_qkv, in_proj_z
and out_proj, the full-attention q/k/v/o_proj, and gate/up/down_proj, on all four layers (three linear-attention, one
full-attention). PEFT initialises lora_B to zero, which would make the adapter a no-op and the gate vacuous, so A and B are
redrawn from a seeded generator, sized so each delta's RMS is about the base weight's own. The reference is PEFT's
merge_and_unload() forward: the final-norm hidden state at the last prompt position (what decoder.Model.PromptHidden
returns), five prompts of 2..64 tokens (never 1, where attention is the identity). The base model's vectors are recorded
too, so the test can show the adapter moves them far more than the tolerance.
"""
import json, os
import torch
import peft
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "decoder", "testdata")
BASE = os.path.join(TD, "qwen3_5-tiny")
ADAPTER = os.path.join(TD, "qwen3_5-tiny-lora")
OUT = os.path.join(TD, "qwen35_lora_golden.json")
TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
R, ALPHA = 4, 8
LENGTHS = [2, 5, 13, 32, 64]


def last_hidden(model, ids):
    with torch.no_grad():
        out = model(torch.tensor([ids]), output_hidden_states=True)
    h = out.hidden_states[-1][0, -1]
    return [float(x) for x in h]


base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
vocab = base.config.vocab_size
g = torch.Generator().manual_seed(20260930)
prompts = [torch.randint(0, vocab, (n,), generator=g).tolist() for n in LENGTHS]
base_h = [last_hidden(base, p) for p in prompts]

pm = get_peft_model(base, LoraConfig(r=R, lora_alpha=ALPHA, target_modules=TARGETS, lora_dropout=0.0, bias="none"))
scale = ALPHA / R
hit = []
with torch.no_grad():
    for name, mod in pm.named_modules():
        if not hasattr(mod, "lora_A") or "default" not in mod.lora_A:
            continue
        A, B, W = mod.lora_A["default"].weight, mod.lora_B["default"].weight, mod.base_layer.weight
        # delta = scale * B @ A, each entry a sum of R products: std(A)=std(B)=s gives delta RMS ~ scale*sqrt(R)*s^2
        s = (float(W.std()) / (scale * R ** 0.5)) ** 0.5
        A.copy_(torch.randn(A.shape, generator=g) * s)
        B.copy_(torch.randn(B.shape, generator=g) * s)
        hit.append(name)
kinds = sorted({n.rsplit(".", 1)[-1] for n in hit})
assert kinds == sorted(TARGETS), f"PEFT wrapped {kinds}, want every target in {sorted(TARGETS)}"
pm.save_pretrained(ADAPTER, safe_serialization=True)
os.chmod(os.path.join(ADAPTER, "adapter_model.safetensors"), 0o644)
os.remove(os.path.join(ADAPTER, "README.md"))  # PEFT's generated model card: boilerplate, not fixture

merged = pm.merge_and_unload().eval()
ref_h = [last_hidden(merged, p) for p in prompts]

gold = {
    "base": "qwen3_5-tiny",
    "adapter": "qwen3_5-tiny-lora",
    "r": R, "lora_alpha": ALPHA, "targets": TARGETS, "wrapped_modules": len(hit),
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    "prompts": [{"ids": p, "hidden": h, "base_hidden": b} for p, h, b in zip(prompts, ref_h, base_h)],
}
with open(OUT, "w") as f:
    json.dump(gold, f)
print(f"wrapped {len(hit)} modules ({', '.join(kinds)}); wrote {ADAPTER} and {OUT}")

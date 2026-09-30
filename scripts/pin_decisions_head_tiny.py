#!/usr/bin/env python3
"""Route B reference on a tiny JEV-shaped judge (D4, docs/tasks/task-constrained-confidence.md).

    ~/.venv-peft/bin/python scripts/pin_decisions_head_tiny.py  ->  testdata/decisions/judge-tiny/  (the judge dir)
                                                                    testdata/decisions/head_tiny_golden.json

The judge dir has JEV-9B's layout: judge_config.json (24 slots: noul 0-2, score 2-8, choice 8-24, bare-v1, weights_mode
unmerged, adapter_subfolder adapter), head.safetensors (proj.weight [24, 64], proj.bias [24], random, seeded),
calibration.json (temperatures away from 1, so a missing division shows) and adapter/ (decoder/testdata/qwen3_5-tiny-lora,
the PEFT adapter D3's gate uses). The backbone is decoder/testdata/qwen3_5-tiny, which has no tokenizer, so each case's
token ids are drawn from a seeded generator; the Go test serves them for the prompt text, which it also checks against
its own renderer.

The reference is jev_core.decide() as pin_decisions_d0.py ports it: the adapter UNMERGED (PeftModel over the f32 base),
the last token's post-final-norm hidden state, W @ h + b, the kind's first n slots, z / T, softmax.
"""
import json, os, shutil, sys
import torch
import peft
import transformers
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pin_decisions_d0 import build_decision_prompt  # noqa: E402  (jev_core's template, verbatim in behaviour)

ROOT = os.path.join(HERE, "..")
BASE = os.path.join(ROOT, "decoder", "testdata", "qwen3_5-tiny")
LORA = os.path.join(ROOT, "decoder", "testdata", "qwen3_5-tiny-lora")
JUDGE = os.path.join(ROOT, "testdata", "decisions", "judge-tiny")
OUT = os.path.join(ROOT, "testdata", "decisions", "head_tiny_golden.json")
VERB = ["false", "true", "0", "1", "2", "3", "4", "5"] + list("ABCDEFGHIJKLMNOP")
RANGES = {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}
TEMPS = {"noul": 1.3, "choice": 0.7, "score": 1.1}
CASES = [  # kind, n options, prompt length in tokens (never 1: attention is the identity there)
    ("noul", 2, 7), ("noul", 2, 40), ("score", 6, 23), ("choice", 3, 17), ("choice", 16, 55),
]

torch.set_grad_enabled(False)
g = torch.Generator().manual_seed(20260930)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
hidden, vocab = base.config.hidden_size, base.config.vocab_size

os.makedirs(os.path.join(JUDGE, "adapter"), exist_ok=True)
for fn in ("adapter_config.json", "adapter_model.safetensors"):
    shutil.copyfile(os.path.join(LORA, fn), os.path.join(JUDGE, "adapter", fn))
W = torch.randn(24, hidden, generator=g) * 0.3
b = torch.randn(24, generator=g) * 0.5
save_file({"proj.weight": W.contiguous(), "proj.bias": b.contiguous()}, os.path.join(JUDGE, "head.safetensors"))
os.chmod(os.path.join(JUDGE, "head.safetensors"), 0o644)
judge = {
    "hidden_size": hidden,
    "slots": {"num_slots": 24, "ranges": RANGES, "verbalizers": VERB, "template_version": "bare-v1"},
    "verbalizer_ids": list(range(100, 124)),  # Route B never reads them; the tiny checkpoint has no tokenizer
    "kinds": ["noul", "choice", "score"],
    "model_name": "judge-tiny", "model_version": "0.0.1",
    "weights_mode": "unmerged", "adapter_subfolder": "adapter", "softcap": None,
}
json.dump(judge, open(os.path.join(JUDGE, "judge_config.json"), "w"), indent=2)
json.dump({"version": 1, "per_kind": TEMPS, "per_kind_family": {}}, open(os.path.join(JUDGE, "calibration.json"), "w"), indent=2)

model = PeftModel.from_pretrained(base, os.path.join(JUDGE, "adapter")).eval()  # UNMERGED, as the Space
causal = model.base_model.model
cases = []
for i, (kind, n, length) in enumerate(CASES):
    opts = ["false", "true"] if kind == "noul" else [str(v) for v in range(6)] if kind == "score" else [f"option {j}" for j in range(n)]
    state, question = f"case {i}: the state of the world", f"question {i}?"
    prompt, labels = build_decision_prompt(kind, state, question, opts)
    ids = torch.randint(0, vocab, (length,), generator=g).tolist()
    h = causal.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1].float()
    lo = RANGES[kind][0]
    z = (W @ h + b)[lo:lo + n]
    cases.append({"kind": kind, "state": state, "question": question, "options": opts, "prompt": prompt, "ids": ids,
                  "logits": z.tolist(), "T": TEMPS[kind], "p": torch.softmax(z / TEMPS[kind], 0).tolist()})
gold = {"judge": "judge-tiny", "base": "qwen3_5-tiny", "reference": "PEFT unmerged, f32",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "cases": cases}
json.dump(gold, open(OUT, "w"), indent=1)
print(f"wrote {JUDGE} and {OUT}: {len(cases)} cases")
for c in cases:
    print(f"  {c['kind']:6s} n={len(c['options']):2d} len={len(c['ids']):2d} p={[round(v, 3) for v in c['p']][:6]}")

#!/usr/bin/env python3
"""HF reference for decoder.Model.PromptHidden (D2, docs/tasks/task-constrained-confidence.md): the final-norm hidden state at the
last prompt position, output_hidden_states[-1][0, -1], from the EXISTING tiny checkpoints (not regenerated), in f32.

    python3 scripts/pin_prompt_hidden.py   ->  decoder/testdata/prompt_hidden_golden.json

Five prompts per checkpoint, token ids fixed by a seeded generator, lengths 2..64: never length 1, where softmax attention is the
identity and a broken attention path would still pass (CLAUDE.md, "a minimal repro can be minimal in exactly the dimension that
hides the bug"). The tiny checkpoints' final-norm weights are all 0, which Qwen3.5's add-one RMSNorm makes a scale of exactly 1,
so their reference vectors have RMS 1 and neither cosine nor scale can show a missing, doubled or mis-weighted final norm. So the
script also writes qwen3_5-tiny-normw/, the dense checkpoint with a random final-norm weight (everything else identical), and
pins it first. The script asserts that output_hidden_states[-1] equals last_hidden_state, i.e. that HF's last entry is already
past the final norm, instead of assuming it.
"""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "decoder", "testdata")
OUT = os.path.join(TD, "prompt_hidden_golden.json")
CKPTS = ["qwen3_5-tiny-normw", "qwen3_5-tiny", "qwen3_5_moe-tiny"]

# the derived fixture: the dense tiny checkpoint with a non-trivial final norm, so the gate can see it
_src = AutoModelForCausalLM.from_pretrained(os.path.join(TD, "qwen3_5-tiny"), dtype=torch.float32).eval()
with torch.no_grad():
    _src.model.norm.weight.copy_(torch.randn(_src.model.norm.weight.shape, generator=torch.Generator().manual_seed(7)) * 0.5)
_src.save_pretrained(os.path.join(TD, "qwen3_5-tiny-normw"), safe_serialization=True)
print("wrote qwen3_5-tiny-normw: final-norm weight", float(_src.model.norm.weight.min()), "..", float(_src.model.norm.weight.max()))
LENGTHS = [2, 5, 13, 32, 64]

torch.manual_seed(0)
out = {"transformers": __import__("transformers").__version__, "torch": torch.__version__, "dtype": "float32", "fixtures": []}
for name in CKPTS:
    path = os.path.join(TD, name)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    vocab = m.config.vocab_size
    g = torch.Generator().manual_seed(1234)
    prompts = []
    for L in LENGTHS:
        ids = torch.randint(0, vocab, (1, L), generator=g)
        with torch.no_grad():
            o = m(ids, output_hidden_states=True)
            base = m.model(ids).last_hidden_state
        last = o.hidden_states[-1][0, -1]
        if not torch.allclose(last, base[0, -1], atol=0, rtol=0):
            sys.exit(f"{name}: output_hidden_states[-1] is not last_hidden_state; the last entry is not post-final-norm here")
        prompts.append({"ids": ids[0].tolist(), "hidden": [float(x) for x in last]})
    out["fixtures"].append({"checkpoint": name, "model_type": m.config.model_type, "hidden_size": m.config.hidden_size, "prompts": prompts})
    print(f"{name}: {m.config.model_type}, hidden {m.config.hidden_size}, {len(prompts)} prompts")
json.dump(out, open(OUT, "w"), indent=1)
print("->", OUT, os.path.getsize(OUT), "bytes")

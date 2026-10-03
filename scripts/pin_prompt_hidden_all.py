#!/usr/bin/env python3
"""HF reference for decoder.Model.PromptHiddenAll (D11, docs/tasks/task-constrained-confidence.md Route C): the final-norm hidden state at EVERY
prompt position, last_hidden_state[0, :], from the EXISTING tiny checkpoints, in f32. The Clef joint head reads the text model's last_hidden_state at
all positions (decisions-d10-clef-2026-10-02.md), so this is what D12 feeds it.

    python3 scripts/pin_prompt_hidden_all.py   ->  decoder/testdata/prompt_hidden_all_golden.json

The same checkpoints and the same five prompts per checkpoint as D2's golden (prompt_hidden_golden.json: seeded generator, lengths 2..64), asserted equal
to it here, so the two goldens cannot drift apart. One extra prompt of length 1 per checkpoint, from its own seed: it checks the shape of a single-row result
and is NOT a correctness bar (softmax over one key is the identity, so a broken attention path would still pass it; CLAUDE.md, "a minimal repro can be minimal
in exactly the dimension that hides the bug"). The script asserts that last_hidden_state is the last entry of output_hidden_states, i.e. that HF's last_hidden_state
is past the final norm, instead of assuming it. qwen3_5-tiny-normw (random final-norm weight) is written here if its model.safetensors is missing (see below).
"""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "..", "decoder", "testdata")
OUT = os.path.join(TD, "prompt_hidden_all_golden.json")
D2 = os.path.join(TD, "prompt_hidden_golden.json")
CKPTS = ["qwen3_5-tiny-normw", "qwen3_5-tiny", "qwen3_5_moe-tiny"]
LENGTHS = [2, 5, 13, 32, 64]

# The derived fixture, written exactly as scripts/pin_prompt_hidden.py writes it (the dense tiny checkpoint with a seeded random final-norm weight) when its model.safetensors is
# absent. D2 committed the directory's config files but not this file, so on a fresh checkout D2's gate skipped the one fixture that can see a missing or doubled final norm;
# D11 commits it. Running this does NOT rewrite D2's golden: if the regenerated weights were not D2's, D2's test (TestPromptHidden_matchesHF/qwen3_5-tiny-normw) fails.
_normw = os.path.join(TD, "qwen3_5-tiny-normw")
if not os.path.exists(os.path.join(_normw, "model.safetensors")):
    _src = AutoModelForCausalLM.from_pretrained(os.path.join(TD, "qwen3_5-tiny"), dtype=torch.float32).eval()
    with torch.no_grad():
        _src.model.norm.weight.copy_(torch.randn(_src.model.norm.weight.shape, generator=torch.Generator().manual_seed(7)) * 0.5)
    _src.save_pretrained(_normw, safe_serialization=True)
    print("wrote qwen3_5-tiny-normw/model.safetensors: final-norm weight", float(_src.model.norm.weight.min()), "..", float(_src.model.norm.weight.max()))

d2 = {f["checkpoint"]: f for f in json.load(open(D2))["fixtures"]}
torch.manual_seed(0)
out = {"transformers": __import__("transformers").__version__, "torch": torch.__version__, "dtype": "float32", "fixtures": []}
for name in CKPTS:
    path = os.path.join(TD, name)
    if not os.path.exists(os.path.join(path, "model.safetensors")):
        sys.exit(f"{path} has no model.safetensors")
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    vocab = m.config.vocab_size
    g = torch.Generator().manual_seed(1234)
    prompts = []
    for L in LENGTHS:
        ids = torch.randint(0, vocab, (1, L), generator=g)
        with torch.no_grad():
            o = m(ids, output_hidden_states=True)
            base = m.model(ids).last_hidden_state
        if not torch.allclose(o.hidden_states[-1], base, atol=0, rtol=0):
            sys.exit(f"{name}: output_hidden_states[-1] is not last_hidden_state; the last entry is not post-final-norm here")
        want = d2[name]["prompts"][len(prompts)]["ids"]
        if ids[0].tolist() != want:
            sys.exit(f"{name}: the prompt ids differ from D2's golden at length {L}: the seeds drifted")
        prompts.append({"ids": ids[0].tolist(), "hidden": [[float(x) for x in row] for row in base[0]], "shape_only": False})
    g1 = torch.Generator().manual_seed(99)
    ids1 = torch.randint(0, vocab, (1, 1), generator=g1)
    with torch.no_grad():
        base1 = m.model(ids1).last_hidden_state
    prompts.append({"ids": ids1[0].tolist(), "hidden": [[float(x) for x in row] for row in base1[0]], "shape_only": True})
    out["fixtures"].append({"checkpoint": name, "model_type": m.config.model_type, "hidden_size": m.config.hidden_size, "prompts": prompts})
    print(f"{name}: {m.config.model_type}, hidden {m.config.hidden_size}, {len(prompts)} prompts ({sum(len(p['ids']) for p in prompts)} positions)")
json.dump(out, open(OUT, "w"))
print("->", OUT, os.path.getsize(OUT), "bytes")

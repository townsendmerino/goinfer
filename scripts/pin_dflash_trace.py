#!/usr/bin/env python
"""Per-position reference dump for the DFlash drafter — P10 increment 2's fixture, and
the input to kill-gate 1 (the Go forward must match this before any acceptance run; the
05 lesson, priced in).

Runs ONE drafting round of z-lab/Qwen3-4B-DFlash-b16 against its paired target
Qwen/Qwen3-4B, through the checkpoint's OWN shipped reference implementation
(`dflash.py`, MIT, trust_remote_code) — the parity oracle, not a reimplementation.
Everything is f32 and greedy, so the dump is deterministic and reproducible; the prompt is
fixed and recorded in the JSON.

Dumps, per position, the tensors a Go forward has to reproduce in order:

    fused_context   hidden_norm(fc(concat h[1],h[9],h[17],h[25],h[33]))   [ctx, 2560]
    block_in        target.embed_tokens(block_ids)                        [16, 2560]
    layer_out.{0..4}  each decoder layer's output                         [16, 2560]
    trunk_out       norm(layer_out.4)                                     [16, 2560]
    draft_logits    target.lm_head(trunk_out[-15:])  argmax + top-8       [15, ...]

    ~/.venv-vl/bin/python scripts/pin_dflash_trace.py
    -> testdata/dflash_qwen3_4b_golden.json  (committed; weights are NOT)
       (named *_golden not *_trace: .gitignore excludes testdata/*_trace.json as
        per-machine layer-dump output, and this fixture is meant to be tracked)
"""
import json
import os
import sys
import types

import torch

# The checkpoint's utils.py does `from datasets import load_dataset, Features, Sequence,
# Value` at MODULE scope, purely for load_and_process_dataset() — an eval-harness helper
# this trace never calls. ~/.venv-vl (torch 2.12.0+cpu, safetensors 0.8.0, transformers
# 5.12.0 — checked, and what 05 used) does not carry `datasets`, and installing it would
# drag pyarrow/fsspec into the venv every other parity gate on this box shares.
#
# So we satisfy the import instead of installing it, and the REFERENCE CODE IS RUN
# UNMODIFIED — that is the whole point of using it as the oracle. If a future trace needs
# a real dataset, build a separate venv rather than reaching for this stub.
if "datasets" not in sys.modules:
    _stub = types.ModuleType("datasets")
    for _name in ("load_dataset", "Features", "Sequence", "Value"):
        setattr(_stub, _name, None)
    sys.modules["datasets"] = _stub

from safetensors.torch import save_file  # noqa: E402
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

DRAFT = os.path.expanduser("~/models/qwen3-4b-dflash")
TARGET = os.path.expanduser("~/models/qwen3-4b")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "testdata", "dflash_qwen3_4b_golden.json")
REF = os.path.join(HERE, "..", "testdata", "dflash_qwen3_4b_ref.safetensors")
PROMPT = "The capital of France is"
# The raw prompt above is deliberately off-distribution for an instruction-tuned target;
# the chat-templated one is what real traffic looks like. Both are dumped, because a
# fixture drawn only from a bare completion prompt is how this repo once manufactured a
# false "the quantizer is broken" signal that survived a week.
CHAT_PROMPT = "Write a Python function that returns the nth Fibonacci number."
TOPK = 8


def trace_one(tok, target, draft, ids, name):
    B = int(draft.block_size)
    tlids = list(draft.target_layer_ids)

    with torch.no_grad():
        # 1. Target prefill. extract_context_feature indexes hidden_states[layer_id + 1],
        #    so hidden_states[0] (the embedding output) is layer_id -1 — the +1 offset is
        #    the wiring detail a Go port gets wrong silently.
        out = target(ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        ctx_cat = torch.cat([hs[i + 1] for i in tlids], dim=-1)      # [1, ctx, 5*2560]
        first = int(out.logits[0, -1].argmax())                       # greedy anchor token

        # 2. The drafter's own fusion of that context.
        fused = draft.hidden_norm(draft.fc(ctx_cat))                  # [1, ctx, 2560]

        # 3. Block: slot 0 = the anchor (a real token), slots 1..B-1 = MASK.
        block_ids = torch.full((1, B), int(draft.mask_token_id), dtype=torch.long)
        block_ids[0, 0] = first
        block_in = target.model.embed_tokens(block_ids)               # TARGET's embedding

        # 4. Trunk, capturing every layer output. Positions are absolute: the block sits
        #    right after the context, and rotary is built over [0, ctx+B).
        ctx_len = ids.shape[1]
        pos = torch.arange(ctx_len + B).unsqueeze(0)
        layer_outs = []
        h = block_in
        fused_ctx = fused
        pe = draft.rotary_emb(h, pos)
        for layer in draft.layers:
            h = layer(hidden_states=h, target_hidden=fused_ctx, attention_mask=None,
                      position_ids=pos, position_embeddings=pe, use_cache=False)
            layer_outs.append(h)
        trunk = draft.norm(h)                                         # [1, B, 2560]

        # 5. Draft logits through the TARGET's lm_head, from slot 1 (slot 0 is the anchor).
        logits = target.lm_head(trunk[:, -B + 1:, :])                 # [1, B-1, vocab]
        drafted = logits[0].argmax(-1)
        topk = torch.topk(logits[0], TOPK, dim=-1)

    def stats(t):
        f = t.reshape(-1).double()
        return dict(shape=list(t.shape), mean=float(f.mean()), std=float(f.std()),
                    min=float(f.min()), max=float(f.max()),
                    first8=[float(x) for x in t.reshape(-1)[:8]])

    g = dict(
        name=name,
        prompt_ids=ids[0].tolist(), prompt_text=tok.decode(ids[0]), block_size=B,
        target_layer_ids=tlids, mask_token_id=int(draft.mask_token_id),
        hidden_states_offset=1,
        anchor_token=first, anchor_text=tok.decode([first]),
        block_ids=block_ids[0].tolist(),
        tensors=dict(
            fused_context=stats(fused[0]), block_in=stats(block_in[0]),
            **{f"layer_out.{i}": stats(t[0]) for i, t in enumerate(layer_outs)},
            trunk_out=stats(trunk[0]),
        ),
        drafted_ids=[int(x) for x in drafted],
        drafted_text=tok.decode([int(x) for x in drafted]),
        draft_logits_topk=[[[int(i), float(v)] for v, i in zip(topk.values[p].tolist(), topk.indices[p].tolist())]
                           for p in range(logits.shape[1])],
        draft_logits_pos0_first64=[float(x) for x in logits[0, 0, :64]],
    )
    print(f"[{name}] ctx={ids.shape[1]} anchor={first} ({g['anchor_text']!r})")
    print(f"[{name}] drafted={g['drafted_ids']}")
    print(f"[{name}] drafted_text={g['drafted_text']!r}")

    # Full tensors for the Go parity gate. The JSON carries stats only (it stays
    # human-readable and diffable); the Go forward needs the actual INPUTS to run at all
    # — fused_context and block_in — and the per-layer outputs to localize a mismatch to
    # a layer instead of reporting "the logits are wrong". ~1 MB per trace, f32.
    ten = {f"{name}/fused_context": fused[0], f"{name}/block_in": block_in[0],
           f"{name}/trunk_out": trunk[0]}
    for i, t in enumerate(layer_outs):
        ten[f"{name}/layer_out.{i}"] = t[0]
    return g, {k: v.contiguous() for k, v in ten.items()}


def main():
    tok = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    draft = AutoModel.from_pretrained(
        DRAFT, dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True).eval()
    print(f"block_size={draft.block_size} target_layer_ids={list(draft.target_layer_ids)} "
          f"mask_token_id={draft.mask_token_id} attn={draft.config._attn_implementation}")

    raw_ids = tok(PROMPT, return_tensors="pt").input_ids
    chat_text = tok.apply_chat_template(
        [{"role": "user", "content": CHAT_PROMPT}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    chat_ids = tok(chat_text, return_tensors="pt").input_ids

    out = dict(
        note=("DFlash one-round reference traces (z-lab/Qwen3-4B-DFlash-b16 + Qwen/Qwen3-4B, "
              "f32, greedy) dumped through the checkpoint's own MIT reference implementation"),
        drafter="z-lab/Qwen3-4B-DFlash-b16", target="Qwen/Qwen3-4B", dtype="float32",
        traces=[],
    )
    tensors = {}
    for ids, nm in ((raw_ids, "raw"), (chat_ids, "chat")):
        g, ten = trace_one(tok, target, draft, ids, nm)
        out["traces"].append(g)
        tensors.update(ten)
    save_file(tensors, REF, metadata={"format": "pt"})
    print(f"saved {REF} ({len(tensors)} tensors)")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("saved", OUT)


if __name__ == "__main__":
    main()

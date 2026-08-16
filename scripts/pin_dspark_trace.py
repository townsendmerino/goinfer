#!/usr/bin/env python
"""Per-position reference dump for the DSpark block-7 drafter — P10 kill-gate 1 for DSpark.

Runs ONE drafting round of deepseek-ai/dspark_qwen3_4b_block7 against Qwen/Qwen3-4B through
DeepSpec's OWN modeling code (MIT), unmodified: `_forward_backbone`, `compute_logits`,
`sample_draft_tokens`, `predict_confidence_step`. f32, greedy, fixed prompts.

Dumped in the SAME tensor shape as the DFlash fixture on purpose. DSpark's trunk is not merely
similar to DFlash's — `_forward_backbone` is `hidden_norm(fc(ctx))` → rotary → 5 layers → `norm`,
its attention is the same non-causal cross-attention with `is_causal=False`, and its
`apply_rotary_pos_emb` (q takes `cos[..., -q_len:, :]`, k takes the full `cos`) is byte-identical
to DFlash's. So the same Go trunk should reproduce both, and this fixture is what proves it.

What DSpark adds beyond the trunk, and which this also dumps:
    base_logits   its OWN lm_head over all 7 block positions (logits_start = 0, unlike
                  DFlash's 1 — slot 0 both embeds the anchor AND predicts)
    markov        rank-256 chain: step_logits[i] = base[i] + w2(w1[prev]), SEQUENTIAL
    confidence    per-position accept-rate logit, gating adaptive block length

    ~/.venv-vl/bin/python scripts/pin_dspark_trace.py
    -> testdata/dspark_qwen3_4b_golden.json      (committed)
    -> testdata/dspark_qwen3_4b_ref.safetensors  (committed via .gitignore exception)

Checkpoint license=None; Francis accepted for exploration 2026-08-15 (see
docs/prompts/dspark-license-issue.md).
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

DEEPSPEC = os.environ.get(
    "DEEPSPEC_DIR",
    "/tmp/claude-1000/-home-francis-mycode-goinfer/"
    "758aa8a7-c89d-4413-90bb-6dc78b85eb48/scratchpad/DeepSpec",
)
sys.path.insert(0, DEEPSPEC)

DRAFT = os.path.expanduser("~/models/dspark-qwen3-4b")
TARGET = os.path.expanduser("~/models/qwen3-4b")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "testdata", "dspark_qwen3_4b_golden.json")
REF = os.path.join(HERE, "..", "testdata", "dspark_qwen3_4b_ref.safetensors")
PROMPT = "The capital of France is"
CHAT_PROMPT = "Write a Python function that returns the nth Fibonacci number."
TOPK = 8


def trace_one(tok, target, draft, ids, name):
    from deepspec.modeling.dspark.common import extract_context_feature

    B = int(draft.block_size)
    tlids = list(draft.target_layer_ids)
    with torch.no_grad():
        out = target(ids, output_hidden_states=True, use_cache=False)
        ctx_cat = extract_context_feature(out.hidden_states, tlids)
        anchor = int(out.logits[0, -1].argmax())

        fused = draft.hidden_norm(draft.fc(ctx_cat))
        blk = torch.full((1, B), int(draft.mask_token_id), dtype=torch.long)
        blk[0, 0] = anchor
        block_in = draft.embed_tokens(blk)          # DSpark's OWN embedding, not the target's

        ctx_len = ids.shape[1]
        pos = torch.arange(ctx_len + B).unsqueeze(0)
        pe = draft.rotary_emb(block_in, pos)
        h, layer_outs = block_in, []
        for layer in draft.layers:
            h = layer(hidden_states=h, target_hidden_states=fused, attention_mask=None,
                      position_ids=pos, position_embeddings=pe, use_cache=False)
            if isinstance(h, tuple):
                h = h[0]
            layer_outs.append(h)
        trunk = draft.norm(h)

        base = draft.compute_logits(trunk)          # all B positions: logits_start = 0
        sampled, corrected = draft.markov_head.sample_block_tokens(
            base, first_prev_token_ids=blk[:, 0], hidden_states=trunk, temperature=0.0)
        prev = torch.cat([blk[:, :1], sampled[:, :-1]], dim=1)
        conf = draft.predict_confidence_step(trunk, prev_token_ids=prev)
        topk = torch.topk(corrected[0], TOPK, dim=-1)

    def stats(t):
        f = t.reshape(-1).double()
        return dict(shape=list(t.shape), mean=float(f.mean()), std=float(f.std()),
                    min=float(f.min()), max=float(f.max()),
                    first8=[float(x) for x in t.reshape(-1)[:8]])

    g = dict(
        name=name, prompt_ids=ids[0].tolist(), prompt_text=tok.decode(ids[0]),
        block_size=B, target_layer_ids=tlids, mask_token_id=int(draft.mask_token_id),
        hidden_states_offset=1, logits_start=0,
        anchor_token=anchor, anchor_text=tok.decode([anchor]), block_ids=blk[0].tolist(),
        tensors=dict(fused_context=stats(fused[0]), block_in=stats(block_in[0]),
                     **{f"layer_out.{i}": stats(t[0]) for i, t in enumerate(layer_outs)},
                     trunk_out=stats(trunk[0])),
        drafted_ids=[int(x) for x in sampled[0]],
        drafted_text=tok.decode([int(x) for x in sampled[0]]),
        confidence_logits=[float(x) for x in conf.reshape(-1)[:B]],
        markov_topk=[[[int(i), float(v)] for v, i in
                      zip(topk.values[p].tolist(), topk.indices[p].tolist())]
                     for p in range(corrected.shape[1])],
        base_logits_pos0_first64=[float(x) for x in base[0, 0, :64]],
    )
    print(f"[{name}] ctx={ids.shape[1]} anchor={anchor} ({g['anchor_text']!r})")
    print(f"[{name}] drafted={g['drafted_ids']}")
    print(f"[{name}] drafted_text={g['drafted_text']!r}")
    print(f"[{name}] confidence={[round(c,3) for c in g['confidence_logits']]}")

    # Only POSITION 0's logits go in the binary. The full [7, 151936] pair would be 8.5 MB
    # per trace and the gate does not need it: the drafted ids already check the whole chain
    # end-to-end exactly, and one full-width logit row checks that the VALUES are right rather
    # than only their argmax. Keeps the committed fixture proportionate (cf. DFlash's 2.6 MB).
    ten = {f"{name}/fused_context": fused[0], f"{name}/block_in": block_in[0],
           f"{name}/trunk_out": trunk[0],
           f"{name}/markov_logits_pos0": corrected[0, 0]}
    for i, t in enumerate(layer_outs):
        ten[f"{name}/layer_out.{i}"] = t[0]
    return g, {k: v.contiguous() for k, v in ten.items()}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

    tok = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    draft = Qwen3DSparkModel.from_pretrained(
        DRAFT, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    print(f"block_size={draft.block_size} target_layer_ids={list(draft.target_layer_ids)} "
          f"markov_rank={draft.markov_head.markov_rank} "
          f"conf_with_markov={draft.confidence_head_with_markov}")

    raw_ids = tok(PROMPT, return_tensors="pt").input_ids
    chat_ids = tok(tok.apply_chat_template(
        [{"role": "user", "content": CHAT_PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids

    out = dict(note="DSpark one-round reference traces (dspark_qwen3_4b_block7 + Qwen3-4B, f32, "
                    "greedy) through DeepSpec's own modeling code",
               drafter="deepseek-ai/dspark_qwen3_4b_block7", target="Qwen/Qwen3-4B",
               dtype="float32", traces=[])
    tensors = {}
    for ids, nm in ((raw_ids, "raw"), (chat_ids, "chat")):
        g, ten = trace_one(tok, target, draft, ids, nm)
        out["traces"].append(g)
        tensors.update(ten)
    save_file(tensors, REF, metadata={"format": "pt"})
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"saved {OUT}\nsaved {REF} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()

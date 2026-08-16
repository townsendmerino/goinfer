#!/usr/bin/env python
"""Reference-side acceptance for the DSpark block-7 drafter — P10 kill-gate 2 for DSpark,
and the measurement that decides the re-pricing in docs/spec/08-dspark-dflash.md.

The DSpark re-pricing projects 1.75x on code, beating DFlash's 1.29x, on ONE transferred
input: a per-token accept probability fitted to DFlash's measured acceptance and carried to
a 7-token block. That transfer is the load-bearing assumption. This measures the real thing.

It runs DeepSpec's OWN loop, not a reimplementation: `generate_decoding_sample` is called
with the DSpark evaluator's actual `_init_context` / `_propose` / `_update` methods, bound to
a shim that supplies only the attributes they read. So the acceptance numbers are the
reference implementation's, on our prompts.

Two thresholds, because the pivot's strongest claim is about the confidence head:

    confidence_threshold = 0.0   gating OFF  — raw acceptance, comparable to DFlash's
    confidence_threshold > 0.0   gating ON   — the built-in router, which is the whole
                                               reason DSpark was preferred over DFlash

    ~/.venv-vl/bin/python scripts/ref_dspark_accept.py

Checkpoint: deepseek-ai/dspark_qwen3_4b_block7 (license=None; Francis accepted for
exploration 2026-08-15 — see docs/prompts/dspark-license-issue.md).
"""
import os
import sys
import types

import torch

DEEPSPEC = os.environ.get(
    "DEEPSPEC_DIR",
    "/tmp/claude-1000/-home-francis-mycode-goinfer/"
    "758aa8a7-c89d-4413-90bb-6dc78b85eb48/scratchpad/DeepSpec",
)
sys.path.insert(0, DEEPSPEC)

DRAFT = os.path.expanduser("~/models/dspark-qwen3-4b")
TARGET = os.path.expanduser("~/models/qwen3-4b")
MAXNEW = 160

# Same suites as decoder/dflash_accept_test.go, so the two drafters are compared on
# identical traffic rather than on each project's favourite prompts.
SUITES = {
    "code": ["Write a Python function that returns the nth Fibonacci number.",
             "Write a Go function that reverses a slice of ints in place.",
             "Write a SQL query that selects the top 5 customers by total order value."],
    "math": ["What is 17 * 23? Show your working.",
             "A train travels 120 km in 1.5 hours. What is its average speed in km/h?"],
    "chat": ["Explain what a hash table is, in two sentences.",
             "Give me three tips for keeping houseplants alive."],
}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepspec.eval.base_evaluator import generate_decoding_sample
    from deepspec.eval.dspark.evaluator import Qwen3DSparkEvaluator as DSparkEvaluator
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

    tok = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    draft = Qwen3DSparkModel.from_pretrained(
        DRAFT, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    B = int(draft.block_size)
    print(f"block_size={B} target_layer_ids={list(draft.target_layer_ids)} "
          f"mask={draft.mask_token_id} confidence_head={draft.confidence_head is not None} "
          f"markov={draft.markov_head is not None}", flush=True)

    eos = [tok.eos_token_id]

    def run(threshold):
        # Shim carrying exactly the attributes DSparkEvaluator's three callbacks read, so
        # DeepSpec's own methods drive the loop.
        shim = types.SimpleNamespace(
            draft_model=draft,
            max_proposal_tokens=B,
            confidence_head_recorder=None,
            args=types.SimpleNamespace(temperature=0.0, confidence_threshold=threshold),
        )
        init = DSparkEvaluator._init_context.__get__(shim, DSparkEvaluator)
        prop = DSparkEvaluator._propose.__get__(shim, DSparkEvaluator)
        upd = DSparkEvaluator._update.__get__(shim, DSparkEvaluator)

        out = {}
        for suite, prompts in SUITES.items():
            rounds = toks = proposed = 0
            for p in prompts:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                ids = tok(text, return_tensors="pt").input_ids
                r = generate_decoding_sample(
                    target_model=target, input_ids=ids, max_new_tokens=MAXNEW,
                    max_proposal_tokens=B, temperature=0.0, stop_token_ids=eos,
                    init_context=init, propose=prop, update=upd)
                rounds += r.verify_count
                toks += sum(r.acceptance_lengths)
                proposed += sum(r.proposal_lengths)
            out[suite] = (rounds, toks, proposed)
            print(f"  thr={threshold:.2f} {suite:5s} {rounds:4d} rounds {toks:4d} tokens "
                  f"| mean proposal {proposed/rounds:4.2f}/{B} => {toks/rounds:.2f} tok/verify",
                  flush=True)
        return out

    with torch.no_grad():
        print("\n--- gating OFF (raw acceptance, comparable to DFlash) ---", flush=True)
        run(0.0)
        print("\n--- gating ON (the confidence head as router) ---", flush=True)
        run(0.5)


if __name__ == "__main__":
    main()

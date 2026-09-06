#!/usr/bin/env python
"""Tiny-random Olmo 3 (Olmo3ForCausalLM) checkpoint + forward golden.

allenai/Olmo-3-{7B,32B}: two real departures from every existing family, both verified against
the real modeling_olmo3.py (docs/task-families-2026-09.md batch 2, G2): NO pre-norm at all (the
sublayer reads the raw residual stream; only the OUTPUT is normalized before the residual add --
NormPostOnly, genuinely different from Sandwich4's pre+post), and QK-norm over the FULL projected
q/k vector rather than per head (QKNormWhole). Otherwise MHA (num_key_value_heads ==
num_attention_heads on the real release), sliding/full 3:1 interleave, YaRN RoPE with an explicit
attention_factor.

    ~/.venv-nemotron3/bin/python scripts/pin_olmo3_tiny.py
    -> testdata/olmo3_forward_golden.json + testdata/olmo3_forward_full.json
       + testdata/olmo3-tiny/
"""
import json, os, math, random, torch
from transformers.models.olmo3 import Olmo3Config, Olmo3ForCausalLM

HERE = os.path.dirname(__file__)
TD = os.path.join(HERE, "..", "testdata")
GOLDEN = os.path.join(TD, "olmo3_forward_golden.json")
FULL = os.path.join(TD, "olmo3_forward_full.json")
CKPT = os.path.join(TD, "olmo3-tiny")

CFG = dict(
    vocab_size=512, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=8,  # MHA, matching the real release
    max_position_embeddings=64, sliding_window=8,
    layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    rope_scaling={
        "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8,
        "beta_fast": 32.0, "beta_slow": 1.0, "attention_factor": 1.1,
    },
    rms_norm_eps=1e-6, rope_theta=500000.0, tie_word_embeddings=False,
)
# 12 in-vocab ids, > sliding_window=8 so the ring/window path is genuinely exercised (already a
# proven mechanism elsewhere, but exercised for real here rather than trivially).
PROMPT = [1, 2, 7, 42, 100, 5, 200, 13, 88, 250, 9, 300]
SAMPLE_SEED = 1234
N_SAMPLE = 256
N_TOPK = 32


def main():
    torch.manual_seed(0)
    c = Olmo3Config(**CFG)
    m = Olmo3ForCausalLM(c).eval().to(torch.float32)
    with torch.no_grad():
        logits = m(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0, -1].float()
    lg = logits.tolist()
    vocab = len(lg)
    argmax = int(torch.tensor(lg).argmax())

    order = sorted(range(vocab), key=lambda i: lg[i], reverse=True)
    top_k = [[i, lg[i]] for i in order[:N_TOPK]]
    rng = random.Random(SAMPLE_SEED)
    sample_ids = rng.sample(range(vocab), min(N_SAMPLE, vocab))
    sample = [[i, lg[i]] for i in sample_ids]
    stats = dict(n=vocab, sum=sum(lg), sum_sq=sum(v * v for v in lg),
                 min=min(lg), max=max(lg))

    golden = dict(
        model_id="testdata/olmo3-tiny (seeded Olmo3ForCausalLM)",
        note="tiny Olmo3ForCausalLM forward oracle; HF float32, next-token logits at the last "
             "position. ids are raw token ids (the Go test is tokenizer-independent). argmax must "
             "match; top_k/sample to small tol; full cosine in the gitignored "
             "olmo3_forward_full.json. Regenerate: pin_olmo3_tiny.py",
        dtype="float32", prompt="", config=CFG, ids=PROMPT,
        argmax=argmax, argmax_token="", vocab_size=vocab, stats=stats,
        top_k=top_k, sample_seed=SAMPLE_SEED, sample=sample,
    )
    os.makedirs(TD, exist_ok=True)
    json.dump(golden, open(GOLDEN, "w"))
    json.dump(dict(argmax=argmax, logits=lg), open(FULL, "w"))
    m.save_pretrained(CKPT, safe_serialization=True)
    print(f"vocab={vocab} argmax={argmax} layers={CFG['num_hidden_layers']}")
    print(f"stats min={stats['min']:.4f} max={stats['max']:.4f}")
    print("wrote", GOLDEN, FULL, CKPT)


if __name__ == "__main__":
    main()

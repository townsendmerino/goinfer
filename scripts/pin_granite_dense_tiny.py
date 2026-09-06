#!/usr/bin/env python
"""Tiny-random dense Granite 4.2 (GraniteForCausalLM) checkpoint + forward golden.

ibm-granite/granite-4.2-{3b,8b,30b} (model_type "granite"): a plain llama skeleton (GQA,
SwiGLU, single-base RoPE, no bias, no QK-norm — confirmed byte-identical tensor names to
llama by instantiating GraniteForCausalLM and reading its state_dict) plus three of
Granite's four scalar multipliers set to NON-trivial values, so a loader/forward that drops
any of them fails parity. residual_multiplier is left at 1.0 (the only value every released
4.2 size ships, and the one goinfer's dense adapter accepts — see validateGraniteDense).

Self-contained + reproducible (fixed seed), same pattern as pin_mixtral_tiny.py /
pin_granite_tiny.py (the Granite-4.0-H hybrid's own fixture).

    ~/.venv-nemotron3/bin/python scripts/pin_granite_dense_tiny.py
    -> testdata/granite_dense_forward_golden.json + testdata/granite_dense_forward_full.json
       + testdata/granite-dense-tiny/
"""
import json, os, math, random, torch
from transformers import GraniteConfig, GraniteForCausalLM

HERE = os.path.dirname(__file__)
TD = os.path.join(HERE, "..", "testdata")
GOLDEN = os.path.join(TD, "granite_dense_forward_golden.json")
FULL = os.path.join(TD, "granite_dense_forward_full.json")
CKPT = os.path.join(TD, "granite-dense-tiny")

CFG = dict(
    vocab_size=512, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
    max_position_embeddings=64,
    rms_norm_eps=1e-6, rope_theta=10000000.0, tie_word_embeddings=False,
    # Non-trivial Granite scalars (residual_multiplier stays 1.0 -- the only value the
    # dense adapter accepts, and the only value any released 4.2 checkpoint ships).
    embedding_multiplier=12.0, attention_multiplier=0.5,
    residual_multiplier=1.0, logits_scaling=6.0,
)
PROMPT = [1, 2, 7, 42, 100, 5]  # arbitrary in-vocab ids (the Go test is tokenizer-independent)
SAMPLE_SEED = 1234
N_SAMPLE = 256
N_TOPK = 32


def main():
    torch.manual_seed(0)
    c = GraniteConfig(**CFG)
    m = GraniteForCausalLM(c).eval().to(torch.float32)
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
        model_id="testdata/granite-dense-tiny (seeded GraniteForCausalLM)",
        note="tiny GraniteForCausalLM forward oracle; HF float32, next-token logits at "
             "the last position. ids are raw token ids (the Go test is tokenizer-"
             "independent). argmax must match; top_k/sample to small tol; full cosine in "
             "the gitignored granite_dense_forward_full.json. Regenerate: pin_granite_dense_tiny.py",
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

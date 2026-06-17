#!/usr/bin/env python
"""Tiny-random Mixtral (MixtralForCausalLM) checkpoint + forward golden — the MoE
parity fixture. Mixtral is the Llama descriptor with the dense FFN replaced by a sparse
mixture of experts: a router softmaxes over all experts, the top-k run as SwiGLU MLPs,
and their outputs combine weighted by the renormalized router probabilities. This
exercises the router, top-k selection, and weighted expert combine through the generic
goinfer forward.

Self-contained + reproducible (fixed seed), replacing the original gate's external
oracle (/home/younes/.../converted-small-mixtral, unrecoverable) — the same pattern as
pin_deepseek_tiny.py / pin_glm_tiny.py. The committed checkpoint (testdata/mixtral-tiny,
small) + golden let the gate run on any box; the full-logit dump is gitignored (the
cosine check skips when it is absent).

    ~/.venv-vl/bin/python scripts/pin_mixtral_tiny.py
    -> testdata/mixtral_forward_golden.json + testdata/mixtral_forward_full.json
       + testdata/mixtral-tiny/
"""
import json, os, math, random, torch
from transformers import MixtralConfig, MixtralForCausalLM

HERE = os.path.dirname(__file__)
TD = os.path.join(HERE, "..", "testdata")
GOLDEN = os.path.join(TD, "mixtral_forward_golden.json")
FULL = os.path.join(TD, "mixtral_forward_full.json")
CKPT = os.path.join(TD, "mixtral-tiny")

CFG = dict(
    vocab_size=512, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=8,
    head_dim=8, max_position_embeddings=64,
    num_local_experts=8, num_experts_per_tok=2,
    rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
)
PROMPT = [1, 2, 7, 42, 100, 5]  # arbitrary in-vocab ids (the Go test is tokenizer-independent)
SAMPLE_SEED = 1234
N_SAMPLE = 256
N_TOPK = 32


def main():
    torch.manual_seed(0)
    c = MixtralConfig(**CFG)
    m = MixtralForCausalLM(c).eval().to(torch.float32)
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
        model_id="testdata/mixtral-tiny (seeded MixtralForCausalLM)",
        note="tiny MixtralForCausalLM forward oracle; HF float32, next-token logits at "
             "the last position. ids are raw token ids (the Go test is tokenizer-"
             "independent). argmax must match; top_k/sample to small tol; full cosine in "
             "the gitignored mixtral_forward_full.json. Regenerate: pin_mixtral_tiny.py",
        dtype="float32", prompt="", config=CFG, ids=PROMPT,
        argmax=argmax, argmax_token="", vocab_size=vocab, stats=stats,
        top_k=top_k, sample_seed=SAMPLE_SEED, sample=sample,
    )
    os.makedirs(TD, exist_ok=True)
    json.dump(golden, open(GOLDEN, "w"))
    json.dump(dict(argmax=argmax, logits=lg), open(FULL, "w"))
    m.save_pretrained(CKPT, safe_serialization=True)
    # transformers 5.x serializes rope as nested rope_parameters; the goinfer llama/
    # mixtral loader reads the classic flat top-level rope_theta. Patch it into the saved
    # config.json (consumed only by the Go loader — HF already produced the golden above).
    cpath = os.path.join(CKPT, "config.json")
    cj = json.load(open(cpath))
    cj["rope_theta"] = CFG["rope_theta"]
    json.dump(cj, open(cpath, "w"), indent=2)
    print(f"vocab={vocab} argmax={argmax} "
          f"experts={CFG['num_local_experts']}x top-{CFG['num_experts_per_tok']} "
          f"layers={CFG['num_hidden_layers']}")
    print(f"stats min={stats['min']:.4f} max={stats['max']:.4f}")
    print("wrote", GOLDEN, FULL, CKPT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""k=3 variant of pin_qwen3moe_tiny.py, for cuda/moe_expert_major_test.go (R11/P20,
docs/measurements/p20-expert-major-2026-09-21.md): the k=2 fixture cannot catch an accumulation-ORDER
regression in the expert-major fold, because float addition of exactly 2 terms is exactly commutative
in IEEE754 (checked directly: reversing the fold order passes cleanly at k=2, fails 482/512 logits at
k=3). Identical to pin_qwen3moe_tiny.py otherwise; see that file for the family/shape rationale.

    ~/.venv-vl/bin/python scripts/pin_qwen3moe_tiny_k3.py
    -> testdata/qwen3moe_k3_forward_golden.json + testdata/qwen3moe_k3_forward_full.json
       + testdata/qwen3moe-tiny-k3/
"""
import json, os, math, random, torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

HERE = os.path.dirname(__file__)
TD = os.path.join(HERE, "..", "testdata")
GOLDEN = os.path.join(TD, "qwen3moe_k3_forward_golden.json")
FULL = os.path.join(TD, "qwen3moe_k3_forward_full.json")
CKPT = os.path.join(TD, "qwen3moe-tiny-k3")

CFG = dict(
    vocab_size=512, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
    head_dim=8, max_position_embeddings=64,
    moe_intermediate_size=32, num_experts=8, num_experts_per_tok=3,  # k>=3: order-sensitive (2-term float add is exactly commutative)
    norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[],
    rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=False,
)
PROMPT = [1, 2, 7, 42, 100, 5]  # arbitrary in-vocab ids (the Go test is tokenizer-independent)
SAMPLE_SEED = 1234
N_SAMPLE = 256
N_TOPK = 32


def main():
    torch.manual_seed(0)
    c = Qwen3MoeConfig(**CFG)
    m = Qwen3MoeForCausalLM(c).eval().to(torch.float32)
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
        model_id="testdata/qwen3moe-tiny (seeded Qwen3MoeForCausalLM)",
        note="tiny Qwen3MoeForCausalLM forward oracle; HF float32, next-token logits at "
             "the last position. ids are raw token ids (the Go test is tokenizer-"
             "independent). argmax must match; top_k/sample to small tol; full cosine in "
             "the gitignored qwen3moe_forward_full.json. Regenerate: pin_qwen3moe_tiny.py",
        dtype="float32", prompt="", config=CFG, ids=PROMPT,
        argmax=argmax, argmax_token="", vocab_size=vocab, stats=stats,
        top_k=top_k, sample_seed=SAMPLE_SEED, sample=sample,
    )
    os.makedirs(TD, exist_ok=True)
    json.dump(golden, open(GOLDEN, "w"))
    json.dump(dict(argmax=argmax, logits=lg), open(FULL, "w"))
    m.save_pretrained(CKPT, safe_serialization=True)
    # Real finding, not assumed: transformers 5.15.0's Qwen3MoeConfig.save_pretrained
    # writes "num_local_experts", but the REAL RELEASED Qwen/Qwen3-30B-A3B config.json
    # (fetched from the hub directly) uses "num_experts": 128 — confirmed by diffing the
    # two field names, not by reading a model card. goinfer's qwen3_moe adapter reads
    # num_experts to match the real release, so the tiny fixture is rewritten the same
    # way pin_qwen3next_tiny.py rewrites its saved config to the release's flat shape —
    # otherwise this fixture would silently test a schema no real checkpoint ships.
    # rope_parameters is left nested (also matches the real release); the adapter
    # backfills that via backfillFlatRope.
    cpath = os.path.join(CKPT, "config.json")
    cj = json.load(open(cpath))
    if "num_local_experts" in cj:
        cj["num_experts"] = cj.pop("num_local_experts")
    json.dump(cj, open(cpath, "w"), indent=2)
    print(f"vocab={vocab} argmax={argmax} "
          f"experts={CFG['num_experts']}x top-{CFG['num_experts_per_tok']} "
          f"layers={CFG['num_hidden_layers']}")
    print(f"stats min={stats['min']:.4f} max={stats['max']:.4f}")
    print("wrote", GOLDEN, FULL, CKPT)


if __name__ == "__main__":
    main()

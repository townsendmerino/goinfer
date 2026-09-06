#!/usr/bin/env python
"""Tiny-random Ministral 3 (Ministral3ForCausalLM) checkpoint + forward golden.

mistralai/Ministral-3-{3b,8b,14b}: Mistral's GQA skeleton (reused verbatim -- confirmed
byte-identical tensor names to llama/mistral by instantiating Ministral3ForCausalLM and reading
its state_dict) plus two real deltas found in Phase 0 (docs/task-families-2026-09.md batch 2,
G3): no sliding window at all on the released checkpoints (sliding_window: null), and YaRN RoPE
scaling with a THIRD field, llama_4_scaling_beta, that multiplies the query by
1 + beta*log(1 + floor(pos/original_max_position_embeddings)) AFTER RoPE, on every layer --
Llama 4's own attention-temperature-tuning formula, generalized to a new Architecture field pair
(AttnTempBeta/AttnTempOrigMaxPos) since llama4Architecture's own version is an either/or with
RoPE and this family needs both together.

The fixture deliberately uses a SHORT original_max_position_embeddings (8) and a prompt LONGER
than that (12 tokens), so floor(pos/8) > 0 for the last 4 positions and the attn-temp scale is
genuinely exercised, not identity -- a fixture that never crosses the threshold would pass even
with the mechanism entirely absent (the "minimal repro hides the bug" trap this repo's own
culture names explicitly). mscale/mscale_all_dim are also DISTINCT (0.5 / 0.8, not both 1.0 like
the real release) so the YaRN attention_factor override actually gets exercised as a real ratio,
not the trivially-1.0 case a broken formula would also pass.

    ~/.venv-nemotron3/bin/python scripts/pin_ministral3_tiny.py
    -> testdata/ministral3_forward_golden.json + testdata/ministral3_forward_full.json
       + testdata/ministral3-tiny/
"""
import json, os, math, random, torch
from transformers.models.ministral3 import Ministral3Config, Ministral3ForCausalLM

HERE = os.path.dirname(__file__)
TD = os.path.join(HERE, "..", "testdata")
GOLDEN = os.path.join(TD, "ministral3_forward_golden.json")
FULL = os.path.join(TD, "ministral3_forward_full.json")
CKPT = os.path.join(TD, "ministral3-tiny")

CFG = dict(
    vocab_size=512, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2, head_dim=8,
    max_position_embeddings=64, sliding_window=None,
    rope_parameters={
        "rope_type": "yarn", "rope_theta": 1000000.0, "factor": 4.0,
        "original_max_position_embeddings": 8, "beta_fast": 32.0, "beta_slow": 1.0,
        "llama_4_scaling_beta": 0.2, "mscale": 0.5, "mscale_all_dim": 0.8,
    },
    tie_word_embeddings=False,
)
# 12 in-vocab ids, deliberately > original_max_position_embeddings=8 so the attn-temp
# scale is exercised on the last few positions (the Go test is tokenizer-independent).
PROMPT = [1, 2, 7, 42, 100, 5, 200, 13, 88, 250, 9, 300]
SAMPLE_SEED = 1234
N_SAMPLE = 256
N_TOPK = 32


def main():
    torch.manual_seed(0)
    c = Ministral3Config(**CFG)
    m = Ministral3ForCausalLM(c).eval().to(torch.float32)
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
        model_id="testdata/ministral3-tiny (seeded Ministral3ForCausalLM)",
        note="tiny Ministral3ForCausalLM forward oracle; HF float32, next-token logits at "
             "the last position, prompt deliberately longer than original_max_position_embeddings "
             "so the attn-temp scale is exercised. ids are raw token ids (the Go test is "
             "tokenizer-independent). argmax must match; top_k/sample to small tol; full cosine "
             "in the gitignored ministral3_forward_full.json. Regenerate: pin_ministral3_tiny.py",
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

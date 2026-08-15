#!/usr/bin/env python
"""Real-model forward oracle for GPT-2 small (gpt2) — the HF float32 reference behind the
family's T3 row. GPT-2 is the family that breaks the Llama mold (LayerNorm with bias,
learned absolute position embeddings, non-gated GELU MLP, fused qkv + bias, Conv1D
[in,out] layout, tied wte head), so its oracle is what proves those seams on released
weights rather than on a seeded fixture.

Replaces the dangling `scripts/pin_llama_forward.py` reference in decoder/gpt2_test.go:
that script no longer exists, so testdata/gpt2_forward_golden.json had no regenerator in
the tree and gpt2_forward_full.json — the file the gate's cosine reads — was never
produced on this box (the gate logged `cosine=NaN` and still passed).

Writes two files:

    testdata/gpt2_forward_golden.json   argmax + stats + top_k + a seeded 256-id sample
                                        (COMMITTED; small)
    testdata/gpt2_forward_full.json     the whole 50257-wide logit vector, for the cosine
                                        (gitignored by testdata/*_forward_full.json; local)

    ~/.venv-vl/bin/python scripts/pin_gpt2_real.py
"""
import json, os, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "testdata", "gpt2")
GOLDEN = os.path.join(HERE, "..", "testdata", "gpt2_forward_golden.json")
FULL = os.path.join(HERE, "..", "testdata", "gpt2_forward_full.json")
PROMPT = "The capital of France is"
SEED = 1234
N_SAMPLE = 256
TOP_K = 32


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(PROMPT, return_tensors="pt").input_ids  # GPT-2 prepends no BOS
    m = AutoModelForCausalLM.from_pretrained(
        CKPT, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    with torch.no_grad():
        logits = m(input_ids=ids, use_cache=False).logits[0, -1].float()
    v = logits.tolist()
    n = len(v)

    top = torch.topk(logits, TOP_K)
    # Reuse the committed golden's sample indices when it is present. The retired generator
    # drew them with an RNG whose scheme was not recorded, so re-drawing here would churn 256
    # tracked lines for no signal — and the reproduction check that matters (argmax, stats,
    # top_k) is exact against the committed file either way.
    sample_ids = None
    if os.path.exists(GOLDEN):
        prev = json.load(open(GOLDEN))
        if prev.get("vocab_size") == n and len(prev.get("sample", [])) == N_SAMPLE:
            sample_ids = [int(kv[0]) for kv in prev["sample"]]
    if sample_ids is None:
        sample_ids = sorted(random.Random(SEED).sample(range(n), N_SAMPLE))

    g = dict(
        model_id="testdata/gpt2",
        note=("forward oracle for testdata/gpt2. HF float32; next-token logits at the last "
              "position. ids are HF token ids (the Go test is tokenizer-independent). argmax "
              "must match; top_k/sample to small tol; full cosine in the gitignored "
              "gpt2_forward_full.json."),
        dtype="float32",
        prompt=PROMPT,
        ids=ids[0].tolist(),
        argmax=int(logits.argmax()),
        argmax_token=tok.decode([int(logits.argmax())]),
        vocab_size=n,
        stats=dict(n=n, sum=float(logits.double().sum()), sum_sq=float((logits.double() ** 2).sum()),
                   min=float(logits.min()), max=float(logits.max())),
        top_k=[[int(i), float(x)] for x, i in zip(top.values.tolist(), top.indices.tolist())],
        sample_seed=SEED,
        sample=[[i, v[i]] for i in sample_ids],
    )
    with open(GOLDEN, "w") as f:
        json.dump(g, f, indent=2)
        f.write("\n")
    json.dump(dict(argmax=g["argmax"], logits=v), open(FULL, "w"))
    print(f"argmax={g['argmax']} ({g['argmax_token']!r}) vocab={n}")
    print("saved", GOLDEN)
    print("saved", FULL)


if __name__ == "__main__":
    main()

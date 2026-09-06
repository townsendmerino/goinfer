#!/usr/bin/env python
"""Tiny-random SmolLM3-3B (SmolLM3ForCausalLM) checkpoint + forward golden.

HuggingFaceTB/SmolLM3-3B: a plain llama-shaped dense GQA model (tensor names byte-identical,
confirmed by instantiating SmolLM3ForCausalLM and reading its state_dict) with per-layer NoPE on
every 4th layer via `no_rope_layers` -- a field whose VALUES are the opposite of what its name
suggests, verified against the real modeling_smollm3.py: `self.use_rope =
config.no_rope_layers[layer_idx]`, so a 1 means "has RoPE", 0 means NoPE. Reuses the SAME Config
field and boolean convention llama4_text already established for its own NoPE layers, and the
SAME layerNoPE Architecture hook cohere2Architecture already populates.

    ~/.venv-nemotron3/bin/python scripts/pin_smollm3_tiny.py
    -> testdata/smollm3_forward_golden.json + testdata/smollm3_forward_full.json
       + testdata/smollm3-tiny/
"""
import json, os, math, random, torch
from transformers.models.smollm3 import SmolLM3Config, SmolLM3ForCausalLM

HERE = os.path.dirname(__file__)
TD = os.path.join(HERE, "..", "testdata")
GOLDEN = os.path.join(TD, "smollm3_forward_golden.json")
FULL = os.path.join(TD, "smollm3_forward_full.json")
CKPT = os.path.join(TD, "smollm3-tiny")

CFG = dict(
    vocab_size=512, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
    max_position_embeddings=64, no_rope_layer_interval=4,
    rms_norm_eps=1e-6, rope_theta=5000000.0, tie_word_embeddings=True,
    pad_token_id=0, bos_token_id=1, eos_token_id=2,
)
PROMPT = [1, 2, 7, 42, 100, 5]  # arbitrary in-vocab ids (the Go test is tokenizer-independent)
SAMPLE_SEED = 1234
N_SAMPLE = 256
N_TOPK = 32


def main():
    torch.manual_seed(0)
    c = SmolLM3Config(**CFG)
    assert c.no_rope_layers == [1, 1, 1, 0], c.no_rope_layers  # sanity: layer 3 (last) is NoPE
    m = SmolLM3ForCausalLM(c).eval().to(torch.float32)
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
        model_id="testdata/smollm3-tiny (seeded SmolLM3ForCausalLM)",
        note="tiny SmolLM3ForCausalLM forward oracle; HF float32, next-token logits at the last "
             "position. ids are raw token ids (the Go test is tokenizer-independent). argmax must "
             "match; top_k/sample to small tol; full cosine in the gitignored "
             "smollm3_forward_full.json. Regenerate: pin_smollm3_tiny.py",
        dtype="float32", prompt="", config=CFG, ids=PROMPT,
        argmax=argmax, argmax_token="", vocab_size=vocab, stats=stats,
        top_k=top_k, sample_seed=SAMPLE_SEED, sample=sample,
    )
    os.makedirs(TD, exist_ok=True)
    json.dump(golden, open(GOLDEN, "w"))
    json.dump(dict(argmax=argmax, logits=lg), open(FULL, "w"))
    m.save_pretrained(CKPT, safe_serialization=True)
    print(f"vocab={vocab} argmax={argmax} layers={CFG['num_hidden_layers']} no_rope_layers={c.no_rope_layers}")
    print(f"stats min={stats['min']:.4f} max={stats['max']:.4f}")
    print("wrote", GOLDEN, FULL, CKPT)


if __name__ == "__main__":
    main()

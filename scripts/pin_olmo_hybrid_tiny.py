#!/usr/bin/env python
"""Tiny-random Olmo Hybrid (OlmoHybridForCausalLM) checkpoint + forward golden.

allenai/Olmo-Hybrid-7B: qwen3_5's Gated DeltaNet (3-of-4 layers) + olmo3's own full-attention
shape (whole-vector QK-norm, 1-of-4 layers) -- but NOT a straight composition, verified against
the real modeling_olmo_hybrid.py (docs/task-families-2026-09.md batch 2, G2):

  - Norm placement differs BY LAYER KIND within one model: full-attention layers use olmo3's
    NormPostOnly (no pre-norm at all), DeltaNet layers use plain NormPre2 -- the real reason G2
    paused for a design decision (NormPlacementLinear).
  - linear_allow_neg_eigval doubles the write-gate beta to [0,2) after the sigmoid (default true
    on the release).
  - q_proj/k_proj/v_proj are THREE separate tensors (more unfused than qwen3_5's own
    pre-concatenated in_proj_qkv); the depthwise causal conv is ALSO three separate tensors
    (q_conv1d/k_conv1d/v_conv1d, split at the same q/k/v channel boundaries); the DeltaNet output
    gate is o_norm/o_proj (qwen3_5: norm/out_proj), and o_norm's epsilon is hardcoded 1e-5 in HF's
    source regardless of config.rms_norm_eps (1e-6 here) -- FLA's FusedRMSNormGated default.
  - rope_parameters is {"rope_theta": null} on the release: NO RoPE anywhere, on any layer.

VERIFIED AGAINST THE REAL allenai/Olmo-Hybrid-7B CHECKPOINT (HTTP Range on its safetensors
header, not the full 7.43GB file), not just modeling_olmo_hybrid.py's source: the real file's
linear layers carry input_layernorm/post_attention_layernorm (unchanged from source) and THREE
separate q_conv1d/k_conv1d/v_conv1d tensors split at exactly [key_dim, key_dim, value_dim] rows.
This transformers version's OWN save_pretrained (via its registered conversion_mapping.py) does
NOT reproduce that layout -- it renames the norms to attention_layer_norm/feedforward_layer_norm
AND splits conv1d into three ARBITRARY roughly-equal thirds, neither of which matches the real
release. So this script bypasses save_pretrained's safetensors write entirely: it builds the
model, takes state_dict() directly (which DOES match source/the real norm names), manually
re-splits each linear layer's combined conv1d.weight at the true q/k/v boundaries, and writes the
result with safetensors.torch.save_file -- reproducing the real file's actual tensor set, not
either wrong intermediate guess.

    ~/.venv-nemotron3/bin/python scripts/pin_olmo_hybrid_tiny.py
    -> testdata/olmo_hybrid_forward_golden.json + testdata/olmo_hybrid_forward_full.json
       + testdata/olmo_hybrid-tiny/
"""
import json, os, random, torch
from safetensors.torch import save_file
from transformers.models.olmo_hybrid import OlmoHybridConfig, OlmoHybridForCausalLM

HERE = os.path.dirname(__file__)
TD = os.path.join(HERE, "..", "testdata")
GOLDEN = os.path.join(TD, "olmo_hybrid_forward_golden.json")
FULL = os.path.join(TD, "olmo_hybrid_forward_full.json")
CKPT = os.path.join(TD, "olmo_hybrid-tiny")

CFG = dict(
    vocab_size=512, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=8,  # MHA, matching the real release
    max_position_embeddings=64,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_num_key_heads=4, linear_num_value_heads=4,  # no GVA (rep=1), matching the real release
    linear_key_head_dim=8, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    linear_allow_neg_eigval=True,
    rms_norm_eps=1e-6, rope_parameters={"rope_theta": None}, tie_word_embeddings=False,
    pad_token_id=None, bos_token_id=None, eos_token_id=None,
)
PROMPT = [1, 2, 7, 42, 100, 5, 200, 13, 88, 250]
SAMPLE_SEED = 1234
N_SAMPLE = 256
N_TOPK = 32


def main():
    torch.manual_seed(0)
    c = OlmoHybridConfig(**CFG)
    m = OlmoHybridForCausalLM(c).eval().to(torch.float32)
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
        model_id="testdata/olmo_hybrid-tiny (seeded OlmoHybridForCausalLM)",
        note="tiny OlmoHybridForCausalLM forward oracle; HF float32, next-token logits at the "
             "last position. ids are raw token ids (the Go test is tokenizer-independent). argmax "
             "must match; top_k/sample to small tol; full cosine in the gitignored "
             "olmo_hybrid_forward_full.json. Regenerate: pin_olmo_hybrid_tiny.py",
        dtype="float32", prompt="", config=CFG, ids=PROMPT,
        argmax=argmax, argmax_token="", vocab_size=vocab, stats=stats,
        top_k=top_k, sample_seed=SAMPLE_SEED, sample=sample,
    )
    os.makedirs(TD, exist_ok=True)
    json.dump(golden, open(GOLDEN, "w"))
    json.dump(dict(argmax=argmax, logits=lg), open(FULL, "w"))

    # Write config.json/tokenizer files normally (their format is unaffected by the conv1d/norm
    # question above), then overwrite model.safetensors with a state_dict reproducing the REAL
    # release's actual tensor layout -- see the module docstring for why save_pretrained's own
    # output can't be used directly here.
    m.save_pretrained(CKPT, safe_serialization=True)
    key_head_dim, num_key_heads = CFG["linear_key_head_dim"], CFG["linear_num_key_heads"]
    value_head_dim, num_value_heads = CFG["linear_value_head_dim"], CFG["linear_num_value_heads"]
    key_dim, value_dim = key_head_dim * num_key_heads, value_head_dim * num_value_heads
    linear_layers = {i for i, t in enumerate(CFG["layer_types"]) if t == "linear_attention"}
    sd = {}
    for name, t in m.state_dict().items():
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers" and parts[3] == "linear_attn" \
                and name.endswith("conv1d.weight") and int(parts[2]) in linear_layers:
            prefix = ".".join(parts[:4])
            q, k, v = t[:key_dim], t[key_dim:2 * key_dim], t[2 * key_dim:]
            assert q.shape[0] == key_dim and k.shape[0] == key_dim and v.shape[0] == value_dim
            sd[f"{prefix}.q_conv1d.weight"] = q.contiguous()
            sd[f"{prefix}.k_conv1d.weight"] = k.contiguous()
            sd[f"{prefix}.v_conv1d.weight"] = v.contiguous()
        else:
            sd[name] = t.contiguous()
    save_file(sd, os.path.join(CKPT, "model.safetensors"), metadata={"format": "pt"})
    print(f"vocab={vocab} argmax={argmax} layers={CFG['num_hidden_layers']}")
    print(f"stats min={stats['min']:.4f} max={stats['max']:.4f}")
    print("wrote", GOLDEN, FULL, CKPT)


if __name__ == "__main__":
    main()

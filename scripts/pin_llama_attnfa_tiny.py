#!/usr/bin/env python
"""Build a tiny-random Llama checkpoint shaped to satisfy canUseAttnFA's dispatch guard
(metal/model.go) -- dense (no MoE/DeltaNet/adapters), no sliding window, no attention sink,
head_dim EXACTLY 128, GQA with nKV>0. Every other committed Metal snapshot-golden fixture
(mixtral-tiny: head_dim=8; gemma4-dense-scaled: head_dim=256) fails that guard, so
`attention_fa` -- shipped as Metal decode's default past depth 1536 (R2,
docs/tasks/red-october.md) -- has never been covered by TestMetalSnapshotGolden, the suite's
only bit-exact (not cosine/tolerance) reference. This fixture closes that gap.

Deliberately plain, same reasoning as pin_llama_tiny.py: the point is to isolate the ONE
property (head_dim==128) the other fixtures lack, not to add a second exotic geometry.

    python3 scripts/pin_llama_attnfa_tiny.py
    -> testdata/llama-attnfa-tiny/   (HF safetensors checkpoint; no golden JSON --
       TestMetalSnapshotGolden is self-referential, it bakes its own hash and does not
       compare against an HF oracle)
"""
import os

from transformers import LlamaConfig, LlamaForCausalLM

HERE = os.path.dirname(__file__)
CKPT = os.path.join(HERE, "..", "testdata", "llama-attnfa-tiny")

CFG = dict(
    vocab_size=256, hidden_size=512, intermediate_size=256, num_hidden_layers=2,
    # hidden_size / num_attention_heads = 128 -- the exact head_dim canUseAttnFA requires.
    # GQA 2:1 (nKV=2>0) so the fixture exercises the grouped case production actually ships.
    num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=4096, rms_norm_eps=1e-5,
    rope_theta=500000.0,
    tie_word_embeddings=False,
    attention_bias=False, mlp_bias=False,
)


def main():
    import torch
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(**CFG)).eval().to(torch.float32)
    assert model.config.head_dim == 128, f"head_dim={model.config.head_dim}, expected 128"
    os.makedirs(CKPT, exist_ok=True)
    model.save_pretrained(CKPT, safe_serialization=True)
    print(f"saved checkpoint -> {CKPT}  (head_dim={model.config.head_dim}, "
          f"num_key_value_heads={model.config.num_key_value_heads})")


if __name__ == "__main__":
    main()

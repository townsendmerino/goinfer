#!/usr/bin/env python3
"""Write a synthetic GGUF with a real model's tensor SHAPES and ggml BLOCK TYPES but random weights.

For profiling a transcode: dequant/quant/write cost depends on shapes and block formats, not on what
the weights mean. Every Q4_K/Q6_K block gets a fixed, finite f16 super-scale so no value is inf/NaN.

usage: synth_gguf.py <out.gguf> <preset>   (presets: qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-7b)
"""
import sys
import numpy as np
import gguf

PRESETS = {
    # hidden, layers, heads, kv_heads, ffn, vocab, tied
    "qwen2.5-0.5b": (896, 24, 14, 2, 4864, 151936, True),
    "qwen2.5-1.5b": (1536, 28, 12, 2, 8960, 151936, True),
    "qwen2.5-7b": (3584, 28, 28, 4, 18944, 152064, False),
}

QK = 256
BYTES = {gguf.GGMLQuantizationType.Q4_K: 144, gguf.GGMLQuantizationType.Q6_K: 210}
rng = np.random.default_rng(1)
F16_SMALL = np.frombuffer(np.float16(0.0123).tobytes(), dtype=np.uint8)
F16_MIN = np.frombuffer(np.float16(0.0045).tobytes(), dtype=np.uint8)


def kquant(out_rows, in_cols, typ):
    assert in_cols % QK == 0, (in_cols, typ)
    nb = in_cols // QK
    bpb = BYTES[typ]
    b = rng.integers(0, 256, size=(out_rows, nb, bpb), dtype=np.uint8)
    if typ == gguf.GGMLQuantizationType.Q4_K:  # d, dmin are the first two f16s
        b[:, :, 0:2] = F16_SMALL
        b[:, :, 2:4] = F16_MIN
    else:  # Q6_K: ql[128] qh[64] scales[16] d(f16) — d is the last two bytes
        b[:, :, 208:210] = F16_SMALL
        b[:, :, 192:208] &= 0x3F  # keep the int8 sub-scales modest
    return b.reshape(out_rows, nb * bpb)


def main():
    out, preset = sys.argv[1], sys.argv[2]
    hidden, layers, heads, kvh, ffn, vocab, tied = PRESETS[preset]
    hd = hidden // heads
    kv = kvh * hd
    Q4, Q6 = gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K
    w = gguf.GGUFWriter(out, "qwen2")
    w.add_context_length(32768)
    w.add_embedding_length(hidden)
    w.add_block_count(layers)
    w.add_feed_forward_length(ffn)
    w.add_head_count(heads)
    w.add_head_count_kv(kvh)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1000000.0)
    w.add_uint32("tokenizer.ggml.eos_token_id", 151645)

    def add(name, out_rows, in_cols, typ):
        w.add_tensor(name, kquant(out_rows, in_cols, typ), raw_dtype=typ)

    def vec(name, n, val=1.0):
        w.add_tensor(name, np.full(n, val, dtype=np.float32) + rng.standard_normal(n).astype(np.float32) * 0.01)

    add("token_embd.weight", vocab, hidden, Q6 if tied else Q4)
    vec("output_norm.weight", hidden)
    if not tied:
        add("output.weight", vocab, hidden, Q6)
    for i in range(layers):
        p = f"blk.{i}."
        more = i < layers // 8 or i >= 7 * layers // 8 or (i - layers // 8) % 3 == 2  # llama.cpp use_more_bits
        vec(p + "attn_norm.weight", hidden)
        add(p + "attn_q.weight", hidden, hidden, Q4)
        vec(p + "attn_q.bias", hidden, 0.0)
        add(p + "attn_k.weight", kv, hidden, Q4)
        vec(p + "attn_k.bias", kv, 0.0)
        add(p + "attn_v.weight", kv, hidden, Q6 if more else Q4)
        vec(p + "attn_v.bias", kv, 0.0)
        add(p + "attn_output.weight", hidden, hidden, Q4)
        vec(p + "ffn_norm.weight", hidden)
        add(p + "ffn_gate.weight", ffn, hidden, Q4)
        add(p + "ffn_up.weight", ffn, hidden, Q4)
        add(p + "ffn_down.weight", hidden, ffn, Q6 if more else Q4)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()

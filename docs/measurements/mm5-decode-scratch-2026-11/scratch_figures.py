"""Metal decode scratch on the Mac's served checkpoints, from their configs: G-MS3's arithmetic (TestDecodeScratch_handTable), which
equals the build's list on every registered fixture. No model is loaded."""
import math

MiB = 1 << 20
MODELS = [  # name, H, V, I, nH, nKV, hd, nL, q4k, source
    ('Qwen2.5-Coder 0.5B', 896, 151936, 4864, 14, 2, 64, 24, False, 'GGUF'),
    ('Qwen2.5-Coder 1.5B', 1536, 151936, 8960, 12, 2, 128, 28, False, 'GGUF'),
    ('Qwen2.5 3B', 2048, 151936, 11008, 16, 2, 128, 36, False, 'published config (no checkpoint on this Mac)'),
    ('Qwen2.5 7B', 3584, 152064, 18944, 28, 4, 128, 28, False, 'GGUF'),
    ('Phi-3 mini 4k (q4k)', 3072, 32064, 8192, 32, 32, 96, 32, True, 'GGUF'),
    ('Gemma 3 1B', 1152, 262144, 6912, 4, 1, 256, 26, False, 'GGUF'),
]


def fa_partial(hd, nH, nKV):  # attnFAPartialFloats: hd 128, or 64 at G = 7 (attnFABlk64On), G <= 8
    G = nH // nKV if nKV and nH % nKV == 0 else 0
    ok_hd = hd == 128 or (hd == 64 and G == 7)
    return nH * (hd + 2) * 32 if ok_hd and 0 < G <= 8 else 0


for name, H, V, I, nH, nKV, hd, nL, q4k, src in MODELS:
    q, kv, gu = nH * hd, nKV * hd, I
    fa = fa_partial(hd, nH, nKV)
    f32 = (H + 1 + q + 2 * kv + 2 * gu + q + 1 + H + 1 + 1 + H + V + 2 * math.ceil(V / 8) + 1 + 2 * math.ceil(V / 1024) + 1 + fa
           + ((H + gu) if q4k else 0))
    twins = (4 * H if q4k else H) + q + H + 2 * H + 2 * H + 2 * q + gu
    uni = 22 + (2 if fa else 0) + 7 + 2 * nL  # core, FA's two, one geometry, each layer's sink flag and window
    seg_lo, seg_hi = (4 * nL, 7 * nL) if q4k else (0, 0)  # the q4k lane's segment row counts, 4 to 7 a layer
    decode = 4 * f32 + twins + 4 * uni
    batch = 16 * (H + V) * 4 + 48 * 4
    chain_greedy, chain_sampled = 12 * 4, 7 * 4 + 2 * V * 4
    B = 8
    ffn = I
    stf = (B * H + B * (q + 2 * kv) + B * 2 * ffn + B * V + 4 * B + max(H, 2 * I) + B + 2 * B * 64 + B + B + B + B * q + 5 * B + B * fa
           + ((B * H + B * ffn) if q4k else 0))
    stb = B * H + B * q + B * H + B * ffn
    sth = 2 * (max(H, q, I) * 8) + 2 * (H * 8)
    stu = 4 + 3 + 9 * B + 1 + 1 + (1 if fa else 0) + (B + 1)
    step = 4 * stf + stb + sth + 4 * stu
    kvpos = nL * 2 * nKV * hd * 2  # f16 K and V, every layer over the whole context (64-row blocks)
    one_slot = decode + batch + chain_greedy + chain_sampled + 4 * seg_hi
    print(f'{name} [{src}]')
    print(f'  decode list, one row: {decode/MiB:.2f} MiB ({decode:,} B; fused-attention partial {4*fa/MiB:.2f} MiB, logits {4*V/MiB:.2f} MiB)'
          + (f'; plus the q4k segments {4*seg_lo}-{4*seg_hi} B' if q4k else ''))
    print(f'  batchRows at 16 rows: {batch/MiB:.2f} MiB; chainBuffers: greedy {chain_greedy} B, sampled +{chain_sampled/MiB:.2f} MiB')
    print(f'  total, one slot: {one_slot/MiB:.2f} MiB; MC3 step list at 8 rows (2+ slots): {step/MiB:.2f} MiB; all: {(one_slot+step)/MiB:.2f} MiB')
    print(f'  KV f16 per position: {kvpos:,} B ({kvpos/1024:.0f} KiB): one slot = {one_slot/kvpos:,.0f} positions, with the step '
          f'{(one_slot+step)/kvpos:,.0f}; 384 MiB margin = {384*MiB/kvpos:,.0f} positions')

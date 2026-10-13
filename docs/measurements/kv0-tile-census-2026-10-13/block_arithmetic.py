#!/usr/bin/env python3
"""KV0 tile census, item 2: bytes per block, blocks per 32K-token sequence and waste, for three models, at B = 64, 128, 256.

Geometries are read from the checkpoints (GGUF metadata for Qwen2.5-7B and Gemma-3-4B, config.json for Qwen3.6-35B-A3B) and
written down here. KV bytes are f16 (what CUDA's default now stores for cleared families) and f32 beside it.

  Qwen2.5-7B-Instruct      28 layers, 4 KV heads x head_dim 128 (kvDim 512), all full attention
  Gemma-3-4B-it            34 layers, 4 KV heads x head_dim 256 (kvDim 1024); 5 local (window 1024) : 1 global, so layers
                           5, 11, 17, 23, 29 are global (5) and the other 29 are local
  Qwen3.6-35B-A3B          40 layers: every 4th is full attention (10), 2 KV heads x head_dim 256 (kvDim 512); the other 30 are
                           Gated DeltaNet (recurrent state, 62.8 MiB per sequence, docs/spec/09-mtp-heads.md)

A block is B consecutive positions of one sequence in one group, at the same block index in every layer of the group.
Window group: a sequence holds ceil(W/B) + 1 blocks (the window's blocks plus the one it is filling); the design's extra
headroom block is not counted here. Waste: the unfilled part of a sequence's last block per group, B/2 positions on average and
B-1 at worst, as bytes and as a share of the sequence's KV in that group.
"""
import math

SEQ = 32768
MODELS = {
    "Qwen2.5-7B": dict(groups={"full": dict(layers=28, kvdim=512, window=None)}),
    "Gemma-3-4B": dict(groups={"global": dict(layers=5, kvdim=1024, window=None), "local": dict(layers=29, kvdim=1024, window=1024)}),
    "Qwen3.6-35B-A3B": dict(groups={"full": dict(layers=10, kvdim=512, window=None)}, state_mib=62.8),
}


def mib(x):
    return x / 2**20


for name, m in MODELS.items():
    print(f"\n== {name}")
    for elem, ename in ((2, "f16"), (4, "f32")):
        per_pos = {g: d["layers"] * 2 * d["kvdim"] * elem for g, d in m["groups"].items()}   # K and V, all layers of the group
        print(f"  {ename}: bytes per position per group: " + ", ".join(f"{g} {v:,} B ({v/1024:.1f} KiB)" for g, v in per_pos.items()) +
              f"; all groups {sum(per_pos.values()):,} B")
        print(f"  {'B':>4} | {'group':<7} | {'bytes/block':>12} | {'blocks per 32K seq':>18} | {'bytes held':>11} | {'avg waste':>10} | {'worst waste':>11} | share (avg)")
        for B in (64, 128, 256):
            for g, d in m["groups"].items():
                bb = per_pos[g] * B
                if d["window"] is None:
                    nblk = math.ceil(SEQ / B)
                    held = nblk * bb
                else:
                    nblk = math.ceil(d["window"] / B) + 1
                    held = nblk * bb
                avg, worst = per_pos[g] * B / 2, per_pos[g] * (B - 1)
                print(f"  {B:>4} | {g:<7} | {bb:>12,} | {nblk:>18} | {mib(held):>8.1f} MiB | {mib(avg):>6.2f} MiB | {mib(worst):>7.2f} MiB | {100*avg/held:5.2f}%")
        if m.get("state_mib"):
            print(f"  recurrent state pool: {m['state_mib']} MiB per running sequence, independent of B")
        if name == "Gemma-3-4B" and ename == "f16":
            full_alloc = 34 * 2 * 1024 * 2 * SEQ
            print(f"  for comparison, every layer holding all 32K positions (Metal and WebGPU today): {mib(full_alloc)/1024:.2f} GiB; "
                  f"paged, 5 global layers full + 29 local layers at 1,024 + 1 block: see rows above")
    # option B: one page size for every group (vLLM's choice). A group must then hold the same number of layers per page.
    if name == "Gemma-3-4B":
        # layers per page = 5 (the global group's layer count): local 29 layers -> 6 groups of 5 = 30 slots, one padded
        pad = (math.ceil(29 / 5) * 5 - 29) / (math.ceil(29 / 5) * 5)
        print(f"  option B (uniform page, 5 layers per group): the 29 local layers fill 6 groups of 5, one slot of 30 padded: {100*pad:.1f}% of the local group's bytes wasted")

# Transcode audit, 2026-10-10: exploratory profile

**Exploratory, not a result.** These runs set direction and the projection bands in
[`docs/tasks/task-giw-v16-transcode-2026-10.md`](../../tasks/task-giw-v16-transcode-2026-10.md). They were taken on
neither rig, so no figure here is quoted as a measurement. That task's gates measure on the Mac and on nobara.

## Setup

- **Box:** a Cowork cloud workspace, 2 vCPU amd64, 7 GB RAM. Go 1.27.2, built from source. aikit v1.64.0, goinfer
  `5839762a` (origin/main on 2026-10-10). The transcode files are the same as on the Mac's `992ba55a`, apart from the
  int4mix label fix in `decoder/serialize.go`.
- **Model:** `synth_gguf.py qwen2.5-1.5b`, a GGUF with Qwen2.5-1.5B's shapes (hidden 1536, 28 layers, 12/2 heads, FFN
  8960, vocab 151,936, tied) and Q4_K_M's block types (Q4_K, with Q6_K on the embedding and on llama.cpp's
  `use_more_bits` layers for attn_v and ffn_down). The weights are random bytes with a fixed finite f16 super-scale per
  block. 980 MB, the size of the real Q4_K_M.
  - Why synthetic is enough here: dequantize, quantize and write cost depend on shapes and block formats, not on what
    the weights mean.
  - `embedInt4` was off (`cmd/prequant`'s default). The serve-side default sidecar is `e4h`, which also sends the
    embedding through the int4 quantizer, so int4's share there is if anything higher.
  - What it cannot show: arm64's split (F3, F6), and anything about a real checkpoint's tensor mix beyond the
    Qwen2.5-1.5B shapes.
- **Harnesses** (kept here as `.go.txt` so they never compile into the module):
  - `synthprofile_test.go.txt` lived in `internal/prequant` under the build tag `synthprofile`. It drives
    `decoder.StreamTranscodeGGUF` to a counting sink (with a running sha256), or through `giw.WriteStream` to a file
    followed by `selfCheck`.
  - `fusedexpert_proto_test.go.txt` lived in `decoder` under the same tag. It times one Qwen3-VL-30B-A3B layer of
    `loadFusedExperts`' per-expert work, and compares transpose variants.

## Raw lines

```
# int4, counting sink, CPU profile
RESULT mode=discard quant=int4 target="" bytes=971560639 writes=3218 stream=18.69s
Duration: 18.74s, Total samples = 18.34s (97.85%)
  cum 14.94s 81.46%  aikit/linalg.QuantizeGroupInt4Row      (flat 9.69s)
  cum  5.23s 28.52%  math.Round (inline)
  cum  2.33s 12.70%  aikit/embed.dequantRange               (dequantQ4KBlock 1.35s, dequantQ6KBlock 0.95s)
  cum  0.29s  1.58%  aikit/linalg.quantizeRowInt8Core       (the int8-pinned embedding)
  cum  0.39s  2.13%  decoder.(*giwWriter).weightMatKind
  flat 0.10s  0.55%  hash/crc32.ieeeCLMUL

# int4 to a real file, then selfCheck
RESULT mode="" quant=int4 target="" size=971580419 writes=3218 stream=19.17s selfcheck=0.13s

# int8int8, counting sink
RESULT mode=discard quant=int8int8 target="" bytes=1547351743 writes=3218 stream=3.90s

# int4 with sha256 of the stream: baseline, then T4a (fast round), then T4a + row-parallel, then T4a + layer window
RESULT ... stream=19.46s sha256=b21c4c2d19528d0c   (baseline)
RESULT ... stream=16.21s sha256=b21c4c2d19528d0c   (T4a)
RESULT ... stream=9.74s  sha256=b21c4c2d19528d0c   (T4a + rows split across workers in streamQuantized*)
RESULT ... stream=9.73s  sha256=b21c4c2d19528d0c   (T4a + K-layer window in the generic GGUF loop)

# the window build at GOMAXPROCS=1, cumulative
Duration: 15.29s; per-layer loaders (parallelLayers) 13.38s 88.5%; streamQuantizedEmbed 0.59s 3.9%

# T4a's rounding, exhaustive (a standalone program, both signs)
checked 1266679809 bit patterns x2 signs up to 2^24: mismatches=0   (plus ±Inf, NaN, ±0.49999997, ±0.5, ±6.5, ±7.5)

# one Qwen3-VL-30B-A3B layer of loadFusedExperts work (hidden 2048, moe_inter 768, 128 experts, transposed layout),
# single core, with T4a's rounding already in the quantizer
RESULT one layer (128 experts) total=22.13s conv=2.28s transpose=6.52s copies=1.74s quant=11.60s; x48 layers ≈ 1062s

# transposing gate_up [2048 x 1536] from bf16, 64 repetitions
convert + transposeF32 (as today)            2.141s
fused, 32x32 blocked                         2.149s   (1.0x)
reused output buffer, naive / 32x32          1.823s / 1.936s   (not page faults)
16x16 through a local tile buffer            0.336s   (5.2x vs 1.756s naive in the same run), bit-identical
```

## What the prototypes were

None of this is in the tree. Each was a throwaway edit in the cloud clone, reverted.

- **T4a:** in aikit's `QuantizeGroupInt4Row`, `int(math.Round(float64(x)))` became
  `f := float64(x); if f < 0 { return int(f - 0.5) }; return int(f + 0.5)`.
- **Row split:** contiguous row ranges per worker in `streamQuantizedRow4`'s int4 case and `streamQuantizedRepackable`,
  each worker with its own scratch. This is safe only because the GGUF reader is pure. T2 explains why the production
  form takes a reader factory instead.
- **Window:** in the generic GGUF `sink != nil` loop, `parallelLayers` over windows of `GOMAXPROCS` layers, then
  write in order.

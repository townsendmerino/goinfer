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

## arm64: the Mac, 2026-10-10 (T0's profile)

**Exploratory, not a result:** by day beside the owner's session (load average 2.9-4.1), the disk 98% full (4.3 GB
free), one cold run and one warm run per target, nothing interleaved.

- **Box:** the MacBook Pro, M1 Pro (8 cores), 16 GB, macOS 26.6.2. Go 1.27.2, aikit v1.64.0, goinfer `55c6a6da` (branch
  `giw-v16-2026-10`: T0's instruments, before T6).
- **Model:** the real `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` from `~/models` (1,117,320,768 bytes).
- **Command:** `prequant -quant int4 -target <target> -cpuprofile <file> -o <out> <gguf>`, `-embed-int4` off (the CLI's
  default, as in the amd64 profile above). The numbers below the phase lines come from `/usr/bin/time -l`.

### Phase lines

```
cpu-arm64, first run (source not in page cache)   wrote ...: 1155 MB (int4) in 20.3s: stream 16.5s, self-check 3.5s
metal, straight after                             wrote ...: 1312 MB (int4) in 14.6s: stream 10.9s, self-check 3.6s
cpu-arm64, warm source                            wrote ...: 1155 MB (int4) in 19.5s: stream 16.4s, self-check 3.0s
    19.75 real  10.73 user  2.09 sys; peak memory footprint 1,246,857,400 B (maximum RSS 2,294,726,656 B)
metal, warm source                                wrote ...: 1312 MB (int4) in 15.6s: stream 12.0s, self-check 3.5s
    15.88 real  10.27 user  1.42 sys; peak memory footprint 1,146,226,800 B (maximum RSS 2,383,577,088 B)
```

### pprof, cumulative (the warm runs; the rows that carry the time)

```
# cpu-arm64 (warm source)
Duration: 19.68s, Total samples = 12.29s (62.44%)
         0     0%     0%     12.15s 98.86%  goinfer/internal/prequant.TranscodeTimed
         0     0%     0%     11.56s 94.06%  goinfer/decoder.StreamTranscodeGGUF
     0.01s 0.081% 0.081%      8.18s 66.56%  aikit/embed.dequantRange
         0     0% 0.081%      7.11s 57.85%  goinfer/decoder.streamQuantizedRepackable
     6.05s 49.23% 49.31%      6.72s 54.68%  aikit/embed.dequantQ4KBlock
         0     0% 49.31%      4.12s 33.52%  goinfer/decoder.streamQuantizedRow4
     2.08s 16.92% 66.23%      2.08s 16.92%  encoding/binary.littleEndian.Uint16 (inline)
     1.78s 14.48% 80.72%      1.78s 14.48%  aikit/linalg.QuantizeGroupInt4Row
     0.01s 0.081% 80.80%      1.45s 11.80%  aikit/embed.dequantQ6KBlock
         0     0% 80.80%      0.84s  6.83%  goinfer/decoder.(*giwWriter).weightMatKind
     0.61s  4.96% 85.76%      0.61s  4.96%  hash/crc32.ieeeUpdate
         0     0% 85.76%      0.58s  4.72%  goinfer/decoder.streamQuantizedEmbed (inline)
         0     0% 85.76%      0.56s  4.56%  goinfer/decoder.(*giwWriter).fusedGroup
         0     0% 90.24%      0.54s  4.39%  goinfer/internal/prequant.selfCheck
         0     0% 90.24%      0.52s  4.23%  syscall.write
         0     0% 92.76%      0.25s  2.03%  aikit/linalg.(*WeightMat).RepackInt4Row4
         0     0% 94.79%      0.25s  2.03%  aikit/linalg.quantizeRowInt8Core
         0     0% 94.79%      0.22s  1.79%  goinfer/decoder.repackRow4ForEmit
         0     0% 94.79%      0.20s  1.63%  runtime.mallocgc
         0     0% 97.23%      0.15s  1.22%  goinfer/decoder.(*giwWriter).int4Scales
# metal (warm source)
Duration: 15.84s, Total samples = 11.24s (70.96%)
         0     0%     0%     10.99s 97.78%  goinfer/internal/prequant.TranscodeTimed
         0     0%     0%     10.32s 91.81%  goinfer/decoder.StreamTranscodeGGUF
         0     0%     0%      7.56s 67.26%  aikit/embed.dequantRange
         0     0%     0%      6.44s 57.30%  goinfer/decoder.streamQuantizedRepackable
     5.66s 50.36% 50.36%      6.30s 56.05%  aikit/embed.dequantQ4KBlock
         0     0% 50.36%      3.57s 31.76%  goinfer/decoder.streamQuantizedRow4
     1.86s 16.55% 66.90%      1.86s 16.55%  encoding/binary.littleEndian.Uint16 (inline)
     1.56s 13.88% 80.78%      1.57s 13.97%  aikit/linalg.QuantizeGroupInt4Row
     0.02s  0.18% 80.96%      1.26s 11.21%  aikit/embed.dequantQ6KBlock
     0.68s  6.05% 87.01%      0.68s  6.05%  hash/crc32.ieeeUpdate
         0     0% 87.01%      0.64s  5.69%  goinfer/internal/prequant.selfCheck
         0     0% 91.81%      0.53s  4.72%  syscall.write
     0.01s 0.089% 91.90%      0.39s  3.47%  goinfer/decoder.(*giwWriter).fusedGroup
         0     0% 91.90%      0.38s  3.38%  goinfer/decoder.streamQuantizedEmbed (inline)
         0     0% 91.90%      0.28s  2.49%  goinfer/decoder.(*giwWriter).weightMatKind
         0     0% 94.22%      0.26s  2.31%  aikit/linalg.quantizeRowInt8Core
         0     0% 94.22%      0.21s  1.87%  aikit/linalg.(*WeightMat).RepackInt4Row4
         0     0% 96.26%      0.07s  0.62%  runtime.mallocgc
```

### Reading

| share of sampled CPU | amd64 (above) | arm64, cpu-arm64 target | arm64, metal target |
|---|---|---|---|
| `QuantizeGroupInt4Row` | 81.5% (`math.Round` 28.5%) | 14.5% | 14.0% |
| GGUF dequant (`dequantRange`) | 12.7% | 66.6% | 67.3% |
| of which `dequantQ4KBlock` | 7.4% | 54.7% (`binary.LittleEndian.Uint16` 16.9%) | 56.1% (16.6%) |
| of which `dequantQ6KBlock` | 5.2% | 11.8% | 11.2% |
| int8 quantize (the embedding) | 1.6% | 2.0% | 2.3% |
| in-RAM row4 repack (`WeightMat.RepackInt4Row4`, F6) | none (a no-op off arm64) | 2.0% (0.25 s) | 1.9% (0.21 s) |
| the writer's own row4 repack for kind 5 (`repackRow4ForEmit`) | none | 1.8% (0.22 s) | none |
| self-check (`selfCheck`, nearly all CRC32) | not in that profile | 4.4% (0.54 s) | 5.7% (0.64 s) |

- **F3: arm64's split is not amd64's.** The int4 quantizer is about 14% of the CPU here, not 81%, and `math.Round` is not
  a node in the profile at all (it compiles to one instruction). The cost that dominates is the GGUF dequantizer, about
  two thirds, most of it `dequantQ4KBlock`, with `encoding/binary.LittleEndian.Uint16` alone at 17%.
  - So on the Mac a faster int4 quantizer (T4) can remove at most about 14% of the transcode's CPU.
  - The dequantizer, which the task doc ranks after T4 from its 12.7% on amd64, is already the first cost on arm64.
- **F6: the wasted in-RAM repack is there, and it is small.** `RepackInt4Row4` runs on both targets (0.21-0.25 s, about
  2% of the CPU). On the cpu-arm64 target the writer then repacks from canonical again for kind 5 (0.22 s). T6 removes
  the first of those.
- **One core is busy,** as on amd64: user plus system time is 12.8 s of 19.8 s wall on the cpu-arm64 run, and nothing
  in the profile runs in parallel.
- **The self-check is not negligible here (F9):** 3.0-3.6 s of wall on a 1.2-1.3 GB file that fits in RAM, 15-25% of the
  total, against 0.13 s on the cloud box. Its CPU is 0.5-0.6 s, the CRC. The rest is off-CPU and this profile cannot say
  where; the disk was 98% full and the box was in use.
- **Wall exceeds CPU in the stream too** (16.4 s of stream on the warm cpu-arm64 run against 12.8 s of CPU for the whole
  process). A CPU profile does not attribute that either.

## What the prototypes were

None of this is in the tree. Each was a throwaway edit in the cloud clone, reverted.

- **T4a:** in aikit's `QuantizeGroupInt4Row`, `int(math.Round(float64(x)))` became
  `f := float64(x); if f < 0 { return int(f - 0.5) }; return int(f + 0.5)`.
- **Row split:** contiguous row ranges per worker in `streamQuantizedRow4`'s int4 case and `streamQuantizedRepackable`,
  each worker with its own scratch. This is safe only because the GGUF reader is pure. T2 explains why the production
  form takes a reader factory instead.
- **Window:** in the generic GGUF `sink != nil` loop, `parallelLayers` over windows of `GOMAXPROCS` layers, then
  write in order.

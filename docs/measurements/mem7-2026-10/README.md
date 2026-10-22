# MEM7: the Metal prefill block sized to what the memory budget leaves, 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, §4 MEM7, "MEM7 gates" (BLK-G), registered before the code.
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2, darwin/arm64. BLK-G3 read
`~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw` (local disk), under the timing lock, started at 08:39 with
load 3.26 and 4.90 GB available (the void bars: 4.0, and the 1.5B's 1.6 GB need plus 2 GB).

| gate | reading | result |
|---|---|---|
| BLK-G1 identity | `TestI8_blocksMatchOnePass_dense` (64, 100, 256 rows) and `_MoE` (mixtral, qwen3moe, qwen3moe-k3, qwen3.5-moe hybrid): tokens, log-probabilities and resident state byte-equal to one pass (`blk-g1-g2.log`) | **PASS** |
| BLK-G2 sizing | on the dense fixture with a 3 GiB need: 1 GiB of headroom gives today's 47,616 rows; 200 MiB gives 18,560 rows (209,134,080 bytes, the most 64-row multiple under 200 MiB); 1 MiB gives the 256-row floor; the exact-kernel and int8-KV cases give 0. The planted defect (headroom ignored) red at 200 MiB and 1 MiB | **PASS** |
| BLK-G3 speed | 12,288 tokens on the 1.5B: today's block 8,000 rows (passes 8,000 + 4,288) against the 128 MiB block's 1,984 rows (seven passes); five pairs after a warm-up, each pair's logits byte-equal: ratios 1.0116–1.0204, **median 1.0137** (17.28–17.33 s against 17.47–17.68 s) (`blk-g3.log`) | **PASS** |

The 1.4% is what the smaller block costs when the budget forces it: on this Mac at 4.9 GB available the 1.5B's block stays
at 512 MiB. A first BLK-G3 attempt at 08:38 loaded the `.gguf` directly and the fit guard refused it (3.3 GB against a 3.2
GB budget); no timing was read, and the test now loads the sidecar.

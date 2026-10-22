# MEM12-f32res: a quantized CPU KV cache reserves no f32 capacity, 2026-10-21

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, §4 MEM12, "MEM12-f32res gates" (F32R-G), registered before the code.
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2, darwin/arm64; the owner's desktop open.

| gate | reading | result |
|---|---|---|
| F32R-G1 bytes | `llama-tiny` and `gemma2-tiny`, `NewCache(512)`: under f16 and int8 every layer's f32 K/V capacity is 0 and the quantized capacities are 512 × the KV width; under f32 the f32 capacities are unchanged. What the hint costs over `NewCache(0)` equals the quantized reservation to the byte: llama f16 262,144, int8 163,840; gemma2 f16 524,288, int8 294,912 (`g1.log`) | **PASS** |
| F32R-G2 identity | `TestKVF16_*` (7), `TestKVI8_*`'s three tiny tests, and the 18 prefix- and session-share tests PASS (`g2.log`); `scripts/refresh_parity_hashes.sh` green, forward goldens 33 passed, 0 failed, 31 skipped (heavy enabled), 42 `deps_hash` lines refreshed. **Deviation:** the three heavy `TestKVI8_*` tests default to `gemma-3-4b-it`, which the fit guard refused (5.2 GB against a 4.6 GB budget with the owner's apps open); not bypassed. Run on `qwen2.5-0.5b-instruct` instead, branch and `origin/main` read identically: the snapshot round-trip's logits bit-identical (PASS on both), and the two int8-against-f32 quality bars at cosine 0.98859, argmax 89.5% on both (FAIL on both: the 0.99 bar is the 4B's, `g2-int8-0p5b.log`) | **PASS** (identity; deviation stated) |

Planted defect, run once on the committed tree and reverted: `NewCache` passing the f32 hint again and `setQuant` keeping the
f32 capacity, the tree before this change. G1 red: the hint costs llama f16 786,432 and int8 688,128 bytes, gemma2 f16
1,572,864 and int8 1,343,488, the quantized reservation plus the f32 one (an f16 cache's hint cost three times its own
storage).

Seen while reading, not part of this item: a ring (sliding-window) layer gets an append-forever reservation it never
uses, in every storage form. `NewKVCache` reserves the hint on every attention layer before `enableRings` makes the
local layers rings, and `setQuant` reserves its quantized arrays on every layer too (gemma2-tiny's figures above include
both).

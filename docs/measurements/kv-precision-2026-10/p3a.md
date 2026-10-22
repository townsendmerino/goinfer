# P3a: Metal --kv f32 back, 2026-10-22

Gates: `docs/tasks/task-kv-precision-everywhere-2026-10.md`, "P3 gates" (P3-G), registered before the code (5ccb5dde). Machine:
MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2. **Status: PASS. P3-G2 failed as registered on one fixture; the owner amended it after the data (D6, owner, 2026-10-22), and it
passes as amended.**

| gate | reading | result |
|---|---|---|
| P3-G1 identity | `attention_f32` (now derived from `attention`'s source) against `attention` on f16-exact K/V, hd 64 and 128, 37 to 9,000 keys, with and without a sink and a window: bit-identical (`TestP3_attentionF32MatchesF16OnExactInputs`); `TestAttentionKernelsPastTileBound` now runs `attention_f32`'s sink cases too. The old hand-written kernel, planted back: red on every sink case | **PASS** |
| P3-G2 agreement, as registered | Metal at f32 against the CPU at f32, 32 positions teacher-forced, beside Metal at f16, max |logit diff| over all positions: llama 8.9e-8 (f16: 0.0095), gemma2 6.0e-7 (0.19), cohere 3.0e-8 (0.0010), qwen25vl 1.2e-7 (0.0053; top-1 32/32 against 31/32), qwen35 0.0045 (0.0054), qwen3next 0.0057 (0.0058), nemotron 0.0109 (0.0115), deepseek 0.0081 (0.0081, its latent is f32 either way). **qwen3moe: 0.0123 at f32 against 0.0076 at f16**, top-1 32/32 both | FAIL on qwen3moe (D6) |
| P3-G2 agreement, as amended after the data (D6, owner, 2026-10-22) | mean over positions of the per-position max |diff|, f32 (f16): llama 4.2e-8 (0.0059), qwen3moe 0.00571 (0.00575), gemma2 1.7e-7 (0.094), cohere 7.5e-9 (0.00069), qwen25vl 3.4e-8 (0.0038), qwen35 0.002845 (0.002835, +0.4%), qwen3next 0.00368 (0.00364, +1.1%), nemotron 0.00750 (0.00758), deepseek 0.00515 (0.00515); top-1 no lower on all nine | **PASS** |
| P3-G3 bytes | all nine fixtures at `--kv f32` (`ResidentContext` 100, padded): K/V buffers 4 B per element and equal to `ResidentKVBytes("metal", …)` | **PASS** |
| P3-G4 paths | under f32 on all nine: `PrefillPath` sequential naming the f32 cache, `PrefillBlockRows` 0, batched decode ineligible, flash decode off. Plant (the prefill decline removed): red | **PASS** |
| P3-G5 unchanged | the generated matrix: exactly the 12 Metal `--kv f32` cells move, refused → f32; `TestKVPrecisionDelivered`, `TestResidentKVPrecision_metalReportsWhatRuns` PASS; Metal logits at `auto` and `--kv i8` over 32 positions byte-identical to main's on all nine fixtures (18 arms; deepseek's i8 arm runs on the CPU on both trees, D5); `touched.py` green, `metal/kernels` included; goldens 33/0/31 | **PASS** |

**P3-G2 on qwen3moe, diagnosed.** Per position, both arms sit at the same floor against the CPU (0.003–0.007 at every position for
either precision; means about 0.0057 at f32 and 0.0058 at f16): on this MoE fixture Metal's difference from the CPU is the MoE path's
own arithmetic, not the KV cache's. One position (24) reads 0.0123 at f32, which sets the max. A bar on the maximum over positions is
decided by that one position; on every fixture where the cache is the error that matters (the dense ones), f32 is 4 to 6 orders of
magnitude closer. The owner chose D6's option (a), the mean of the per-position max within 5%, on 2026-10-22; the reading above is the gate as amended.

Planted defects: the old `attention_f32` (G1 red), one f32 decline removed (G4 red), f32 priced at 2 B (G3 red).

A first G3 readback double-counted deepseek's buffers (an MLA layer's V aliases its latent, `metal/mla.go`); the test counts it once,
and Metal's MLA pricing is right.

After `main` brought in G3 (`TestKVDeliveredBytes_Metal`, `docs/measurements/kv-precision-2026-10/g3.md`), its readback runs on this
branch's f32 cells too: 35 Metal cells read back, every resident f32 cell 4 B per element, and the three no-resident `/f32` cells
leave its matrix-differs list (the matrix's Metal answer is now f32, which is what their CPU cache holds).

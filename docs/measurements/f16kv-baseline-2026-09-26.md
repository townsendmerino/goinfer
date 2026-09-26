# R6 phase 3 (f16 resident KV on CUDA): baseline before any kernel work, 2026-09-26

Why this exists: `docs/prompts/cuda-r6-flash-decode.md` §"Phase 3" pre-registered its band against
phi3-mini at int4 (goinfer 56.2 tok/s, Ollama 73.3 at depth 3900). As of `cad1973b`, Phi-3 loads at
int8int8 with per-32 activation scales (queue-engineering.md H2), because int4 without per-32 gives
junk output. This record re-measures the baseline under that default before anyone rules on the band.

**Provenance.** `nobara-pc`, RTX 2070 SUPER 8 GB, driver 595.91.07. goinfer `cad1973b`, serve binaries
`~/bench-peer-claim/serve-{cuda,cpu}-cad1973b`. Ollama from `~/ollama-0325`, model `p3m-local`
(same weights as the goinfer file, 195/195 tensors, per `peer-claim-2026-09-25.md`). llama.cpp from
`~/mycode/peers/llama.cpp`. Model `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`. Greedy,
essay-v2 prompts, `scripts/bench_peer.py`, `BENCH_RUNS=3`, engines interleaved cell by cell with a
restart between cells, idle box (preflight loadavg 0.20, GPU 49 °C). Raw data:
[`f16kv-baseline-2026-09-26/`](f16kv-baseline-2026-09-26/).

## Results (decode tok/s, mean of 24 completions; spread ≤ 0.6 on every cell)

| run | goinfer config | depth | goinfer | Ollama | llama.cpp | goinfer / llama.cpp |
|---|---|---|---|---|---|---|
| A | int8int8 + per-32 (the default), `BENCH_CTX=3072` | 128 | 89.8 | 125.9 | 127.6 | 0.70 |
| A | same | 2048 | 63.0 | 91.4 (token gate void) | 93.1 | 0.68 |
| B | explicit `-quant int4`, per-row | 128 | **9.0, on the CPU** | 125.8 | 127.7 | — |
| B | same | 2048 | **5.0, on the CPU** | 91.5 (token gate void) | 93.1 | — |
| B | same | 3900 | **3.1, on the CPU** | 74.5 | 75.2 | — |

The two Ollama cells marked void reported no token count (the known `usage` gap). Their rates are
shown for scale only; llama.cpp is the peer column for this record.

## What it says

1. **The band's original cell can no longer be measured.** Explicit int4 Phi-3 is declined on CUDA by
   design: `residentGateReasonAct` waives the H2 hazard only for per-32 without int4 weights. So
   run B ran on the CPU. The last resident int4 figure is yesterday's 59.7 tok/s at 3900
   (`peer-claim-2026-09-25.md`, table b), and that path produced junk output.
2. **Under the default (int8int8 + per-32), depth 3900 does not fit resident at f32 KV.** 3.6 GB of
   int8 weights leave about 3.09 GB for KV after the 384 MiB margin. At 786 KB per position in f32,
   that is about 3930 positions, and a 3930-token prompt plus 64 generated tokens needs about 3994.
3. **Decode is bandwidth-bound, and the fit is tight.** Bytes per token = weights + f32 KV × positions.
   Both run A cells then imply the same effective bandwidth: 335 GB/s at 128 and 330 GB/s at 2048.
4. **Projection for f16 KV, from that fit (a projection, not a measurement):**

   | depth | f32 KV today | f16 KV projected | gain | vs llama.cpp |
   |---|---|---|---|---|
   | 2048 | 63.0 | ~74.7 | 1.19× | 0.80 |
   | 3900 | does not fit (~49 if it did) | ~64 | resident at all | 0.85 |

   So f16 KV alone does not reach the phase-3 band's 70 tok/s ship bar at 3900 for int8 Phi-3. The
   same fit projects under 70 there, which is inside the 62–70 "parked" band.
5. **At shallow depth, the gap is the weights, not the KV.** At depth 128, Phi-3 is now 0.70× the
   peers, where yesterday's int4 junk-output path read 1.14× AHEAD. That is the speed cost of the H2
   quality fix: int8 weights are about 1.6× the bytes of the peers' q4. f16 KV cannot touch that.
   The lever there is int4 weight quality (Track B, parked: asymmetric int4 or keeping Q4_K).
6. **f16 KV mostly helps full-attention, non-GQA geometries.** For a GQA model like Qwen2.5-7B, the
   same fit projects only ~1.05× at 3900, because its KV is 114 KB per position against Phi-3's 786 KB.

## The decision this record leaves for the owner (nothing here amends the band)

The phase-3 band's premise, int4 Phi-3 at 3900 as a KV-bound cell, no longer holds for the default
configuration. Choosing a replacement band is the owner's call, made before any f16 run. The options
are set out in the reply that shipped this record. This file records only the measurements and the
projection.

# Mellum2.1 on CUDA: where a cold prefill spends its time, and the plan at 16k and 32k

Exploratory by-day readings (one run each, no repeat, not quotable as results) that decide whether a grouped, exact MoE prefill GEMV (option A in the scoping note
given to Francis, 2026-10-19) is worth building. Run by `run-ffn-share.sh` under the timing lock, 2026-10-19 08:49:42 to 08:52:30 PDT, on nobara
(RTX 2070 SUPER, driver 595.91.07), nothing else on the card, `~/models/mellum2.1-thinking.int4.e4h.cuda.v17.giw` (local path), serve
`~/serve-cuda-b2b5e816` for the plans, this branch's test for the phases. Logs: `~/goinfer-logs/mellum21-ffn-share-2026-10-19/`.

## Phases (`TestPrefillDecomp`, `--moe-cache-experts`, `-ctx 8192`, random embeddings, one profiled run after one unprofiled warm pass)

| M | unprofiled wall | gemv | attention | glue | MoE FFN | share of the profiled sum |
|---|---|---|---|---|---|---|
| 512 | 5.29 s (96.8 tok/s, 10.33 ms/token) | 38 ms | 25 ms | 23 ms | 4.986 s | 98.3% |
| 4096 | 39.98 s (102.4 tok/s, 9.76 ms/token) | 306 ms | 888 ms | 187 ms | 40.09 s | **96.7%** (attention 2.1%) |

The category boundaries are stream syncs, so the profiled sum (41.47 s at 4096) runs 1.5 s over the pipelined wall time. "MoE FFN" is the whole routed-expert FFN of a
layer (route, expert DMA, expert GEMVs, fold): `prefill.go` now times it as its own category (`moeCat`), which the profiler used to count under gemv.
The unprofiled rate agrees with KV7's cold prefills (95 to 98 tok/s at 6 to 7.5k ids).

## Plans (`plan_banner.py`, the server's own banner, free VRAM left in)

| `-ctx` | expert-cache slots per layer | free VRAM when sizing | KV |
|---|---|---|---|
| 16384 | 57 of 64 | 6.2 GB | 0.6 GB, windowed layers 1,536 positions |
| 32768 | 52 of 64 | 5.7 GB | 1.0 GB |

Both are `decode path: cuda-resident (int4)`, one conversation. (The scoping note's 40 to 45 slots at 32k was arithmetic and was wrong: 52 are planned.)

## What it decides

The expert FFN is nearly all of a cold prefill (96.7% at 4096). Attention is 2%. Any lever that does not shorten the expert FFN cannot move Mellum2.1's
prefill by more than a few percent; one that cuts the FFN by a factor f moves it by about 1 / (0.033 + 0.967 / f).

# Mellum2.1 on CUDA: where a cold prefill spends its time, and the plan at 16k and 32k

Exploratory by-day readings (one run each, no repeat, not quotable as results) that decide whether a grouped, exact MoE prefill GEMV (option A in the scoping note
given to Francis, 2026-10-19) is worth building. Run by `run-ffn-share.sh` under the timing lock, 2026-10-19 08:49:42 to 08:52:30 PDT, on nobara
(RTX 2070 SUPER, driver 595.91.07), nothing else on the card, `~/models/mellum2.1-thinking.int4.e4h.cuda.v17.giw` (local path), serve
`~/serve-cuda-b2b5e816` for the plans, this branch's test for the phases. Logs: `~/goinfer-logs/mellum21-ffn-share-2026-10-19/`.


> **Read this first (B37, 2026-10-19).** Every in-process reading in this file except where it says otherwise was taken before the harness loaded the way serve does: `Options.ResidentWindowedKV` was off
> (serve's default is on), so the expert cache had fewer slots than serve plans at the same flags (55 at `-ctx 8192` here; 26 against serve's 53 at 32768 in the 16k sections). The figure to quote is the served one,
> `docs/measurements/mellum21-served-16k-2026-10-19/README.md`: 16,308 tokens, 162 s to 45 s, 3.60x, 53 slots. The 4k tables and the FFN-share profile below are labelled "in-process, windowed KV off (pre-B37)";
> they were not re-run on the serve-shaped harness.

## Phases (in-process, windowed KV off (pre-B37); `TestPrefillDecomp`, `--moe-cache-experts`, `-ctx 8192`, random embeddings, one profiled run after one unprofiled warm pass)

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

## Option A: grouped, exact expert kernels (built 2026-10-19, `cuda/moe_rows.cu`)

The 4096-token profile above shows the expert FFN at 96.7% and about 14.5 us per launch (12k launches per layer per 512-row chunk: three per (row, rank) pair), so the expert-major path was launch-bound.
`gemv_w4a8_moe_rows` and `glu_quant_rows` take all of one expert's pairs in one launch per stage (three launches per expert) and keep each pair's accumulator in the order, with the rounding, of
`gemv_w4a8_moe` (the audited `moe.ptx`), so the result is the same bits; the weight words are loaded once per tile of 8 pairs.

**Tier 0** (`go test -tags 'cuda goinfer_testhooks' -run '^TestMoERows_'`, 8 s): the kernels equal the per-pair kernels on 228 output sets at Mellum2.1's geometry (gate||up 1792 x 2304,
down 2304 x 896, a tiny K that is only the ragged tail; 1, 7, 8, 9 and 37 pairs; permuted activation rows and output index; both activations for `glu_quant_rows`). The residual of every prompt row equals
the per-pair path's on both tiny MoE fixtures at M = 5, 40 and 133 with the cache evicting (the existing `TestMoEExpertMajorCUDA_residualIdentical` and `_bitIdentical` also pass, and now run the grouped
path). The plant, each pair's output filed under the wrong rank, makes 2560 of 2560 residual elements differ.

**Tier 1** (in-process, windowed KV off (pre-B37); `TestMoERows_realMellumIdentityAndAB`, real Mellum2.1 `.giw`, `--moe-cache-experts -ctx 8192`, 55 slots per layer, 2026-10-19 17:37:28 to 17:39:48 PDT, under the lock, serial; exploratory, one session):

| | per-pair (before) | grouped |
|---|---|---|
| residual of all 512 rows, 28 layers | reference | **identical on all 1,179,648 elements** |
| 4096-token cold prefill, run order off, on, on, off | 39.896 s, 39.999 s | 11.264 s, 11.325 s |
| rate | 102.5 tok/s (9.75 ms/token) | **362.6 tok/s (2.76 ms/token)** |
| paired ratio | | **3.54x** (3.55x and 3.53x in the two pairs) |

The last row's logits are identical across all four runs. 16k was not measured; the 3.5x is at M = 4096. Part of the remainder is expert DMA (about 3 MB per admitted expert, synchronous) and the
dense attention half, and a profile of the grouped path is the next reading if more is wanted.

### 16384 tokens (2026-10-19 17:59:39 to 18:07:34 PDT, test at `c875f294` on main `e006afbb`, under the lock; exploratory, one session)

Same test (`TestMoERows_realMellumIdentityAndAB`), `GOINFER_ROWS_AB_M=16384 GOINFER_ROWS_AB_CTX=32768`, `--moe-cache-experts`, order per-pair, grouped, grouped, per-pair.
Per run, as it started: load average 2.77 / 3.70 / 4.26 / 4.52 (the test's own threads included), `MemAvailable` 52.0 GB, VRAM free 475 MiB (the process's own).

| | per-pair (before) | grouped |
|---|---|---|
| residual of all 512 rows | reference | identical on all 1,179,648 elements |
| 16384-token cold prefill, runs in order | 2m50.868s, 2m52.178s | 54.335 s, 54.359 s |
| rate | 95.5 tok/s (10.47 ms/token) | **301.5 tok/s (3.32 ms/token)** |
| paired ratio | | **3.16x** |

The last-row logits are identical across the four runs. The per-token cost at 16k is higher than at 4k for both arms (10.47 against 9.75 ms, 3.32 against 2.76 ms): the full-attention layers' share grows with the context.

**Plan caveat.** This load capped the expert cache at **26 slots per layer** ("64 slots/layer would need 6.3 GB VRAM but only 3.1 GB free"), not the 52 that `-ctx 32768` plans on a quiet card (5.7 GB free).
About 2.9 GB of the card was held by something else when the model loaded, and I did not find out what (no tack process was running at 17:59; tack's next one started at 18:08). Both arms ran
the same 26 slots, so the ratio is a like-for-like reading at that plan; at 52 slots both arms miss the cache less and the ratio may differ. A re-run on a quiet card would settle it.


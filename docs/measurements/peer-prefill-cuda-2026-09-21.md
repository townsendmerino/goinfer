# CUDA prefill vs Ollama, re-measured 2026-09-21 after the chunk-demotion fix and the 128-row attention tile

`scripts/bench_peer_prefill.py`, goinfer commit `fd34ce30` (serve-cuda built from it; the header's `dirty: true` is one untracked log file, no tracked change), Ollama v0.32.5 (`~/ollama-0325`; the header's `peer_version` reads a warning because the header is captured before Ollama starts),
RTX 2070 SUPER, driver 595.91.07, Nobara 44, idle box (only the compositor on the GPU), qwen2.5-coder 0.5B / 1.5B q4_k_m, n=6 unique-prefix prompts per cell, greedy. Goinfer and Ollama alternate inside each cell, per the harness.
Raw: `peer-prefill-cuda-2026-09-21.json` / `.log`. Ratio below = Ollama TTFT / goinfer TTFT (>1 = goinfer faster).

## TTFT (user-visible time to first token; each engine's own request overhead included)

| model | K | goinfer | Ollama | Ollama/goinfer |
|---|---:|---:|---:|---:|
| 0.5B | 512 | 35 ms | 373 ms | 10.8x |
| 0.5B | 1024 | 72 | 399 | 5.6x |
| 0.5B | 2048 | 168 | 443 | 2.6x |
| 0.5B | 3900 | 377 | 592 | 1.57x |
| 1.5B | 512 | 88 | 444 | 5.05x |
| 1.5B | 1024 | 184 | 501 | 2.7x |
| 1.5B | 2048 | 432 | 655 | 1.51x |
| 1.5B | 3900 | 955 | 951 | **1.00x (parity)** |

goinfer's cell spreads are 0.6-6.5% at K >= 1024 (19.6% at 0.5B K=512, where TTFT is 35 ms); Ollama's are 4-26%. **TTFT is not prefill speed:** Ollama carries a ~330-350 ms fixed cost per request (fitted overhead 329 / 350 ms), goinfer none, so the short-prompt "wins" are mostly that floor.

## Prefill throughput (marginal, the engine's own speed with the overhead cancelled)

The harness's single least-squares marginal for goinfer is flagged **`LINEAR_FIT_INVALID`** (negative fitted overhead: goinfer's TTFT is superlinear in K), so the headline fit ratios it prints (0.5B 1.58x, 1.5B 1.71x behind) are NOT quoted; the per-interval (local) marginals are:

| tok/s over the interval | 0.5B goinfer / Ollama | 1.5B goinfer / Ollama |
|---|---|---|
| ~512-1024 | 13,727 / 19,617 (Ollama 1.43x ahead) | 5,328 / 8,983 (1.69x) |
| ~1024-2048 | 10,736 / 23,492 (2.19x) | 4,169 / 6,736 (1.62x) |
| ~2048-3900 | 8,883 / 12,480 (1.40x) | 3,545 / 6,255 (**1.76x**) |

So on throughput Ollama is still **1.4-2.2x ahead**; goinfer's marginal still degrades with K (5,328 -> 3,545 tok/s on the 1.5B) while Ollama's is flatter, the residual O(K^2) attention term. The two views disagree because of the fixed overhead, exactly as the harness docstring warns.

## Against the previous anchor (2026-09-05..09, `benchmarks.md`)
Marginal "1.5B 3.16x / 0.5B 1.89x behind" was a whole-curve fit; the comparable local figures now are 1.5B ~1.7x and 0.5B ~1.4-2.2x (not a like-for-like fit, so no single "improved by" number is claimed). **The old `serve-cuda` (2026-09-09 build) was NOT re-run this session**, so this is not a same-session before/after; the in-repo before/after is `TestPrefillDecomp` (`prefill-chunk-demotion-2026-09-21.md`, `attn-fused-tile128-default-2026-09-21.md`): 1.5B K=3900 goinfer-side 5.0 s (regressed) -> 1.42 s -> 0.95 s.
Where the 1.5B lands at K=3900: TTFT 955 ms, i.e. R5's 0.71-0.84 s target is NOT met (the 0.5B is 377 ms).

## Notes / not established
- goinfer's repeat-vs-fresh reads 0.02-0.14 (identical prompts are served far faster than fresh ones): the serve binary now has a prompt cache that the 2026-09-01 record said it did not. The unique `Session NNNN.` prefix defeats it (fresh cells rise with depth: 35 -> 377 ms, 88 -> 955 ms; the scaling check passes), so the table is fresh-prompt prefill. Not investigated further here.
- Ollama caches too (repeat/fresh 0.36-0.96, healthy). Only 0.5B/1.5B and hd128 K<=3900 were measured; 7B was not; nothing here says anything about batch > 1 or long contexts past 3900.

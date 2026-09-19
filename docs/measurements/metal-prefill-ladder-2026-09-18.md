# Metal prefill ladder, re-measured post M-03/M-04 — 2026-09-18

**R4 step 0** (`docs/tasks/red-october.md`): replace the pre-fix Metal prefill peer rows
(`legacy-benchmarks.md` §A, "Metal prefill — 2026-09-09") with a same-session measurement of the
current kernels, and decide whether R4 step 2 (a further GEMM tile change) is funded.

**Result: step 2 is KILLED.** K=512 reads 2.54× behind Ollama on TTFT — inside neither the ship
band (≤1.8×) nor the park band (1.8–2.4×) the audit's own projection registered, but past the kill
line (above 2.4×). Per the pre-registered rule: *"the tile is not the lever the audit thought and
the parity path needs a different kernel design."* M-03 and M-04 did measurably help — see
"Against the pre-fix baseline" below — just not enough to reach the projected band.

## Provenance

goinfer `e2927f41` (working tree carried one untracked file at run time —
`docs/measurements/r4-step0-run.log`, this run's own log; not a code change, and the run's header
JSON's `goinfer_dirty: true` is that file, not a dirty tree). Ollama v0.32.5 (Flash Attention on,
`OLLAMA_KV_CACHE_TYPE=q8_0` — this Mac's standing peer-comparison convention, `benchmarks.md` §B3
provenance line). Apple M1 Pro, 16 GB, macOS 26.6.2. Model: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`
(1.5B int4), `~/models` (internal SSD) on both sides — the Ollama tag `q15` was `ollama create`d
`FROM` this same file in an earlier session, so both engines load the identical checkpoint, not a
different quantizer's build.

`serve-metal` (`~/bench-cur/serve-metal`) rebuilt fresh from `e2927f41` before this run — the
previously cached binary at that path was from 2026-09-09, predating both M-03 and M-04.

Three arms interleaved per depth: `goinfer_exact` (`--exact-prefill`, sequential per-token,
skips the batched path entirely), `goinfer` (fast path, default), `ollama`. n=6 distinct prefixes
per cell, medians reported. `scripts/bench_peer_prefill.py --backend metal --models 1.5B --depths
256,512,1024,2048,3900 --n 6`. Cache-check confirmed every arm's repeat/fresh ratio stayed low
(fresh prompts miss the engine's own cache) at every depth. Raw output:
`docs/measurements/metal-prefill-ladder-2026-09-18-raw.json`.

## TTFT tok/s (prompt_tokens / wall-clock TTFT — includes per-request overhead)

| K | goinfer exact | goinfer fast | fast / exact | Ollama | Ollama / goinfer fast |
|---|---|---|---|---|---|
| 256 | 84.9 (spread 2.4%) | 366.2 (spread 31.5%) | 4.31× | 773.0 (spread 7.8%) | 2.11× |
| 512 | 83.0 (1.2%) | 349.5 (16.4%) | 4.21× | 886.3 (3.8%) | **2.54×** |
| 1024 | 77.6 (1.7%) | 294.1 (8.2%) | 3.79× | 923.1 (1.0%) | 3.14× |
| 2048 | 68.2 (0.5%) | 262.5 (2.5%) | 3.85× | 874.5 (0.7%) | 3.33× |
| 3900 | 56.4 (0.8%) | 225.0 (1.6%) | 3.99× | 781.8 (0.2%) | 3.47× |

K=256 is no longer identical between exact and fast, unlike the 2026-09-09 row — R3's brief notes
the fast-path floor moved from 512 to 256 (M-02) since that measurement, so the fast path is live
at every depth swept here. goinfer fast's spread is markedly higher than exact's or Ollama's,
worst at K=256 (31.5%) and falling as depth grows (16.4% → 8.2% → 2.5% → 1.6%) — consistent with a
short, noisy dispatch-floor region rather than a depth-dependent effect; not investigated further
here (out of scope for step 0).

## Overhead-free marginal throughput

**The whole-curve linear fit is invalid for all three arms** — the script's own check:
`LINEAR_FIT_INVALID: "negative fitted overhead: TTFT is superlinear in K, so marginal_tok_s is not
a constant for this engine — read local instead."` This includes Ollama, unlike the 2026-09-09 row
where Ollama's fit was valid (near-flat 900–1140 tok/s) — worth a note for whoever revisits this,
since it means Ollama's curve isn't flat as depth grows here either, not only goinfer's:

| K interval | goinfer exact | goinfer fast | Ollama |
|---|---|---|---|
| 256→512 | 81.2 tok/s (12.31 ms/tok) | 333.9 tok/s (2.99 ms/tok) | 1058.7 tok/s (0.94 ms/tok) |
| 512→1024 | 72.7 tok/s (13.75 ms/tok) | 253.3 tok/s (3.95 ms/tok) | 965.3 tok/s (1.04 ms/tok) |
| 1024→2048 | 60.8 tok/s (16.45 ms/tok) | 237.3 tok/s (4.22 ms/tok) | 830.0 tok/s (1.20 ms/tok) |
| 2048→3900 | 47.3 tok/s (21.12 ms/tok) | 194.0 tok/s (5.16 ms/tok) | 698.3 tok/s (1.43 ms/tok) |
| whole-curve fit (invalid, orientation only) | 54.9 tok/s | 217.6 tok/s | 778.7 tok/s |

**goinfer fast's marginal cost is itself growing with depth (2.99 → 5.16 ms/token, ~1.72×), not
flat.** Ollama's grows too, but more mildly (0.94 → 1.43 ms/token, ~1.52×) — so on the local
intervals goinfer is losing ground to Ollama as K grows (ratio 3.17× → 3.60× → 3.49× → 3.60×), not
holding a constant multiple behind. The 2026-09-09 row attributed the whole gap to a flat GEMM
term with attention as the K-growing residual; this run's shape (both terms growing) is not
inconsistent with that reading, but it isn't a re-confirmation of it either — no per-kernel
profile was taken here, this is TTFT only, which is exactly step 0's scope. A mechanism read (in the spirit of R5's
"profile before designing the next fix" discipline) is what step 2 would need if it were
re-opened.

## Against the pre-fix baseline (2026-09-09, `legacy-benchmarks.md` §A)

M-03 and M-04 measurably helped — this is not a regression:

| K | pre-fix ratio (Ollama/goinfer fast) | this run | direction |
|---|---|---|---|
| 512 | 3.33× | 2.54× | improved |
| 3900 | 8.81× (TTFT) / ~4.3× (L2 record, contemporaneous) | 3.47× | improved |

The internal fast/exact speedup also changed shape: pre-fix it fell from 3.75× at K=512 to 2.04×
at K=3900 (the fast path's own advantage shrinking with depth); this run holds flat around
3.8–4.3× at every depth. Both are consistent with M-04's attention-tiling work doing something
real to the O(K²) term. What did not happen is reaching the audit's registered post-M-03
projection of ≤1.8× at K=512 for a ship verdict — the realized improvement (3.33×→2.54×) is
real but roughly half the ~1.85× (3.33/1.8) the projection implied a 2× GEMM speedup should buy.

## Decision rule (as registered in `docs/tasks/red-october.md` R4)

*"Step 2's band: K=512 ≤1.8× behind ships (the audit's own projection for a 2× GEMM), 1.8–2.4×
parked, above 2.4× killed (the tile is not the lever the audit thought and the parity path needs
a different kernel design — say so)."*

**K=512 measured at 2.54× behind → KILLED.** No further GEMM-tile work is funded on this evidence.
This is a negative result recorded with the same care as a win, per `CLAUDE.md`'s measurement
discipline — it closes step 2 as scoped, it does not close R4's line of inquiry (a mechanism
investigation, not a re-roll of the same tile design, would be the next registered step if this
gap is picked up again).

## Out of scope

Attention below 18% of TTFT (unchanged from R4's original scope), MoE prefill (R11), the
short-prompt floor (R3), a per-kernel `ncu`-style profile of where the remaining GEMM/attention
split actually sits post-M-03/M-04 (not attempted here — TTFT only).

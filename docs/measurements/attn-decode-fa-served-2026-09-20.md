# attn_decode_fa served speed (R6 step 2): the lane clears both ship bands; fidelity is still unscored

Speed only. **Fidelity is NOT established:** the registered gate (`attn-decode-fa-fidelity-PREREGISTERED.md`) needs the held-out set B
reference, which is still being built, so the lane stays opt-in (`GOINFER_CUDA_FLASH_DECODE=16`) and nothing here promotes it.

Instrument: `scripts/bench_splitkv.py` (fresh `serve` per arm, arms adjacent with alternating order, greedy, decode-only client
timing from the first streamed token), new modes `flash` / `flash-gate` / `aa-flash`. goinfer serve built from `4e571d2b` (+ the
floor constant in the last run), RTX 2070 SUPER, driver `595.91.07`, Nobara 44, `nobara-pc`. **The box was made idle by pausing
(SIGSTOP) the CPU reference build for each window; load average < 0.8 at every start** (the harness refuses otherwise; two runs were
refused once and re-run). The reference process was resumed after each window; the pause spans are in
`attn-decode-fa-served-progress.txt`. Raw cells `attn-decode-fa-served-*.json`, logs alongside.

## Forced on at every depth (`GOINFER_CUDA_FLASH_DECODE_MIN_KEYS=0`, S=16) vs the shipped exact path, tok/s

| geometry | 128 | 256 | 512 | 1024 | 2048 | 3900 | 8000 |
|---|---|---|---|---|---|---|---|
| 0.5B (hd64) | 0.911 | 0.931 | 0.928 | 0.954 | 1.176 | **1.468** | — |
| 1.5B (hd128) | 0.989 | 1.026 | 1.108 | 1.123 | 1.278 | **1.557** | — |
| gemma3-1b (hd256, windowed) | 0.957 | 0.949 | 0.945 | 0.973 | 1.058 | 1.002 | — |
| phi3-mini (hd96) | 1.001 | — | 1.000 | — | 1.001 | 1.000 | — |
| D7 qwen2.5-7b (hd128) | — | — | — | — | 1.178 | 1.274 | **1.543** |

phi3-mini is exactly 1.000: hd=96 is not a supported head dim, so the lane declines and both arms are the same code (the registered
"phi3 class declines" expectation, confirmed).

**Registered bands (served, greedy):**
- **1.5B at 3900: 194.1 tok/s (exact 124.7) — band >= 170 ships: MET** (194.1 / 194.2 in two independent runs).
- **D7 at 8000: 61.1 tok/s (exact 39.6) — band >= 50 ships: MET.** (The previous V-sum spike alone read 44.8 there.)
- Decode is close to depth-flat with the lane: 0.5B ~305 tok/s at every depth from 128 to 3900; 1.5B 224 -> 194 from 128 to 3900
  (exact: 227 -> 125); D7 68.8 / 66.3 / 61.1 at 2048 / 3900 / 8000 (exact 58.4 / 52.0 / 39.6).

These clear the bands. They do NOT yet mean "ships": the bands are speed conditions, and R6's decision rule makes the fidelity gate a
precondition.

## The floor: 2048 attended keys (`GOINFER_CUDA_FLASH_DECODE_MIN_KEYS`, default now 2048)

Forced on, the lane loses at shallow depth on the small-KV geometries (0.5B: 0.91-0.95 through 1024; gemma3-1b 0.95-0.97 through 1024)
and wins from 2048; the 1.5B wins from 256. 2048 is the lowest floor with no measured regression on any geometry, so it is the default.
It gates on the attended span (nWin), so gemma's 512-key local layers never take it. **Verified at the shipped default** (`flash-gate`):

| ratio, lane at default floor / exact | 128 | 512 | 1024 | 2048 | 3900 |
|---|---|---|---|---|---|
| 0.5B | 0.983 | 1.001 | 0.994 | 1.152 | 1.476 |
| 1.5B | 1.000 | 1.000 | 1.000 | 1.276 | 1.560 |
| gemma3-1b | 1.014 | 0.993 | 0.998 | 1.086 | 1.016 |
| phi3-mini | 1.000 | 1.001 | 1.001 | 1.001 | 1.001 |

Below 2048 both arms run the exact path, so the shallow cells are an A/A: the 0.5B and gemma3-1b cells scatter 0.983-1.014 with
run-to-run spreads of 3-7 tok/s on those two (their per-arm spread, not the lane, is the noise). **No shallow regression beyond
2%: met** — the largest deviation (0.983) is identical code on both sides.

## A/A floor

Both arms the exact path, same fresh-serve-per-arm procedure: 1.5B 1.003 / 1.000 at 3900 / 8000; D7 1.003 / 1.000. So the D7 and 1.5B
effects above (1.27-1.56) sit two orders of magnitude over the floor.

## What this does NOT establish

- **Fidelity** (registered gate, set B): unscored. Do not quote these numbers as a shipped speedup.
- **A same-session peer comparison.** Registered as the last step (`bench_peer.py`, `BENCH_DEEP_CTX` scoped to the 8000 cells). For
  scale only, Ollama's depth cells recorded earlier for these geometries (not re-run today): 1.5B 175 tok/s at 3900, D7 56.7 at 8000,
  0.5B ~268 flat. The lane would put goinfer ahead on all three, but that is a comparison across sessions.
- Depths between the measured ones, S other than 16, models with attention sinks / GQA > 8 / hd outside {64,128,256} (decline), the
  spec-decode interaction (the lane refuses to coexist with speculative decoding), and Metal/WebGPU (out of scope for R6).

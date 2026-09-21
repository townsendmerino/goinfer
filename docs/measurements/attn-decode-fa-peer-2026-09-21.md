# attn_decode_fa vs Ollama, same session (R6 last registered step)

Three engines interleaved cell by cell with a server restart between cells, one session, `scripts/bench_peer.py` (both over HTTP,
decode-only timing from the first streamed token, greedy, 128 tokens x 2 completions x 2 runs where deep, spread shown):
**goinfer exact** (the shipped path), **goinfer + lane** (`GOINFER_CUDA_FLASH_DECODE=16`, default floor 2048, same binary via an env
wrapper, run as the harness's `goinfer_old` slot), and **Ollama v0.32.5** (`~/ollama-0325`). RTX 2070 SUPER, driver `595.91.07`, Nobara 44,
goinfer serve built from HEAD at 2026-09-21 05:50 (`serve-cuda-fa2`; the lane is the same binary with the env var set), int4 / q4_K_M, same weights on both engines as in the earlier
sweeps. Raw cells `attn-decode-fa-peer-peer-fa-run1.json` / `...-run2-8000.json`, logs `attn-decode-fa-peer-run{1,2}.log`.
The 8000 cells ran with `BENCH_DEEP_CTX=8192` in their own invocation (the deep-context setting is scoped to them only).
**Idle bar:** every cell started under the harness's 1.0 load-average cap and the loadavg recorded per cell is 0.24-0.98; that is inside the
rule, not deeply idle, and a first attempt at the 8000 run was refused at 1.16 and re-run after waiting for < 0.3. There is no repeat
sweep, so cross-session drift (~3.5% on this box) is not bounded. **The lane is opt-in and fidelity-gated on two cells only**
(`attn-decode-fa-fidelity-2026-09-20.md`); read every "lane" figure with that in mind.

## tok/s, and ratio to Ollama

| model | depth | goinfer exact | goinfer + lane | Ollama | exact / Ollama | **lane / Ollama** |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 128 | 334.2 | 334.4 (lane off: below floor) | 268.3 | 1.25 | 1.25 |
| | 2048 | 259.9 | 300.7 | 270.4 | 0.96 | **1.11** |
| | 3900 | 206.8 | 306.3 | 259.5 | 0.80 | **1.18** |
| 1.5B | 128 | 227.6 | 227.8 (off) | 195.1 | 1.17 | 1.17 |
| | 2048 | 161.4 | 206.3 | 180.2 | 0.90 | **1.14** |
| | 3900 | 124.7 | 194.8 | 175.2 | 0.71 | **1.11** |
| | 8000 | 83.8 | 170.1 | 138.3 | 0.61 | **1.23** |
| D7 (qwen2.5-7b) | 128 | 72.9 | 72.9 (off) | 74.2 | 0.98 | 0.98 |
| | 2048 | 58.4 | 68.8 | 71.0 | 0.82 | 0.97 |
| | 3900 | 52.0 | 66.3 | 69.6 | 0.75 | **0.95** |
| | 8000 | 39.7 | 61.1 | 56.5 | 0.70 | **1.08** |

(The 1.5B and D7 128-token cells in the 8000 invocation, run at `-ctx 8192`, read 225.6 / 226.4 / 188.5 and 72.7 / 72.7 / 71.4 — the same picture.)

## What it says

- With the lane, goinfer is **ahead of Ollama on every 0.5B and 1.5B cell (1.11-1.25x) and on D7 at 8000 (1.08x)**, where the exact path
  was 0.61-0.71x at depth. Ollama stays **slightly ahead on D7 at 2048 and 3900 (0.97x / 0.95x)**: the lane closes most of that gap
  (0.82 / 0.75 without it) but not all of it, on the 7B.
- The 1.5B at 3900 clears the registered band at 194.8 tok/s (>= 170), D7 at 8000 at 61.1 (>= 50) — the same figures the paired A/B measured
  (194.1, 61.1), from an independent session.
- Below the floor the lane is off, and the lane's shallow cells equal the exact path's within the usual noise (0.5B 334.2 vs 334.4).

## Not established

The earlier cross-session Ollama figures quoted for scale (1.5B 175 at 3900, D7 56.7 at 8000, 0.5B ~268) reproduce here (175.2, 56.5, 259-270),
so they were fair, but that is one more session, not a bound on drift. No gemma3-1b or phi3-mini peer cell was run (the lane is a no-op on
phi3-mini and only marginal on gemma3-1b at 3900: 1.016). No quality comparison to Ollama's output. Nothing here promotes the lane.

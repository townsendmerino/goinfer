# Mellum2.1 served cold prefill at 16k: before and after the grouped expert kernels

By-day reading, under the timing lock, 2026-10-19 20:32:55 to 20:40:42 PDT, nobara (RTX 2070 SUPER, driver 595.91.07), nothing else on the card (only `kwin_wayland`, 17 MiB; 7,386 MiB free before every cell).
Exploratory: one session, two cells per arm. Script `ttft_mellum.py` (the Gemma 4 E4B reading's script with Mellum2.1's model and flags); raw `ttft-mellum.json`.

Two binaries, a fresh server per cell, order before / after / after / before: `~/serve-cuda-2f375a10` (main with B36 and not the grouped kernels) and `~/serve-cuda-e006afbb` (the same plus them, #105).
Both: `-model mellum2.1-thinking.int4.e4h.cuda.v17.giw -backend cuda -ctx 32768 -moe-cache-experts`. The prompt is 16,308 tokens of filler on `/v1/completions`, one token out, greedy; the clock runs from the request to the first token, after a 400-character warm-up request.
Every cell planned **53 expert slots per layer** (5.8 GB free), the normal 32k plan.

| arm | TTFT | rate |
|---|---|---|
| before (`2f375a10`) | 161.16 s, 162.19 s | 100.8 tok/s |
| after (`e006afbb`) | 44.82 s, 44.94 s | 363.4 tok/s |
| ratio, before / after | | **3.60x** |

Memory: 56.1 GB available before every cell. Load average at cell start: 1.98, 3.44, 3.51, 3.72. The last three are above the harness's 2.0 flag, and they are the 1-minute average carrying the previous cell's own
server (decode threads and the prompt just served); nothing else ran (`other GPU processes: 0` before the first cell, and the lock was held by this run alone).

## The in-process test at 16k, and its 26-slot plan

`TestMoERows_realMellumIdentityAndAB` at 16384 tokens, `-ctx 32768` (two sessions, 17:59:39 and 20:23:26 PDT, the second on a card with only `kwin_wayland` on it): per-pair 2m51.5s / 2m50.7s (95.5 / 96.0 tok/s)
against grouped 54.3 s (301.5 / 301.8 tok/s): 3.16x and 3.14x, residuals of all 512 rows identical, last-row logits identical.
That test plans **26 slots per layer** (3.1 GB free after the build) on an empty card, where the server plans 53 with the same flags. The 2.7 GB difference is in how the test builds its options, not a
process holding the card (none did at 20:23). The option is `Options.ResidentWindowedKV`: serve sets it (`--windowed-kv`, default on, `internal/loadflags`), the library default is off, and the test did not set it, so its 21 sliding-window layers held the whole 32,768-position context: 21 x (32,768 - 1,536) x 4,096 B (f32 K and V, 4 KV heads x 128) = 2.69 GB, the gap (B37; by arithmetic, not yet confirmed by a run). The slower in-process grouped time (54 s against 45 s served) is the smaller cache.

## What it says

At the normal plan the grouped kernels take a 16k cold prefill from about 162 s to about 45 s, 3.6x, against 3.5x at 4k (in process, 55 slots). The ratio is not smaller at 16k when the cache is the normal size.

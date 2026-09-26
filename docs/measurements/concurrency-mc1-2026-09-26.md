# MC1 — multi-slot resident KV on Metal: interleaved conversations hold the 1-client figure (1.22–1.24× at 4 clients) (2026-09-26)

MC1 of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md): N conversations' KV stay resident at
once, so interleaved turns reuse their own prefix instead of evicting each other. It is still one generation at a
time. Shipped in `c2f1532e`; the design is in that commit and in the doc's MC1 section.

**Result: every gate holds, and it ships.**
- With 4 resident KV slots (`-kv-sessions` default 4), the W7 workload's aggregate at 2 and 4 clients is **0.99× /
  1.01×** the 1-client figure. The expected band was 0.90–1.0×.
- The previous build, one slot, reads 0.87× / 0.87×.
- **4-client aggregate: 70.9 against 57.0 tok/s (1.24×), and 70.2 against 57.7 on the reversed-order repeat
  (1.22×).**
- Every client reuses its whole history on every turn. The slowest turn at 4 clients drops from 10.4–11.2 s to
  7.4–8.3 s.

## Setup

- M1 Pro 16 GB, macOS 26.6.2.
- qwen2.5-coder-1.5b-instruct q4_k_m from `~/models`, `-quant int4`.
- `scripts/bench_w7_plain.py`: 6 plain turns per client, 128 tokens per turn, greedy, a fresh server per cell, and a
  nonce on each client's first turn.
- **old** = `serve-metal-lrufix`: the tree just before MC1, with MC0's LRU fix, so the Metal resident has one slot.
- **new** = `serve-metal-mc1`: the MC1 tree. `-kv-sessions` defaults to 4, so the resident allocates 4 KV slots
  (~117 MB each at the 4k resident context).
- Order: old 1/2/4, new 1/2/4, then the deciding 4-client cell again in reversed order (new, old).
- One session, 2026-09-26 16:10–16:30 PDT, idle-gated (load1 1.84 at start).
- Raw: [`concurrency-mc1-2026-09-26/`](concurrency-mc1-2026-09-26/) (`mc1.json`, `run.log`, `run-mc1.sh`).

## Results

| build | clients | aggregate tok/s | ÷ its 1-client | prompt − reused, turns 2–6 (every client) | slowest turn |
|---|---:|---:|---:|---|---:|
| old | 1 | 65.33 | — | 26, 27, 24, 20, 27 | 2.75 s |
| old | 2 | 56.95 | 0.872× | 190, 345, 497, 645, 800 | 5.6 s |
| old | 4 | 57.00 | 0.873× | 187, 342, 494, 642, 797 | 11.2 s |
| old (repeat) | 4 | 57.68 | 0.883× | 187 … 797 | 11.2 s |
| **new** | 1 | 70.34 | — | 26, 27, 24, 20, 27 | 1.96 s |
| **new** | 2 | 69.54 | **0.989×** | 26, 27, 24, 20, 27 | 4.2 s |
| **new** | 4 | 70.85 | **1.007×** | 26, 27, 24, 20, 27 | 7.9 s |
| **new** (repeat) | 4 | 70.19 | 0.998× | 26, 27, 24, 20, 27 | 8.3 s |

"Prompt − reused" is how many tokens each turn prefilled. The reuse counts themselves differ by up to 3 tokens between
runs (168 against 171 at turn 2) because each run's nonce tokenizes to a different length. What each turn prefills —
the new user turn only — is identical in every new cell and in both 1-client cells.

The new build's 1-client figure is 7.7% above the old one's (70.3 against 65.3). That is not MC1's doing: with one
client nothing switches slots. The difference sits in the first turn (1.96 against 2.75 s), a cold-start effect of
server launch, so the MC1 comparison is made within each build (2 or 4 clients ÷ 1) and at 4 clients across builds.

## Gates

| gate (pre-registered, speed bar as amended 2026-09-26) | result |
|---|---|
| per-conversation output bit-identical to the same conversation served alone | **pass.** `TestMetalKVSlots_interleavedMatchesAlone`, through the production `Generate` path: identical ids every turn on the synthetic tiny model and on qwen2.5-coder-1.5b int4 (48 tokens × 3 turns × 2 conversations). The one-slot control thrashes, so the test sees the failure it guards. |
| `prefill_reused_tokens` at 2 and 4 clients equals the 1-client run's, turn for turn | **pass**: prompt − reused is 26, 27, 24, 20, 27 on every client at 1, 2 and 4 clients |
| aggregate at 2 and 4 clients: *expected* 0.90–1.0× of 1 client | 0.989× and 1.007× |
| ships on the hard gates plus a real 4-client improvement over the previous build | **1.24× (70.85 vs 57.00), repeat 1.22× (70.19 vs 57.68)** |
| the fit guard declines a slot count that does not fit, and names the count it chose | implemented (`metalKVSlots` clamps to the memory guard's budget; the banner names a clamp). **Not exercised**: 4 slots of the 1.5B fit this machine, and no clamping load was run. |

## Against W7 and MC0

- W7 (2026-09-19) read 60.08 → 36.14 → 36.39 tok/s at 1 / 2 / 4 clients. This build reads 70.3 → 69.5 → 70.9. W7's
  loss has been removed twice, by faster Metal prefill since 09-19 and by MC1. The cross-session comparison gets no
  more precision than that.
- llama-server's own W7 row (84.8 → 95.9 → 149.7) scales *past* its 1-client figure through batching, and this does
  not; that is MC2's question. MC1 closed the thrash, the part of the gap that was not batching.

## Scope, stated

- **Metal only.** CUDA and WebGPU keep one slot until converted.
- **One slot for recurrent families** (Gated DeltaNet, Mamba-2, LFM2): their state is not part of a KV slot.
- **VL turns keep the bound slot.** They release and re-claim the resident mid-generation, so they write whichever
  slot is bound. That is correct but not multi-slot.
- **An embedding (`HiddenLast`) request writes the bound slot and forgets it**, as before, so it can still cost one
  conversation a cold turn.

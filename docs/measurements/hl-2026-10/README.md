# HL on Metal: the readings (2026-10-14)

The registration is `docs/tasks/task-forward-architecture-2026-10.md` §6.1, committed (`cba440ad`) before any HL code.
This file records what each gate read. The branch is `hl-metal-2026-10`; it is not merged.

Machine: MacBook Pro (M1 Pro, 16 GB), macOS, by day. No gate here is timed.

## HL-G1, HL-G2, HL-G3: identity, PASS

Recorded run at `e9f23357`. The log is `~/goinfer-logs/hl-g1-g3-2026-10-14.log` on the Mac; it also holds the existing
chain and MC3-chain tests, run beside the new ones, all PASS.

| gate | cell | result |
|---|---|---|
| HL-G1 | MC3 fixture, tied head | 80 tokens, 57 with rows, 12 redos: 0 row elements, 0 tokens, 0 K/V elements differ; the token after the chain equal |
| HL-G1 | MC3 fixture, untied head | the same counts, 0 differences |
| HL-G2 | fake chain resident, through `Generate` | 24 tokens: 16 closed steps on `GreedyChainNext`, 8 open steps on rows (the processor's 8 calls), redos at positions 7, 9, 15 and 17, the positions after the four banned picks; tokens equal to the chain-unavailable turn |
| HL-G3 | `llama-attnfa-tiny`, 1 slot | 64 tokens equal; 64 served by the chain, 17 rows (17 processor calls), 5 redos (5 bans) |
| HL-G3 | `llama-attnfa-tiny`, 2 slots (MC3, held) | the same |
| HL-G3 | Qwen2.5-Coder-0.5B q4_k_m (`.gguf`) | the same |
| HL-G3 | Qwen2.5-Coder-1.5B (`.int4.metal.v16.giw`) | the same |

The 1.5B cell was first pointed at the `.gguf`. The load-time fit guard refused it: 2.9 GB was available and the
`.gguf` load needs 2.9 GB, because the file stays mapped during the load. That run is void and was not counted. The
cell then loaded the same model's Metal `.giw`, which the guard admits.

## The gates can go red

Each defect was planted after a commit and removed with `git checkout`. Each made the gate fail:

- **The decoder's redo call removed:** HL-G2 fails, and the tokens part at the first banned pick. HL-G3 fails on the
  fixture and on the 0.5B.
- **The head committed after the run-ahead forward instead of before:** HL-G1 fails. The row then holds the next
  forward's activations.

## HL-G4: speed, not run

`run-hl-g4.sh` (this directory) runs `TestHLChainAB` from a binary pinned at `e757d54c`: the 1.5B and 7B, the closed
and toolcall shapes, 9 reps each, about 25 minutes. It has not been queued, and the registration's ship, park and off
bands decide on its METRIC lines. Its harness was checked once on the tiny fixture, 2 reps per shape. Identity held,
and the toolcall shape ran 49 rows and 6 redos per turn. That check's rates are not a measurement and are not recorded.

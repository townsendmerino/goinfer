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

## HL-G4: speed, graded OFF (the night of 2026-10-14)

`run-hl-g4.sh` ran from the binary pinned at `e757d54c`, on 2026-10-14 from 18:00:36 to 18:08:56 PDT. The run held the
timing lock, with load 0.91 at its start and no thermal warning. The results and provenance are in
`g4-night-2026-10-14/`. Identity held in every rep of every cell: HL's tokens were today's.

| model | (a) closed: the prose part | (b) toolcall: the gate open for the last 48 tokens, 6 redos a turn |
|---|---|---|
| Qwen2.5-Coder-1.5B | **1.0350** (1.0260 to 1.0371; 9 of 9 pairs above 1) | **0.9859** (0.9768 to 0.9879; 0 of 9 above 1) |
| Qwen2.5-7B | **1.0103** (1.0052 to 1.0107; 9 of 9 above 1) | **0.9728** (0.9721 to 0.9745; 0 of 9 above 1) |

**Verdict: OFF.** The registered rule says "Off: (a) < 1.00, or (b) < 1.00", and (b) is under 1.00 on both models
with every pair below 1. On (a) alone the 1.5B would ship (≥ 1.02) and the 7B would park.

The chain wins on the prose part of a tool turn, as C-B01 found for plain turns. The open steps cost more than it
saves there:

- every open step adds a head dispatch and a row readback;
- each redo throws away one queued forward.

At 48 open steps and 6 redos a turn, that is 1.4% of the 1.5B's turn and 2.7% of the 7B's.

**What this leaves:**

- The branch is not merged with HL on by default.
- The bar is not moved: the toolcall shape was fixed before any timed run. Its redo rate (one ban every 8 open
  steps) is the registered shape. It is not a measured rate of real tool calls.
- A re-run needs a named mechanism. Two would change the open-step cost:
  1. run open steps without the chain, as today's path does, and keep the chain only while the gate is closed;
  2. measure how often a real tool call's mask disagrees with the argmax.

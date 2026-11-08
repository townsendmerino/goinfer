# HL-R on Metal: the readings (2026-11-07)

The registration is `docs/measurements/hl-2026-11/registration.md`, committed before any HL-R code. This file records what each gate
read. The branch is `hl-r-2026-11`; the change is `Model.generateInto` letting the greedy chain run under a gated logit processor
while its gate is closed (outside MC3), and stopping it at each opening.

Machine: MacBook Pro (M1 Pro, 16 GB, macOS 26.6.2), go1.27.0 darwin/arm64, by day, 2026-11-07.

## HL-R1 and HL-R2: identity, PASS

- **HL-R1** (`decoder.TestHLR_chainRunsWhileTheGateIsClosedAndStopsAtEachOpening`, a fake chain resident whose logits depend on every
  earlier position, through `Generate`): a gate opening twice, a processor that bans the argmax at every open step. Tokens equal the
  chain-off turn; the chain started once per closed run (3) and stopped at each opening and at the end (3); every open step ran
  `Forward`; the processor never ran with the chain open; no closed step took `ForwardArgmax`. Three planted defects each turn it
  red: the stop removed (the processor then runs with the chain open, and the tokens differ), a gated turn that does not use the
  chain (0 starts, 11 `ForwardArgmax` calls), and an open step that stays on the chain (tokens differ).
- **HL-R2** (`metal.TestHLR_gatedChainMatchesTodaysPath`): Generate on Metal, chain on against `greedyChainOff`, 64 tokens, the gate
  open for generated-token counts 5 to 11 and 30 to 39, the processor banning the row's argmax on every third open step. On the
  committed fixture, the Qwen2.5-Coder 0.5B (`.gguf`) and 1.5B (`.int4.metal.v17.giw`): tokens identical, the processor ran 17 times
  (5 bans) on both arms, and the chain served 47 tokens' forwards, which is the 64 emitted less the 17 open steps. Planting the removed
  stop turns all three cells red (`greedy chain is at position 28, the decode loop at 35`).

## HL-R3: speed

**Superseded reading.** A first run (2026-11-07 20:07:48 to 20:15:27 PST, binary `0c392c2c`) ran without the registered free-memory
check (the runner checked the load only, and recorded no free memory or swap), so it does not count as registered. Its numbers are in
the history of this file (the 1.5B: 1.1134 and 1.0252; the 7B: 1.0110 and 1.0039). The reading below replaces it.

**The counted run.** `run-hlr-r3.sh` ran a test binary pinned at `1eb6ed26` under the exclusive timing lock, 2026-11-07 from 20:47:58 to
21:01:57 PST. At each cell's start the runner checks the 1-minute load (cap 4.0) and `cmd/hostmem -reclaimable` against the model's
file size plus 2 GB, and records both with the swap in use (`r3-2026-11-07/provenance.txt`). Swap in use was 1,404 MB of 2,048 MB
throughout, with no growth; reclaimable memory was 6,230 and 6,200 MiB at the 1.5B cells' starts against a need of 3,203. Identity held
in every rep (10 reps of 256 tokens each, the first discarded as the warm-up), and each rep also asserts that the chain served exactly
the closed steps' forwards. Checkpoints from `~/models`: `qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw`.

| model | (a) closed: the prose part | (b) toolcall: the gate open for the last 48 tokens |
|---|---|---|
| Qwen2.5-Coder-1.5B | **1.1351** (1.0315 to 1.1729; 9 of 9 pairs above 1); today 88.9, HL-R 101.1 tok/s | **1.0519** (1.0026 to 1.0992; 9 of 9); today 92.6, HL-R 97.8 tok/s |
| Qwen2.5-7B | not counted by day: both cells VOID by the guard; see the override cells below | |

**The 7B cells did not run by day.** The 7B's need is 6,596 MiB (its 4,466 MiB file plus 2 GB); reclaimable memory was 6,095 MiB when its
first cell came up and fell to 3,900 to 4,700 MiB (the load average peaked at 8.6) while the machine's other work ran. The runner waited six
60-second attempts per cell and started neither; it exited non-zero, as registered.

**The 7B cells, taken under Francis's override of the free-memory void, 2026-11-08 03:12.** Francis ("Force the Mac runs", relayed by the
coordinator) relaxed the free-memory void from the file size plus 2 GB to the file size; the load cap of 4.0 and the load's own fit guard
stayed. `run-hlr-r3-override.sh` (a copy of the runner with that one change; its provenance carries an `OVERRIDE` line) ran the same pinned
binary `1eb6ed26` under the exclusive lock on 2026-11-08 from 03:14:06 to 03:19:50 PST, at the Mac's idle hour. These two cells are **not registered
cells** and are reported separately: the first (closed) started with reclaimable memory at 5,630 MiB, under the registered 6,596; the
second (toolcall) started at 7,331 MiB, above it, with load 2.44. Swap in use was 913 MB of 2,048 MB, unchanged. Identity held in every rep, and the
chain served exactly the closed steps' forwards in every rep. Records: `r3-7b-override-2026-11-08/`.

| model | (a) closed: the prose part | (b) toolcall: the gate open for the last 48 tokens |
|---|---|---|
| Qwen2.5-7B (override) | **1.0188** (1.0063 to 1.0427; 9 of 9 pairs above 1); today 32.5, HL-R 33.1 tok/s | **1.0226** (0.9747 to 1.0385; 8 of 9); today 32.3, HL-R 33.0 tok/s |

**Against the registered bands** (Ship: median (a) >= 1.02 and (b) >= 1.00; Park: (a) in [1.00, 1.02); Off: (a) < 1.00 or (b) < 1.00):
- **1.5B: Ship.** (a) 1.1351 and (b) 1.0519.
- **7B (under the override): Park.** (a) 1.0188 is in the park band, 0.0012 under the ship line; (b) 1.0226 clears 1.00 on the median, with one pair below 1 (0.9747).

**The registration does not say how the two models combine.** The 2026-10-14 reading of the rows-and-redo form gave one verdict because
(b) was under 1.00 on both. If the 7B parks, the models differ, and read model by model the 1.5B ships and the 7B parks; read as "both must
ship", the result is Park. This record does not choose: a bar is not moved after a reading, and a combination rule added now would be chosen
with the numbers in view. An ambiguous case goes to the owner.

**What changed against the rows-and-redo form (HL, graded OFF on 2026-10-14):** (b) on the 1.5B went from 0.986 to 1.052 and (a) from 1.035 to
1.135. The open steps no longer carry an extra head dispatch and a row readback, and there is no redo; the cost left is one chain stop at
the opening, which waits out the forward the chain had queued past it, and one chain start at the reopening (at the end of a tool turn
there is none). In the toolcall shape the gate opens once and never closes, so the shape has one opening and no reopening; a turn whose
gate closes and reopens many times (a tool call per few tokens) pays that start each time and is not what (b) measures.

## Not shown

- **Real tool-call turns.** (b) is the registered synthetic shape: the gate open for the last 48 of 256 tokens, with the processor
  banning the argmax every 8th open step. With no redo, the ban rate matters little to HL-R, but how long a real gate stays open, and
  how often it closes and reopens, is unmeasured here.
- **MC3, CUDA, sampled turns, an ungated processor:** out of scope (registration).
- **Other sizes and depths.** 1.5B and 7B at the 64-token prompt and 256 generated tokens only.
- **Served end to end.** The cell is in-process `Generate`, as C-B01's and HL-G4's were.

## The decision (2026-11-08 06:50)

**Francis chose to merge HL-R as is** (relayed by the coordinator; PR #765, merge `ac56f39b`). The registration did not say how the two models
combine, and this record had left that to him. His call:

- **Readings he decided on.** The 1.5B, on registered cells (the 2026-11-07 20:47:58 run): (a) 1.1351 and (b) 1.0519, in the Ship band. The
  7B, on cells taken under his own override of the free-memory void (2026-11-08 03:12; the closed cell started below the registered
  free-memory line, the toolcall cell above it): (a) 1.0188 and (b) 1.0226, in the Park band by 0.0012 on (a), faster on all 9 pairs of
  (a) and clearing (b) on the median (8 of 9 pairs).
- **Why, in his words as relayed:** tokens are identical in every rep, there is no memory cost, no median is under 1.00, and the 7B's miss is
  0.0012.
- **What the decision is not.** It is not a moved band: the bands stand as registered, and the 7B's (a) is graded Park above. It is a
  ruling on how the two grades combine, made with both numbers in front of him. It does not cover a turn whose gate opens and closes many
  times, which remains unmeasured (each opening waits out one queued forward and each reopening starts the chain).
- **What follows.** The code stays on for gated greedy turns where the resident offers a chain (Metal today), with a changelog entry
  (`changelog.d/2026-11-08-metal-chain-under-a-gated-processor.md`). CUDA's chain and the sampled chain remain separate steps.


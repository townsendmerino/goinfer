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

`run-hlr-r3.sh` ran a test binary pinned at `0c392c2c` (sha256 prefix in `r3-2026-11-07/provenance.txt`) under the exclusive timing lock,
2026-11-07 from 20:07:48 to 20:15:27 PST. Load at each cell's start was 2.07, 1.06, 0.81 and 0.35, under the 4.0 void line; no thermal
warning; no cell was void. Identity held in every rep of every cell (10 reps of 256 tokens each, the first discarded as the warm-up).
Checkpoints: `qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw` and `qwen2.5-7b-instruct-q4_k_m.int4.metal.v17.giw`, from `~/models`.

| model | (a) closed: the prose part | (b) toolcall: the gate open for the last 48 tokens |
|---|---|---|
| Qwen2.5-Coder-1.5B | **1.1134** (1.0237 to 1.2128; 9 of 9 pairs above 1); today 90.4, HL-R 100.9 tok/s | **1.0252** (1.0173 to 1.0869; 9 of 9) |
| Qwen2.5-7B | **1.0110** (1.0095 to 1.0243; 9 of 9); today 32.3, HL-R 32.7 tok/s | **1.0039** (0.9967 to 1.0325; 8 of 9) |

**Against the registered bands** (Ship: median (a) ≥ 1.02 and (b) ≥ 1.00; Park: (a) in [1.00, 1.02); Off: (a) < 1.00 or (b) < 1.00):
- **1.5B: Ship.** (a) 1.1134 and (b) 1.0252.
- **7B: Park.** (a) 1.0110 is in the park band; (b) 1.0039 clears 1.00 on the median, with one pair below 1.

**The registration does not say how the two models combine.** The 2026-10-14 reading of the rows-and-redo form graded each model
against the bands and gave one verdict because (b) was under 1.00 on both. Here the models differ: read model by model, the 1.5B ships
and the 7B parks; read as "both must ship", the result is Park. This record does not choose: a bar is not moved after a reading, and a
combination rule added now would be chosen with the numbers in view. The ambiguous case goes to the owner.

**What changed against the rows-and-redo form (HL, graded OFF on 2026-10-14):** (b) went from 0.986 and 0.973 to 1.025 and 1.004, and
(a) from 1.035 and 1.010 to 1.113 and 1.011. The open steps no longer carry an extra head dispatch and a row readback, and there is no
redo; the cost left is one chain stop at the opening, which waits out the forward the chain had queued past it, and one chain start at
the reopening (at the end of a tool turn there is none). In the toolcall shape the gate opens once and never closes, so the shape
has one opening and no reopening; a turn whose gate closes and reopens many times (a tool call per few tokens) pays that start each
time and is not what (b) measures.

## Not shown

- **Real tool-call turns.** (b) is the registered synthetic shape: the gate open for the last 48 of 256 tokens, with the processor
  banning the argmax every 8th open step. With no redo, the ban rate matters little to HL-R, but how long a real gate stays open, and
  how often it closes and reopens, is unmeasured here.
- **MC3, CUDA, sampled turns, an ungated processor:** out of scope (registration).
- **Other sizes and depths.** 1.5B and 7B at the 64-token prompt and 256 generated tokens only.
- **Served end to end.** The cell is in-process `Generate`, as C-B01's and HL-G4's were.

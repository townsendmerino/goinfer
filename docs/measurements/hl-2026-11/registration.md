# HL-R on Metal: registration (2026-11-07, before any HL-R code)

**Question.** Does keeping Metal's greedy chain running while a gated logit processor's gate is closed, and stopping it at the
step the gate opens, make the prose part of an agent turn faster without making the whole turn slower?

This is HL-R of `docs/tasks/task-forward-architecture-2026-10.md` (§6.2 on the unmerged branch `hl-metal-2026-10`, filed
2026-10-14; the branch's HL-G4 graded the rows-and-redo form OFF). It is rebuilt on main, not merged: the branch predates the
chain's KV5-6 rework and the move of `metal/` to `backend/metal/`, and HL-R uses none of the rows-and-redo capability.

**Mechanism.** In `Model.generateInto`, a turn with a gated processor (`SamplingParams.LogitProcessorGate`) may use the greedy
chain. A closed step takes the chain, as a turn with no processor does. At a step the gate opens, the loop stops the chain
(`GreedyChainStop` waits out the forward the chain queued past the opening), runs `Forward` for the full row, applies the
processor and samples, as today's path does. The chain reopens at the next closed step. The cost of an opening is one wasted
forward (the one the chain queued for the position `Forward` then rewrites) and one chain start at the reopening.

**Scope.** Metal, one slot (no MC3 batcher), greedy and `top_k = 1` turns, a gated processor. Out of this step: the sampled chain,
an ungated processor, CUDA's chain, and a turn under the MC3 batcher. Each is its own step.

**Gates.** Each must pass before the next is read.

| id | tier | instrument | passes when |
|---|---|---|---|
| **HL-R1** | 0, identity | a fake chain resident in `decoder` whose logits depend on the token at every earlier position, through `Generate` | a gate that opens and closes twice: tokens equal the same turn with the chain unavailable; the chain started once per closed run and stopped exactly at each opening (and at the end); every open step ran `Forward`; no `GreedyChainNext` call for an open step. Red when the stop is removed, and red when the chain is left running through an open step. |
| **HL-R2** | 1, identity | `Generate` on Metal, chain on against `greedyChainOff` (today's path: `ForwardArgmax` while closed, `Forward` while open): the MC3 fixture's single slot, Qwen2.5-Coder 0.5B and 1.5B | the same gate and mask shapes as HL-R1: tokens identical; the chain served every closed step; at least one opening ran, or the cell is void. |
| **HL-R3** | 1, speed, by day | in-process interleaved A/B in HL-G4's shape (`TestHLChainAB`, ported from the branch): 9 reps after a discarded warm-up, Metal 1.5B and 7B, one process per model and shape, under the exclusive timing lock | two quantities, HL-R ÷ today: (a) a gated turn whose gate stays closed (the prose part of a tool turn); (b) a whole turn whose gate opens once for a tool call: 256 tokens, the gate open for the last 48, the processor banning the row's argmax on every 8th open step. **Ship:** median (a) ≥ 1.02 and median (b) ≥ 1.00. **Park:** (a) in [1.00, 1.02). **Off:** (a) < 1.00, or (b) < 1.00. |

**Stopping rule.** One run of 9 reps per model and shape. A rep whose two token streams differ voids the comparison. A cell
started with load above 4.0 or reclaimable memory (`cmd/hostmem -reclaimable`) below the model's file size plus 2 GB is void and re-run in place; the runner checks both at each cell's start, records them with the swap in use, and exits non-zero unless all four cells counted. The do-nothing arm is
today's path in the same process; the arms alternate.

**What each outcome means.** Ship: HL-R goes on by default for gated greedy turns on Metal, with a changelog entry. Park: the
branch's reading stands and HL-R stays off by default, behind no new setting (the code is removed). Off: the code is removed and
the record says so. The bands are the 2026-10-14 ones and are not moved.

**Cost of the run.** About 25 minutes: 4 cells (two models, two shapes), 18 timed arms (9 pairs) and 3 warm-up runs each, plus a load per cell.
Start time and expected finish are stated before it runs.

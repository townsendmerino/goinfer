# HL-C on CUDA: registration (2026-11-08, before any HL-C code)

**Question.** Does a CUDA greedy chain (the next token's forward queued on the device before the host has seen this token) make plain
greedy decode and the prose part of an agent turn faster, without making a turn that opens a gate slower?

HL-C is the CUDA half of HL (`docs/completed/task-forward-architecture-2026-10.md` §6; Metal's half is `registration.md` here and
`README.md`, merged #765). It borrows HL-R's loop and bands; the chain itself is new on CUDA.

**Mechanism.** CUDA today picks the argmax on the device (`backend/cuda/resident.go:cudaResident.ForwardArgmax`) but syncs and uploads
the next token's embedding row every token. The chain implements `decoder/residency.go:ResidentGreedyChain` on `cudaResident`: the
argmax kernel's index feeds a device gather of the next embedding row, the next forward is queued on the same stream, and the host reads 4
bytes per token. `Model.generateInto` already runs a chain under a gated processor while the gate is closed and stops it at each opening
(HL-R), so the loop needs no change. `GreedyChainAvailable` is false where the chain would not be exact: graphs on, a windowed or MLA
model, a Gemma 4 E-model (its row carries per-layer inputs the host computes from the token id), a row that is not a plain table row, LoRA.

**Scope.** nobara (RTX 2070 SUPER, 8 GB), one slot (no MC3), greedy and `top_k = 1`, plain and gated turns, CUDA graphs off (their default).
Out of this step: the sampled chain, MC3, graphs, an ungated processor.

**The gather's table is the open cost.** The device embedding must equal the host row bit for bit. If the checkpoint's embedding is already
on the device in the form the host reads it from, the gather reads it; if not (an untied head is quantized differently), the chain needs a
device copy in the checkpoint's stored dtype, dequantized by the host's own formula. That copy is declared to the memory planner like any
other device buffer, so the fit check and the banner price it.

**Gates.** Each must pass before the next is read.

| id | tier | instrument | passes when |
|---|---|---|---|
| **HL-C1** | 0, identity, seconds | the gather kernel on tiny fixtures of every embedding dtype the resident accepts, against `Embed.Row` | every vocabulary id's gathered row is bit-identical to the host row; the plan's device bytes include the table (the memory-gap gate names the new buffer, no gap). Red when one id's row is read from its neighbour, and red when the table is left out of the plan. |
| **HL-C2** | 0, loop | `decoder.TestHLR_chainRunsWhileTheGateIsClosedAndStopsAtEachOpening` | stays green with CUDA's chain available (no new assertion). |
| **HL-C3** | 1, identity, nobara, by day | `Generate` on CUDA, chain on against the chain off (`ForwardArgmax` while closed, `Forward` while open): Qwen2.5-Coder 0.5B and 1.5B, HL-R2's gate and mask shapes, plus a plain turn | tokens identical on every cell; the chain served every closed step; at least one opening ran, or the cell is void. |
| **HL-C4** | 1, speed, nobara, by day | in-process interleaved A/B in HL-R3's shape: 10 reps, the first discarded, one process per model and shape, under the exclusive timing lock; Qwen2.5-Coder 1.5B and Qwen2.5-7B | two quantities, HL-C ÷ today: (a) a gated turn whose gate stays closed; (b) a whole turn whose gate opens once for a tool call (256 tokens, the gate open for the last 48, the processor banning the row's argmax on every 8th open step). **Ship:** median (a) ≥ 1.02 and median (b) ≥ 1.00, on every counted model. **Park:** (a) in [1.00, 1.02) on any counted model and no (b) under 1.00. **Off:** (a) < 1.00 or (b) < 1.00 on any counted model. |

**Stopping rule.** One run of 9 pairs per model and shape. A rep whose token streams differ voids the comparison. A cell is void by the
heavy-load rule (`docs/measurements/noise-registry.md` §8: it starts only with available memory at or above the plan's need and load
at most 8.0, and is void if swap grows beyond nobara's swap-in bar) and is re-run in place; the runner exits non-zero unless every cell
counted. Each load is fit-checked by the server's own plan before the run; a cell the plan refuses (the 7B plus its table) is dropped and
listed, and the verdict reads from the cells that ran. The models combine as written above: every counted model must ship, so a split is
Park, not a question for the owner. The do-nothing arm is today's path in the same process; the arms alternate.

**What each outcome means.** Ship: the chain is on by default where `GreedyChainAvailable`, with a changelog entry. Park: the code is
removed and the record says so, behind no new setting. Off: the same. The bands are HL-R's and are not moved. If HL-C1 shows the table
cannot be priced or does not fit the 1.5B, HL-C stops there and the record says so.

**Cost of the run.** Two models and two shapes, 9 timed pairs each plus a warm-up, plus a load per cell; the 7B cells are the long ones.
The start time and the expected finish are stated at launch, from a first-cell rate, and repeated in every progress line.

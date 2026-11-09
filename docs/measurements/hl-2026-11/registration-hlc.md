# HL-C on CUDA: registration (2026-11-08, before any HL-C code)

**Question.** Does a CUDA greedy chain (the next token's forward queued on the device before the host has seen this token) make plain
greedy decode (c) and the prose part of an agent turn (a) faster, without making a turn that opens a gate (b) slower?

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

**The gather's table is the open cost, and it must not move any load's fit.** The device embedding must equal the host row bit for bit.
A tied head's gather reads the head's own buffers; an untied head (the bench Qwen2.5 `.gguf`s ship a separate `output.weight`, tied in
name only; read each checkpoint's tensors, not the config's `tie_word_embeddings`) needs a device copy in the stored dtype, dequantized
by the host's own formula. The copy follows Metal's precedent (`backend/metal/greedy_chain.go:resident.chainEmbedTable`): made on the
first chain request, priced by the plan on top of the build's own price, and where it does not fit `GreedyChainAvailable` is false and the
turn takes today's path on the same load. So no load that fits today stops fitting: a build that never asks for a chain allocates and
prices nothing more.

**Gates.** Each must pass before the next is read.

| id | tier | instrument | passes when |
|---|---|---|---|
| **HL-C1** | 0, identity and fit, seconds | the gather kernel on tiny fixtures of every embedding dtype the resident accepts, tied and untied, against `Embed.Row`; the plan with and without a chain request | every vocabulary id's gathered row is bit-identical to the host row, on a tied and an untied fixture; the plan's device bytes with no chain request equal main's for every fixture, and with one include the table (the memory-gap gate names the buffer, no gap); a build with no room for the table loads as today and reports the chain unavailable. Red when one id's row is read from its neighbour, when the table is left out of the plan, and when the table is allocated with no chain request. |
| **HL-C2** | 0, loop | `decoder.TestHLR_chainRunsWhileTheGateIsClosedAndStopsAtEachOpening` | stays green with CUDA's chain available (no new assertion). |
| **HL-C3** | 1, identity, nobara, by day | `Generate` on CUDA, chain on against the chain off (`ForwardArgmax` while closed, `Forward` while open): Qwen2.5-Coder 0.5B and 1.5B (the bench `.gguf`s, untied heads: the run records each file's `output.weight`), HL-R2's gate and mask shapes, plus a plain turn; a tied-head checkpoint if one is on the box, else HL-C1's tied fixture stands for it | tokens identical on every cell; the chain served every closed step; at least one opening ran, or the cell is void; at least one cell has an untied head, and the record lists each cell's head. |
| **HL-C4** | 1, speed, nobara, by day | in-process interleaved A/B in HL-R3's shape: 10 reps, the first discarded, one process per model and shape, under the exclusive timing lock; Qwen2.5-Coder 1.5B and Qwen2.5-7B | three quantities, HL-C ÷ today: (a) a gated turn whose gate stays closed; (b) a whole turn whose gate opens once for a tool call (256 tokens, the gate open for the last 48, the processor banning the row's argmax on every 8th open step); (c) a plain greedy turn, no processor, 256 tokens. Medians per model. **Off:** any median of (a), (b) or (c) under 1.00 on any counted model. **Ship:** not Off, and (a) and (c) at least 1.02 and (b) at least 1.00 on every counted model. **Park:** neither: no median under 1.00, and (a) or (c) in [1.00, 1.02) on some model. The three outcomes are exclusive, and Off wins when models split. |

**Stopping rule.** One run of 9 pairs per model and shape. A rep whose token streams differ voids the comparison. A cell is void by the
heavy-load rule (`docs/measurements/noise-registry.md` §8: it starts only with available memory at or above the plan's need and load
at most 8.0, and is void if swap grows beyond nobara's swap-in bar) and is re-run in place; the runner exits non-zero unless every cell
counted. Each load is fit-checked by the server's own plan before the run, with and without the table, and both plans' free VRAM are recorded
(the fit before and after). A model whose table the plan does not fit is a REPORTED outcome, not a dropped cell: its load must still fit as
today (HL-C1), its cells run with the chain unavailable and are listed as such, and the verdict reads from the models where the chain ran.
If the 7B's table does not fit, that is a finding for the memory planner, stated in the record. The models combine as written: every counted
model must ship, a split between Park and Ship is Park, and any Off is Off. The do-nothing arm is today's path in the same process; the
arms alternate.

**What each outcome means.** Ship: the chain is on by default where `GreedyChainAvailable`, with a changelog entry. Park: the code is
removed and the record says so, behind no new setting. Off: the same. The bands are HL-R's and are not moved. If HL-C1 shows the table
cannot be priced or does not fit the 1.5B, HL-C stops there and the record says so.

**Cost of the run.** Two models and three shapes, 9 timed pairs each plus a warm-up, plus a load per cell; the 7B cells are the long ones.
The start time and the expected finish are stated at launch, from a first-cell rate, and repeated in every progress line.

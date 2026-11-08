# Metal's decode scratch on the Mac's served checkpoints (MM5, 2026-11-07)

<!-- citations-at: a5812274 -->

What Metal's decode scratch costs on the checkpoints served on the Mac, read for Francis's call on whether it is charged in
`NeedBytes` (`docs/tasks/task-subsystems-2026-10.md`, MM5b). Tier 0: the figures are arithmetic on each checkpoint's GGUF
header (`scratch_figures.py`). That is the arithmetic `TestDecodeScratch_handTable` (G-MS3, forward-architecture §4.9c)
holds the build's decode list to on every registered fixture. No model was loaded and nothing was timed. Qwen2.5 3B is from its
published config, since no 3B checkpoint is on the Mac.

**The columns.**
- *decode*: the decode list at one row.
- *batchRows*: the 16 rows the build allocates (`ensureBatchCap(16)`), which grow to the next power of two when a `ForwardBatch`
  asks for more.
- *sampled chain*: what a sampled greedy chain adds. A greedy chain alone holds 48 B.
- *1-slot*: decode, batchRows and the chain, the sampled chain included.
- *MC3 step*: MM5a's list at `batchMaxSeqs` = 8, built only with two or more slots.
- *KV per position*: f16, Metal's default, every layer over the whole context in 64-row blocks.

| model | decode | batchRows | sampled chain | 1-slot | MC3 step | 1-slot + step | KV per position | 1-slot / + step, in positions | 384 MiB, in positions |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder 0.5B | 0.91 MiB | 9.33 | 1.16 | 11.39 MiB | 6.11 | 17.51 MiB | 12 KiB | 972 / 1,494 | 32,768 |
| Qwen2.5-Coder 1.5B | 1.04 | 9.37 | 1.16 | 11.56 | 7.20 | 18.76 | 28 KiB | 423 / 686 | 14,043 |
| Qwen2.5 3B | 1.13 | 9.40 | 1.16 | 11.69 | 7.96 | 19.65 | 36 KiB | 333 / 559 | 10,923 |
| Qwen2.5 7B | 1.44 | 9.50 | 1.16 | 12.10 | 10.43 | 22.53 | 56 KiB | 221 / 412 | 7,022 |
| Phi-3 mini 4k (q4k) | 0.38 | 2.14 | 0.24 | 2.77 | 2.66 | 5.44 | 384 KiB | 7 / 14 | 1,024 |
| Gemma 3 1B | 1.34 | 16.07 | 2.00 | 19.41 | 8.79 | 28.21 | 26 KiB | 765 / 1,111 | 15,124 |

**What dominates.** batchRows' 16 rows of logits (16 x vocabulary x 4 B), not decode's own list. Everything decode-side is 5 to 28
MiB, 1.5% to 7% of `mem.MetalFitMargin` (384 MiB).

**What else rides in that room.**
- **MM7's two other Metal gap rows, both negligible:**
  - the dense vectors: per-layer rope tables and the like, about hd/2 floats a layer, roughly 8 KiB on the 7B;
  - Gemma 4's MoE: 19,016 B on `gemma4-moe-kv-tiny32`, not diagnosed, and no model above is one.
- **The prefill pass's scratch,** which `metalResident.prefillBlockBudget` sizes at up to 512 MiB, against the room under the
  ceiling past the need. On a tight fit it can take all of the margin, by design.
- **What the ledger does not see:** compiled pipelines, command buffers in flight, the driver's own memory, and the Go heap.

The margin's documented purpose (`decoder.MetalFitMarginBytes`) is drift. The load guard reads the live budget again after the
context is chosen, and a context taken at the exact edge would be halved by a few megabytes of drift.

**Francis's call, 2026-11-07 17:26 (relayed by the coordinator):** Metal's decode scratch is charged in `NeedBytes`, and
`MetalFitMargin` stays at 384 MiB, a drift cushion only. The work is MM5b's.

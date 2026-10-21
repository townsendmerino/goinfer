# Gemma 4 E4B cold prefill past the shared-memory bound, before and after #102 (pre-registration, 2026-10-19)

**Registered before the run** (owner, 2026-10-19: "after the merge, by day"). This is the motivating case of the Mac's finding 1: Gemma 4 E4B prompts past 12,160 attended keys declined `attn_batched`'s 48 KiB shared-memory bound and fell to the sequential prefill. #102 (the row-batched exact
attention) is meant to end that.

## Design

- **Arms.** *before* = `~/serve-cuda-333407c3` (main with #95, #101 and #100 and without #102); *after* = `~/serve-cuda-<sha8>` built from main once #102 is merged. Nothing else differs.
- **Cells.** 12,288, 16,384 and 24,576 prompt tokens (by the server's own `count_tokens`), cold: a fresh nonce at the start of the prompt, `/v1/completions`, plain text, one new token, `-ctx 32768`, `~/models/gemma-4-e4b-gguf/gemma-4-E4B_q4_0-it.gguf` (its existing sidecar). One server process per cell (a fresh load each time), so nothing is held between cells.
- **Interleaved.** The two arms alternate cell by cell, and the order flips by size: 12k before then after, 16k after then before, 24k before then after. Six cells, one run each: a drift in the box shows against both arms.
- **Recorded per cell:** the 1-minute load average and `MemAvailable` before the server starts and after the first token, the GPU's free memory before the load and at the first token, the load time to readiness, the time to the first token, the prompt tokens, and the cell's wall clock. The box is in daytime use, so a cell with a load average above 2.0 before it starts is flagged, not hidden.
- **Stop rule.** A cell is stopped at 900 s and recorded as such (the *before* cells are the sequential path and may be long). No retry of a cell to get a better number.

## Pre-registered reading

1. **Resolved:** the *after* time is below the *before* time in all three cells and the gap is more than the load-flagged noise (a flagged cell is reported but not counted for or against).
2. **Not resolved:** any cell where *before* and *after* are within 10% of each other, or a cell stopped at 900 s on one arm only: reported as is, no conclusion beyond the cell.
3. Reported, not graded: tokens per second per cell, and whether the *after* arm's time grows with the prompt faster than the 9B's did (101 tok/s past the bound, `docs/measurements/kv3-tiled-exact-2026-10-13.md` §13.1), which would say the E-model's per-layer inputs, not the attention, are the cost.
4. **Void** if another timed run held the lock, or the server exits before the first token.

## Estimated wall time

About 40 minutes: the three *before* cells are the sequential path (the KV3 record had Gemma 4 E4B at 12,215 tokens unfinished at 150 s before the route; minutes each), the three *after* cells about 1 to 2.5 minutes each (55.9 s at 12,215 tokens on the branch), and six loads of about 20 s. Over the day-run budget of ten minutes, on the owner's explicit request; it holds the timing lock throughout, and the owner can stop it at any cell (`timing_lock.py status` names it).

## Results (appended when read)

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

Run 2026-10-21, 16:05 to 16:20 PDT (15 minutes, against the 40 estimated), by day, under the exclusive timing lock; driver 595.91.07, RTX 2070 SUPER; the log is `~/goinfer-logs/kv3-e4b-cold-ttft-2026-10-21.log` and the records are in `run-2026-10-21/`. *before* is `~/serve-cuda-333407c3`; *after* is `~/serve-cuda-388be34b`, built for this run from a clean worktree at `388be34bb` (main at the #102 merge, aikit as `go.mod` pins it; no local aikit), `-tags cuda`, Go 1.27.0 (the same toolchain as *before*), so the arms differ by the commits between 333407c3 and 388be34bb, #102 among them. One run per cell; the prompt is each size's own count (the target is the character budget: 12,166, 16,214 and 24,298 tokens by the server's `count_tokens`).

| prompt tokens | arm | load at start | time to first token | tokens/s | note |
|---|---|---|---|---|---|
| 12,166 | before | 0.39 | 60.51 s | 201.0 | |
| 12,166 | after | 1.26 | 56.18 s | 216.5 | |
| 16,214 | after | 1.78 | 86.36 s | 187.8 | |
| 16,214 | before | 1.40 | 134.85 s | 120.2 | |
| 24,298 | before | 2.98 | 306.89 s | 79.2 | load flagged (above 2.0) |
| 24,298 | after | 3.19 | 169.69 s | 143.2 | load flagged (above 2.0) |

Every cell started with 54 GB available and 7,382 MiB of VRAM free; no cell stopped at the 900 s cap, none exited early, and the lock was held throughout.

**Reading, by the registered rules: NOT RESOLVED (rule 2), with the direction in every cell the same.**
- 12,166 tokens: *after* is 7.2% below *before* (ratio 1.077), **inside the 10% band**, so by rule 2 it is reported as is and the cell concludes nothing. This is the cell nearest the bound (12,160): the old path was already only 1.08x slower there.
- 16,214 tokens: *after* is 36.0% below (ratio 1.56), unflagged, outside the band: resolved for *after*.
- 24,298 tokens: *after* is 44.7% below (ratio 1.81), but both arms started above load 2.0, so the cell is reported and not counted for or against.
- Rule 1 needs all three cells; one is inside the band and one is flagged, so it is not met. The one unflagged cell outside the band favours *after*.

Reported, not graded (rule 3): *after* runs 216.5, 187.8 and 143.2 tokens/s at 12.2k, 16.2k and 24.3k, falling with the prompt (the last is a flagged cell); *before* runs 201.0, 120.2 and 79.2. The 9B's rate past the bound was 101 tokens/s (`docs/measurements/kv3-tiled-exact-2026-10-13.md` §13.1); E4B's *after* is above it at every size, and its fall from 12.2k to 16.2k (13%) and on to 24.3k (24%) could be the attention's growth with depth or the E-model's per-layer inputs, and one cell per size cannot separate them. The 16k and 24k cells are the ones to repeat, in a quiet window, if the number is to be quoted: this is one run per cell, exploratory by the registration's own design.

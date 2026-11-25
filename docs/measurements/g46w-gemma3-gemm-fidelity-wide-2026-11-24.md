# G46-W: Gemma 3's tensor-core GEMM prefill at depth, ten prompts from distinct sources (registration, 2026-11-24)

Registered 2026-11-24, before any code or run. **Decision it follows (Francis, 2026-11-24 about 17:30 PST, relayed by the coordinator):** G46's re-read
(`docs/measurements/g46-gemma3-gemm-fidelity-2026-11-24.md`) is AMBIGUOUS, and the 10-21 read FAILed at 10,240 tokens on other text, so no remedy
now: the tensor-core GEMM stays on for Gemma 3, and a wider gate reads it before any remedy is decided. It runs by day, not on the night it is
registered.

## Question

Does Gemma-3-4B's default batched prefill (the tensor-core GEMM on above the 512-token floor) stay within the 0.99 bar of the per-token path at every
depth below the shared-memory bound, across prompts from distinct sources, so that a depth at which it fails, if there is one, is a property of
depth and not of two prompts' text?

## Instrument and cells

- `backend/cuda/kv3_tiled_test.go:TestG46_gemma3GEMMFidelityLadder`, extended to take a list of prompt sources and to classify an argmax
  disagreement by the near-tie rule (`decoder/fidelity_testhook.go:NearTieArgmaxForTest`: a hard flip when the gap exceeds 3% of the per-token
  logits' range). The extension leaves G46's own two-prompt run unchanged. It is checked (build, vet, a dry run that prints each prompt's token
  count) before the run.
- **Ten prompts, each the first 12,160 tokens of one source** (Gemma 3's tokenizer, all counted at 12,887 to 107,166 tokens):
  - eight prose documents snapshotted in the tree, so their text does not move with it: from `testdata/prefill-gate-prose-a/`, `QUEUE.md`,
    `audit-2026-09-02.md`, `benchmarks.md`, `ollama-chase.md` and `task-w4a8-neon-bandwidth.md`; from `testdata/prefill-gate-prose-b/`,
    `task-gpu-paths-2026-09.md`, `queue-performance.md` and `task-attention-decode-cost.md`;
  - G46's two code prompts (seeds 4242 and 9001, encoded from the tree's own `decoder/` sources, so their text is the tree's).
- **Depths:** G46's six, 4,096, 6,144, 8,192, 10,240, 12,000 and 12,160 tokens: 60 default cells. One per-token pass per prompt reads all six
  references. Checkpoint `~/models/gemma-3-4b-it-Q4_K_M.gguf` at int4, f32 KV, as G46.
- **Controls:** a levers-off arm at 4,096 and 12,160 on every prompt (20): each must read 0 logits differing and 0 tensor-core launches, or the
  run's attribution of any difference to the GEMM fails. Each default cell must launch the GEMM and no row-kernel group.

## Metric and threshold

Per cell, last-row logits against the per-token path: **PASS** cosine >= 0.992 and no argmax disagreement; **FAIL** cosine < 0.99 or a hard flip;
**PARKED** otherwise (cosine in [0.990, 0.992), or a near-tie flip).

Whole gate: **PASS** if no cell FAILs, all 20 controls hold, and at most 3 of the 60 cells (5%) are PARKED, each listed. **FAIL** if any cell FAILs
or a control does not hold. Otherwise **PARKED**, reported cell by cell.

## Stopping rule

Fixed: all 60 cells and 20 controls, no early stop. A load failure, a NaN or a decline stops the run with that named. The run is VOID if, at its
start, the 1-minute load is above 8.0 or `MemAvailable` is under the load's need plus 2 GB (about 5 GB); it starts only after the load settles
under 4.0 with no other holder of the timing lock, and holds the lock exclusively.

## What each outcome means

- **PASS:** the GEMM stays on for Gemma 3 at every depth read; G46 is closed.
- **FAIL:** the record names the lowest depth with a FAIL cell and how many of the ten prompts fail at each depth. The remedies (a depth cap for
  Gemma 3's GEMM, the GEMM off for Gemma 3, or accept and document) go to Francis with those depths and the speed each costs; the bar is not moved.
- **PARKED:** reported with no conclusion; Francis decides what the parked cells mean.

## Expected nobara time

About 50 minutes under the exclusive lock, by day: per prompt the per-token reference over 12,160 rows (136 s on 2026-11-24), six default
batched prefills (96 s together on 2026-11-24) and two levers-off controls (about 45 s, estimated), so about 4.6 minutes a prompt for ten
prompts, plus the load and the settle.

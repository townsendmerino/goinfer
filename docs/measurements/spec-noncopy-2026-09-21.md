# Speculation with the lane on non-copy traffic, and a lossless fix to the n-gram loop

Three follow-ups to `attn-decode-fa-verify-2026-09-21.md`: (1) the two-model and block-drafter speculative paths with the lane on, (2) what speculation is worth as traffic
stops being copy-heavy, with acceptance recorded, (3) a fix in the n-gram loop that the (2) measurement exposed. `TestSpecNonCopyLane` (`cuda/spec_noncopy_lane_test.go`).

## 1. The other speculative paths (correctness)

`TestFlashDecodeTwoModelSpecLane` (qwen2.5-coder-0.5b drafting for the 1.5B) and `TestFlashDecodeBlockSpecLane` (real Qwen3-4B + its DFlash drafter, via the synchronous
`spec.Generate`, the core the streaming entry shares) pass in both modes: **multi-row verify** (output == plain LANE greedy, `fa_partial_rows` counted inside verify: 532 / 936
launches) and **`GOINFER_CUDA_FLASH_DECODE_VERIFY=0`** (output == plain EXACT greedy, 0 lane launches). `MIN_KEYS=0` forces the lane on the short prompts.
**Mutation check, honestly:** removing the exact-attention scope from `BlockSpec.generate` / `GenerateSpeculative` does NOT fail those tests (the mutants survive). The reason
is structural: neither loop takes an M=1 decode step (both seed and verify in batches), so under option A their verify rows are exact whether or not the scope is held. The scope is
load-bearing only for the n-gram loop (its mutant is killed by `TestFlashDecodeSpeculativeScope`); for the other two it is defense in depth, and the comments now say so.

## 2. Non-copy prompts (1.5B, ~4900-token document + an instruction, greedy, 160 new tokens, one process, arms interleaved 3 rounds, median, decode time = time(160) - time(1))

Arms: P0 plain exact, PL plain lane, S0 exact + adaptive n-gram spec (what `serve --spec ngram` runs), SL lane + spec (multi-row verify). Acceptance is the engine's own
(`SpecStats`). **A first "fresh" prompt ("ignore the text above...") was wrong:** the 1.5B ignored the instruction and kept copying the document (identical acceptance to COPY), so
the run prints each kind's output head and the FRESH kinds put a real question BEFORE the document.

Before the fix in section 3 (tok/s):

| prompt | accept. | P0 | PL | S0 | SL | S0/P0 | SL/PL |
|---|---|---:|---:|---:|---:|---:|---:|
| COPY (rewrite exactly) | 8.42 tok/round, 99% | 114.2 | 196.6 | 233.6 | 370.7 | 2.05 | 1.88 |
| SUMMARIZE | 3.41 tok/round, 55% | 114.7 | 194.8 | 131.4 | 210.5 | 1.15 | 1.08 |
| FRESH essay (photosynthesis) | 1.04 tok/round, 12% | 114.3 | 197.0 | **74.5** | **140.1** | **0.65** | **0.71** |
| FRESH story (lighthouse) | 1.03 tok/round, 10% | 113.8 | 197.2 | **75.2** | **142.9** | **0.66** | **0.72** |

So on genuinely novel text `--spec ngram` was a **29-35% loss** against plain decoding, on the exact tree as well as with the lane. The adaptive controller was doing its job (Theta 0.251
on CUDA; alpha 0.1 gives depth 0): the loss was not over-drafting.

## 3. The cause and the fix

Every round verified through `resident.ForwardN`, which returns the FULL logits row per verified row (608 KB device-to-host each at a 152k vocab) and takes the argmax on the host.
A no-draft round is a plain decode step, but it paid the batched path plus that readback instead of the device-argmax fast path plain decode uses (`ResidentGreedy`, a 4-byte readback). The
block-drafter loop already avoids this; the n-gram loop never did. The drafter's own scan was measured too and is not the cause (88 us per call at 4900 tokens, ~1%).

`genNgramInto` now, for a greedy, resident, untraced run, verifies by argmax ids only: one row via `ForwardArgmax`, several via `PrefillLastNArgmax` (the verify primitive block
speculation uses). Any error falls back to the full-logits path once and permanently, so a resident whose batched pass declines keeps working. Sampled and traced runs are untouched.
The ids equal the logits path's argmax, so output is unchanged (asserted per arm in the test: S0 == P0, SL == PL, token for token, every kind). It also skips the drafter's context
scan when the depth controller is already at 0 (worth ~1%).

After:

| prompt | P0 | PL | S0 | SL | S0/P0 | SL/PL | SL/S0 | S0 was -> now | SL was -> now |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| COPY | 114.1 | 196.6 | 263.1 | 444.9 | 2.31 | **2.26** | 1.69 | 233.6 -> 263.1 (+13%) | 370.7 -> 444.9 (+20%) |
| SUMMARIZE | 114.7 | 192.4 | 150.9 | 249.0 | 1.32 | **1.29** | 1.65 | 131.4 -> 150.9 (+15%) | 210.5 -> 249.0 (+18%) |
| FRESH essay | 114.3 | 196.0 | 101.5 | 176.1 | 0.89 | **0.90** | 1.73 | 74.5 -> 101.5 (+36%) | 140.1 -> 176.1 (+26%) |
| FRESH story | 113.9 | 196.7 | 105.8 | 181.1 | 0.93 | **0.92** | 1.71 | 75.2 -> 105.8 (+41%) | 142.9 -> 181.1 (+27%) |

## Reading it

- **Lane + spec beats exact + spec by 1.65-1.73x on every kind** (SL/S0), copy or not: the lane's verify rows are what pay.
- Against the best plain path (PL, the fair comparison for a lane server), speculation is worth **2.26x on copy-heavy, 1.29x on partial reuse, and still a ~8-10% loss on fresh
  text.** The loss on fresh text shrank from 28-35% to 8-11% but is not gone. What remains is not measured here (the periodic D=1 probe rounds, per-round host bookkeeping such as
  the history copy and the drafter call, and the plain-step path inside the spec loop are candidates, none isolated).
- Practical reading: `--spec ngram` should be enabled for copy-heavy traffic (edits, RAG, agent loops), and is now safe-ish but still slightly negative on free-form chat.
- These are single prompts of each kind on one 1.5B at ~4.9k context; acceptance varies by prompt. Not measured: other models or contexts for the after-fix numbers, the served HTTP
  path (this is in-process), the two-model and block-drafter paths' speed.

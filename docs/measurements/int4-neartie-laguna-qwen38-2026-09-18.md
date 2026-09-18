# Laguna and Qwen3.8 real-checkpoint oracle divergences: real, but fixed by int8 — not "accept forever"

**RETRACTED, SAME DAY. The original verdict below ("accept as permanently non-blocking") was
WRONG.** This doc's first version argued `TestLagunaReal_oracle` and `TestQwen38Real_oracle`'s
divergence from HF was inherent int4 precision loss at a near-tie, unfixable, and should go into
`cmd/gate/parity.go`'s `neverConfirmed` map forever. Independent review (Gemini, asked to try to
break the conclusion, not confirm it) checked the one premise the whole argument rested on — "int8
does not fit alongside f32 activations in 62 GB of RAM" — and found it had never actually been
run. It hadn't. Measured the same day: **both gates pass cleanly at int8**, well within this box's
RAM, with the divergence completely gone. Both test files now load int8 instead of int4
(`decoder/laguna_real_test.go`, `decoder/qwen3_5_realckpt_test.go`) and both gates PASS. Neither
entry went into `neverConfirmed`. This is the corrected account, kept — not deleted — because how
a plausible-sounding wrong conclusion got caught is as much the record as the numbers are.

## What was real, and stayed real

The int4 divergence itself was not a false reading — everything measured about it holds:

- Both gates' int4 forward diverges from the real HF bf16 reference a few greedy-continuation
  tokens in, at a position where goinfer's own logits have the HF-reference token as a close
  runner-up (Laguna: rank 2, 0.97 logits back; Qwen3.8: rank 1 — the immediate runner-up — 0.32
  logits back), not a wildly wrong answer.
- Floating-point-rounding-sized noise alone (`GOINFER_NORM_ULP_NOISE=1`, `decoder/normnoise.go`,
  no code change) reproduces the identical wrong token in both cases, which is real evidence the
  int4 divergence is a genuine near-tie sensitive to exactly this magnitude of perturbation.
- This is a real extension of this repo's own prior "MoE router-flip noise floor" finding
  (previously characterized only for MoE routing at int8) to a second mechanism: a DENSE,
  non-MoE family (Qwen3.8) showing the identical signature at int4, which says the sensitivity is
  about int4's coarser grid in general, not specifically about expert routing.

All of that is still true and still worth knowing. **What was wrong was the conclusion drawn from
it**: that because int4 has this real sensitivity and int8 supposedly didn't fit, the only honest
option was to accept the int4 failures forever. The "didn't fit" half of that was never checked.

## The retraction, in one table

| | claimed (first version, same day) | measured (same day, after review) |
|---|---|---|
| Laguna int8 RAM peak | "does not fit in 62 GB" (asserted, not measured) | **43 GB peak RSS**, 62 GB total, comfortable headroom |
| Laguna int8 gate result | not run | **PASS** — cosine 0.998845, all 8 continuation tokens exact |
| Laguna int8 wall time | not run | 485s (~8 min) |
| Qwen3.8 int8 RAM peak | "does not fit in 62 GB" (asserted, not measured) | **27 GB peak RSS** |
| Qwen3.8 int8 gate result | not run | **PASS** — cosine 0.999886, all 8 continuation tokens exact |
| Qwen3.8 int8 wall time | not run | 89s |
| Policy | move both to `neverConfirmed`, permanent | **fix the gates to int8**, both promoted as genuine PASS |

Both int4 forwards' recorded evidence above (margins, noise control) is unchanged by this
retraction — that measurement was real and reproducible. What changed is what to DO about it: the
gates were asking their oracle check to run at a precision with a known, real, and (as it turned
out) entirely avoidable sensitivity, when a precision that avoids it was available the whole time.

## Why the "doesn't fit" premise was wrong, and how it got made without being checked

`decoder/laguna_real_test.go` and `decoder/qwen3_5_realckpt_test.go`'s oracle gates carried a
comment asserting int8 didn't fit, dated to when the gates were first written (2026-09-08) and
apparently carried forward by inference from int4's OWN comment ("int4 weights: 33B bf16 is ~63GB
on disk and would not fit alongside f32 activations in 62GB of RAM") — which is a true statement
about *unquantized bf16*, not int8, and does not actually say anything about int8's footprint. The
oracle gates' own comments then asserted int8 "does not fit... either", extending a true claim
about bf16 into an unchecked claim about int8, and citing each other ("the same capacity-forced
choice qwen3next's own real gate makes") as if mutual citation were evidence. It read as
well-supported specifically because it was stated three times in three places, all originating
from the same never-run assumption.

**The actual numbers, worked from what int4 already measures**: Laguna's int4 weights load at
19.5 GB RSS (this session's own earlier measurement, before this file was corrected). int8 is
roughly double that per-weight, so ~39 GB of weights was always the right order-of-magnitude
estimate — squarely inside 62 GB alongside a KV cache the fit-planner already sizes down to fit
(6.6–6.7 GB here, well below int4's 18.7 GB, because int8's larger weight footprint makes the
planner shrink context first). The "doesn't fit" belief was not merely unmeasured; the numbers
needed to sanity-check it by arithmetic alone were already sitting in this same file's own
earlier measurements, one section up.

## What actually happened at int8 (measured 2026-09-18, this box, real checkpoints)

Both runs via the officially edited gates (`Options{Quant: "int8"}`), not a standalone diagnostic:

**Laguna-XS.2 (33B-A3B)**: `GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run
TestLagunaReal_oracle -v -timeout 60m`. Fit planner: weights 31.6 GB + KV 6.7 GB (context capped
to 85949 of the model's 262144 to keep KV inside budget — an int8-specific tradeoff, not a
problem: the oracle gate only ever exercises `len(prompt)+8` positions). Peak measured RSS during
the run: 43 GB. Total wall time: 485s. Result: argmax exact (22345), last-logit cosine
**0.998845**, full continuation `[22345 83 268 785 9626 377 14220 395]` — **exact match** to the
golden, including position 3 (785), the exact token int4 got wrong (110).

**Qwen3.8 (27.8B dense)**: same command shape, `-run TestQwen38Real_oracle`. Fit planner: weights
26.3 GB + KV 12.1 GB (context capped to 99143). Peak measured RSS: 27 GB. Total wall time: 89s.
Result: argmax exact (11751), cosine **0.999886**, full continuation
`[11751 13 198 760 6511 314 9564 369]` — exact match, including position 2 (198), the newline
token int4 replaced with a repeated "Paris".

## What this changes going forward

- `decoder/laguna_real_test.go`'s `TestLagunaReal_oracle` and
  `decoder/qwen3_5_realckpt_test.go`'s `TestQwen38Real_oracle` now load `Options{Quant: "int8"}`,
  not `"int4"`. Their doc comments are corrected in place — not silently, with the wrong claim
  quoted and marked wrong, per this repo's own retraction discipline of leaving the trail visible
  rather than just fixing the number.
- Neither gate needed `cmd/gate/parity.go`'s `neverConfirmed` map. Both were promoted to the
  ledger as genuine, real PASS results
  (`scripts/gate_ledger.py promote --gate TestLagunaReal_oracle --value PASS --by francis` /
  same for `TestQwen38Real_oracle`), the same as `TestQwen25VLReal_gate` and
  `TestQwen2MoeReal_oracle` earlier the same day.
- The coherence-only gates above each oracle (`TestLagunaReal_gate`, `TestQwen38Real_gate`) are
  UNCHANGED, still int4 — their own comments make the narrower, still-true claim ("bf16 does not
  fit", not "int8 does not fit"), and they don't compare logits against an independent reference,
  so int4's near-tie sensitivity was never a correctness risk for what those two specifically
  check (loader/wiring correctness, not numeric parity).
- `oracleCosFloor`'s int4 bar (0.98) and int8 bar (0.99) are both unchanged. Nothing here argues
  either floor is miscalibrated — the point was never that 0.98 was too strict, it was that these
  two gates didn't need to run at the precision the floor was calibrated defensively for.

## The actual lesson

Not "int4 is broken" (it isn't — the near-tie sensitivity measured here is real and expected of a
coarser quantization grid, and nothing here suggests goinfer's int4 path is wrong for serving).
The lesson is narrower and more useful: **a capacity/infrastructure claim ("X doesn't fit in Y")
is exactly as much a claim that needs verifying as a claim about code correctness — and it's
easier to skip verifying precisely because it looks like a fact about the environment rather than
a decision anyone made.** The first version of this doc applied real rigor (margin measurement,
a noise-injection control, cross-checking against a second family) to the CODE question and none
at all to the CAPACITY question the whole policy conclusion depended on — and the capacity
question was the one that was actually wrong. Asking "did anyone run this, or are we all citing
each other" is now worth doing explicitly whenever a "this won't fit" claim is about to become a
`neverConfirmed` decision rather than a measurement.

## Reproduce this

```sh
GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestLagunaReal_oracle -v -timeout 60m
GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestQwen38Real_oracle -v -timeout 60m
```

Both should now PASS directly — no env var, no code change needed; the fix is in the tracked test
files themselves. The int4 margin/noise-control numbers in the "what was real" section above came
from a throwaway diagnostic (not committed) built the same way
`docs/tasks/task-webgpu-nogqa-decode-bug.md`'s WebGPU investigation did the same day: load the
checkpoint, run the prompt + continuation, and at the diverging position sort the full logit
vector to read off where the golden's expected token ranks.

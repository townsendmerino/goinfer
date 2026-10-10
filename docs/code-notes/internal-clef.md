# internal/clef: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/clef`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## pipelineProbBar

Moved from `internal/clef/pipeline_test.go` (the comment above `pipelineProbBar`) on 2026-10-09.

```text
The WHOLE pipeline against the reference: request -> Go encoder -> Go backbone (decoder.PromptHiddenAll on the committed tiny qwen3_5, hidden 64, UNTIED
lm_head, random final-norm weight) -> Go head (width 32, seeded random weights in every parameter) -> per-option probabilities, compared with the official
joint_schema_model.py chained as ClefModel.forward chains it, at f32 (scripts/pin_clef_e2e_tiny.py). It exists because the encoder, the backbone seam and the
head are each gated in isolation, and nothing else runs them together: where the lm_head rows come from, whether the head sees the post-final-norm hidden state,
and whether the encoder's spans index the rows the backbone returns are only visible end to end.

Everything it reads is committed, so a missing file FAILS (a skip would hide the absence). PROBABILITY BAR, written before the first run: 1e-4 absolute against the
reference. The head alone matched its reference to 1.2e-7 given identical inputs; the backbone adds its own f32 differences (D11's bar is relative L2 1e-5 on the
hidden state), which the head's attention and softmax pass through, so the end-to-end bar is looser than the head's 1e-5 by design and the measured value is logged.
```

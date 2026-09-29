# D7 — decisions against constrained generation (speed): the projection, before measuring — 2026-09-28

Item D7 of [`task-constrained-confidence.md`](../tasks/task-constrained-confidence.md). **This header is the projection
the task requires before anything is measured.** Nothing here was timed for D7; every input is a figure already on
record, named where it is used. The measurement, when it runs, fills §4 and is compared with the band in §3. D7 is not a
kill gate. The calibrated distribution is the product whether or not it is faster.

## 1. The model

- **A decision (Route A, `internal/decide`)** is one `Generate` of one token: the prompt's prefill, then a restricted
  softmax over the option tokens at the last position (`internal/decide/model.go`, `ModelPrefill`). The readout is a
  handful of logits, so t_head ≈ 0 and **t_decision ≈ t_prefill(K)**.
- **Constrained generation** is plain `Generate` with the grammar as the logit processor. Every output token costs a
  decode step, the forced JSON structure included: only the `--spec` path drafts forced runs
  (`internal/serveapp/openai.go`, the `lm.spec && gr.masker != nil` case). The first output token comes from the
  prefill's logits, so **t_constrained ≈ t_prefill(K) + n_out · t_decode**, with n_out counted up to but not
  including EOS.

## 2. The inputs

**Model:** Qwen3.5-9B Q4_K_M, the model D6a grades, because D7's "same model" is that one.

**The machines:**
- **nobara CUDA (RTX 2070 SUPER): the only machine with figures on record.** They are from
  [`task-cuda-deltanet-prefill-2026-09.md`](../tasks/task-cuda-deltanet-prefill-2026-09.md), gate 3, 2026-09-28:
  - batched prefill 344–348 tok/s at 145 / 561 / 621 prompt tokens;
  - its profile at 621: GEMVs 1.654 s, attention 58 ms, recurrence 38 ms, glue 25 ms;
  - the per-token path, 9.270 s for 561 tokens. That is 16.5 ms per token, taken here as t_decode.
  - `docs/benchmarks.md` has no 9B-class row. Its largest dense model is a 7B, which is not this family.
- **The Mac: not projected.** The 9B does not run resident. Metal's memory guard allows 5.4–5.8 GB against the
  6.04 GB the resident needs, and the CPU fallback took 10–79 s a row (the D6a pre-registration in the task doc).
- **nobara CPU: no 9B figure on record.** D7 measures it or states it out of scope.

**Extrapolation:** t_prefill at K = 1024 and 4096 is extrapolated past the measured 621 tokens, as a band:
- **linear:** 345 tok/s flat;
- **quadratic attention:** the 621-token profile, with the GEMVs, recurrence and glue scaling as K and the 58 ms of
  attention as K². This model reads 13.8 s at 4096, against the linear 11.9 s.

**n_out, counted with the model's own tokenizer** (`tokenizer.LoadGGUF` on the 9B's `.gguf`):

| answer | tokens |
|---|---:|
| schema, one field: `{"answer":false}`, `{"answer":0}`, `{"answer":"A"}` (every kind, every label) | 5 |
| a Hermes tool call, `<tool_call>\n{…"answer":"A"}}\n</tool_call>` (the Qwen `<tool_call>` dialect, `chat/tools.go`) | 16 compact, 20 with default JSON spacing |
| five fields in one object, e.g. `{"refund":false,"urgency":3,"category":"B","escalate":true,"channel":"D"}` | 23–26 |

A tool call also puts the tool's definition into the prompt. Measured with `chat.Template.RenderTools` for one
`decide` tool, that is **128–143 prompt tokens** (boolean / six-level integer / four-letter enum). The projection uses
135.

## 3. The projection (nobara CUDA, Qwen3.5-9B, batch 1)

**One question.** A ratio is constrained time ÷ decision time, so > 1 means the decision is faster.

| K | t_decision (s) | schema answer (s) | ratio | tool call (s) | ratio |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.72–0.74 | 0.80–0.83 | **1.11** | 1.37–1.46 | **1.88–2.00** |
| 1024 | 2.97–2.99 | 3.05–3.07 | **1.03** | 3.62–3.74 | **1.22–1.25** |
| 4096 | 11.9–13.8 | 12.0–13.9 | **1.01** | 12.5–14.7 | **1.06** |

**Five questions about one state** (TypeSafe's request shape):
- Five decisions are five prefills of K. On qwen3_5 each question pays a full prefill today (task doc §3 and D8).
- One constrained pass answers all five fields: one prefill, then 23–26 tokens of decode.

| K | five decisions (s) | one constrained pass (s) | decisions slower by |
|---:|---:|---:|---:|
| 256 | 3.71 | 1.09–1.17 | **3.2–3.4×** |
| 1024 | 14.84 | 3.32–3.40 | **4.4–4.5×** |
| 4096 | 59.4 | 12.2–12.3 | **4.8–4.9×** |

**What the band says:**
- **One question:** a decision beats a schema-constrained answer by 11% at a short prompt, and by almost nothing at
  4K. Prefill dominates both, and the constrained answer is only five decode steps. Against a tool call, the
  decision's edge is the call's wrapper plus its ~135 definition tokens: ~2× at 256, 6% at 4K.
- **Many questions about one state:** decisions are 3–5× *slower* than one constrained pass, and repeated prefill is
  ~80% of the request. **D8's trigger (state prefill ≥ 70% of a multi-question request) is projected to fire.** The
  trigger stands on the measurement, not on this projection.
- **This is qwen3_5-specific.**
  - A dense family keeps its prefix across questions when the template puts the question after the state (the
    resident prefix reuse `ModelPrefill` inherits). There, N questions cost roughly one prefill plus N short suffixes.
  - A hybrid DeltaNet family cannot reuse a prefix until the state-checkpoint work D8 names exists.
- **At saturating concurrency:** MC3's batched decode lowers constrained generation's per-request decode cost, and
  both routes stay prefill-bound. The single-question ratios should move toward 1. The multi-question gap depends on
  prefill throughput under concurrency, which has no figure on record for this family.

## 4. Measurement

*(Not run. It goes on the night queue, and its pre-registration is written here first: same model, same prompts,
quiet box, paired differences, decisions/s at batch 1 and at saturating concurrency. Each measured ratio is set
beside §3's band.)*

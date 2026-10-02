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

## 4. Measurement: pre-registration (2026-09-28, before any harness or run exists)

### 4.1 Setup

- **Machine:** nobara CUDA, on the night queue, with the harness defaults (`BENCH_MAX_LOADAVG=1.0`, idle-gated per
  request block) and the TE9 timing lock.
- **Model:** Qwen3.5-9B Q4_K_M from `~/models`, served by one pinned `goinfer-serve` binary, named with its commit in
  the record.
  - **Context:** `--ctx 8192`, so the 4096-token state plus a tool definition fits. The run must show the resident
    decode path.
  - If the resident declines at 8192, the harness retries at `--ctx 5120`, records the decline and the context used,
    and goes on. A CPU fallback voids the run.
- **Concurrency:** batch 1 only. `docs/server.md` gives the Gated-DeltaNet families one slot, so "saturating
  concurrency" is a queue for qwen3_5 and measures nothing new. Stated here, not measured.

### 4.2 Prompts

- **Setup:** 8 distinct states at each K ∈ {256, 1024, 4096}, calibrated to K state tokens with the model's own
  tokenizer, as `bench_prompts_calibrate.py` does for `bench_peer.py`. They are support-ticket-like prose, and every
  state is different, so no prefix is shared.
- **One question per state**, rotating over the kinds: noul, a six-level score, and a four-option choice.
- **The five-question set** is those three plus a second noul and a second choice, all about the same state.
- **Built and frozen with the harness:** the prompt file's sha256 is recorded here before the run.

### 4.3 The arms, all over HTTP, timed client-side from request sent to response complete

| arm | request | what it answers |
|---|---|---|
| **decision** | `POST /v1/systemone`, the state and one question | one question |
| **schema** | `POST /v1/chat/completions`: the state plus the question's instructions as the user turn, `response_format` a one-field `json_schema` (the question's enum, boolean or integer), greedy | one question |
| **tool** | the same chat request with one `decide` tool whose `answer` parameter carries the same enum, `tool_choice: "required"`, greedy | one question |
| **decision ×5** | `POST /v1/systemone`, the state and all five questions | five questions |
| **schema ×5** | one chat request whose `json_schema` has all five fields | five questions |

- **Recorded per request:** wall time, `usage.input_tokens` and `usage.output_tokens`, and the arm's actual prompt
  length.
- **Order:** the three single-question arms rotate through a 3 × 3 Latin square over consecutive states. The two
  five-question arms alternate AB / BA by state.
- **Warm-up:** 2 requests per arm, discarded, after the server loads.

### 4.4 What is computed

- **Per state, the paired ratio** t_arm ÷ t_decision (single-question) or t_decision×5 ÷ t_schema×5. The point
  estimate is the geometric mean over the 8 states, with a t-interval on the log ratios (df 7).
- **Against §3's band.**
  - A ratio whose 95% interval overlaps its projected band, widened by ±10%, is **as projected**.
  - Otherwise it is **off projection**. The record then names which input was wrong, by recomputing t_prefill and
    t_decode from the measured requests (usage tokens ÷ time), before the number is quoted anywhere.
- **D8's trigger:** the state-prefill share of a five-question decision request,
  **share = 1 − t_decision×5(K = 32) ÷ t_decision×5(K)**. This is what a longer state adds, since the 32-token cell has
  almost none. It needs a K = 32 cell, 8 states, run the same way. **D8 fires if the share is ≥ 0.70 at K = 1024.**
  1024 is the "realistic workload" D8 asked for: a support ticket plus a little history.

### 4.5 What is quoted

- The docs and site quote the measured ratios, with the machine, date and interval. They never quote §3's
  projection.
- The single-question ratio goes into `docs/server.md` only as "about the same as" or "N× faster than" constrained
  generation. The five-question ratio goes there as the cost warning it already carries, now with its number.

### 4.6 Cost

**About 35 minutes, one night job** (from §3's projections, plus about 25% for warm-up, load and gating):
- single-question arms: about 7 minutes;
- five-question arms: about 12.5 minutes;
- the K = 32 cell: about 1 minute.

**Harness built, smoke-tested and queued, 2026-10-02** (`decisions-d7-2026-09-28/`: `d7_bench.py`, `run-d7.sh`, `prompts.json`). Frozen before any registered run:
- **Prompt file** `prompts.json`, sha256 `2d56d87f800e2b7235bbeb9e1148039daee1e3d22c53a503ea26711ba0910dd9`: 8 states at each of K = 256, 1024, 4096 and 32, every one calibrated by goinfer's own `usage.input_tokens` (state tokens = the decision request's input tokens minus a one-character state's, plus 1) to within 2 percent of K (at least 3 tokens). Reached: K=256 251 to 261, K=1024 1006 to 1030, K=4096 4052 to 4109, K=32 31 to 34. Support-ticket-like prose; all 32 states differ from their 60th character on, **and share only the 7-character opener "Ticket "** (about two tokens), which the pre-registration's "no prefix is shared" did not anticipate; a hybrid family reuses no prefix, so it does not matter to the measurement, but it is not literally none.
- **Binary** `serve-cuda-644d8008` (goinfer `644d8008`, sha256 `365425d917281c6e90c7723891ad28276611bd7e0f84366dafc2d9461b014ee9`), **harness** `d7_bench.py` sha256 `643762c48b336f933d491f554b352ac6d1e24e03408ddf83c90c62c6577a0ca4`, both copied under `~/goinfer-bench/decisions-d7/` and sha256-checked by the night script.
- **Requests, as the pre-registration lists them**, with the details it left open: the questions are one noul, one six-level score (levels sent as an ordered array: an object is a 422), one four-option choice, and a second noul and choice for the five-question set; the schema arm is `response_format` `json_schema` (strict, one `answer` field for a single question, five named fields for five); the tool arm is one `decide` tool, `tool_choice: "required"`;
  all greedy, `max_tokens` 88 / 64 / 248 for single schema / tool / five-field schema. The single-question order is a 3 x 3 Latin square over consecutive states, the five-question order alternates AB / BA by state, 2 discarded warm-up requests per arm, an idle gate (`BENCH_MAX_LOADAVG` 1.0) before each state block, the resident decode path checked from the server's own banner (ctx 8192, then 5120; a CPU fallback voids the run).
  An invalid answer (no readable answer in the reply) is recorded with `ok: false` and excluded from its pair, never silently dropped; the analysis prints how many.
- **Analysis** as registered: per-state paired ratio, geometric mean over the 8 states with a t-interval (df 7) on the log ratios, against the section 3 band widened by 10 percent ("as projected" if the interval overlaps it, otherwise "off projection"), and D8's trigger share at K=1024.
- **Smoke, EXPLORATORY and not quotable** (2 states at K=256, 1 pair each, run by day while the box was otherwise idle): all five arms answered validly; it found one harness bug (the score question's `criteria` must be an ordered array, not an object; fixed before freezing). It also hints that requests are slower than section 3 projected and that the schema x5 reply writes about 49 tokens, not the projected 23 to 26; that is what the registered run measures, and it is used here only to revise the estimate.
- **Estimate revised to 60 minutes** (from 35): the projection-based figure plus the smoke's observation that K=256 requests ran about 1.4 times the projected time. Queued as `d7-decisions-speed` on nobara's night queue; the owner starts the queue.

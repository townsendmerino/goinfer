# Task: confidence — per-field probabilities on constrained output, and a typed `/v1/decisions` endpoint (C0–C2, D0–D9) — 2026-09

> **Status, 2026-09-30: C0–C2, D0–D5 done; D6a GRADED → BUILD D2–D4 (built); D6b queued on nobara for tonight's run; D7 projected.**
> - **D6a** ([`decisions-d6a-2026-09-28.md`](../measurements/decisions-d6a-2026-09-28.md)): arm A (Qwen3.5-9B, chat-v1, calibrated)
>   reads top-1 0.4197 and ECE 0.1656, against JEV-9B's 0.9181 and a bar of 0.0632, so the registered rule says **build D2–D4**.
>   The control failed as registered (bare-v1 0.3378 against the authors' B0 0.5180). The investigation found the inputs identical
>   to transformers' on the D0 fixture (prompts, token ids, label tokens: 150/150) and the gap numeric (4-bit weights plus goinfer's
>   re-quantization; 60% argmax agreement with f32). On those same items the reference's own Route A reads 0.559 and the trained head
>   0.934, so label scoring is about 37 points behind a trained head **in the reference itself**; the decision holds whatever the
>   numeric gap is. Owed by night: the 150 items at `--quant q4k` (CPU-only for this model on CUDA), to split that gap.
> - **D7's projection** is in [`decisions-d7-2026-09-28.md`](../measurements/decisions-d7-2026-09-28.md). For one
>   question, a decision beats a schema answer by 1.11× at 256 prompt tokens and by ~1.01× at 4K. For five questions
>   about one state on qwen3_5, decisions are 3–5× slower, so D8's trigger is projected to fire.
> - Previous status, 2026-09-27: C0 and D0 done, D1 and C1 built. C2 done. D5 built (`POST /v1/systemone`,
>   TypeSafe-compatible).
> - **C0 clears for enum, boolean and integer fields** ([`confidence-c0-2026-09-27.md`](../measurements/confidence-c0-2026-09-27.md)).
>   AUROC on the 1.5B: 0.847 / 0.727 / 0.680. The readout costs 4.20% of a token on the 1.5B and 1.44% on the 7B.
>   Number and string fields are parked: the labelled set drew too few wrong answers to judge them.
> - **D0** ([`decisions-d0-prior-art-2026-09-27.md`](../measurements/decisions-d0-prior-art-2026-09-27.md)) found
>   that autotrust's server is unpublished, so the reference fixture comes from the card's `decide()`.
>   - TypeSafe's `/v1/systemone` JSON is now recorded, and every SDK plus jevx takes a base-URL override.
>   - **The authors' own B0 report already puts Route A with the bare-v1 template far behind JEV-9B** (choice
>     top-1 0.532 against 0.898). D6a with that template would land in its "build D2–D4" branch; only a better
>     goinfer template could change that.
>   - The corrections to this doc are marked "(D0, 2026-09-27)" where they apply.
> - Two measurements decide what gets built: **C0** (does a per-field number mean anything? — answered above) and
>   **D6a** (is label scoring on any model good enough, or do the trained decision heads earn their build?).
>
> **Provenance.** This doc merges two drafts, neither of which was ever committed:
> - the C-items: a per-field confidence draft written 2026-09-25 under this file name, delivered
>   in chat and never saved into the tree. It is recreated here, corrected against `9bf7f3a7`;
> - the D-items: `task-decisions-2026-09.md` (D0–D9, written 2026-09-27, untracked). It was folded in
>   here with its item numbers unchanged, and the original moved to `_to_delete/`.
>
> **Correction to the 2026-09-25 draft.** It described the typed-output API as
> `goinfer.Into[T](ctx, prompt)`. No such function exists. The real surfaces are
> `constrain.GrammarFromStruct` / `constrain.JSONSchema` → `constrain.NewMasker(...).Process` set as
> `SamplingParams.LogitProcessor` (the README's "A Go struct the model cannot violate" section), and
> `response_format: {"type": "json_schema"}` on the server (`internal/serveapp/openai.go:558`). C1
> is written against those.
>
> **Siblings.** [`task-tool-grammar-union-2026-09.md`](task-tool-grammar-union-2026-09.md)
> (constrained generation; D7 measures decisions against it) ·
> [`task-work-queue-2026-09.md`](task-work-queue-2026-09.md) (J1/J3/J4 plumbing D5's batch route
> rides) · [`task-concurrency-2026-09.md`](task-concurrency-2026-09.md) (MC1/MC3 slots, D7's
> concurrency cell) · [`task-site-2026-09.md`](task-site-2026-09.md) (D9) ·
> [`task-never-swap-2026-09.md`](task-never-swap-2026-09.md) (D3's merge-at-load memory) ·
> [`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) (mode 3, "point my tools at it").

---

## 1. One idea, two surfaces

goinfer can already guarantee the *shape* of an answer: a JSON the model could not malform, or (with
D1) a pick from a closed option set. It says nothing about how sure the model was. Both surfaces
below close that gap with the same computation: **a softmax over a restricted set of tokens at a
position the grammar or the template controls**, renormalized, optionally temperature-scaled.

| | **C: confidence on constrained output** | **D: decisions** |
|---|---|---|
| Request | any schema-constrained generation | state + a question with a finite answer set |
| Work | full constrained decode | one prefill, no decode |
| Number | per field, aggregated over the field's free tokens | one distribution over the options |
| Where the probabilities come from | the masked logits `constrain` already computes (`MaskAt` at `constrain/constrain.go:148`, `Process` at `:208`) and discards after sampling | the label tokens' logits at the last prompt position (Route A), or a trained head over the final hidden state (Route B) |

**Where they meet.** An `enum` or `boolean` field in constrained output is a decision: the masked
distribution at the position that decides the value *is* a distribution over a closed answer set.
So:
- C and D share one readout helper (restricted softmax + per-kind temperature) and one calibration
  sidecar format (autotrust's `calibration.json`, read by one loader).
- D0's labelled fixture doubles as C0's gate-2 set for enum and boolean fields.
- If D6a shows Route A is well-calibrated after temperature scaling, enum-field confidence in C
  inherits that calibration through the same `choice`/`noul` temperatures. Free-string and numeric
  fields get no such inheritance and stay raw.

**Why this is worth doing.** A product category launched in September 2026 on the premise that a
structured answer should carry a probability you can threshold on (route, escalate, hand to a
person). TypeSafe's Jev (2026-09-15, closed, hosted) is the prominent one; open students of it
already exist on a family goinfer loads (§3). No local inference engine exposes confidence on
structured output. It composes with what goinfer already leads on, and it is a pitch that does not
need a speed claim, which matters under the Sep 11 promotion position (no promotion before parity
with Ollama). Decisions are also prefill-only, so the decode gap does not arise.

**Why it might not be.** The wave is two weeks old, and the category's published numbers are
contested: the claims are self-benchmarked, and one open model's headline requires fine-tuning on
the benchmark's own training split while it scores near a random baseline zero-shot. Route A may be
good enough, which would make D2–D4 unnecessary. C0 may show most field tokens are grammar-forced,
which would make C1 decoration. Both are measured before anything is built.

**Naming.** "Confidence" and "decisions" in code, flags and docs. "Jev" and "System One" belong to
TypeSafe and appear only in prose that identifies the prior art.

## 2. The claim we must not make

**A masked token probability is not a calibrated confidence.** It is the model's probability over
what the grammar or the option set allowed, which is narrower than P(this answer is correct). The
decision-model category reports Brier scores and ECE against gold labels; goinfer reports that only
where it has run a labelled evaluation (D6), and says `calibrated: false` everywhere else. Given how
much of this project rests on not overstating what a number means, the naming and the docs matter as
much as the mechanism. Every surface that exposes a number links to the paragraph that says what it
is.

## 3. What goinfer has today (verified against `9bf7f3a7`, 2026-09-27)

**Constrained decoding.**
- `MaskAt` (`constrain/constrain.go:148`) and `Process` (`:208`) walk the full logit vector at
  every constrained position and set illegal entries to −∞. The surviving distribution is discarded
  once a token is sampled.
- `ForcedRun` (`:97`) and `ForcedBytesRun` (`:166`) exist because the grammar often forces the next
  token outright. A forced token has probability 1.0 by construction and says nothing about the
  model. The caveat recorded above `ForcedRun` says the grammars allow optional whitespace at every
  structural boundary, so strict forcing fires mainly inside fixed literals (object keys, enum and
  const values), not the scaffolding between them.
- The struct → schema path is `constrain/reflect.go` and `constrain/schema.go`.

**Decisions.**
- **Hidden-state seam exists and is wired for `qwen3_5`.** `ForwardCapture`
  (`decoder/model.go:1292`) returns logits plus captured residuals. `qwen3_5` dense shares
  `qwen3_5_moe`'s own-forward row, which has Captures = true and Recurrent = true
  (`decoder/arch.go:954`; the predicate is `a.qwen35 != nil`, so it matches both). The capture
  contract (`decoder/capture.go:14`) is the residual *after* layer l, before the final norm, so D2
  applies the final RMSNorm itself and must match HF's `hidden_states[-1]`, which is already normed
  on Qwen. Verify in D2; do not assume.
- **LoRA cannot reach this family.** Both paths are closed:
  - Merge-at-load (`--lora`): `validateTargets` (`decoder/lora.go:144`, called at
    `decoder/weights.go:695`) knows only the Q/K/V/O/gate/up/down suffixes, so an adapter that also
    targets the GDN projections is refused whole. The merge itself runs inside the `loadProj`
    closure (`decoder/weights.go:758`), and the GDN projections are loaded outside it.
  - Compute-time (`--adapter`): `LoadAdapter` (`decoder/lora.go:353`) refuses every own-forward
    family by design.
- **Recurrent state cannot be rewound.** `KVCache.TruncateTo` (`decoder/kvcache.go:540`) reports
  inexact on any partial rewind when the model has recurrent state, and `.giw-kv` snapshots skip
  recurrent state (`decoder/kvsnapshot.go:62`). So "prefill the shared state once, branch per
  question" is not available on `qwen3_5` today (D8).
- **Route A is approximable from outside already.** `/v1/completions` with `max_tokens: 1,
  logprobs: true, top_logprobs: 20` (`internal/serveapp/openai.go:556`, cap at `:33`) gives a client
  the label-token logprobs, with no renormalization over the option set, no calibration, and no
  guarantee the labels are in the top 20. That is the baseline D1 improves on.

**Cost denominator.** The sampler's share of a token is 5.4% (phi3-mini) to 18.2% (the model
measured in the G26 work) per `docs/spec/10-optfwd-gate.md:177`. A softmax over a 152k-vocabulary
masked logit vector at every constrained position is work of the same order.

---

## 4. Items

### C0 — does a per-field number mean anything? (measurement only; may close C)

Deliverable: a short written answer to 1–4 with the measured forced-fraction table. No API, no code
beyond what the measurement needs.

1. **Forced fraction per field kind.** On a real schema with enum, bool, int, number and free-string
   fields, count what fraction of each field's tokens had more than one legal continuation. If most
   of a field kind's tokens are forced, its per-field average measures the schema, not the model.
2. **Aggregation per field kind.** Candidates: product of per-token probabilities (length-biased),
   geometric mean (length-normalized), minimum (weakest link), first free token only (enums and
   booleans, where one position usually decides the value). State which one each field kind uses
   and why. Do not pick one for all of them.
3. **Is the enum/boolean case the whole viable feature?** For those fields the deciding position's
   masked distribution *is* the answer distribution (§1), with no aggregation question. If C0 shows
   free-string confidence does not discriminate, C1 ships enum/boolean only.
4. **Cost.** Softmax over the masked logits at every constrained position, as a percentage of a
   token, against the sampler share above. Quiet box, report spread. If it costs more than sampling
   does, it is opt-in (it is opt-in regardless; this sets the documented cost).

**C0 pre-registration (2026-09-27, committed before any graded run).**
- **Harness:** `metal/confidence_c0_test.go` (`TestConfidenceC0`).
- **Analysis:** `docs/measurements/confidence-c0-2026-09-27/analyze.py`, written before any graded run.
  The gates below are that script's arithmetic.
- **Models (graded):** qwen2.5-coder-1.5b-instruct q4_k_m (`.gguf`) and qwen2.5-7b-instruct q4_k_m
  (`.int4.metal.giw`, tokenizer from its `.gguf`), both from `~/models`, at `-quant int4` on Metal, greedy,
  up to 200 tokens per answer.
  - A 3-ticket smoke run on qwen2.5-coder-0.5b checked the harness before this was written. It is not graded,
    and it is disclosed below because it was seen.
- **Data:** `docs/measurements/confidence-c0-2026-09-27/tickets.jsonl`, 60 support tickets written for
  this (sha256 `71787d8fd0aad4368646bff65ad70e1a00fcca5a7d74a9d29b3a8619df6936ef`).
  - Every gold label follows from rules stated in the prompt. The system prompt is in the harness.
  - The schema has one field of each kind: `category` (enum of 5), `urgent` (boolean), `order_count`
    (integer), `refund_amount` (number), `customer_name` (string, graded), and `summary` (string, no gold:
    gates 0 and 1 only).
  - D0's gold-labelled JEV items join the enum/boolean set when D0 delivers them. This registration covers
    the hand-built set.
- **Definitions.**
  - *Free token*: at some byte of the token, the grammar allowed two or more non-whitespace next bytes.
    Otherwise the token is *forced*: its probability is tokenization preference, not the model's view of
    the value.
  - *Field attribution*: a token belongs to the field whose value bytes it overlaps; otherwise it is
    scaffolding.
  - *p*: the token's probability under the model's distribution restricted to the grammar-legal tokens,
    at T = 1.
  - *Decision* (enum/boolean): at the first token of the value consistent with exactly one option, the mass
    of every legal token summed by the option it spells and renormalized. The field's decision confidence is
    the chosen option's share.
- **Primary aggregation per kind** (item 2; the others are reported as secondary, never used to decide):
  - enum and boolean: *decision*;
  - integer and number: the *minimum* p over free tokens (a number is as weak as its weakest digit);
  - string: the *geometric mean* p over free tokens (length-normalized).
- **Correctness:** exact for `category`, `urgent` and `order_count`; |Δ| < 0.005 for `refund_amount`;
  `customer_name` case-insensitive after trimming whitespace and trailing punctuation. An unparseable output
  or missing field is excluded for that field, and counted.
- **C-gate 0 (enough free tokens):** a kind is surfaceable only if at least 80% of its instances have at
  least one free value token. Its confidence is computed over free tokens only; forced tokens never
  enter a number.
- **C-gate 1 (cost):** per model, the mean in-situ readout time (log-sum-exp over the legal logits, timed
  inside the processor) is at most 5% of the mean mask-only decode token time. That is the bottom of the
  sampler's measured share, 5.4%. The mean is the gate, because the readout costs several times more
  inside a free string, where nearly the whole vocabulary is legal, than at a structural position. The
  median is reported beside it.
  - The 0.5B smoke run read 7.8% mean and 3.3% median.
  - Zero cost when disabled is C1's to verify: no hook, no code on the path.
  - A FAIL here does not close C. The feature is opt-in regardless, so it means the readout is optimized
    before C1 ships (float32, or reading only at free positions), then re-measured.
- **C-gate 2 (it discriminates):** directional only, per field kind and per model, on the primary
  aggregation, for a model with at least 8 correct and 8 wrong instances of that field:
  - AUROC (P(a correct answer's confidence > a wrong one's), ties ½) ≥ 0.65 → PASS;
  - AUROC < 0.55 → FAIL, and the kind does not ship;
  - in between → ambiguous, parked.
  - A kind passes only if every qualifying model passes. No qualifying model → *insufficient errors*,
    parked (not failed), with the counts on the page.
- **Item 3's answer:** if enum and boolean pass gate 2 and the integer, number and string kinds do not, C1
  ships enum/boolean only.

**C0 result, 2026-09-27** ([`confidence-c0-2026-09-27.md`](../measurements/confidence-c0-2026-09-27.md)):
- **Enum, boolean and integer pass every gate.**
  - AUROC on the 1.5B: 0.847 (48/12), 0.727 (38/22) and 0.680 (43/17).
  - The 7B makes too few mistakes to qualify on any kind; its AUROCs of 0.97 / 0.91 / 0.94 point the same way.
- **Number and string are parked.** There were too few wrong answers: `refund_amount` was right 60/60 on both
  models.
- **Cost:** a mean 4.20% (1.5B) and 1.44% (7B) of a decode token. The ungraded 0.5B read 7.8%, so C1 should read
  only at free positions.
- **The consequence for C1:** it may surface enum, boolean and integer fields. Number and string wait for a harder
  labelled set. Every kind that passed decides its value in one free token.

### C1 — capture, attribute, surface (only if C0's gates clear)

**Built, 2026-09-27** (C0 cleared enum, boolean and integer).
- **Capture:** `constrain.Masker.CaptureConfidence(ConfidenceOptions)`.
  - Inside `Process` it records, at each position outside a free string, the normalizer and the legal tokens' logits
    (all of them when there are at most 256, else the top 64).
  - Off, `Process` pays one nil check. `TestFieldConfidence_offAndRefusals` pins identical masking and no capture
    state.
- **Attribution:** no plumbing was needed; the schema grammar's frame stack already carries the path.
  `FieldConfidence(generated)` replays the tokens through a fresh copy of the grammar, byte by byte. That gives each
  byte's owning value and path (`meta.level`, `tags[1]`) and each token's byte-level forcedness.
- **Aggregations, C0's registered ones:**
  - enum and boolean: option mass at the deciding token, where BPE splits sum per option;
  - integer: the minimum over free tokens.
  - Number and string fields, and fields with no free token, are omitted.
  - `ConfidenceOptions.EnumTemperature` / `BooleanTemperature` apply a fitted temperature and mark the field
    calibrated. Serve passes none today.
- **Serve:** `goinfer_confidence: true` on `/v1/chat/completions` and `/v1/completions` with `response_format`
  `json_schema`.
  - The response gets a top-level `goinfer_confidence` array; a stream sends it as one event after the finish chunk.
  - Other routes refuse the flag with a 400, so none drops it silently. The response without the flag is unchanged.
  - A confidence request runs plain constrained decode: the grammar-fused speculative path drives the masker
    without `Process`.
  - Documented in `docs/server.md` with the §2 caveat.
- **Tests:**
  - a synthetic generation pins paths, kinds, BPE-summed distributions, the integer minimum, the temperature, the
    omissions and the skipped free strings. Mutation-checked: counting only the chosen token turns it red;
  - serve refusals run in CI;
  - end to end on the local 0.5B: one record per enum/boolean/integer field, none for the string, the answer
    identical with and without the flag, one stream event, and tools refused.
- **Cost** (`TestConfidenceCost_C1`, real 151,936-token vocab; the box was not quiet, load 3, so indicative):
  capture adds 0.29 ms at an object-key position, 0.28 ms at an enum value, and 0.000 ms inside a free string.
  Against C0's decode tokens that is about 2% (1.5B) and 0.8% (7B) at the positions it reads.

- **Capture, not compute.** A hook at the masking seam that records, per position, the probability
  of the sampled token among the legal set, plus the full restricted distribution at positions the
  grammar marks as value-deciding for enum/boolean fields. Off by default; zero cost when off,
  verified.
- **Attribute positions to fields.** The grammar knows where it is in the schema. Establish whether
  a JSON-path-per-position mapping is already available from the grammar state or needs plumbing.
  This may be the bulk of the work.
- **Surface it, opt-in, without changing anything for callers who do not ask:**
  - Library: `Masker` gains an accessor returning `[]FieldConfidence{Path, Kind, Value, Confidence,
    Distribution (enum/bool only), FreeTokens}` after generation. `GrammarFromStruct`, `NewMasker`
    and `Process` keep their signatures and behaviour.
  - Server: a request-level opt-in on `response_format: json_schema` requests returns the same
    records in a vendor extension field. The OpenAI-shaped response is byte-identical when the
    opt-in is absent. Field name decided at build time and recorded in `docs/server.md`.
- Enum/boolean fields carry `calibrated: true` only when a `calibration.json` with the matching
  kind is loaded (§1); every other field is `calibrated: false`.
- **Do not build** a calibration layer for free-string fields, a routing helper, or anything that
  turns the number into a decision for the caller.

### C2 — docs for C

**Done, 2026-09-28.**
- `docs/server.md`, with the §2 caveat (landed with C1).
- A README paragraph under "A Go struct the model cannot violate", with the caveat linked.
- [`examples/confidence`](../../examples/confidence/main.go): a complete program. Its test runs the real binary on a
  small Qwen checkpoint when one is present. On the 0.5B it printed category `billing` 0.70 (right), `urgent`
  `true` 0.62 (wrong, and less sure) and orders 1 at 0.51.

`docs/server.md` (the opt-in and the §2 caveat), the README's "A Go struct the model cannot
violate" section (one paragraph, the caveat linked), and a worked example in `examples/`.

### D0 — prior art, reference fixture, API shape (no code in goinfer)

**Result, 2026-09-27** ([`decisions-d0-prior-art-2026-09-27.md`](../measurements/decisions-d0-prior-art-2026-09-27.md),
every fact read from a primary artifact at a pinned revision). Where it contradicts the text below, the record wins:
- **autotrust's `jev_judge` server is not published**, and the `{distribution, decision, confidence, latency_ms}`
  fields appear in no source. The reference fixture is the JEV-9B card's `decide()` (transformers 5.16.1 + peft
  0.21.0), run on the Linux box: `docs/prompts/nobara-decisions-d0-fixture-2026-09.md`.
- **The template is bare-v1**, with no chat template and no BOS. The labels are bare `false`/`true`, `0`–`5` and
  `A`–`P` (no leading space). `slots.template_version` holds the version.
- **The 9B's numbers:**
  - T = noul 1.0022 / choice 0.9840 / score 1.0122;
  - adapter 160,486,456 B;
  - JEV-9B's base is the post-trained `Qwen/Qwen3.5-9B`, not `-Base`.
- **Gold labels exist only on the `openjev_v2` rows** of `SargeDev/jev-distill-corpus-v3`:
  - calibration: 1,109 noul and 463 choice, with no score rows;
  - ood: 9,767 noul, 3,219 choice and 72 score.
  - The `yuri_v3` targets are Jev's own distributions, a teacher and not gold.
- **API shape:** TypeSafe's `POST /v1/systemone` is recorded verbatim. `@typesafe-ai/sdk`, `typesafe-sdk` (PyPI),
  the Vercel and LangChain providers and jevx all take a base-URL override. The decision is still the owner's.

- **Prior-art sweep (mandatory):** autotrust's `jev_judge` server code (the prompt template and its
  version in `judge_config.json`, the verbalizer token ids, how fewer than 16 options occupy the
  choice slots, the batch route); `kyegomez/open-jev` (a different, random-weights architecture;
  note it and move on); the other open reproductions the awesome-jev list names; and whether
  llama.cpp, Ollama or vLLM ship a decisions-shaped endpoint (vLLM does not, as far as autotrust's
  card says: autotrust wraps it).
- **API shape (his call).** Three candidates:
  - autotrust's `POST /v1/decisions` + `POST /v1/decisions:batch`, the only open schema:
    `{kind, state, question, options}` → `{distribution, decision, confidence, latency_ms}`;
  - TypeSafe's `POST /v1/systemone`: `{model, state, questions: {name: spec}}`, many questions per
    state;
  - both, over one core.

  Record the exact request/response JSON from TypeSafe's public docs rather than reconstructing it
  from third-party pages. Check whether the TypeSafe SDKs (`@typesafe-ai/sdk`,
  `typesafe-sdk-python`) and the Vercel/LangChain providers accept a base-URL override.

  **A ready client exists (added 2026-09-27).** [`muthuishere/jevx`](https://github.com/muthuishere/jevx)
  is a CLI for Jev-style decisions (`is`, `ask --choice/--score`, `pick`, `rank`, `filter`) whose
  exit codes map confidence to shell control flow (≥0.6 → 0, `unsure` → 3, error → 4; **(D0)** and no → 1, see D5). It talks to
  any server that speaks `/v1/systemone` through a profile and runs no models itself. A
  `/v1/systemone`-compatible route therefore gives goinfer a working terminal/CI/agent client on day
  one, which moves the decision toward **both**: TypeSafe's shape as the compatibility surface,
  autotrust's one-question shape kept only if it costs nothing extra. jevx's scenario guide becomes
  D5's acceptance harness.
- **Reference fixture.** Run autotrust's Python server on JEV-9B on the Linux box (bf16 ~18 GB; CPU
  transformers is fine offline). Record `distribution` for 100 items from their calibration split
  plus 50 from the Open-Jev OOD split, all three kinds. Commit inputs and outputs under
  `testdata/decisions/` as goldens (the pin-generator pattern). Where items carry gold labels, they
  are also C0's gate-2 set for enum and boolean fields.

### D1 — Route A: label-token scoring on any model

**Built, 2026-09-27.**
- **`internal/decide`:** the readout (`internal/confidence.RestrictedLogSoftmax`), the templates, `calibration.json`
  load/save, and the per-kind temperature fit.
  - `bare-v1` (the default) is byte-identical to JEV's template; the D0 record's rendered example is pinned by
    `TestRender_bareV1`.
  - `chat-v1` puts the same content in the model's chat template, with thinking off.
  - Each verbalizer must be a single bare token in the model's tokenizer, or `New` refuses and names it.
  - A calibration fitted under the other template is refused.
- **CLI:** the `goinfer-chat decide` / `decisions-calibrate` subcommands (`internal/decidecmd`). Every backend's chat
  binary has them.
  - `decide --model <f> [--template chat-v1] [--calibration c.json] [--permute n] in.jsonl`: one JSONL line in and
    one out. Lines are in the corpus's row shape.
  - `decisions-calibrate … -o calibration.json labelled.jsonl` fits T per kind on mean KL(target ‖ p_T), by
    golden-section search on 1/T, in autotrust's format. It warns when T hits the search bound, where the model's
    ranking disagrees with the labels beyond what a temperature can fix.
- **One prefill per line**, through `Model.Generate` with a logit processor. Every backend, the resident prefix reuse
  and the KV slots apply. The prompt is encoded as plain text, so special-token text in a state stays text.
- **Owed:** D1 against transformers' B0 on identical bytes (the nobara fixture's `route_a_b0.jsonl`): the token ids
  first, then the distributions.
- **Not yet decided: the template for D6a.** On the 0.5B (anecdotal, not graded), chat-v1 separated a refund /
  no-refund pair better than bare-v1 (P(true) 0.81 against 0.68, where bare-v1 read 0.45 against 0.39). But
  `permute 3` flattened a 3-way choice to near uniform, which is position bias. D6a has to say which template it
  grades, and D0's B0 already covers bare-v1 on Qwen3.5-9B.

- A `decider` in `internal/serveapp` (not `decoder`). Build a prompt from `{state, question,
  options}` with a fixed template; label options A, B, C… so each label is one token in the model's
  vocab (check per tokenizer; fall back to digits). Disable thinking in the chat template. One
  prefill; take the last-position logits; read the label tokens; log-softmax over just those; apply
  the per-kind temperature. This is the shared readout helper C1 also uses.
- `noul` uses two labels in yes/no form; `score` uses six labels 0–5 and returns the distribution
  plus the expected score. **(D0, 2026-09-27)** The reference (bare-v1) labels noul as bare `false`/`true` and
  uses no chat template, so "disable thinking in the chat template" above applies only to a chat-templated
  variant. D1 renders bare-v1 by default, so Route A and a later Route B read the same prompt.
- **Calibration.** A `decisions-calibrate` subcommand fits the per-kind temperature on a labelled
  JSONL (golden-section or L-BFGS on one scalar; no dependency) and writes `calibration.json` in
  autotrust's format, so Route A, Route B and C1's enum fields share one loader.
- **Position bias.** Label-order bias under shuffled options is a known failure. Offer opt-in
  `permute: n` that averages over n option orders (n prefills) and document the cost.

### D2 — final-norm hidden seam for the last prompt token (Route B)

- Batched-prefill the prompt minus its last token (existing path), then run the last token through
  `ForwardCapture` with `layers = [NumLayers-1]` and apply the final norm. Expose one decoder method
  (`Model.PromptHidden(prompt) ([]float32, error)`) so the recurrent/rewind and resident questions
  live in one place.
- **CPU first.** If the resident CUDA/Metal executors do not expose the capture, D2 ships CPU-only
  and GPU becomes a follow-up with its own measurement. No silent fallback: the response's `backend`
  field says which ran.
- **Gate:** cosine ≥ 0.9999 against HF `output_hidden_states[-1]` at the last position on 5 prompts
  (f32 weights), before any head is attached.
- **DONE 2026-09-30.** `decoder.Model.PromptHidden(ctx, prompt)` (`decoder/embed.go`): every family, its own layer loop
  included, since it runs `runLayers` per token in a fresh cache and then the same final norm `logitsFromHidden` applies (the
  generic families take the batched path). CPU only; no resident executor exposes this hidden state yet. **Gate passed:**
  `TestPromptHidden_matchesHF`, cosine 1.00000000 and relative L2 at most 1e-5 on 5 prompts (2–64 tokens) × 3 fixtures, against
  `scripts/pin_prompt_hidden.py` (transformers 5.16.1, f32). **A trap found on the way:** the tiny checkpoints' final-norm
  weights are all 0, a scale of 1 under the add-one RMSNorm, so cosine alone could not see a missing or doubled final norm (a
  mutation dropping it passed). Fixed with a derived fixture, `qwen3_5-tiny-normw` (random final-norm weight), and the L2
  bound; both mutations now fail 20 checks.

### D3 — LoRA merge-at-load for the GDN projections (Route B)

- Extend the `qwen3_5` loader so `in_proj_qkv`, `in_proj_z` and the linear-attention `out_proj`
  pass through the same f32 merge as `loadProj` before quantization. Extend `validateTargets` with
  the GDN suffixes *from the schema*, not a hand list (lora.go's own V-12 comment records why). Mind
  `FusedDeltaNetProj` (`decoder/arch.go:368`): the merge has to hit the on-disk tensor before any
  repack.
- **Consequence, stated:** a merged model is a decision model; its text generation is no longer the
  base Qwen's. One set of weights serving both is out of scope (§6).
- **DONE 2026-09-30 (the merge; `transcode --lora` not built).**
  - **What changed.** `loadQwen35Attn` reads every tensor through the loader's merge-aware readers (`loadMatMerged`,
    `loadF32Merged` in `decoder/weights.go`), so a delta is merged on the on-disk f32 before qwen3_next's fused
    `in_proj_qkvz` split or Olmo Hybrid's q/k/v concatenation. With no adapter those readers are exactly `loadMatQ` /
    `TensorF32`, so an adapter-free load is unchanged.
  - **Validation, departing from the bullet above.** `validateTargets` was not extended. The qwen35 tensor names live
    in `loadQwen35Attn` across four layouts, so a list would be the second copy V-12 warns about. Instead
    `loraAdapter.checkAllMerged`, called once in `loadWeights`, refuses any delta the load did not merge. It is
    derived from what the loader did, so it cannot drift from it.
  - **Two silent no-ops it also closes.** `buildInternLM2Weights` and `buildGptOssWeights` take no adapter and
    returned before any LoRA check, so an adapter on either loaded clean and changed nothing. It also catches a
    listed name on a layer that does not load it (a `self_attn` projection on a linear-attention layer).
  - **Gate passed:** `TestLoRA_mergeAtLoad_qwen35MatchesPEFT` (`decoder/lora_qwen35_test.go`).
    - **The reference.** A PEFT-written adapter (`scripts/pin_qwen35_lora.py`; peft 0.21.1, transformers 5.16.1,
      f32) on `qwen3_5-tiny`: r 4, α 8, all ten of autotrust's target kinds, 25 modules over three DeltaNet
      layers and one full-attention layer. B is redrawn non-zero so the adapter is not a no-op.
    - **The result.** `PromptHidden` matches PEFT's `merge_and_unload()` to relative L2 3.1–4.9e-7 on five prompts
      (2–64 tokens), where the bar is 1e-5. The base is 0.72–1.02 away, and the test fails if it is under 1e-3.
    - **Mutation-checked.** Recording but not applying the merge for `in_proj_z`, the DeltaNet `out_proj`,
      `in_proj_qkv`, or the full-attention `q_proj` each goes red, 0.2–0.48 off.
    - **The two refusal tests** go red with `checkAllMerged` disabled: an unloaded name on `qwen3_5-tiny`, and an
      adapter on `internlm2-tiny`.
  - **Not tested:** the fused `in_proj_qkvz` (qwen3_next) and Olmo Hybrid layouts are merged by the same readers
    but have no PEFT gate of their own. JEV-9B's real adapter has not been loaded yet (that is D4's first step).
  - **Memory.** The merge runs tensor by tensor on the streaming-quant path, so the extra peak is one tensor's f32
    copy, not the model's.
  - **`transcode --lora`: built 2026-09-30** as `prequant -lora DIR` (`internal/prequant/lora.go`).
    - **What it writes:** a `.giw` from a safetensors directory with the adapter merged at load, plus
      `<bundle>.lora.json` recording the adapter and the sha256 of its weights. The sidecar is published only after
      the bundle passes its self-check, and a plain rebuild at the same path removes it.
    - **How loads use it:** `prequant.AdapterLoRA` decides what a decision head's load merges. A safetensors model
      gets the adapter; a `.giw` whose sidecar records that exact adapter gets nothing; any other `.giw` is refused.
      Serve's `head=` and `goinfer-chat decide --head` both call it.
    - **Test:** `TestTranscodeLoRA`. On the tiny fixtures at f32, the bundle's `PromptHidden` is bit-identical to a
      merge-at-load and differs from the base; another adapter and a sidecar-less bundle are refused; a GGUF input is
      refused.
    - **Peak memory:** the transcode itself holds the whole quantized model (it is `transcodeDir`), so a 27B is built
      on a box that can hold it, and the Mac then mmaps the result.

### D4 — readout head + calibration (Route B)

- autotrust's recipe: a LoRA on the backbone (r=16, α=32, targets `in_proj_qkv, in_proj_z,
  out_proj, q/k/v/o_proj, gate/up/down_proj`, 416 MB unmerged); the last prompt token's final-norm
  hidden state through an fp32 linear head `H → 24 slots` (`head.safetensors`: noul 0–1, score 2–7,
  choice 8–23); a per-kind temperature (`calibration.json`: noul 1.014, choice 1.016, score 1.004);
  **(D0, 2026-09-27)** those are JEV-27B's figures. JEV-9B's adapter is 160,486,456 B and its T is noul 1.0022 /
  choice 0.9840 / score 1.0122;
  softmax over the slots the request's options occupy. Published: JEV-27B mean KL 0.104 on the
  Open-Jev OOD split, JEV-9B 0.234.
- Load `head.safetensors` and `judge_config.json` (slot layout, verbalizer ids, template version,
  `weights_mode`). Refuse `weights_mode` values other than `unmerged`/`merged` with a clear error.
- Slot selection per kind as above; confirm in D0 that masking unused choice slots before the
  softmax is what the reference does.
- The template must be byte-identical to the reference's version. Pin it, and refuse a
  `judge_config.json` whose version the loader does not know.
- **DONE 2026-09-30.** `decide.LoadHead` and a Route B `Decider` (`internal/decide/head.go`); `goinfer-chat decide
  --head DIR` and `decisions-calibrate --head DIR` (`internal/decidecmd`).
  - **Loading.** `judge_config.json`, `head.safetensors` (`proj.weight` [slots, hidden], `proj.bias`) and
    `calibration.json`. The loader refuses:
    - a `weights_mode` other than `unmerged`/`merged`;
    - a `template_version` other than bare-v1;
    - a `softcap`;
    - a kind goinfer does not have;
    - slot verbalizers that are not goinfer's labels in order;
    - ranges outside the head, and any shape mismatch.

    An unmerged head's adapter becomes the load's LoRA (merged at load, D3), and a different `--lora` is refused.
  - **Decision.** `PromptHidden` (D2), then W·h + b, then the kind's first n slots, then ÷ T, then softmax. That is
    the reference `decide()`, which slices the slots before the softmax, so masking unused choice slots is confirmed.
    The head's own `calibration.json` applies unless `--calibration` replaces it. Refused: chat-v1, option
    descriptions, a score other than the head's six levels.
  - **Gate passed:** `TestHead_matchesReference`. The reference is `scripts/pin_decisions_head_tiny.py` (peft 0.21.1,
    the adapter **unmerged**, as the demo Space runs it) on a tiny JEV-shaped judge over `qwen3_5-tiny`
    (`testdata/decisions/judge-tiny`).
    - **Result:** max |Δp| 1.5e-8 to 2.1e-7 on five cases (noul ×2, score, choice at 3 and 16 options), against a bar
      of 1e-5.
    - **Mutation-checked:** each of these goes red:
      - every kind read from slot 0 (0.62);
      - the bias dropped (0.107);
      - the temperature skipped (0.018);
      - a softmax over all of a kind's slots (0.685).
  - **Real-model probe (exploratory; nobara-pc CPU, 2026-09-30, binary at 3d333203).** Two D0 items on autotrust's
    JEV-9B.
    - **It loads:** the 200 deltas merge and pass `checkAllMerged`, and the head and temperatures apply.
    - **It answers like the reference:** the prompt token counts match (516, 112), and both items give the
      reference's [1.0, 0.0]. At f32, KL(ref ‖ goinfer) is 3.5e-9. Both items are saturated, so this proves the
      wiring, not the numbers; D6b does the numbers.
    - **Cost:**

      | quant | load | per token | peak RSS |
      |---|---|---|---|
      | int4 | ~45 s | ~0.2 s | 33 GB |
      | f32 | — | ~1.2 s | 52.7 GB |

      `PromptHidden` runs Qwen3.5 token by token on the CPU (no resident executor exposes the hidden state, D2),
      which is decode-speed and memory-bandwidth bound.
  - **Not done:**
    - ~~Serve's `/v1/systemone` still answers by Route A only~~ **done 2026-09-30:** a `--model` entry's `head=DIR`
      loads the head at startup, and its unmerged adapter becomes the entry's LoRA (a different `lora=` is refused).
      `/v1/systemone` on that entry answers by Route B, with the head's template and calibration.
      - **Validation:** questions are validated against the head's limits before any prefill (422), via
        `Decider.Validate`.
      - **Descriptions:** they are left out of the prompt, as `jev_core` builds it, and listed in
        `goinfer.descriptions_dropped`. They are not refused, since TypeSafe's score questions always carry them.
      - **Reporting:** `/v1/models` reports route `head`. The load banner says the entry's text generation is the
        decision model's.
      - **Tests:** `TestSpecHead`, `TestSystemOne_head`, `TestHead_dropsDescriptions`. Documented in
        `docs/server.md`.
      - **End-to-end probe** (exploratory; nobara-pc CPU, `goinfer-serve` at 11837723, default int4):
        `--model jev=~/models/JEV-9B,head=~/models/JEV-9B` loads, and `/v1/models` reports route `head`. One D0 item
        (`v3_63e22867e7c69c10_n`) answers P(true) 0.6175 against the f32 reference's 0.602, with 71 input tokens, the
        reference's count.
    - ~~`jev_core`'s state truncation~~ **done 2026-09-30** (`internal/decide/truncate.go`). A state over 1024 tokens
      is cut to its first 614 and last 410 tokens and decoded back to text, with a split character replaced exactly as
      CPython's `errors="replace"` does it.
      - **Gate:** `TestTruncateState_matchesReference`, eight states built to cut badly (CJK, emoji, accents, rare
        ideographs, a multi-byte seam, exactly 1024 and 1025 tokens). The kept text and the whole prompt's token ids
        match transformers with JEV-9B's own tokenizer (`scripts/pin_decisions_truncation.py`). A 50/50 cut goes red.
      - **CPython's rule:** `TestPyReplaceInvalidUTF8` holds the replacement to CPython's output on 16 byte strings,
        the cases where Go's `strings.ToValidUTF8` differs.
      - **Scope:** it runs on the head route only; Route A's prompts are unchanged. None of the pinned corpus's 26,824
        states reaches 1024 tokens (the longest is 541), so it matters for served requests, not for D6.
      - **Still different from the reference:** a state containing special-token text such as `<|im_start|>`, which
        transformers parses as one special token and goinfer encodes as plain text.
    - A batched or resident hidden-state path for Qwen3.5, which D6b's cost depends on (below).
- **D6b sizing, from that probe.** The 150 items are 27,861 tokens:

  | arm | estimate |
  |---|---|
  | f32 | ≈ 9 h |
  | int4 | ≈ 1.6 h |
  | int8int8 | not probed; likely 2–3 h |

  That was ~13–14 h of night time under the 3 h job cap.
  - **Owner's choice, 2026-09-30:** make `PromptHidden` batched first. Done: `runLayersQwen35N` (7ba740fe).
  - **Re-measured on the same JEV-9B probe** (exploratory; nobara-pc CPU, binary at 90f8dbc9, single items):

    | | per-token (before) | batched (now) | per-token RSS |
    |---|---|---|---|
    | int4 | 186–201 ms/token | 65–66 ms/token (≈3×) | 33 GB |
    | f32 | 1,188 ms/token | 74–84 ms/token after the first item (≈15×; the first read 241, warm-up) | 52.7 GB |

  - **Numerics on four unsaturated items:**
    - f32 batched matches the reference to KL ≤ 6e-8.
    - At int4, batched and per-token differ from each other by KL 3.5e-5 to 7.4e-3, each about as far from the
      reference as the other, in both directions. That is int4's own error, which D6b's int4 arm grades.
  - **D6b is now ~2 h for the three arms**, run under the amendment below.

### D5 — the endpoint

**Built, 2026-09-28: `POST /v1/systemone`, TypeSafe-compatible** (the owner's pick). autotrust's `/v1/decisions`
schema was never published (D0).
- **Code:** `internal/serveapp/systemone.go`, answered by `internal/decide` with the same auth → haltGate → inf →
  maxBytes chain as its siblings.
- **Request:**
  - `{model, state, questions: {name: {type, instructions, criteria}}}`. The question and option order is preserved
    (choice letters follow the order sent).
  - A state or instructions given as an object or array is compacted to JSON, as jevx does.
  - Up to 256 questions per request. Every question is validated before any prefill, and a bad one is a 422.
- **Response:** TypeSafe's per-kind answers (`noul`; `choice` + `probabilities` + `confidence`; `score` + `legend`
  + `probabilities` + `confidence`), `usage` (output 0) and a `goinfer` block (route, template, calibrated per kind).
- **`confidence`** = the top probability's margin over uniform, `(n·p − 1)/(n − 1)`, TypeSafe's demo form
  generalized. TypeSafe publishes no formula.
- **Limits:** choice 2–16 options (A–P), score 2–10 levels. `decide` gained described options and N-level scores
  for this; bare-v1 is byte-identical to JEV's template only without descriptions.
- **Flags:** `--decisions-template` (default chat-v1) and `--decisions-calibration`, both validated at startup.
- `/v1/models` lists each entry's `decisions` support. Adapter entries are refused.
- **Tests:**
  - a CI run of the handler with an injected decider, checked against jevx's fail-closed rules: exactly the
    questions asked, noul in [0,1], each probability set summing to 0.98–1.02, the choice among the offered keys,
    the options in the order sent, and `usage` present;
  - the 422/404/400 cases;
  - the confidence map;
  - a real-model end to end on the local 0.5B.
- **Documented** in `docs/server.md` with the §2 caveat, the SDK/jevx base-URL settings, and the authors' measured
  quality of this method.
- **Not done:** the `:batch` route (a single request already carries many questions); the J3 job object; the
  `goinfer-chat -decide` JSONL mode is `goinfer-chat decide`, from D1.
  - Running jevx's own binary against it: its request shape and fail-closed checks are reproduced in the test
    instead of executing third-party code here.

- `POST /v1/decisions` (+ `:batch`, ≤256 items) and the TypeSafe-shaped alias if D0 says so,
  registered with the same `auth → haltGate → inf → maxBytes` chain as its siblings
  (`internal/serveapp/main.go:633`). Batch goes through J1 admission and, when asked, the J3 job
  object, so a long batch is re-attachable.
- Response: `distribution`, `decision`, `confidence`, `latency_ms`, plus `model`, `route` (`label` |
  `head`), `backend`, and `calibrated` (false when no `calibration.json` was found — legal, but
  loud).
- `goinfer-chat -decide in.jsonl -o out.jsonl` shares the batch route's line format (the J5
  pattern).
- `/v1/models` advertises `decisions: {routes: [...], kinds: [...]}` per loaded model.
- **Acceptance:** jevx's scenario guide (https://muthuishere.github.io/jevx/guides/scenarios/) runs
  unchanged against a goinfer profile, and exit codes match on the non-borderline items.
  - **(D0, 2026-09-27)** The exit codes are 0 = yes/decided, 1 = no, 3 = unsure (noul between 0.2 and 0.8, or a
    choice/score below `min_confidence` 0.6) and 4 = error.
  - About nine scenarios depend on unpublished input files, so "runs unchanged" covers the inline ones.
  - jevx sends choice descriptions (`criteria`) and 2–10-level scores. bare-v1 has no place for either, and JEV's
    score is fixed at six levels.
  - TypeSafe's `GET /v1/models` is `{models:[…]}`, which conflicts with goinfer's OpenAI-shaped `/v1/models`. It
    matters only to a client calling `models.list()`; jevx and `systemOne()` do not.

### D6 — fidelity gates (two, in order)

- **(D0, 2026-09-27)** The authors' B0 report already measured Route A with the bare-v1 template on Qwen3.5-9B:
  - choice top-1 0.532 against JEV-9B's 0.898 (test_set_30k);
  - overall top-1 0.518 against 0.918 on OOD, with ECE 0.072 there.
  - Temperature cannot move top-1, so a bare-v1 Route A lands in the "more than 10 points behind" branch.
  - D6a remains worth running only for a goinfer template that could differ, such as a chat-templated prompt on an
    instruct model, and it must say which template it grades.
- **D6a (right after D1; decides D2–D4).** Route A on Qwen3.5-9B base, calibrated on the
  calibration split, against JEV-9B's reference outputs on D0's held-out items: top-1 agreement with
  gold where it exists, ECE, mean KL to the JEV-9B reference. Also run Route A on one small model
  (Qwen2.5-1.5B) to show what the any-model path gives.
  - Within 3 points of top-1 and ECE ≤ 0.05 → ship D1 + D5 only; park D2–D4 with this measurement
    as the reopen trigger.
  - More than 10 points behind → build D2–D4.
  - In between → his call, with the numbers on the page.

**D6a pre-registration (2026-09-28, committed before any graded run; the owner chose chat-v1 as the graded
template).** The artifacts are in `docs/measurements/decisions-d6a-2026-09-28/`: `select.py`, `analyze.py` and
`run-d6a.sh`, all written before any graded run.
- **Machine: nobara-pc, CUDA** (`docs/prompts/nobara-decisions-d6a-2026-09.md`). The Mac cannot run it.
  - The 9B fell back to the CPU there twice: first because the auto-pinned context exceeded Metal's 32,768
    ceiling (fixed by `--ctx 4096`), then because Metal's memory guard budget (5.4–5.8 GB of live memory against
    the 6.04 GB the resident needs) declined it.
  - The CPU path took 10–79 s a row, which puts the run at about 60 hours.
  - A 6-row smoke run on rows outside every graded sample was seen before this was written, and is disclosed:
    chat-v1 got 3/6 right and bare-v1 1/6. That decides nothing.
- **Model:** `unsloth/Qwen3.5-9B-GGUF` Q4_K_M (5,680,522,464 B; its base is `Qwen/Qwen3.5-9B`, JEV-9B's own base),
  from `~/models`, at the default quant with `--ctx 4096`. The run must show the resident decode path.
- **Arms** (the production CLI: `decisions-calibrate` on the calibration sample, then `decide` uncalibrated on the
  eval sample; `analyze.py` applies each arm's fitted T):
  - **A:** Qwen3.5-9B with **chat-v1**. Graded.
  - **B:** Qwen3.5-9B with **bare-v1**. The control: the authors' B0 is this computation in bf16. Its raw top-1 must
    land within 0.05 of B0's OOD 0.5180, or goinfer's Route A differs from theirs and arm A is not trusted until
    that is explained.
  - **C:** qwen2.5-coder-1.5b-instruct with chat-v1. Reported only ("what the any-model path gives").
- **Data** (`SargeDev/jev-distill-corpus-v3` @ `fc99c635…`; calibration.jsonl sha256 `c5e232a0…b068155`, ood.jsonl
  `70d0f017…885781`):
  - **calibration:** 1,500 rows, excluding `yuri_v1`'s placeholder rows, stratified by kind as the rest of the
    split is (noul 598, choice 468, score 434). Sample sha256 `0cde7143…4e15`.
  - **evaluation:** OOD gold rows: 1,000 noul, 1,000 choice and all 72 score. Sample sha256 `307fe998…07b3`.
  - Within each stratum, the rows with the lowest `sha256(id)`.
- **Metrics:**
  - top-1: argmax(p) = argmax(target);
  - ECE: top-label, over 15 equal-width bins;
  - noul AUROC (reported);
  - per kind, then combined with the OOD split's own kind proportions (9,767 / 3,219 / 72), so the overall figures
    compare with JEV-9B's published full-split OOD numbers (top-1 0.9181, ECE 0.0396, bf16).
  - The KL to the JEV-9B reference waits on the D0 fixture, and is reported when it exists. It does not decide.
- **The decision** is the rule above, on arm A calibrated:
  - top-1 ≥ 0.8881 and ECE ≤ 0.05 → Route A is enough: ship D1 + D5, park D2–D4;
  - top-1 < 0.8181 → build D2–D4;
  - between → the owner's call.
  - **This is not a paired comparison.** JEV-9B's figures are the authors' bf16 full-split run. A gap near an edge
    is read with that in mind, and the fixture's same-rows comparison settles it.
- **Parked here:** the Metal decline of an *auto-pinned* context above its ceiling. It declines to the CPU instead
  of clamping to 32,768, which is a real usability bug in its own right, and is fixed separately.
**D6a amendment (2026-09-28, owner; before any graded result).** The first nobara run (`5c85f7c0`) was stopped
~25 minutes into arm A's calibration. It had written no calibration and no evaluation, so no graded number
exists.
- **Why it stopped:** its estimate was ~6 h. CUDA prefills a Gated-DeltaNet prompt one token at a time (~14 ms per
  token). The owner: never a 6-hour run. Two changes:
  1. **Batched DeltaNet prefill on CUDA**, as its own gated task
     ([`task-cuda-deltanet-prefill-2026-09.md`](task-cuda-deltanet-prefill-2026-09.md)). D6a runs on the commit that
     ships it, named then.
  2. **Smaller samples**, from the same rule (lowest `sha256(id)` per stratum), so each is the head of the original:
     - calibration: 600 rows (noul 239 / choice 187 / score 174), sha256 `91b3cc2a…95cb`;
     - evaluation: 872 rows (noul 400 / choice 400 / all 72 score), sha256 `3b2f496f…69d8`.

     They come from `select.py <data> <out> 600 400`, run with `S=<that dir>`. The defaults still reproduce the
     original sample hashes.
- **Unchanged:** the arms, the metrics, the control and the decision rule.
- **What the smaller sample costs:** 1,472 prefills per arm instead of 3,572.
  - The kind-weighted top-1 is dominated by noul (weight 9,767 / 13,058), estimated here from 400 rows. Its standard
    error near p ≈ 0.9 is about ±1.5 points, against a 3-point band and a 10-point band. A result within ~1.5 points
    of an edge is therefore read as the owner's call, whichever side it lands on.
  - Calibration fits one temperature per arm, and 600 rows is ample for one parameter.
- **Runtime:** re-estimated from the resident check once the prefill ships. Over ~1 hour is a stop, not a start.

**D6a amendment 2 (2026-09-28, owner; before any graded result): the ECE bar is bias-adjusted.**
- **Why:** ECE is biased upward at small N, so a perfectly calibrated model does not read 0. TE3
  (`docs/measurements/noise-registry.md`) put that floor at ~0.02–0.05 for 400 / 400 / 72 rows, against a 0.05 bar.
  The bare bar could therefore fail a calibrated model on sampling noise alone.
- **The bar:** ECE ≤ √(0.05² + floor²). The floor is `ece_floor()` in
  `docs/measurements/decisions-d6a-2026-09-28/analyze.py`: it takes arm A's own calibrated confidences, draws each
  row correct with probability equal to its confidence, scores the draw with the same 15 bins and kind weights, and
  averages 2,000 draws (seed 0). The top-1 bands are unchanged.
- **Why quadrature and not 0.05 + floor.** The owner chose "0.05 + the floor". A simulation run before recording
  it showed that the linear form over-corrects:
  - Setup: 400 / 400 / 72 rows, 40 simulated runs per cell. The confidences are either concentrated near 0.9 or
    spread out, and the model is overconfident by a fixed true gap.

    | true gap | floor | bare 0.05 | 0.05 + floor | √(0.05² + floor²) |
    |---|---|---|---|---|
    | 0 (calibrated), concentrated | 0.026 | 40/40 | 40/40 | 40/40 |
    | 0 (calibrated), spread | 0.040 | 37/40 | 40/40 | 40/40 |
    | 0.05, concentrated | 0.026 | 13/40 | 37/40 | 22/40 |
    | 0.05, spread | 0.040 | 8/40 | 39/40 | 24/40 |
    | 0.08, concentrated | 0.026 | 0/40 | 12/40 | 0/40 |
    | 0.08, spread | 0.040 | 1/40 | 24/40 | 4/40 |

  - The linear form puts the effective bar near a true gap of 0.07. The quadrature form puts it at ~0.05, which is
    the bar's stated intent: a model right at the boundary passes about half the time. It still passes a calibrated
    model in every simulated run.
  - Sampling noise and a true gap combine roughly in quadrature inside |accuracy − confidence|, which is why
    quadrature lands on the intent.
- **Reported alongside:** the bare ECE, the floor and the bar, so the reading at the bare 0.05 remains visible.

- **D6b (Route B parity).** goinfer JEV-9B against the D0 goldens at f32, int8int8 and int4
  (default). Pre-registered: mean KL ≤ 0.01 at f32 and ≤ 0.03 at int4, top-1 agreement ≥ 98%.
  Report ECE per arm: quantization can hold top-1 and still shift calibration, which is the
  product's actual promise. If int4 fails calibration but passes top-1, the default for decision
  models is int8, recorded in the capability matrix, not in a comment.
- **Amendment, 2026-09-30, before any D6b run** (registered with its grader, [`decisions-d6b-2026-09/grade.py`](../measurements/decisions-d6b-2026-09/grade.py)):
  - **"Fails calibration" is made operational.** Calibration is the top-label ECE against gold on the 84 gold rows of
    the 150 items (15 equal-width bins), for each arm and for the reference on the same rows. An arm fails calibration
    if the 95% paired-bootstrap interval of (arm ECE − reference ECE) over those rows lies wholly above 0. An interval
    that reaches 0 is recorded as unresolved, not as a failure. At 84 rows the ECE alone cannot resolve small gaps, so
    the paired difference is what can.
  - **int8int8 is graded on int4's band** (mean KL ≤ 0.03, top-1 ≥ 98%), since it is the fallback default the rule
    names.
  - **Validity.** Every arm must answer all 150 items by route `head` with the reference's prompt token count, or the
    arm is void.
  - **It runs on the batched Qwen3.5 `PromptHidden`** (`runLayersQwen35N`, 7ba740fe), the path goinfer ships. Its CPU
    cost measured 65–66 ms/token at int4 and ~80 ms/token at f32, against the per-token path's ~0.19 / ~1.2 s, so the
    three arms are ~2 h in all.

### D7 — decisions vs constrained generation (speed)

- Projection, derived before measuring: `t_decision ≈ t_prefill(K) + t_head` against
  `t_constrained ≈ t_prefill(K) + n_out · t_decode`, where `n_out` is the token length of the
  shortest valid JSON answer under the tool-grammar union. Fill in `t_prefill` and `t_decode` from
  `docs/benchmarks.md` for 9B-class on each box at K ∈ {256, 1024, 4096} before running; record the
  band in the measurement file's header.
- Same model, same prompts, quiet box, paired differences. Report decisions/s at batch 1 and at
  saturating concurrency (MC1/MC3 slots where they exist).
- Not a kill gate: the calibrated distribution is the product whether or not it is faster. It is the
  number the docs and site quote, and it is quoted as measured.
- **Projection done, 2026-09-28** ([`decisions-d7-2026-09-28.md`](../measurements/decisions-d7-2026-09-28.md)),
  for nobara CUDA on Qwen3.5-9B, the only machine with figures on record.
  - One question: the decision is faster by 1.11 / 1.03 / 1.01× than a schema answer at K = 256 / 1024 / 4096, and
    by 1.9–2.0 / 1.2 / 1.06× than a tool call.
  - Five questions about one state: decisions are **3.2–4.9× slower**, because each question pays a full prefill on
    qwen3_5.
  - The Mac cannot run the 9B resident.
  - **Measurement pre-registered, 2026-09-28** (the record's §4): nobara CUDA, batch 1 (qwen3_5 serves one slot),
    8 states per K, paired ratios against the band, and D8's trigger as a state-prefill share ≥ 0.70 at K = 1024.
    It is about 35 min and not queued; it needs `d7_bench.py` built and smoke-tested first.

### D8 — shared state, many questions (deferred)

TypeSafe's shape asks many questions about one state. On `qwen3_5` each question costs a full
prefill of the state today (§3). The fix is an in-memory recurrent-state checkpoint (conv window +
GDN state at a position, copied rather than rewound): the "state checkpoints" track
[`../completed/qwen3_5_moe.md`](../completed/qwen3_5_moe.md) already defers, which would also unlock
prefix reuse and spec decode on every hybrid family.

**Trigger:** D7 shows state prefill ≥ 70% of a multi-question request's time on a realistic
workload. Until then, the template puts the question after the state so it is at least a clean
suffix, and the cost is documented.

### D9 — docs for D

**Done 2026-09-28, except the parts that wait on D6a or D7:**
- **`docs/server.md`:** the Decisions section, with the §2 caveat, came with D5. It now also says that many
  questions about one state cost a full prefill each.
- **README:** the Decisions paragraph, with no speed claim.
- **`docs/README.md`:** the index entry updated.
- **The site:** [`task-site-2026-09.md`](task-site-2026-09.md) S2 gains a decisions row on each model page, and a
  decision-model tag reserved for Route B.
- **The recipe:** [`../integrations/typesafe-jevx.md`](../integrations/typesafe-jevx.md), for jevx and TypeSafe's
  SDKs. It is marked not yet run end to end.
- **The capability matrix: deferred to D2–D4.**
  - Route A (label scoring) is the same on every family, so a per-family column would be a constant.
  - Whether a checkpoint's tokenizer has the single-token labels is a property of the checkpoint, not the registry
    the matrix is generated from, and `New` refuses at load when it does not.
  - Routes differ per family only once a trained head exists.
- **Waits on:** D6a's result for the route recommendation, and D7's measurement for any speed claim.

`docs/server.md` (a Decisions section with the §2 caveat), README (one paragraph; no speed claim
unless D7 is in), the capability matrix (decision routes per family),
[`task-site-2026-09.md`](task-site-2026-09.md) (a models-page tag for decision models), and a recipe
under `docs/integrations/` if D0 found the SDK base-URL override or when jevx acceptance passes.

---

## 5. Pre-registered gates for C (D's are in D6)

- **C-gate 0 — enough free tokens.** Before measuring, state a per-field-kind threshold for the
  fraction of non-forced tokens below which that kind's confidence is mostly schema artefact and is
  not surfaced. Fails for every kind → close C; D is unaffected.
- **C-gate 1 — cost.** Opt-in cost below a stated percentage of a token; zero when disabled,
  verified.
- **C-gate 2 — it discriminates.** On a hand-labelled set (D0's gold-labelled items for enum and
  boolean fields, plus a small hand-built set for free strings and numbers), low-confidence answers
  are wrong more often than high-confidence ones. Directional only, not a calibration claim; state
  the threshold in advance. If confidence does not correlate with correctness for a field kind, that
  kind does not ship.

Ambiguous → parked.

## 6. Order

**D0 → C0 ∥ D1 → D6a** (C0 and D1 build the same readout helper, so they share a session) →
C1 if C's gates clear; [D2 → D3 → D4 → D6b] if D6a says so → D5 → C2, D9 → D7. D8 waits on its
trigger. D5 can land after D1 alone if D6a says Route A is enough.

## 7. Not in scope, stated

- **Compute-time LoRA for own-forward families** (one loaded model serving generation and
  decisions). The right eventual shape; a separate, larger task touching every resident backend's
  `SetAdapter`.
- **Training or distilling** a decision head or a calibration model. goinfer runs published heads
  and fits one scalar temperature per kind.
- **Calibrating free-string or numeric fields.** Raw, documented probabilities only.
- **Open-ended "decisions"** (free text, unbounded option sets). That is generation, and the
  tool-grammar union covers structured generation.
- **Claiming parity with TypeSafe's hosted Jev.** goinfer is measured only against the open
  students and against gold labels; any comparison with the hosted model goes through autotrust's
  published side-by-side, cited as theirs.

## Constraints

- C0 and D6a are measurements. No API design in either.
- `GrammarFromStruct`, `JSONSchema`, `NewMasker`, `Process` and the OpenAI-shaped responses are
  unchanged for callers who do not opt in.
- No cgo.
- Quiet box for every cost and speed number; report spread.

## Sources

`constrain/constrain.go:98`, `:147`, `:166`, `:208` (`ForcedRun`, `MaskAt`, `ForcedBytesRun`,
`Process`) · `decoder/model.go:1292` (`ForwardCapture`) · `decoder/capture.go:14` (the capture
contract) · `decoder/arch.go:954` (the `qwen3_5` / `qwen3_5_moe` own-forward row) ·
`decoder/arch.go:368` (`FusedDeltaNetProj`) · `decoder/lora.go:144` (`validateTargets`) ·
`decoder/lora.go:353` (`LoadAdapter` refuses own-forward) · `decoder/weights.go:695`, `:744`
(merge-at-load) · `decoder/kvcache.go:540` (`TruncateTo`) · `decoder/kvsnapshot.go:62` (snapshot
skips recurrent state) · `internal/serveapp/openai.go:34`, `:536`, `:538` (`top_logprobs` cap,
`logprobs`, `response_format`) · `internal/serveapp/main.go:633` (route middleware) ·
`docs/spec/10-optfwd-gate.md:177` (sampler share) ·
[autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) (adapter, head, calibration, API) ·
[autotrust/JEV](https://huggingface.co/autotrust/JEV) · [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) ·
[Jev (AI model), Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model)) (unpublished weights/architecture) ·
[TechTarget, 2026-09-22](https://www.techtarget.com/it-infrastructure/news/366650696/Jev-decision-model-touted-as-quicker-cheaper-LLM-alternative) (launch, Vercel adoption figure) ·
[awesome-typesafe-jev](https://github.com/valentynkit/awesome-jev-typesafe/wiki) (API endpoint, gateways, SDKs) ·
[kyegomez/open-jev](https://github.com/kyegomez/open-jev) (random-weights reconstruction) ·
[muthuishere/jevx](https://github.com/muthuishere/jevx) (CLI client)

<!-- doc-reviewed: 2026-09-27 -->

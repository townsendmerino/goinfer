# Task: confidence — per-field probabilities on constrained output, and a typed `/v1/decisions` endpoint (C0–C2, D0–D9) — 2026-09

> **Status: SCOPED, nothing built.** Two measurements decide what gets built: **C0** (does a
> per-field number mean anything?) and **D6a** (is label scoring on any model good enough, or do the
> trained decision heads earn their build?). Everything else waits on one of them.
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
> `response_format: {"type": "json_schema"}` on the server (`internal/serveapp/openai.go:538`). C1
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
| Where the probabilities come from | the masked logits `constrain` already computes (`MaskAt` at `constrain/constrain.go:147`, `Process` at `:208`) and discards after sampling | the label tokens' logits at the last prompt position (Route A), or a trained head over the final hidden state (Route B) |

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
- `MaskAt` (`constrain/constrain.go:147`) and `Process` (`:208`) walk the full logit vector at
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
  (`decoder/model.go:1288`) returns logits plus captured residuals. `qwen3_5` dense shares
  `qwen3_5_moe`'s own-forward row, which has Captures = true and Recurrent = true
  (`decoder/arch.go:954`; the predicate is `a.qwen35 != nil`, so it matches both). The capture
  contract (`decoder/capture.go:14`) is the residual *after* layer l, before the final norm, so D2
  applies the final RMSNorm itself and must match HF's `hidden_states[-1]`, which is already normed
  on Qwen. Verify in D2; do not assume.
- **LoRA cannot reach this family.** Both paths are closed:
  - Merge-at-load (`--lora`): `validateTargets` (`decoder/lora.go:137`, called at
    `decoder/weights.go:682`) knows only the Q/K/V/O/gate/up/down suffixes, so an adapter that also
    targets the GDN projections is refused whole. The merge itself runs inside the `loadProj`
    closure (`decoder/weights.go:744`), and the GDN projections are loaded outside it.
  - Compute-time (`--adapter`): `LoadAdapter` (`decoder/lora.go:309`) refuses every own-forward
    family by design.
- **Recurrent state cannot be rewound.** `KVCache.TruncateTo` (`decoder/kvcache.go:540`) reports
  inexact on any partial rewind when the model has recurrent state, and `.giw-kv` snapshots skip
  recurrent state (`decoder/kvsnapshot.go:62`). So "prefill the shared state once, branch per
  question" is not available on `qwen3_5` today (D8).
- **Route A is approximable from outside already.** `/v1/completions` with `max_tokens: 1,
  logprobs: true, top_logprobs: 20` (`internal/serveapp/openai.go:536`, cap at `:33`) gives a client
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

### C1 — capture, attribute, surface (only if C0's gates clear)

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

`docs/server.md` (the opt-in and the §2 caveat), the README's "A Go struct the model cannot
violate" section (one paragraph, the caveat linked), and a worked example in `examples/`.

### D0 — prior art, reference fixture, API shape (no code in goinfer)

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
  exit codes map confidence to shell control flow (≥0.6 → 0, `unsure` → 3, error → 4). It talks to
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

- A `decider` in `internal/serveapp` (not `decoder`). Build a prompt from `{state, question,
  options}` with a fixed template; label options A, B, C… so each label is one token in the model's
  vocab (check per tokenizer; fall back to digits). Disable thinking in the chat template. One
  prefill; take the last-position logits; read the label tokens; log-softmax over just those; apply
  the per-kind temperature. This is the shared readout helper C1 also uses.
- `noul` uses two labels in yes/no form; `score` uses six labels 0–5 and returns the distribution
  plus the expected score.
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

### D3 — LoRA merge-at-load for the GDN projections (Route B)

- Extend the `qwen3_5` loader so `in_proj_qkv`, `in_proj_z` and the linear-attention `out_proj`
  pass through the same f32 merge as `loadProj` before quantization. Extend `validateTargets` with
  the GDN suffixes *from the schema*, not a hand list (lora.go's own V-12 comment records why). Mind
  `FusedDeltaNetProj` (`decoder/arch.go:368`): the merge has to hit the on-disk tensor before any
  repack.
- **Consequence, stated:** a merged model is a decision model; its text generation is no longer the
  base Qwen's. One set of weights serving both is out of scope (§6).
- Offer `goinfer transcode --lora` → `.giw` so the whole-model f32 merge is paid once, not per load.
  On the 16 GB Mac a 27B merge-at-load in the Go heap is exactly the anonymous-memory spike
  never-swap S1 exists to avoid. JEV-9B is the Mac target; JEV-27B is the Linux box target.

### D4 — readout head + calibration (Route B)

- autotrust's recipe: a LoRA on the backbone (r=16, α=32, targets `in_proj_qkv, in_proj_z,
  out_proj, q/k/v/o_proj, gate/up/down_proj`, 416 MB unmerged); the last prompt token's final-norm
  hidden state through an fp32 linear head `H → 24 slots` (`head.safetensors`: noul 0–1, score 2–7,
  choice 8–23); a per-kind temperature (`calibration.json`: noul 1.014, choice 1.016, score 1.004);
  softmax over the slots the request's options occupy. Published: JEV-27B mean KL 0.104 on the
  Open-Jev OOD split, JEV-9B 0.234.
- Load `head.safetensors` and `judge_config.json` (slot layout, verbalizer ids, template version,
  `weights_mode`). Refuse `weights_mode` values other than `unmerged`/`merged` with a clear error.
- Slot selection per kind as above; confirm in D0 that masking unused choice slots before the
  softmax is what the reference does.
- The template must be byte-identical to the reference's version. Pin it, and refuse a
  `judge_config.json` whose version the loader does not know.

### D5 — the endpoint

- `POST /v1/decisions` (+ `:batch`, ≤256 items) and the TypeSafe-shaped alias if D0 says so,
  registered with the same `auth → haltGate → inf → maxBytes` chain as its siblings
  (`internal/serveapp/main.go:548`). Batch goes through J1 admission and, when asked, the J3 job
  object, so a long batch is re-attachable.
- Response: `distribution`, `decision`, `confidence`, `latency_ms`, plus `model`, `route` (`label` |
  `head`), `backend`, and `calibrated` (false when no `calibration.json` was found — legal, but
  loud).
- `goinfer-chat -decide in.jsonl -o out.jsonl` shares the batch route's line format (the J5
  pattern).
- `/v1/models` advertises `decisions: {routes: [...], kinds: [...]}` per loaded model.
- **Acceptance:** jevx's scenario guide (https://muthuishere.github.io/jevx/guides/scenarios/) runs
  unchanged against a goinfer profile, and exit codes match on the non-borderline items.

### D6 — fidelity gates (two, in order)

- **D6a (right after D1; decides D2–D4).** Route A on Qwen3.5-9B base, calibrated on the
  calibration split, against JEV-9B's reference outputs on D0's held-out items: top-1 agreement with
  gold where it exists, ECE, mean KL to the JEV-9B reference. Also run Route A on one small model
  (Qwen2.5-1.5B) to show what the any-model path gives.
  - Within 3 points of top-1 and ECE ≤ 0.05 → ship D1 + D5 only; park D2–D4 with this measurement
    as the reopen trigger.
  - More than 10 points behind → build D2–D4.
  - In between → his call, with the numbers on the page.
- **D6b (Route B parity).** goinfer JEV-9B against the D0 goldens at f32, int8int8 and int4
  (default). Pre-registered: mean KL ≤ 0.01 at f32 and ≤ 0.03 at int4, top-1 agreement ≥ 98%.
  Report ECE per arm: quantization can hold top-1 and still shift calibration, which is the
  product's actual promise. If int4 fails calibration but passes top-1, the default for decision
  models is int8, recorded in the capability matrix, not in a comment.

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

`constrain/constrain.go:97`, `:147`, `:166`, `:208` (`ForcedRun`, `MaskAt`, `ForcedBytesRun`,
`Process`) · `decoder/model.go:1288` (`ForwardCapture`) · `decoder/capture.go:14` (the capture
contract) · `decoder/arch.go:954` (the `qwen3_5` / `qwen3_5_moe` own-forward row) ·
`decoder/arch.go:368` (`FusedDeltaNetProj`) · `decoder/lora.go:137` (`validateTargets`) ·
`decoder/lora.go:309` (`LoadAdapter` refuses own-forward) · `decoder/weights.go:682`, `:744`
(merge-at-load) · `decoder/kvcache.go:540` (`TruncateTo`) · `decoder/kvsnapshot.go:62` (snapshot
skips recurrent state) · `internal/serveapp/openai.go:33`, `:536`, `:538` (`top_logprobs` cap,
`logprobs`, `response_format`) · `internal/serveapp/main.go:548` (route middleware) ·
`docs/spec/10-optfwd-gate.md:177` (sampler share) ·
[autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) (adapter, head, calibration, API) ·
[autotrust/JEV](https://huggingface.co/autotrust/JEV) · [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) ·
[Jev (AI model), Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model)) (unpublished weights/architecture) ·
[TechTarget, 2026-09-22](https://www.techtarget.com/it-infrastructure/news/366650696/Jev-decision-model-touted-as-quicker-cheaper-LLM-alternative) (launch, Vercel adoption figure) ·
[awesome-typesafe-jev](https://github.com/valentynkit/awesome-jev-typesafe/wiki) (API endpoint, gateways, SDKs) ·
[kyegomez/open-jev](https://github.com/kyegomez/open-jev) (random-weights reconstruction) ·
[muthuishere/jevx](https://github.com/muthuishere/jevx) (CLI client)

<!-- doc-reviewed: 2026-09-27 -->

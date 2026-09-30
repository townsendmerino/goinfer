# Task: Qwen3.5's generation prompt — serve renders it differently from the model's own template (2026-09)

> **Status 2026-09-30: phases 1, 2 and 3 DECIDED AND BUILT — serve's default is `-thinking template`** (owner decision
> 2026-09-30: each model follows its own chat template's default). Found by P8a gate G4
> (`docs/measurements/p8a-qwen35-vl-2026-09/g1-g4-results.md`, finding 1), widened the same day when the 9B leg of P8a died
> on it: **the 0.8B and the 9B disagree about the default, so the G4 write-up's "Qwen3.5 defaults to thinking OFF" is true of
> one size only.** What is built is in "Built" below; what is not is in "Not built". `-thinking asis` restores the old bytes.
>
> **Measured on the way:** the 9B, given a prompt *without* its open `<think>\n`, writes `<think>\n` itself as its first two
> tokens and then continues identically to the with-opener run (question 1, three images, 32 tokens). Before this work
> serve had no think-block handling at all, so that self-written opener and the whole reasoning block reached the client as
> ordinary `content`. The P8a pin script's 0.8B-only assertion that killed the 9B night job is fixed and verified on the 9B.

## Built (2026-09-30)

What shipped, against the design in "Proposed fix" below (which is kept as the record of the reasoning):

- **`chat/reasoning.go` — the per-family spec.** `chat.Template` carries a `Reasoning` read from the checkpoint's own template
  text (Qwen3: default on, nothing written; Qwen3.5-0.8B: default off, closed block; Qwen3.5-9B / JEV-9B: default on, open
  block; Gemma 4: default off, `<|think|>` system marker when on). `WithThinking(ThinkAsIs|Template|On|Off)` renders each; a
  template control that is not one of those shapes gets no spec — no prefill, no splitting. `ThinkSplitter` is the output
  half, chunk-boundary safe. **Gate:** 36 text prompts byte-equal to HuggingFace in every mode, token ids equal against the
  real tokenizers, Gemma 4 tool prompts byte-equal (`testdata/chat_think_goldens/think_modes.json`, made by
  `scripts/pin_chat_think_modes.py`); the splitter is property-tested over every split point and random partitions. Nine
  mutations (9B default flipped, trailing-newline trim, partial-close hold, re-entry from content, splitter bypassed, flush
  dropped, effort mapping, constrained-template rule, legacy raw) each turn a gate red.
- **`internal/serveapp/think.go` — the wire half.** One splitter wrap inside `streamTokens`, so every route gets clean
  `content` from one place: OpenAI chat (`reasoning_content`, streamed, all before content), tools, vision, Anthropic
  (`thinking` blocks with `signature_delta`, block indices shifted, only when the request asked), Responses (reasoning
  dropped), jobs (reasoning in the event log, not the result) and batches. Controls: `chat_template_kwargs.enable_thinking`,
  `reasoning_effort`, `reasoning_format`, Anthropic `thinking`; flags `-thinking` (default `asis` = today's prompt bytes)
  and `-reasoning-format` (default `deepseek`; `none` restores the old raw output). count_tokens renders exactly as
  generation does. The web UI stitches `reasoning_content` back into the `<think>` form its own fold already parses.
- **Constrained requests render thinking-off** (`response_format` json, a forced or lone tool, `required`): the grammar
  governs token 1, so a prompt ending inside an open block would contradict it. The lazy `auto` tool union is not
  constrained this way; a model that quotes the tool opener inside its reasoning can arm it early (hazard kept).
- **Deviation from the design: `reasoning_effort` can only turn thinking OFF.** `none` → off; any other value changes
  nothing. The design mapped every non-`none` value to on. `docs/server.md` records that the DeepSeek Harness sends a bare
  `reasoning_effort` to every endpoint it does not recognise; reading it as "thinking on" would change that client's prompts
  and, at a small `max_tokens`, hand it an empty answer. A client that wants thinking says so in `enable_thinking`.
- **Found and fixed on the way:** `renderGemma4Tools` wrote a newline between a system prompt and the first declaration that
  Gemma's template does not; the old tool goldens had no system-prompt case. The per-checkpoint goldens found it.

- **The reasoning budget (built 2026-09-30, owner request)** closes the truncation hole: a gated logit processor
  (`internal/serveapp/budget.go`) force-closes the think block once it has used its budget (request's own, else
  `-reasoning-budget`, else `auto` = three quarters of `max_tokens`), so a thinking reply always has room to answer. Gated:
  fast paths are untouched until due. Composes with the lazy tool-union masker; steps aside under `-spec`/`-drafter` and for
  grammar-constrained requests. Gates: pure-function table, the real decode loop on the tiny model (gated and ungated;
  identical to an unprocessed run up to the budget, close token forced at exactly the budget, never forced twice), composition
  and skip rules, six mutations each red. On the real 0.8B: thinking on at `max_tokens` 160 ends with an answer; a request
  `thinking_token_budget` of 16 bounds the reasoning and leaves an answer; Anthropic `budget_tokens` gives a thinking block then
  a text block; streaming equals non-streaming.
- **History handling (built 2026-09-30, owner request).** A client's replayed reasoning is rendered the way the model's own template
  would: kept for the turns of the tool loop in progress, dropped before the last user query (`chat/history.go`). Per family, read
  from the real templates: Qwen3.5 always writes the `<think>` block for those turns; Qwen3 only for the last message or a turn with
  reasoning; Gemma 4 a thought channel when there is reasoning and strips channel spans from model turns. `<think>…</think>` left in
  content is extracted as the Qwen templates do. Inputs: OpenAI `reasoning_content` / `reasoning`, Anthropic `thinking` blocks
  (`Turn.Reasoning`). **Gate:** 144 HuggingFace prompts byte-equal per checkpoint and mode (plain conversations for all three;
  tool loops byte-exact for Qwen3 — the JSON inside `<tool_call>` compared structurally — and Gemma 4, through the assistant turn's
  first call for Qwen3.5); `-thinking asis` asserted unchanged by replayed reasoning for every case; nine mutations each red.
  - **Deviation, deliberate: `Turn.ToolLoop`.** Claude Code puts reminder text inside the same user message as a `tool_result`.
    Applied literally, Qwen's rule reads that text as a new query and strips the reasoning of the turns just before it — on the turn
    right after the first tool result, for exactly the client this matters most for. Such text is marked as a loop continuation and
    does not move the "last query". An OpenAI client cannot produce this shape (its tool results are separate `tool` messages).
  - **Found by the new goldens:** Gemma 4 with thinking on ends a prompt inside an OPEN thought channel after a tool response, so a
    reply starts mid-reasoning; serve's splitter would have put that reasoning in `content`. "The prompt ends inside an open block"
    is now per conversation (`Template.PromptOpensThinkFor`); the reasoning budget uses it too.
  - **Not replicated, recorded:** the templates trim every message's content (user and system too), which goinfer's renderers
    have never done; Gemma 4's `preserve_thinking`. **Pre-existing Gemma 4 renderer differences, unrelated to reasoning, found and
    skipped by name in the gate:** consecutive assistant messages share one model turn in Gemma's template (goinfer opens one per
    message; serve merges adjacent turns first, so it is unreachable there), and Gemma writes an assistant message's TEXT after its
    tool responses where goinfer's tool renderer writes it before the calls.
  - **Consequence for the KV cache:** when a new user query arrives the previous loop's turns lose their reasoning, so a cached
    prefix is reusable only up to the first such turn — the templates' own design.
- **Stop strings on the answer only (built 2026-09-30, owner request).** `streamTokens` now takes the reasoning out BEFORE the stop
  logic (`thinkOut.feed`), so `firstStop` / `stopTailHold` / the UTF-8 holdback see the answer alone; the reasoning goes to its own
  callback with its own UTF-8 carry (a multi-byte character split across two tokens reaches the client whole). The splitter's
  end-of-reply flush runs through the stop logic once more. deepseek-legacy (content is the raw text) and `-reasoning-format none`
  keep watching the raw text by design. Replaces the pinned "KNOWN LIMITATION" test. Gates: stop in the answer / in the reasoning /
  in both / split across the boundary / inside an unfinished block, both tag encodings; legacy and none unchanged; rune-split
  reasoning; the existing differential stop-string test against the pre-R-08 algorithm still passes unchanged; four mutations red.
- **`goinfer-chat` and the demo agent (built 2026-09-30, owner request).** The splitter's UTF-8 handling and the budget moved into the
  public `chat` package (`ReplySplitter`, `ReasoningBudget`, `BudgetRoom`, `Template.NewReasoningBudgetFor`) so serve, the CLI and the
  agent share one copy of "where reasoning ends" and "how much of a turn thinking may take". CLI: `--thinking` / `/think`,
  `--show-thinking`, reasoning dimmed before a cyan answer, history holds the answer alone, budget except under JSON / `--draft` /
  `--spec ngram`. Agent: `Options.Thinking` + `-thinking`, DECIDE always thinking-off (its grammar governs token 1), answer through the
  splitter into an optional `Events.Reasoning`, budget on the answer phase only. Gates: separation with rune splits in all three
  consumers, per-mode prompt tails, where the budget installs and where it must not, seven mutations red; the real `goinfer-chat`
  binary on the 0.8B prints a thinking run dimmed then the answer, and the answer-only default for that model.
  - **Known build caveat:** `demo/agent` is its own module pinned to the released goinfer (v0.19.0); in the workspace (CI's build) it
    uses the new `chat` API, but a standalone `GOWORK=off` build resolves the old release and fails until the next release — the
    between-releases state CLAUDE.md records.
- **Claude Code itself, tested 2026-09-30** (2.1.284, isolated config dir, dummy key, tools off and with `Read`, through a
  logging proxy; the real 0.8B served under a Claude model name): it sent `thinking: {type: enabled, budget_tokens}`, parsed the
  `thinking_delta` stream (its own thinking-token counter climbed), accepted the empty `signature`, printed the answer, and
  exited 0 (`is_error: false`). In a tool loop it replayed our thinking blocks in history with `signature: ""`; serve dropped
  them and answered 200 on every turn. Executing the tool took ~0.1 s; the 125 s was three ~40 s model turns on CPU. What went
  wrong was the 0.8B's own tool-call quality (a wrong path, junk arguments, and one call written as text with broken JSON,
  which serve returns as prose) — not the protocol. Not tested: a model that calls tools reliably (the 9B), which would make the
  loop meaningful.

**Verified against real checkpoints** (client matrix, `scripts/think_matrix.py`, exploratory daytime run on Qwen3.5-0.8B; the
9B is a night job). See the record in "Matrix results" below for what passed, what did not, and what was not exercised.

## Matrix results (exploratory, daytime, Qwen3.5-0.8B int4 CPU, 2026-09-30)

`scripts/think_matrix.py` against a running `serve` (logs kept in `~/goinfer-logs/think/`). Not a measurement of speed; a
check that the invariant holds on every client's request shape. **Final run: 37 of 37 cells pass, 1 not exercised.**

- **Passed, and what it proves on a real model:** OpenAI chat with thinking unset / off / on, streaming vs non-streaming
  byte-identical at temperature 0; all reasoning deltas before all content deltas; usage identical between the two; the
  prompt half end to end (`prompt_tokens` moves by exactly +4 for off and +2 for on against the as-is prompt — the closed
  block and the open `<think>\n`); a reply truncated inside the block (empty `content`, `reasoning_content` present,
  `finish_reason` length); `reasoning_format` none (no reasoning field), deepseek-legacy, and a 400 on a bad value;
  `reasoning_effort: none` (no reasoning) and `high` (prompt unchanged); tools under each mode; Anthropic with `thinking`
  absent / enabled / disabled, streaming and not (block order, contiguous indices, every block opened and closed, a
  `signature_delta` before a thinking block stops, no thinking block unless asked, no markup in text); Anthropic tools;
  `count_tokens` equal to the generation's `input_tokens` in all three settings; the Responses API with effort unset / none /
  high.
- **NOT exercised: a reply that thinks, closes the block, and then answers, on a real model.** The 0.8B never finishes
  thinking under greedy decoding — two probes at `max_tokens` 3000 (`Say hello.`, `Reply with only the word: yes`) ended
  `finish_reason length` with 17,580 and 10,986 characters of reasoning and no content — and the sampled, seeded retry inside
  the matrix was still thinking at its 600-token limit (1,587 characters of reasoning, no content) — a short limit, so that
  run says less than the 3000-token probes do. That path is covered by the unit tests (synthetic vocabulary, both "tag token" and "tag
  spelled out" encodings) and by `TestThinkModes_delimitersDecodeToTheirSurfaceForm`, which checks against the real Qwen3 /
  Qwen3.5 / Gemma 4 tokenizers that the delimiter tokens decode to their literal text. It has not run end to end on a model
  that closes its block. The matrix now reports this state as NOT EXERCISED (exit status 2) instead of passing or failing it,
  and tonight's job runs it on the 9B, Qwen3-1.7B and Gemma-4-E2B.
- **Not covered at all:** Claude Code itself (the empty `signature`), vision routes with thinking, jobs and batches with
  thinking on a real model, any concurrent-request interaction (the K1 progress counter now also counts reasoning
  fragments; unit-untested).
- **How the first runs went:** run 1 stopped on a 503 when another session's 52 GB probe pushed the box into swap and
  serve's own swap guard refused requests (the matrix now retries a 503); run 2's completion cell failed for the reason above
  (a test-design error, not a server defect) and was reworked.

## What flipping the default changes (and a correction about G3)

**Correction, 2026-09-30.** Earlier text in this record and in the hand-off said a default flip "moves G3". That was
imprecise. G3 as registered (`docs/measurements/p8a-qwen35-vl-2026-09/preregistration.md`, `TestQwen35_textIdentityHashes`)
hashes raw f32 logits for **hard-coded token ids** fed straight to the model on two tiny fixtures and the real 0.8B. No chat
template is involved, so changing the renderer's default cannot fail that test. What G3 stood for in P8a was a *scope
promise* — "the image work must not change a Qwen3.5 text turn" — and that promise ended when P8a did. The finding that
deferred this fix ("doing so alters every Qwen3.5 text turn, which G3 says must not move") was about that promise, not about a
test that would have gone red.

**What a flip to `-thinking off` or `template` does change:**

1. **Every client of a recognised model gets a different prompt and a different model behaviour.**

   | model | today (`asis`) | `-thinking template` | `-thinking off` |
   |---|---|---|---|
   | Qwen3.5-0.8B | prompt without the block (22 tokens in the matrix prompt); the model answered directly | + closed block (26); no thinking — same as HF's default | same as `template` |
   | Qwen3.5-9B | prompt without the block; the model writes `<think>\n` itself | + open `<think>\n` (+2); thinks, as HF's default does | + closed block (+4); **stops thinking by default** |
   | Qwen3 (1.7B/4B/30B-A3B) | nothing written; the model decides | nothing written (its template default is on) | + closed block (+4); stops thinking |
   | Gemma 4 | closed scaffold = its default, no change | no change | no change |

   Prompt-token deltas are from the 0.8B matrix run and the per-checkpoint goldens. `off` changes the 9B and Qwen3 most: shorter,
   cheaper replies, and no empty-`content`-at-small-`max_tokens` failure — but also weaker answers on prompts that benefit from
   reasoning, and behaviour that differs from what those models' own cards call the default. `template` matches each model's
   own default exactly (so it fixes nothing for the 9B's truncation hole, and fixes the 0.8B's mismatch with HF).
2. **One test must be rewritten on purpose:** `TestServe_qwen35Image_G4` pins serve's prompt as "HF's default ids minus a
   4-token think block". Under `template` serve's ids equal HF's, so that assertion fails by design and becomes "equal".
3. **Comparability of anything measured through serve's chat endpoint on these families.** A flip changes the rendered
   prompt and, for the 9B and Qwen3, how many tokens a reply takes. I found no served benchmark row for Qwen3 / Qwen3.5 / Gemma 4
   in `docs/benchmarks.md` (six mentions of those names, all in family lists or notes — found by name, not read row by row), and
   the decisions work (`internal/decide`) builds its own prompt suffix and does not go through serve's renderer. So nothing
   published is invalidated today; any *future* row is comparable only within one default, which the row's provenance must name.
4. **It is one flag and reversible.** `-thinking asis` restores today's bytes exactly; a request can still override either way.

## Not built

- **Harmony (gpt-oss)** needs its own parser (several channel messages per reply); the interface admits one, none written.
- **Qwen3.5's XML tool-call format** (separate task), and what signature Claude Code wants on a thinking block — still to
  settle with the outstanding manual Claude Code smoke test.

## The defect

serve renders every Qwen family through the generic ChatML renderer (`chat/templates.go`, `ChatML()`), whose generation
prompt is `<|im_start|>assistant\n` and nothing more. Qwen3.5's own `chat_template.jinja` appends a think block after that
line, and **which block depends on the checkpoint**. The two files differ in one conditional
(`diff ~/models/qwen3.5-0.8b/chat_template.jinja ~/models/qwen3.5-9b/chat_template.jinja`, lines 149-152):

| checkpoint | template default | generation prompt ends with | `enable_thinking` flips it |
|---|---|---|---|
| Qwen3.5-0.8B | thinking **OFF** | `<think>\n\n</think>\n\n` (empty block, closed) | `true` → open `<think>\n` |
| Qwen3.5-9B | thinking **ON** | `<think>\n` (open) | `false` → the empty closed block |

(Read from `tokenizer_config.json` / `chat_template.jinja` of the checkpoints on this box at the revisions P8a pinned; the 9B
is `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Other Qwen3.5 sizes (2B, 4B, 27B, the MoEs) were **not** read — do not assume
a rule by size; read each template.)

So serve's prompt differs from HF's default in *different* ways per size:

- **0.8B:** serve is 4 tokens short (the empty block). HF's default answers with thinking suppressed; serve leaves the
  model free to start its own reasoning. Measured in G4 (`TestServe_qwen35Image_G4` pins the 4-token delta as a named
  difference).
- **9B:** serve is short by the open `<think>\n` (2 tokens: `<think>`, `\n`). The model writes the opener itself at step 0 (question 1, measured), so the
  difference shows up as a literal `<think>\n` at the start of serve's reply text.

The gap applies to **every Qwen3.5 text turn through serve**, not only image turns, and to any other caller of the ChatML
renderer for these checkpoints. `enable_thinking` is not plumbed anywhere in product code (`grep enable_thinking` in `*.go`
finds only test comments), so a caller who wants HF's `enable_thinking=false` behaviour has no way to ask for it.

## Why it was not fixed in P8a

Changing the renderer alters the prompt of every Qwen3.5 text turn. P8a's gate G3 says the text path must not move
(baseline `docs/measurements/p8a-qwen35-vl-2026-09/g3-baseline-0a3f0a7f.json`), and a prompt change fails it by
construction. The P8a goldens (`golden_*` = the model's own template; `serve_*` = the same minus the think opener) therefore
carry both variants, and G2 compares against `golden_*`, so the vision gate does not depend on this decision.

## Questions, in the order they unblock each other

1. **What does the 9B do when the open `<think>\n` is missing? — ANSWERED 2026-09-30 (HF f32 greedy, 9B at the pinned
   revision, three grid-aligned images, 32 tokens, log `~/goinfer-logs/p8a/pin-9b-goldens-2026-09-30.log`).** It writes the
   opener itself: on all three images the `serve_*` variant's first two generated tokens are `<think>` (id 248068) and
   `\n` (id 198), and its remaining 30 tokens equal the `golden_*` variant's first 30 exactly. The prompts differ by
   exactly 2 tokens (82/80, 114/112, 178/176). Min top-1/top-2 gaps 0.063-0.189 in both variants, so none of this is a
   near-tie. So for this prompt the 9B is not run in a mode HF's default never exercises — the missing opener is
   regenerated at step 0 — but **the generated text of a serve reply starts with a literal `<think>\n`**, where an HF reply
   built from the model's template does not (the opener lives in the prompt there). That moves the weight onto question 3.
   Caveats: an image prompt, three images, 32 tokens (all inside the think block); says nothing about long replies,
   text-only turns, tool calls, or sampling at temperature > 0. The 0.8B's side (closed empty block missing) is the
   G4-measured delta and was not re-tested here.
2. **What should the contract be?** Options, none chosen:
   a. *Follow the checkpoint's own template default* — render the generation prompt from the model's `chat_template`
      (or detect its `enable_thinking` default), so each size gets what HF gives it. Closest to "the model's own template"
      but means the default differs by checkpoint.
   b. *One fixed serve policy for the family* (always non-thinking, or always open) — uniform for users, differs from HF
      default on one size by construction.
   c. *Leave the default, add the knob* — plumb `enable_thinking` (OpenAI-compat `chat_template_kwargs` on serve, a
      `chat.Turn`/render option in the library) and keep today's bytes as the default. No G3 movement; does not fix the
      default gap, only makes it reachable.
   What the 9B measurement does to each: (c) leaves the 9B's replies opening with `<think>\n` and all reasoning in
   `content` (question 3) until the knob is used; (b)-non-thinking makes the 9B's replies shorter than HF's default ever is;
   (b)-open and (a) put the opener in the prompt, which matches HF for the 9B but leaves the reasoning block to be surfaced
   or stripped (question 3) — so question 3 has to be settled for every option except non-thinking. The 0.8B is the reverse:
   its default is already non-thinking, so (a) and (b)-non-thinking agree there and (c) is the no-op.
   Any option that changes default bytes changes every client's prompt; see "What flipping the default changes" (the G3 logit
   baseline itself does not move — corrected 2026-09-30).
3. **Output handling when thinking is on — PARTLY ANSWERED 2026-09-30 (grep, no run).** Non-test Go contains no
   `reasoning_content` field and no `</think>` stripping on the serve reply path; the only `</think>` uses are
   `internal/decide/model.go` (which appends the empty no-think block *itself* — `<think>\n\n</think>\n\n` — and refuses
   a tokenizer where that does not encode as `[<think> \n </think> \n]`, so it is a precedent for "the caller supplies the
   suffix" and a place that would need to learn the 9B's open form) and a test hook (`decoder/testhooks.go`). So today a
   Qwen3.5-9B chat reply through serve begins with a literal `<think>\n`, then reasoning, then `</think>`, then the answer,
   all as `content`. Still **not measured**: how long the 9B's reasoning runs on a text prompt (the 32-token runs never left the
   block), and whether a client such as Claude Code's Messages path or an OpenAI SDK chokes on or hides it.
4. **Is it only Qwen3.5?** Qwen3 (default ON) is served with its template adding nothing and the model left to open `<think>` itself
   (`decoder/testhooks.go` calls this "thinking left ON"); that is long-standing behaviour, not a measured-correct one. The
   same self-opening question applies to any family whose template pre-writes a think opener. Survey the other detected
   ChatML families' templates for a pre-written think block before concluding Qwen3.5 is special.

## Gates for whatever is chosen

- Byte-exactness of the rendered prompt against `tok.apply_chat_template(...)` **per checkpoint size actually shipped**,
  for text-only and image requests, with `enable_thinking` unset, true and false. A test that renders with only one size's
  template proves nothing about the other — this defect exists because the script and the G4 gate were built from the 0.8B
  alone.
- If default bytes change, the G4 test's pinned 4-token delta is rewritten on purpose (it becomes 0 for the 0.8B) and the 9B equivalent is
  added.
- Quality is judged through the real template, not raw completion (see the quant-eval note in memory); a change that makes
  the model think by default also changes latency per reply, so report tokens-to-answer, not only token identity.

## What P8a does and does not cover for the 9B

The night job's goinfer G2 (`TestQwen35VLReal_G2_9B`) feeds the decoder `golden_*` ids — the model's own template, opener
*in* the prompt — so it gates the vision tower, splice and decode on the 9B, **not** serve's prompt rendering. There is no
9B G4: the serve-level test (`TestServe_qwen35Image_G4`) runs on the 0.8B only. The `serve_*` goldens exist for the 9B now and
a serve-level 9B gate could be built on them, but it would pin today's behaviour (literal `<think>\n` in the reply), so
build it after the contract is chosen, not before.

## Proposed fix — research and design (2026-09-30, not built, nothing measured beyond what is marked)

**What "all clients, all the time" can honestly mean.** It cannot mean one wire format: clients split into families that
want contradictory things, and the other servers did not find a single format either — llama.cpp ships three
(`--reasoning-format none | deepseek | deepseek-legacy`, where legacy "keeps `<think>` tags in `message.content` while also
populating `message.reasoning_content`"), vLLM and SGLang ship a per-model reasoning parser, and Open WebUI documents that
it reads BOTH structured deltas (`reasoning_content` / `reasoning` / `thinking`) and inline tags, with inline tags the most
common path. What it can mean is **one invariant that every client family survives**:

> `content` is always the answer, never contains think markup, and is never silently empty when an answer was produced;
> reasoning, when it exists, goes in a separate field that a client is free to ignore; and a client that asked for
> thinking in its own dialect gets it in that dialect.

**Client families and what each needs.**

| family | reads | needs |
|---|---|---|
| OpenAI-SDK / curl / LangChain, unaware of reasoning | `choices[0].message.content`, `delta.content` | clean `content`; extra fields ignored (true for the OpenAI SDKs — unknown fields are dropped) |
| Reasoning-aware OpenAI-compat (Open WebUI, Cline-style, DeepSeek-API clients) | `reasoning_content` (also `reasoning`, `thinking`) | the separated field, streamed as `delta.reasoning_content` before content |
| Tag-parsing UIs (Open WebUI's inline path, LM Studio-style) | `<think>…</think>` inside `content` | raw tags in `content` — today's behaviour; the escape hatch below |
| Anthropic Messages (Claude Code, the SDKs) | `content[]` blocks; `thinking` request field | `text` blocks always; `thinking` blocks only if the request enabled thinking (the real API's `thinking_delta` events, then a `signature_delta` before `content_block_stop`) |
| Responses API | `output_text` items | clean text; reasoning items are out of scope (file header says so) |
| goinfer's own web UI | its JS `splitThinking` | keeps working on clean content (no tags left to split) |

**The mechanism: one splitter at the one choke point.** Every reply funnels through `drive`'s `onText(string)`
(`streamTokens` in `internal/serveapp/openai.go` decodes special tokens to literal `<think>` / `</think>` text, one token
each, so tags never arrive split across *special-token* boundaries — but a model can also write them as ordinary tokens, so
chunk-safety is still required). Add a pure, chunk-boundary-safe state machine in `chat/` next to `ProseStreamer`
(`chat/tools.go`, same shape: `Push(chunk) → (reasoning, content)`, holds back a partial tag at the tail, `Flush`):

1. start state from what the PROMPT ended with: `forcedOpen` (prompt ended `<think>\n`) → Reasoning; otherwise Undecided;
2. Undecided: optional leading whitespace then `<think>` → strip it, Reasoning; anything else → Content (the closed
   empty-block case is Content from token 1);
3. Reasoning: up to `</think>` → reasoning; on `</think>` strip the following `\n`s, → Content; a `<think>` seen later in
   Content is ordinary content (never re-enters);
4. `Flush` at end: still in Reasoning ⇒ **truncated inside the block** ⇒ everything so far is reasoning, content is empty
   (vLLM's rule, verified in its `Qwen3ReasoningParser` docs: "output was truncated. Everything generated so far is
   reasoning"). This is the one case the invariant cannot rescue by splitting — see "thinking budget / default" below.

It must run **before** `runToolTurn`'s `ProseStreamer` and `ParseToolCallsFor`: a model that quotes `<tool_call>` inside its
reasoning would otherwise trip the tool parser, and a leading `<think>` defeats the bare-JSON recovery (first non-space byte
check). vLLM has the same rule ("tool calling only parses functions from the `content` field"). Properties to test, in the
style of `TestProseStreamerMatchesParser`: (i) output is independent of chunking, byte-by-byte included; (ii) reasoning+content
reassembled with the tags equals the raw text (lossless — so `reasoning_format=none` can be built from the same state
machine); (iii) equals a non-streaming whole-string split; (iv) every tag spelling above, including `<think>` as several
ordinary tokens.

**Wire formats.**
- *OpenAI chat:* add `reasoning_content` (`omitempty`) to the non-streaming `message` map and to `delta` (`helpers.go`);
  stream reasoning deltas first, then content. Emit `reasoning_content` only — not also `reasoning`/`thinking` — because a
  client that reads several and concatenates would show it twice; vLLM moved to `reasoning` while llama.cpp/DeepSeek kept
  `reasoning_content`, so this is a real fork, and the choice should be made knowingly (recommend `reasoning_content`, the
  one llama.cpp and the DeepSeek API use, and the only one the OpenAI-compat reasoning clients all read; Open WebUI reads all three).
- *`reasoning_format` (llama.cpp's name, so configs carry over):* server flag and per-request override. `deepseek` (default,
  separated), `none` (raw, today's bytes, for tag-parsing UIs), `deepseek-legacy` (both). Cheap, because (ii) makes it a
  re-join.
- *Anthropic:* before this work the `thinking` request field was accepted and ignored (a comment on `anthropicReq` said so; it is now rewritten). Emit `thinking` blocks
  only when the request has `thinking.type` `enabled`/`adaptive`; otherwise run non-thinking (below) and emit text only.
  `anthropicTextStream` hardcodes block index 0 and `finish()` returns 0|1 — it needs real index bookkeeping, in
  `streamMessages`, `streamMessagesTools` and the duplicate in `vision_serve.go`. **Open, unverified:** what `signature` a
  non-Anthropic server should send (the real API validates it on replay; serve drops `thinking` blocks on input
  already, `anthropicTurns` default branch, so replay is safe), and whether Claude Code accepts an empty one. Settle it with
  the manual Claude Code smoke test that is still outstanding (memory: anthropic-messages-api), not by reading docs.
- *Responses API, `/v1/completions`:* Responses: drop reasoning, emit clean `output_text`. Completions is raw-prompt,
  untouched. Jobs and batches record the split form (`resultText` in `drive` currently sees raw text — decide once).

**Controlling it: `enable_thinking`.** Plumb it as a render option, not a per-request env read (the env-config rules in
`task-env-config-2026-09.md` apply): accept `chat_template_kwargs.enable_thinking` (vLLM / llama.cpp / SGLang all use exactly
that), Anthropic `thinking`, and Ollama-style `think` where cheap; add a render-options argument to `chat.Template.render`
(today `func(system string, turns []Turn) []Segment`; every caller listed in the serve map must pass it — `serveChatText`,
`tools.go`, `anthropic.go`, `responses.go`, vision routes, jobs, batches, and **`count_tokens`, which must render identically
to generation**). Decide the checkpoint's default by matching its own template's `enable_thinking` test
(`… is false` vs `… is true`, the two forms read above), and **fail toward today's bytes when the pattern is unrecognised** —
there is no jinja engine here, so a string match is fragile and must be pinned by a test against every size's actual
template file.

**Two rules that remove whole failure classes.**
1. *Grammar-constrained tool turns force non-thinking.* A forced/named/lone tool installs a masker from token 1
   (`constrainForcedTool`), and `required` uses `ToolCallsGrammar` from token 1: the model cannot write `<think>` at all, so
   an open `<think>\n` in the prompt would be contradicted. With `auto`, the `LazyMasker` must be armed only **after** the
   splitter has left Reasoning, or a `<tool_call>` quoted in reasoning arms it.
2. *Unaware clients get a bounded answer.* The truncation hole: a client that sent `max_tokens: 64` and never heard of
   reasoning gets `content: ""`, `finish_reason: length` from a thinking model — the failure that looks exactly like a broken
   server. vLLM's docs are silent on it. Options: (a) default non-thinking unless the request opts in (the closed empty block
   in the prompt; no reasoning is ever generated; also the fastest), (b) thinking by default with a reasoning budget that
   force-closes `</think>` (llama.cpp `--reasoning-budget`: -1 unlimited, 0 immediate end, N>0 budget), (c) leave it and
   document. **Recommendation: (a) as the default, opt-in to thinking, with (b) as a follow-up for opted-in clients** — but
   (a) changes default bytes (0.8B: +4 tokens; 9B: moves the default off the model's own),
   so it is its own decision, with the consequences listed in "What flipping the default changes" ("prompt now equals the model's template with
   `enable_thinking=false`"), not folded into the splitter.

**History.** Serve already drops Anthropic `thinking` blocks on input and ignores `reasoning_content` (not a `chatMessage`
field), so a client that splits reasoning and replays only the answer is safe. A client that replays raw `<think>` in
`content` gets it re-rendered as plain text every turn (context cost, differs from the template). Qwen3.5's template keeps
reasoning only for assistant turns after the last user query (`loop.index0 > ns.last_query_index`) — that matters for agent
loops, where tool-call turns after the last query carry their reasoning. Do not build history handling in the first cut;
changing how history renders breaks KV prefix reuse from the changed turn onward (`lm.sessions.acquire`). Accept
`reasoning_content` on input and drop it, and write down that this is a deliberate deviation.

**Hazards found in the code map** (file:line as reported by the map; re-verify before editing):
- *(RESOLVED 2026-09-30 — see "Stop strings on the answer only" under Built; kept as the record of the reasoning.)* *Stop strings are matched on raw decoded text* inside `streamTokens` (`firstStop`, `stopTailHold`), which includes
  reasoning: a stop string appearing in the reasoning ends the reply early with no answer. A splitter downstream of `onText`
  sees already-truncated text. Fixing it means matching stops on the content side only, which restructures `streamTokens` —
  and `TestStreamTokens_decodesAsAContinuation` pins exactly one `DecodePiece` call and zero `Decode` calls in it.
  First cut: document it; it affects thinking clients only.
- *Several tests are AST guards:* `TestStreamSurfaces_allSendUsageBeforeDone` counts exactly 5 `sseDone` completions,
  `TestDriveUsesRequestContext`, `TestServe_noRouteEncodesARenderedPrompt`. A new streaming site updates the count.
- *No shared reply builder:* the text-reply logic is duplicated in `serveChatText`, `serveVisionChat`,
  `serveVisionMessages`, the tool paths, `batches_run.go`, `jobs_run.go`, plus library copies in `internal/chatapp/main.go`
  and `demo/agent/agent/agent.go`. The splitter belongs in `chat/` (importable by all) and each site needs wiring; budget
  for missing one, and grep for `onText` callers as the completeness check.
- *Adjacent, separate defect:* Qwen3.5's own tool format (`<tool_call><function=NAME><parameter=K>…`) is not parsed anywhere
  in Go; serve prompts it with Hermes JSON, and if the model emits the XML form the call is dropped silently and returns as
  prose. It will make a "client works" claim false for tool-using clients on Qwen3.5 regardless of the think fix. Out of
  scope here; worth its own task.
- *No fake-token serve harness exists* (end-to-end serve tests need `GOINFER_SERVE_MODEL` and skip without it — a skip is
  not a pass). The usable model-free seam is `lm := &loadedModel{tk: tk}` plus a hand-fed `chan int` into `streamTokens`
  (`streamtokens_windowing_test.go`), and `httptest` against a bare `server`. Build splitter and wire-format tests there;
  use the real 0.8B and 9B only for the client matrix.

**Phasing, each shippable alone.**
1. *Splitter + `reasoning_content` + `reasoning_format` + Anthropic thinking blocks, default bytes UNCHANGED.* Works today
   without a prompt change: the 9B self-opens `<think>` (measured), so Undecided→Reasoning handles it, and the 0.8B mostly
   emits none. No G3 movement. Delivers the invariant except for the truncation hole.
2. *`enable_thinking` plumbed* (OpenAI `chat_template_kwargs`, Anthropic `thinking`), render option through every caller,
   byte-exactness gate per size, tool-grammar rule 1. Default bytes still unchanged.
3. *Default decision* (rule 2 a/b); consequences in "What flipping the default changes". Owner decision.
4. *Stops on content only; reasoning budget; history.*

**Gate: the client matrix, run for real (night queue if it exceeds ~10 min; the 0.8B is quick by day).** For each of 0.8B
and 9B, thinking unset/on/off, text and tools: OpenAI Python SDK non-streaming and streaming; the same with `max_tokens`
small enough to truncate inside the block (expect empty content + reasoning + `length`, and a documented note); raw `curl`;
Anthropic SDK with and without `thinking`; Claude Code (the outstanding smoke test); Open WebUI with its inline-tag path
(`reasoning_format=none`) and its structured path. Every cell asserts the invariant, not just "a reply came back". A cell
that passes only because the reply was empty is a fail — the same trap as a skipped test.

**Sources read for this section** (web, 2026-09-30; not re-verified line by line): vLLM reasoning-outputs page and its
`Qwen3ReasoningParser` API docs (Qwen3.5's template "places `<think>` in the prompt so only `</think>` appears in the
generated output"; `enable_thinking` via `chat_template_kwargs`; field renamed `reasoning_content` → `reasoning`; thinking
disabled → content, via `prompt_is_reasoning_end`); the llama.cpp server README and `common/chat` (`thinking_forced_open`
when the template ends `<think>\n`; `--reasoning-format`; `--reasoning-budget`); Open WebUI's reasoning-models page;
Anthropic's streaming and extended-thinking docs (`thinking_delta`, `signature_delta`, `display: omitted`; the final turn of a
thinking request must begin with a thinking block in manual mode); Ollama's `think` parameter and `message.thinking`.

## It is not only Qwen — measured survey, and the general shape (2026-09-30)

Rendered every chat template on this box (`~/models`, HF `apply_chat_template`, one user turn, `add_generation_prompt=True`)
with `enable_thinking` unset / `False` / `True` and read the tail. Templates that could not be loaded or carry no
`chat_template` in their tokenizer files (gpt-oss HF, SmolLM3, OLMo-3-think, LFM2.5, Ministral, Granite-4.2, Gemma-4-E2B, internlm2
— needs `trust_remote_code`) are **not surveyed** here; for gpt-oss the repo's own comment (`chat/templates.go`, Harmony) is the
only evidence used.

| family (checkpoints on this box) | generation prompt, `enable_thinking` unset | `=False` | `=True` | reasoning delimiters |
|---|---|---|---|---|
| Qwen3 (1.7B, 4B, 30B-A3B) | nothing (model decides) | closed empty `<think>\n\n</think>\n\n` | nothing | `<think>…</think>`, model writes both |
| Qwen3.5 0.8B | closed empty block | same | open `<think>\n` | `<think>…</think>` |
| Qwen3.5 9B, JEV-9B | open `<think>\n` | closed empty block | open `<think>\n` | only `</think>` is generated |
| Gemma 4 26B-A4B | closed scaffold `<\|channel>thought\n<channel\|>` | same | nothing after `<\|turn>model\n` (`<\|think\|>\n` goes in the system turn instead) | `<\|channel>thought\n…<channel\|>` |
| gpt-oss (Harmony; repo comment only) | no non-thinking form: always a channel protocol | — | — | `analysis` / `commentary` / `final` messages, several per reply |
| Gemma 3, Granite-hf, Mellum2, Phi-3, Qwen2.5/1.5, TinyLlama, Qwen3-Next instruct | nothing | kwarg ignored | kwarg ignored | none in the template |

What the table shows, in order of weight:
1. **The generation prompt has exactly four shapes** — nothing; open prefill; closed prefill; an always-on channel protocol — and
   **the same knob name (`enable_thinking`) means different things per family and per size**: default ON (Qwen3, Qwen3.5-9B),
   default OFF (Qwen3.5-0.8B, Gemma 4), or absent. Code that hard-wires "Qwen does X" or "ChatML does Y" is wrong by
   construction; this defect was the ChatML renderer doing exactly that.
2. **The output side has two parser shapes, not one per family:** a *delimited* span (`<think>…</think>`, Gemma 4's channel
   markers — same machine, different strings, optionally "forced open" by the prompt) and Harmony's *multi-message channel
   routing*. vLLM and SGLang land on the same split (a parser per model family, parameterised).
3. **History rules are also shared.** Gemma 4's template, like Qwen3.5's, re-renders reasoning only for assistant turns after
   the last user turn (`loop.index0 > last_user_idx`), additionally gated by `preserve_thinking` when there are tool calls,
   and it reads `reasoning` **or** `reasoning_content` from the assistant message (`chat_template.jinja`, lines ~238-242).
   So inputs should accept both field names.

**The general solution: make the prompt half and the parse half one declared object per family, and make the machinery
generic over it.** The defect class is *a renderer and a parser that were written separately and disagree*; fixing Qwen3.5
alone fixes one instance. Concretely:
- Add to `chat.Template` a `Reasoning *ReasoningSpec`, filled in by `Detect` from the template string and the vocabulary
  (`HasToken`), e.g. `{Open, Close string; Default ThinkDefault /*On|Off|Absent*/; Prefill func(mode) string; History HistoryRule}`.
  `nil` means "this family does not think in the template" (Phi-3, Gemma 3, …): no splitting, no prefill, today's bytes.
- One render option `Thinking{Default, On, Off}` through the same render signature for every family; each family's spec knows
  its three prompt endings (the table's columns). A test renders the Go template and the real HF template for **every
  checkpoint on the box** in all three modes and asserts byte equality — a generated golden per checkpoint, not a hand-written
  one per family, so a family nobody has written yet fails loudly when its first checkpoint is added (the repo's
  `validateResolved()` chokepoint idea, applied to templates).
- One generic *delimited* splitter parameterised by `{Open, Close, forcedOpen}` (Qwen, Qwen3.5, GLM, DeepSeek, MiniMax, Gemma 4
  with its channel strings) plus a separate `HarmonyParser` for the channel protocol, both behind one interface
  `Push(chunk) (reasoning, content)`; the wire layer (`reasoning_content`, Anthropic `thinking` blocks, `reasoning_format`) only
  ever sees that interface, so it is written once.
- Tool-call parsing already dispatches per template (`Template.ParseToolCalls`); put it behind the same per-family object so
  prompt, reasoning parse and tool parse cannot drift apart. The Qwen3.5 XML-tool-format gap above is the same defect class:
  the renderer speaks Hermes JSON, the model speaks XML, nothing checks them against each other.
- **Fail toward today's behaviour:** a template whose `enable_thinking` pattern is not recognised gets `Reasoning == nil`
  (no prefill change, no splitting), and a startup log line names the model as "thinking unmanaged". Never guess a default.

Alternative considered and **not** recommended now: embed a real Jinja engine and render each checkpoint's own
`chat_template` for every family, which removes hand-written renderers and their drift entirely and makes every kwarg work
without per-family code. Cost: a new dependency in a repo that hand-writes byte-exact renderers with goldens on purpose,
no test that the engine matches HF's Jinja (custom `raise_exception`, `strftime_now`, `tojson`, `namespace`, macros — the
Qwen3.5 and Gemma 4 templates use all of them), and a per-request render cost. It is the long-term option if the per-family
specs keep multiplying; the spec approach is the smaller step and keeps the goldens.

## Not in scope

The vision path itself (P8a G0-G4 stand on `golden_*`), Qwen3 (served with thinking left on, as today — though the phase-1
splitter would cover it for free), and Qwen3.5's XML tool-call format (separate task).

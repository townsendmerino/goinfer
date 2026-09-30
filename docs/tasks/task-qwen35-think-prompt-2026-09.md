# Task: Qwen3.5's generation prompt — serve renders it differently from the model's own template (2026-09)

> **Status 2026-09-30: open, owner decision needed (scope below).** Found by P8a gate G4
> (`docs/measurements/p8a-qwen35-vl-2026-09/g1-g4-results.md`, finding 1), then widened the same day when the 9B leg of P8a
> died on it: **the 0.8B and the 9B disagree about the default, so the G4 write-up's "Qwen3.5 defaults to thinking OFF" is
> true of one size only.** Nothing here has been changed in product code.
>
> **Measured since (same day):** the 9B, given a prompt *without* its open `<think>\n`, writes `<think>\n` itself as its first two
> tokens and then continues identically to the with-opener run (question 1, three images, 32 tokens). And serve has **no
> think-block handling at all** in non-test Go (no `reasoning_content`, no stripping — question 3), so the self-written
> opener, and the whole reasoning block that follows it, reach the client as ordinary reply content. The P8a pin script's
> 0.8B-only assertion that killed the 9B night job is fixed (`e6b41252`, local) and verified on the 9B.

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
   Any option that changes default bytes moves G3 and needs the baseline **re-registered with a mechanism**, not
   re-baselined because the number moved.
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
- G3 re-registered before the run if default bytes change; the 4-token G4 delta becomes 0 (0.8B) and the 9B equivalent is
  added.
- Quality is judged through the real template, not raw completion (see the quant-eval note in memory); a change that makes
  the model think by default also changes latency per reply, so report tokens-to-answer, not only token identity.

## What P8a does and does not cover for the 9B

The night job's goinfer G2 (`TestQwen35VLReal_G2_9B`) feeds the decoder `golden_*` ids — the model's own template, opener
*in* the prompt — so it gates the vision tower, splice and decode on the 9B, **not** serve's prompt rendering. There is no
9B G4: the serve-level test (`TestServe_qwen35Image_G4`) runs on the 0.8B only. The `serve_*` goldens exist for the 9B now and
a serve-level 9B gate could be built on them, but it would pin today's behaviour (literal `<think>\n` in the reply), so
build it after the contract is chosen, not before.

## Not in scope

The vision path itself (P8a G0-G4 stand on `golden_*`), Qwen3 (served with thinking left on, as today), and any change to thinking-mode
output parsing beyond the check in question 3.

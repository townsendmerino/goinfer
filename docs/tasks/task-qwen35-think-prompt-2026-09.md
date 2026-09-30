# Task: Qwen3.5's generation prompt — serve renders it differently from the model's own template (2026-09)

> **Status 2026-09-30: open, owner decision needed (scope below).** Found by P8a gate G4
> (`docs/measurements/p8a-qwen35-vl-2026-09/g1-g4-results.md`, finding 1), then widened the same day when the 9B leg of P8a
> died on it: **the 0.8B and the 9B disagree about the default, so the G4 write-up's "Qwen3.5 defaults to thinking OFF" is
> true of one size only.** Nothing here has been changed in product code.

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
- **9B:** serve is short by the open `<think>\n` (the token count is **not measured**). Whether the model then writes
  `<think>\n` itself, answers directly, or does something else is **not measured** — that is the first question below.

The gap applies to **every Qwen3.5 text turn through serve**, not only image turns, and to any other caller of the ChatML
renderer for these checkpoints. `enable_thinking` is not plumbed anywhere in product code (`grep enable_thinking` in `*.go`
finds only test comments), so a caller who wants HF's `enable_thinking=false` behaviour has no way to ask for it.

## Why it was not fixed in P8a

Changing the renderer alters the prompt of every Qwen3.5 text turn. P8a's gate G3 says the text path must not move
(baseline `docs/measurements/p8a-qwen35-vl-2026-09/g3-baseline-0a3f0a7f.json`), and a prompt change fails it by
construction. The P8a goldens (`golden_*` = the model's own template; `serve_*` = the same minus the think opener) therefore
carry both variants, and G2 compares against `golden_*`, so the vision gate does not depend on this decision.

## Questions, in the order they unblock each other

1. **What does the 9B do when the open `<think>\n` is missing?** P8a's 9B golden step produces both variants on the same
   three images under HF f32 greedy (`golden_*` with the opener, `serve_*` without), 32 tokens each: if the `serve_*` text
   starts `<think>\n` the model re-opens the block itself and the prompt difference is cosmetic at the first token; if it
   answers directly, serve is running the 9B in a mode HF's default never exercises. Free once the pin step has run; read
   `hf_text` in both files. (32 tokens, three images, an image prompt — suggestive only for text turns.)
2. **What should the contract be?** Options, none chosen:
   a. *Follow the checkpoint's own template default* — render the generation prompt from the model's `chat_template`
      (or detect its `enable_thinking` default), so each size gets what HF gives it. Closest to "the model's own template"
      but means the default differs by checkpoint.
   b. *One fixed serve policy for the family* (always non-thinking, or always open) — uniform for users, differs from HF
      default on one size by construction.
   c. *Leave the default, add the knob* — plumb `enable_thinking` (OpenAI-compat `chat_template_kwargs` on serve, a
      `chat.Turn`/render option in the library) and keep today's bytes as the default. No G3 movement; does not fix the
      default gap, only makes it reachable.
   Any option that changes default bytes moves G3 and needs the baseline **re-registered with a mechanism**, not
   re-baselined because the number moved.
3. **Stop and output handling if thinking is on by default.** A 9B that opens `<think>` answers with a reasoning block
   first: check how serve surfaces it (streamed as content? stripped? `reasoning_content`?) before making thinking-ON the
   default anywhere. Not looked at in this doc.
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

## Not in scope

The vision path itself (P8a G0-G4 stand on `golden_*`), Qwen3 (served with thinking left on, as today), and any change to thinking-mode
output parsing beyond the check in question 3.

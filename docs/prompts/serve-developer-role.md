# goinfer task: `role: "developer"` compat on the serve surface

> **Box:** either. Small, self-contained. Stimulus: DeepSeek Harness sends system prompts as
> `role: "developer"` for reasoning-class models (its escape hatch is a per-model
> `compat.supportsDeveloperRole: false`), and the same convention is spreading from OpenAI's
> newer APIs — so this is general harness compat, not a dsh special case. Verified 2026-08-21:
> `internal/serveapp` has no handling for the role (the only "developer" in the tree is the
> harmony template's internal channel in `chat/templates.go` — unrelated).

## Step 0 — characterize before fixing

Send a `role: "developer"` message to each surface and record what actually happens today —
400, silent drop, or pass-through into the template as an unknown role. Do this for
`/v1/chat/completions`, `/v1/completions` (n/a expected), `/v1/responses` (both the `input`
message-item form and `instructions`), and `/v1/messages` (Anthropic — `system` is a top-level
field there, so developer-role items should already be impossible; confirm the error is clean).
The current behavior goes in the commit message; a fix without the before-state is the class of
change this repo doesn't make.

## The change

Treat `role: "developer"` as `role: "system"` on the OpenAI-compatible surfaces:

- Same merge/precedence semantics as an explicit system message in the same position — an alias,
  not a new concept. If both `system` and `developer` messages appear, follow OpenAI's stated
  semantics at implementation time (verify current; expected: developer supersedes/means system
  for reasoning models) and document the choice in one sentence.
- `/v1/responses`: accept `developer` in message items, same aliasing.
- Unknown roles other than this one keep today's behavior — this is not a general
  unknown-role-tolerance change.
- No template changes: by the time the chat renderer sees it, it is a system message.

## Gates

- Table-driven test per surface: developer-only, system+developer, developer in mid-conversation
  (should behave as the equivalent system-message request — byte-identical prompt rendering vs
  the aliased form).
- The Step-0 characterization test kept as a regression guard for the *rejected* cases that stay
  rejected.
- README/serve docs: one line in the OpenAI-surface section ("`developer` is accepted as an
  alias for `system`"), so the C2 audit reads a claim that matches the code.

## Non-goals

dsh-specific anything (their compat flag remains a fine workaround and this change simply makes
it unnecessary); OpenAI `reasoning_effort`/o-series parameters (accept-and-ignore already covers
the pattern if they arrive — separate item if a harness actually needs them honored).

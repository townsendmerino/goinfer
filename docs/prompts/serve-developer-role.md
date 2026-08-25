# goinfer task: `role: "developer"` compat on the serve surface

> **Box:** either. Small, self-contained — one short session, mostly tests. Stimulus: DeepSeek
> Harness sends system prompts as `role: "developer"` for reasoning-class models, and the same
> convention is spreading from OpenAI's newer APIs — general harness compat, not a dsh special
> case.
>
> **This is NOT a blocker, and the doc must not read as one.** dsh ships a per-model
> `compat.supportsDeveloperRole: false` that makes it send `system` instead, so the dsh Tier-0
> run can happen today with no goinfer change. The reason to do this FIRST anyway is that the
> current behavior is **silent-wrong**: the role isn't rejected, it's demoted (below), so a
> harness whose flag is forgotten — or one with no such flag — delivers its entire agent
> scaffold as the user's first message, and the failure reads as "the model is bad at agent
> work," not "the server mangled the request." That is exactly the trap a characterization run
> must not be sitting on top of. Silent-wrong and cheap ⇒ fix before the Tier-0 run.

## Step 0 — characterization (the chat path is ANSWERED; finish the rest)

**Answered, verified on-device at `dc8355e`:** `messagesToTurns`
(`internal/serveapp/openai.go` — cite the function, not a line; it will move) maps roles with a
`switch` whose arms are `system` / `tool` / `assistant` and a `default:` that appends a **user**
turn. `developer` therefore hits `default` and is **silently demoted to a user message, in
position** — not a 400, not a drop. This before-state goes in the fix's commit message verbatim.
All four OpenAI-side surfaces funnel through this one function (call sites in `openai.go`,
`responses.go`, `tools.go`, `vision_serve.go`), which is why the fix is a single chokepoint.

**Remaining Step 0:** `/v1/messages` (Anthropic) takes `system` as a top-level field, so a
developer-role content item should be structurally impossible — confirm the error for one is
clean, and keep a regression test pinning it. `/v1/responses`' `instructions` field is separate
plumbing — confirm it is unaffected either way.

## The change

One `case "developer":` arm in `messagesToTurns` — the verified single chokepoint — treating
`role: "developer"` as `role: "system"` on the OpenAI-compatible surfaces:

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
- A test pinning the OLD behavior is written first and shown failing against the fix — the
  demotion was silent once; the gate exists so it cannot become silent again.
- README/serve docs: one line in the OpenAI-surface section ("`developer` is accepted as an
  alias for `system`"), so the C2 audit reads a claim that matches the code.

## Non-goals

dsh-specific anything (their compat flag remains a fine workaround and this change simply makes
it unnecessary); OpenAI `reasoning_effort`/o-series parameters (accept-and-ignore already covers
the pattern if they arrive — separate item if a harness actually needs them honored).

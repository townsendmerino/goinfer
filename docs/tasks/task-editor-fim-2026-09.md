# Task: fill-in-the-middle, and the editor integration it unlocks (F0–F7) — 2026-09

> **Status: SCOPED 2026-09-18, unstarted. F0 is a measurement and gates everything after it.**
>
> Filed after the owner asked how hard a VS Code extension would be. The answer this doc argues:
> **the extension is the easy half and the wrong place to start.** goinfer cannot do fill-in-the-
> middle at all today, and FIM — not chat — is the one job where running locally genuinely wins.
>
> Sibling: [`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) mode 3, "point my tools
> at it", of which this is the strongest instance.

---

## 1. Why FIM rather than a chat extension

**A chat extension should not be built.** Continue, Cline and Roo already point at goinfer today,
because `serve` is OpenAI-compatible. Building a fourth one means competing with funded incumbents
at the thing they are good at, and it wins goinfer nothing it does not already have.

**Inline completion is different, and it is the one place a local engine beats a remote frontier
model on the merits:**

- It is **latency-bound**, not intelligence-bound. Ghost text that arrives late is worse than no
  ghost text. A round trip to a hosted model is a floor you cannot get under; a 1.5B on the same
  machine has no network at all.
- It runs **constantly**, on every pause in typing, which is exactly the traffic profile where
  paying per token hurts and where a local model costs nothing but watts.
- It sends **your code, continuously**, which is the privacy case at its sharpest.
- A 1.5B coder model at 40 tok/s is *adequate* for a few tokens of completion in a way it is not
  adequate for a conversation. The task is sized to the model.

## 2. What exists, and the one thing that does not

- **`/v1/completions` exists** and is the right surface (FIM belongs on the legacy Completions API,
  not on chat).
- **`completionReq` has `Model`, `Prompt`, `Stream`, `Logprobs` — and no `suffix`.** That is the
  whole gap at the API level. `suffix` is part of the legacy OpenAI Completions API, so adding it is
  compatibility work, not invention.
- **The tokens are already tokenizable.** Mellum2's own `tokenizer_config.json` (a gitignored local
  test-fixture slice, not a committed path — described here rather than cited, since nobody else
  can resolve it) carries `<fim_prefix>`, `<fim_middle>` and `<fim_suffix>` in its added/special
  token set. **There is no tokenizer work in this doc** — the pieces are already encodable today.
- **Mellum is already a supported family** — JetBrains' code-completion model, built for exactly
  this — but only as a *chat* template (`mellum2`). The FIM model is being served as if it were a
  chatbot.
- **The spelling differs per family.** Mellum uses `<fim_prefix>`; Qwen-family coders use
  `<|fim_prefix|>`. So the wrapper is per-family — which is precisely the shape
  `ToolCallWrapper()` already has in `chat/tools.go`, switching on the template name and returning
  `ok=false` for families that have none.

## 3. The property to build around: prefix reuse makes typing forward nearly free

A FIM prompt is `<prefix-token> A <suffix-token> B <middle-token>`, where A is the code before the
cursor and B the code after.

**When you type forward at the cursor, A grows at its end.** goinfer's reuse is longest-common-
prefix matching, so the next request shares everything up to the change — the KV for the whole file
before the cursor is already resident and only the newly typed tokens are prefilled. A completion
while typing forward should cost a handful of tokens of prefill, not a whole file.

That is what makes the latency budget in F0 reachable at all, and it is a property the obvious
competitor does not have in the same shape.

**Where it collapses, and the design must say so:** editing backwards, jumping the cursor, or any
change to B resets the match. Those requests pay a full prefill and will be visibly slower. F0
measures both cases separately rather than reporting one blended number.

**Editor traffic needs its own KV session.** `-kv-sessions` defaults to 4; a completion firing every
few hundred milliseconds must not evict the chat the user also has open.

## 4. F0 — the measurement that gates the rest

Nothing else in this doc is worth building if the numbers are wrong, and they are cheap to get.

- **What:** p95 latency for one completion, on the Mac, 1.5B coder model at int4, at file contexts
  of roughly 1k / 4k / 16k tokens. **Two arms, reported separately:** typing forward (reuse hits)
  and cursor-jumped (cold prefill).
- **Also record:** the prefill/decode split — a completion is a few tokens, so this is TTFT-
  dominated — and the reuse hit rate across a realistic typing session.
- **Pre-registered thresholds, typing-forward arm:**
  - **p95 under 150 ms** — good; F3 is worth building.
  - **150–300 ms** — usable with debounce; F3 proceeds with the debounce tuned to the number.
  - **above 300 ms** — **kill F3.** Ghost text at that latency is an irritation, not a feature. F1
    and F2 may still ship, because a slower completion is still worth having in an editor the user
    already configured, but goinfer should not ship an extension that makes the engine feel slow.
- Measured the way everything here is measured: quiet box, paired, same session, recorded in
  `docs/measurements/`.

## 5. F1 — the server surface

- **`suffix` on `/v1/completions`**, per the legacy OpenAI shape.
- **`FIMWrapper()` on `chat.Template`**, mirroring `ToolCallWrapper()`: returns the three tokens
  plus `ok`. First families: `mellum2` and the Qwen coders.
- **A family with no wrapper declines with a reason.** It must never fall through to treating
  prefix-plus-suffix as a plain prompt — that produces confident nonsense with no error, which is
  the failure mode this repo keeps naming. Same discipline as a named `tool_choice` that cannot be
  constrained returning a 400 rather than an unconstrained decode.
- **Stops are the FIM family's own**, not the chat template's. Getting this wrong is how gpt-oss
  spent a release never reaching its final channel.
- A completion-appropriate `max_tokens` default — tens, not hundreds — and a hard cap, because a
  runaway completion in an editor is worse than a short one.

## 6. F2 — reach the editors that already exist, before writing one

`docs/integrations/continue.md`, alongside the `claude-code` and `opencode` recipes that already
exist. Continue's autocomplete expects a FIM-capable endpoint; once F1 lands, goinfer is one.

**This is deliberately ahead of F3.** It serves every editor that speaks the completions API rather
than one extension that then has to be maintained forever, and it is a doc page plus F1.

## 7. F3 — the extension, if F0 says yes

Thin, and explicitly not an assistant:

- An `InlineCompletionItemProvider` against `/v1/completions`.
- Debounced to F0's measured number; **cancel in flight on the next keystroke** — K1's cancel-by-id
  is the right mechanism, and none of the `/v1/jobs` machinery is needed for something this short.
- Status: which model is loaded, its decode path, tok/s — the model-card facts X7 exposes.
- Start, stop, and load a model without leaving the editor.
- One click to write the config for whichever assistant extension is already installed, pointing it
  at the local server. That is mode 3 made one click instead of a recipe.

**Not** a chat panel. **Not** an agent. Those exist and already work against goinfer.

## 8. Not in scope

- A chat or agent extension, per §1.
- Repo-wide context or retrieval for completions — that is ken and aikit, and it is a different
  project.
- JetBrains, Neovim or Zed ports until the VS Code one has earned them.
- Training or fine-tuning a completion model.

## 9. Gates

- A FIM request against a family with no wrapper is **declined with a reason**, never silently
  completed as a plain prompt. Mutation-checked.
- **Typing forward reports high `prefill_reused_tokens`; a cursor jump reports low.** This is the
  §3 property, asserted rather than assumed. (X2 and X8 in
  [`task-turn-telemetry-2026-09.md`](task-turn-telemetry-2026-09.md) make it directly observable.)
- Editor traffic does not evict a chat's KV session.
- Stops come from the FIM family and a completion terminates on its own rather than at `max_tokens`.
- **F0's p95 is re-measured after F1 lands** and still meets the bar it was gated on — the
  implementation is where a latency budget usually goes missing.

## Sources

`internal/serveapp/openai.go` (`completionReq` — `Model`/`Prompt`/`Stream`/`Logprobs`, no `suffix`) ·
`chat/tools.go` (`ToolCallWrapper` — the per-family shape `FIMWrapper` mirrors) ·
`chat/templates.go` (`mellum2`, registered as a chat template) ·
Mellum2's own `tokenizer_config.json` (a gitignored local test-fixture slice — the FIM tokens, already special) ·
`decoder/resident_reuse.go`, `internal/serveapp/sessions.go` (the LCP reuse §3 depends on) ·
`internal/serveapp/main.go` (`-kv-sessions`) ·
[`task-halt-2026-09.md`](task-halt-2026-09.md) K1 (cancel-by-id, F3's debounce mechanism) ·
[`task-turn-telemetry-2026-09.md`](task-turn-telemetry-2026-09.md) X7, X2/X8 ·
`docs/integrations/` (the two recipes F2 joins)

<!-- doc-reviewed: 2026-09-18 -->

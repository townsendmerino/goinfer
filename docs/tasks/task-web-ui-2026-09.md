# Task: `serve -web` as a real chat interface — the Claude-app gap (W1–W26) — 2026-09

> **Status: SCOPED 2026-09-13, SCOPE DECIDED 2026-09-14, IN PROGRESS — §6.1, Tier A (W1–W8), Tier B done except W12 (skipped for now, owner 2026-09-14): W9–W11 and W13–W18. Tier C next, each needing its own design decision (§1). W27–W32 added 2026-09-15 (§7) now that J1–J4 have shipped; W27–W32 all DONE 2026-09-15. Tier C is what remains.** Filed from
> a feature comparison against the Claude desktop/web app, read against the tree at `9d29d625`.
>
> **The scope question is settled: the web UI is a product surface, to be made as fully useful for
> users as possible** (owner decision, 2026-09-14). That reverses the lean recorded in
> [`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) §F.3 — *chat and pull only, the
> first-run surface, not a product* — which has been amended in place to point here. All three tiers
> are in scope; §1 records what that does and does not change.
>
> Siblings: [`task-fit-to-hardware.md`](task-fit-to-hardware.md) §3 already owns W12 (fit before
> download) and is cited rather than restated; [`task-work-queue-2026-09.md`](task-work-queue-2026-09.md)
> J3 is the only route to W27; [`task-halt-2026-09.md`](task-halt-2026-09.md) K1/K2/K5 is what a
> W22 admin panel would surface. `docs/completed/task-web-ui-ambient.md` is the closed record of
> the current visual design and is not superseded by anything here.

---

## 0. What exists today

The page is **a small embedded directory** — `//go:embed webui` (`internal/serveapp/webui.go:51`):
`index.html` plus `ui/app.css` and `ui/app.js`, hand-written HTML, CSS and vanilla JS, no build
step, no external stylesheet, font or script. *(Until 2026-09-14 it was one 1,828-line file; §6.1
records the split.)* That is deliberate and load-bearing: a CDN reference would make the UI of an
offline-capable engine require the network. It is off by default behind `-web`
(`internal/serveapp/webui.go:545`), and it is a client of the same `/v1` routes any other client
uses — so it cannot drift from the API, because it *is* the API's user.

Two tabs (`internal/serveapp/webui/index.html:13`):

- **Chat** — model dropdown, temperature, max tokens, streaming reply, Stop, and per-response
  token count / tok/s / wall time attached to the message itself.
- **Models** — enter a Hugging Face repo, list its `.gguf` files, pull one with a progress bar,
  transfer rate, ETA and sha256 verification.

Plus an API-key field. That is the whole surface.

**The five things it already does better than the Claude app**, listed first because a
feature-parity exercise is exactly how they get sanded off:

1. **Per-response measurement, with the response.** Tokens, tok/s and seconds on the bubble, not
   in a status line the next turn overwrites.
2. **The machine is visible.** Model id and the server's own `decode_path` in the header chrome.
   No other chat UI tells you which kernel answered.
3. **Model pull with sha256, rate and ETA** — including the deliberate "no sha256 published — NOT
   verified" when the repo published none.
4. **It works with the network off.** One file, no CDN, no font host, no account, no telemetry.
   **Nothing in Tier A may change this.**
5. **Stop actually stops.** The abort cancels the generation server-side and frees the worker.

---

## 1. The decision — DECIDED 2026-09-14

**Owner decision: make the web UI as fully useful for users as possible.** The page is a product
surface, not only the first-run surface. §F.3's lean was right for what the page was; it is
superseded, and amended in place in [`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md)
so it no longer reads as current.

*What this replaced, kept for the record:* the recommendation here was to build Tier A only, treat
Tier B as a deliberate widening needing sign-off, and leave Tier C unauthorised.

**What the decision puts in scope:**

- **Tier A (W1–W8)** — in scope, and built first. These are still the items a first-run user hits
  within five minutes, and W1/W3/W7 are prerequisites for much of what follows.
- **Tier B (W9–W18)** — in scope. No further sign-off needed.
- **Tier C (W19–W26)** — in scope in principle. The "is `-web` a product?" objection no longer
  blocks any of them. **Each still owes its own technical decision before it is built**, and those
  are real design calls, not permission: where the tool loop runs (W19), a PDF extractor against
  the no-new-root-dependency rule (W20), the gate for page-initiated admin actions — answered once
  for W5 and W22 together — the RAG stack (W24), and the sandbox model (W25). Items that already
  belong to other docs stay owned there (W12 → `task-fit-to-hardware.md` §3, W27's re-attach →
  `task-work-queue-2026-09.md` J3).

**What the decision does NOT change** — none of these were the "not a product" argument, so none
are reopened by it:

- **§5's non-goals stand.** Accounts/sync/sharing, voice, parallel generation and connectors are out
  for reasons of architecture and model support, not scope. Revisit one only by naming it.
- **§6's constraints still govern every item.** One binary, fully offline, no CDN and no bundler
  (§6.1); model output rendered as DOM nodes, never an HTML string (§6.2 — this matters *more* as the
  page does more); every mutating route behind the existing gates (§6.3).

**Build order:** §6.1's move to `//go:embed webui/*` first, before Tier A doubles the file; then
Tier A in its ranked order; then Tier B; then Tier C item by item, each opening with its own decision.

---

## 2. Tier A — the eight that make it usable (build first)

Ranked by what a person notices in the first five minutes.

### W1 — Markdown and code blocks — DONE 2026-09-14
~~Today the page renders **plain text only**~~ — model output now renders as Markdown through
`ui/markdown.js`, a renderer that builds DOM nodes from a fixed element allow-list and never an HTML
string (the rule is stated at `internal/serveapp/webui/ui/app.js:935`). Paragraphs and line breaks,
headings, fenced code with a language label (an unclosed fence renders as code, so streaming does not
flicker), inline code, strong/em/strike, links, bare URLs, blockquotes, nested lists and GFM tables.
Raw HTML in model output is shown as text; links are live only for `http:`/`https:`/`mailto:`; images
are never loaded (a Markdown image becomes a link to it). Streaming re-renders at most once per frame.

**Not done the way this entry originally said, and why.** It advised extending agent-web's
`renderMarkdownLite` (`demo/agent/cmd/agent-web/index.html:183`). That function assembles an HTML
*string* and assigns it with `innerHTML` — exactly the pattern §6.2 forbids — so it was not a safe
base and was not used. It escapes `<`, `>` and `&` first, so it is probably not exploitable as written,
but agent-web renders model output that way today; that is a separate item, noted here rather than
fixed under W1's scope.

**Gates**, each shown able to go red by a mutation: `scripts/webui_md_gate.mjs` drives the shipped
renderer in headless Chrome — 24 hostile payloads rendered into live DOM with a canary that must never
fire, 11 pathological inputs under a 1 s budget (ReDoS), 25 structural cases — and runs from
`go test` as `TestWebUI_markdownGateInBrowser`; `TestWebUI_noHTMLStringSinks` fails the build on any
`innerHTML`/`outerHTML`/`insertAdjacentHTML`/`document.write`/`eval`/`new Function` in the page. One
finding from mutating it: a `<meta http-equiv=refresh>` inserted via `innerHTML` really navigates the
page — an `innerHTML` regression would be an open redirect even without script.
**Effort was: M.** **This item was also the security decision — §6.2.**

### W2 — Copy, per message and per code fence — DONE 2026-09-14
~~No copy affordance at all~~ — every code block has a Copy button in its header, and every finished
message (yours, the model's, and a *stopped* answer) has a Copy action. A message copies the **raw
Markdown** it was rendered from, not the rendered text, so a paste keeps its formatting. Copy uses the
async clipboard API and falls back to a selected textarea, because that API only exists in a secure
context and a page opened at a plain-http LAN address (W16's phone) is not one; if both refuse, the
button says "Copy failed". Code-block buttons are rebuilt on every streamed frame, so one delegated
handler serves them all.

Gate: `scripts/webui_app_gate.mjs` (also `TestWebUI_appGateInBrowser`) drives the shipped page through
its own `send()` and checks exact copied text on both clipboard paths and with no clipboard API at
all, failure feedback and its reset, copy on a stopped answer, and copy after a re-render rebuilds the
buttons — each shown able to go red by a mutation. **Effort was: S.**

### W3 — The conversation survives a reload — DONE 2026-09-14
~~History is `const history = []` in page memory~~ — the conversation is a `transcript` of turns
(`internal/serveapp/webui/ui/app.js:309`) saved to `localStorage` under a versioned key, and the
messages sent to the API are derived from it, so what is shown, saved and sent cannot drift apart.
On load it is restored and rendered exactly as live turns are — model output through the Markdown
renderer, the user's text as text — so stored content earns no more trust than fresh output.

- **A reload mid-stream keeps the partial answer**: it is saved about once a second and on
  `pagehide`, and comes back labelled "interrupted". Stopped answers come back labelled "stopped".
- **Storage failures are survivable**: blocked storage or a full quota leaves the chat working and
  shows a note saying the conversation won't survive a reload; a stored value the page can't read is
  set aside under its key + `.unreadable`, not destroyed.
- ~~**New chat** clears it (after a confirm — there is no undo)~~ — superseded by W9: New chat starts
  another conversation and keeps this one, so it no longer asks. Still disabled mid-reply.
- ~~**One conversation per browser**, shared by its tabs~~ — superseded by W9: many conversations, one
  key each; which one is open is per tab. An idle tab still follows another tab's change to the
  conversation it has open.

`-session-dir` holds KV snapshots, not transcripts, and was correctly not used.

Gate: `scripts/webui_app_gate.mjs` now runs in five phases separated by real page reloads —
what is saved, what is restored (and that Copy still copies the source), a mid-stream reload,
unreadable storage, a quota failure, New chat confirmed and cancelled, tab sync, and a hostile
stored transcript (invalid entries skipped, model name shown as text, canary never fires). Seven
mutations, each red. Two of them first exited as "could not run", which `go test` turns into a
SKIP — a broken restore would have reached CI as a skipped test; the shared gate driver now reports
any failure inside a loaded page as a failure. **Effort was: M.**

### W4 — A system prompt box — DONE 2026-09-14
~~There is no way to set a system message at all~~ — a collapsible "System prompt" box in the settings
card (`internal/serveapp/webui/ui/app.js:623`), whose summary reads "· active" when set so it is
visible while collapsed. When non-blank it is sent **first, trimmed**, with every request; blank or
whitespace-only sends nothing. The route accepts a system message for every family — templates
without a system role fold it into a user turn server-side (`messagesToTurns`) — so the box needs no
per-model exceptions.

Decided here, and why: it is a **setting, not part of a conversation**. It has its own storage key,
survives New chat and reloads, and is never written into the saved transcript — it is applied to
each request, not said once. People reuse a system prompt across chats; New chat wiping it would be
the wrong default. An idle tab follows another tab's change, but never overwrites a box the user is
typing in. W9 (separate conversations) may revisit per-conversation prompts.

Gate: phases 6–7 of `scripts/webui_app_gate.mjs` read the actual request body — empty, set,
whitespace-only and padded prompts; not in the transcript; kept by New chat; restored and sent after a
reload; a hostile prompt inert; tab sync, and a focused box not clobbered. Six mutations, each red.
**Effort was: S.**

### W5 — Load a model from the page — DONE 2026-09-14
~~The pull flow dead-ends on its own success line: *"Downloaded, not loaded — restart the server with
--model &lt;path&gt; to serve it"*~~ — a finished pull now ends on a **Load it now** button
(`internal/serveapp/webui/ui/app.js:1581`). It loads the file, shows a heartbeat while the load runs,
refreshes the model list, and selects the new model, so the next message goes to it. The header
stats now follow whichever model is selected, not always the first one listed.

**The gate decision: a narrow route, not the admin load.** `POST /admin/models/load`
(`internal/serveapp/admin.go:113`) takes any caller-named path and stays behind
`-allow-admin`/`-admin-socket`, unchanged. The page gets its own `POST /web/models/load`
(`internal/serveapp/main.go:739`), registered only under `-web` and wrapped like pull
(`sameOrigin`, `auth`, body cap). It will load only a **regular `.gguf` file inside the pull cache**
(`webLoadPath`, `internal/serveapp/webui.go:346`). Symlinks are resolved on both the path and the
cache root *before* the containment check, and the resolved path is what gets loaded, so neither
`../` nor a symlink planted in the cache can point the loader outside it. The suffix check also
rejects a half-finished `.part` download. The page can load what it pulled, and nothing else.

Also decided here:
- **The server's own settings.** Backend, quant and KV come from the command line, with no
  per-request override. The page offers "load what you pulled", not a second admin API.
- **One load at a time.** A second load while one runs gets `409 already running`.
- **Closing the tab does not cancel a load.** The load keeps running, the model is still
  published, and the next model list shows it. A load stopped part-way would have wasted minutes
  for nothing. (A pull, by contrast, still stops when the tab closes and deletes its `.part` file.)
- **"Already loaded" counts as success.** The page selects that model instead of showing an error.
- **Loading and publishing are shared with the admin load.** Both go through `loadDecoder` and
  `publishLoaded` (`internal/serveapp/admin.go:171`), so the M-24 close-on-race and the session
  restore live in one place, and a web-loaded model unloads like any other.

**What this settles for W22** (the question was to be answered once, for both): a `-web` route
may act only on **what the page's own flows created**, behind the same gate stack as pull, and
**the admin routes are never widened**. Cancel and halt have no "created by this page" limit (they
act on generations from any client), so applying the rule to them is W22's own design call. The
rule is decided; how it applies to W22 is not.

Gates: the confinement table and the route tests in `internal/serveapp/webui_load_test.go` (13 path
cases, a symlinked cache root, streaming and publishing, heartbeat, single-flight, closing the tab
mid-load), plus the route guards in `webui_test.go` extended to the new route (auth, sameOrigin,
`-web`-only). Eleven mutations, each red. Phase 8 of `scripts/webui_app_gate.mjs` drives the page
(24 checks: the exact path posted, heartbeat, selection and stats, the next chat's model, three
failure modes that leave the button usable, already-loaded, hostile path and error text inert).
Twelve mutations, each red. End to end, `TestWebLoad_realModel` (gated on `GOINFER_SERVE_MODEL`)
loaded a real 0.5B GGUF from a fake pull cache through the route and got a chat answer from it.

Not done, noted for later: no route lists models pulled in *earlier* sessions, so after a restart
the page cannot offer those without a fresh pull. That is a small follow-up (list the cache) or
part of W9/W14.
**Effort was: M.**

### W6 — Fold the thinking block — DONE 2026-09-14 (page side)
~~Reasoning tokens stream inline as body text~~ — a reply's thinking is now folded into a collapsed
"Thinking…" section above the answer, relabelled "Thought for Ns" when it closes
(`splitThinking`, `internal/serveapp/webui/ui/app.js:357`). The server still does not split
thinking from the answer (`stop_reason` is never `thinking` in v1, `internal/serveapp/anthropic.go:35`),
so the page does it.

The recognised shapes come from real `serve` output, captured before writing any code, not from
documentation:
- **Qwen3**: the reply's content starts with `<think>\n…</think>`. The opening tag counts only as the
  very first thing in a reply, so an answer that merely mentions the tag is not folded. A `</think>` with
  no opening tag also folds everything before it, for templates that open the thinking in the prompt.
- **gpt-oss**: `<|channel|>analysis<|message|>…<|end|><|start|>assistant<|channel|>final<|message|>…`.
  The analysis channel is folded and the final channel is the answer, with no markers left on screen.

Also decided here:
- **Nothing raw flashes while streaming.** A tag split across chunks is held back until it is known
  to be a tag (or not). A reply's first `<th` shows nothing yet.
- **The fold is updated in place**, so a reader who opens it mid-stream keeps it open.
- **Copy copies the answer**, or the whole reply if there is no answer (stopped mid-thought).
- **Earlier replies go back to the model without their thinking.** This is what Qwen3's own template
  does with history, and it keeps long reasoning from filling the context window on every turn.
- **The saved conversation keeps the thinking and its duration.** A restored reply is folded again.
- **A finished reply that thought but never answered says so**, instead of showing an empty bubble.
  A stopped or interrupted reply does not, since its label already explains it.

**Server defect found here, FIXED 2026-09-14: gpt-oss never answered through `serve`.** Captured
before the fix (`gpt-oss-20b-MXFP4.gguf`, CPU): the stream held only the analysis channel, ending
with `finish_reason: stop`. The harmony template stopped on `<|end|>`, which closes the *analysis
message*, so generation ended before the model reached its `final` channel. The stops are now
upstream's own, from gpt-oss-20b's `generation_config.json`: `<|return|>`, `<|call|>` and
`<|endoftext|>` (`chat/templates.go`, pinned by `TestHarmony_stopsAreUpstreams`). Re-run on the same
model and prompt, the reply now carries the final channel (`17 × 23 = 391.`), and streaming and
non-streaming agree. The change reaches every user of the template, not just the page:
`goinfer-chat` and the demo agent read the same stops. The page's "No answer after the thinking" note
stays, for any model that really does stop mid-thought.

**Still open on the server:** the API's `content` carries gpt-oss's raw channel markers, and Qwen3's
`<think>` block, to every client. The page hides them; an OpenAI SDK client sees them. Splitting
thinking into its own field (or content block, on the Anthropic route) is its own server item.

Gate: phases 9–10 of `scripts/webui_app_gate.mjs`, 35 checks. A DOM observer fails the gate if any raw
tag fragment is ever on screen during streaming. Streams are fed chunk by chunk to test the held-back
tags and the fold staying open, and there are checks for the template-opened and gpt-oss shapes, a
mention of the tag, hostile content, stopping mid-thought, the history sent, and restore after a reload.
Two replies of **real captured serve output** (`scripts/webui-gate/captured-thinking.json`, Qwen3-1.7B
and gpt-oss-20b after the stop fix, real token boundaries) are replayed chunk for chunk. Fifteen mutations, each red.
**Effort was: M.**

### W7 — Regenerate, edit-and-resend, delete a turn — DONE 2026-09-14
~~None of the three~~ — all three, as buttons on each message's action row next to Copy
(`internal/serveapp/webui/ui/app.js:841`). History is now addressable: each action changes the
transcript, re-renders the log from it and saves, so what is shown, saved and sent can't disagree.

- **Regenerate** appears only on the **last** reply, including a stopped or interrupted one. It
  resends the history up to the last question and replaces the reply. It is not offered on earlier
  replies, where it would have to silently discard everything after them; Edit covers that case.
- **Edit** opens one of your messages in place (Ctrl/Cmd+Enter saves, Escape cancels). Saving replaces
  the message, drops everything after it, and resends from there. If that would drop more than the one
  reply being replaced, it asks first and says how many messages go. An empty edit keeps the box open.
- **Delete** removes a whole **exchange** (your message and its reply), after asking. There is no
  undo. Deleting half an exchange would leave two user turns in a row, which some chat templates refuse.
- **Nothing changes the conversation while a reply is generating or a message is open for editing.**
  The buttons are disabled, and the functions refuse on their own too, since W17's keyboard shortcuts
  will call them with no button in between. Send can't fire during an edit either, not even by
  keyboard. Another tab's change doesn't close an open edit, and New chat cancels it.

Decided here, and why: **edits truncate; they don't branch.** A branching history (keep both versions,
switch between them) is what the Claude app does. It needs conversation storage to hold a tree, which is
W9's territory (separate conversations). Truncation with a warning is the honest single-thread version.

Gate: phases 11–12 of `scripts/webui_app_gate.mjs`, 30 checks. After each action they check exactly
what is sent, shown and saved; they also cover the disabled states (including direct calls), the confirm
prompts and declining them, Escape and Ctrl+Enter, a hostile edit inert, and the actions and Regenerate
working after a reload. Seventeen mutations, each red on its own W7 checks.

**Found while proving it: the browser gates were leaking Chrome.** `scripts/webui-gate/cdp.mjs` killed
the `google-chrome` launcher script, not the browser it starts. Every gate run since W1 left a Chrome
running (181 had built up), and the random debugging port sometimes landed on a leaked browser still
holding an old page and old localStorage. That made two W7 mutation runs fail on *unrelated* W3/W4
checks. Fixed: Chrome now picks its own port and writes it into the run's own profile directory, so a
run can only reach its own browser; the whole process group is killed; the profile is removed. After
17 back-to-back mutant runs, 0 browsers remain. **What this means for earlier records:** a collision
could have made a mutant look red for the wrong reason. The kept W5 and W6 mutation outputs all fail on
their own checks. The W1–W4 runs kept only final outputs, so they are not individually re-verified,
though fewer browsers had leaked then and a collision was much less likely.
**Effort was: M.**

### W8 — Context meter, and a warning before the wall — DONE 2026-09-14
~~You discover the context limit by hitting `400 context_length_exceeded` mid-conversation~~ — the
composer now shows *"about N of W tokens used (P%)"* with a bar
(`showContext`, `internal/serveapp/webui/ui/app.js:86`). At 80% it warns that the conversation is
nearing the limit. At 95% it says the next message will likely not fit, and what to do: start a new
chat, or delete earlier exchanges (W7). If a request does hit the wall, the error says the same thing
in plain words, with the server's own message underneath. Other 400s are left as they are.

**Server: `/v1/models` (and `/health`) publish `context_window`.** It comes from one function,
`contextWindow` (`internal/serveapp/openai.go:821`), which `prepare` also uses to enforce the limit,
so the number a client plans against is exactly the one that rejects it. On a resident GPU backend
that is the resident KV cap, not the model's `MaxPositions`. Measured on this box (CUDA, Qwen3-1.7B):
`context_window: 8192` rather than Qwen3's native maximum. A prompt of 8192 tokens is rejected naming
8192, one of 8191 is served, and its reported usage totals exactly 8192.

**"Used" is the server's own count.** The page now requests `stream_options.include_usage`. "Used" is
the latest reply's `prompt_tokens + completion_tokens`, saved with that reply, so it survives reloads
and follows W7's deletes. Per-reply stats now show the server's `completion_tokens` too. Counting
stream chunks, as the page did before, undercounts whenever a partial UTF-8 rune or stop string holds
a token back. A server that sends no usage falls back to counting chunks.

Decided here, and why:
- **Approximate on purpose, and labelled "about".** The next request also carries your new message,
  and sends earlier replies without their thinking (W6). A prediction that pretended to be exact would
  be worse than an honest estimate.
- **Measured against the selected model's window.** After switching models the count comes from the
  previous model's tokenizer, so it's close but not exact until the next reply.
- **No meter** when the model publishes no window, or before any reply has reported usage.

Gate: `TestServe_contextWindowIsTheEnforcedOne` (published = enforced, at the exact boundary) and
`TestServe_prepareEnforcesThePublishedWindow` (the wiring for the resident case CI cannot load), with
three server mutations, each red. Phases 13–14 of `scripts/webui_app_gate.mjs`, 19 checks: usage
requested and used, both thresholds, the capped bar, re-measuring on delete and model switch, no window,
the wall explained (and only the wall), usage saved, restored, and unreadable stored usage ignored.
Fourteen page mutations, each red. Writing the gate found one real bug on the way: New chat left the old
conversation's meter showing.
**Effort was: M.**
**Correction 2026-09-15:** the empty context meter (before any usage, or on a model with no window) shipped visible when it should have been hidden. A CSS `display`
rule on its class overrode the `hidden` attribute, and the gate only read `el.hidden`, which stayed true.
Fixed with a global `[hidden]{display:none !important}`, and every gate phase now ends by checking that
no element marked hidden takes up space (found by W15's screenshot pass).

---

## 3. Tier B — the ten that make it a daily driver (in scope)

Six are S. Each is a reason someone opens Open-WebUI or LM Studio instead of the page that shipped
inside the binary.

| # | Item | Today | Effort |
|---|---|---|---|
| **W9** | Conversation list with generated titles | **DONE 2026-09-14** — see below | M (after W3) |
| **W10** | Full sampling controls | **DONE 2026-09-14** — see below | S |
| **W11** | Image attach for vision models | **DONE 2026-09-14** — see below | M |
| **W12** | Fit verdict before a multi-GB pull | size only. **Already scoped** — `task-fit-to-hardware.md` §3; `pull.File` carries `Size` (`pull/pull.go:179`) | M — **skipped for now** (owner, 2026-09-14) |
| **W13** | Errors that say what to do | **DONE 2026-09-14** — see below | S |
| **W14** | Export the conversation | **DONE 2026-09-14** — see below | S |
| **W15** | Dark mode | **DONE 2026-09-15** — see below | S |
| **W16** | Phone layout | **DONE 2026-09-15** — see below | S |
| **W17** | Enter sends, ↑ edits last, Esc stops | **DONE 2026-09-15** — see below | S |
| **W18** | Label which turn came from which model | **DONE 2026-09-15** — see below | M |

### W18 — Label which turn came from which model — DONE 2026-09-15
The reply header already showed the model's name. What was missing is what the row names: nothing marked
a **change** of model, and the per-reply stats had no context to be compared in
(`labelReply`, `internal/serveapp/webui/ui/app.js:946`).

- **Each reply records the compute path its model reported when the reply was sent** (`decode_path` from
  `/v1/models`, e.g. `cuda-resident (int4)`). It is shown beside the model name, saved with the reply, and
  exported in its heading. It is recorded at send time because a model can later be reloaded onto another
  backend, and the stats are only comparable along with the path they were measured on.
- **A divider marks each place where consecutive replies came from different models** ("Model changed:
  alpha-4b → beta-9b"). It is worked out from the conversation, not stored, so it follows Regenerate
  (regenerate the reply on the first model and the divider disappears), Edit and Delete. A request that
  failed before producing any text leaves no divider behind.
- Model names and paths from storage are shown as text; an over-long stored path is dropped.

Not done: side-by-side "ask both models" comparison. That is two generations for one turn, closer to
W7-style branching than to a label.

Gate: phases 33–34 of `scripts/webui_app_gate.mjs`, 15 checks: label and path, saved; no divider without a
change; the divider exactly before the new model's reply; relabelling and the divider following a
Regenerate; a model with no path; no divider after a failure; the path in the export; after a reload, the
exact order of messages and dividers; hostile or over-long stored labels. Ten mutations, each red.
**Effort was: S** (less than the M the row estimated, since the name was already there).

### W17 — Enter sends, ↑ edits last, Esc stops — DONE 2026-09-15
~~Only Ctrl/Cmd+Enter~~ — as the row asked, **a setting, not a swap** (`internal/serveapp/webui/ui/app.js:1434`).
Ctrl/Cmd+Enter always sends. **Enter sends** is a checkbox beside the composer, off by default (long
prompts want Enter for new lines). When on, Enter sends and Shift+Enter is a new line. The placeholder
says which is in effect. The setting is saved, followed across tabs, and applies to the **edit box** too
(W7), so one habit works in both.

- **An input method mid-composition never sends.** In Japanese, Chinese or Korean input, Enter
  *confirms* the characters, and a send there would ship half a word. Both signals browsers use are
  honoured (`isComposing`, and key code 229). The rename box (W9) gets the same guard.
- **↑ in an empty message box** opens your last message for editing (W7). With text in the box, ↑ just
  moves the cursor.
- **Esc stops a reply that is generating**, from anywhere on the page, unless something closer used it:
  in an edit or rename box, Esc still cancels that.

Gate: phases 31–32 of `scripts/webui_app_gate.mjs`, 24 checks. The keys are delivered as real `keydown`
events with modifiers, `isComposing` and key code 229. The checks cover what is sent and whether the key
was swallowed, both settings in the message box and the edit box, ↑ with and without text, Esc stopping
a reply and cancelling an edit, the rename box under composition, and persistence and tab sync.
Fourteen mutations, each red.

**A gate defect the mutations found:** with Esc-to-stop removed, the check waited on a reply that could
never end, and the gate **hung** instead of failing (a hang in CI is a timeout, not a clear red). That
check now has a 5-second deadline and fails.
**Effort was: S.**

### W16 — Phone layout — DONE 2026-09-15
~~One `max-width:920px` column, no media query~~. The cards already stacked. The header was what broke a
phone: brand, stats, Book, Theme and both tabs in one row forced a 400 px screen's layout to **446 px**
(sideways scrolling) and collapsed the Theme select to 12 px. At 600 px and below the header now wraps into
rows (brand and tabs, then the model stats, then Book and Theme) and the gutters tighten. On a narrow
screen or a coarse pointer, the small text-sized controls grow to at least 32 px tall. Several had been
under WCAG 2.5.8's 24 px minimum: the Export buttons (21 px) and the System prompt / Sampling toggles
(20 px). Desktop with a mouse keeps its density.

Found by looking, which is the only reason they were found: screenshots from headless Chrome at 400 and 360
px, in both themes. The same pass caught one regression of this item's own (a `display:flex` summary loses
its disclosure triangle) and two older bugs (hidden elements that still rendered, and the Chats list laid
out sideways), fixed in their own commit.

Gate: phases 28–30 of `scripts/webui_app_gate.mjs`, 24 checks. They use a real mobile metrics override at
400 and 360 px, with a page crowded with what breaks narrow layouts: an unbroken 300-character URL, a
400-character code line, a 12-column table, an error with buttons, a pending image and an open edit box.
The checks:
- the page does not scroll sideways;
- nothing reaches past the screen edge except inside its own scroll box;
- the code and table can **actually be scrolled** to their end (a clipping box reports the same sizes);
- every visible control is at least 24×24 px (83 checked);
- header items don't overlap, and the stats get their own row;
- the key controls are on screen, and the toggles keep their triangles;
- at desktop width, the header is still one row, with desktop padding and the 920 px column.

Ten mutations in all. The first run left three survivors. Two were rules nothing needed once the header
wrapped (removed, not tested), and one needed a sharper check: scrolling the code block instead of comparing
sizes. A later survivor (phone rules applied on desktop) got the desktop-padding check.
**Not covered:** on desktop, the dense controls stay under 24 px tall. WCAG 2.5.8 allows that when targets
are spaced apart, which was not measured here.
**Effort was: S–M.**

### W15 — Dark mode — DONE 2026-09-15
~~One light surface; no `prefers-color-scheme` rule anywhere~~ — a **Theme** control in the header:
**System** (the default, following `prefers-color-scheme`), **Light** or **Dark**, saved per browser,
followed across tabs, and applied before anything else on load (`internal/serveapp/webui/ui/app.js:7`).

**Only tokens change.** The dark palette overrides the `--gi-*` tokens in two places: under the system
preference unless the page is set to Light, and whenever Dark is chosen. `color-scheme` goes with it,
so native controls and scrollbars follow. AmbientCSS shades a surface as albedo × light, so a dark
albedo is most of the change. Its light-tuned chamfer highlights were too bright on dark (the ambient
design notes called light and dark "two different tuning problems"), so dark also dims the key and fill
lights (0.6 / 0.55), tuned by looking at screenshots from headless Chrome.

**Contrast is measured, not chosen by eye, in every mode.** The gate computes the WCAG ratio for every
text colour on every real surface: the page, a card, the header, a reply bubble, and the message field
at **each stop of its gradient**, plus white on the Send button. It does this under System-light,
System-dark, Light forced over a dark preference, and Dark forced over a light one. AA text (4.5:1)
holds everywhere. The worst cases are 4.51:1 (light, the Send label) and 6.60:1 (dark).

**That measurement found real failures in the existing light theme**, which the ambient pass had checked
against the flat page colour only. On the shaded end of a reply bubble (rgb 212,217,231):
- accent text was 3.51:1 (every link and "Thinking…"),
- muted grey was 4.14:1 (every reply's stats line),
- success green was 3.66:1,
- warning ink was 4.19:1.

The fixes: accent text `#0a5cb3`, muted `#535c67`, green `#186534`, warning `#744b0f`, each 4.57–4.95:1.
The Send button keeps its designed `#0b6fd4` as a separate fill token, `--gi-accent-fill`. The same pass
found two layout bugs from earlier items, now fixed in their own commit (see the W8, W9 and W11
corrections). `docs/completed/task-web-ui-ambient.md` carries a correction next to its old figures.

Gate: phases 25–27 of `scripts/webui_app_gate.mjs`, 13 checks. The browser's colour scheme is switched
from Node between phases (DevTools emulation, via `cdp()` added to `cdp.mjs`). **Headless Chrome defaults
to a dark preference**, so the light phase sets light explicitly. Eleven mutations, each red.
**Effort was: M** (the contrast fixes were the unplanned part).

### W14 — Export the conversation — DONE 2026-09-14
~~Nothing~~ — **Export this chat: Markdown · JSON** beside the Chats heading
(`chatMarkdown`, `internal/serveapp/webui/ui/app.js:1288`). No share link: that would be a server-side
copy, the anti-goal this row names.

- **Markdown** is for reading. A `## You` / `## <model>` section per turn, attached images embedded as
  data URIs so the file stands alone, thinking folded in `<details>`, and each reply's stats and state
  (stopped, incomplete, cancelled) in italics.
- **JSON** is for keeping or processing. The conversation exactly as stored, plus
  `format: "goinfer.chat"`, `version: 1` and `exported_at`.
- **The system prompt is exported labelled "at export time".** It is a page setting (W4), not recorded
  per message, so the file must not imply the conversation was had under it.
- **File names** are the title reduced to letters, digits and dashes, plus the date, and never a path:
  `../../Rust: ownership & borrowing?` becomes `rust-ownership-borrowing-<date>.md`. A title with nothing
  file-safe becomes `conversation-<date>`.
- Disabled on an empty chat and while a reply is generating.

Not done: **import**. JSON export is the half that makes import possible later; nothing asked for it yet.

Gate: phase 24 of `scripts/webui_app_gate.mjs`, 13 checks. The download is intercepted and the file's
exact contents compared, for a conversation built to include every shape: an image, thinking, stats,
a failed reply, a two-line system prompt, and a hostile title. Also covered: the fallback name, a null
system prompt, and disabled when empty or busy. Ten mutations, each red.
**Effort was: S.**

### W13 — Errors that say what to do — DONE 2026-09-14
~~Any non-200 becomes the first 400 characters of the response body in a red bubble~~ — every failure
is explained as what happened and what to do, with a button for the remedy (`problemFor`,
`internal/serveapp/webui/ui/app.js:122`). The server's own message stays underneath, as text.

| What the server said | What the page says | Buttons |
|---|---|---|
| no HTTP answer at all | Can't reach the server. Check that goinfer serve is still running. | Retry |
| 401 | The server needs an API key. Enter it on the Models tab. | Enter API key, Retry |
| 404 | The model "X" isn't loaded. | Refresh models, Retry |
| 413 | This request is larger than the server accepts. Remove an image, or start a new chat. | New chat |
| 429 | The model is busy: its request queue is full (with the server's `Retry-After`). | Retry |
| 503 `{"error":"halted"}` | The server has halted new generations: the reason, and that an admin must resume it. | Retry |
| 503 | The server is at capacity (with `Retry-After`). | Retry |
| 400 `context_length_exceeded` | W8's wall. | New chat |
| other 4xx / 5xx | The server rejected the request / hit an error, with its message. | — / Retry |

**Failures part-way through a reply keep what arrived.** Before this, a dropped connection or a server
error mid-reply discarded a partial answer that had already been saved as it streamed. Now it stays,
labelled *"incomplete — the connection to the server was lost"* (or the server's error), and Regenerate
is the retry. The server reports a mid-stream failure as an `{"error":…}` event, not a status, and the
page used to ignore those entirely. A failure before any text is still not a turn: it is an explained
error message with Retry.

**How a reply ended, when that is actionable:** `finish_reason: "length"` adds *"stopped at the Max
tokens limit — raise it to get more"*, and `"cancelled"` (a K1/K2 halt or cancel) says *"cancelled by
the server"*. Both are saved and come back after a reload. The model list, too, says *"API key needed
— enter it on the Models tab"* or *"can't reach the server"* instead of `HTTP 401`.

**Measured, not assumed:** every HTTP body the gate replays was provoked from a real `serve`
(Qwen2.5-0.5B, CPU) and saved in `scripts/webui-gate/captured-errors.json`. Those were `-api-key` with no key, an unknown model,
`-max-queue 1` and `-max-inflight 1` under concurrent load (both carry `Retry-After: 1`, which is why
the 503 message uses it too), a K5 halt through `-admin-socket` (body exactly
`{"error":"halted","reason":"maintenance window"}`), and a 30 MB prompt. That last one is **not** a 413
at default settings: the context pre-check rejects it first as `context_length_exceeded`, so the 413
row rests on the server's standard error shape, not a capture.

**A real bug the gate caught:** the first version told a user to enter the key "at the top of the
page", and its button focused the field. The key field is on the **Models** tab, hidden while
chatting, so the words were wrong and the button did nothing. Both now point at the Models tab, and the
button opens it.

Gate: phases 22–23 of `scripts/webui_app_gate.mjs`, 29 checks: each captured error explained exactly;
no raw JSON shown; Retry, Enter API key, Refresh models and New chat each doing their job; an
unreachable server; a mid-reply server error and a dropped connection keeping the partial answer
(saved as `failed`, regenerable, hostile message inert); a pre-text server error; `length`,
`cancelled` and a normal stop; the model list's errors; the labels after a reload; unknown stored
states dropped. Nineteen mutations, each red. One survived at first (the stored-state whitelist,
invisible on screen); its check now confirms an unknown state is not written back.
**Effort was: M.**

### W11 — Image attach for vision models — DONE 2026-09-14
~~No control, though `-vision` works on the same route~~ — an **Attach image** button beside Send, plus
drop anywhere on the chat pane or paste into the message box, with a preview and Remove
(`internal/serveapp/webui/ui/app.js:205`). The image shows on the message it was sent with, is saved
with it, and comes back after a reload.

**Server: `/v1/models` and `/health` publish `vision` per model**, from `visionCapable` — the same check
the vision path uses to refuse an image with "this model has no vision tower". The page shows Attach
only when it is `true`, which is the "per-model capability check" this row asked for, done from the
server's own answer rather than a guess from the model's name.

What the server allows decided most of the rest:
- **One image per request, counted across the whole history** (`maxImagesPerTurn = 1`,
  `internal/serveapp/vision_serve.go:20`). The server attaches it to the *latest* user turn, wherever it
  was sent. So the page sends **only the most recent image**, as an `image_url` part on the message it
  came with; earlier ones go as plain text. Follow-up questions about that image still work. Attaching
  a second image says it replaces the first in what the model sees.
- **PNG and JPEG only.** Those are the decoders the process registers. Anything else is drawn onto a
  canvas and re-encoded as JPEG in the browser. So is anything larger than **1344 px on its longest
  side** or over 1.5 MB, scaled to fit. Vision towers resize to their own input anyway (SigLIP: 896 px),
  and a phone photo kept whole would fill browser storage in a few messages.
- **A text-only model is sent no images.** Attaching on one is refused, a pending image blocks Send with
  a reason, and a conversation with images says they won't be sent. Otherwise the request would be a 400.

**Security:** a stored or pending image is rendered, so it must be a strict PNG/JPEG base64 `data:`
URI. A remote URL (which would let a stored conversation make the page fetch something), a script
scheme, SVG, or anything malformed is dropped, not rendered or sent. The page also carries no `<img>`
at all until the user attaches one; the preview is created only while an image is pending.

Measured end to end before gating (CPU, Qwen2.5-VL-3B, requests shaped like the page's, a generated
red|blue test image):
- `/v1/models` reported `vision: true`.
- Text plus image: "Red and blue", in 3 s.
- A follow-up question, with the image on the earlier message: "Red" for "which color is on the left?".
- An image with no text: accepted.
- Two images in one request: a 400, "v1 supports 1 image per request", which is why only the newest is sent.

Gate: `TestServe_modelsReportsVision` (present and false for a text-only model, true with a tower, and
red on a wrong capability check), and phases 20–21 of `scripts/webui_app_gate.mjs`, 28 checks. The page
makes its own test images on a canvas. They cover picker, paste and drop; a small PNG kept byte for
byte; WebP converted and a 3000×1000 PNG scaled, both **checked by decoding the result**; exactly the
content parts sent; only the newest image; none to a text-only model; non-image, corrupt and
over-25 MB files refused; image-only send and edit; nothing attachable mid-reply; saved, restored, and
hostile stored images never rendered or sent. Twenty-four mutations, each red. **Not gate-able:**
cancelling `dragover`. Real browsers need it for a drop to happen at all, but a synthetic drop event
fires regardless, so a gate cannot tell. It was checked by reading the code, not by a mutation.
**Effort was: M.**
**Correction 2026-09-15:** the image preview row, with its "Remove" button, shipped visible when it should have been hidden. A CSS `display`
rule on its class overrode the `hidden` attribute, and the gate only read `el.hidden`, which stayed true.
Fixed with a global `[hidden]{display:none !important}`, and every gate phase now ends by checking that
no element marked hidden takes up space (found by W15's screenshot pass).

### W10 — Full sampling controls — DONE 2026-09-14
~~The page sends `temperature`/`max_tokens` only~~ — a collapsible **Sampling** section beside the
system prompt adds every other sampling field `/v1/chat/completions` accepts: `top_p`, `top_k`, `seed`,
`stop`, `frequency_penalty` and `presence_penalty` (`internal/serveapp/webui/ui/app.js:641`).
Temperature and max tokens stay in the top row.

**A correction to what this row used to say:** it listed `logit_bias` among the fields the route
accepts. It does not. The request struct (`internal/serveapp/openai.go:456`) has no such field. The
sampler supports `LogitBias`, `MinP` and `RepeatPenalty`, but the HTTP route exposes none of them, so
W10 covers what the route actually takes. Exposing the other three is a server item of its own.

Decided here, and why:
- **A setting, like the system prompt:** its own key, kept across reloads and New chat, followed across
  tabs, but never over a field the user is typing in. **Reset** returns to the defaults.
- **Blank means not sent**, so the server's default applies. A blank seed is a fresh random one each
  time (M-03).
- **Ranges are OpenAI's documented ones, and an out-of-range value is refused, not clamped.** The
  server rejects only some of them (negative temperature, `top_p` outside [0,1]) and passes the rest
  through unchecked, so the page checks all of them. The field is marked and the reason shown. Send,
  Regenerate and Edit all refuse **before changing the conversation**; otherwise a regenerate could
  remove the old reply and then fail to send. A seed must be a safe JavaScript integer, since a larger
  one cannot round-trip exactly.
- **Stop sequences, one per line.** `\n` and `\t` stand for a newline and a tab, since a line can't
  hold a newline; blank lines are ignored.
- **The background title request (W9) keeps its own fixed settings**, whatever is set here.

Measured on a real model before writing the gate (CPU, Qwen2.5-0.5B, request bodies shaped like the
page's). The same seed twice gave identical output, and a different seed gave different output.
`top_k: 1` at temperature 1.5 matched greedy decoding exactly. A `,` stop sequence cut the reply to
"apple". Penalties of 2 changed the output. `top_p: 1.5` came back as a 400 from the server itself.

Gate: phases 18–19 of `scripts/webui_app_gate.mjs`, 39 checks. They cover exactly what is sent for
defaults, set fields and cleared fields; stop parsing; the title request unaffected; ten out-of-range
values, each marked, explained and refused before anything changes (including by Regenerate, Edit, and
`generate()` called directly); saving, tab sync, reset, restore after reload, and unreadable stored
values ignored. Twenty mutations, each red, run detached with the source backed up (W9's lesson).
**Effort was: S–M.**

### W9 — Conversation list with generated titles — DONE 2026-09-14
A **Chats** list above the settings card. New chat now starts another conversation and keeps the one
you were in, so it no longer asks first. Each conversation can be opened, renamed or deleted (deleting
asks, since there is no undo). The list is most-recently-updated first, and the open conversation is
highlighted. Nothing in the list works while a reply is generating or a message is being edited.

**Storage** (`internal/serveapp/webui/ui/app.js:489`): one localStorage key per conversation,
`goinfer.chat.v2.<id>`, holding `{v, id, title, titled, updated, messages}`.
- **One key each, not one key for all:** the once-a-second save while a reply streams rewrites only
  that conversation, and an unreadable value costs only itself (it is set aside, as in W3).
- **No separate index:** the list is read from the keys, so there is nothing to fall out of step.
  A value whose `id` doesn't match its key is treated as unreadable.
- **The open conversation is per tab** (sessionStorage), so two tabs can have different chats open;
  a new tab opens the most recent one.
- **An empty New chat is never stored:** it isn't a conversation until something is said in it.
- **Migration:** W3's single conversation (`goinfer.chat.v1`) is migrated once into a conversation of
  its own, opened, and the old key removed.

**Titles** (`autoTitle`, `internal/serveapp/webui/ui/app.js:1358`). A title starts as the first message,
cut to fit. After the first reply, the model is asked **once**, in the background, with a non-streaming
24-token request, for a title of at most six words. The reply is cleaned: first line only, no
"Title:" label, quotes, markup or final period, and it is rejected if it still contains thinking or
channel markers.
- **Your reply always wins:** the request is cancelled as soon as you ask for a reply, and re-asked
  after that reply.
- **A rename always wins:** a title that arrives after a rename is discarded.
- **An unusable reply is not retried:** the conversation keeps its first-message title.

Measured before deciding (CPU, the page's exact request): **Gemma 3 1B** gave "Capital of France",
"Date Parsing Function" and "Deadly good?" in about 0.6 s each. **Qwen3-1.7B** spent all 24 tokens
thinking (about 2 s), so it keeps the first-message title. Qwen3's `/no_think` switch was tried and
**not adopted**: it made Qwen3 reply, but with worse titles than the first message ("Paris is the capital
of France.", "sticker on starter"), and it would put a model-specific switch into every model's prompt.

Decided here, and why:
- **The system prompt stays global** (W4), not per conversation. Nothing asked for it, and a per-chat
  prompt is a bigger change to what W4 settled.
- **Edits still truncate** (W7). Branching now has somewhere to live, but it is its own item.
- **The list sits above the chat, not in a sidebar**, with a scroll cap. A layout that uses a wide
  screen belongs with W16's phone layout, which will need a media query anyway.

Gate: phases 15–17 of `scripts/webui_app_gate.mjs`, 40 checks, plus the W3/W4/W6/W7/W8 phases moved onto
the new storage. The prelude answers the page's background title request itself, so it never becomes
the "last request" other checks inspect. Twenty-five mutations, each red. Proving them found four gaps
in the first version of these checks, each closed: `save()` on an empty chat, cancellation actually
observed, a reasoning model that finishes its thinking, and renaming the *open* conversation. The
cleaner itself had a real bug, caught before any mutations: `**Title**.` kept its `**`, because the
period sat outside the markup. One run was cut off by a disconnect and left a mutant in the source;
it was caught by checking every mutant's original text, restored, and the whole run redone detached
with a backup that refuses a restart over a dirty file.
**Effort was: M.**
**Correction 2026-09-15:** the list rendered as a row squeezed beside its heading. The header's
`nav{display:flex}` also matched `<nav id="chats">`. Fixed with `#chats{display:block}`, and the gate now
checks that the list is laid out below its heading at full width (found by W15's screenshot pass).

---

## 4. Tier C — projects, not sessions. In scope as of 2026-09-14; each needs its own design decision first (§1).

- **W19 — Tool calls visible in the thread.** The server does tool calling and constrained JSON;
  `demo/agent/cmd/agent-web/index.html:198` has the collapsible chip UI for exactly this. The
  blocker is architectural: someone must run the tool loop, in the browser or behind a new server
  endpoint. **This is the F.3 question in its sharpest form.** L.
- **W20 — File attach.** Text and source files are S (read, fence, prepend). PDF is L and is really
  a dependency decision — extraction in pure Go with no new root module dependency, against
  `task-embed-and-harness-ux.md` §0's rule.
- **W21 — Search across conversations.** Needs W3 and W9 first. Then it is the obvious place to
  dogfood skiff: a client-side WASM index over local transcripts, running inside the engine's own
  UI, is that product's pitch demonstrated rather than described. M–L.
- **W22 — Admin panel: running generations, cancel, halt.** K1/K2/K5 shipped and are CLI/socket
  only, so a person watching a slow generation in the browser can neither see nor stop it. The gate
  rule is W5's (a `-web` route acts only on what the page created, and admin routes are never
  widened). It already settles W28 and the own-jobs half of W31. What it leaves open is exactly this
  item and W31's other-clients half: acting on work the page did not create. M.
- **W23 — Structured-output workbench.** Paste a JSON schema, watch a grammar-constrained
  generation fill it, see the token cost. `response_format` is live. The README's "a Go struct the
  model cannot violate" currently has nowhere a visitor can see it work. M.
- **W24 — Projects / knowledge / RAG.** ken + aikit + `/v1/embeddings`. A stack decision, not a UI
  one. XL.
- **W25 — Rendered preview of generated code.** The sandboxing *is* the job: a sandboxed iframe at
  a null origin, or not at all. L.
- **W26 — Prompt library.** M, and the weakest pull on the list for this audience.
- **W33 — A model picker instead of blind text entry.** Filed 2026-09-15 from a live session on
  this box: asked to load a bigger coding model, the obvious move — pull `Qwen2.5-Coder-7B-Instruct`
  over the just-fixed pull flow, Load it — silently landed on single-threaded CPU at ~2 tok/s. The
  server was still running `-quant int8int8` from an earlier, smaller model, so the 7B's weights
  needed 7.6 GB against 6.5 GB free (the smaller model was also still resident — load does not
  auto-evict, W32's own rule) and CUDA declined; nothing on the page said any of that, or offered a
  way to see or change it. Two related gaps, both real, both worth naming separately:
  1. **Repo entry is free text with no memory of what's known-good.** `pull.Curated()` /
     `CuratedNames()` (`pull/pull.go:96`, `:110`) already exists — reachable today only by typing
     `demo:0.5b` into the same box (`#repo`, `internal/serveapp/webui/index.html:98`), undiscoverable
     unless someone already knows the trick. A dropdown of curated names beside the free-text box —
     pick one, or paste a repo — is a small, additive change over what already exists.
  2. **The quant/backend a model loads at is invisible and unchangeable from the page.** `-quant`
     and `-backend` (`internal/serveapp/main.go:497`, `:502`) are server-startup flags with "no
     per-request override" (W5's own decision, §2). That was the right call for *requests*; it is
     what made *this* incident invisible — the page had no way to show what quant was about to be
     used, or that a bigger model would blow the budget under it. Arguably the higher-value half of
     this item: a visible, changeable quant control on the load flow would have caught the problem
     before it happened, not after.
  Both still owe the design decision §1 asks of every Tier C item — in particular, part 2 changes
  what W5 called settled ("the server's own settings … no per-request override"), so it needs its
  own explicit re-examination, not an assumption that W5's reasoning still holds unchanged for a
  *load-time* (not per-chat-request) choice. M for either half alone; committing to both together
  is more, and they do not have to ship in the same pass.

---

## 5. Not worth building here, stated

Named so they do not return as "gaps" in a later pass.

- **Accounts, sync, sharing links, org admin.** There is no server to sync to, and adding one
  inverts the product.
- **Voice in or out.** Needs model classes goinfer does not serve.
- **Several conversations generating at once.** One decode worker per model is deliberate, with
  `-max-queue` backpressure behind it. The UI may show queue position
  (`task-work-queue-2026-09.md` J9); parallel decode is J8's measurement to earn, not a UI feature.
- **Connectors / MCP.** A harness's job, and mode 3 is "point my tools at it" — the tools have this
  already.

---

## 6. Three constraints that govern the build

**6.1 One embedded file stops scaling here. — DONE 2026-09-14.** Tier A alone roughly doubles 1,828
lines. The no-build-step, no-external-asset property is worth keeping exactly as it is. The move that
keeps it and fixes the file is **`//go:embed` over a directory** — several source files, still no
toolchain, still one binary, still offline. What not to do is add a bundler; that trades away the
property the project exists for.

*Done as specified:* `index.html` (66 lines) + `ui/app.css` + `ui/app.js`, served by a new
unauthenticated `GET /ui/{file}` route with an explicit type allow-list, `nosniff`, `no-store`, and
404 for anything else, registered only under `-web`. Assets are referenced relatively, so the page
also loads from `file://`. Proven behaviour-preserving three ways: inlining the two files back
reproduces the original byte for byte; in headless Chrome the post-JS DOM is identical and all 57
elements have identical computed styles, with zero exceptions before and after; and the web UI tests
now scan every embedded file, fail on a reference to a non-embedded asset, and pin the route's type,
gating and auth — each shown able to go red by a mutation.

**6.2 W1 is where model output stops being inert, and the current rule must survive it. — HELD
2026-09-14 (see W1).** The page says it in the source: content goes in via `textContent` or
`Markdown.render`, never `innerHTML` (`internal/serveapp/webui/ui/app.js:935`), and a test now fails the
build if that changes. **Build DOM nodes from the parsed tree; never assemble an HTML string.** Otherwise a model — possibly one pulled from a stranger's Hugging Face repo
minutes earlier, by this very page — gets script execution on the same origin as the API, with the
user's key in a field on that page. The gate for W1 is a test that feeds the renderer hostile
markdown (`<img onerror>`, `javascript:` links, raw `<script>`) and asserts no element is created
outside the allowed set.

**6.3 Every new mutating route inherits the gates already there** — `sameOrigin`, the `-web`
opt-in, and the non-loopback `-api-key` rule (`internal/serveapp/webui.go`). V-20 in
`docs/completed/review-2026-09-04.md` is the record of what it cost to learn that the first time.
W5's load route was the first real test of them, and takes all of them (W5).

---

## 7. The queue surface, and unload — W27–W32, added 2026-09-15

`task-work-queue-2026-09.md`'s J1–J4 shipped 2026-09-15, which makes five things buildable that
were not when this doc was filed. They are **Tier B class** — daily-driver items, in scope under
§1's decision, no further sign-off — and they are numbered from W27 so the existing items keep
their recorded numbers.

**A correction this section also makes.** Until now both this doc and the queue doc cross-referenced
"W18" as the re-attach item. W18 is *Label which turn came from which model*; re-attach never had a
number. It is **W27** below, and the four stale references have been repointed.

### W27 — Re-attach to a generation after a reload — DONE 2026-09-15
~~Closing the tab still cancels the work~~ — a text reply now submits as a server-side job
(`POST /v1/jobs`) streamed through J3's `GET /v1/jobs/{id}/events`
(`internal/serveapp/webui/ui/app.js:1032`, the `asJob` branch of `generate`), instead of the plain
chat stream. The job id is saved with the reply the moment the server returns it, before anything
else can go wrong, so a reload — or opening another conversation mid-reply (W29) — no longer loses
it. Opening a conversation whose last reply is a still-running job re-attaches and replays the job's
events from the start (`resumeJob`, `internal/serveapp/webui/ui/app.js:1244`, called from
`openChat`, `:1306`); a job that finished while nobody was watching is filled in from its own
finished record instead; a job the server no longer has (a restart without `-job-dir`, or
expiry) is labelled "not finished here" with a Resume button, not left spinning forever. Stop now
DELETEs the job on the server (`stopReply`, `:1225`), not just the local stream — before this, Stop
only ended the page's own copy, leaving the job to run to completion for nobody. A reply carrying an
image still uses `/v1/chat/completions`: jobs take text only (`asJob = !messages.some(m =>
Array.isArray(m.content))`), so an image-carrying turn stays tied to the tab exactly as before.

This also laid the groundwork W28 and W29 build on directly: the job id is what a queue-position
poll and a background-send need to ask about, and the "detaching, not stopping" distinction (below)
is what lets W29 leave a job running when the tab moves on.

**A mutation-testing defect found here, not in the code but in its own gate.** One mutation —
skipping the branch that distinguishes "this reply was let go on purpose" from "this reply was
stopped" — did not turn any check red. Tracing it down: `detachReply` (`:1234`) saves the entry as
still-`"generating"` and aborts the fetch, but by the time that abort's `catch` actually runs
(asynchronously, after the fetch promise rejects), `openChat`/`startFresh` has already run
synchronously and cleared `$("chat-status")` for the *new* conversation. The old entry and its
bubble are genuinely orphaned by then — but `$("chat-status")` is a single shared element, not
per-conversation, so if the mutated code falls through to the generic "stopped" branch, it writes
"stopped" into the *new*, idle conversation's status line, after the fact, with nothing to catch it.
A stray element write invisible to every existing check is worth finding exactly the way it was
found here — not because the branch does anything visible in the common case, but because the one
thing it prevents (clobbering someone else's status text) has no other check watching it.

Gate: phases 35–36 of `scripts/webui_app_gate.mjs`, driven by real captured `POST`/`GET`/`DELETE`
`/v1/jobs` traffic (`scripts/webui-gate/captured-jobs.json`, Qwen2.5-0.5B, CPU) — a submit, a
waiting job's queue field, a full-queue refusal, a finished job's record and full replayed event
stream, and a job cancelled while queued. Eighteen mutations, each red, including the one above
(closed by a check that leaving a job-backed reply does not leave the next conversation's status
saying it was stopped). **Effort was: M.**

### W28 — Queue position while waiting — DONE 2026-09-15
~~A busy server is currently an indistinguishable spinner~~ — most of this shipped as part of the
server's own job-status work this morning (`de0f13c5`) and W27's client, ahead of this doc catching
up: `GET /v1/jobs/{id}` reports a still-queued job's `queue: {position, waiting, running}`
(`internal/serveapp/jobs_http.go:114`, from `lm.turns.position`), and the page polls it once a
second while nothing has streamed yet, showing *"Waiting for its turn: 2nd in line (3 waiting, 1
running)."* in place of a spinner (`pollQueue`, `internal/serveapp/webui/ui/app.js:1101`).
**Gate, as anticipated:** this is a field on the page's own request's own state (its W27 job), never
a `/web/queue` listing every client's requests — the same rule W5 already settled, so nothing new
needed deciding.

**What was still missing, and is the actual work of this entry: the 429 itself carried no number.**
W13's *"its request queue is full"* was true but not informative — it didn't say how full, or
compared to what. `loadedModel.queueFullMsg()` (`internal/serveapp/openai.go:254`) now names the
configured depth: `cap(lm.queue)-1`, the one number that is always accurate of a queue reported as
FULL (a live waiting count would already be stale by the time a client could read it) — *"model "q"
queue full (max 2 queued); retry"*. All three refusal sites share it: the synchronous chat-route and
job-submit 429s, and `notAdmitted`'s rarer job-already-accepted race. The page folds the number into
W13's title instead of parroting the server's own wording: *"The model is busy: its request queue is
full (max 2)."*

Measured real, not assumed: `TestJobs_realModelEventsQueueAndRefusal` drives `-max-queue 1` against
a real loaded model and asserts the exact captured text, including the number.

Gate: the widened `TestNotAdmitted_queueFullIsAFailureNotACancellation` (both the no-queue and
bounded-queue message shapes) plus the real-model queue test above; phase 22 and phase 35's captured
429 checks now assert the numbered title, from bodies re-captured against real `serve` output after
this change (`scripts/webui-gate/captured-errors.json`, `captured-jobs.json`) rather than the
pre-W28 wording. A mutation clearing the client's depth-parse regex to always miss was confirmed red
on both the W13 and W27 checks that read it. **Effort was: S** (the position half was already done;
this was the 429's number).

### W29 — Send in the background and come back — DONE 2026-09-15 (landed with W27)
~~Submit through `/v1/jobs` rather than the streaming route, leave the page, and the answer is on
the job when you return~~ — this fell out of W27's own machinery rather than needing its own build:
once a text reply *is* a job (W27), opening another conversation or hitting New chat mid-reply no
longer has to block or stop it. `detachReply` (`internal/serveapp/webui/ui/app.js:1234`) saves the
in-flight reply as still-`"generating"` with its job id and lets the local stream go; `openChat`
and New chat call it instead of refusing when a job-backed reply is running
(`internal/serveapp/webui/ui/app.js:1306`, `:1333`) — only an image-carrying (non-job) reply still
blocks them, since there is nothing server-side to detach from. The conversation list marks that
chat *"reply in progress"* while it runs (`internal/serveapp/webui/ui/app.js:1412`), and the list
and New chat both stay usable, not disabled, for the whole time it is away.

`resumeJob` (`internal/serveapp/webui/ui/app.js:1244`) is what makes coming back work: still
running → re-attach and replay from the start (W27); finished while away → filled in from the job's
own record, labelled *"finished while you were away"*; failed or cancelled while away → shown as
such, with the server's own reason; the job itself gone (restarted without `-job-dir`) →
*"interrupted — the server lost this job when it restarted"*, same as any other lost job. This is
the one Claude-app behaviour the page could not match at any price before J3 — closing the tab, or
even just looking at another conversation, used to be the same as Stop.

Gate: the same phases 35–36 as W27 (`scripts/webui_app_gate.mjs`), sharing its 18 red mutations —
"leaving the conversation does not cancel the job," "the list marks that conversation as having a
reply in progress," "a reply that finished while you were away is filled in from the job's real
record," "a job that failed/cancelled while away is shown as such." **Effort was: none beyond
W27** — recorded here because the doc had it as a separate M-effort item; in practice detaching
*is* what a job-backed reply already permits, not a second thing to build.

### W30 — A Batch tab — DONE 2026-09-15
~~Upload a JSONL, watch it run, download the results~~ — a third tab
(`internal/serveapp/webui/index.html`, `pane-batch`) over J4's OpenAI-shaped batch API: choose a
`.jsonl` file, **Run batch** uploads it (`POST /v1/files`, multipart — the one request on this page
that is not JSON, so it gets its own `authHeader()` rather than `headers()`, since fetch will not
set its own multipart boundary if `Content-Type` is already forced), then submits it
(`POST /v1/batches`, always `/v1/chat/completions` — the only endpoint the server accepts this
pass). The Models tab's pull idiom transplants directly: a terminal line and a progress bar
(`runBatch`, `internal/serveapp/webui/ui/app.js:1830`), just driven by polling
`GET /v1/batches/{id}` once a second instead of pull's SSE stream, since a batch has no single
connection to stream progress down — lines finish out of order, on their own schedule.

`pollBatch` (`internal/serveapp/webui/ui/app.js:1868`) renders `request_counts` as it arrives —
*"N of T done (C ok, F failed)"*, the bar at `(completed+failed)/total` — and on the terminal status
(`completed` or `cancelled`) stops polling, re-enables the form, and offers downloads: **Download
results** always (the server always writes an output file, even an empty one), **Download errors**
only when `error_file_id` is present (an all-succeeded batch has none). Both fetch
`GET /v1/files/{id}/content` and hand the real bytes to the existing `download()` helper (W14) — a
real file save, not a re-serialization of anything the page parsed, so the download is provably the
server's own output. **Cancel** posts `POST /v1/batches/{id}/cancel`.

Decided here, and why:
- **No per-request model field.** Each JSONL line already carries its own `body.model`; a batch is a
  file of already-complete requests, not a form that composes them.
- **No persisted history.** The doc's "watch it run, download the results" does not ask for one, and
  a real history belongs to W31 (the job store's own journal), not a second, batch-specific list.
- **`completion_window` is sent as a fixed `"24h"`.** The server does not enforce it (no polling
  arrives that instructs an SLA); it is required by the OpenAI shape, so a constant satisfies the
  shape without inventing a setting nothing reads.

**No Claude-app analog: this is a goinfer-specific surface, not a gap being closed.**

Gate: phase 37 of `scripts/webui_app_gate.mjs`, 17 checks against a scripted `/v1/files`/`/v1/batches`
fetch stub (progress across real polls, the finished line's exact ok/failed count, both downloads'
exact byte-for-byte content and per-batch filenames, an error-free batch offering only one download,
Cancel calling the real route and halting further polling, an upload failure shown with the server's
own reason and the tab left usable). Nine mutations, each red on this phase's own checks.

**A gate defect the mutations found, distinct from a hang: an uncaught exception loses the whole
phase's results, not just the one check downstream of it.** Breaking the file-select → enable-Run
wiring left every later `await` in the phase timing out (bounded, not a hang) — except one check
read `window.__lastBatchBody.input_file_id` with no guard, and dereferencing a property on
`undefined` (the request that never happened) threw. That exception unwound the phase's whole async
IIFE before it returned its `results` array, so the report showed `FAILED (gate program threw)`
with **zero** checks recorded — including the earlier, correctly-failing "a chosen file enables Run"
check, which never got the chance to be counted. Fixed with `?.` on every check that reads a value
only a real request populates (`window.__lastBatchBody?.…`, the downloaded-file capture). Confirmed
by re-running the same mutation after the fix: 14 checks now fail cleanly, by name, instead of the
whole phase reporting nothing.
**Effort was: M.**

### W31 — Cancel by job id, and job history — DONE 2026-09-15 (the allowed half)
~~J3's `DELETE /v1/jobs/{id}` cancels an addressed job... today's Stop button only aborts the local
stream~~ — most of this half was **already true by the time this entry was reached**: W27 made Stop
call `DELETE /v1/jobs/{id}` for whatever is on screen, and W27/W29's `resumeJob` already fills in a
reply's outcome (done, failed, cancelled, or lost) from the job's own record the moment its
conversation is opened — which **is** "show their history" for a job this page's own flows created;
there is no separate history view to build, the same way W29 turned out to already be built by W27.

**What was genuinely still missing: cancelling a job-backed reply that is running in the
*background* — a different conversation than the one on screen — without switching to it first.**
Before this, the only way to stop it was to reopen that conversation (which re-attaches, W27) and
then click Stop. Now the chat list's "reply in progress" badge (W29) carries its own **Cancel**
button (`renderChatList`, `internal/serveapp/webui/ui/app.js:1401`) when that conversation's own
storage holds a job id, wired through `cancelBackgroundJob`
(`internal/serveapp/webui/ui/app.js:1459`): `DELETE /v1/jobs/{id}` for that job, then — without
waiting for the conversation to be reopened — the same outcome `resumeJob` would derive is written
back into its storage directly (`state: "cancelled"`, `running` cleared), so the badge disappears
immediately and a later open shows it correctly without ever re-attaching to a job that is no longer
running.

**Still exactly W5's rule, applied to a new spot.** The button only ever acts on a job id read back
from the very storage key it is cancelling — never a value handed to it — so it can only cancel what
that conversation's own flows created, the same scope W27's Stop already had. No Cancel is offered
on the conversation actually being viewed (Stop already covers that one), nor when the stored job id
fails the same validation W27 uses (a hostile or malformed value never becomes a request).

**Gate, still split as filed:**
- **Cancel jobs this page submitted, and show their history: allowed. Built.**
- **Show or cancel generations this browser never saw: not decided**, unchanged — still W22's open
  question, answered together with it.

Gate: phase 38 of `scripts/webui_app_gate.mjs`, 10 checks: no Cancel on the conversation being
viewed, a just-detached-from conversation's Cancel starting *disabled* (its own detach has not
settled — a real, if narrow, race the mutations found needed its own check, below), Cancel offered
next to a background reply's badge, disabled while editing (same as rename and delete), the real
`DELETE` call naming the *other* conversation's job (not whatever is on screen), the badge and
button disappearing immediately rather than waiting for a reopen, the cancelled state correctly
written back to storage, a reopen afterward correctly *not* re-attaching (it is no longer
"interrupted"), a hostile stored job id offered the badge but never a Cancel button, and — a second
hostile-id case, distinct from the first — a job id that is valid when the button renders but turns
hostile before the click resolves (a storage race), still refused because the handler re-reads
storage fresh rather than trusting what the render implied.

Seven mutations, each red on this phase's own checks. Two were not, on the first pass, and both
were real gaps, not weak assertions to paper over:
- **The busy-disabled check only ever observed `syncActions()`'s correction, never the button's own
  creation-time value.** `syncActions()` re-disables every list button whenever `editing`/`ac`
  changes, which happens to mask a broken `cancelBtn.disabled = busy` at the moment it is *created* —
  except in the one window where `renderChatList()` runs with no `syncActions()` immediately behind
  it: the instant a conversation is detached from (W29), when `ac` is still the old, still-aborting
  controller (the abort's rejection is a microtask, not synchronous) and its row's Cancel button is
  being created for the very first time. Closed by checking that exact instant, synchronously, before
  any `await`.
- **The hostile-job-id guard inside `cancelBackgroundJob` was unreachable through the UI**, because
  `readChat` already refuses to let a bad id become a clickable button — so the function's own
  defense-in-depth check had no test that ever exercised it. Closed by simulating the race it
  actually guards against: a valid id at render time, corrupted before the click resolves.

One of the original seven (a stale `nextSubmit` left over from setting up the background job,
leaking into a later, unrelated send within the same phase) was a defect in the gate script itself,
not the page — caught before it could produce a false pass.
**Effort was: S** (small once W27/W29 existed to build it on).

### W32 — Unload a model, and see what is resident — DONE 2026-09-15

**W5 shipped half a door.** ~~nothing on the page can free the first model to make room~~ — a
`POST /web/models/unload` route now exists over `unloadByName`
(`internal/serveapp/admin.go:243`), the two-phase unpublish-then-drain `handleAdminUnload` already
did (unpublish under `regMu`, drain-and-close detached) — pulled out into its own function
specifically so the web route and the admin route share it verbatim, rather than the page getting a
second, parallel implementation of the same use-after-free-avoiding logic.

**No path policy needed here, unlike load.** `webLoadPath` (W5) exists because a load names a
filesystem path the admin route would otherwise trust unconditionally; unload names nothing but a
registry key, and the only keys that exist are ones `GET /v1/models` already publishes to every
client. `handleWebUnload` (`internal/serveapp/webui.go:519`) is `sameOrigin(auth(...))` behind
`-web` — W5's exact gate stack (`internal/serveapp/main.go:743`) — with `s.models[req.Name]` under
`regMu` (inside `unloadByName`) as the entire "policy": a name not loaded is a 404, the same shape
as any other unknown model. `TestWebUI_disabledByDefault` and the AST wiring guard
(`TestWebUI_listAndPullAreWrappedInSameOrigin`, `internal/serveapp/webui_test.go`) were both
extended to this route rather than left to trust it by resemblance.

**The Models tab now lists what is resident** (`renderResidentModels`,
`internal/serveapp/webui/ui/app.js:213`) — filtered to entries with a decoder (an embedding-only
entry has nothing to unload) — showing quant, decode path and resident size per row, and marking
whichever one Chat is pointed at. `quant` and `resident_bytes` are new fields on `pathFields`
(`internal/serveapp/openai.go:431`), the one function `/v1/models` and `/health` already share, so
both surfaces gained them for free rather than the page needing a second, web-only request just to
ask the registry twice. Each row's **Unload** button names its own free-able size.

**"Unloaded" is not "memory freed", stated on screen.** `unloadModel`
(`internal/serveapp/webui/ui/app.js:237`) reads the response's `status`/`freed` and says exactly one
of three things: *"unloading — finishing an in-flight request; its memory frees as that completes"*
(202: a spinner that resolved here would be lying), *"unloaded — memory freed"* (200, last owner),
or *"unloaded — its memory is shared with another loaded entry, so it was not freed"* (200, not the
last owner). All three leave the model gone from `/v1/models` either way — draining only withholds
the *native memory* claim, never the routing one — so the list refreshes and the row disappears
regardless of which of the three it was.

**Confirm before unloading, worded to what this page can actually know.** Another client's in-flight
turn on the same model is invisible here, so the default wording states the drain guarantee rather
than claiming to observe it: *"A request already in flight elsewhere will finish; nothing new will
route to it until it is loaded again."* When *this* page's own reply is the one running on that
model (`generating.model === name`), the wording says so specifically instead.

**The fit refusal now names a way out.** `unloadSuggestion` (`internal/serveapp/webui.go:482`)
checks `errors.Is(err, decoder.ErrWontFitResident)` — the exported sentinel `FitDeclineError`
unwraps to, chosen over `errors.As` on the concrete type specifically so the check (and its test)
never need that type's unexported fields — and, if something is resident, appends *"Unload
`"qwen2.5-1.5b"` (4.0 GB) to make room."* to the load error. Prose, not a line break: the page renders
this as plain `textContent` with no `white-space:pre-line`, so a literal `\n` would just collapse to
a space. Nothing to suggest (a fresh server's first load) leaves the message exactly as the decoder
raised it, so the page never implies a fix that doesn't exist.

**Decided here, and why:**
- **`?wait=false` is not plumbed through the web route.** The doc floated "`?wait=false` plus a poll
  of `/health`" as the better UI shape; in practice the client has to handle both 200 and 202
  regardless (bullet 3), and the common case — an idle model — drains within the default wait and
  returns 200 without an extra round trip, so there was nothing to gain by forcing every unload
  through the slower path.
- **No CUDA-specific re-gate.** The doc's own gate list asks for one because the drain is precisely
  the use-after-free fix that CUDA can SIGSEGV on if it's wrong — but `unloadByName` is the *exact*
  function `handleAdminUnload` already called, unmodified, and that safety property is already
  proven backend-agnostically by `unloaddrain_testhooks_test.go`'s `preamblePark` hook. This route
  adds no new path through the drain, so re-running that proof here would be re-verifying code this
  change never touched.
- **Load still does not auto-evict.** Untouched, as specified — the user chooses which model goes.

Gate: `TestWebUnload_publishesRemoval`, `TestWebUnload_unknownNameIs404`,
`TestWebUnload_freedTrueForARealModel` (the committed tiny fixture, a real `*decoder.Model`, proving
`freed:true` end to end rather than only through the already-proven shared function) and
`TestUnloadSuggestion` (`internal/serveapp/webui_unload_test.go`), plus the two existing route-guard
tests widened to the new route. Phase 39 of `scripts/webui_app_gate.mjs`, 12 checks: the list
excludes the embedding-only entry, a row's exact text (quant, decode path, size, the "in Chat"
mark), the Unload button's own size label, the confirm's two wordings (generic and "your current
reply"), the real POST naming exactly that row, all three response shapes' exact wording, a failed
unload leaving the row usable without touching the list, and no card at all when nothing has a
decoder. Nine mutations, each red on this phase's own checks.

### What J5–J9 would add later
J5 (the resumable `goinfer-chat -batch` runner) is a CLI, not a UI item. J6 (prefix-aware
scheduling) was measured 2026-09-15 and **killed** — 1.024× against a 1.3–2.0× pass band
(`docs/measurements/j6-prefix-scheduling-2026-09-15.md`) — so admission stays plain FIFO and W28's
"position while waiting" is exactly arrival order, unqualified. J8 (N decode workers) is still open
and would change *how* the queue is served without changing this.

---

## Sources

`internal/serveapp/webui.go:51`, `:545` (the embed, the `-web` gate) ·
`internal/serveapp/webui/ui/app.css:1513` (layout) · `internal/serveapp/webui/index.html:13` (tabs) ·
`internal/serveapp/webui/ui/app.js:309`, `:935`, `:122`, `:1434`, `:1581`, `:357`, `:841`, `:86`, `:489`, `:1358`, `:641`, `:205`, `:1288`, `:7`, `:946` (the conversation transcript, the
rendering rule, the error explanations, the keyboard handling, the load offer that replaced the dead-end line, the thinking split,
regenerate/edit/delete, the context meter, conversation storage, generated titles, sampling controls, images, export, theme, model labels) · `internal/serveapp/openai.go:813` (`contextWindow`) ·
`internal/serveapp/admin.go:113` (`handleAdminLoad`) ·
`internal/serveapp/openai.go:466` (the sampling fields the page never sends) ·
`internal/serveapp/anthropic.go:35` (no thinking block in v1) · `pull/pull.go:179` (`Size`, for the
fit verdict) · `demo/agent/cmd/agent-web/index.html:129`, `:183`, `:198` (the image composer, the
markdown TODO, the tool chips — all transplantable) ·
[`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) §F.3, §0 ·
[`task-fit-to-hardware.md`](task-fit-to-hardware.md) §3 ·
[`task-work-queue-2026-09.md`](task-work-queue-2026-09.md) J3, J8, J9 ·
[`task-halt-2026-09.md`](task-halt-2026-09.md) K1/K2/K5 ·
`docs/completed/task-web-ui-ambient.md` (the current visual design) ·
`docs/completed/review-2026-09-04.md` V-20 (the cross-origin lesson)

<!-- doc-reviewed: 2026-09-13 -->

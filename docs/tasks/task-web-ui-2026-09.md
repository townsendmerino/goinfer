# Task: `serve -web` as a real chat interface — the Claude-app gap (W1–W26) — 2026-09

> **Status: SCOPED 2026-09-13, SCOPE DECIDED 2026-09-14, IN PROGRESS — §6.1, Tier A (W1–W8), W9–W11 and W13 done; W12 skipped for now (owner, 2026-09-14); Tier B continues with W14.** Filed from
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
> J3 is the only route to W18; [`task-halt-2026-09.md`](task-halt-2026-09.md) K1/K2/K5 is what a
> W22 admin panel would surface. `docs/completed/task-web-ui-ambient.md` is the closed record of
> the current visual design and is not superseded by anything here.

---

## 0. What exists today

The page is **a small embedded directory** — `//go:embed webui` (`internal/serveapp/webui.go:50`):
`index.html` plus `ui/app.css` and `ui/app.js`, hand-written HTML, CSS and vanilla JS, no build
step, no external stylesheet, font or script. *(Until 2026-09-14 it was one 1,828-line file; §6.1
records the split.)* That is deliberate and load-bearing: a CDN reference would make the UI of an
offline-capable engine require the network. It is off by default behind `-web`
(`internal/serveapp/webui.go:458`), and it is a client of the same `/v1` routes any other client
uses — so it cannot drift from the API, because it *is* the API's user.

Two tabs (`internal/serveapp/webui/index.html:12`):

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
  belong to other docs stay owned there (W12 → `task-fit-to-hardware.md` §3, W18's re-attach →
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
string (the rule is stated at `internal/serveapp/webui/ui/app.js:903`). Paragraphs and line breaks,
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
(`internal/serveapp/webui/ui/app.js:284`) saved to `localStorage` under a versioned key, and the
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
card (`internal/serveapp/webui/ui/app.js:597`), whose summary reads "· active" when set so it is
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
(`internal/serveapp/webui/ui/app.js:1400`). It loads the file, shows a heartbeat while the load runs,
refreshes the model list, and selects the new model, so the next message goes to it. The header
stats now follow whichever model is selected, not always the first one listed.

**The gate decision: a narrow route, not the admin load.** `POST /admin/models/load`
(`internal/serveapp/admin.go:113`) takes any caller-named path and stays behind
`-allow-admin`/`-admin-socket`, unchanged. The page gets its own `POST /web/models/load`
(`internal/serveapp/main.go:724`), registered only under `-web` and wrapped like pull
(`sameOrigin`, `auth`, body cap). It will load only a **regular `.gguf` file inside the pull cache**
(`webLoadPath`, `internal/serveapp/webui.go:321`). Symlinks are resolved on both the path and the
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
(`splitThinking`, `internal/serveapp/webui/ui/app.js:332`). The server still does not split
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
(`internal/serveapp/webui/ui/app.js:809`). History is now addressable: each action changes the
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
(`showContext`, `internal/serveapp/webui/ui/app.js:61`). At 80% it warns that the conversation is
nearing the limit. At 95% it says the next message will likely not fit, and what to do: start a new
chat, or delete earlier exchanges (W7). If a request does hit the wall, the error says the same thing
in plain words, with the server's own message underneath. Other 400s are left as they are.

**Server: `/v1/models` (and `/health`) publish `context_window`.** It comes from one function,
`contextWindow` (`internal/serveapp/openai.go:804`), which `prepare` also uses to enforce the limit,
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
| **W14** | Export the conversation | nothing. A share link is an anti-goal; Markdown and JSON to a file are not | S |
| **W15** | Dark mode | one light surface; no `prefers-color-scheme` rule anywhere. The AmbientCSS palette is already token-shaped (`docs/completed/task-web-ui-ambient.md`) | S |
| **W16** | Phone layout | one `max-width:920px` column, no media query (`internal/serveapp/webui/ui/app.css:1480`). A server on the LAN is a plausible phone client | S |
| **W17** | Enter sends, ↑ edits last, Esc stops | only Ctrl/Cmd+Enter (`internal/serveapp/webui/ui/app.js:1288`). Make it a setting, not a swap — the current behaviour suits long prompts | S |
| **W18** | Label which turn came from which model | the dropdown is read at send time so switching half-works, but nothing marks the turns, and the per-response stats are the one place that comparison would mean something | M |

### W13 — Errors that say what to do — DONE 2026-09-14
~~Any non-200 becomes the first 400 characters of the response body in a red bubble~~ — every failure
is explained as what happened and what to do, with a button for the remedy (`problemFor`,
`internal/serveapp/webui/ui/app.js:97`). The server's own message stays underneath, as text.

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
(`internal/serveapp/webui/ui/app.js:180`). The image shows on the message it was sent with, is saved
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

### W10 — Full sampling controls — DONE 2026-09-14
~~The page sends `temperature`/`max_tokens` only~~ — a collapsible **Sampling** section beside the
system prompt adds every other sampling field `/v1/chat/completions` accepts: `top_p`, `top_k`, `seed`,
`stop`, `frequency_penalty` and `presence_penalty` (`internal/serveapp/webui/ui/app.js:615`).
Temperature and max tokens stay in the top row.

**A correction to what this row used to say:** it listed `logit_bias` among the fields the route
accepts. It does not. The request struct (`internal/serveapp/openai.go:439`) has no such field. The
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

**Storage** (`internal/serveapp/webui/ui/app.js:464`): one localStorage key per conversation,
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

**Titles** (`autoTitle`, `internal/serveapp/webui/ui/app.js:1217`). A title starts as the first message,
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
  only, so a person watching a slow generation in the browser can neither see nor stop it. Same
  gate question as W5 — answer it once, for both. M.
- **W23 — Structured-output workbench.** Paste a JSON schema, watch a grammar-constrained
  generation fill it, see the token cost. `response_format` is live. The README's "a Go struct the
  model cannot violate" currently has nowhere a visitor can see it work. M.
- **W24 — Projects / knowledge / RAG.** ken + aikit + `/v1/embeddings`. A stack decision, not a UI
  one. XL.
- **W25 — Rendered preview of generated code.** The sandboxing *is* the job: a sandboxed iframe at
  a null origin, or not at all. L.
- **W26 — Prompt library.** M, and the weakest pull on the list for this audience.

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
`Markdown.render`, never `innerHTML` (`internal/serveapp/webui/ui/app.js:903`), and a test now fails the
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

## Sources

`internal/serveapp/webui.go:50`, `:458` (the embed, the `-web` gate) ·
`internal/serveapp/webui/ui/app.css:1480` (layout) · `internal/serveapp/webui/index.html:12` (tabs) ·
`internal/serveapp/webui/ui/app.js:284`, `:903`, `:97`, `:1288`, `:1400`, `:332`, `:809`, `:61`, `:464`, `:1217`, `:615`, `:180` (the conversation transcript, the
rendering rule, the error explanations, the keybinding, the load offer that replaced the dead-end line, the thinking split,
regenerate/edit/delete, the context meter, conversation storage, generated titles, sampling controls, images) · `internal/serveapp/openai.go:796` (`contextWindow`) ·
`internal/serveapp/admin.go:113` (`handleAdminLoad`) ·
`internal/serveapp/openai.go:449` (the sampling fields the page never sends) ·
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

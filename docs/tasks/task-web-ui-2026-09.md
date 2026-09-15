# Task: `serve -web` as a real chat interface — the Claude-app gap (W1–W26) — 2026-09

> **Status: SCOPED 2026-09-13, SCOPE DECIDED 2026-09-14, unstarted.** Filed from a feature comparison
> against the Claude desktop/web app, read against the tree at `9d29d625`. Nothing here is built.
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

The page is **a small embedded directory** — `//go:embed webui` (`internal/serveapp/webui.go:47`):
`index.html` plus `ui/app.css` and `ui/app.js`, hand-written HTML, CSS and vanilla JS, no build
step, no external stylesheet, font or script. *(Until 2026-09-14 it was one 1,828-line file; §6.1
records the split.)* That is deliberate and load-bearing: a CDN reference would make the UI of an
offline-capable engine require the network. It is off by default behind `-web`
(`internal/serveapp/webui.go:303`), and it is a client of the same `/v1` routes any other client
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
string (the rule is stated at `internal/serveapp/webui/ui/app.js:248`). Paragraphs and line breaks,
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
(`internal/serveapp/webui/ui/app.js:66`) saved to `localStorage` under a versioned key, and the
messages sent to the API are derived from it, so what is shown, saved and sent cannot drift apart.
On load it is restored and rendered exactly as live turns are — model output through the Markdown
renderer, the user's text as text — so stored content earns no more trust than fresh output.

- **A reload mid-stream keeps the partial answer**: it is saved about once a second and on
  `pagehide`, and comes back labelled "interrupted". Stopped answers come back labelled "stopped".
- **Storage failures are survivable**: blocked storage or a full quota leaves the chat working and
  shows a note saying the conversation won't survive a reload; a stored value the page can't read is
  set aside under `goinfer.chat.v1.unreadable`, not destroyed.
- **New chat** clears it (after a confirm — there is no undo), and is disabled mid-reply.
- **One conversation per browser**, shared by its tabs: an idle tab follows another tab's change.
  Separate conversations are W9.

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
card (`internal/serveapp/webui/ui/app.js:148`), whose summary reads "· active" when set so it is
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

### W5 — Load a model from the page
The pull flow dead-ends on its own success line: *"Downloaded, not loaded — restart the server with
--model &lt;path&gt; to serve it"* (`internal/serveapp/webui/ui/app.js:471`). A first-run user is
sent back to a terminal in the middle of the one flow the page exists for.

`POST /admin/models/load` already exists (`internal/serveapp/admin.go:113`) but is gated behind
`-allow-admin`/`-admin-socket` **for a real reason** — it loads caller-named paths. So this is a
policy decision, not wiring: either a narrow `-web`-only load route restricted to files under the
pull cache directory (recommended — the page can only load what it just downloaded), or an explicit
opt-in flag. Do not simply widen the admin gate. **Effort: M, plus the gate decision.**

### W6 — Fold the thinking block
Reasoning tokens stream inline as body text, so any reasoning-class model looks like it is
rambling before it answers. The server does not split them either — `stop_reason` is never
`thinking` in v1 (`internal/serveapp/anthropic.go:35`). Page-side tag folding is the cheap half and
worth doing alone; a real content-block split is a server change and its own item.
**Effort: S–M page-side.**

### W7 — Regenerate, edit-and-resend, delete a turn
None of the three. A bad turn is permanent, and the only recovery is a reload, which costs the
whole conversation (W3). Needs history to become addressable rather than append-only.
**Effort: M.**

### W8 — Context meter, and a warning before the wall
You discover the context limit by hitting `400 context_length_exceeded` mid-conversation. The
server knows the real number since R13 fixed `Config.MaxPositions` for the 16 GGUF architectures
that had it unset; `/v1/models` would need to publish it alongside `decode_path`.
**Effort: M, plus one field on `/v1/models`.**

---

## 3. Tier B — the ten that make it a daily driver (in scope)

Six are S. Each is a reason someone opens Open-WebUI or LM Studio instead of the page that shipped
inside the binary.

| # | Item | Today | Effort |
|---|---|---|---|
| **W9** | Conversation list with generated titles | one unnamed conversation, until reload | M (after W3) |
| **W10** | Full sampling controls | page sends `temperature`/`max_tokens` only; the route already accepts `top_p`, `top_k`, `seed`, `stop`, penalties and `logit_bias` (`internal/serveapp/openai.go:391`) | S |
| **W11** | Image attach for vision models | no control, though `-vision` works on the same route; `demo/agent/cmd/agent-web/index.html:129` has the whole composer (click, drag, paste, preview) to transplant, plus a per-model capability check so it hides on text-only models | M |
| **W12** | Fit verdict before a multi-GB pull | size only. **Already scoped** — `task-fit-to-hardware.md` §3; `pull.File` carries `Size` (`pull/pull.go:179`) | M |
| **W13** | Errors that say what to do | any non-200 becomes `(await r.text()).slice(0, 400)` in a red bubble (`internal/serveapp/webui/ui/app.js:294`), so a queue-full 429, a halted 503 and a bad key read alike — while the server's error shapes are typed | S |
| **W14** | Export the conversation | nothing. A share link is an anti-goal; Markdown and JSON to a file are not | S |
| **W15** | Dark mode | one light surface; no `prefers-color-scheme` rule anywhere. The AmbientCSS palette is already token-shaped (`docs/completed/task-web-ui-ambient.md`) | S |
| **W16** | Phone layout | one `max-width:920px` column, no media query (`internal/serveapp/webui/ui/app.css:1480`). A server on the LAN is a plausible phone client | S |
| **W17** | Enter sends, ↑ edits last, Esc stops | only Ctrl/Cmd+Enter (`internal/serveapp/webui/ui/app.js:373`). Make it a setting, not a swap — the current behaviour suits long prompts | S |
| **W18** | Label which turn came from which model | the dropdown is read at send time so switching half-works, but nothing marks the turns, and the per-response stats are the one place that comparison would mean something | M |

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
`Markdown.render`, never `innerHTML` (`internal/serveapp/webui/ui/app.js:248`), and a test now fails the
build if that changes. **Build DOM nodes from the parsed tree; never assemble an HTML string.** Otherwise a model — possibly one pulled from a stranger's Hugging Face repo
minutes earlier, by this very page — gets script execution on the same origin as the API, with the
user's key in a field on that page. The gate for W1 is a test that feeds the renderer hostile
markdown (`<img onerror>`, `javascript:` links, raw `<script>`) and asserts no element is created
outside the allowed set.

**6.3 Every new mutating route inherits the gates already there** — `sameOrigin`, the `-web`
opt-in, and the non-loopback `-api-key` rule (`internal/serveapp/webui.go`). V-20 in
`docs/completed/review-2026-09-04.md` is the record of what it cost to learn that the first time.
W5's load route is the first real test of them.

---

## Sources

`internal/serveapp/webui.go:47`, `:303` (the embed, the `-web` gate) ·
`internal/serveapp/webui/ui/app.css:1480` (layout) · `internal/serveapp/webui/index.html:12` (tabs) ·
`internal/serveapp/webui/ui/app.js:66`, `:248`, `:294`, `:373`, `:471` (the conversation transcript, the
rendering rule, the error path, the keybinding, the dead-end line) · `internal/serveapp/admin.go:113` (`handleAdminLoad`) ·
`internal/serveapp/openai.go:391` (the sampling fields the page never sends) ·
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

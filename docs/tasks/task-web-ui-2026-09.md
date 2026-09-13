# Task: `serve -web` as a real chat interface — the Claude-app gap (W1–W26) — 2026-09

> **Status: SCOPED 2026-09-13, unstarted.** Filed from a feature comparison against the Claude
> desktop/web app, read against the tree at `9d29d625`. Nothing here is built.
>
> **This doc reopens a question that was already answered once.**
> [`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) §F.3 asked exactly "where does
> `-web` stop" and recorded the lean: *chat and pull only — it is the first-run surface, not a
> product.* Tier A below survives that reading intact, since all eight items are about the first
> run working at all. **Tiers B and C do not.** §1 puts that decision up front rather than letting
> it erode one feature at a time.
>
> Siblings: [`task-fit-to-hardware.md`](task-fit-to-hardware.md) §3 already owns W12 (fit before
> download) and is cited rather than restated; [`task-work-queue-2026-09.md`](task-work-queue-2026-09.md)
> J3 is the only route to W18; [`task-halt-2026-09.md`](task-halt-2026-09.md) K1/K2/K5 is what a
> W22 admin panel would surface. `docs/completed/task-web-ui-ambient.md` is the closed record of
> the current visual design and is not superseded by anything here.

---

## 0. What exists today

The page is **one embedded HTML file** — `//go:embed webui/index.html`
(`internal/serveapp/webui.go:35`), 1,828 lines of hand-written HTML, CSS and vanilla JS, no build
step, no external stylesheet, font or script. That is deliberate and load-bearing: a CDN reference
would make the UI of an offline-capable engine require the network. It is off by default behind
`-web` (`internal/serveapp/webui.go:243`), and it is a client of the same `/v1` routes any other
client uses — so it cannot drift from the API, because it *is* the API's user.

Two tabs (`internal/serveapp/webui/index.html:1530`):

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

## 1. The decision this doc needs first

§F.3's lean was correct for what the page was. The question is whether the page is now the product
surface for mode 1 ("try it"), in which case Tier A is not scope creep but the minimum for the
mode to work at all — a first-run user currently hits a dead end inside the one flow the page
exists for (W5).

**Recommended split, to be accepted or rejected as a whole:**

- **Tier A is in scope under the existing §F.3 reading.** Every item is "the first run works".
- **Tier B is a deliberate widening** of §F.3, and §F.3 should be amended in place to say so, with
  a date, rather than left to read as still-current.
- **Tier C is not authorised by this doc.** Each item there needs its own decision, and two of
  them (W17, W18) belong to other docs already.

Until that is settled, build Tier A and nothing else.

---

## 2. Tier A — the eight that make it usable

Ranked by what a person notices in the first five minutes.

### W1 — Markdown and code blocks
Today the page renders **plain text only**: `textContent`, never `innerHTML`
(`internal/serveapp/webui/index.html:1658`). A code answer arrives as one unbroken run of
characters. This is the single largest usability gap and the one every visitor meets.
`demo/agent/cmd/agent-web/index.html:183` already carries a `renderMarkdownLite` whose own comment
says "Deliberately tiny — full markdown is a TODO"; extend that rather than starting over.
**Effort: S–M.** **See §4 — this item is also the security decision.**

### W2 — Copy, per message and per code fence
No copy affordance at all; you select by hand and catch the stats line. Falls out of W1's node
tree. **Effort: S.**

### W3 — The conversation survives a reload
History is `const history = []` in page memory
(`internal/serveapp/webui/index.html:1646`). A refresh loses the conversation, including the answer
you were about to copy. `localStorage` is the cheap correct answer; `-session-dir` holds KV
snapshots, not transcripts, and is not this. **Effort: S.**

### W4 — A system prompt box
There is no way to set a system message at all. One textarea, prepended to `messages`; the route
has always accepted it. It is also the only way the page can show the thing harness users care
about. **Effort: S.**

### W5 — Load a model from the page
The pull flow dead-ends on its own success line: *"Downloaded, not loaded — restart the server with
--model &lt;path&gt; to serve it"* (`internal/serveapp/webui/index.html:1815`). A first-run user is
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

## 3. Tier B — the ten that make it a daily driver

Six are S. Each is a reason someone opens Open-WebUI or LM Studio instead of the page that shipped
inside the binary.

| # | Item | Today | Effort |
|---|---|---|---|
| **W9** | Conversation list with generated titles | one unnamed conversation, until reload | M (after W3) |
| **W10** | Full sampling controls | page sends `temperature`/`max_tokens` only; the route already accepts `top_p`, `top_k`, `seed`, `stop`, penalties and `logit_bias` (`internal/serveapp/openai.go:391`) | S |
| **W11** | Image attach for vision models | no control, though `-vision` works on the same route; `demo/agent/cmd/agent-web/index.html:129` has the whole composer (click, drag, paste, preview) to transplant, plus a per-model capability check so it hides on text-only models | M |
| **W12** | Fit verdict before a multi-GB pull | size only. **Already scoped** — `task-fit-to-hardware.md` §3; `pull.File` carries `Size` (`pull/pull.go:179`) | M |
| **W13** | Errors that say what to do | any non-200 becomes `(await r.text()).slice(0, 400)` in a red bubble (`internal/serveapp/webui/index.html:1689`), so a queue-full 429, a halted 503 and a bad key read alike — while the server's error shapes are typed | S |
| **W14** | Export the conversation | nothing. A share link is an anti-goal; Markdown and JSON to a file are not | S |
| **W15** | Dark mode | one light surface; no `prefers-color-scheme` rule anywhere. The AmbientCSS palette is already token-shaped (`docs/completed/task-web-ui-ambient.md`) | S |
| **W16** | Phone layout | one `max-width:920px` column, no media query (`internal/serveapp/webui/index.html:1485`). A server on the LAN is a plausible phone client | S |
| **W17** | Enter sends, ↑ edits last, Esc stops | only Ctrl/Cmd+Enter (`internal/serveapp/webui/index.html:1717`). Make it a setting, not a swap — the current behaviour suits long prompts | S |
| **W18** | Label which turn came from which model | the dropdown is read at send time so switching half-works, but nothing marks the turns, and the per-response stats are the one place that comparison would mean something | M |

---

## 4. Tier C — projects, not sessions. Not authorised by this doc.

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

**6.1 One embedded file stops scaling here.** Tier A alone roughly doubles 1,828 lines. The
no-build-step, no-external-asset property is worth keeping exactly as it is. The move that keeps it
and fixes the file is **`//go:embed webui/*` over a directory** — several source files, still no
toolchain, still one binary, still offline. What not to do is add a bundler; that trades away the
property the project exists for.

**6.2 W1 is where model output stops being inert, and the current rule must survive it.** The page
says it in the source: `textContent` only, never `innerHTML`
(`internal/serveapp/webui/index.html:1658`). **Build DOM nodes from the parsed tree; never assemble
an HTML string.** Otherwise a model — possibly one pulled from a stranger's Hugging Face repo
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

`internal/serveapp/webui.go:35`, `:243` (the embed, the `-web` gate) ·
`internal/serveapp/webui/index.html:1485`, `:1530`, `:1646`, `:1658`, `:1689`, `:1717`, `:1815`
(layout, tabs, in-memory history, the textContent rule, the error path, the keybinding, the
dead-end line) · `internal/serveapp/admin.go:113` (`handleAdminLoad`) ·
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

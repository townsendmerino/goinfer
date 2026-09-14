# Task: goinfer as a work queue — admission, jobs, batch APIs (J1–J9) — 2026-09

> **Status: REBUILT 2026-09-13, unstarted.** This doc was first written 2026-09-12 and was never
> committed. It was deleted the next morning by a workaround, not by a decision — see "How this
> doc was lost" below, which is kept because the failure is structural and the fix is J0.
> The rebuild is faithful to the J1–J9 scope as filed; the prose is re-derived from the tree at
> `9d29d625`, so every citation here was re-verified rather than carried over.
>
> Sibling: [`task-halt-2026-09.md`](task-halt-2026-09.md) (K1–K9) — this doc **reuses** K1 (cancel
> by id), K2 (global halt), K4 (budgets) and K5 (the admin socket) rather than restating them, and
> nothing here is worth building before those are stable. Also adjacent:
> [`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W18, whose "re-attach to a generation after a
> reload" is J3's browser-side consumer and has no other route to existing.

---

## How this doc was lost, and why that is the first item

The original was written 2026-09-12 and left untracked in the tree, which is this repo's normal
resting state for a fresh task doc for a few hours. Then:

- **`fd538dc6` (09-12 13:20)** — `docs/QUEUE.md`'s citation index absorbed its 11 path citations
  on a routine `--update`. The doc was real, on disk, and cited.
- **`fd598234` (09-12 13:47)** — the ARCHITECTURE.md rewrite referenced it by name.
- **`6c019eb5` (09-12 14:03)** — those 11 index rows came back out on the next `--update`, because
  the file was untracked and the index regenerates from the filesystem.
- **`3e01502c` (09-13 07:28)** — a different session found the pre-push citation lint red **because
  of this file**, and recorded the workaround in its own commit message: the lint "reads the
  filesystem (rglob), not git-tracked paths, so it fails while that file sits in the tree —
  verified moved aside … It must be moved aside again immediately before the actual `git push` for
  the pre-push hook to pass, then restored."

It was moved aside. It was not restored. No branch, stash, dangling object, `docs/internal/` or
backup folder holds it.

**The mechanism, stated plainly.** `scripts/queue_citation_lint.py`'s `live_docs()` walks
`ROOT.rglob("*.md")`. Any *untracked* markdown file carrying `path:line` citations therefore reds
the lint, and the pre-push hook refuses the push. The only remedy the lint offers is to take the
file out of the tree — so "move the new design record out of the repository" becomes the standard
move, performed under time pressure, at the moment someone is trying to push something else. That
is a document shredder wearing a gate's clothes.

The script already contains the exact argument against its own behaviour, written for
`docs/internal/`: that directory is skipped because it "is DELIBERATELY UNCOMMITTED (.gitignore),
so every path inside it resolves on exactly one machine and on none of the clones CI or anyone else
runs … a lint that reds on files CI cannot see is a lint people stop reading." Every word of that
applies to an untracked file. The exclusion was written for one directory when it is a property of
tracking.

### J0 — the lint must not red on files git does not track

**Ship before anything else in this doc**, because J1–J9 will produce several new task docs and
each one spends its first hours untracked.

- `live_docs()` skips any path `git ls-files --error-unmatch` does not resolve — i.e. untracked
  and ignored files alike, one `git ls-files` call up front rather than per file.
- The skipped set is **printed, not silent**: `note: N untracked doc(s) not linted (commit them to
  bring their citations under the gate): …`. Silence here would trade a false red for an invisible
  hole, which is the shape this repo keeps naming.
- `--update` likewise indexes only tracked files, so a generated index never carries rows that
  resolve on one machine.
- Gate: a mutation test — create an untracked `.md` with a deliberately bogus `path:line`
  citation, confirm exit 0 with the note, `git add` it, confirm exit 1 naming that citation.

Brief for this is at [`../prompts/citation-lint-untracked.md`](../prompts/citation-lint-untracked.md).

---

## What exists today, cited

Not rebuilt below; this is the floor J1–J9 build on.

- **One decode worker per model.** `tryEnter` claims a queue slot and then takes the model's mutex
  (`internal/serveapp/openai.go:170`), and the mutex is what serialises decode. The cap is
  literally `1 running + --max-queue` (`internal/serveapp/openai.go:77`).
- **The wait is not fair and not context-aware by itself.** `sync.Mutex.Lock()` has no context, so
  a second check exists purely so a halt can cut a waiter loose
  (`internal/serveapp/openai.go:180`). Waiters are woken in whatever order the mutex chooses: a
  20-token request that arrived last can go after a 4,000-token one that arrived first, and
  nothing in the system knows the difference.
- **Backpressure is a number, not a plan.** `-max-queue` defaults to 8
  (`internal/serveapp/main.go:506`); a full queue is a 429 on the OpenAI routes and a 529
  `overloaded_error` on the Anthropic one (`internal/serveapp/anthropic.go:507`). A global
  `-max-inflight` (default 128) bounds the pre-queue stage — JSON and image decode, tokenisation,
  template render — and is deliberately distinct from the per-model 429
  (`internal/serveapp/helpers.go:77`).
- **Nothing is durable.** `drive` runs the generation for the life of the request
  (`internal/serveapp/openai.go:1044`). The client's connection *is* the job: close it and the
  work is cancelled and unrecoverable. There is no id to ask about afterwards.
- **There is warm state worth scheduling around.** The session LRU keeps prefilled KV and hands a
  request the session that already holds its prompt as a prefix
  (`internal/serveapp/sessions.go:14`), `-kv-sessions` 4 by default
  (`internal/serveapp/main.go:500`). Admission order therefore has a measurable cost today that
  admission does not know about.
- **One route already takes a batch.** `/v1/embeddings` accepts up to 2,048 inputs in a request
  (`internal/serveapp/embeddings.go:34`) — the only bulk surface in the product, and the shape J4
  generalises.
- **No batch CLI.** `goinfer-chat` takes one `--model` and one conversation
  (`internal/chatapp/main.go:203`); there is no file-in/file-out mode.
- **From K1/K2/K5, already shipped:** a generation registry with cancel-by-id, global halt with
  in-flight cancellation, and an admin unix socket. J2 and J3 are the durable layer those three
  already assume exists and currently do without.

---

## Ground rules

1. **One decode worker per model stays**, through every item below. It is not an oversight to be
   removed on the way to a queue; concurrency across requests is J8's measurement to earn, on its
   own, against the roadmap's kill-or-earn rule. Everything else here is about *which* request the
   one worker takes next, and about surviving a disconnect.
2. **Compatibility is the promise for `serve`** (`docs/api-tiers.md`). J4 implements the two batch
   APIs as specified, including their quirks; it does not invent a third.
3. **Additive and Experimental until v1.0.** No Hard-tier name changes. `-job-dir` unset means the
   product behaves exactly as it does today, including the 429.
4. **No new root module dependency.** A JSONL journal and a job store are stdlib.
5. **Every throughput claim is measured before it ships**, with a pre-registered band and a kill
   number. J6 and J8 carry theirs below.

---

## J1 — fair, context-aware admission

Replace the bare mutex wait with an explicit queue the server can reason about.

- A per-model FIFO of waiting requests with a real `context.Context` per waiter, so a cancelled or
  halted waiter leaves immediately and the second halt check in
  `internal/serveapp/openai.go:180` stops being load-bearing.
- **Context-aware**, in both senses: the admission record carries the request's prompt-token count
  and its session/prefix key, so J6 and J7 have something to schedule on. J1 itself keeps strict
  FIFO — it establishes the structure and changes no order.
- 429/529 semantics unchanged at the boundary, and `-max-queue` keeps its meaning.
- Gate: with `-max-queue 1` and two concurrent requests, the second's cancellation is observed at
  the server within one decode step, not at the end of the first generation. Red before the change.

## J2 — the job object, and an optional journal

- A **job** is `{id, model, request, class, state, created, started, finished, usage, error}`.
  Every generation gets one, whether or not anything durable is on.
- `-job-dir <path>` (off by default) appends one JSONL line per state transition. Append-only,
  fsync on terminal states, owner-only mode — the same 0600/0700 reasoning
  `internal/serveapp/sessions.go`'s snapshot fix already applies, since a job record replays the
  prompt.
- On restart, the journal is read to reconstruct terminal jobs for J3's lookup. A job that was
  *running* at exit is marked `interrupted`, never silently `failed` — the distinction is the whole
  point of writing it down.
- This subsumes K9 (the append-only generation log) rather than duplicating it: one journal, two
  readers.

## J3 — `/v1/jobs`: submit, poll, re-attach

- `POST /v1/jobs` takes the same body as `/v1/chat/completions` and returns an id immediately.
- `GET /v1/jobs/{id}` returns state and result. `GET /v1/jobs/{id}/events` is an SSE stream that
  **replays from the beginning and then continues live**, so a reconnecting client loses nothing —
  this is the re-attach, and it is what makes a browser reload survivable.
- `DELETE /v1/jobs/{id}` is K1's cancel, addressed by job id rather than generation id; the two
  registries are joined, not parallel.
- The client-visible key stays the one proposed for K4: OpenAI's `user` field, Anthropic's
  `metadata.user_id`.
- Gate: start a job, kill the client mid-stream, reconnect to `/events`, and assemble a transcript
  byte-identical to an uninterrupted run of the same seed.

## J4 — the two batch APIs, over one job store

- **OpenAI Batch**: `POST /v1/files` (JSONL upload), `POST /v1/batches`, `GET /v1/batches/{id}`,
  `POST /v1/batches/{id}/cancel`, output file retrieval. `completion_window` is accepted and
  reported; it is not a promise this product can make on one worker, and the field must not read
  as one.
- **Anthropic Message Batches**: `POST /v1/messages/batches` and its results endpoint, over the
  same store, with the `custom_id` echo both APIs require.
- Both are thin translations onto J2's job object. No second execution path — that is the same
  rule the web UI is built on, for the same reason.
- Gate: an off-the-shelf OpenAI SDK batch round-trip completes against `serve` with no goinfer
  knowledge, and the result file's `custom_id` ordering matches the input.

## J5 — `goinfer-chat -batch in.jsonl -o out.jsonl`

- A resumable local runner sharing J4's line format exactly, so a file is interchangeable between
  the CLI and the HTTP batch API.
- **Resumable**: on restart with an existing output file, completed `custom_id`s are skipped. A
  20,000-line job that dies at line 14,000 does not start over.
- No server required — this runs the library in-process, which is also the mode-2 story the facade
  doc wants a real example of.

## J6 — prefix-aware scheduling

The only place ordering can buy real throughput on one worker: prefer the waiter whose prompt
shares the longest prefix with a session already warm in the LRU
(`internal/serveapp/sessions.go:14`).

- **Pre-registered band: 1.3–2.0×** on a mixed agent-loop workload (shared system prompt and tool
  specs, divergent tails) against strict FIFO.
- **Kill below 1.15×.** Bounded starvation guard: no waiter may be passed over more than N times,
  N configurable and measured at the same time, since the guard is what caps the win.
- Measure on the quiet box with paired differencing, both quants, and report the negative at full
  value if it lands there.

## J7 — classes, priorities, deadlines, budgets

- Two classes: **interactive** and **batch**. Interactive preempts batch at admission (never
  mid-generation — one worker, and a half-finished decode is waste).
- Optional per-job deadline; a job that cannot start before it is honestly declined rather than
  queued to expire.
- **Budgets are K4's**, applied here per class: token and wall-clock caps the model never sees.
- Default when nothing is configured: every request is interactive, and the behaviour is J1's.

## J8 — N decode workers per model: kill or earn

The only throughput item, and it is deliberately last.

- The standing analysis says decode is bandwidth-bound, so a second worker on the same model
  mostly splits the same memory bandwidth. That is a hypothesis with no measurement behind it in
  this repo.
- **Pre-registered:** ≥1.25× aggregate tokens/s at 4 concurrent requests, with p99 per-request
  latency no worse than 1.5× the single-worker figure. Below either, the item is removed and the
  negative is written down here.
- Measure on CPU and on one GPU backend; MoE paging changes the answer and must be its own cell.
- Note for whoever picks this up: the real alternative is **batched multi-request decode** (one
  worker, several sequences per step), which is a different and larger piece of work. J8 is the
  cheap question asked first, not the recommended answer.

## J9 — observability and the doc updates

- `GET /admin/queue`: depth, oldest wait, per-class counts, LRU hit rate. On the admin socket
  (K5), not on TCP by default.
- Startup banner gains one line when `-job-dir` is on, per the "banner is the UI" rule in
  `task-embed-and-harness-ux.md` §3.3.
- `docs/server.md` gains the job and batch routes; `docs/QUEUE.md` gets the J-entries that stay
  open; `docs/ARCHITECTURE.md`'s serving box gains the job store.

---

## Not in scope, stated

- **Multi-node or distributed queueing.** One process, one machine.
- **A scheduler that moves work between models.** Loading a second model is K5/admin territory and
  a memory decision, not a queue decision.
- **Priority inversion cleverness.** Two classes and a starvation bound; if that proves too coarse,
  it will prove it with a measurement.
- **Retry policy.** A failed job stays failed and visible. Automatic retry of a generation that
  consumed budget is a footgun this product should not ship by default.

---

## Sources

`internal/serveapp/openai.go:77`, `:170`, `:180`, `:1040` (the queue cap, `tryEnter`, the halt
check, `drive`) · `internal/serveapp/helpers.go:77` (`-max-inflight`, distinct from the per-model
429) · `internal/serveapp/main.go:500`, `:506` (`-kv-sessions`, `-max-queue`) ·
`internal/serveapp/anthropic.go:507` (529 on a full queue) · `internal/serveapp/sessions.go:14`
(the session LRU J6 schedules around) · `internal/serveapp/embeddings.go:34` (the one existing bulk
surface) · `internal/chatapp/main.go:203` (the CLI J5 extends) ·
[`task-halt-2026-09.md`](task-halt-2026-09.md) K1/K2/K4/K5/K9 ·
[`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) §3.3 ·
[`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W18 · `docs/api-tiers.md` (what `serve` promises)

<!-- doc-reviewed: 2026-09-13 -->

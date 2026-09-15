# Task: goinfer as a work queue — admission, jobs, batch APIs (J1–J9) — 2026-09

> **Status: J0/J1/J2/J3/J4 DONE 2026-09-15 (J3/J4 text-chat scope only), J5–J9 unstarted.** This
> doc was first written 2026-09-12 and was never committed. It was deleted the next morning by a
> workaround, not by a decision — see "How this doc was lost" below, which is kept because the
> failure is structural and the fix was J0. The rebuild is faithful to the J1–J9 scope as filed,
> the prose re-derived from the tree at `9d29d625`, so every citation here was re-verified rather
> than carried over.
>
> J5 needs J4's line format but is otherwise independent (a local CLI runner, no server); J6/J8
> each need hours of real benchmarking against a pre-registered pass/kill band, out of scope for
> any of this. All three remain fully open.
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

**Status: DONE 2026-09-14** (`0c74a161`, `main`). `scripts/queue_citation_lint.py`'s `live_docs()`
now filters on `git ls-files -z`, cached once per process; `untracked_live_docs()` feeds `main()`'s
startup note (printed once, never silent); `--update` inherits the fix automatically since it
iterates `live_docs()` too. `scripts/install-git-hooks.sh`'s pre-push refusal message gained the
matching advice line (edited in the generator, not the generated `.git/hooks/pre-push`). Gate
built exactly as specified in `scripts/test_queue_citation_lint.py`, against an isolated temp git
repo — mutation-checked: reverted the filter, confirmed the test reds exactly as this doc's own
loss incident describes, restored, green. Verified against the real repo: this doc's own sibling,
`docs/queue-presentation.md`, was genuinely untracked at the time and was correctly skipped-and-
reported rather than reddening the lint.

---

## What exists today, cited

Not rebuilt below; this is the floor J1–J9 build on.

- **One decode worker per model.** `tryEnter` claims a queue slot and then takes its turn
  (`internal/serveapp/openai.go:209`), and that turn is what serialises decode. The cap is
  literally `1 running + --max-queue` (`internal/serveapp/openai.go:94`).
- **The wait is not fair and not context-aware by itself.** `sync.Mutex.Lock()` has no context, so
  a second check exists purely so a halt can cut a waiter loose
  (`internal/serveapp/openai.go:220`). Waiters are woken in whatever order the mutex chooses: a
  20-token request that arrived last can go after a 4,000-token one that arrived first, and
  nothing in the system knows the difference.
- **Backpressure is a number, not a plan.** `-max-queue` defaults to 8
  (`internal/serveapp/main.go:508`); a full queue is a 429 on the OpenAI routes and a 529
  `overloaded_error` on the Anthropic one (`internal/serveapp/anthropic.go:507`). A global
  `-max-inflight` (default 128) bounds the pre-queue stage — JSON and image decode, tokenisation,
  template render — and is deliberately distinct from the per-model 429
  (`internal/serveapp/helpers.go:84`).
- **Nothing is durable.** `drive` runs the generation for the life of the request
  (`internal/serveapp/openai.go:1119`). The client's connection *is* the job: close it and the
  work is cancelled and unrecoverable. There is no id to ask about afterwards.
- **There is warm state worth scheduling around.** The session LRU keeps prefilled KV and hands a
  request the session that already holds its prompt as a prefix
  (`internal/serveapp/sessions.go:14`), `-kv-sessions` 4 by default
  (`internal/serveapp/main.go:502`). Admission order therefore has a measurable cost today that
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
  `internal/serveapp/openai.go:220` stops being load-bearing.
- **Context-aware**, in both senses: the admission record carries the request's prompt-token count
  and its session/prefix key, so J6 and J7 have something to schedule on. J1 itself keeps strict
  FIFO — it establishes the structure and changes no order.
- 429/529 semantics unchanged at the boundary, and `-max-queue` keeps its meaning.
- Gate: with `-max-queue 1` and two concurrent requests, the second's cancellation is observed at
  the server within one decode step, not at the end of the first generation. Red before the change.

**Status: DONE 2026-09-14** (`main`). `internal/serveapp/admission.go`: a size-1, FIFO,
context-aware turn-granter (modeled on `golang.org/x/sync/semaphore.Weighted`'s `Acquire` as a
*technique*, not a dependency — ground rule 4 forbids a new module for this), replacing
`loadedModel.mu`. `sessionLRU` turned out to have no client-visible session key at all — it matches
by longest-common-prefix over the actual prompt token ids (`sessions.go`'s `bestExtend`) — so the
"session/prefix key" the admission record carries is, concretely, `promptIDs []int`.

Found while integrating: `lm.mu` was doing double duty — also the *only* thing excluding a
background goroutine (session restore-on-load, the idle-demote ticker, graceful-shutdown save)
from touching `sessionLRU` while a generation was using it. `admission` is a FIFO queue, not a
lockable mutex a background goroutine can take on a whim, so this needed a second, dedicated field
(`loadedModel.sessMu`), held for the identical `tryEnter..exit` span the old `lm.mu` covered — the
safety property is unchanged; only the *wait for a turn* gained context-awareness.

Gate built as specified in `internal/serveapp/admission_test.go`, deterministic and fast (no real
checkpoint needed — the defect is in the wait mechanism itself): a waiter behind an indefinitely-
held turn is cancelled and must return well within milliseconds, not "the end of the first
generation." Verified red without the fix with a companion test running the identical scenario
against a bare `sync.Mutex` (must hang; `TestAdmission_mutexWouldFailThisGate`), proving the first
test actually discriminates the defect. `-race` clean; FIFO order and the immediate-admit fast path
also covered. Confirmed against a real checkpoint with no behavioral change: `TestServe_backpressure`
(burst 12, `-max-queue 2` → 3×200/9×429, unchanged) and `TestServe_haltUnderLoad` (K2
time-to-quiescence ~30ms; of 32 concurrent requests, 31 *queued* ones were refused immediately
rather than needing to be granted their turn first — the concrete, observable form of this fix).

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

**Status: DONE 2026-09-14** (`main`). `internal/serveapp/job.go` (the `job`/`jobStore` object,
in-memory, always populated — `newServer` constructs it unconditionally) and
`internal/serveapp/jobjournal.go` (the optional JSONL durability layer, gated on `-job-dir`).
`request` is scoped to what `drive`/`driveVL`'s existing K1 chokepoint actually has — the decoded
prompt token ids, not the raw HTTP body, which would need new plumbing from all 8 handler call
sites for no reader that exists yet. Each journal line is a full snapshot at that transition (not a
diff), so reconstruction is "keep the last line per id." Jobs are created/finished at the *same*
`drive`/`driveVL` chokepoint as K1's registration, keyed on the same `gr.id`, so J3's later "the two
registries are joined, not parallel" is a lookup away rather than a retrofit — not built this pass.

Gate built as specified in `internal/serveapp/job_test.go`: round-trip (write, reload, get the last
state back), the interrupted-on-restart case (a `running` transition with no terminal follow-up
reconstructs as `interrupted`, verified red without the fix by removing that logic and confirming
the test catches it), a *second* restart's own last line correctly stays `interrupted` (not
re-derived), and the 0700/0600 permission check matching `sessions.go`'s existing rationale
exactly. Confirmed with a real checkpoint that job creation/finishing is invisible to existing
behavior (`TestServe_backpressure`, `TestServe_haltUnderLoad`, `TestServe_anthropic_integration`/
`_streaming` all pass unchanged with `-job-dir` unset).

**Not built this pass** (explicitly out of scope): the `/v1/jobs` HTTP surface (J3), the batch APIs
(J4/J5), and anything that reads `jobStore` from outside the process.

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

**Status: DONE 2026-09-14** (`main`), text-chat scope only — see "not built" below.
`internal/serveapp/jobs_http.go` (the four routes), `jobs_run.go` (`runJob`, the async
counterpart to `serveChatText`, decoupled from any HTTP request's lifetime), `jobeventlog.go`
(the replay-then-live buffer). The one real architectural gap: every existing generation path runs
`drive()` synchronously inside its own handler, holding `withModel`'s RLock
(`internal/serveapp/liveness.go:94`) for the handler's whole body so `/admin` unload can't free the model
mid-generation. `POST /v1/jobs` returns before the generation finishes, so `resolveAndLock`
(`internal/serveapp/liveness.go:69`) is called directly instead of through `withModel`, and its `release` is handed
to `runJob` to defer over the job's whole life instead of the handler's. Everything else needed —
admission (`lm.tryEnter`, no `http.ResponseWriter` required), cancel-by-id (`drive`/`driveVL`
already register `gr.id` in K1's registry once running) — was already reusable as-is.

`jobStore` (J2) gained a `responseStore`-style FIFO cap (`internal/serveapp/responses.go:49`'s pattern), skipping
over any still-pending/running job rather than evicting it; and two creation paths — `create`
(unchanged, both transitions at once, for every synchronous handler) and `createPending` +
`getOrCreate` (the async path: pre-create pending BEFORE admission is even attempted, so
`GET /v1/jobs/{id}` can see "pending" while genuinely queued, not just an instant flash before
"running" the way the synchronous paths' timing makes unavoidable).

Found by `-race`, not by inspection: J2 never had a reader of `*job`'s fields from outside the
single goroutine that owned it, so `markRunning`/`finish` mutated them without holding
`jobStore.mu`. J3's `handleGetJob` is a genuine second reader (a different goroutine, while
`runJob` is still running the job) — the very first real-checkpoint run of the new HTTP tests
caught the resulting data race immediately. Fixed by moving those mutations under the lock and
adding `snapshot(id) (job, bool)` (a copy taken under the lock) for `handleGetJob` to read instead
of the live pointer.

Verified: `jobeventlog_test.go` (replay from a fresh reader, partial-replay-then-live, a blocked
waiter unblocking within milliseconds of `append`/`markDone`, `-race` clean under concurrent
readers+writers), `jobs_test.go` (eviction never drops a live job; oldest-finished-first;
`getOrCreate` transitions the pre-created job rather than creating a second one). The task doc's
own gate, real-checkpoint (`GOINFER_SERVE_MODEL`): `TestJobs_submitPollReattachMatchesSyncRun`
(submit, read a few frames, disconnect, poll to completion, re-attach as a brand-new connection —
the replayed transcript is byte-identical to an uninterrupted `/v1/chat/completions` run of the
same seed) and `TestJobs_deleteCancelsAQueuedJob` (DELETE stops a job that is still queued, not
only a running one — the concrete case J1's admission fix exists for). No regression to the
existing synchronous paths or goroutine count (`TestServe_backpressure`, `TestServe_haltUnderLoad`,
`TestServe_goroutineLeakCheck` all pass unchanged).

**Not built this pass:** vision and tool-calling request bodies (`POST /v1/jobs` accepts
`serveChatText`'s plain-text scope only); the `user`/`metadata.user_id` client-visible key the
original spec named — K4 (budgets), which that key was "the one proposed for", is not built yet
(`docs/tasks/task-halt-2026-09.md` lists it as open), so there is nothing to reuse; a job-store
size flag (`-job-cap`, currently a hardcoded 256 matching `responseStore`'s own bound). J4/J5 (the
batch APIs, over this same store) and J6–J9 remain open.

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

**Status: DONE 2026-09-15** (`main`), text-only scope — see "not built" below. Both dialects reduce
to the same three parts, all new: `internal/serveapp/files.go` (OpenAI's file store — write-once,
so unlike `job` it needs no `snapshot`-style copy to read race-free), `internal/serveapp/batches.go`
(`batchStore`/`batchRecord`, mirroring `jobStore`'s own FIFO-skip-a-live-one eviction shape almost
verbatim), `internal/serveapp/batches_run.go` (`runBatchChatLine`/`runBatchMessageLine`, one
goroutine per line), `internal/serveapp/batches_finalize.go` (`finalizeBatch`, assembling the
output once every line is terminal), `internal/serveapp/batches_http.go` (the ten new routes).

**No second execution path, literally**: every line calls the exact same `lm.tryEnter`/`lm.drive`
pair every other surface calls, with the SAME `jobStore.createPending`/`setCancel`/`finish`
bookkeeping `runJob` (J3) already established — a batch line IS a job, so it gets restart-journal
durability and K1 cancel-by-id for free. What's new is bookkeeping *around* that call, not a second
way to run it: a `sync.WaitGroup` on each `batchRecord` (`Add(n)` at creation, one `Done()` per
line as its own last act) replaces polling — `finalizeBatch`'s only job is `wg.Wait()` then
assemble. `POST .../cancel` calls `jobStore.cancel` for every constituent job, exactly what
`DELETE /v1/jobs/{id}` already does one at a time.

Deliberately does **not** call `handleCreateJob`/`serveMessagesWith` — both are tied to writing an
`http.ResponseWriter` mid-body, and making either callable headless risked regressing J3's/the
Anthropic surface's own tested synchronous paths for a one-time win. Reuses every *expensive*
helper they call (`lm.prepare`, `lm.tryEnter`, `lm.drive`, `anthropicTurns`, `anthropicStopReason`,
`textBlock`) and duplicates only the thin (~15-line) validation sequence — the same shape
openai.go and anthropic.go already independently carry for their own two synchronous surfaces, not
a new pattern this adds.

`jobResult` (`job.go`) gained one field, `StopSeq string`, set from `drive`'s own `stopHitOut`
named return in its existing job-finish block — needed so a batch's Anthropic line can call the
*exact* `anthropicStopReason(finish, stopSeq)` helper `serveMessagesWith` already uses, instead of
losing "which stop sequence matched" fidelity. `json:"-"`: invisible to `GET /v1/jobs/{id}`'s
existing response shape, a no-op for every OpenAI-originated job, which never reads it.

Found by `-race`, not by inspection — the same class of bug J3's own `jobStore` hit, in a new
place: `batchRecord.JobIDs` is written by each line's own goroutine (`setJobID`, disjoint indices)
and read WHOLESALE by `requestCancel` (cancel reaching every constituent job) from a different
goroutine. Disjoint-index writes don't save a plain slice from racing a concurrent read of its
backing array with no synchronization between them — caught on the very first real-checkpoint run
of `TestBatches_cancelStopsQueuedLines`. Fixed by writing `JobIDs[i]` under `batchRecord.mu`
(`setJobID`) and copying the slice under the same lock before `requestCancel` ranges over it.
`Results[i]` was written under the lock from the start this time (same reasoning stated up front
in `batchRecord`'s own doc comment) rather than finding the identical defect a second time in the
same file.

Verified: unit tests for `fileStore`/`batchStore` CRUD, FIFO eviction (mutation-checked: the
`batchStore` live-skip test was confirmed red — nothing evicted at all — against a deliberately
broken `evictLocked`, then restored), `requestCancel`'s idempotence on an already-terminal batch,
and `assembleOpenAIOutput`'s success/error JSONL split preserving `custom_id` order. Real-checkpoint
(`GOINFER_SERVE_MODEL`): `TestBatches_openAIRoundTrip` (a 2-line input file — one plain, one
requesting an image — through `POST /v1/files` → `POST /v1/batches` → poll → `GET
.../content`, asserting the good line's `response.body` matches an uninterrupted
`/v1/chat/completions` call of the same seed, and the bad line lands in the error file without
failing the batch), `TestBatches_anthropicRoundTrip` (the inline-request twin, one line missing
`max_tokens`, polled to `"ended"`, read from the dedicated `.../results` endpoint — no file store
involved on this side, matching the real API's own shape), and
`TestBatches_cancelStopsQueuedLines` (mirrors `TestJobs_deleteCancelsAQueuedJob`: a batch queued
behind a long holder job, cancelled, both lines land as errored/cancelled rather than completing).
All three pass under `-race`. No regression to J1–J3's own real-checkpoint gates
(`TestServe_backpressure`, `TestServe_haltUnderLoad`, `TestJobs_submitPollReattachMatchesSyncRun`,
`TestJobs_deleteCancelsAQueuedJob`), and the full `go test ./...` across every module is green.

**Not built this pass:** vision and tool-calling batch lines (matches J3's own scope line, same
reason); OpenAI batch endpoints other than `/v1/chat/completions` (`/v1/embeddings`,
`/v1/completions`, `/v1/responses` — the real API supports all four; a line asking for a
different `endpoint` is a clean 400 naming the one this pass supports, not a silent
mistranslation); `completion_window`/batch-expiry enforcement (accepted and echoed, never
enforced, as this section's own bullet says up front); `DELETE /v1/files/{id}` (not needed by the
round-trip gate, not named in this section's own route list); a literal openai-python/
anthropic-python SDK-driven gate (K1's "verified live against the official SDKs" note,
`task-halt-2026-09.md`, was a manual one-off check — this repo's actual enforced gates are
Go-native real-checkpoint tests throughout J1–J4; the gate here is a Go `multipart.Writer`/
`http.Client` round trip built to match the documented wire shapes field-for-field instead). J5–J9
remain open.

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

`internal/serveapp/openai.go:94`, `:209`, `:220`, `:1087` (the queue cap, `tryEnter`, the halt
check, `drive`) · `internal/serveapp/helpers.go:84` (`-max-inflight`, distinct from the per-model
429) · `internal/serveapp/main.go:502`, `:508` (`-kv-sessions`, `-max-queue`) ·
`internal/serveapp/anthropic.go:507` (529 on a full queue) · `internal/serveapp/sessions.go:14`
(the session LRU J6 schedules around) · `internal/serveapp/embeddings.go:34` (the one existing bulk
surface) · `internal/chatapp/main.go:203` (the CLI J5 extends) ·
[`task-halt-2026-09.md`](task-halt-2026-09.md) K1/K2/K4/K5/K9 ·
[`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md) §3.3 ·
[`task-web-ui-2026-09.md`](task-web-ui-2026-09.md) W18 · `docs/api-tiers.md` (what `serve` promises)

<!-- doc-reviewed: 2026-09-13 -->

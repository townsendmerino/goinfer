# REVIEW-10: "Work that survives a disconnect"

Body ~730 words. Build passes (`-drafts`). `figures: []` on purpose: no measured number is quoted. The 256, 8 and 9 in the text are code constants, not measurements.

## Claims and sources

| Claim | Source |
|---|---|
| `POST /v1/jobs` returns id + `pending` (202); GET, `/events`, DELETE routes exist | `internal/serveapp/main.go:714-717`; `jobs_http.go` |
| Job runs on a background context, not the request; disconnect does not cancel it | `jobs_run.go` `runJob` (bgCtx); `jobs_http_test.go` (comment "only DELETE /v1/jobs/{id} cancels a job") |
| Events replay from the start, then live; reconnect = new reader | `jobeventlog.go` doc comment; `handleJobEvents` |
| Stream closes with finish_reason + usage chunk, or error event; queue position on GET | task doc J3 "Additions 2026-09-15"; `jobs_run.go` `jobTerminalEvents`; `handleGetJob` |
| Job holds the model loaded so admin unload cannot free it | `jobs_run.go` `runJob` comment; task doc J3 |
| `-job-dir`: JSONL journal `jobs.jsonl`, 0700/0600, last line per id, running becomes `interrupted` | `jobjournal.go`; flag text `internal/serveapp/main.go:397` |
| Journal has prompt token ids, not text; result is NOT journaled | `job.go` (`PromptIDs` json tag; `result` unexported, comment "not journaled") |
| After restart: GET gives state, no result; `/events` gives 410 | `handleGetJob` (result only if non-nil); `handleJobEvents` (410 branch) |
| Without `-job-dir`, all in memory; files and batch records are in memory only | `job.go`, `files.go`, `batches.go` (no journal); `internal/serveapp/main.go:978-986` |
| Cap 256 each for jobs, files, batches; only terminal jobs evicted | `internal/serveapp/main.go:978-989`; `job.go` `evictLocked` |
| Text chat only; images/tools refused; batch endpoint must be `/v1/chat/completions` | `jobs_http.go`; `internal/serveapp/batches_http.go:84-86`; task doc J3/J4 "Not built" |
| `completion_window` accepted, not enforced | task doc J4 "Not built" |
| One generation per model at a time | `docs/releases/v0.19.0.md` ("One generation at a time per model") |
| Shipped in v0.19.0 | `CHANGELOG.md` [v0.19.0] 2026-09-18; `docs/releases/v0.19.0.md`; `gh release view v0.19.0` (published 2026-09-19T02:08Z) |
| Web Batch tab and reload re-attach use these routes | `CHANGELOG.md` (inside [v0.19.0]) |
| `goinfer-chat -batch` not built | task doc status line ("J5/J7/J9 unstarted") and J5 section; grep of `internal/chatapp/*.go` finds no `-batch` flag or resume code |
| `-max-queue` default 8, queue cap = 1+8, 0 = unbounded, 429 + Retry-After when full | `internal/serveapp/main.go:439,965`; `jobs_http.go` |
| Tests listed exist and assert what the table says | `jobs_http_test.go`, `job_test.go` (read the bodies) |

## Conflicts

1. **The brief's `goinfer-chat -batch` (J5) does not exist.** Task doc says unstarted; the source agrees. Presented as designed, not built.
2. **Release date.** CHANGELOG says v0.19.0 is 2026-09-18; `gh release view` says published 2026-09-19T02:08Z (that is the evening of the 18th in US time zones). The page says only "v0.19.0".
3. **`docs/server.md` does not document `/v1/jobs`, `/v1/files` or `/v1/batches`.** `docs/ARCHITECTURE.md` says "server.md documents each route". The page's curl examples come from the code and tests, not from server.md. `docs/server.md` only mentions `/v1/jobs` as a route that refuses the confidence flag. This is a doc gap to fix.
4. **`docs/tasks/task-work-queue-2026-09.md` J2 says the journal keeps "usage" and "error" as small fields**; the code also stores prompt ids. I followed the code.

## Left out as unverified

- No measurement or timing exists for jobs or batches, so none is quoted.
- No test kills a real server process. The restart case is journal-file-level only. The page says so.
- Whether SIGTERM shutdown waits for or interrupts running jobs: not checked, not claimed.
- Whether the journal file is ever compacted: I found no rotation code in `jobjournal.go`, but did not state it.
- Client disconnect on the synchronous `/v1/chat/completions` route: not claimed either way.
- API-key header form for these routes: not checked; the page says only "takes it like the others".
- The real-model tests' pass status is the task doc's word (2026-09-14/15). I did not run them.

## Questions for the owner

1. The large-batch limit (last item of "It doesn't") is from reading the code: `handleCreateBatch` starts one goroutine per line, each calls `tryEnter`, and the queue holds 1+`-max-queue` entries. Do you want that confirmed with a real run before it is published, or the bullet cut?
2. Should this page wait until `docs/server.md` documents the job and batch routes, or is code-derived curl acceptable?
3. Is "designed, not built" for `goinfer-chat -batch` the framing you want on a public page, or should the J5 mention be dropped?

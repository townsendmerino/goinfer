# Review notes: writeup 01, "Stop means stop" (drafted 2026-09-29)

Draft, `reviewed:` empty. Body 900 words (tables and code included). The site build with `-drafts` passes, so the three figures (`~10ms`, `1 of 32`, `31 of 32`) appear in the body and in the record. Nothing was run for this page: no server, tests or benchmarks. Everything below was read from files.

## Claims and sources

| Claim | Source |
|---|---|
| Cancel by id: `GET /admin/generations`, `POST /admin/generations/{id}/cancel {"reason"}`; reason required (400 otherwise); reply `{"id","found"}` | `internal/serveapp/admin.go` (`registerAdminRoutes`, `handleAdminGenerationCancel`) |
| Generation list fields (id, model, started, tokens_so_far) | `internal/serveapp/generations.go` (`generationSnapshot`) |
| Halt: `POST /admin/halt {"reason"}` (reason required), reply `{"halted":true,"reason","quiesced_in_ms"}`, blocks until quiescent, bounded 30 s; `POST /admin/resume`; `GET /admin/status` | `halt.go` (`halt`, `handleAdminHalt`), `admin.go` (`handleAdminStatus`) |
| 503 `{"error":"halted","reason"}` on inference routes while halted; `/health` gets `halted`/`halt_reason`/`halt_at` | `halt.go` (`haltGate`), `health.go` |
| Halt checked before the in-flight cap and again when a queued request is admitted; the missing second check was a bug found while building | `docs/tasks/task-halt-2026-09.md`, K2 "Found while building"; `openai.go` (`tryEnter`) |
| Stream ends `finish_reason:"cancelled"` plus a `goinfer_cancelled` event; non-stream gets 499 | `openai.go` (~line 797); `CHANGELOG.md` v0.18.0 K1 entry; `helpers.go` (`statusCancelled = 499`) |
| Socket: `-admin-socket`, mode 0600, no API key, `/admin/*` not registered on TCP (404 not 403) | `admin_socket.go`, `main.go` (~lines 393, 396, 618); `TestServe_adminSocket` |
| `--allow-admin` on TCP requires an API key, ignored when the socket is set | `main.go` (~lines 393, 476) |
| CLI `goinfer-serve status\|ls\|cancel\|halt\|resume`; default socket paths; flag-order trap | `admin_cli.go`, `admin_socket.go` (`defaultAdminSocketPath`); binary name from `docs/server.md` |
| `-halt-file` polled every 250 ms, `touch` halts, `rm` resumes; resume with the file still present re-halts | `halt.go` (`haltFilePoller`, comment on `resume`) |
| SIGUSR1 halt / SIGUSR2 resume; none on Windows | `haltsignal_unix.go`, `haltsignal_windows.go` |
| `-halt-exit-code N` | `main.go` (~line 395), `halt.go` |
| Mechanism: per-token context check; nothing asks the model to stop | task doc "What exists today" and ground rules; `generations.go` |
| Shipped in v0.18.0, 2026-09-13 | `CHANGELOG.md` (`## [v0.18.0] — 2026-09-13`, K1/K2/K5 entries); `git tag --contains 972bd2d7` lists `v0.18.0` |
| Measurement table (MacBook Apple Silicon arm64 CPU, Qwen2.5-Coder-0.5B Q4_K_M to int8int8, `4b979d0`, 32 requests, ~10ms, 1 of 32 / 31 of 32) | `docs/measurements/kill-switch-quiescence-2026-09-12.md` (header, "The number") |
| 10 ms poll granularity of the quiescence reading | `generations.go` (`waitEmpty`, `const poll = 10 * time.Millisecond`) |
| The test logs the time and asserts no bound (only a 15 s hang guard) | `halt_test.go` (~lines 125-140) |
| `TestServe_cancelByID` asserts (cancelled finish, event, id leaves registry, next greedy request identical) | `cancel_test.go` header comment and body; skips without `GOINFER_SERVE_MODEL` |
| Halt-file and signals smoke-tested by hand, not in a test | task doc K2 status; no test found for them |
| doesnt: client executes tools, cannot recall a call | task doc "Where goinfer sits" |
| doesnt: lease, budgets, session cancel, broker, drill, deployment units not built | task doc K3, K4, K6-K9 (no status line); grep found no `lease-file`, `max-gen-seconds`, `token-budget` in Go code and no `deploy/` dir |
| doesnt: SDK caveats (Anthropic SDK drops the event; OpenAI SDK strict validation raises) | task doc K1 "Found while building" (openai-python 3.8.0, anthropic-python 1.5.0) |
| doesnt: TCP admin shares the `/v1` key | `main.go` `-allow-admin` help text; task doc "Auth and exposure" |

## Conflicts between records

- **The brief said `docs/releases/v0.18.0.md`; it does not exist.** There is no such file under `docs/releases/`. The v0.18.0 facts come from `CHANGELOG.md` instead.
- **`docs/server.md` has no admin/control section.** It mentions `--allow-admin` for model load/unload only. The halt and cancel routes are documented in the task doc, `docs/ARCHITECTURE.md` (Control bullet) and `CHANGELOG.md`. The page says so in "Use it". The most useful doc fix is probably a real section in `server.md`.
- **Quiescence under `-race`.** The task doc says "up to ~40ms under `-race`"; the measurement record's table says quiesced 10 ms under `-race`, with a 37 ms admin round trip. I used the record (the primary source) and quote only `~10ms`. Both are dated 2026-09-12.
- **The record's "~2200× inside K2's own 2× per-token-latency gate bound"** is not quoted. I could not reproduce that ratio from the record's own table, and the test asserts no such bound.
- **Task-doc status is stale in places** (its header says K3+ open, which matches the code; some `docs/task-halt-2026-09.md` paths in older comments predate the move to `docs/tasks/`). I cite the current path.

## Left out as unverified

- Whether halt or cancel reaches embeddings requests, or every route that generates (I confirmed `gr.id` is set in chat, completions, responses, messages, vision, tools and jobs handlers, but did not audit batches or embeddings). The page says nothing about them.
- Any latency on Metal or CUDA, or with a batched GPU step (MC3, several generations at once). The record predates that work and the registry-empty wait was not re-measured. The page says only that nothing measures it.
- Halt-file and signal latency; no record.
- A Windows halt path beyond "file or socket" (code comment only).
- Whether a halt file already present at server start halts on the first poll (code suggests so, not tested or documented). Left out.
- The `resident_batch` field in `/admin/status` (added for a benchmark harness; not part of this story).

## Questions for the owner

1. Is the "same user can open the socket" limit fair to state as a limit? It is my reading of mode 0600 plus the task doc's threat model (which excludes a root adversary and defers separate-user deployment to the unbuilt K6). The code does not say it outright.
2. Should the page name the task-doc IDs (K1/K2/K5) or keep them out? I kept them out, apart from the file path.
3. The measurement is a single run by hand on the Mac by day, with no re-run since MC3 changed decode. Do you want it re-measured (on the night queue, both boxes, GPU backends) before this page is reviewed, or is "measured once" acceptable as written?

---
title: "Work that survives a disconnect"
area: "Serving"
order: 10
summary: "A generation submitted as a job keeps running after the client leaves, and a new connection can replay it. With -job-dir, a restart records what happened."
stand: "goinfer-serve can run a generation as a background job with its own id. The client can disconnect and come back to read the whole answer. A restart is a smaller promise, and this page says how much smaller."
measured: 2026-09-15
reviewed: 2026-09-29
facts:
  - {label: "survives a client disconnect", value: "yes, the job runs on and its text can be replayed"}
  - {label: "survives a server restart", value: "state only, and only with -job-dir; the text is lost"}
  - {label: "shipped in", value: "v0.19.0 (job queue, /v1/jobs, two batch APIs)"}
  - {label: "goinfer-chat -batch", value: "designed, not built"}
doesnt:
  - title: "It doesn't keep the answer through a restart."
    text: "With -job-dir, the journal (a log file of job states) records each job's state and its prompt token ids, but not the generated text. After a restart, GET /v1/jobs/{id} reports the state and no result, and the events stream answers 410 Gone. A job that was running when the server stopped is marked interrupted and is not run again."
  - title: "It doesn't keep anything without -job-dir."
    text: "Jobs, the replay buffer, uploaded files and batch records all live in memory, and the journal is off by default. A restart without it forgets every job id. Even with it, batches and files are not journaled, so a restarted server cannot tell you about a batch."
  - title: "It has no resumable batch runner yet."
    text: "goinfer-chat -batch in.jsonl -o out.jsonl is a planned command-line batch runner that would skip finished lines and continue after a crash. It is not built yet: the project's work-queue plan lists it as unstarted, and we found no such flag in the goinfer-chat source."
  - title: "It is text chat only, one at a time, and small."
    text: "Jobs and batch lines take plain text chat. A request with images or tools is refused, and a batch line must use /v1/chat/completions. A model runs up to four generations at once where it can (`-max-concurrent`, default 4) and one at a time otherwise (for example a vision model, an adapter, or a speculating GPU model). The stores hold 256 entries each. The completion_window field (the batch deadline in OpenAI's API) is accepted and never enforced."
  - title: "A large batch may not fit the queue."
    text: "Every batch line asks for a place in the model's queue at once. A model's queue holds the generations it is running plus `-max-queue` waiting ones (default 8): 9 places for a model that runs one at a time, 12 for one that runs four. By reading the code, lines past that fail with a 429 message saying the queue is full. We have not run a batch that large to confirm it."
figures: []
sources:
  - "docs/tasks/task-work-queue-2026-09.md"
  - "docs/releases/v0.19.0.md"
  - "docs/ARCHITECTURE.md"
  - "CHANGELOG.md"
  - "internal/serveapp/jobs_http.go"
  - "internal/serveapp/jobs_run.go"
  - "internal/serveapp/job.go"
  - "internal/serveapp/jobjournal.go"
  - "internal/serveapp/jobeventlog.go"
  - "internal/serveapp/batches_http.go"
  - "internal/serveapp/batches_run.go"
  - "internal/serveapp/jobs_http_test.go"
  - "internal/serveapp/job_test.go"
---

## The problem

A chat request normally lives as long as its HTTP connection. If a laptop sleeps, a browser tab reloads, or a script is killed halfway through a long reply, the connection goes and the generation (the reply being produced) goes with it. The machine spent the time, and you have nothing.

The same is true of a pile of requests. If you send a few hundred prompts one after another from a script and the script dies at the two-hundredth, you have to work out where it stopped.

## What goinfer does

`goinfer-serve` can run a generation as a job. You submit it, get an id back straight away, and the server keeps working whether or not anyone is connected:

```
curl -s localhost:8080/v1/jobs -d '{"model":"local","max_tokens":48,"messages":[{"role":"user","content":"Write a haiku about Go."}]}'
# {"id":"job_...","status":"pending"}

curl -N localhost:8080/v1/jobs/job_.../events   # stream it, then Ctrl-C at any point
curl -N localhost:8080/v1/jobs/job_.../events   # a new connection replays from the first token, then goes live
curl -s localhost:8080/v1/jobs/job_...          # state, usage and the finished text
curl -s -X DELETE localhost:8080/v1/jobs/job_...   # cancels it, queued or running
```

Closing the events stream does not cancel the job. Only `DELETE` does, or an operator's cancel or halt (see [Stop means stop](/different/01-stop-means-stop/)). A reconnecting client is treated as a new reader and gets the whole backlog, so there is no offset to remember.

For many requests there are OpenAI-style batches: upload a JSONL file (one request per line) to `/v1/files`, start it with `/v1/batches`, poll, then fetch the output file. Anthropic-style Message Batches are at `/v1/messages/batches`. Each line runs as its own job.

## How it works

`POST /v1/jobs` records a pending job and starts a goroutine that is tied to a background context, not to the request. That goroutine waits its turn in the model's queue, then runs the same generation path as `/v1/chat/completions`. It holds the model loaded until it finishes, so an operator's request to unload the model cannot free it while the job runs.

Every token is appended to an in-memory event log. The events endpoint reads that log from the start and then waits for more, so a late or repeated reader sees the same text. The stream ends with a `finish_reason` chunk and a usage chunk, or an error event if the job failed. While a job is queued, `GET /v1/jobs/{id}` reports its place in line.

With `-job-dir <path>`, each state change (pending, running, done, failed, cancelled) is appended to `jobs.jsonl` as one line. The file has owner-only permissions, because each line holds the prompt's token ids (the prompt as the numbers the model reads). On the next start the server replays the file and keeps the last line per id. It turns any job still marked running into `interrupted`, never `failed`, because nobody saw it fail.

## What was measured

Nothing here has a benchmark. The project's measurement records hold no timing or throughput figure for the job queue, and this page quotes no speed. What exists is tests that check the behaviour. The first two need a real model file, named by `GOINFER_SERVE_MODEL`, and skip without one.

| What is checked | Test | Needs a model? |
|---|---|---|
| Submit, read a frame, disconnect, poll to done, re-attach as a new connection: the replayed text equals an uninterrupted chat run with the same seed (the random seed used to pick tokens) | `TestJobs_submitPollReattachMatchesSyncRun` | yes |
| `DELETE` stops a job that is still queued | `TestJobs_deleteCancelsAQueuedJob` | yes |
| A job with no final line in the journal comes back as `interrupted`, and stays so on later restarts | `TestJobJournal_crashedJobReconstructsAsInterrupted`, `TestJobJournal_reopenCompletesTheInterruptedRecord` | no |
| Journal round trip and file permissions | `TestJobJournal_roundTrip`, `TestJobJournal_permissions` | no |

The first two tests are in [jobs_http_test.go](https://github.com/townsendmerino/goinfer/blob/main/internal/serveapp/jobs_http_test.go), the journal tests in [job_test.go](https://github.com/townsendmerino/goinfer/blob/main/internal/serveapp/job_test.go). The project's [work-queue plan](https://github.com/townsendmerino/goinfer/blob/main/docs/tasks/task-work-queue-2026-09.md) records the real-model tests as passing on 2026-09-14 and 2026-09-15. It does not say which machine or model they ran on, and we did not re-run them for this page. The restart case is tested by writing journal lines and reopening the file, not by killing a live server.

## Use it

- `goinfer-serve -model local=<file.gguf> -job-dir ~/goinfer-jobs` turns the journal on. Without `-job-dir`, jobs are memory-only.
- `-max-queue` sets how many requests may wait per model (default 8; `0` is unbounded). A full queue answers `POST /v1/jobs` with a 429 and `Retry-After`.
- Batches: `curl -F purpose=batch -F file=@in.jsonl localhost:8080/v1/files`, then `curl localhost:8080/v1/batches -d '{"input_file_id":"file-...","endpoint":"/v1/chat/completions","completion_window":"24h"}'`. Poll `GET /v1/batches/{id}` until it says `completed`, then fetch the file named by `output_file_id` from `/v1/files/{id}/content`. Failed lines go in a separate error file.
- Where the server needs an API key, these routes take it like the others.
- goinfer-serve's built-in web UI uses these same routes for its Batch tab and for re-attaching after a page reload. That shipped in v0.19.0 ([release notes](https://github.com/townsendmerino/goinfer/blob/main/docs/releases/v0.19.0.md)).
- The server guide, [docs/server.md](https://github.com/townsendmerino/goinfer/blob/main/docs/server.md), does not describe these routes yet. The Serving section of [docs/ARCHITECTURE.md](https://github.com/townsendmerino/goinfer/blob/main/docs/ARCHITECTURE.md) lists them.

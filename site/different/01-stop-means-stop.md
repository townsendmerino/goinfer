---
title: "Stop means stop"
area: "Control"
order: 1
summary: "Cancel one generation by id, or halt them all from an HTTP call, a file or a signal. Control sits on its own Unix socket, away from the API."
stand: "If a client loops, or you just want it to stop, one command stops the tokens: for one generation, or for the whole server. The controls live on a separate channel from the one the client talks to."
measured: 2026-09-12
reviewed: 2026-09-29
facts:
  - {label: "cancel one", value: "by generation id"}
  - {label: "halt all", value: "HTTP call, file, or signal"}
  - {label: "control channel", value: "its own Unix socket, mode 0600"}
  - {label: "halt time measured", value: "once, on one CPU box"}
doesnt:
  - title: "It doesn't sandbox what a client does with text it already has."
    text: "goinfer produces tokens and, when asked, tool calls. The client runs the tools. A halt stops the tokens; it cannot recall a call the client already received, and it cannot undo anything. The design doc says this in its first section."
  - title: "The socket is protected by file permissions, nothing more."
    text: "Anything that runs as the same user can open it, agent included. On the TCP listener, --allow-admin uses the same API key as /v1, so a client with the key can halt too. Separating users is a deployment job, and the systemd and launchd units that would do it are scoped but not built."
  - title: "The halt time was measured once."
    text: "One CPU run on one MacBook, a 0.5B model, 32 requests. The test logs the time and asserts no bound on it. Nothing measures it on Metal or CUDA, or with a large model."
  - title: "The rest of the plan isn't built."
    text: "A lease that halts when nobody renews it, token and time budgets, cancel by session, an agent-side broker and a drill that pulls every switch are still open in docs/tasks/task-halt-2026-09.md. The flags for a lease and budgets do not exist."
  - title: "Some SDKs hide the reason."
    text: "The cancel reason travels in an extra event. The Anthropic Python SDK drops that event, and the OpenAI Python SDK raises on it if strict response validation is on. The finish reason still reads cancelled."
figures:
  - {text: "~10ms", source: "docs/measurements/kill-switch-quiescence-2026-09-12.md"}
  - {text: "1 of 32", source: "docs/measurements/kill-switch-quiescence-2026-09-12.md"}
  - {text: "31 of 32", source: "docs/measurements/kill-switch-quiescence-2026-09-12.md"}
sources:
  - "docs/tasks/task-halt-2026-09.md"
  - "docs/measurements/kill-switch-quiescence-2026-09-12.md"
  - "CHANGELOG.md"
  - "docs/ARCHITECTURE.md"
  - "internal/serveapp/halt.go"
  - "internal/serveapp/admin.go"
  - "internal/serveapp/admin_socket.go"
---

## The problem

A program that drives a model server can go wrong in boring ways: a retry loop, or a model that keeps asking for tool calls. Sooner or later you want it to stop, while the misbehaving client is still running.

Killing the server works, but it unloads the model, and it cannot stop one generation and leave the rest alone.

## What goinfer does

Every generation has an id, the one already in the response (`chatcmpl-...` for chat). You can list the running ones and cancel one by id, or halt everything without a restart. The model stays loaded, so resuming is instant.

Start the server with a control socket:

```sh
goinfer-serve -model local=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
  -admin-socket /tmp/goinfer-admin.sock
```

With the socket set, the same binary is the control tool:

```sh
goinfer-serve status
goinfer-serve ls                         # id, model, start time, tokens so far
goinfer-serve cancel chatcmpl-... "stuck in a loop"
goinfer-serve halt "operator stop"
goinfer-serve resume
```

If the socket is not at the default path (`/run/goinfer/admin.sock`, or under `~/Library/Application Support/goinfer/` on macOS), pass `-admin-socket <path>` before any reason text. Go's flag parser stops at the first plain word, so a reason typed first hides the flag.

The routes are plain HTTP, so `curl --unix-socket /tmp/goinfer-admin.sock -X POST http://localhost/admin/halt -H 'Content-Type: application/json' -d '{"reason":"operator stop"}'` works too. Cancel replies `{"id": ..., "found": true}`; `found` is false if the generation had already finished. Halt replies `{"halted": true, "reason": ..., "quiesced_in_ms": <n>}` only after every cancelled generation has stopped. A `reason` is required for both.

What the client sees: a streaming reply ends with `finish_reason: "cancelled"` and one more event, `goinfer_cancelled`, carrying the id and reason, so it cannot be taken for a normal stop. A non-streaming reply gets a 499 error. While halted, every inference route answers `503 {"error":"halted","reason":...}`, and `GET /health` reports `halted`.

Two other ways to halt, with no HTTP client: `-halt-file <path>` (polled every 250 ms; `touch` halts, `rm` resumes), and `SIGUSR1` to halt and `SIGUSR2` to resume. Windows has no `SIGUSR1`; it uses the file or the socket.

## How it works

Every generation registers its cancel function under its id. A cancel cancels that generation's context, which the token loop already checks on every token, so it stops at the next token. Nothing asks the model to stop.

A halt does three things in order: set a flag, cancel every registered generation, and wait (up to 30 seconds) until the registry is empty. The flag is checked before the in-flight cap, so a halt does not wait for a slot. It is checked again when a queued request is finally admitted. Without it, requests waiting behind the model's one decode worker would have started after the halt. The task doc records this as a bug found while building.

Control lives on its own socket for a plain reason: a client that can reach `/v1` should not also be able to reach the stop. Set `-admin-socket` and `/admin/*` is not registered on the TCP listener at all. A request there gets a 404, not a 403, so it does not confirm the routes exist. The socket has no API key; its permissions (mode 0600) are the check. Without a socket, `--allow-admin` puts the routes on the TCP listener behind the API key, and it refuses to start without one.

## What was measured

One record, dated 2026-09-12: `docs/measurements/kill-switch-quiescence-2026-09-12.md`.

| | |
|---|---|
| Machine | MacBook, Apple Silicon (arm64), CPU backend |
| Model | Qwen2.5-Coder-0.5B-Instruct, Q4_K_M quantized to int8int8 at load |
| Build | goinfer `4b979d0` |
| Test | `TestServe_haltUnderLoad`: 32 concurrent streaming chats, `-max-inflight 32 -max-queue 32`, then `POST /admin/halt` |
| Time from halt to every cancelled generation stopped | ~10ms, with and without `-race` |
| Outcome | 1 of 32 cancelled mid-stream; 31 of 32 refused with 503 before starting |

Only one request streams at a time, because this build runs one generation per model. The other 31 were queued, and the halt refuses them at admission.

The ~10ms is coarse. The server checks for an empty registry every 10 ms, so a reading of 10 means "within about one poll". The record calls it a single-machine sample, not a gate threshold.

Two tests assert behaviour, not time. `TestServe_cancelByID` (needs a real `.gguf`) cancels a 4096-token stream by id and checks the `cancelled` finish, the `goinfer_cancelled` event, that the id leaves the registry, and that the next greedy request matches one taken before. `TestServe_adminSocket` needs no model: halt, status and resume work over the socket without a key, and `/admin/halt` on TCP returns 404. The model-backed tests skip unless `GOINFER_SERVE_MODEL` is set. The halt-file and signal triggers have no test; the task doc says they were smoke-tested by hand.

## Use it

- `-admin-socket <path>` puts `/admin/*` on a mode-0600 Unix socket and off TCP. Off by default.
- `--allow-admin` puts `/admin/*` on the TCP listener. It needs `--api-key`, and is ignored when `-admin-socket` is set.
- `-halt-file <path>` halts while the file exists. If you `resume` while the file is still there, the next poll (within 250 ms) halts again.
- `-halt-exit-code N` exits the process with code N after a halt, so a restarter can be told not to undo it.
- These shipped in v0.18.0 (CHANGELOG, dated 2026-09-13). The routes are not yet described in `docs/server.md`; the task doc and `docs/ARCHITECTURE.md` are the references.

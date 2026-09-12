# Task: halting goinfer — cancel, halt, lease, and the kill drill — 2026-09 (K1–K9)

> **Status: OPEN, drafted 2026-09-12** against `main` + branch `aikit-v1.41.0-bump`. Scoping doc
> for the question "if an agent built on goinfer misbehaves, what stops it, and how do we know
> the stop works". Companion to `docs/task-gpu-paths-2026-09.md` (nothing here touches kernels).
>
> Suggested order: K1 → K2 → K5 → K3 → K4 → K6 → K8 → K7 → K9. K1+K2+K5 are the useful minimum
> (cancel a task, halt the server, from a channel the agent cannot reach); K3 is the one that
> changes the guarantee; K8 is what makes any of it believable.

## Where goinfer sits, and what that means for a kill switch

goinfer is the **model server**. `serve` turns prompts into tokens and, when a client asks for
tools, emits `tool_calls` (`internal/serveapp/openai.go:396`); **the client executes them**.
The one in-tree agent (`demo/agent`) has a single read-only tool — ken `search` over MCP
(`demo/agent/agent/kenclient.go:42`). So goinfer cannot, by itself, stop a client that already
received a `tool_call` from acting on it, and it cannot undo anything. What it can do, and what
this doc is about:

1. stop producing tokens and tool calls — per generation, per session, or globally — from a
   channel the agent cannot reach, with a bound on how long that takes;
2. stop **by default** when nobody is telling it to continue (a lease, fail-closed), so every
   failure of the supervisor, the network, or the operator's attention stops the agent rather
   than leaving it running;
3. ship the executor pattern clients should use, in `demo/agent`, so "tool calls go through a
   broker that checks the lease before every call" is the path of least resistance;
4. prove it — a drill that pulls each switch under load and measures time-to-quiescence.

The threat model is a runaway or misbehaving agent loop (a client in a retry storm, a model
emitting tool calls forever, a prompt-injected client doing something it should not), and an
operator who needs one command that works. It is not an adversary with root on the box; a
process that can rewrite its own service unit is out of scope for anything inside the process.

## What exists today (cited, so nothing is rebuilt)

- **Per-token cancellation.** The generation loops check `ctx.Err()` every token
  (`decoder/model.go:1097,1128,1263`; `decoder/generate_vl.go:19,42`). Cancel a request's context and
  it stops within one token. This is the mechanism every level below builds on; nothing new is
  needed in `decoder/`.
- **Graceful shutdown.** SIGINT/SIGTERM → stop accepting, 30 s drain, exit
  (`internal/serveapp/main.go:662–683`). Cooperative: a stuck handler holds it for 30 s.
- **Concurrency cap.** `-max-inflight` (default 128) over the inference handlers
  (`internal/serveapp/main.go:572–580`). A cap, not a budget: it bounds parallelism, not total work.
- **Auth and exposure.** Loopback by default; `-api-key` required off-loopback; `/admin/*`
  opt-in behind `-allow-admin` **on the same listener and the same key as `/v1`**
  (`internal/serveapp/main.go:606–607`). Today an agent that can call `/v1` can also call `/admin` if admin is on.
- **Unguessable ids** for responses/messages/tool calls (`internal/serveapp/helpers.go:493–504`) — the handle a
  cancel-by-id needs already exists; it is just not registered anywhere.
- **Client-side interrupt** in the demo agent: `signal.NotifyContext(os.Interrupt)`
  (`demo/agent/cmd/stdlib-agent/main.go`, the `signal.NotifyContext(os.Interrupt)` call near line 105 — no `:NNN` citation here since the lint's path regex cannot parse a hyphenated directory segment) — Ctrl-C from the terminal, nothing else.

Missing: cancel someone else's generation by id; a global halt short of SIGTERM; any notion of
a lease; budgets beyond the parallelism cap; a control channel separate from the data channel;
a supervised deployment shape; an executor pattern; a drill.

## Ground rules

- **The stop never depends on the model.** No level below asks the model to stop; every level
  cancels a context, flips an atomic, closes a socket, or kills a cgroup.
- **Fail closed where a lease is on.** Missing/late signal → halt. Never "assume fine".
- **Control and data on different channels.** Nothing a `/v1` client can reach may halt,
  resume, or renew. If the same process serves both, they are different listeners with
  different credentials.
- **Every halt is loud and attributed:** a distinct `finish_reason`, a health field, a log line
  with the reason and who/what issued it. A silent stop is as bad as a silent continue.
- **Every switch has a measured time-to-quiescence** in the gate ledger, and a drill that
  re-measures it. An untested kill switch is a belief.
- One doc. Findings go into the per-item status lines here.

---

## K1 — Cancel by id: the task-level switch

**Where.** `internal/serveapp/`: the handlers that create a generation context
(`internal/serveapp/openai.go:977,1081`, the anthropic/responses twins), `internal/serveapp/helpers.go`'s `reqID()`.

**Fix.** A process-wide registry `map[id]*generation{cancel, started, model, session, tokens,
toolCalls}` populated when a handler mints its id and cleared on completion. `GET
/admin/generations` lists in-flight; `POST /admin/generations/{id}/cancel` calls the cancel and
records `reason`. The token loop exits at its next `ctx.Err()` check (already present); the
stream terminates with `finish_reason: "cancelled"` (OpenAI) / `stop_reason: "cancelled"`
(Anthropic) and one final SSE event naming it, so a client cannot mistake it for a natural stop.
Resident state stays consistent: `resBusy` released, `resIDs` forgotten (the same cleanup the
error path does today). Also `POST /admin/sessions/{id}/cancel` for every generation of one
session, since an agent loop is a session.

**Gate.** Start a 4k-token generation, cancel at a random point, assert: no token with a
timestamp after the cancel; the final event names `cancelled`; the next request on the same
model is bit-identical to a fresh process (resident state sane).

**Size.** Small.

## K2 — Global halt: the server-level switch, no restart

**Where.** `main.go` handler chain (`inf`, `auth`), `handleHealth`.

**Fix.** An atomic `halted` with a reason. `POST /admin/halt {reason}`: set it, cancel every
generation in K1's registry, and from then on every inference route returns `503 {"error":
"halted", "reason": …}` before taking an inflight slot; `/health` reports `"halted": true` with
the reason and timestamp; `POST /admin/resume` clears it. Two triggers besides HTTP: **a halt
file** (`-halt-file /run/goinfer/HALT`, polled every 250 ms — `touch` from any shell halts,
`rm` resumes; the serve user must have read but not write on its directory, see K6), and
**SIGUSR1** = halt, **SIGUSR2** = resume, so a supervisor can halt without opening a socket.
Halt is not shutdown: the model stays loaded, resume is instant. `-halt-exit-code N` makes
halt exit the process instead (for K6's "do not auto-restart after a halt").

**Gate.** 32 concurrent generations, halt, assert every stream ended `cancelled` and
**time-to-quiescence** (halt → last token emitted) recorded and under 2× the per-token latency.
Under `-max-inflight` saturation too — a halt that has to wait for a slot is not a halt.

**Size.** Small.

## K3 — The lease: fail-closed by default when nobody is renewing

**Where.** New `internal/serveapp/lease.go`; `main.go` flags.

**Fix.** `-lease-file <path> -lease-ttl 60s` (and `-lease-url` for a supervisor endpoint, later).
While set, serve is **halted unless** the lease file's mtime is within TTL; a renewer `touch`es
it. Expiry → K2's halt with reason `lease expired`; renewal → resume. Startup with a stale or
absent lease starts halted, not running. Default **off** so nothing changes for today's users;
documented as **the** mode for anything that executes tool calls. The property this buys, and
the reason it is the item that changes the guarantee: every failure path — supervisor crash,
partition, operator away, disk full so the touch fails — resolves to *stopped*. A signal-based
kill has the opposite property on every one of those.

`goinfer lease renew <path> --every 20s` as a tiny renewer for the by-hand case; a systemd timer
for K6.

**Gate.** Run under lease, stop the renewer, assert halt within TTL + poll interval; restart the
renewer, assert resume. Start with no lease file, assert halted from the first request.

**Size.** Small.

## K4 — Budgets: hard caps the model never sees

**Where.** `serveapp` request handling; per-session state (`lm.sessions`).

**Fix.** Caps enforced in serve, each producing K1's `cancelled` (per generation) or K2's halt
(process-wide) with the cap named as the reason: per generation — wall-clock (`-max-gen-seconds`)
and tokens (exists as `max_tokens`; add a server-side ceiling the request cannot raise); per
session — total tokens, total `tool_calls` emitted, generations per minute; per process —
total tokens since start or since resume (`-token-budget`), after which serve halts and stays
halted until an explicit resume. The session and process budgets are the ones that catch a loop:
a client in a retry storm never trips a per-request cap.

**Gate.** Exceed each cap in a test; assert the stop, the reason string, and that resume clears
only what it should (a process budget is not reset by resume unless asked).

**Size.** Small–medium (the session accounting is the work).

## K5 — Control on its own channel: the admin socket

**Where.** `internal/serveapp/main.go:606–607` admin routes; new listener.

**Fix.** `-admin-socket /run/goinfer/admin.sock` (unix, mode 0600, owned by the supervisor's
user): `/admin/*` — load/unload, generations, halt, resume, lease status — move there, off the
`/v1` listener entirely, and `-allow-admin` on the TCP listener becomes a loud opt-in that the
docs call out as "the agent can reach this". `goinfer serve ls|cancel <id>|halt|resume|status`
CLI subcommands over the socket, so the operator's command is one word and needs no curl and no
key. On macOS the same, under `~/Library/Application Support/goinfer/`.

**Why before K3.** A lease renewed over the data channel is a lease the agent can renew.

**Gate.** With the socket set, `/admin/*` on TCP returns 404; halt over the socket works with
no api-key; a `/v1` client with the key cannot halt or resume.

**Size.** Small.

## K6 — Supervised deployment: the layer that does not run goinfer's code

**Where.** New `deploy/systemd/goinfer-serve.service`, `goinfer-lease.timer`, and a launchd
plist; `docs/serving.md` (or wherever serve's deployment doc lives — one place).

**Fix.** A unit that makes the layers below real: own user; own slice with
`KillMode=control-group` and `TimeoutStopSec` = the K2 quiescence bound + grace, so `systemctl
kill` and `stop` take every descendant; `Restart=on-failure` with the K2 halt exit code in
`RestartPreventExitStatus` so a halt is not undone by the restarter; `ProtectSystem=strict`,
`ReadWritePaths=` only the model cache, `NoNewPrivileges=yes`, `PrivateTmp`; `MemoryMax` and
`CPUQuota` as budgets the process cannot argue with. The halt file's directory and the lease
file are owned by the supervisor user, **read-only to the serve user** — the process reads the
switches, it cannot flip them. A `goinfer-lease.timer` that renews only while a named condition
holds (a file the operator maintains, a check script — the site decides). For the container
case, the same as a compose file with the socket and halt dir as mounts.

**Gate.** `systemctl kill -s KILL goinfer-serve` with a detached child alive: assert the child
is dead too. Halt exit code → unit stays stopped. Serve user cannot write the halt file.

**Size.** Small; the doc is most of it.

## K7 — The executor pattern: `demo/agent` gets a broker

**Where.** `demo/agent/agent/agent.go` (the loop), `kenclient.go` (the one tool).

**Fix.** Every tool call goes through `agent.Broker`: before each call it checks the lease
(the same file K3 uses, or serve's `/health` `halted` flag over the data channel as a weaker
fallback), the session budget (calls, wall-clock), and a per-tool policy (`read-only` |
`reversible` | `irreversible`); irreversible calls go into a hold queue with a delay and are
discarded if a halt lands inside the window; every call — allowed, held, refused — is appended
to a JSONL log with the generation id serve gave it. The demo's only tool is read-only, so the
queue is exercised by a test tool, not by ken. Kept to one file: this is a pattern for people
who execute goinfer's `tool_calls`, not a framework. The README gets a section: "if your agent
executes tool calls, this is the shape; here is why the lease check is per call and not per
turn."

**Gate.** Halt mid-turn: assert no tool call is *issued* after the halt timestamp (serve stops
the tokens; the broker stops the effects — two independent stops).

**Size.** Small–medium.

## K8 — The kill drill: `gate kill`

**Where.** `cmd/gate` (runner exists: `run.go`, `parity.go`, `gpu.go`); gate ledger.

**Fix.** A gate that starts serve under lease with the admin socket, drives a long tool-using
loop through `demo/agent` against a small model, and at a random point pulls **each** switch in
turn across runs: cancel-by-id, session cancel, halt over the socket, halt file, lease expiry,
each budget, SIGTERM, SIGKILL of the process group. For each: no token and no tool call after
the switch timestamp; process tree empty within the grace; resident state sane after resume (a
parity check against a fresh process); time-to-quiescence recorded in the ledger with the
machine label. CPU-only, so it runs on every box and in CI. Counts, like perfgate, as release
evidence: a release that has not pulled the switches has not shipped a kill switch.

**Size.** Medium. This is the item that makes the rest true.

## K9 — The record: an append-only generation log

**Where.** `serveapp`; a path chosen in K6.

**Fix.** One JSONL line per generation — id, session, model, start, end, `finish_reason`,
tokens, `tool_calls` emitted, budget state at end, issuer of any cancel/halt — written to a
file the serve user can append to but not truncate or rewrite (`chattr +a` on Linux; a
separate logger process or the supervisor's journal where that is not available). After a
halt, this is how the operator answers "what did it do before we stopped it", which is the
question every rogue-agent story ends on.

**Size.** Small.

---

## Not in scope, stated

- Undoing effects. A kill stops future actions; the hold queue (K7) narrows the window, it does
  not reverse anything. Reversibility is the tool's problem, and the client's.
- Stopping a client that has already received a `tool_call`. K7 is the offered pattern; a
  client that ignores it is outside what serve can do.
- Adversaries with the serve user's write access to the unit, the socket dir, or the lease.
  K6 makes those root-only; an attacker with root is not this doc.

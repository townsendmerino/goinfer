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
tools, emits `tool_calls` (`internal/serveapp/openai.go:432`); **the client executes them**.
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
  (`internal/serveapp/main.go:685–683`). Cooperative: a stuck handler holds it for 30 s.
- **Concurrency cap.** `-max-inflight` (default 128) over the inference handlers
  (`internal/serveapp/main.go:501,619-621`). A cap, not a budget: it bounds parallelism, not total work.
- **Auth and exposure.** Loopback by default; `-api-key` required off-loopback; `/admin/*`
  opt-in behind `-allow-admin` **on the same listener and the same key as `/v1`**
  (`internal/serveapp/main.go:658`). Today an agent that can call `/v1` can also call `/admin` if admin is on.
- **Unguessable ids** for responses/messages/tool calls (`internal/serveapp/helpers.go:500-511`) — the handle a
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

**Where (original, pre-K1 locations — see the Status paragraph below for the shape that shipped).** `internal/serveapp/`: the handlers that create a generation context (`internal/serveapp/openai.go:1058,1081` at the time this was drafted, now superseded by `drive`/`driveVL`'s own `ctx, cancel := context.WithCancel(parent)` at `internal/serveapp/openai.go:1058,1197`), `internal/serveapp/helpers.go`'s `reqID()`.

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

**Status: DONE 2026-09-12** (branch `killswitch-k1-k2-k5`, off `main`). Registered inside
`drive`/`driveVL` (the two functions every generation surface funnels through), not at each
handler — a caller opts in by setting the already-existing id on `genRequest` before calling
either. `GET /admin/generations`, `POST /admin/generations/{id}/cancel {"reason"}` shipped, both
behind the existing `-allow-admin` gate on the TCP listener (K5 moves them to the socket).
`TestServe_cancelByID` is the gate test, run against a real local `.gguf`
(`GOINFER_SERVE_MODEL`, this box's `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`); it caught a real
bug before landing — see "found while building" below.

**NOT done: `POST /admin/sessions/{id}/cancel`.** goinfer's own "session" (`sessionLRU`/
`decoder.Session`, `sessions.go`) is a content-addressed KV-reuse cache selected by
longest-common-prefix match (`bestExtend`) — it has no client-visible, stable identifier an
operator could type into a cancel request, and none of the four request surfaces carries a
session/user/thread id either (`embeddings.go`'s `User` field is the only "user" field in the
tree, and it is explicitly "accepted, ignored"). Flagging this rather than inventing an id
scheme unilaterally, per this doc's own instruction.

**Found while building, not assumed:**
- The registry's cancel function MUST cancel `parent` (the context the handler received), not
  the child context `drive`/`driveVL` derive for their own stop-string handling — registering
  the child's cancel makes an admin cancel indistinguishable from a natural stop-string hit to
  `streamTokens`' own `parent.Err()` check (M-23's mechanism), so the generation genuinely stops
  but is silently reported as if it had stopped naturally. Caught by the gate test itself
  failing (the cancel worked — the stream ended fast — but no chunk ever said `"cancelled"`),
  not by inspection.
- No cleanup beyond cancelling the context is needed or added: `decoder/model.go`'s
  `ctx.Done()` branch already calls `residentCommitIDs` on any cancellation, and `resBusy`'s
  release is a plain `defer` in the same function — both already run for a client disconnect
  today, and an admin cancel is, from the decoder's point of view, exactly that.
- A pre-existing static guard (`TestStreamSurfaces_allSendUsageBeforeDone`, an AST check that
  every normal stream completion sends its usage chunk before `[DONE]`) caught a real omission:
  the tools-path cancellation exit needed its own `sendUsage` call, matching every other exit —
  the guard's own count of "normal completions" moved from 4 to 5 as a result, which is exactly
  what the guard exists to force a human to notice and justify, not silently absorb.

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

**Status: DONE 2026-09-12** (branch `killswitch-k1-k2-k5`, off `main`). `halt.go`: atomic
`haltInfo{reason, at, trigger}` (an `atomic.Pointer`, lock-free reads on the hot path), `POST
/admin/halt`/`POST /admin/resume`, `GET /health`'s `halted`/`halt_reason`/`halt_at`,
`-halt-file` (250ms poll, edge-triggered so a standing halt doesn't re-log every tick),
`SIGUSR1`/`SIGUSR2`, `-halt-exit-code`. All smoke-tested against a real running binary, not just
compiled: `-halt-file` touch/rm, `SIGUSR1`/`SIGUSR2`, and `-halt-exit-code` (confirmed the
process actually exits with the configured code after quiescence) — `SIGUSR1` against `go run`
looked broken on the first attempt (silently no-op) purely because `go run` wraps the binary in
a child process that doesn't forward the signal; the real built binary receives it correctly.
Gate test: `TestServe_haltUnderLoad`, real local `.gguf`.

**Found while building:**
- **A second, more serious version of K1's own bug** (see K1's status line): `-max-inflight 32`
  admits all 32 requests, but goinfer serializes every generation for one model behind a single
  mutex (`loadedModel.mu`, "the single decode worker"). Only the ONE request currently holding
  that mutex is ever registered in K1's registry; the other 31 are blocked in `sync.Mutex.Lock()`
  — which is not context-aware — and would have run to natural completion one at a time
  regardless of the halt, ignoring it entirely in practice for 31 of 32 requests. Fixed by
  re-checking halted status a second time, INSIDE `tryEnter`/`enter`, immediately after the
  (possibly long) mutex wait and before any generation starts — closing the gap between "admitted
  past the front-of-chain gate" and "actually running." Found by reasoning through this gate's
  own "32 concurrent, halt, every stream cancelled" scenario BEFORE writing the test (unlike K1's
  bug, which the test itself caught) — then confirmed by the test: 1 of 32 requests actually
  reaches `finish_reason: "cancelled"` (the one genuinely streaming when halt landed), the other
  31 get an immediate `503 {"error":"halted"}` without ever running a real generation. Both are
  "the halt worked" outcomes for this architecture; a literal 32/32 `"cancelled"` is not
  achievable with one decode worker per model and isn't what this fix produces or should aim for.
- Measured time-to-quiescence on this box (M1 Pro-class Apple Silicon, CPU backend, Qwen2.5-Coder
  0.5B int8int8): **~10-40ms**, dominated by the one active generation's next per-token ctx
  check, not by anything K2 itself adds. First value for the K8 ledger this doc's own intro
  names as owed.

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

**Where.** `internal/serveapp/main.go:658` admin routes; new listener.

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

**Status: DONE 2026-09-12** (branch `killswitch-k1-k2-k5`, off `main`). `-admin-socket <path>`:
unlink-stale-then-listen, mode 0600, a second `http.Server` on `net.Listen("unix", …)` serving
`/admin/*` (load/unload, K1's generations list/cancel, K2's halt/resume, plus a new `GET
/admin/status` — see below) with no wrapper at all, not even `-allow-admin`'s check: the socket's
own file permissions are the entire gate. When set, the TCP listener does not register `/admin/*`
at all — confirmed as a genuine 404 (`http.ServeMux` on an unregistered pattern), not a
handler-internal 403, so a probe against the TCP listener cannot even confirm the surface exists.
Default paths implemented exactly as scoped: `/run/goinfer/admin.sock` (everywhere but macOS),
`~/Library/Application Support/goinfer/admin.sock` (macOS, via `os.UserHomeDir()`). CLI: `serve
status|ls|cancel <id> [reason]|halt [reason]|resume`, dispatched the same way `pull`/`check`
already are (`os.Args[1]`, before `flag.Parse`), talking to the socket over a plain
`net.Dialer.DialContext("unix", …)` transport. Gate test `TestServe_adminSocket` needs no model
(halt/resume/status/generations never touch one) and passed first try once the wiring below was
right; also re-ran `TestServe_admin`/`TestServe_cancelByID`/`TestServe_haltUnderLoad` afterward to
confirm the refactor below didn't regress K1/K2 — all green.

**Found while building, not assumed:**
- **A real refactor, not just new code.** The admin handlers (`handleAdminLoad`, `…Unload`,
  K1's `…GenerationsList`/`…GenerationCancel`, K2's `…Halt`/`…Resume`) all checked
  `-allow-admin` INTERNALLY (`s.adminEnabled(w)`, first line of each handler body) — fine when
  every registration went through the same TCP path, but wrong for K5: registering those same
  handlers on the socket with that check still inside them would have made `-admin-socket`
  useless without ALSO passing `-allow-admin`, contradicting this item's own "no api-key check"
  line. Moved the check out to a chain-level `requireAdmin` wrapper (`admin.go`, same shape as
  `auth`/`haltGate`/`inf` in `main.go`) applied only at the TCP registration site;
  `registerAdminRoutes` (also `admin.go`) now registers the six-plus-one routes once, called
  twice — TCP with `auth(requireAdmin(...))`, socket with the identity wrapper. Caught a real,
  if narrow, regression this uncovered: `admin_test.go`'s existing "`--allow-admin` off → 403"
  case built its own bare test mux with the handlers registered directly (no wrapper) — it had
  been relying on the since-removed internal check the whole time. Fixed by wiring
  `srv.requireAdmin(...)` into that test's mux too, matching what `main.go` now does for real.
  `cancel_test.go`/`chaos_test.go`/`halt_test.go` were unaffected: they already ran with
  `allowAdmin: true`, so they never exercised the check either way.
- **No admin-scoped status endpoint existed.** The CLI's own `status` verb (named in this
  section's own "Fix" line) had nothing to call: `/health` carries K2's halted fields but is
  deliberately NOT one of "the existing load/unload plus K1/K2's" routes this item says move to
  the socket, and the socket serves `/admin/*` only — registering `/health` there too would have
  been scope creep in the other direction (a `/v1`-shaped route on an admin-only channel). Added
  `GET /admin/status` (halted/halt_reason/halt_trigger/halt_at plus K1's live generation count)
  instead — the minimal thing the doc's own CLI line requires to be true.
- **A real CLI usability trap, found by running the built binary, not by reading the code.**
  `serve halt <reason> -admin-socket <path>` silently used the WRONG (default) socket path and
  failed to connect — Go's `flag` package stops parsing at the first non-flag argument, so a
  reason typed before the flag hides the flag entirely. `serve halt -admin-socket <path>
  <reason>` (flag first) works. Documented in both the flag's own help text and a doc comment;
  not fixed structurally, since fixing it would mean hand-rolling flag parsing this package
  doesn't do anywhere else.
- **Doc citation was stale, fixed during the K1/K2/K5-into-main consolidation:** this section's
  own "Where" line named a `main.go` line pair for the admin routes that, after K1/K2/K5, no
  longer described anything real — the gating logic it pointed at, the inline `adminEnabled`
  check, no longer exists at all (see the refactor above). Now points at
  `internal/serveapp/main.go:658` (`registerAdminRoutes`'s call site, wrapped in the same `auth`
  middleware `/v1` uses). Left as a historical note rather than deleted, since the underlying
  observation — a bare line-number citation into a file under active refactor drifts on the next
  unrelated edit regardless of how carefully it was pinned — still stands and is worth keeping
  visible.

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

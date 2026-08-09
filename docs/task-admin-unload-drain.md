# Design: draining `/admin/models/unload` so it can free device memory safely

**Status:** design only — nothing implemented. Read this before any code. It is the durable target
for the two comments that describe the hazard inline: `handleAdminUnload` (internal/serveapp/admin.go:95–121)
and the reciprocal note at `resident.Close` (metal/model.go:972–976); update both to cite this file
when the fix lands.

**One line:** `handleAdminUnload` must release the model's native memory, but the obvious
`lm.model.Close()` after the registry delete is a use-after-free. The fix is a *drain* — wait out
every in-flight request that touches the model, then close — implemented with a **liveness
`sync.RWMutex` keyed to the `*decoder.Model`**, a **regMu-atomic unpublish+decide**, and a
**detached drain-and-close** with a bounded-wait HTTP response.

**Revision (design reviewed).** Five points were raised in review; all are resolved below, and
resolving them changed two things from the first draft: the liveness lock is now **per-`*decoder.Model`,
not per-entry** (§A, closes a straggler-sibling UAF), and the unload response is a **bounded wait →
200-or-202**, not an unbounded block (§Q1/§Q4).

---

## The problem, as established

`handleAdminUnload` (admin.go:122) deletes the model from the registry and never calls `Close()`.
purego has no ARC and there are no finalizers in `metal/`, `cuda/`, or `gpu/`, so GC reclaims the Go
wrappers and never the native device allocations. Measured on Metal: **+451 MB per unload/reload
cycle** of a ~450 MB model, memory does not return. **Backend-agnostic** — CUDA VRAM and WebGPU
buffers leak identically — and on an 8 GB card an unload-then-reload of a large model fails to
allocate because the card cannot hold two copies.

`Model.Close` is complete and idempotent on all three backends (Metal N-11; CUDA `reqCh==nil` guard;
WebGPU C-26 `closed` flag). So *closing is safe.* **Closing at the wrong time is not.**

### Why the one-liner is a use-after-free

Every handler has the shape `pick() → work → enter()`. `pick` (openai.go:213) returns the
`*loadedModel` under `regMu`; `enter` (openai.go:166) is where `lm.mu` is finally taken. Between them
the **preamble** touches `lm.model`/`lm.tk` with **no lock** — in `handleChat`: `pick` (openai.go:464)
→ `promptTooLargeForContext` → `chatPrompt` (a full BPE) → `prepare` → `enter` (openai.go:489).
`handleAdminUnload`'s `lm.mu.TryLock()` sees an idle model and would grant the unload while that
request is mid-tokenize against weights about to be freed; the request then `drive()`s on a torn-down
backend. On CUDA that is a driver SIGSEGV that kills the server; on CPU it is a quieter read of reused
memory. Seven handlers share the shape:

| handler | `pick` |
|---|---|
| `handleChat` | openai.go:464 |
| `handleCompletions` | openai.go:542 |
| `handleMessages` | anthropic.go:406 |
| `handleCountTokens` | anthropic.go:535 |
| `handleChatTools` | tools.go:19 |
| `handleResponses` | responses.go:86 |
| `serveVisionChat` | vision_serve.go:119 |

The in-generation case is already safe (past `enter`, the request holds `lm.mu`, so a mid-stream
unload 409s — verified on Metal). The hole is the preamble.

---

## The design

Two locks, one detached worker, one guarded decision.

- **`mu` (existing, per entry):** serializes generation, held `enter`→`exit`. Unchanged.
- **liveness (new): a `sync.RWMutex` keyed to the `*decoder.Model`,** shared by every registry entry
  backed by that model (§A explains why per-model, not per-entry). A server side-table
  `map[*decoder.Model]*modelLiveness` guarded by `regMu`, where `modelLiveness{ rw sync.RWMutex; refs int }`;
  `refs` counts registry entries backed by the model. Every request path takes `rw.RLock()` at model
  resolution and holds it until the handler returns — spanning the preamble *and* the generation,
  which is what the existing `mu` misses.

### §A — why liveness is keyed to the model, not the entry (straggler UAF)

A base and its compute-time adapters share one `*decoder.Model` (main.go:537). With a *per-entry*
lock and detached drains, this sequence UAFs: unload adapter A (not last owner, so it does not close)
while a request of A is still in flight; later unload base B (now last owner) drains only *B's* lock
and closes the model — while A's straggler request is still reading it. The registry scan says "no
entry routes to the model," which is true, yet a deleted entry can still have in-flight work on the
model. A per-**model** lock closes this: A's straggler holds the model's `RLock`, so B's close (the
model's `Lock`) waits for it. Consequence that also simplifies things: a **non-last** unload does not
drain at all — it just unpublishes; only the **last** unload drains the model and closes.

### Phase 1 — unpublish + decide, atomically under `regMu` (answers #1 and #2)

Under a single `regMu.Lock`:
1. `delete(s.models, name)` — the entry is now unroutable; no new `pick` can find it.
2. `ml := s.liveness[lm.model]; ml.refs--` — **decrement after delete.**
3. decide: `last := ml.refs == 0`. If `last`, remove `s.liveness[lm.model]` and hand the model to the
   detached closer (Phase 2). Otherwise the model stays; nothing is freed.

Release `regMu`. The **close-decision is made under `regMu`; the `Close` itself runs outside it**
(Phase 2) — holding `regMu` across a multi-second drain would freeze all request routing serverwide.

**#1 — a concurrent load cannot race the decision.** A load that would share an existing model
(today only startup `--adapter`; any future runtime equivalent) must look up the base entry under
`regMu` to obtain its `*decoder.Model`. Phase 1 deletes the entry and decrements `refs` under the same
`regMu`, so the load's lookup is serialized against it: if the lookup precedes Phase 1 it bumped
`refs` first, so Phase 1 sees `refs>0` and does **not** close; if it follows Phase 1 the entry is
gone, so the share fails to resolve. There is no interleaving in which a new sharer is published
against a model already decided-to-close. **Invariant for the future:** any publish that shares an
existing `*decoder.Model` must do its lookup-and-`refs++` inside one `regMu` section — the same rule
that makes today's fresh-model loads (which allocate a new model, `refs==1`) trivially safe.

**#2 — delete/decrement strictly before the last-owner test, and why.** Two concurrent unloads of
entries sharing one model must not *both* decline to close — that orphans the model with no entry left
to retry, and unlike a double-close (idempotent, harmless) a double-decline is unrecoverable. With
decrement-before-test under `regMu`, the two Phase-1 sections serialize: the first decrements to
`refs=1` and declines; the second decrements to `refs=0` and closes. Exactly one closes. Reverse it —
test the count while still counting yourself — and both can read `refs≥1` and both decline. Write this
into the code comment; the ordering looks arbitrary and is not. (Stated in refcount terms; the
equivalent scan-based phrasing is "`delete` self from the registry *before* scanning it for
siblings," so the second unloader necessarily sees an empty field.)

### Phase 2 — detached drain-and-close (answers #3)

If Phase 1 decided `last`, a **server-lifetime goroutine** — not tied to `r.Context()` — takes the
model's `rw.Lock()`. Because the model is now absent from both the registry and the liveness table, no
new `RLock` can appear, so the write lock is acquired once the finite set of in-flight holders (across
all now-deleted sibling entries) drains — **guaranteed to terminate**. Then it calls `lm.model.Close()`
and signals done.

**The drain must outlive the request.** The entry is already unpublished, so nothing will ever retry
it; if an admin client disconnects (or a `curl` times out) mid-drain and that aborted the close, the
model would be orphaned for the process lifetime. So the goroutine owns the close and runs to
completion regardless of the handler; the handler only *observes* it (below). `sync.RWMutex.Lock` is
not context-cancellable, which is correct here — we do **not** want cancellation.

### The HTTP response — bounded wait, 200 or 202 (answers #4, supersedes the draft's pure block)

The draft said "block until freed, 200." Review's steelman of 202 is right on one point: because the
drain now waits for the **whole queue**, not just the active generation (see §Q5), a pure block is an
admin endpoint that can hang for a long time with no recourse. So:

- Phase 1 is synchronous and instant → the model is unroutable the moment the call is accepted.
- The handler waits up to a bounded **`T`** (a few seconds) on the closer's done signal:
  - drained within `T` → **200** `{status:"unloaded", freed:true}`.
  - not yet → **202** `{status:"unloading", freed:false}` and the detached closer keeps going.

This keeps the draft's good contract — **`freed:true` means the memory is actually released** — for
the common fast case, which is what makes unload→reload safe on an 8 GB card. It adds recourse for the
pathological busy-queue case instead of hanging. The residual: after a 202 an operator who reloads
before `freed:true` can still hit the two-copies OOM — but that is now an observable, documented
contract ("reload after `freed:true`"), not a silent hang and not a crash. Pure block does not
actually beat this: it only moves the same wait to inside the one call, at the cost of no recourse.
`T` can be a flag later; not needed for v1.

### `withModel` is the only route to a request-path model (answers #5)

A returned release func relies on every handler deferring it on every exit path (preamble 400s,
panic, disconnect); a single miss leaks a **reader**, which hangs the model's `Lock` forever — worse
than the memory leak. And "handlers remember to defer" is not enforcement. So:

**Eliminate `pick` as a callable function.** Fold its lookup into `withModel(w, name, fn func(*loadedModel))`,
which is the *sole* code that turns a request's model name into a `*loadedModel`: it does the
`regMu`+`RLock` acquisition, `defer`s the `RUnlock` once (covering panic / early return / disconnect),
resolves not-found to the standard 404, and calls `fn`. With no `pick` symbol in the package, the
author of the eighth handler *cannot* call it — the safety is a property of the code, not of memory.
The non-request lookups that legitimately need no liveness (`modelByName`, `servedNames`, `handleModels`)
stay separate and visibly distinct. A lint test asserts no handler file reads `s.models` directly.
Each of the seven handlers is restructured to `return s.withModel(w, req.Model, func(lm){ ... })`.

---

## The seven original questions (updated)

**Q1 — SSE streams.** Bounded wait, not 409 and not unbounded block. 409-on-any-reader makes unload
impossible on a busy server (never a zero-holder instant); unbounded block hangs with no recourse
(§Q4). Bounded wait → 200 if drained within `T`, else 202 with the drain detached (§Phase 2). The
model is unroutable immediately either way.

**Q2 — release safety.** A `withModel` wrapper that is the *only* route (pick removed), so no exit
path and no future handler can skip the release. Enumerated seven handlers above.

**Q3 — where the ownership check fits.** Two orthogonal checks, now unified in one structure. The
`modelLiveness.refs` count is the **structural** check ("does any registry entry still route to this
model") and replaces the standalone scan; the `modelLiveness.rw` lock is the **temporal** check ("is
any in-flight request still touching it," across *all* sibling entries — §A). Close requires both:
`refs==0` **and** the drain complete. Neither subsumes the other — an idle-but-referenced model
(`refs>0`, `rw` free) must not close, and a de-referenced-but-still-draining model (`refs==0`, `rw`
held) must not close. (The stashed last-owner-scan work maps onto maintaining `refs`; an O(1)
decrement replaces the O(n) scan, but the scan is an acceptable equivalent if refcount wiring at every
load site is judged too invasive.)

**Q4 — 202 re-argued.** Adopted as the >`T` branch (§the HTTP response). Recommendation is the
bounded-wait hybrid, which survives the strongest 202 argument (recourse for the long-queue case)
while keeping `freed:true`⇒safe-reload for the common case.

**Q5 — queued requests.** Premise corrected: admission is non-blocking *before* generation
(`limitInflight`, helpers.go:78, is a non-blocking semaphore → 503; `tryEnter`, openai.go:153, does a
non-blocking queue send → 429). The only post-`pick` block is `mu.Lock`. So a "queued" request has
already resolved its model and holds the liveness `RLock`; it is drained like any other and **served
to completion** on the still-valid model (the close waits for it), not errored. Requests arriving
after unpublish get 404.

**Q6 — shutdown.** Leave graceful shutdown unchanged; do not use the drain there. Process exit
reclaims device memory via the OS; the leak only matters for a long-lived unload/reload process.
Wiring the drain into shutdown adds executor-teardown/checkpoint risk for no benefit, and a detached
unload-drain in flight at shutdown simply dies with the process (harmless). `Model.Close`'s startup
caller (main.go:707, `.giw` `--quant` cleanup, pre-executor) stays untouched.

**Q7 — blast radius.** The lock/drain/refcount logic is pure Go in `internal/serveapp`, backend-
independent → fully unit-testable on the Mac with the CPU backend (no device), including the §Q4
regression test. Metal: real reclaim re-verifiable here (+451 MB→~0). WebGPU: `gpu-darwin` CI / a Mac
WebGPU device. **CUDA cannot be verified here (no hardware), and its failure mode — `drive()` on a
torn-down context, a driver SIGSEGV that kills the server — is the fatal one the drain exists to
prevent. Confirming it on real CUDA requires the Linux box.** That is the residual risk to call out.

---

## Regression test (the honesty hook)

Park a request **inside the preamble window** (holding the model `RLock`, before `enter`) and assert
unload cannot free the model out from under it. Use the existing **`goinfer_testhooks`** build tag
(CI already runs `go test -race -tags goinfer_testhooks ./...`; pattern in `gpu/testhooks_gen.go`):
production build defines `func preamblePark() {}` (empty, inlined, not a branch); the test build
defines `var preamblePark = func(){}`, settable by a test. `withModel` calls `preamblePark()`
immediately after taking the `RLock`. The test blocks the hook, fires a request (which parks holding
the lock), then fires an unload and asserts it returns 202/`freed:false` and that **no `Close` ran**
while parked; releasing the request then lets the drain complete.

**Honest against reordering:** the invariant is "a request parked in the preamble holds the liveness
lock ⇒ the drain cannot complete." The hook lives at the acquisition site, so if someone moves the
`RLock` down to `enter` (reopening the window), the parked request no longer holds it, the drain
completes and closes while it is parked, and the test fails — which is exactly the reintroduced bug.

---

## Compatibility / CHANGELOG

Mid-stream unload changes on an observable public endpoint: today `409 busy; retry`, after the fix an
immediate unpublish + `200 freed:true` (fast) or `202 freed:false` (slow). This belongs in the
`[Unreleased]` CHANGELOG when the code lands, with the `freed` field documented.

**`?wait=false`?** Not worth preserving the old `409` — it was safe only because nothing was ever
freed, and it forced callers into a retry loop that could never succeed on a busy model. A scripted
non-blocking caller is better served by `?wait=false` meaning **"skip the bounded wait, return 202
now"** (still unpublishes and still drains-and-closes detached), then polling `freed` via
`GET /v1/models`. That gives scripts a clean non-blocking path *and* real reclamation, which the old
`409` never did. So: add `?wait=false` = immediate 202; drop the `409-busy` semantics.

---

## Scope

- Do **not** touch `Model.Close`'s startup caller (main.go:707) — pre-executor, no drain involved.
- Do **not** wire the drain into shutdown (§Q6).
- `Model.Close` itself is unchanged — already complete and idempotent; only the *timing* (drain-gated,
  `refs==0`-gated, detached) is new.

## Recommendation summary

Per-`*decoder.Model` liveness `sync.RWMutex` + `refs` in a `regMu`-guarded side-table; `RLock` taken
only through a `withModel` wrapper that is the sole route to a request-path model (pick removed);
unload = `regMu`-atomic {delete, `refs--`, decide} → detached drain (`rw.Lock`) + last-owner
`Close` → bounded-wait 200/202 (`freed` field), `?wait=false` for immediate 202. Decide under `regMu`,
close outside it; delete/decrement strictly before the last-owner test; drain detached from the
request. Keep `mu` for generation; leave shutdown and the startup `.giw` path alone. CUDA's fatal path
is the one thing the Mac cannot verify.

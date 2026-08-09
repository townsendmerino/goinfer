# Design: draining `/admin/models/unload` so it can free device memory safely

**Status:** design only — nothing implemented. This note exists to be read before any code, and is
the durable target for the two code comments that already describe the hazard inline:
`handleAdminUnload` (internal/serveapp/admin.go:95–121) and the reciprocal note at `resident.Close`
(metal/model.go:972–976). Update both to cite this file when the fix lands.

**One line:** `handleAdminUnload` must release the model's native memory, but the obvious
`lm.model.Close()` after the registry delete is a use-after-free. The fix is a *drain* — wait out
every in-flight request that holds the model pointer, then close — implemented with a per-model
liveness `sync.RWMutex` and a two-phase unload (unpublish, then drain-and-close).

---

## The problem, as established

`handleAdminUnload` (admin.go:122) deletes the model from the registry and never calls `Close()`.
purego has no ARC and there are no finalizers anywhere in `metal/`, `cuda/`, or `gpu/`, so GC
reclaims the Go wrappers and never the native device allocations. Measured on Metal:
**+451 MB per unload/reload cycle** of a ~450 MB model, and memory does not return. It is
**backend-agnostic** — CUDA VRAM and WebGPU buffers leak identically — and on an 8 GB card an
unload-then-reload of a large model fails to allocate, because the card cannot hold two copies.

`Model.Close` is complete and idempotent on all three backends (verified: Metal N-11 `ReleaseAll`
empties the ledger; CUDA `reqCh == nil` guard; WebGPU C-26 `closed` flag + register-at-creation
release list). So *closing is safe*. **Closing at the wrong time is not.**

### Why the one-liner is a use-after-free

Every request handler has the shape `pick() → work → enter()`. `pick` (openai.go:213) returns the
`*loadedModel` under `regMu`; `enter` (openai.go:166) is where `lm.mu` is finally taken. Between them
the **preamble** touches `lm.model` and `lm.tk` with **no lock held** — in `handleChat`, that is
`pick` (openai.go:464) → `promptTooLargeForContext` → `chatPrompt` (a full BPE over the prompt) →
`prepare` → `enter` (openai.go:489). `handleAdminUnload`'s current `lm.mu.TryLock()` sees an idle
model (the preamble holds no mutex) and would grant the unload while that request is mid-tokenize
against weights about to be freed. The request then `enter`s and `drive()`s on a torn-down backend.

The seven handlers sharing the shape:

| handler | file:line of `pick` |
|---|---|
| `handleChat` | openai.go:464 |
| `handleCompletions` | openai.go:542 |
| `handleMessages` (Anthropic) | anthropic.go:406 |
| `handleCountTokens` (Anthropic) | anthropic.go:535 |
| `handleChatTools` | tools.go:19 |
| `handleResponses` | responses.go:86 |
| `serveVisionChat` | vision_serve.go:119 |

The in-generation case is already safe: once a request is past `enter`, it holds `lm.mu`, so a
mid-stream unload gets 409 (verified on Metal with a 400-token stream). The hole is the preamble.

---

## The design

Two mechanisms, both needed (see Q3 for why neither subsumes the other):

1. **Liveness lock** — a second `sync.RWMutex` on `loadedModel` (call it `live`), *distinct from the
   existing generation mutex `mu`*. It answers "is any request using this entry *right now*."
   - The request path takes `live.RLock()` at `pick` and holds it until the handler returns —
     spanning the preamble *and* the generation, which is what closes the window (today's `mu` only
     covers `enter`→`exit`).
   - `pick` acquires `live.RLock()` **while still holding `regMu`** (atomic with the registry
     lookup), then releases `regMu`. This is the ordering hinge (proof below).

2. **Two-phase unload** in `handleAdminUnload`:
   - **Phase 1 — unpublish (synchronous, under `regMu.Lock`):** `delete(s.models, name)`. After this,
     no new `pick` can find the model. This is the instant the model is "unloaded" from the API's
     point of view; other models keep serving.
   - **Phase 2 — drain + close:** acquire `live.Lock()` (write). Because unpublish already happened,
     **no new `RLock` holder can appear**, so existing holders decrease monotonically to zero and the
     write lock is *guaranteed to be acquired* — this is the drain. Then, **iff this is the last
     registry entry backed by the `*decoder.Model`** (Q3), call `lm.model.Close()`. Respond 200.

**Ordering-correctness (why the drain terminates and never races the delete).** `pick` takes
`live.RLock` under `regMu.RLock`; unload's `delete` runs under `regMu.Lock`; the two are mutually
exclusive. So for any request and any unload, exactly one of:
- unload's `delete` wins → the later `pick` doesn't find the model → no `RLock` taken → nothing to
  drain from that request; or
- `pick` wins → it holds `live.RLock` *before* unload deletes → unload's later `live.Lock` waits for
  it.

Unload releases `regMu` **before** taking `live.Lock` (it never holds both), so there is no lock-order
cycle. After `delete`, no code path can find the model to `RLock` it, so `live.Lock` cannot be
starved by new readers and completes once the finite set of in-flight holders drains. Go's RWMutex
writer-priority is irrelevant here precisely because no new readers arrive.

This **replaces** unload's `mu.TryLock` with `live.Lock`; `mu` stays as the generation serializer,
untouched.

---

## Questions the design must answer

### Q1 — SSE streams: block, don't 409, don't time-out-into-leak

A long generation holds `live.RLock` for its whole duration (minutes for a max-length stream).
Three options for unload when a reader is present:

- **409 immediately (the starting shape's `TryLock`).** Rejected. On a busy server there may never be
  a zero-holder instant, which makes unload *effectively impossible* — the operator can never reclaim
  the memory. This is the failure the prompt flagged, and it is real.
- **Time out, then give up.** Rejected. Leaves the model unpublished but its memory unreleased — a
  leak with extra steps, and now the model is gone from the API too.
- **Block on the drain, return 200 when the memory is actually freed. ← recommended.** Terminates
  because unpublish (Phase 1) stops new readers, so the drain is bounded by the in-flight set, which
  is itself bounded by `max_tokens` and by client disconnect (`r.Context()` cancel → `drive` stops →
  the deferred release fires). The model is unroutable the instant Phase 1 runs, so the rest of the
  server is unaffected while unload waits. **"200 means freed"** is the contract that makes
  unload→reload safe on a small card.

Note this **changes the mid-stream behavior** from today's `409 busy; retry` to *block until the
stream ends, then 200*. That is the point: `409` was only ever safe because we never actually freed
anything. A second unload of the same name while the first drains gets 404 (already unpublished),
which is acceptable idempotency.

Rejected alternative worth recording: **202 + background drain-and-close.** Returns immediately, closes
in a goroutine. Rejected because "unloaded" would then *precede* the memory actually being freed, so
an immediate reload on an 8 GB card hits the two-copies OOM this whole fix exists to prevent.
Synchronous drain is what makes the response mean something.

### Q2 — a release func is not enough; scope the lock in one place

Seven handlers share the shape today, and the eighth will be written by someone who never read this
note. A bare `release := pick(...)` relies on every handler deferring it on **every** exit route —
the preamble's early-return 400s, a panic, a client disconnect mid-preamble. Miss one and you leak a
**reader**, which is strictly worse than leaking memory: a permanently-held `RLock` means unload's
`live.Lock` never completes, so unload hangs forever.

Recommendation: **do not hand back a bare func.** Put acquisition and release in a **single
choke-point wrapper** — `s.withModel(w, name, func(lm *loadedModel) { ... })` — that does the
`regMu`+`RLock` dance, `defer`s the `RUnlock` once, and runs the handler body. One `defer` in one
place covers panic, early return, and disconnect for all seven handlers and every future one; a
handler physically cannot skip the release because it does not own it. The cost is restructuring each
handler body into a closure — mechanical, and it also centralizes the `modelNotFound` path.

If the wrapper is judged too invasive, the fallback is `pick` returning `(lm, release)` with a
`goinfer_testhooks` lint test asserting every `pick(` call site is immediately followed by
`defer` — but the wrapper is the design that is *hard to get wrong*, which is the requirement.

### Q3 — the last-owner scan is orthogonal and still required

The stashed last-owner-refcount work (a scan of `s.models` for another entry sharing the same
`*decoder.Model`) and the liveness lock answer **different** questions:

- **Liveness lock** — is any *request* using **this loadedModel entry** right now? (temporal)
- **Last-owner scan** — does any *other registry entry* (a base and its compute-time adapters share
  one `*decoder.Model`; main.go:537) still reference this model? (structural)

Neither subsumes the other. An adapter can be perfectly idle — its own `live` fully drained — yet
still share `base.model`; closing the base on the base's unload would free weights the adapter's
*next* request reads. And a base with no adapters still needs the liveness drain against its own
in-flight requests. So **`Close` fires only when both hold: this entry's `live` is drained *and* no
sibling shares the `*decoder.Model`.** The per-entry liveness lock deliberately does not span
siblings — the scan is what bridges entries. `Model.Close` already releases every adapter registered
on the base, so the last-owner close frees the whole family at once.

### Q4 — regression test via the `goinfer_testhooks` seam, honest against reordering

The test must park a request **inside the preamble window** (holding `live.RLock`, before `enter`)
and assert unload does not free the model out from under it. The park point must not be a test-only
branch in production code.

Use the existing **`goinfer_testhooks`** build tag (CI already runs `go test -race -tags
goinfer_testhooks ./...`; the pattern is `gpu/testhooks_gen.go`). Add, in `internal/serveapp`:
- production build (`//go:build !goinfer_testhooks`): `func preamblePark() {}` — empty, inlined to
  nothing, not a branch;
- test build (`//go:build goinfer_testhooks`): `var preamblePark = func() {}`, settable by a test.

The shared `withModel` wrapper calls `preamblePark()` **immediately after taking `live.RLock`**, i.e.
inside the window. The test sets it to block on a channel, fires a request (which parks holding the
lock), then fires an unload and asserts it **blocks / does not complete** until the parked request is
released — and that no `Close` ran meanwhile.

Honesty against reordering: the invariant under test is "a request parked in the preamble holds the
liveness lock ⇒ unload cannot drain." If someone later moves the `RLock` acquisition from `pick`/the
wrapper down to `enter` (reopening the window), the parked request no longer holds the lock, unload
drains and closes while it is parked, and the test **fails** — which is exactly the reintroduced bug.
The hook lives at the acquisition site, so it tracks the lock, not a line number.

### Q5 — queued requests have already picked; serve them, don't fail them

The prompt's premise ("a request waiting on `--max-queue` has not picked yet") is **incorrect** —
worth stating plainly so no one designs around a false model. Admission is entirely non-blocking
*before* generation: `limitInflight` (helpers.go:78) is a non-blocking semaphore (503 if full),
and `tryEnter` (openai.go:153) does a non-blocking channel send for the queue slot (429 if full).
The only place a request **blocks** is `mu.Lock` inside `tryEnter` — which is **after** `pick` and
the preamble. So a "queued" request is one holding an inflight slot + a queue slot + (under this
design) `live.RLock`, blocked on `mu.Lock`.

Consequence: such a request is a live holder and is **drained like any other** — unload's `live.Lock`
waits for it. It runs its generation to completion on the still-valid model (`Close` cannot run until
the drain completes), then releases. So it is **served normally, not errored**. No generic failure,
no special "unloaded" error path needed. Requests arriving *after* unpublish simply 404. This is
simpler than injecting an unload-aware error and is correct.

### Q6 — do not use the drain at shutdown

Recommendation: **leave graceful shutdown unchanged.** It never calls `Close` today, which is
harmless: at process exit the OS reclaims all device memory regardless. The leak only matters for a
**long-lived** process doing repeated unload/reload — the admin path, not shutdown. Wiring the drain
into shutdown would add executor-teardown ordering and checkpoint-interaction risk for zero benefit
(the process is dying). Keep it out. This also keeps `Model.Close`'s startup caller (main.go:707,
`.giw` `--quant` mismatch cleanup, before the executor starts) untouched, per scope.

### Q7 — blast radius: what can be verified here, and what cannot

- **Lock/drain semantics** are pure Go in `internal/serveapp`, independent of any backend. Fully
  unit-testable on the Mac with the CPU backend (no device): the Q4 park-and-assert test needs no GPU.
  This covers the *correctness of the drain itself*, which is the part most likely to be wrong.
- **Metal** — real memory reclaim is verifiable on this Mac: the existing repro (+451 MB/cycle → ~0)
  re-run after the fix.
- **WebGPU** — `gpu-darwin` CI builds and tests it; a Mac WebGPU device can exercise the reclaim.
- **CUDA — cannot be verified here (no hardware).** And the CUDA failure mode is the fatal one: the
  one-liner's UAF surfaces as `drive()` on a torn-down context — a driver SIGSEGV that kills the
  server. The drain is designed to make that path unreachable, but **confirming it on real CUDA
  requires the Linux box.** This is the residual risk to call out explicitly: everything except the
  CUDA fatal-path is verifiable on the Mac; the CUDA fatal-path is exactly what we cannot test here.

---

## Scope

- Do **not** touch `Model.Close`'s startup caller (main.go:707). It runs before the executor starts,
  so `stopExec` is a no-op and no drain is involved; it stays working unchanged.
- Do **not** wire the drain into graceful shutdown (Q6).
- `Model.Close` itself is not modified — it is already complete and idempotent. Only the *timing* of
  the call (drain-gated, last-owner-gated) is new.

## Recommendation summary

Liveness `sync.RWMutex` per `loadedModel`, acquired at `pick` under `regMu` and released by a single
`withModel` wrapper; unload = unpublish-then-`live.Lock`-drain-then-(last-owner)-`Close`, blocking
until freed (200 = freed). Keep the last-owner scan; keep `mu` for generation serialization; leave
shutdown and the startup `.giw` path alone. Regression test parks in the window via the
`goinfer_testhooks` seam. CUDA's fatal path is the one thing the Mac cannot verify.

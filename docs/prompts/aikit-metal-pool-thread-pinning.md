# Task (aikit + goinfer): pin the OS thread around aikit's Metal autorelease pools

> **For:** Claude Code, in `~/tmcode/aikit` (kernel work lands there) with sibling
> `~/tmcode/goinfer` for verification. Written 2026-08-26. **Box: mac** — this needs a real Metal
> device; CI cannot run it (`macos-latest`'s paravirtual Metal layer SIGSEGVs inside
> `objc_msgSend`, which is why `.github/workflows/ci.yml`'s `metal-darwin` job is build + vet + one
> device-free test). Queue entry: **G22** in `docs/queue-engineering.md`.
>
> **This is not a bug hunt. The mechanism is already known and written down** — see below. The
> deliverable is the fix in the right place, plus a gate that can actually fail.

## The mechanism — already documented in goinfer, do not re-derive it

`goinfer/metal/model.go`, `resident.Forward`:

> *"Pin to one OS thread for the whole call: the NSAutoreleasePool (begin/end) is per-OS-thread, and
> Go can migrate goroutines mid-call — draining a pool on a different thread than it was pushed is
> UB (intermittent SIGSEGV). Same discipline the CUDA backend's LockOSThread executor uses."*

The observed crash is exactly that drain:

```
SIGSEGV: segmentation violation
PC=0x… addr=0x10   signal arrived during cgo execution
github.com/ebitengine/purego/objc.ID.Send(...)
github.com/townsendmerino/aikit/gpu.(*Encoder).End      ← e.pool.Send(selDrain)
github.com/townsendmerino/goinfer/metal.TestSAQVFusion_correctnessAndThroughput
```

## Why it only ever bites tests — the asymmetry that IS the bug

- **goinfer's production Metal paths pin.** `resident.Forward` and `BuildResident` both
  `runtime.LockOSThread` / `defer runtime.UnlockOSThread`. They are safe, and the shipped product
  has never shown this crash.
- **aikit's `Encoder` API does not.** It creates its pool inline (three sites in `gpu/metal.go`)
  and drains it in `End()`. `LockOSThread` appears **nowhere in aikit's production code** — only in
  aikit's own tests. Any caller that does not pin is on UB.
- **97 of goinfer's 108 `metal/*_test.go` files do not pin.**

That asymmetry explains every observed property, and any proposed fix must keep explaining them:
the crash is **probabilistic** (it needs a scheduler migration inside the window), its **site
migrates** between runs, it **survives isolation** (`TestSAQVFusion` alone: 3/3 in <0.5 s), and it
is **absent from the shipped product**.

## Observed rates (2026-08-26, M1 Pro, goinfer `65441a3`)

| invocation | result |
|---|---|
| `go test -tags metal ./metal/` | 4/4 pass |
| `go test -tags metal -short ./metal/` (what the gate runs) | 1 fail / 3 |
| `go run ./cmd/gate gpu`, clean tree, quiet machine, sequential | PASS |

`docs/task-gate-runner.md` records ~50%, and once 6-of-7, with the operator's read that the rate
tracks cumulative GPU dispatch volume — consistent with a migration-window race. **Rate is
load-dependent, so any "it stopped crashing" claim needs N large enough to beat it (see Gates).**

## The change

**Pin where the pool is created, unpin after it drains.** The invariant belongs to the pool, not to
every caller — the same shape `aikit/gpu/cuda.go` already documents for CUDA ("gocudrv's Context
OWNS a dedicated `runtime.LockOSThread`'d executor goroutine and funnels every driver call through
it").

Three sites in `gpu/metal.go` create an `NSAutoreleasePool` (`Run1D`, `Run2D`, and the `Encoder`
path around line 649). Each must hold the OS thread from creation until after `selDrain`.

**`runtime.LockOSThread` nests** — it is a counter, and the thread unlocks when it reaches zero — so
adding it inside aikit is safe alongside goinfer's existing outer pins in `Forward` /
`BuildResident`. Verify that rather than trusting this sentence.

**The hazard to think about before writing code:** an `Encoder`'s pool is created in one call and
drained in a *later* one (`End()`). A pin taken inside `End()` alone is useless — the pool was
pushed on whatever thread the earlier call ran on. So either the lock is held across the encoder's
whole lifetime (and then a caller that hands an `Encoder` to another goroutine is broken and must
be rejected or documented), or the encoder path is restructured so create-and-drain happen under
one pin. **State which you chose and why in the commit.**

## What NOT to do

- **Do not pin per test.** Fixing the 97 files leaves the invariant unenforced for the next caller,
  and the crash site migrates precisely because *any* unpinned caller can be the victim. It also
  makes the gate green while the API stays unsafe.
- **Do not "fix" it by serializing tests or adding retries.** That hides a UB race behind timing.
- **Do not change goinfer's existing pins** to compensate; they are correct and are the evidence
  that this diagnosis is right.

## Gates

1. **`-short` metal suite, N ≥ 20 consecutive runs, zero `fault 0x10` deaths.** N is set by the
   observed rate: at ~30% a handful of passes proves nothing. Record N and the machine load —
   `docs/benchmarks.md`'s methodology now requires machine state beside any timing-sensitive claim,
   and this is one.
2. **`go run ./cmd/gate gpu` PASS on a clean tree**, run sequentially with nothing else on the box.
   A `+dirty` tree is a PROVENANCE failure and a concurrent build poisons the run — both were
   demonstrated on 2026-08-26 by the author of this brief, twice.
3. **Zero golden churn and no numeric change.** Pinning changes *which thread* runs the work, never
   what it computes. If any golden moves, stop: something other than thread affinity changed.
4. **A mutation check**: revert the pin, show the crash returns within N runs. A fix for a
   probabilistic failure that cannot be shown to fail without it is not yet evidence.

## Coordination (this is a cross-module change)

aikit is **not** in goinfer's `go.work`, so goinfer builds against the module cache. Sequence:
aikit change → aikit version bump → goinfer `require` bump → re-run the gates above in goinfer.
Per `docs/…/goinfer-gpu-require-bump-on-tag`, the `gpu` module's `goinfer` require must move in the
same release commit if this rides a tag.

## Why this is worth doing

§C1-M is a **release gate that a human must run on the only machine that can run it**, and it
currently fails roughly a third to a half of the time for reasons unrelated to the code under test.
That is not a gate; it is a coin flip someone re-flips until it lands — and the habit of re-flipping
is exactly how a real failure gets waved through some later evening.

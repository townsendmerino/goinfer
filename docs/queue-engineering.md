# Engineering-discipline queue

Gates, lints, censuses, tooling, process rules, and the audit sweeps. Anything whose success criterion is that **a check can be trusted** — that it runs, that it can fail, that its green means what it says. If the question is *would we find out*, it belongs here.

> **One of four queues.** The work list is split by *success criterion*, not by component:
> [performance](queue-performance.md) · [correctness](queue-correctness.md) ·
> [engineering](queue-engineering.md) · [release](queue-release.md).
> [`QUEUE.md`](QUEUE.md) is the index over all four and holds the cross-cutting sweeps.
>
> **Task docs are NOT queues.** `docs/task-*.md` are *design records* — why a thing is built as it
> is — and they are cited from 88 code comments. A queue entry cannot carry that, so the task docs
> stay put and the queues hold only the open work.
>
> Entries keep the section they were filed under (`In flight`, `Queued`, …) and their original IDs,
> so a citation to an ID still finds it.


> **Thirteen closed entries were archived 2026-08-31** to
> [`docs/completed/queue-engineering.md`](completed/queue-engineering.md) — G21, G19, G18, G14,
> A12, B0, B5, B3, E9, F1, G13, R1 and R2. What is below is the open work.
>
> **Four entries that READ as closed were deliberately left here**: `B8` (audited, "the gates are
> clean, the PROSE is not"), `B7` (swept, actions not all discharged), `F2` ("two lack an anchor")
> and `F3` ("status unconfirmed"). Classifying by header keyword would have archived all four and
> buried the residual work — the stale-header defect, caused by the cleanup instead of found by it.

## In flight

**G22 · The Metal `fault 0x10` flake is an UNPINNED AUTORELEASE POOL in aikit's Encoder — fix it
where the pool lives** — `aikit`+`mac`, **QUEUED, diagnosed 2026-08-26.**

**Not a shipping bug, and not a mystery — the mechanism is already written down in this repo.**
`metal/model.go`'s `resident.Forward` says it outright: *"the NSAutoreleasePool (begin/end) is
per-OS-thread, and Go can migrate goroutines mid-call — draining a pool on a different thread than
it was pushed is UB (intermittent SIGSEGV). Same discipline the CUDA backend's LockOSThread executor
uses."* The crash is exactly that drain: `aikit/gpu.(*Encoder).End` → `e.pool.Send(selDrain)`.

**Why it only ever bites tests.** goinfer's production Metal paths PIN — `resident.Forward` and
`BuildResident` both `runtime.LockOSThread`. aikit's `Encoder` API does not: it creates its pool
inline (three sites in `metal.go`) and drains it in `End()`, with `LockOSThread` appearing nowhere
in aikit's production code — only in its own tests. Any caller that does not pin is on UB, and
**97 of goinfer's 108 metal test files do not pin.** That accounts for every observed property: the
crash is probabilistic (it needs a scheduler migration in the window), its site MIGRATES between
runs, it survives isolation (`TestSAQVFusion` alone: 3/3 pass in <0.5 s), and it never appears in
the shipped product.

**Observed rate here 2026-08-26:** `-short` full suite 1 fail / 3 runs; without `-short` 4/4 pass;
a clean sequential gate run passed. `docs/task-gate-runner.md` records ~50% and once 6-of-7, with
the operator's read that the rate tracks cumulative GPU dispatch volume — consistent with a
migration-window race.

**The fix belongs in aikit, not in 97 test files.** Whoever creates the pool should pin the thread
and unpin after the drain, making the API safe regardless of caller — the same discipline
`aikit/gpu/cuda.go` already documents for CUDA's locked executor. Pinning per-test is the wrong
shape: it leaves the invariant unenforced for the next caller, and the crash site migrates precisely
because any unpinned caller can be the victim.

**Coordination:** aikit is not in goinfer's `go.work`, so this needs an aikit change, a version bump
and a goinfer `require` bump — which is why it is filed rather than started.

**Gate for it:** the `-short` metal suite run N times with zero `fault 0x10` deaths (N large enough
to beat the observed rate — 20+ given ~30-50%), plus the existing goldens unchanged.

**A13 · A device-draining predecessor poisons the `attention` kernel — zeros at hd≥64, with the
card empty and every call succeeding** — `linux`, **open, BISECTED 2026-08-13**

**The predecessor is found, and it is a category rather than a single test.** Bisected over the ten
tests that precede the victim in run order, each paired with it, two repeats each:

| predecessor | reproduces | what it does |
|---|---|---|
| `TestAllocFloor` | **yes, 4/4** | allocates until the device refuses |
| `TestA10ReportingGap` | **yes, 4/4** | allocates until the device refuses |
| `TestA10FloorIsPerProcessOrPerDevice` | **yes, 4/4** | balloons VRAM to measure the floor |
| the other seven | **0/2 each** | none of them drain |

**Every poisoner drains the device to exhaustion; nothing that does not drain, poisons.** Two of the
three were fixed earlier to release everything (`5ece205`), and they still poison — so it is **not
residual allocation**.

**Four things ruled out by measurement, not by argument:**

- **Memory pressure.** Free VRAM at the moment attention runs is **7,661,092,864 B (7306 MiB)** — the
  card is essentially empty.
- **A failed allocation.** `mustAlloc` fatals on an error *and* on a nil buffer with no error.
- **A dropped error.** Every call is checked now (`e682eb2`): `CopyHtoD`, `LaunchOn`, `Synchronize`,
  `CopyDtoH` all succeed.
- **Parallelism.** `-p 1`, one package, no `t.Parallel`.

**A correction to the first filing: the "hd=64/128 only" bracket was an artefact of the full-tier
ordering.** In the minimal pair, **hd=64, 128, 256 and 512 all fail and only hd=16 passes**. So the
rule is *everything above the smallest width*, not a middle band — which removes the "a resource
ceiling would fail the large dims" puzzle, because the large dims *do* fail.

**So: a drain-to-exhaustion leaves the context in a state where a later `attention` launch writes
zeros, on a card with 7.3 GB free, reporting success at every step.** That is the finding, and no
mechanism is proposed for it here.

**MECHANISM FOUND (2026-08-13): the DRAIN EVICTS THE MODULE'S DEVICE CODE, and the cached handle
survives it.** Two tests, and they only look like they disagree:

| probe | result | reading |
|---|---|---|
| **1. interrogate the cached handle** (`cuFuncGetAttribute` ×6, poisoned vs clean) | **byte-identical, all valid, no errors** — `maxThreadsPerBlock=1024 sharedSizeBytes=0 localSizeBytes=0 numRegs=62 ptxVersion=75 binaryVersion=75` | the handle *answers*; it looks live |
| **2. force a module reload before the launch** | **all five widths PASS, 3/3 runs.** Control without reload in the same session: 4 failures, 2/2 runs | the module *was* the problem |

**They reconcile, and the reconciliation is the finding: `cuFuncGetAttribute` is NOT A LIVENESS
PROBE.** It answers out of metadata that outlives the device code — max threads, register count, PTX
version are all still there after the underlying module has been evicted. So the handle stays
*queryable* while what it names is gone, the launch reports success, and nothing executes. That is
exactly the "returns success and does nothing" shape, and it explains every observation at once:
identical launch arguments, all calls succeeding, full card, zeros out.

## `gate gpu` is RED on nobara for an environmental reason, and its message misdirects

**Measured 2026-09-01.** `go run ./cmd/gate gpu` exits rc=1 with 170 PASS / 4 FAIL:
`TestAllocFloor` and `TestMoERouteDemandThreshold` (plus their child/marker pair), in the drain
group.

**It is not a regression.** The identical failure, with the identical measured byte value,
reproduces at `82dda2a` — before any of the 2026-09-01 changes. Verified by checkout, not assumed.

**The cause is a foreign CUDA context, and the arithmetic names it exactly:**

```
measured demand   156,958,720 B
expected          140,181,504 B   (residual 138,412,032 + cold floor 1,769,472)
difference         16,777,216 B   = exactly 16 MiB
```

`nvidia-smi --query-compute-apps` reports **pid 1777, `/usr/bin/kwin_wayland`, 16 MiB** — the KDE
Plasma compositor. The test's COLD regime means *"this launch is the first context on the device
and pays the reserve itself"*, and that premise is false while the compositor holds a context.

**Why this is worth a note rather than a shrug.** The failure message is otherwise excellent — it
tells you to check components first — but it ends with *"ONLY if the identity does not close has
the KERNEL's launch requirement moved … and then docs/QUEUE.md A1/A5/A7/A9 need re-deriving"*.
Someone reading a red gate on a desktop session would be pointed at re-deriving four pinned
allocation figures, when the whole discrepancy is one compositor's 16 MiB. **Do not edit those
pins on the strength of this failure.**

Options, none taken here: run the gate from a TTY with the compositor stopped; teach the COLD
regime to subtract foreign contexts' reported usage; or have the gate refuse to start (as
`bench_peer.py` does for load) rather than fail deep inside. The third is probably right — §B8's
own provenance says "GPU exclusive at start (the supervisor refuses to launch if anything beyond
the compositor holds the card)", so the *benchmark* supervisor already treats exclusivity as a
precondition while this gate does not.

## A13 — THE TAG ARGUMENT, AS ONE ENUMERABLE CLAIM

> **No shipped path drives the device to refusal and then continues using the context.**

The trigger is **exhaustion** — `TestAllocFloor` allocates until even a 1 MiB request is refused
(~144 MiB free) and poisons every time, 5/5. Partial holds are a weaker, intermittent stimulus. So
the claim is checkable path by path rather than argued from sizes:

| path | reaches refusal? | continues on the context? | evidence |
|---|---|---|---|
| **prefill scratch** | **no** — min free **1820.2 MiB** with a **7B** resident (the largest this box runs) at M=2048: **12.6× the refusal floor**, 1676 MiB of margin | yes | `docs/measurements/prefill-peak-vram.md` — VRAM tracer, 1347 samples. **CORRECTED from 39.9×**, which was measured against a 1.5B and overstated the headroom ~3×; scratch competes with resident weights, so the small model was not the worst case |
| **admin unload** | **no** — frees, never allocates toward refusal | yes (shared primary context) | `Model.Close` → `ReleaseObjects`; refcount reading |
| **multi-model unload** | **no** | yes | measured clean 3/3 — 7B (≈4.9 GB) unloaded under a co-resident B, B's logits byte-identical |
| **A5 cap search** | **no** — allocates nothing | n/a | `fits(n)` is arithmetic against `free` |
| **resident build OOM** | **YES** | **no** — declines and the fallback issues no CUDA | `(nil,false,nil)`; `cudaBackend.MatmulBT` → `linalg.MatmulBT`; refusals harmless at 1000× |

**One property, five checks.** The only path that reaches the trigger is the one that stops using the
context, and the only paths that keep using the context stay orders of magnitude away from it. That
is a stronger and more falsifiable statement than the four shifting properties it replaces — any new
path need only be tested against the one claim.

### A13 — CLOSING STATE (2026-08-13). Understood, uncharacterised in one part, NON-BLOCKING.

| | |
|---|---|
| **status** | **UNDERSTOOD, not solved.** Open deliberately, not parked by neglect |
| **trigger** | **named and reproducible: drain the device until a 1 MiB request is refused (~144 MiB free). 5/5.** Deterministic |
| **mechanism** | **established.** The module's device code becomes unusable while the cached handle stays *queryable* — `cuFuncGetAttribute` returns byte-identical valid values across poisoned and clean runs, because that metadata outlives the device code. Launches then return SUCCESS and write nothing. No CUDA call reports an error, and free VRAM is back to ~7.3 GB by the time it shows |
| **remedy, measured** | re-load the module and re-resolve the function **immediately before** the launch — 3/3. Re-loading earlier does not work: allocations between load and launch re-invalidate it |
| **NOT characterised** | the **thread factor** — why a pinned test goroutine poisons where the resident's `LockOSThread` executor does not. Pinning alone was tested and is *not* the variable. This is the honest gap |
| **why it does not block** | the enumeration above: no shipped path drives the device to refusal and then continues using the context. Five paths, each with its own evidence, four of them measurements |

**THE FALSIFIER — the one sentence a future implementer needs:**

> **Any future change that drives the device to refusal and then continues using the same context
> breaks this.**

That is what makes A13 non-blocking rather than merely unobserved, and it is what a reader must check
a change against. It is deliberately a *property*, not a list of forbidden functions: a retry loop, an
evict-and-rebuild, a "try a smaller cache and continue", a second model sized against free VRAM — none
of those is named anywhere, and all of them break it. The failure will be **silent zeros, not an
error**, so it will not announce itself.

**It is recorded in two places on purpose**, because the entry is not where the change gets written:
here, and at the module/pipeline cache site in `cuda/backend.go` — the code someone adding residency
recovery is actually reading. Neither location is decorative; a rule filed only in a queue entry is a
rule nobody encounters at the moment they break it.

Chasing the thread factor past this point is research, not release work.

**THE SWEEP'S STIMULUS IS INTERMITTENT; THE REAL ONE IS NOT. The percentage figures are weaker than
recorded (2026-08-13).**

| stimulus | reproducibility, measured today |
|---|---|
| `TestAllocFloor` drain → `attention` (the original) | **4 subtest failures in 5 of 5 runs — fully deterministic** |
| synthetic hold+release 50% (`GOINFER_A13_HOLDPCT`) | **intermittent** — one 6-run block came back `C C C C P C` |

Earlier the synthetic probe returned POISON three times running, and the sweep's "reliably poisoned
at ≥25%, unstable at 15%" was built on it. **That characterisation is withdrawn**: the percentages
were measuring an unreliable proxy, not the phenomenon. What is stable is the *real* stimulus —
`TestAllocFloor` drains until even a 1 MiB request is refused, i.e. to **actual exhaustion** (~144
MiB free), and that reproduces every time.

**So the trigger is exhaustion, not held bytes.** A partial hold — even 90% — is a different and much
weaker stimulus than driving the device to refusal. That reconciles the non-monotonic sweep: it was
noise around a threshold effect the probe only sometimes reached.

**Consequence for the persistent-allocation test.** It was run against the *flaky* probe and returned
`P P C C C C` at 1024 MiB against a control of `C C C C P C` — **no separation, and neither arm
reliable**. It says nothing either way about the live-set-collapse hypothesis. **The test needs
re-running against `TestAllocFloor`'s drain**, which is the only stimulus with a stable positive:
hold ~1 GB that the drain cannot take, let the drain exhaust and release, then launch. That is a real
experiment; the one just run was not.

**PINNING IS NOT THE VARIABLE (2026-08-13) — the eviction story SURVIVES.** The cheapest possible
alternative explanation was that CUDA usage from a goroutine Go is free to migrate across OS threads
is the real mechanism, and the resident's `LockOSThread`-pinned executor is immune merely by being
pinned. One line tested it:

    UNPINNED  hold+release 50%   POISON POISON POISON
    PINNED    hold+release 50%   POISON POISON POISON   (runtime.LockOSThread + defer Unlock)

**Three of three either way.** So `cuFuncGetAttribute` staying valid while a reload fixes the result
still stands as the finding, `cuda/backend.go`'s cache-site comment stands as written, and nothing
downstream of eviction needs revisiting.

**What it does leave open: the thread factor is real but NOT pinning.** A pinned test goroutine
poisons; the resident's executor does not. Both are pinned, both use the same primary context, both
go through gocudrv. Whatever distinguishes them is something other than OS-thread affinity —
candidates not yet separated include how context currency is established on each thread, and the
fact that the resident's context permanently holds a model's worth of live allocations while the
test's does not. **Not characterised, and not guessed at here.**

**THE 1×2 IS FILLED, AND BOTH SHIPPED-PATH RESULTS ARE NULLS THAT DO NOT YET COUNT (2026-08-13).**

| cell | stimulus | result |
|---|---|---|
| **(LoadModule prebuilt PTX, test goroutine)** | **drain to refusal** (deterministic, 5/5); the synthetic hold+release percentages once cited here are withdrawn | **POISONS** — the only combination ever observed to |
| **(CompileLibrary, resident executor)** | hold+release 3648 MiB via `rf.do` | clean |
| **(LoadModule prebuilt PTX, resident executor)** | hold+release 3648 MiB via `rf.do`, module loaded and launched on the same executor | **clean** — `before=768 after=768` non-zero |
| **multi-model unload** (`A` = 7B int4 ≈ 4.9 GB ≈ 67% of the card, `B` co-resident, `A` closed through the shipped path) | the real feature | **clean, 3/3** — B's logits byte-identical to its pre-unload baseline |

**The context factor is eliminated by reading, not by testing:** `CreateSystemDefaultDevice` calls
`dev.Primary()` and gocudrv never binds `cuCtxCreate`, so every cell above uses the *same* primary
context. What is left is the **thread**, and the missing cell says resident-executor launches are
**not** poisonable by this stimulus — which is exactly why the multi-model null is not yet evidence.
**Nothing has ever poisoned a launch made on the resident's executor.**

So the honest position: **the shipped paths look clean and the harness cannot yet demonstrate that it
could show otherwise.** Two nulls from a route with no positive control are not a clearance.

**CORRECTION (2026-08-13): "each resident model builds its own context" IS FALSE, and it was doing
load-bearing work in the tag argument.** Read rather than assumed, after the prefill control failed
to fire:

- `CreateSystemDefaultDevice()` (aikit `gpu/cuda.go`) calls **`dev.Primary()`** — the device's
  **primary context**, retained by refcount. gocudrv **does not bind `cuCtxCreate` at all**.
- `Context.Close()` calls **`PrimaryCtxRelease`** — a refcount **decrement**, not a destroy.

So every resident model in a process shares **one primary context per device**, and that context is
destroyed only when the **last** holder releases it.

**What that breaks, and what it does not:**

- **`TestResidentCloseFreesVRAM` is still valid** — its three load→forward→close cycles are
  sequential with no other holder, so each `Close` drops the refcount to zero and the context really
  is torn down between cycles. Its green is honest for the single-model case.
- **The multi-model case is NOT covered, and I argued it was.** With model B loaded, unloading model
  A releases gigabytes **inside a context that stays live for B** — which is precisely the sweep's
  stimulus, in a shipped feature (`POST /admin/models/unload`), reached by ordinary operation.
- **"Resident immunity" was a misnomer** before it was ever tested: the resident's context and the
  test's context are the same object, so the context cannot be the variable in the 2×2. The
  remaining factors are the **module route** (`ctx.LoadModule` of prebuilt PTX vs `CompileLibrary`
  via NVRTC) and the **thread** (test goroutine vs the resident's pinned executor).

**Consequence for the tag: the admin-unload path returns to the open list.** Its earlier "CLEAN —
destroys the context" verdict rested on the same false premise and is withdrawn. What it actually
does is release a model's device memory into a context that other models may still be using.

**PREFILL MEASUREMENT: INCONCLUSIVE, because the POSITIVE CONTROL DOES NOT FIRE. Reported as
inconclusive rather than clean.** (`cuda/a13_prefillchurn_test.go`, 2026-08-13)

`ReleaseBuf` was read first and the stimulus is real: `ReleaseBuf` → `Buffer.Close` →
`cudaresult.MemFree`. **No pool, no reuse** — prefill scratch genuinely returns hundreds of MB to
the driver inside a live context, on every long prompt.

The measurement ran, and both symptoms came back clean: 8 repeated prefills produced **identical
logits, 128/128 non-zero, zero drift**, and a decode forward on the same context afterwards was
also clean. **That result is not evidence, because the control that should have poisoned the same
context did not.**

| control attempt | stimulus | poisoned? |
|---|---|---|
| v1 — allocate/free 3648 MiB via `dev.Primary()` | wrong context entirely: `BuildResident` creates its **own** context, so this never touched the subject | no |
| v2 — allocate/free 3648 MiB via `rf.dev` on the resident's own executor thread (`rf.do`) | **the right context** | **still no** |

**So one of two things is true, and the measurement cannot distinguish them:** either a resident
context is genuinely immune to this stimulus — which would be the real reason production is safe, and
a far better answer than any of the four enumerated properties — or the harness still cannot express
the effect and the prefill question remains open.

**What separates them:** demonstrate the poisoning on a *resident-context* launch at all. Every
observed instance so far used a module loaded by a test via `ctx.LoadModule(gluePTX)` in the primary
context; the resident compiles its own via `CompileLibrary` on a pinned thread. If the effect cannot
be produced there by any stimulus, resident immunity is the finding and it is worth more than the
enumeration. **Until that is shown, prefill is not cleared, and the tag question stays open.**

*(v1's failure is itself the recorded lesson: a control on the wrong context looked like a clean
result. It was caught by running the control before believing the null — the fifth time in this
campaign that a null needed its forcing mechanism verified first.)*

**THE INVARIANT, STATED AS WHAT IT ACTUALLY IS:** *no large hold-and-release inside a live CUDA
context.* "Per-model contexts" understates it — that is one mechanism that happens to satisfy the
invariant, not the invariant itself.

**ENUMERATION OF PATHS THAT COULD VIOLATE IT (2026-08-13):**

| path | allocates & frees in a live context? | verdict |
|---|---|---|
| **admin unload** (`POST /admin/models/unload`, `--unload-drain-wait`) | **WITHDRAWN — see the correction above.** `cx.Close()` is `PrimaryCtxRelease`, a refcount decrement. With another model loaded the context stays live and this is a large free inside it | **OPEN — the strongest candidate** |
| **A5 cap search** (`capSlots`) | **no — pure arithmetic.** `fits(n)` compares `slotRequirement(...)+slotMarginBytes` against `free`; nothing is allocated to probe | **CLEAN** |
| **`allocSlots` failure** | **no.** OOM arrives as a panic from `MustBuf` and *the resident is discarded on decline* — context torn down | **CLEAN** |
| **expert cache / KV growth** | no mid-life grow/resize path exists in `cuda/` | **CLEAN** |
| **prefill scratch** (`cuda/prefill.go:259`) | **YES.** `scratch` is allocated per call and released by a deferred loop of `r.dev.ReleaseBuf(b)`, inside the live resident context. Its own comment: *"at M=3000 this is hundreds of MB"* | **NEEDS MEASUREMENT — see below** |

**PREFILL SCRATCH IS THE ONE PATH THAT MATCHES THE POISONING CONDITION, and it ships.** Every long
prompt allocates hundreds of MB of scratch and frees it without leaving the context. That is
structurally the sweep's stimulus, on the hot path, in a shipped feature.

**SUPERSEDED — the band it referenced was a proxy's noise.** This paragraph originally sized
prefill's risk against the sweep's "reliably clean ≤~12%" figure. **Those percentages are withdrawn**
(see below): they came from a synthetic hold+release probe that later proved intermittent, while the
real stimulus — drain to refusal — is deterministic. Prefill is now closed on a *measurement of its
own peak*: min free 5752.2 MiB, 39.9× the refusal floor. That is a better argument than the one this
paragraph made, and it does not depend on any band.

**TAG: this is the open question, and it is a shipping path rather than a test one.** If repeated
prefill scratch churn poisons the context, that is a correctness bug on the same constraint as
before — silently wrong output with no error — and the tag waits. If it does not, the tag stands on
four named properties rather than two. The measurement is one test: build a resident, run several
long prefills, then validate a launch on the same context.

**THE PRODUCTION QUESTION, ANSWERED FIRST (2026-08-13): a failed allocation does NOT poison. A
large SUCCESSFUL one, freed, can.**

| stimulus | result |
|---|---|
| **1 refused allocation** (15.1 GiB request) | **CLEAN, 3/3** |
| **2, 10, 100, 1000 refused allocations** | **CLEAN** at every count |
| hold+release 1%, 3%, 6%, 12% of free VRAM (≤ ~876 MiB) | CLEAN |
| hold+release 18%, 21% (~1.3–1.5 GiB) | CLEAN, and 18% is 3/3 |
| hold+release **15%** (~1.1 GiB) | **POISON once, CLEAN twice** |
| hold+release **25%, 50%, 75%, 90%, 95%** | **POISON** |

**So the refusal is not the trigger — the successful allocation is.** A thousand refused requests
leave the context perfectly usable; holding and releasing roughly a gigabyte or more can break it.
That inverts the intuition the draining tests suggested, and it matters because *the decline path
only ever experiences refusals*.

**NO THRESHOLD IS CLAIMED, AND THE RANGE IS WITHDRAWN (2026-08-13).** This paragraph used to close
with "reliably clean at or below ~12%, reliably poisoned at or above ~25%, unstable between." That
range is **retracted as a correction, not softened**: a re-run of the *same* stimulus at a fixed
percentage returned `C C C C P C`, so the probe is **intermittent at a fixed setting** and the
non-monotonicity was its own noise being read as structure. A single run per percentage never
supported a band, and the second run showed it.

What survives is the stimulus that is *deterministic*: **drain until a 1 MiB request is refused**
(~144 MiB free) poisons **5 of 5**. The trigger is exhaustion, not held bytes — which is why the
percentages never resolved. Everywhere the withdrawn band was used to size risk (prefill's "hundreds
of MB", the multi-model unload's 67%, the A13 cell) it has been replaced by a direct measurement or
by the enumeration above.

**What this does to the tag.** The exposure requires a **large successful allocation, freed, followed
by a launch on the same context**. Two things stand between production and that, and they should be
named separately because they are different guarantees:

1. **The decline path never gets there.** On exhaustion `BuildResident` refuses and returns
   `(nil,false,nil)` — refusals only, which the sweep shows are harmless at any count — and the
   fallback runs `linalg.MatmulBT` with no CUDA at all.
2. ~~**Each resident model builds its own context**~~ — **WITHDRAWN, THIS WAS FALSE.**
   `CreateSystemDefaultDevice` calls `dev.Primary()` and `Context.Close` calls `PrimaryCtxRelease`,
   a refcount decrement; gocudrv never binds `cuCtxCreate`. **All models share ONE primary context
   per device per process**, destroyed only when the last holder releases. What survives is
   narrower: `TestResidentCloseFreesVRAM`'s three 7B load→forward→close cycles are green *because
   they are sequential with no other holder*, so each `Close` does drop the refcount to zero. That
   is the **single-model** case only; it says nothing about a co-resident model, and I used it to
   argue the multi-model case was safe. See the correction below.

**The multi-model server case is therefore covered by (2), not by luck** — but it is covered by a
property nobody wrote down as a safety property until now, which is exactly how it gets refactored
away. **A shared long-lived context across models would expose this directly.**

**REFINED, because the obvious remedy failed and the failure is informative.** Reloading the module
*inside the draining test* — repairing what it disturbed — **does NOT fix the victim**:

| sequence | result |
|---|---|
| drain → victim loads module → allocates → launch | **fails** |
| drain → **drain reloads** → victim loads → allocates → launch | **fails** |
| drain → victim loads → allocates → **reload** → launch | **passes, 3/3** |

All three load the module after the drain. The one that works is the one with **no allocation between
the load and the launch**. So this is not a single eviction the drain can undo: **allocation activity
after a load, in a post-drain context, re-invalidates the module.** The victim's own `mustAlloc`
calls are enough to do it.

That also rules out the tidy fix — a draining test cannot restore the invariant for whatever runs
next, because the next test's own allocations break it again.

**OPEN OBSERVATION, deliberately not folded into the story: hd=16 survives.** Every other width
fails in the minimal pair. The reload result says the module is unusable, and that does not
distinguish *"module gone"* from *"module gone except one path"* — a partial survival wants its own
measurement. Reading it as a size threshold would be a guess, and it is recorded here as an
observation precisely so it does not get absorbed into the eviction narrative as though explained.

**THE LAUNCH IS IDENTICAL — the state is in the driver or the context, not in the call.** Every field
the launch depends on was logged in both a clean and a poisoned run (`GOINFER_A13_LAUNCH=1`):

    grid=(12,1,1) block=(128,1,1) shared=1028 nH=12 nKV=2 nKeys=129 scale=0.05
    lens=(768,16512,16512,768)          <- identical for a given hd, both runs

**No argument differs. No extent differs. No stride differs.** And the pointer question is answered
too: **hd=64 FAILS at the very same base address where hd=16 PASSES** (`0x…200000` in both), so a
differing device pointer is not the discriminator either. That is the third pre-registered branch —
everything identical, result different — which **rules out anything goinfer computes** and scopes the
remaining search to context- or driver-level state.

*(A secondary observation, not the cause: after a drain the allocator stops reusing the freed slot
and hands out advancing addresses, where a clean run returns the same address for every subtest.
Consistent with allocator state having changed; it does not explain a zero result, since the failing
launch happens at a reused address.)*

**SCOPE, SHARPENED — "only three measurement tests drain" was too comfortable.** The trigger is
**exhaustion followed by continued use of the same context**, and **production does reach
exhaustion** — the 26B at 34 slots is the origin of this entire campaign. So the question is not
whether production drains the card; it is whether anything keeps using the context afterwards.

**It does not, and that is the reason A13 is unreachable in production — stated rather than left
implicit.** On exhaustion `BuildResident` declines and returns `(nil, false, nil)`, and the fallback
is `cudaBackend.MatmulBT`, which dispatches to **`linalg.MatmulBT` — the shared SIMD kernels, the
same ones the CPU backend uses**. The staged path touches no CUDA. So after a real OOM the process
stops issuing CUDA work entirely, and the poisoned context is never launched into again.

**The decline is load-bearing, and that is the finding.** If the fallback had kept using the context —
a staged path that still issued kernels, say — A13's mechanism would be reachable from a real 26B OOM
and this would be a correctness bug rather than a test-interaction one. It is worth knowing that the
safety comes from the decline rather than from the kernel being robust.

> **DO NOT "SIMPLIFY" THE FALLBACK INTO SOMETHING THAT KEEPS ISSUING KERNELS.** `BuildResident`
> returning `(nil, false, nil)` on exhaustion, and `cudaBackend.MatmulBT` dispatching to
> `linalg.MatmulBT` with no CUDA at all, is the *only* thing standing between a real OOM and
> silently wrong output. It reads like an incidental fallback and it is a safety property.

**AND THE CONSEQUENCE, now that eviction is confirmed:** the shape where this would bite is a
**long-running process that hits memory pressure and recovers** — a server that OOMs once, sheds
load, and carries on. Its cached module handles would survive the pressure event while the device
code behind them did not, and every subsequent launch would succeed and produce zeros. goinfer does
not reach that today only because it stops issuing CUDA work entirely after a decline. A future
change that recovers residency after pressure, instead of declining permanently, must re-load modules
rather than reuse cached handles — and must not rely on `cuFuncGetAttribute` to tell it whether it
needs to.

**TAG IMPACT — a test-interaction defect, not a production one**, for the reason above rather than
because the trigger is rare. The production forward path is unaffected and the resident parity gates
pass. A13 joins the other three A12 items as a test defect.

**A2 (partial) · 26B documentation correction** — `linux`, 2026-08-12

The half that does NOT depend on A1 is done: the README instructed
`GOINFER_MOE_CACHE_SLOTS=48` and claimed it auto-caps to 38, when at the free VRAM the gates
observe it caps to **34 — which fails**. So the published instruction could produce an OOM on the
card it was measured on. Corrected to the highest measured-safe value (30), with the hit-rate curve
and an explicit reproducibility note on 16.98.

**Unblocked — A1 is closed.** The corrected cap is **33**, and the leftover-VRAM column follows from
the closed form (table under A1). Both were previously withheld because the model that would supply
them had been refuted; that is no longer the case. The remaining half is publishing them, and A7
confirms 33 by run before anything is published as safe rather than as computed.


## Queued

**INHERITED · `scripts/shard_checkpoint.py` moved here from aikit** — `unclaimed`, **filed 2026-08-14**.
aikit's zero-Python campaign found this script in `aikit/scripts/`, but it is goinfer's: it splits a
single-file safetensors checkpoint into N shards + `model.safetensors.index.json`, and
`decoder/sharded_test.go` (`const gemmaShardedDir = "../testdata/gemma-3-270m-sharded"`) consumes its
output. It references `decoder.LoadWeights`, which only exists here — a leftover from the aikit/goinfer
split; aikit deleted its copy. It now lives at `goinfer/scripts/shard_checkpoint.py` and runs from the
goinfer root (its `sys.argv` defaults resolve to `goinfer/testdata/gemma-3-270m{,-sharded}`). Action:
none required unless goinfer wants it gated or under a `scripts/` convention — recorded so it is not a
second thing that "exists and is composed into no decision". Uses `safetensors.torch` (a torch dep):
goinfer's to keep or port on goinfer's terms.

**B10 · `aikit` and `aikit/gpu` carry SEPARATE TAG SERIES — the single home for that fact** —
`linux`, **filed 2026-08-13**

Three instances, one fact, and the third was inside the tool built to check the first two:

| instance | how it bit |
|---|---|
| **E6** | `git diff v1.16.0..v1.17.0 -- gpu/` reported a 72-line PTX change that goinfer had already been running — a nested module diffed across the *parent's* tags |
| **B7** | the manifest's single `aikit_version` field cannot represent two versions; it drifted to `v1.12.0` against a `go.mod` saying `v1.16.0` |
| **the citation lint** | `gpu/cuda.go` resolved nowhere, because `required_modules()` read only the root `go.mod` and `aikit/gpu` is not inside `aikit@v1.17.1`'s module-cache directory. Reddened CI on a correct reference |

**THE FACT LIVES IN `scripts/queue_citation_lint.py`'s `required_modules()`**, which now reads *both*
`go.mod` files — the root for `aikit`, `cuda/go.mod` for `aikit/gpu` — and prints both resolutions
with every green:

    CROSS-REPO RESOLUTION: ... answered from the MODULE CACHE at the version go.mod requires
      github.com/townsendmerino/aikit      -> module cache @ v1.17.1
      github.com/townsendmerino/aikit/gpu  -> module cache @ v0.28.0

That is the right home because it is **derived rather than restated** — it reads the versions from
the files that own them, so it cannot drift the way B7's hand-maintained field did, and it is
**executed on every CI run** rather than being a paragraph someone has to remember. The printed lines
mean a fourth instance has somewhere to look and something to compare against.

*Still restated by hand, and therefore still able to drift:* the manifest's `aikit_version` (B7,
open) and `RELEASING.md`'s alignment step (now derived — it reads the versions rather than naming
them, `0898295`).

**B9 · Dropped errors on launch/copy/sync paths — 268 production sites, enumerated** — `linux`,
**filed 2026-08-13**

A12's fourth failure was `attention cosine 0.000000`, silently, with no CUDA error anywhere in the
run. The cause was every call in the block discarding its error (`_ = gc.CopyHtoD`, `_ = LaunchOn`,
`_ = Synchronize`, `_ = CopyDtoH`): the output buffer was never written, `got` stayed zero, and a
resource failure arrived wearing a numerics bug's clothes. **`errcheck -blank` census:**

| module | total | test | **production** |
|---|---|---|---|
| `cuda/` | 622 | 587 | **35** |
| `gpu/` | 573 | 409 | **164** |
| `metal/` | 21 | 18 | **3** |
| root (`decoder/` etc.) | 317 | 251 | **66** |
| | | | **268** |

**The production set is the one that matters, and `cuda/`'s 35 are concentrated on the DECODE FORWARD
PATH** — `r.launch(...)`, `r.doG(...)`, `r.stream.Sync()`, `gpu.Download(...)` in `cuda/resident.go` and
`cuda/prefill.go`. In a test a dropped error reaches an assertion; **in production it reaches a caller**,
as silently wrong output. That is a strictly worse version of the bug A12 just found in a test.

**It can be a gate, unlike the dispatch shape.** `_ =` on an error return is **syntactically
decidable**, so a lint can decide it without knowing intent — which is exactly what made B6's
dispatch census a report rather than a verdict. Deliberately-ignored sites get **declared with a
reason**, in the shape the citation lint already uses for allow-paths, so **the count is the check**
rather than anyone's judgement about a given line.

**Cost, stated so it is chosen rather than discovered:** 268 sites need triage before the gate can
go green, and most are tests where the ignore is defensible. The forward-path production sites are
the ones worth fixing first, and they are ~35 rather than 268.

**A false zero in the census itself, recorded because it nearly shipped:** the first `metal/` run
reported **0** — because exporting `GOOS=darwin` made `go run` build a *darwin* errcheck binary and
try to execute it on Linux (`exec format error`). The tool never ran. A native binary with `GOOS`
set for the analysis only, run from inside the module with `GOWORK=off`, gives the real 21/18/3.
"Zero findings" and "the tool did not run" are the same output, which is the same class the tracer's
zero samples belonged to.

**B8 · A sweep must distinguish "could not evaluate" from "failed"** — `linux`, **filed 2026-08-12**

`scripts/parity_sweep.sh` reported **27 blockers**, then **15**, then **0**. The tree was fine at all
three. Every one of the 42 was a **missing asset or an unset env var** — an *inability to evaluate*,
rendered in the output as a *result*, under one label: `⚠️ SKIP — asset missing (blocker)`.

**Two concrete costs, both paid.** The label says "asset missing" for gates that skipped on
`GOINFER_HEAVY_TESTS` being unset, which sent me looking for checkpoints that were sitting in
`~/models` the whole time. And a run whose blockers are all unevaluated gates reads identically to a
run with real failures — the operator cannot tell a red tree from an unequipped box without opening
the log.

**The fix is in the output, not in the gating.** Both still block a tag: an unevaluated required gate
is not a pass, and that rule is right. But they must be *counted and named separately*:

    27 BLOCKER(S): 0 FAILED, 27 NOT EVALUATED (19 asset absent, 8 env gate unset)

and the env-gated ones should name the variable, since that is a one-line fix rather than a download.

**Same shape as the Mac's 31 silent skips, and as `scripts/gpu_gate.sh` reporting an OOM as "a CUDA forward
moved".** An absence rendered as a result, and a cause rendered as the wrong kind of cause. The
project already has the rule — *a skip is not a pass* — and these are the corollary: **a skip is also
not a failure, and a gate that cannot say which it had is asking the operator to guess.**

**B7 · `aikit_version` is a hand-maintained input to a computed gate** — `linux`, **found 2026-08-12
during the v1.17.0 bump**

`testdata/parity_manifest.json` carries an `aikit_version` field that is **mixed into `deps_hash`**,
so the staleness gate re-stales every family when it changes. That is the right design and it works.
The gap is that **nothing computes the field** — it is typed in by hand. At the v1.17.0 bump it read
`v1.12.0` against a `go.mod` that said `v1.16.0`.

**What the drift means, stated precisely, because it is narrower than it first looks.** An aikit bump
that does not touch the field changes no manifest input, so `deps_hash` does not move, so the
staleness gate stays green and **no goldens run**. At least one prior bump went through that way.
The gate did not fail; it was never given the input that would have made it fire. This is the
absence-of-signal shape — a green that means "nothing asked" rather than "nothing changed".

**Why it is a B and not an A.** goinfer's numerics did not silently drift: aikit's own bit-identity
discipline (`be049df`, `TestKernelFMALint`) is what held, and the v1.17.0 bump was checked by hand
and by a goldens run. So this is a *missing interlock*, not a live defect. It also means the fix
cannot be "trust the field more".

**The fix is to derive it.** Read the aikit `require` lines out of the `go.mod` files at manifest-
build time and fail if they disagree with each other or with the recorded value — the same
derive-both-sides rule `scripts/selector_coverage.py` follows for its selectors. Two versions matter,
not one: the root module *and* `gpu/`, which are separate tag series that do not track each other
(see E6). A single `aikit_version` string cannot represent both, so the field likely becomes a small
object. Until then the enforcement is the bump ritual in `docs/RELEASING.md`, i.e. a person.

**THE SHAPE, which is why this is filed as a class and not a chore.** A hand-maintained constant that
duplicates a value computed elsewhere is **sibling drift with a literal as one of the siblings**. The
existing two shapes are a *check* naming one member and a *dispatch* naming one member (B6,
`docs/parity-coverage-policy.md`); this is the third — a *constant* restating one. The recognition
test is unchanged: **is this value maintained anywhere else?** If yes, the copy will drift, and being
data rather than code buys it nothing — it buys it less, because no compiler, vet or lint reads it.
The remedy is unchanged too: derive it.

Two instances landed the same day, which is what promoted it from a chore to a class. This one, and
`RELEASING.md`'s version-alignment step, which named the versions to align on and went stale
**twice** before being rewritten to read them (`0898295`). Both were literals restating something a
`go.mod` already knew.

*Cross-reference:* B6 carries the check/dispatch shapes; this is the constant shape, and the three
belong to one class.

**B1 · Env-var lint** — either box

`cc238c6` landed the `GINFER_`→`GOINFER_` rename (31 files, real work) and an env-var
classification as `docs/env-vars.md`. The classification is the expensive part and it's done. What's
missing is that it be **executable**.

Convert that table into the Go source of truth and have a lint read it — **do not write a second
registry beside the document**, which leaves two lists to drift apart. The lint fails on any
`os.Getenv` in the tree naming a variable absent from the registry.

**Also catch env reads at package initialisation.** A `var x = os.Getenv(...)` at package scope is
read before any test runs, so `t.Setenv` cannot override it — a test that sets it silently gets the
default and its result reads as a measurement. That cost a full six-minute 26B run on 2026-08-12
(`GOINFER_A1_PROBE`), and the only reason it was caught is that the probe asserted it had recorded
something. The lint has the `os.Getenv` call sites already; flagging the ones in package-level
initialisers is the same pass. Folded in here rather than filed separately, because a second env
registry is exactly what B1 exists to prevent.

Why it matters: 105 variables, exactly one of which anything has ever set. Six have no prefix at all
(`ZZBASE`, `GEMMA3_4B`, `G4_TRACE`, `NOISE_FLOOR_CKPT`, `ROUTER_CAPTURE_OUT`, `GIW_BIG`) and are
only findable *as a class* if something enumerates `os.Getenv` mechanically. A markdown table
maintained by intention drifts on the first variable someone adds.

**B2 · Gate reconciliation — one entry point** — `linux`

Two mechanisms now exist for running the heavy tier: `scripts/gpu_gate.sh` group 2c (linux) and
`scripts/heavy_gate.sh` (`8fecfad`, mac).

Resolve to one: **`scripts/gpu_gate.sh` always declares the heavy group.** When not requested it emits a
counted skip with its reason and the verdict line carries it. Fast runs stay fast; no run silently
omits the tier. `scripts/heavy_gate.sh` becomes the implementation group 2c invokes, or it goes. Two files
is fine; two entry points isn't, because **the verdict has to come from one place**.

**B18 · Release-path restructure (T0–T5)** — `linux` (the diagnosis it's built from is
CUDA-side), **filed 2026-08-16**. Decided 2026-08-12 by Francis as "the first thing after the
release"; v0.13.0 shipped before this was recorded, and it was reconstructed from the
conversation that produced it. Design page:
[`docs/task-release-path-restructure.md`](task-release-path-restructure.md) — the context, the
evidence and the acceptance criteria per item live there; this entry is the claimable work.

**Why:** v0.13.0's tag took two days. Day one found real defects (the 26B slot-cap bug, goldens
covering f32 only, an aikit decode regression, 35 dropped errors). Day two — almost entirely the
CUDA tier — found zero production defects: three test defects, one test-interaction phenomenon
(A13) no shipped path can reach, and a lot of careful measurement establishing exactly that. The
rigor wasn't the problem; it was pointed at something that couldn't affect what ships.

**Five items, budgeted (an afternoon + two days total; overrun >50% on any one item stops and
reports rather than continuing):**

- **T2 — triage before diagnosis, DO FIRST.** Rule for `RELEASING.md` +
  `parity-coverage-policy.md`: a red gate's first question is "can any shipped path reach this?",
  answered by enumeration, not mechanism-hunting. Reachable → blocks the tag; not reachable → file
  it named, don't block. (A12/A13's enumeration took an hour and settled the tag question — done
  on day two, after the mechanism hunt already cost a day.)
- **T1 — instruments are not gates.** `TestAllocFloor`, the A10 probes, the reservation/VRAM
  sweeps deliberately create device states no shipped path creates — move them out of the gated
  package (own package or build tag), results to `docs/measurements/`, not a gate verdict.
- **T3 — four gate outcomes, PARTLY DONE.** `parity-coverage-policy.md` already defines
  pass/fail/cannot-evaluate/first-run; only **fail** should block a tag — enforcement is what's
  missing (a correct decline on an 8GB card blocked a release once already).
- **T4 — re-tier by cost, this is B3 (above), promoted.** Measured wall-clock per tier, printed
  by the gate, not estimated.
- **T5 — the gate owns its environment.** A co-tenancy check at start (refuse with
  cannot-evaluate naming any process already holding device memory, rather than an ambiguous
  result) and a derived unmarked-drainer assertion (free VRAM recorded at test boundaries, fail
  if an unmarked test drains below a floor).

**Explicitly out of scope:** splitting the repo — the coupling that cost the two days was between
kinds of test inside one package, not between modules.

**Acceptance criteria are per-item in the design page** (e.g. T1: the gated package contains no
test that drives the device to refusal; T3: mutation-checked both ways — missing asset →
cannot-evaluate, wrong number → fail).

**B4 · Label or drop `stash@{0}`** — **REOPENED. Absent on `linux-62gb`, UNSEARCHED on
`macbook-arm64`.**

`git stash list` is empty in all four repos here. That is a result about **this box**, and closing
the item on it would repeat the exact distinction the SHA lint learned this turn: a search that could
not have seen the object does not report its absence. `stash@{0}` may only ever have existed on the
mac. **In the mac batch below.**

**macbook-arm64 searched (2026-08-12):** `git stash list` = **0** in `goinfer`, `aikit`, `wgpu`. The
original `stash@{0}` (the "item2 unload-close fix + tests (wip)") DID live here and was **backed up to
`~/goinfer-stash-backup/*.patch` then cleared** — preserved as patches, not lost; the mac stash is now
empty. So goinfer/aikit/wgpu are **closed on both boxes**. **One residue with an owner:** `cpubrrr` is
**not present on macbook-arm64** (could-not-search here), and `linux-62gb`'s "empty in all four repos"
did **not enumerate cpubrrr** — so its status is unconfirmed by name. **Owner `linux-62gb`:** run
`git stash list` in `cpubrrr` specifically and record it as searched (with count) or could-not-search.
That single confirmation closes B4.

**B8 · Position-keyed pins — audited 2026-08-12; the gates are clean, the PROSE is not**

`TestDispatchCensus` went red on a pure line shift (the fused gate+up guard in `decoder/mlp.go`, site
unchanged) and was
re-keyed on trimmed line content. Everything else pinned this campaign was swept for the same
property:

| pin | keyed on | positional? |
|---|---|---|
| `TestDispatchCensus` | trimmed line content | **no** — re-keyed `3d6ae1e` |
| `TestKernelLocalMemoryCensus` | kernel **name** (`moe_route`, `rope_kv`…) | no |
| `TestMoERouteFirstLaunchReservation` | byte value (138,412,032) | no |
| `TestMoERouteDemandThreshold` | byte values (286,916,608 / 289,013,760) | no |
| `TestSlotAllocation_matchesGranularityForm` | measured strides + quantum | no |
| `TestSlotCapArithmetic` | measured free/strides | no |
| `TestInt4_forwardParity` | fixture **name** → recorded metrics | no |
| `applySoftcap` threshold | size value | no |
| queue SHA lint index | sha → **subject** (content) | no |

**No gate remains position-keyed.** The residual surface is **14 `file:line` citations in this file's
prose**, which no lint covers and which drift silently. Already stale, checked:

- `cuda/backend.go:1144` — cited as `allocSlots`'s call site; now points at a bare `//` (A9-FIX
  inserted the warm-up above it).
- `cuda/resident.go:274` — cited for audit C-08's `_ = gpu.Upload`; now a comment about backend locals.
- two citations were **unresolvable**, because they omitted the repo — an aikit `linalg/quant.go`
  line and a bare `decoder/weightmat.go` one. Both
  are aikit paths written as if they were local ones; the SHA lint learned this distinction for
  commits and the same gap exists for paths.

**FIXED in the same change as the lint that found them** (`scripts/queue_citation_lint.py`), because
a lint landing red on its first run is a lint nobody adopts. The stale `allocSlots` call-site
line was corrected (it had drifted when A9-FIX inserted the warm-up above it);
the two bare `decoder/weightmat.go` / `decoder/mlp.go` references repo-qualified or de-numbered; the
`linalg/quant.go` reference resolves in aikit once the lint searches the sibling set (line 113 at the
time — the scalar `int8→f32` widen loop this citation was making the point about; that code is gone,
replaced by the SIMD widen at `linalg/quant.go:138` once aikit v1.18.0/P2 landed and goinfer bumped
to v1.19.0, 2026-08-15 — retargeted so the citation still resolves).

**And one turned out not to be a line drift at all.** `cuda/resident.go:274` was cited for audit
C-08 — `_ = gpu.Upload(...)` discarding errors. That code is **gone**: `recordUpload` captures the
first error into `r.setupErr` and the build declines gracefully. The citation was stale because the
CLAIM was stale, and F2 had been listing a fixed critical as open. A line-number check would have
reported a shift; the content check reported that the file no longer supports what was said about
it, which is the difference worth having.

**B7 · Off-origin work — swept, 2026-08-12** — `linux` for the local half, `mac` for the rest

Branches with no upstream, across all four repos on this box:

| repo | branch | unique commits | action |
|---|---|---|---|
| goinfer | `test/strengthen-mamba-deltanet-goldens` | **1** (`98936cf` strengthen mamba-2 + deltanet parity fixtures) | **PUSHED** |
| goinfer | `task/gemma4-moe-phase1a` | 0 — fully merged | leave; delete when convenient |
| aikit | `decoder-m2-tokenizer` | 0 — fully merged | leave; delete when convenient |
| wgpu, goduct | none | — | — |

Stashes: **none, in any of the four.**

**MAC BATCH — one session, not three interruptions.** Collected because each item needs that machine
and none needs this one:

1. **C3, the Metal consumer window — FIRST.** The largest completely uncovered surface, and it sank
   once already. Ordering it behind the two chores below is how that happened; a session that runs
   out of time should lose a chore, not this.
2. **Push `metal-rope-merge`** so `d682315` resolves from anywhere and P4's "already implemented,
   snapshot-golden byte-exact" becomes checkable. It does not need merging to be safe.
3. **B4's stash check** — `git stash list` in all four repos; the stash is absent here and unsearched
   there.
4. **arm64 f32 goldens read — TAG-GATE (NEW, created 2026-08-12). Minutes.**

   **WHY IT IS OWED — primary reason: the correctness argument has per-architecture branches, and
   only the stronger one has been exercised.** aikit's comment on the rework justifies bit-identity
   **per architecture** (`linalg/matmul_blocked.go`), and the two branches are not equally strong:

   - **amd64** — `dotFMA8` already reduces in-register, so the removed round trip was "32 adds of
     which 24 added literal `0.0`". Adding `0.0` is exact in IEEE-754: **structural**, it cannot move
     a bit whatever the inputs.
   - **arm64** — "the four lanes per column are **real partial sums** and `dot8ColsInto` folds them
     in **this same left-to-right order**". An **ordering claim about the new implementation**, not a
     structural impossibility. f32 addition is not associative, so if the fold order differs anywhere,
     the sums move.

   **goinfer's green goldens ran on amd64 and therefore exercised the STRUCTURAL branch. The weaker
   branch is the one nothing has tested.** That is what this gate closes, and it is why the argmax
   margin needs re-confirming on arm64 rather than being a separate claim about byte-agreement.

   **Secondary reason (independent, and it also holds): the refresh's arch was never recorded.**

   **Provenance — created, not overlooked.** This gate came into existence on 2026-08-12, the moment
   the architecture-exception clause met the aikit **v1.17.0 f32 blocked-matmul rework** (an
   expression-rewrite to a float path, still live in v1.17.1). It did **not** exist before that rework,
   so any earlier search that looked and found nothing **searched correctly — there was nothing to
   find.** Do not record this as a pre-existing gate someone skipped; that distinction is what keeps
   the search trustworthy next time. (The check on `2e8dfb6` — its 19 f32 rows carry no arch, the
   trailer didn't stamp one, git notes are empty, the manifest `machine` field is the preserved *T3*
   machine not the refresh's — asked whether the v1.17.1 refresh *incidentally* discharged the new
   gate. It didn't: the arch isn't recorded and every pointer, incl. today's box refreshes and the
   18/23 `linux-62gb` validation, points to amd64. So the gate is **open**, never yet run on arm64.)

   **What a green PROVES — written here so a green is not over-read.** The f32 goldens are
   **argmax + cosine, not bit-identity.** A green therefore does **NOT** show byte-agreement across
   arm64/amd64 — that cross-arch divergence is real, expected, and decision-irrelevant
   (`parity-coverage-policy.md` "arch-scoped"). What it proves is narrower and exact: **the argmax
   margin survives the summation-order change on the architecture that contracts `x*y+z`** (arm64 fuses
   FMA; amd64's baseline does not). The FMA campaign's **114,431× headroom was measured for the code as
   written**; the rework **changed the summation order**, so that headroom is no longer known to hold.
   Re-confirming it on the fusing arch is the **entire content** of this gate — nothing more, nothing
   less.

   **What a red MEANS — pre-registered, before it can be argued after.** A failure is **the headroom
   collapsing** — the reordered summation pushed a decision across the ~2×10⁻⁵ argmax tolerance on
   arm64 — **not a flaky fixture.** A red is a real numeric finding about the rework and is treated as
   one; it does not get waved off as fixture noise after the fact.

   **How.** Run `scripts/refresh_parity_hashes.sh` (or the f32 forward goldens) on `macbook-arm64`; the
   new `arch=arm64` trailer records the discharge. Second tag-gate alongside the prefill measurement —
   both attach to the same aikit-bump change.

   **DISCHARGED 2026-08-13 on `macbook-arm64` (`f8c4777`). Evidence, not inference.**

   First run (`53a96f6`) was PARTLY discharged — 8 f32, the Mac lacking the gitignored fixtures. The
   fix was NOT "regenerate on arm64" (my earlier claim, **wrong — I inferred it**): the box's 19 f32 run
   on **tiny synthetic fixtures** (`torch.manual_seed(0)`, "sub-second, no download") — deterministic,
   arch-independent data files, ~38M. Nothing ties them to the generating machine; the gitignore keeps
   them out of the *repo*, not off other *machines*. So I **rsync'd the 14 the Mac lacked from the box
   (~38M, minutes)** and re-ran:
   - **arch stamp present** in the proof block (*"forward goldens green at f8c4777 on arch=arm64"*) and
     the trailer (`Deps-Hash-Refresh: f8c4777 goldens=22 arch=arm64`).
   - **Composition: 22 passed / 0 failed / 20 skipped.** Of the 22, 3 quantized → **19 f32 rows green on
     arm64 — equal to the box's 19.** Up from 8. The argmax margin survives the v1.17.0 summation-order
     change across all 19 (Cohere×2, Gemma4 dense×2 / MoE×3 / logit×2, LoRA, Mixtral, Deepseek, Granite,
     Llama4, Nemotron, Phi3, Glm4Moe, Kimi, Gemma3VL, Qwen25VL, Qwen35). **No headroom collapse.**
   - **0 deps_hash lines changed** (both runs): the arch-independence claim holds on arm64. The "hash
     moves on arm64" finding did not trigger.
   - **Residual is arch-INDEPENDENT, not an arm64 gap.** The still-skipped f32 families need **real
     checkpoints absent on BOTH machines** — `qwen2.5-0.5b`, `qwen3-1.7b`, `tinymistral-248m`, `gpt2`,
     `llama-3.2-1b`, `gemma-3-270m` (HF downloads) — plus `tiny-qwen2-moe` (transferred, but its test
     has a secondary file check that still skips; worth a look, 1 family). The box's f32 run **could not
     cover these either**, so they are not a discharge gap for this gate — they are the general
     "we don't keep real checkpoints" tier, equal on amd64. Gate discharged to the box's f32 standard.
   - **Nothing committed** — 0 hash Δ → no refresh commit; this entry is the record.
**Still outstanding, and it needs the mac:** `metal-rope-merge` carrying `d682315`. It is not on
origin and resolves in no clone here, so **P4's "already implemented, snapshot-golden byte-exact" is
unverifiable from any machine but that one**. Pushing the branch is enough — it does not need merging
to make the claim checkable.

**B4 (original) · Label or drop `stash@{0}`** — superseded

"item2 unload-close fix + tests (wip)", a +32 hunk in the admin-unload source file. That filename
resolves in no repo here, so the stash cannot be reconstructed from its description either. Almost
certainly adds `Close()` to the
admin unload path — the change that converts a bounded leak into a use-after-free through the
`pick()`-to-`enter()` window. The safe version needs the drain, which is now implemented
(`588052b`), so this stash may simply be superseded. Either retitle it to say what it is, or drop
it. **An untitled WIP that looks ready is a trap.**

**B6 · Sibling-drift enumeration** — either box

A check that fails when one member of a sibling pair carries a fix or invariant the other does not.
The class is written up in `docs/parity-coverage-policy.md` ("Sibling drift"); this is the executable
half.

Known instances: **W8A8 / W4A8 `Workspace`**; **dense `mlp` / `moeMLP` `decodeScratch`**; **batched
GEMV int8 / int4**; ~~**`capSlots` and its inline copy in `allocSlots`**~~ (closed by A5 `6091e7a` —
`allocSlots` now calls `capSlots`); **SIMD / scalar widen**; and **the final-logit softcap, five
members**.

The softcap set is the largest found so far and is worth writing out, because P3 named exactly one
of them:

| site | status |
|---|---|
| `cuda/resident.go` (decode) | **shares `applySoftcap`** (`4c26a58`) |
| `cuda/prefill.go` | **shares `applySoftcap`** (`4c26a58`) |
| `decoder/forwardn.go:1047` | unchanged (softcap logic itself; line shifted again by later edits elsewhere in the file, retargeted 2026-08-24; previously retargeted 2026-08-15 after P1's edit) — `decoder/` core changes ride the goldens-proof requirement, not a version-gated freeze |
| `decoder/model.go:729` | unchanged — same freeze |
| `metal/model.go:1048` | unchanged — Metal is on hold for core-numerics surfaces |

The three unchanged members are a **deliberate** partial fix, not an oversight, and they are the
reason this row exists: had P3 been taken at face value and only `cuda/resident.go` parallelised,
even the second CUDA site would have drifted from it. Adopting `applySoftcap` (or its equivalent) at
the remaining three is the work that closes the row, and it unblocks with the freeze.

**Enumerate the members; do not name one.** A test that names one member is exactly what the passing
sibling already had — it reproduces the class rather than closing it. Where enumeration cannot be
mechanical, the invariant's own comment carries the full set, so the next fix is written by someone
who has been told the set exists.

**B17 · CI staticcheck verdicts are toolchain-nondeterministic** — either box. Filed 2026-08-14,
decider Francis.

`ci.yml` runs `go run honnef.co/go/tools/cmd/staticcheck@2025.1.1` (root and backend jobs). The
version pin does not pin the verdict: staticcheck's checks track the Go toolchain it is *built*
with, and `go run @version` builds it with whatever toolchain the box or runner has. aikit
diagnosed exactly this on 2026-08-13 (its `docs/task-gpu-lint-staticcheck-drift.md`): identical
linter version, Mac go1.26.1 vs Linux go1.26.5 → ST1000/ST1023 findings on one box only, green
elsewhere. Consequence here: a box can be red where CI is green (or the reverse) at the same
pinned version — the local-vs-CI relationship B0 exists to keep declared. Remedy shape per aikit's
task doc: make the linter's build toolchain part of the pin (pinned-toolchain build or a prebuilt
binary), so the verdict is a function of the pin. The "Pinned like fin/ken" comment (`ci.yml`)
marks where else the same class may live.

### C. Verification surfaces never exercised

**G25 · The idle guard is now written twice, in two languages, from two independent
rediscoveries** — unclaimed. Filed 2026-08-28.

Within one day the same guard was arrived at twice, by different work, without either knowing about
the other: `gate_cell_idle()` in `scripts/bench_peer.py:407` (re-check before every cell, refuse on
timeout), and a `settle()` in the snapshot-cost driver on the `linux` box. Both started as
check-once-at-start, both were found insufficient the same way, and **both converged on the same
non-obvious rule: refuse rather than proceed.**

That rule is the part worth centralising, because it is counter-intuitive enough to be re-litigated
by whoever writes the third one. A loop that waits and then measures anyway does not fail loudly —
it launders a contaminated run into a plausible-looking one. Measured both times: the contaminated
snapshot-cost run gave 0.544% against the clean 0.531–0.537%, close enough to have been believed and
published. **Contamination that produces obviously-wrong numbers is harmless; contamination that
produces plausible numbers is the dangerous kind, and only the second kind survives review.**

Shape, if taken: one helper carrying (a) the loadavg cap and its env override, (b) wait-then-REFUSE
rather than wait-then-proceed, (c) the per-cell re-check rather than start-only, and (d) recording
the observed loadavg into the artifact regardless of outcome — the recording is what caught the
2026-08-27 contamination that the start-only gate missed. `scripts/` has no shared library module
today, so this is partly a question of where such a thing lives.

**Do not fold in the goinfer-side settle loops without checking they want the same policy.** A gate
that refuses is right for a measurement; it is not obviously right for a long-running sweep where
refusing discards hours of completed cells. `bench_peer.py` is resumable and can afford to refuse;
that is a property of that harness, not a universal one.

**G23 · The T3 cosine bar is calibrated for int8 and there is now an int4 T3 under it** —
unclaimed. Filed 2026-08-26 from `qwen3_next`'s first real-checkpoint run.

`decoder/real_oracle_test.go` gates every `real-model-oracle` family on one hardcoded threshold,
and the comment beside it names its population: `if cos < 0.99 { // int8 W8A8 vs bf16 — same bar as
the deepseek real gates }`. Every family under it so far has been int8.

`qwen3_next` cannot be. 80B at int8 is ~80 GB against 62 GB of RAM, so its precision is **forced by
capacity**; it ran int4 weights / f32 activations and measured **0.989876** — 0.000124 under a bar
set for a strictly gentler quantization, with **argmax and all six continuation tokens exact**. The
evidence that this is quant noise and not a defect is in
`docs/measurements/qwen3next-t3-int4-2026-08-26.md`: the same checkpoint through the same loader at
**f32** scores cosine **1.00000000**.

**The question is what the int4 bar should be, and it must not be answered by this family alone.**
One data point, from the family that benefits from a looser answer, is how a threshold gets set to
wherever the number landed. Candidates for a second: `nemotron3nano` (already characterised at two
precisions, 0.978 int8int8 vs 0.9977 int8) and any family that fits int4 *and* int8 so both can be
measured on the same weights — that comparison is what actually calibrates the offset, rather than
inferring it.

**A second defect this exposed, and it belongs with the first.** `emitParityRow` is a no-op when the
gate fails, so a run that misses the bar records **nothing** — right for a failing gate, wrong for a
*calibration* miss, where the measured number is exactly what the next decision needs. As it stands
the only trace of a 0.989876 is a log file somebody has to know to look for. Whoever sets the bar
should decide whether a below-bar run should record a row marked as such.

**G21 · The bench-disk rule is enforced in the two shell/python harnesses and NOT in `cmd/gate`** —
unclaimed. Filed 2026-08-26 when the rule was consolidated.

`scripts/bench_peer.py` and `scripts/bench_compare.sh` now REFUSE a checkpoint that resolves under
`/srv/models` or `/Volumes/` (realpath, so a symlink out of `~/models` is caught). `cmd/gate` reads
`GOINFER_GATE_MODELS` in two places — `cmd/gate/configs.go:14` and `cmd/gate/gpu.go:365`, both defaulting to
`$HOME/models` — and accepts whatever it is given.

Correctness tests do not care what disk they read from, so this is not urgent. The throughput tests
in the same packages do: `TestProdThroughput`, `TestE2EDecodeThroughput_synthetic`,
`TestDecodeDepthThroughput` and the bandwidth tests all report rates, and a rate measured off a
5400 rpm SMR disk is wrong without being an error. **Whoever takes this should check first whether
those tests are actually disk-sensitive** — a fully-resident decode may touch the checkpoint only at
load — and enforce only where a number depends on it. An unnecessary guard on a correctness runner
is worse than none, because it makes the rule look arbitrary.

Related: the rule's prose was consolidated the same day (`CLAUDE.md` states it and names both
roots; `docs/benchmarks.md` § "Model storage" is the authority). It had lived in three places and
was complete in one — the other two named only `/Volumes/`, which does not exist on Linux, while
`/srv/models` is a local mount on the box that measures every CUDA row.

**G20 · `TestMoERouteDemandThreshold`'s WARM branch has been unreachable since 2026-08-21** —
unclaimed. Found 2026-08-26 while re-deriving the pins for P16's re-anchor; recorded rather than
fixed in the same change, because changing what a gate asserts and re-anchoring its constants in one
commit makes neither reviewable.

The discriminator is `warm := firstPass.freeBefore < pinnedDeviceFloor`. In the warm regime the
demand IS the residual, so that condition can only hold when **the floor exceeds the residual**.
That was true when the floor was 151,191,552 (> 138,412,032 residual) and has been false ever
since — 54,263,808 from 2026-08-21, and 1,769,472 from the 2026-08-26 re-anchor. So:

- the WARM branch is now dead code, and
- a warm run does not skip the check, it takes the COLD branch and goes **red**, reporting a broken
  demand identity when nothing is broken.

It has never fired because the drain group always runs this test cold, in its own process. That is
the shape this section exists for: not a check that cannot fail, but a check whose *other* branch
cannot be reached, so the gate silently covers one regime while claiming two. It is also the same
class as `TestMoERouteFirstLaunchReservation`'s unstated must-run-first precondition (A12 resolution,
`e682eb2`) — a test whose correctness depends on process state that nothing asserts.

**Do not "fix" it by widening the comparison until it passes.** The question is what a warm regime
actually costs now that the floor is 1.7 MB, which is an empirical question needing a warm
measurement — run the launch with another CUDA context already alive and see what `freeBefore`
reads. If the warm and cold demands are no longer distinguishable at this floor, the honest fix may
be to delete the regime split and say so, not to repair it.

**D1 · Trace tap and the launch-site coverage table** — `linux`, before D2's migration

For each of the 48 launch sites, two columns: which traces observed it, **and is it covered by any
asserting gate**. The trace proves the migration is faithful, not that the pre-migration mapping was
correct — so any site with no asserting gate carries its current mapping across unverified.
`moe_route` is one known member; enumerate the rest before migrating.

Run traces across sequential prefill, batched prefill, decode, an MoE model, and a partial-rotary
model.

**D2 · Launch-wrapper commit 1, then the migration** — `linux`

Design approved and fully specified. Positional parameters, one generated named type per
**(parameter name, C type)** — 13 names carry more than one C type, so name-only keying is
ambiguous. Buffers typed too. Returns `[]KernelArg` rather than launching, so `r.launch`'s sticky-
error accumulator stays intact and `cuda/prefill.go` and tests share the wrappers.

Extraction: `cuda/internal/gen` parsing `__global__ void NAME(params)`, comments stripped before
splitting, **hard-failing on any parameter outside the closed 9-type table — never skipping**.

aikit's two shipped GEMVs (`gemv_w4a8_fwd`, `gemv_w8a8_fwd`) are **derived, not transcribed** —
aikit ships `gemv_quant.cu` inside the module, resolvable via `go list -m`. The generated header
records aikit's module version; the diff test distinguishes "vendored, `.cu` legitimately absent →
counted skip" from "module dir resolves, file gone → loud failure".

Three gates: byte-compare regenerated against committed; coverage (every `.visible .entry` in
embedded PTX has a wrapper); and the PTX cross-check as a standing assertion, since `glue.ptx`
proved `.cu` and `.ptx` can diverge for months. Plus a second lint: **every generated wrapper is
called at least once from non-test code**, or is on an explicit test-only list — that closes
embedded-but-never-bound, the last uncovered member of the dead-code family.

Commit 1 changes no call sites and must be provably inert. Then migrate per file (`cuda/resident.go` 36,
`cuda/prefill.go` 11, testhooks 1) with a trace comparison at each step.

State the **641 → 0** figure in the commit message with its limit: zero counts cross-name
transpositions the type system prevents; passing a wrong *value* of the right kind still compiles.
The failure moves from an invisible positional slip to a legible mis-assertion at the call site.
**Do not write "eliminates transposition bugs".**

**E7 · No Python in the repo by v1.0** — `linux`/`mac`, **INVENTORY DONE; no migrations until after the v0.13.0 tag (§C1 + CUDA gate first).** Decided 2026-08-12 by Francis.

**Scope of the decision:** the repo contains no Python by v1.0. Analysis moves to Go tests; shell is
minimized to process orchestration. **The reference-tensor / pin-fixture generation is OUT (item 7),
blocked on Francis's torch-replacement research — do not attempt it or design around a guess at it.**

**Inventory (67 tracked Python files, 6843 lines), split by the only axis that matters here — does it need an ecosystem Go cannot reach:**

- **57 scripts import torch / transformers / safetensors / numpy** → the reference-tensor surface. **OUT OF SCOPE** (item 7). ~5000 lines: the whole `scripts/pin_*` family plus the torch oracles, golden/ref generators, and analysis probes (kda-oracle, gptoss oracle/golden, chat/tool golden gen, eagle parity ref, mxfp4 extract, gemma4 recon / scale-probe / 12b-trace, and similar).
- **10 stdlib-only scripts** → the migratable set (~1827 lines). Ranked by **load-bearing × easy** (first migrations cut the most risk per hour):

| # | script | lines | what it decides/produces | CI/gate dep today | difficulty | rank rationale |
|---|---|---|---|---|---|---|
| 1 | `scripts/skip_census.py` | 174 | PASS/SKIP/FAIL census, SKIPs bucketed by reason (release ritual) | release ritual | **stdlib Go** — and *strictly better*: reads `go test -json` structured events instead of scraping text | load-bearing × easy, and the Go version is a *reader* not a parser. **Migrate first.** |
| 2 | `scripts/sweep_composition.py` | 167 | prints the parity-sweep coverage composition along family×quant×loader | `scripts/parity_sweep.sh` + `scripts/refresh_parity_hashes.sh` | **stdlib Go** (grep test source for the quant literal + `go test -json`) | release-gate axis print; high load-bearing, easy |
| 3 | `scripts/ci_checks.py` | 108 | DERIVES CI's hygiene check-set from the CI workflow so the gate can't drift from CI | `scripts/gpu_gate.sh` | **stdlib Go, probably** — it already uses `re` on the workflow, no YAML lib. **Check the actual workflow shape reads with stdlib BEFORE reaching for YAML.** If a real parser is needed → `tools/` module, not main `go.mod` (item 4). | high load-bearing, medium |
| 4 | `scripts/selector_coverage.py` | 117 | tests that EXIST vs tests any selector RUNS (the difference) | census (not a hard gate) | **stdlib Go** | medium load-bearing, easy |
| 5 | `scripts/queue_citation_lint.py` | 773 | validates the queue's commit-SHA + file:line citations | **CI** (the only Python in CI) | **stdlib Go, HARD** — git via `os/exec`, module-cache path resolution against the required aikit version, orphan/reachability, generated index. No external ecosystem, but 773 lines. | **highest load-bearing, lowest easy** — the capstone: do it after the easy wins prove the pattern; it is what actually removes Python from the CI critical path. |
| 6 | `scripts/bench_peer.py` | 229 | goinfer-vs-peer decode A/B over both HTTP servers | `scripts/bench_compare.sh` | **stdlib Go** (`net/http`, `os/exec`) but larger | low load-bearing (benchmark, not a gate) |
| — | `scripts/chatml_tiny_fixture.py` (97), `scripts/diff_gemma4_12b.py` (53), `scripts/bench_prompts_calibrate.py` (41) | | fixture-gen / debug-diff / bench-prompt calibration | none | stdlib | low priority tail — opportunistic |

**Separate category — build tooling / FFI: `cuda/nvrtc_compile.py` (68) — DECIDED 2026-08-12 (Francis): "no Python" COVERS it; do it, LAST, low priority.** It compiles a `.cu` to PTX via **NVRTC** (invoked by `cuda/build_ptx.sh` → `scripts/gpu_gate.sh`); the runtime never calls it (goinfer ships committed PTX, loaded via gocudrv), so it fires only when a `.cu` changes, on the GPU-gate path — **not** always-on CI. Replacement:

- **Mechanism: a purego NVRTC binding**, ~6 C-ABI functions (`nvrtcCreateProgram` / `nvrtcCompileProgram` / `nvrtcGetPTXSize` / `nvrtcGetPTX` / `nvrtcGetProgramLog` / `nvrtcDestroyProgram`), same shape as the Metal binding. **purego is already in the cuda/metal dependency set → no new ecosystem.** A `cuda/cmd/ptxgen` (run from `cuda/build_ptx.sh` / `go:generate`) reads the `.cu`, passes the arch options (`--gpu-architecture=compute_75` for Turing, etc.), writes the `.ptx`. Build tooling only — **not** in the main `go.mod` (item-4 constraint holds by construction).
- **Load-bearing constraint (from the split-KV / cap-bump PTX discipline):** `ptxgen` must dlopen the **PINNED** `libnvrtc.so` (explicit path, e.g. 12.6.85 — not "whatever's on the box"), or a regen is a toolchain bump masquerading as a kernel change. Acceptance = the relay's byte-identical control: the Go tool must reproduce **every currently-committed `.ptx` byte-for-byte** from the current `.cu` at the pinned version before the Python is deleted (criterion **a**), and rebuild-unchanged → identical sha.
- **Honest limit:** removes the *Python*, not the *NVRTC dependency* — someone must still compile `.cu`→PTX, which irreducibly needs libnvrtc at regen time (the Python helper needs it too; no new burden). "No Python," not "no CUDA build dep."
- **Priority: LAST E7 item**, after the `queue_citation_lint` capstone proves the pattern — nvrtc is hardware-gate-path and rarely invoked, unlike the CI-critical citation lint. Independent of the `oracle/` plan (`docs/task-oracle-refforward.md`); shares nothing with it. Deletes `cuda/nvrtc_compile.py` in the landing commit (criterion **c**).

**Acceptance criterion per migration (non-negotiable, from Francis):**
- **a.** Run the Python and the Go against the current tree; outputs must **agree**. Any disagreement is **investigated before the swap**, not explained after.
- **b.** Mutation-check the Go both ways: introduce the defect it exists to catch → assert RED; remove it → assert GREEN.
- **c.** **Delete the Python in the same commit that lands the Go** — the two never coexist as sources of truth (B7's constant shape, applied).
- **d.** The **scope line survives**: whatever the Python printed about what it did and did NOT validate, the Go prints too.

**Dependency constraint:** **no tooling dependency in the main module's `go.mod`.** Stdlib-only where possible; a separate **`tools/` module with its own `go.mod`** where a parser is genuinely needed (the ci-checks YAML question is the only candidate). A consumer's module graph must not grow because a lint changed language.

**Shell — minimize, harden what stays (item 6 audit, 9 shell scripts):**
- **Keep (orchestration):** `scripts/parity_sweep.sh`, `scripts/gpu_gate.sh`, `scripts/heavy_gate.sh`, `cuda/build_ptx.sh`, and the two demo asset-build scripts under `demo/chat/` and `demo/agent/`. **Move (reads tree + decides):** `scripts/mutation_check.sh` MOVED 2026-08-21 (`gate mutation`, script deleted); the deciding half of `scripts/refresh_parity_hashes.sh` remains a candidate; `scripts/bench_compare.sh` folds into the bench-peer Go successor.
- **Rule audit (the four rules):**
  - **Rule 3 (`set -euo pipefail`):** `scripts/refresh_parity_hashes.sh` and the three build scripts compliant. **Violations:** `scripts/bench_compare.sh` (`set -u` only) and `scripts/mutation_check.sh` (`set -u` only) — add `pipefail`. *(mutation_check.sh was deleted 2026-08-21 by E8; its finding died with it rather than being fixed, which is what the do-not-harden-the-condemned rule predicted.)* The gate scripts (`scripts/gpu_gate.sh`, `scripts/heavy_gate.sh` are `set -u`; `scripts/parity_sweep.sh` is `set -uo pipefail`) **omit `-e` DELIBERATELY** — they run N families/packages and tally, and `-e` would abort on the first failure and lose the tally. The real requirement there is per-command rc / `PIPESTATUS` capture, which they already do; **do not blanket-add `-e` to a tallying gate.** Document the reason inline so it is not "fixed" into a regression.
  - **Rules 1/2/4:** no violations found in a targeted pass. The scripts show awareness — the mutation checker's header explicitly records fixing the `command -v staticcheck && staticcheck` (rule 1+4) anti-pattern (that header now lives in `cmd/gate/mutation.go`); `scripts/gpu_gate.sh`'s `command -v nvidia-smi` is backend *detection*, not a skipped check; the `grep -c … || true` counts guard the pipe's exit correctly. A full line-by-line pass on the keep-set is deferred with the migration.

**Item 7 — OUT OF SCOPE of E7's migration, owner named:** the pin-fixture and reference-tensor generation (the 57 torch/HF scripts). **Francis owns it.** Do not start it in parallel and do not design the tooling migration around a guess at what he finds.

**Replacement research: DELIVERED as a scoping plan — `docs/task-oracle-refforward.md`.** The design questions are resolved there (verdict: buildable; a pure-Go `oracle/` submodule with its own `go.mod`, independent safetensors reader + f64 math, anchored against HF once per architecture, emitting the existing golden schema — Python shrinks from ~50 per-model generators to a handful of per-architecture anchor runs, not to zero). It is a **plan, not a start**: still Francis's go/no-go to fund, and gated behind v0.13.0 (§C1 + CUDA gate) like the rest of E7. Its Phase 0 (cluster the 57 by shared forward-math to size the real kernel surface) is the first work if funded — the E7 inventory counted the scripts but did not cluster them by math.

**PARKED, 2026-08-18 (Francis):** the v0.13.0 freeze cleared, but Francis reviewed the plan and declined to fund it now — not rejected, may be picked up someday as a low-priority item, no active trigger. See the status line at the top of `docs/task-oracle-refforward.md` for the record. A future revisit should also fold in aikit's own ~23-file torch-oracle cluster (`aikit/scripts/oracle/*.py`) as one joint Phase-0 pass across both repos, not two separate plans.

**E8 · One Go gate-runner over `go test -json` — collapse the tallying shell + census Python** — `linux`/`mac`, **IN PROGRESS — runner + first two configs LANDED 2026-08-20.** Decided 2026-08-12 by Francis.

**Landed:** `cmd/gate` (stdlib-only, no module's graph grew) with the `census` and `heavy` configs; `scripts/skip_census.py` and `scripts/heavy_gate.sh` deleted in the same commit. Acceptance (a) proved by pointing BOTH programs at one captured `go test -json` stream — byte-identical output and identical rc in both modes (PASS 884 / SKIP 79 / FAIL 0 of 963; rc 1 under `GOINFER_REQUIRE_FIXTURES=1`). Acceptance (b) is in-tree (`cmd/gate/gate_test.go`), against the real toolchain over scratch modules.

**Three findings, recorded in `docs/task-gate-runner.md` §8:** (1) the two migrated scripts DISAGREED about whether subtests count — heavy_gate's column-0 grep counted top-level only, skip_census's JSON keying counted them all — so it is a per-config knob, not a house style; (2) keying results on `(Package, Test)` silently UNDERCOUNTS a matrix that runs one package under several tag sets, which is exactly what the two unmigrated gates do — caught by the tally-integrity mutation test, not by review; (3) `skip_census.py` exits 0 on a package-level failure (build error / native crash), preserved for acceptance (a) but now ANNOUNCED — **flipping it is a verdict change and therefore Francis's call.**

**Step 2 LANDED 2026-08-21:** `gate parity` + `gate composition` + `gate selector`, replacing `scripts/parity_sweep.sh`, `sweep_composition.py` and `selector_coverage.py`. The two censuses were a HARD dependency, not a convergence — both PARSED the shell script (`sweep_composition` regexped its `GATES=(…)` array; `selector_coverage` did the same for its selector), so deleting the sweep would have broken them outright; both now read the same Go gate list the sweep checks. The sweep is a CHECKSET, not a tally: **MISSING** (a named gate that produced no result at all) is an outcome a pass/skip/fail count cannot see, and it blocks. Acceptance (a): both sides run with `EMIT_MANIFEST=1` and the mutated files reverted between them — identical verdict (2 blockers, rc 1), all 50 per-gate rows identical, and the manifest+matrix mutation **byte-identical**; both censuses byte-identical.

**The sweep is RED on this tree, it is not E8's doing, and it is now BISECTED** — `TestQwen35GGUF_gate` / `_weightDiff` fail identically under the unmodified shell script. Culprit is **`6d4fc79`** ("qwen35 family: quantize the projections that were f32 at every quant"): parent `33879dd` PASSES at cosine min 0.99298 / argmax 69/80, `6d4fc79` FAILS at 0.98740 / 68/80. The Go 1.27 bump was the obvious suspect and is REFUTED — bit-identical 0.98740 at go 1.26.6 and go 1.27.0. Not an unnoticed regression but a deliberate bandwidth trade whose cost the commit measured on the SIBLING gate (`TestQwen35Real_gate2FullModel`, floor ≥0.98, re-baked green) while this one's hard-coded 0.992 floor, set at `2583a2b`, was never revisited. It shipped silently because the gate is `realckpt`+heavy: CI never builds it and only the release sweep runs it. **Remedy is a judgement call, not taken:** see `docs/task-gate-runner.md` §9.

**Step 3 LANDED 2026-08-21 — E8 IS STRUCTURALLY COMPLETE:** `gate gpu` replaces `scripts/gpu_gate.sh` (715 lines, the largest of the six). Six scripts → one runner + configs, as the plan claimed. Acceptance (a) on the CUDA half, both sides running the FULL gate including the ~28min heavy tier: identical group accounting (9 declared → 9 reported), identical 9 pass / 1 skip / 2 fail, identical 12 per-check verdict lines, the same two failing tests, identical partition reconciliation, identical FAIL verdict. Two lines differ: the heavy tier's wall clock, and a RUN-line count where **the shell was right and my first version was wrong** — `go test -v` does not indent a subtest's `=== RUN` line, so `grep -cE '^=== RUN'` counts subtests while the `--- SKIP` grep beside it does not. Matched exactly (verified by re-parsing the saved -json stream, not by re-running the gate) and flagged: "ran 238 tests, skipped 22" mixes units.

**Two reds on this box, neither E8's, both reported identically by the script:** `TestQwen36_35B_cache` failed on a STALE ASSET (the 35B `.giw` was format version 2; the build reads 3..6) — **FIXED 2026-08-21** by regenerating from the Q8_0 GGUF (now GINFW v6, 26.4→22.1 GB, smaller because `6d4fc79` also int4s the projections the old bundle stored f32). That uncovered a second defect: the test's tokenizer step had no `.giw` arm although `.giw` is its own default path, so the default invocation could never reach its decode; every passing run had pointed `GOINFER_QWEN36_35B` at the `.gguf`. Both fixed — passes at 14.09 tok/s, 75.5% hit rate, and `TestMoERouteDemandThreshold` broke the demand identity — **DIAGNOSED AND FIXED 2026-08-21, and the gate's own conclusion was wrong.** It blamed the KERNEL; the kernel did not move. The value is bit-stable across four runs (so my "it is moving between runs" note was itself wrong — it came from comparing logs written by two different versions of the test), and it is IDENTICAL at `c6760d7`, the commit that recorded 141557760 and pinned against it. What moved is the DEVICE FLOOR: `TestAllocFloor` now measures 54263808, not 151191552, and the identity then closes to the byte (54263808 + 138412032 = 192675840). It accused the kernel because its comment claimed "the residual and the floor are each pinned by their own gate" — true of the residual, FALSE of the floor, since `TestAllocFloor` says outright "Not a threshold assertion: the number is the finding". An unpinned number another gate depends on does not stop being load-bearing; it stops being watched. Fixed by pinning the floor (to a window — it is a machine property) where the floor lives, and by rewriting the demand message to check components before blaming the kernel. **Safety unaffected in the safe direction:** a smaller floor = more headroom, margin clears by 332.2 MiB. Both pins mutation-checked with `gate mutation`.

**Metal half VERIFIED on the Mac 2026-08-21 — E8 is complete.** Same method (retired script extracted from git, both run against the CURRENT tree so the seven post-deletion `metal/` commits could not confound it), same standard: identical verdict (`FAIL … rc 1`, both failing on the same machine-local `c8b65ba` citation orphan, not a Metal fact), 7 declared → 7 reported, 6 pass / 2 skip / 1 fail, every verdict line matching **including the evidence lines**, skip-notes identical. Homebrew-bash dependency confirmed GONE (stock `/bin/bash` 3.2 runs the binary). **A FOURTH latent shell bug exposed:** the retired script's Darwin selector (its line 617) `MINE='-darwin$'` went to `grep -Eq "$MINE"` without `--`, so the leading dash parsed as FLAGS and every job misclassified as skip — the block printed `PASS 0 CI hygiene check(s) reproduced locally`, a green having run nothing, in the block written to stop exactly that. Invisible on Linux, whose pattern has no leading dash. The Go port reports the correct 8 pass / 14 skip (corroborated on the Linux box against `ci_checks.py`: 22 rows, 8 in `-darwin` jobs). **Loose end:** the INCONCLUSIVE verdict arm was read from source but never demonstrated live — every fresh-clone attempt hit the pre-existing Metal `fault 0x10` crash, 6 of 7 attempts (a rate well above the day's ~50% baseline, and itself worth noting). See `docs/task-gate-runner.md` §12. **`mutation_check.sh` MIGRATED 2026-08-21** as `gate mutation` — the last item §6 named. Agreement: four scratch scenarios (happy / vacuous-expression / already-red-baseline / gate-blind-to-the-mutation) byte-identical including exit codes AND the restored file, plus the documented real example (`int4-quantizer` over `decoder/weightmat.go` against `TestInt4_forwardParity`) identical and tree-restored on both sides.

Distinct from E7 (which migrates scripts one-for-one): **E8 recognizes that six scripts are one program.** The three tallying shell gates (`scripts/parity_sweep.sh`, `scripts/gpu_gate.sh`, `scripts/heavy_gate.sh`) and the three census Python scripts (`scripts/skip_census.py`, `scripts/sweep_composition.py`, `scripts/selector_coverage.py`) all run `go test -json` across a *package × family × quant × tag* matrix, tally PASS/SKIP/FAIL (SKIPs bucketed by reason), and decide — differing only in matrix and decision. **One runner (`cmd/gate`) + committed configs subsumes all six** (~6 scripts → 1 runner + configs).

Why Go is *strictly better* here, not just same-language — it dissolves the item-6 audit's footguns by construction: the deliberate-omit-`-e` tally tension vanishes (Go has no abort-on-error — capture `rc`, append, never lose the count); `PIPESTATUS` capture is direct via `os/exec`; the `command -v tool && tool` silent-skip (the one the mutation checker records fixing) can't recur (`exec.LookPath` miss is an explicit error). Backend/asset detection → SKIP-with-reason in Go, where it can't fail open.

**Stays shell (pure glue, per the decides-vs-orchestrates line):** `cuda/build_ptx.sh`, the two `demo/*` asset-build scripts, env-then-run-one-command wrappers. The runner **shells out to** `go test -json` (stdlib `os/exec`) — orchestrates it, doesn't reimplement it; **no new main-`go.mod` dependency.**

**Acceptance = E7's a–d verbatim** (agree-before-swap incl. the tally-integrity mutation case; delete the script in the same commit; the skip-reason scope line survives). **Sequencing note:** do **not** harden shell you're about to delete — the item-6 `pipefail` fixes for `scripts/bench_compare.sh` and `scripts/mutation_check.sh` were moot (both were on the migration list; mutation_check is now deleted); add `pipefail` only to the surviving glue shells.

**Full plan: `docs/task-gate-runner.md`** (matrix-config shape, core loop, hardware detection, order-within-E8, not-in-scope). E8 and E7's census migrations converge — the three census scripts fold in as configs as each matching gate lands, rather than being migrated twice.

**F2 · §2/§3 criticals — SWEPT 2026-08-12** — `linux`, **most were already fixed; two lack an anchor**

| finding | state | anchor |
|---|---|---|
| C-05 gemma-4 stride on snapshot restore | **fixed**, with a gate | `decoder/kvsnapshot_gemma4_test.go:10` |
| C-06 unvalidated tensor shapes | **fixed**, break-it-first gate | `decoder/serialize_shapecheck_test.go:15` |
| C-08 `_ = gpu.Upload` over zeroed weights | **fixed** — `recordUpload` → `setupErr` → graceful decline | `cuda/resident.go:499` |
| C-14 CUDA argmax has no index tie-break | **fixed** at `c6600fc`, gated | `cuda/argmax_tiebreak_test.go:19` |
| C-31 `make([]byte, u32)` unbounded | **fixed** — bounded against the remaining file size before the allocation | `internal/giw/bundle.go:114` |
| C-21 embeddings batch cap, un-queued | **fixed** — `checkEmbedInputBounds` caps the input count, gated at the boundary and at +1; the un-queued half is a *documented deliberate decision*, not an omission. The body-cap tests are a different concern (bytes, not count) — covered-by-something-else, which is why they did not answer this | `internal/serveapp/embeddings.go:26` |
| C-22 shutdown lock, swallowed second signal | **fixed**, with a named gate — the checkpoint cannot block forever on a busy model, and a second Ctrl-C always kills | `internal/serveapp/main.go:555` |
| C-30 no mutex in the paging paths | **fixed** — both pagers carry an internal mutex, each citing the audit finding | `decoder/layerpaging.go:42` |

**These are correctness and security items, so a wrong entry costs more here than in P or B — in both
directions.** Five listed as open were fixed, which wastes attention; and had any been listed as fixed
while open, the cost would have been the reverse and worse. That asymmetry is why every row above
carries an anchor now: **the lint keeps them honest without anyone re-reading the code.**

**Every listed critical is fixed.** The whole F group was stale.

**And two of the three "open" verdicts in the first pass of this sweep were MY search failing, not the
entries.** C-30 was recorded as "unverifiable — names a paging path that is not a file"; the files are
`decoder/layerpaging.go` and `decoder/moepaging.go`, and the glob used was `decoder/paging*.go`, which
could not have matched them. C-21/C-22 were recorded "unverified" after looking only at the body-cap
tests, which measure bytes where the finding is about counts.

That is **exactly the distinction the citation lint learned this turn** — a search that could not have
seen the target does not report absence — applied to commits and paths on the same day, and then not
applied to my own sweep of the F group. The recognition test is not "did I look" but **"could what I
ran have found it"**, and it has to be asked of prose sweeps, not only of tooling.

**F3 · G-01 class — confirm the sub-shapes landed** — `linux`, **status unconfirmed**

The class entry in `parity-coverage-policy.md` should carry **exercised but never triggered** —
`allocSlots` runs in every MoE test and caps in none of them, so a safety branch reads as fully
covered by every measure the project has. Recognition test: **does any test reach the branch**, not
just the function.

### G. New capability, scoped but not started

**G3 · Consumer-resolution gate — the aikit twin** — either box, **docs-only now; code after the
v0.13.0 tag (§C1 + CUDA gate first)**. Filed 2026-08-14, decider Francis.

Verify from a consumer's position that every published goinfer module resolves and builds through
the Go module proxy — the B-01…B-07 class ("the tagged submodules don't build for anyone",
2026-08-05 audit), which aikit hit independently on 2026-08-12: eight backend modules never tagged,
`require …/gpu v0.0.0` → 404 for any consumer, invisible in-tree because replaces + the gitignored
`go.work` mask it by construction. Nothing inside a repo can see this class; the manual
scratch-module step in `RELEASING.md` is the only current guard and runs at most at tag time.

The shape is built and proven in aikit (merged 2026-08-13, green on both real runner OSes):
`tools/gate` (matrix runner — one cell per check, ok / n-a / FAIL plus INCONCLUSIVE = exit 2, the
count never lost) + `tools/consumergate` (resolve tier: `go get` + `go mod download` per
module@version against the proxy only, no `,direct` fallback; compile tier: `go build module/...`,
wrong-OS modules n/a not FAIL) + a tag-triggered job and a scheduled two-OS watch whose compile
union covers what no single OS can. Port the shape, don't re-derive it. goinfer's set: five modules
(root, `gpu`, `cuda`, `metal`, `demo/agent`); tag patterns `v*`, `gpu/v*`, `cuda/v*`, `metal/v*`,
`demo/agent/v*`. `RELEASING.md`'s out-of-tree verification prose becomes an invocation of the tool,
same commit.

Scoping check before build: whether `ken` is multi-module — it is a promoted repo now, and if
multi-module it is the third instance of the class; the same tool shape covers it as its own task
there.

## Draft: contents of the next release

## B14 — the gate needs the FOURTH outcome implemented (policy filed, code pending)

`docs/parity-coverage-policy.md` now defines four T3 outcomes: pass / fail / cannot-evaluate /
**first-run**. `scripts/parity_sweep.sh` implements three — `classify()` returns PASS / FAIL / SKIP /
MISSING, and everything that is not PASS increments `blockers`.

**What has to be built.** A gate cannot know it is on its first run without a record of prior runs,
so first-run needs a **ledger of gates that have a confirmed prior result**. A gate that produced a
result and is absent from the ledger is `first-run`; the sweep prints it on its own line, counts it
separately, and does **not** increment `blockers`.

**The ledger is the "someone decided it is correct" record**, which is the point rather than an
implementation detail — a gate enters it by a human promoting an observed value to a baseline, never
by the sweep observing itself. Auto-promotion would turn "never checked" into "expected" in one
silent step.

### Ledger entry — the fields, fixed before it is built

Every entry carries all five. An entry missing any of them is not a confirmation, it is a note.

| field | why it is there |
|---|---|
| **gate name** | what the entry is about |
| **confirmed value** | the result a human declared correct — the baseline itself, not a pointer to one |
| **who promoted it** | a confirmation is an act by a person; an entry with no author is an assertion with nobody behind it |
| **when** | absolute date |
| **the commit it was confirmed against** | the state of the tree the human was looking at when they said "this is correct" |

### What the sweep reconciles, and what it reports

Three checks, every sweep, reported next to the counts:

| condition | outcome |
|---|---|
| **gate with no ledger entry** | **`first-run`** — the normal path into the new outcome |
| **ledger entry with no matching gate** | **stale — report and ignore.** Never fail on it: a removed or renamed gate is ordinary, and a ledger that blocks on its own leftovers trains people to delete entries |
| **entry whose gate's source has changed since the confirming commit** | **"confirmed before the gate last changed"** — printed as a **warning beside the counts** |

### Why the third check is the one that matters

**Renaming a gate reverts it to `first-run`, which is SAFE.** The name stops matching, the entry goes
stale, the gate has no confirmation, and it re-enters triage. Loud and harmless.

**A gate KEEPING its name while its assertion changes is not safe.** The ledger then compares a
confirmed value against **different semantics** and reports **pass** — a green built on a
confirmation of something else, with nothing in the name, the count, or the result to reveal it.

**This is the same staleness the citation lint's content keys catch, and it takes the same remedy.**
That lint deliberately does not key on line numbers: a citation survives a line move but breaks when
the cited *content* changes, because the content is what the claim was about. The ledger entry must
bind to the gate's **source content**, not merely to its name — so editing the assertion invalidates
the confirmation while reformatting around it does not.

**The warning does NOT block, and must never become one.** It exists so that a human promoting a
value can see **which confirmations have outlived the code they confirmed** — exactly the list you
want in front of you at the moment you are deciding what is correct, and one nobody can assemble
after the fact.

**IMPLEMENTED 2026-08-14** — `scripts/gate_ledger.py` + the `FAIL` branch of `parity_sweep.sh`.

| behaviour | verified |
|---|---|
| no ledger entry, gate exists | `FIRST-RUN` |
| entry present, source unchanged | `CONFIRMED` |
| entry present, **function body edited** | `SOURCE-CHANGED` |
| edit **elsewhere in the same file** | still `CONFIRMED` — the key is the function body, not the file |
| name in no `_test.go` at all | `UNKNOWN-GATE` |
| **no ledger file at all** | `CONFIRMED` |

**Two safety properties that were NOT in the original design and are the reason it is safe to ship:**

1. **An absent ledger means "we have no idea", not "nothing has ever run."** If a missing file
   produced `FIRST-RUN`, then deleting it — or merely shipping this before anyone seeded it — would
   make **every failing gate non-blocking**. That is a safety regression wearing a new outcome's
   clothes. The mechanism is **inert until the ledger is deliberately created**, and inert means the
   old behaviour: a failure blocks.
2. **`UNKNOWN-GATE` for a name that exists in no test file.** A first-run is a gate that *ran* and
   produced a result with no baseline. A name nobody can locate is a typo or a deleted test, and
   granting it first-run amnesty would hand a free pass to precisely the cases no one can inspect.
   Found by the end-to-end mutation check, not by design.

**Seeded:** 47 of the 48 required gates, from the 2026-08-13 sweep, marked `BULK-SEEDED` **per
entry** — because pretending a bulk import is 48 individual human judgements would be the
false-confirmation this ledger exists to prevent. Upgrade one at a time with
`gate_ledger.py promote`. The 48th, `TestW4A8DecodeParity`, is correctly `FIRST-RUN`: it has never
produced a result at all (its int8 `.giw` half has never been built).

**Reconciliation** runs every sweep and prints next to the counts: first-run, stale (reported and
ignored), incomplete entries, and *"confirmed before the gate last changed"* as a **warning that
does not block**.

## B16 — SHARED ASSET REGISTRY: preflight and gates now apply ONE predicate (IMPLEMENTED)

**Implemented 2026-08-14.** The sweep's preflight and every heavy gate each decided "is this asset
present" separately — the preflight in bash with `[ -e "$path" ]`, each gate inline with its own
default path and its own `os.Stat`. Two implementations, free to disagree, and they did.

**What the divergence actually cost, in three observed forms:**

1. **A directory satisfies `-e`.** Three preflight entries named a directory where the loader wanted
   the `.gguf` file inside it; preflight reported them RESOLVED. **Four gates were costed by that.**
2. **`GOINFER_QWEN35_GOLDEN`'s real requirement is a readable `manifest.json` INSIDE the directory.**
   `-e` on the directory cannot express that, so preflight said present and the gate skipped anyway.
3. **`GOINFER_PREQUANT_GGUF` had three different fallbacks across four call sites** — and at
   `loadInt4Model`, **none at all**, so that one gate skipped whenever the variable was unset while
   its three siblings fell back to `../testdata`. The same box ran different gates against different
   files, with nothing in any output naming the difference.

**The fix is not a better `-e`.** Presence is now DEFINED IN ONE PLACE — `testdata/assets.json`,
carrying `kind`, `members`, `members_any`, `min_bytes` and the candidate paths — and every consumer
applies that definition instead of approximating it.

| piece | what it is |
|---|---|
| `testdata/assets.json` | the registry: 10 assets, each with its predicate and its `used_by` gates |
| `scripts/asset_registry.py` | `preflight` / `check` / `list` / `verdicts` / `census` |
| `decoder/asset_registry_test.go` | `assetPath(t, env)` + `lookupAsset(env)`, and the registry's own gates |
| `scripts/parity_sweep.sh` | preflight now sources the registry; its own resolution table is deleted |

**TWO IMPLEMENTATIONS, COMPARED RATHER THAN TRUSTED.** Bash cannot read JSON and Go cannot be called
from the preflight, so the predicate necessarily exists twice. What makes that acceptable is
`TestAssetRegistry_agreesWithPreflight`, which asks the script for its verdict on all 10 and asserts
the Go side agrees — **and agrees on the same path**, since two sides can both report success about
different files.

**Verified by mutation, both directions:**

| mutation | expected | result |
|---|---|---|
| drop Go's is-a-directory check | disagreement | **neutralised by `IsRegular` downstream — tested nothing** |
| drop Go's is-a-directory *and* `IsRegular` checks | disagreement | ✅ `PREDICATES DISAGREE — python ok=false, Go ok=true` |
| two valid candidates, Go iterates them reversed | same verdict, different path | ✅ `both resolve, but to DIFFERENT paths` |
| re-introduce a direct `os.Getenv` of a registered var | drift gate red | ✅ names the file and the replacement |
| registry unreadable | preflight's loud branch | ✅ fires; the sweep says the blocker count is about the failure, not the tree |
| a `used_by` gate that does not exist | red | ✅ **caught a real one** — the entry named `TestMoonlightReal_gate`; the gate is `TestDeepseekMoonlightReal_gate` |

The first row is the one worth keeping: **a mutation that the code neutralises downstream proves
nothing**, and it looked like a passing check until the verdict was read.

**Also fixed, found while replacing the block:** the preflight's closing line read *"all 10 assets
resolved"* above a table of **nine** rows — a hardcoded count in the sentence announcing that
everything was fine. The count is derived from the registry now. `GOINFER_QWEN35_REAL` was never in
that table at all and is registered (bringing it to a genuine 10).

**THE DENOMINATOR, reported rather than implied.** `asset_registry.py census` classifies all **124**
`GOINFER_*` asset-candidate variables **by USE, not by name**: 10 registered, **34 path-shaped but
unregistered**, 51 switch/scalar, **29 UNCLASSIFIED**. Enforcement is scoped exactly to the
registered set; the rest is counted, not policed.

*The universe is `read-by-a-test OR registered`, not `read-by-a-test`* — and that correction was
itself forced by this change. Registering an asset REMOVES its `os.Getenv` from the test sources,
which is the entire point, so a universe of "variables read by tests" shrank by nine the moment the
conversions landed and the four buckets stopped summing to the total. **The numerator-vs-universe
defect, in the tool built to report it.** The census now asserts its buckets partition the universe,
so that cannot recur silently.

*Unclassified is a real outcome, not a leftover.* `GOINFER_SERVE_MODEL` is a path by any human
reading, but its value flows into a struct field and never touches a filesystem call — no local rule
can see it. Folding it into "switch" would be a bucket absorbing what the rule did not determine,
which is the census failure this line of work exists to stop. An earlier proximity rule ("any
filesystem call within N lines") called `GOINFER_HEAVY_TESTS` — a pure on/off switch — path-shaped;
following the bound identifier fixed that and exposed the third bucket.

**One deliberate behaviour change**, marked at the site: `loadInt4Model` previously skipped whenever
`GOINFER_PREQUANT_GGUF` was unset and now resolves through the candidate list like its three
siblings. Under the sweep nothing changes (the preflight exports the variable either way); under a
bare `go test ./decoder`, `TestDecodeParityInt4` now RUNS where it used to skip. Note that
**B13** records
this gate as **already red for at least two days** — so expect it red, and do not read that as caused
by this change. *(Resolved 2026-08-15 `8f63a7d`: red since 2026-06-14, a stale golden pinning a
superseded W4A8 kernel — so this behaviour change did not cause it, it REVEALED it, and the "two
days" was the visible floor. The gate is green now; `queue-correctness.md` carries the close-out.)*

**End-to-end proof:** `TestPhi3GGUFReal_gate` under `-tags realckpt` loads its model through
`assetPath` and passes (argmax 3681 = golden, continuation exact). **Note the tag**: the 24
`realckpt` files are not compiled by a bare `go vet ./decoder/`, so an earlier "PASS" of this gate
had in fact run **no tests at all**. Vet is clean under `realckpt`, `goinfer_testhooks`, `gpu` and
`race` as well as the default.

**Not done:** the 34 path-shaped unregistered variables. Registering them is the campaign, and it
overlaps **B13**
— the two lists should be reconciled before either is worked, since a dark test and an unregistered
asset are frequently the same test.

## B11 — TWO SERIALIZATION PATHS DIFFER BY 8 BYTES (FIXED)

**Fixed 2026-08-14.** The 8 bytes were `len("int8int8")`: `writeHeadGlobals` decided whether to
write the v5 quant-label field on `wr.sink == nil` — "are we the buffered writer" — as a proxy for
"do we have the real weight data yet". Those agree for the true incremental GGUF streaming
transcode (which writes the header on an all-zero `Layers` slice before any layer streams in — see
below for why that path genuinely can't resolve a label). They **disagree** for a caller that
already holds a fully-loaded `*Weights` and simply chooses the streaming API for its I/O shape:
`internal/prequant` and the qwen35 GGUF branch (a dedicated loader that fully materializes `w`,
*then* calls `SerializeWeightsTo`) both do exactly this, and both — along with this test — were
needlessly dropping a field the buffered path wrote.

**Fixed by testing the real thing:** `hasPopulatedLayers()` checks whether any body matmul weight
actually has `Rows() > 0`, and `writeHeadGlobals` gates the label on that instead of on which
writer is in use. `decoder/serialize.go:677` (`hasPopulatedLayers`).

**The other half mattered more than the byte count.** `quantLabel()`'s own "nothing matched" case
returns `"native"` — a real, valid quant mode, not an empty string. Measured directly on an
unpopulated `Weights{Layers: make([]LayerWeights, 4)}`: `quantLabel()` returns `"native"`. So the
naive fix (drop the `wr.sink == nil` check and always call `quantLabel()`) would have **baked a
false `"native"` label into every genuinely-streamed bundle** — trading an 8-byte omission for a
silently wrong claim in the header. `TestSerialize_unpopulatedLayersOmitsLabel` gates this
specifically, mutation-verified: removing the `hasPopulatedLayers()` guard reproduces exactly this
failure (`label field = "native" on an UNPOPULATED w`).

**Neither side was ever corrupt.** The old comment already documented the fallback: a v5 bundle
with an empty label field is read exactly like a pre-v5 bundle — `LoadSerializedWeights` leaves
`bakedQuant` empty and `Model.Quant()` falls back to re-deriving `quantLabel()` from the loaded
tensor kinds, the same function that would have written the label. So a streamed .giw missing the
optimization hint produces the identical answer at load time as one that has it — the label is
provably cosmetic, and its absence was never a data-loss risk. This resolves the open question
below without qualification, not by trusting the design intent but by tracing the actual fallback
path both ends use.

**Original entry, corrected in place, for the record:**

`TestSerializeWeightsTo_matchesBuffer` fails:

```
streamed length 632821543 != buffered 632821551
```

The assertion is `decoder/serialize_test.go:436`.

**632,821,551 − 632,821,543 = 8 bytes. One uint64.** On a ~633 MB payload that is not drift or a
rounding artifact — it is one field written by one path and not the other, or at a different width.

**CORRECTION — this is NOT a first-run.** The original entry claimed the gate "did not start
failing, it started running", on the belief that `GOINFER_PREQUANT_GGUF` had never been set. It had.
This test **failed identically on 2026-08-12**, in the very sweep C1a was discharged on. It is a
**known-bad of at least two days' standing**, not a newly-exposed unknown, and it is **not** caused
by anything in v0.13.0.

What still stands:

1. ~~There is no basis yet for saying which side is correct.~~ **RESOLVED above** — the streamed
   side was correct to omit the field where it genuinely lacked the data (the true transcode); it
   was wrong to omit it where it had the data in hand (this test, `internal/prequant`, qwen35).
2. ~~If the streamed path is the short one, anything it has written may be affected.~~
   **RESOLVED.** All five real `.giw` files on this box (`gemma4-26b-int4`, `GLM-4.5-Air-Q2_K.int4`,
   `mellum2.int4`, both `qwen3.6-35b-a3b-int4*`) were hand-parsed at the header: every one is
   `weights_v2`/`v3`/`v4` — **pre-v5**, so the label field this bug concerns doesn't exist in any
   of them at all. Not "unaffected because harmless" — unaffected because the field they'd need to
   have gotten wrong didn't exist yet when they were written. Two were also load-tested end to end
   post-fix (`mellum2.int4.giw` → `Quant()="int4mix"`, `gemma4-26b-int4.giw` → `Quant()="int4"`),
   both correct, both via the pre-v5 inference fallback exactly as before.

**Verification:** `TestSerializeWeightsTo_matchesBuffer` passes (632,821,551 bytes, byte-identical).
Both new/changed behaviors mutation-verified — reverting the `hasPopulatedLayers()` guard to the
old `wr.sink == nil` reproduces the original 8-byte failure exactly; removing the guard entirely
reproduces the `"native"`-mislabel hazard exactly. Full `decoder` serialize/giw test group green
(18 tests). `go build`, `go vet -tags goinfer_testhooks ./...` clean.

## B12 — the selector census could not see a gate nobody was running (FIXED)

**RE-FILED 2026-08-13** — destroyed by the same `--update` bug.

`scripts/selector_coverage.py` analysed gating **only for the 49 SELECTED tests**. The 286 "reached
by NONE" were printed as a bare count and never attributed, so an env-gated test that no selector
reaches was invisible *as* env-gated.

A **defect in the census, not a process failure**: nobody ignored a warning, because the information
was never produced. The count was accurate and told you nothing.

**Fixed:** the unselected bucket is now broken down by what *also* gates it, and the report ends with
the env vars gating otherwise-unreached tests — **42 of them**.

## B13 — 42 ENV-GATED TEST SETS THAT NOTHING SELECTS. One campaign, budgeted, AFTER the tag.

**RE-FILED 2026-08-13** — destroyed by the same `--update` bug. **DO NOT ENABLE ANY BEFORE THE TAG.**

**Rationale.** Enabling these converts a **bounded release into an unbounded investigation**: each
failure needs a bisect against the previous tag before anyone can say whether it blocks. **Nothing
about this surface got worse because v0.13.0 happened** — it has been dark for as long as the
selectors have existed.

**Sharpened by what the preflight actually found.** The original entry argued the yield would be high
because one variable was set and produced two failures. The 2026-08-12 comparison shows those two
were **already failing**, so the correct statement is stronger and simpler: **the dark surface hides
standing failures, not just unknowns.** Three of the gates reachable this way
(`TestDecodeParityInt4`, `TestSerializeWeightsTo_matchesBuffer`, `TestQwen35GGUF_vsSafetensors`) have
been red for at least two days with nobody informed.

**"At least two days" was the visible floor, and it was off by a factor of thirty.**
`TestDecodeParityInt4` was bisected 2026-08-15 to `7deb368` (2026-06-14) — **red for two months**,
because it was skipping for all but the last few days of that. The dark surface did not merely hide
a standing failure; it hid **when** the failure started, and therefore what caused it. That is the
argument for the campaign at full strength: the cost of a dark gate is not the red you eventually
see, it is the attribution you lose in between. (Resolved as a stale golden, not a defect — the
W4A8 scale-fold kernel that bump carried made int4 *more* faithful to f32, 5 → 11 leading ids. See
`queue-release.md`; two of the three remain in this campaign.)

**After the release:** batches, each failure triaged by **(a) can a shipped path reach it** and
**(b) is it older than the current tag**, with a **budget that stops the campaign** rather than the
list stopping it.

**COUNT — two numbers, recorded rather than smoothed.** The census says **42**; an independent
re-derivation counting only variables read *inside a test body* says **41**. The difference is
`GOINFER_QWEN35_REAL`, read in a file-level helper. Both attributions are defensible. The next person
to count will get 41 and should know why.

The 42, and the test count behind them (47 test functions), are reproduced by
`python3 scripts/selector_coverage.py` — the report is the list, so it cannot go stale here.

Seven non-`GOINFER_` variables also gate unselected tests and are excluded as out of scope:
`G4_TRACE`, `GEMMA3_4B`, `GOMEMLIMIT`, `HOME`, `NOISE_FLOOR_CKPT`, `ROUTER_CAPTURE_OUT`, `ZZBASE`.

## B19 — the parity sweep is UNOBSERVABLE for its first 30-60 minutes (post-1.0)

**Found 2026-08-19 during the v0.14.0 release prep**, while trying to answer "how far along is it?"
and having nothing but RSS to go on.

`scripts/parity_sweep.sh` phase 1 is one command:

```sh
go test -v -timeout "$TIMEOUT" ./decoder/ ./tokenizer/ >>"$LOG" 2>&1
```

**`go test -v` BUFFERS PER PACKAGE when it is given more than one package.** Nothing reaches `$LOG`
until `./decoder/` completes in full — measured: the log sat at **0 bytes for 30+ minutes** while
the binary ran at 595% CPU and ~78 GB RSS. The run is healthy and completely opaque, which is the
combination that makes a watcher wonder whether to kill it.

`scripts/gpu_gate.sh` does not have this problem, and the reason is instructive: it runs a SINGLE
package (`./cuda/`) per invocation and streams one line per test. Same tool, different arity.

**The fix (small, and NOT to be applied while a sweep is running — bash reads a script lazily by
byte offset, so editing an executing one can corrupt it mid-flight):**

```sh
for pkg in ./decoder/ ./tokenizer/; do
  go test -v -timeout "$TIMEOUT" "$pkg" 2>&1 | tee -a "$LOG" | grep --line-buffered -E '^(--- |ok |FAIL)'
done
```

One package per invocation restores streaming; `tee` keeps the full log the classifier already
parses; the `grep` gives the operator the same one-line-per-test view the GPU gate has. Check the
exit status via `PIPESTATUS`/`set -o pipefail` so a pipe does not swallow a failure — that is the
one trap in this change.

**Why it is worth doing rather than tolerating:** a release ritual that cannot be watched gets
watched by proxy — `ps`, `free`, `/proc/PID/maps` — and the fallback for "is it stuck?" becomes
"kill it and re-run", which on a 2-3 hour sweep is the most expensive possible answer. It also
makes a HANG indistinguishable from a slow real-model gate for the first hour.

**Deferred to post-1.0 deliberately** (the v1.0 gate's §7 scope discipline): it changes the release
tool during a release. File, do not touch mid-flight.

## B15 — the manifest EMITTER promotes experimental → validated on tiny-golden evidence

**Found 2026-08-13 by running the sweep with `EMIT_MANIFEST=1`.** The merge wrote:

| family | committed | emitted |
|---|---|---|
| `glm4_moe`, `mixtral`, `qwen2_5_vl`, `qwen2_moe` | `experimental` / `tiny-golden` | **`validated`** / `tiny-golden` |
| `mellum` | `validated` / `real-model-oracle` | `validated` / **`real-oracle`** |

Two distinct defects: it **promotes status without upgrading the method**, and it **corrupts a valid
method string** (`real-model-oracle` → `real-oracle`, which is not a T3 method).

**`TestParityManifest_methodTier` caught it** — it passes on the committed manifest and fails on the
emitted one. Without that gate this run would have silently upgraded four families to *supported* in
the published capability matrix on tiny-golden evidence. That is the gate doing exactly the job it
was written for, against the tool that writes the file.

**The emission was DROPPED, not committed.** The manifest stays at its committed state. Commit Y for
this release therefore contains **no manifest change**, and that is the honest outcome rather than a
missing step.

**Not fixed.** The emitter needs the promotion rule and the method-name handling repaired before
`EMIT_MANIFEST=1` is trusted again; until then the flag writes claims the tier gate has to catch.

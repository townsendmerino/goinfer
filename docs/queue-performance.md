# Performance queue

Throughput, latency, kernels, residency, memory. Anything whose success criterion is a **measured number** — a benchmark, a profile, a bytes-per-token figure. If the question is *how fast* or *how much memory*, it belongs here.

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


## In flight

**A11 · moe_route's demand threshold — RESOLVED 2026-08-12. The identity now CLOSES; the old pin was
the outlier** — `linux`, **CLOSED**, and it merges with A9-RESID

**A11 and A9-RESID are ONE finding with two observations.** `TestMoERouteDemandThreshold` failed with
both bounds moved by exactly **+589,824 B**, deterministically. That number was already on the record:
A9-RESID called it "baseline drift". It is not drift. It is the amount by which
**demand = floor + residual** failed to close at `MOE_MAX_E=512` while closing exactly at 256:

    256:  151,191,552 + 54,525,952  = 205,717,504   measured 205,717,504   EXACT
    512:  151,191,552 + 138,412,032 = 289,603,584   measured 289,013,760   short by 589,824

**The measurement now reads 289,603,584 — the closed form, to the byte.**

**Both components re-measured, and BOTH HELD** — so this is a fourth outcome, not one of the three
that were pre-registered (floor moved / residual moved / identity wrong):

| component | recorded | re-measured | |
|---|---|---|---|
| floor (allocate-until-failure, fresh context: 7,665,287,168 reported − 7,514,095,616 obtained) | 151,191,552 | **151,191,552** | unchanged |
| residual (`cuMemGetInfo` either side of first launch) | 138,412,032 | **138,412,032** | unchanged |
| demand (balloon bisection) | 289,013,760 | **289,603,584** | = floor + residual |

Nothing about the machine or the kernel moved — consistent with `aikit/gpu` never leaving `v0.28.0`
and the gate's own 21/21 byte-identical PTX. **The OLD PIN was the outlier**, recorded from the one
measurement that did not close, with the 589,824 misattributed to drift rather than read as *a
failure to close*. A9-RESID's "CLOSED — baseline variance" verdict was wrong on the mechanism while
being harmless in effect.

**The pins are RE-DERIVED, not edited to match** (`cuda/moe_route_demand_test.go`), with the identity
recorded at the pin site as their justification, plus the instruction for next time: if these move
again, check the identity first — if `floor + residual` still equals the demand, the components moved
and the pin is downstream of them.

**BLAST RADIUS — narrower than the first filing implied.**

- **A1 is NOT at risk.** Its closed form does not rest on the demand threshold: it was confirmed by
  predicting allocation at 16, 30 and 34 slots and free-after-`allocSlots` at 34, **all to the byte**.
  Nothing in this touches it, and A11 must not be read as reopening the A-chain.
- **In scope, and now corrected:** A9's demand figure, and A10's decomposition — which is *vindicated*
  rather than revisited. The identity it proposed now closes at both `MOE_MAX_E` values instead of
  one.
- **Nothing that ships is affected.** The margin check passes with `slotMarginBytes` 402,653,184 ≥
  289,603,584, **clear by 107.8 MiB**, and the 33-slot cap is a binary search against the margin, not
  against these pins. This was a record-versus-machine mismatch, never a defect.

**The lesson worth keeping: a residue that "is exactly the drift we measured elsewhere" deserves more
suspicion than a residue that is merely small.** 589,824 B against 289 MiB is 0.2% — easy to wave
through. Its appearing *twice, exactly* is what made it a mechanism, and the second observation is
what resolved the first.


## Queued

**A1 · Why the 26B forward failed at 34 slots** — `linux`, **CLOSED**

Waste is **allocation granularity**, fully accounted. Buffers are 123,904 × {1, 2, 8, 16} bytes per
slot, 4 per layer, 30 layers; each is rounded up **independently** to the driver's 2 MiB quantum.

    x = n × 123,904 / 2,097,152
    Q(n)           = ceil(x) + ceil(2x) + ceil(8x) + ceil(16x)      per-layer quanta
    Requirement(n) = 30 × Q(n) × 2,097,152

`roundedPredicted` matched measured actual **to the byte** at 16 and 30 slots, with structure
asserted before totals at both (120 allocations, 4 distinct sizes, 30 occurrences each, distinct
sum = n × 3,345,408), and predicted post-`allocSlots` free matched to the byte at 34 — a slot count
never used to derive the form.

**No memory was ever unaccounted.** The delta was real; the prior closed form under-predicted it by
ignoring the quantum. Machine state for every figure here: free 3,847,880,704 B, margin
402,653,184 B (384 MiB), quantum 2,097,152 B, 30 MoE layers.

| n | Q(n) | requirement | + margin | vs free | free after `allocSlots` |
|---|---|---|---|---|---|
| 30 | 50 | 3,145,728,000 | 3,548,381,184 | 299,499,520 spare | 702,152,704 |
| 32 | 53 | 3,334,471,680 | 3,737,124,864 | 110,755,840 spare | 513,409,024 |
| 33 | 54 | 3,397,386,240 | 3,800,039,424 | 47,841,280 spare | 450,494,464 |
| 34 | 58 | 3,649,044,480 | 4,051,697,664 | **203,816,960 over** | 198,836,224 |

**Corrected cap is 33.** At n = 34, x crosses 2 and all four buffers tip at once — a 4-quanta step.
34 is the worst boundary in the range, and it is the value the README used to recommend.

Known and durable: driver quantum is **2 MiB** (not next-power-of-two — 5→6, 6→6, 9→10 MiB
measured); sub-quantum requests are pool-served but **not free**. Both asserted in
`cuda/allocgran_test.go` with their measurements.

**A2 · 26B documentation correction** — `linux`, **unblocked by A1**

38 slots is unreachable with correct accounting. The published 16.98 tok/s was measured at a cap
that shouldn't have been granted — it worked with ~133 MB leftover, equal to the forward's demand
within measurement error. The README currently instructs `GOINFER_MOE_CACHE_SLOTS=48`.

The corrected-cap figure is **33**, and the leftover-VRAM column can now be filled from the closed
form (the table under A1). Publish what the corrected cap delivers, with that fourth column for
**leftover VRAM after `allocSlots`** — the figure that distinguishes a safe operating point from a
lucky one. Record 16.98 as **measured-but-unsafe rather than retracted**; the correction is a few
percent, not a repudiation.

The hit-rate curve is worth publishing alongside, since it explains the flag better than an
instruction does. **The leftover-VRAM column now falls out of the closed form** — it is what
distinguishes a safe operating point from a lucky one, and it is the column whose absence let 38 be
published:

| slots/layer | LRU hit rate | decode | requirement (rounded) | leftover after `allocSlots` |
|---|---|---|---|---|
| 8 (default) | 0% — **inert** | ~5 tok/s | 838,860,800 | 3,009,019,904 |
| 16 | 57.3% | 11.33 tok/s | 1,698,693,120 | 2,149,187,584 |
| 30 | *not yet measured* | *not yet measured* | 3,145,728,000 | 702,152,704 |
| 33 | *not yet measured* | *not yet measured* | 3,397,386,240 | 450,494,464 |
| 34 | — | **0 tok/s** | 3,649,044,480 | 198,836,224 — **below the 289,013,760 demand** |
| 38 | 81.6% | 16.98 tok/s | 3,900,702,720 | **negative** — unreachable |

Machine state: free 3,847,880,704 B at `allocSlots`, 30 MoE layers. Leftover = free − requirement;
the 384 MiB margin is what the cap additionally reserves, so a row is grantable when
requirement + 402,653,184 ≤ free. The 8/16 rows' leftovers are confirmed by measurement (the A1
instrument read 16 slots' consumption to the byte); 30/33/34/38 are computed from the same form that
matched at 16, 30 and 34.

At the default of 8 the cache is **inert** — the routed set exactly fills it and nothing survives to
the next token.

Pre-registered prediction for 30 slots, to test the curve rather than assume it: **~74–78% hit
rate, ~15.0–15.8 tok/s**.

### B. Enforcement gaps — things that exist but aren't composed into a decision

**A3 · Make the launch OOM say what it is** — `linux`, **DONE `e42e83e`**

The one failure that has resisted a day of investigation also produces the least informative
message: a raw `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY` with nothing tying it to the cache
setting the user chose. The decline floor added in `a15a394` does not catch it — that fires below
`topK`, and this dies at 34 slots with `topK` of 8.

Two changes, both error handling rather than prediction, useful whatever A1 turns out to be:

1. **Name the kernel in the launch error.** One line. As a side effect it collapses much of A1's
   candidate space — a router or eviction kernel failing means something very different from the
   main expert GEMV failing, and right now the message does not distinguish them.
2. **Catch the launch OOM on the resident MoE path and reframe it**: name the configured slot
   count, say it is the likely cause, suggest lowering it. That converts a fatal driver error into
   an actionable decline — the pattern applied everywhere else this fortnight and conspicuously
   absent exactly where a day was spent.

Deliberately its own item rather than folded into A1, so the next investigation starts from a
message rather than a symptom.

**Landing shape, fixed.** Shipped error text is: kernel name, **requested** slot count, **effective**
slot count after capping, cause, remedy. Naming only the effective count sends a user who set 48 to
lower it to 40 — which caps to the same value and fails identically, making the advice look wrong.
**No VRAM readings in user-facing error text**: those are instrumentation, they move with the probe,
and a number whose meaning depends on where it was taken does not belong in a message the reader
cannot situate. `pipeName` must be **total** — no panic on any shape, including nil, unexported, and
a pipeline not held in a named field; it runs only when something has already failed, so it must not
be able to turn a diagnosable error into a crash. staticcheck ST1005: lowercase, unpunctuated.

Split from the A1 VRAM instrumentation before committing — the two are currently interleaved in
`cuda/resident.go` and `cuda/backend.go`, and one ships while the other does not.

**A4 · Do the two cap implementations differ?** — `linux`, **CLOSED, refuted**

Both copies apply the 384 MiB margin; they do not disagree. The benchmark's cap of 38 came from a
machine state with more free VRAM, not from a second implementation computing a different answer.

This mattered because a numeric disagreement between the copies would have accounted for a claimed
cap of 38 against an observed 34 with **no unaccounted VRAM at all** — it had to be excluded before
any accounting branch was believed. That they agree numerically does not make the duplication
harmless: see A5, and the `capSlots` row under sibling drift in `parity-coverage-policy.md`.

**A7 · Confirm the corrected cap by run** — `linux`, **DONE 2026-08-12, every figure as predicted**

Pre-registered before the run: free after `allocSlots` at 33 slots is 450,494,464 B exactly, and the
forward succeeds. Both held.

| reading (real 26B, 33 slots) | measured | predicted |
|---|---|---|
| free before `allocSlots` | 3,847,880,704 | 3,847,880,704 |
| free after `allocSlots` | **450,494,464** | **450,494,464** (exact) |
| free at first launch | 450,494,464 | — |
| free before the last launch | 312,082,432 | — |
| tokens generated | **4** | > 0 (34 slots gives 0) |

**The cross-validation is the valuable part.** The decrement from first launch to last launch is
**138,412,032 B — exactly the `moe_route` residual pinned from the synthetic fresh-context harness**,
reproduced here on a completely different path: real model, real expert cache, real decode. Two
independent routes to the same byte figure.

So the corrected cap of **33 is confirmed by run**, not only by formula, and A2 can publish it.

What this run does *not* do is narrow the demand: 33 passes with 450,494,464 free and 34 fails with
198,836,224, which brackets the 289,013,760 threshold without tightening it. The balloon search is
still the only measurement of the demand itself.

Note the margin reading has changed. The concern written here in advance was that 33 clears the
*margin* by only 47,841,280 B — but the margin is not the binding quantity; the **demand** is, and 33
clears that by 161,480,704 B. The old framing would have called 33 marginal when it is not.

**A8 · Is `fRoute` the first launch?** — `linux`, **CLOSED**

`fRoute` is **not** the first launch of the token — `ropeKV` (from `gmod`) and `fAttn` (from the
glue module) precede it. But it is plausibly the first launch out of **`moePTX`**, so a lazily
deferred module load is attributable to it exactly as a first-launch would be. A9 stands.

**A5 · The corrected cap must be a SEARCH, not a division** — `linux`, **DONE `6091e7a`**

When A1's fix lands, do not write it as a division with a correction term. Per-buffer 2 MiB
rounding makes the requirement a **step function of slot count**, and

    fit := int((free - marginBytes) / nLayers / perLayer)

cannot invert a step function — it is wrong precisely at the boundaries the failure lives on. A
division plus a fudge term reproduces the class A5 exists to close.

The requirement is monotone non-decreasing in n, so binary-search it:

    largest n such that
      nLayers × Σᵢ ceil(n·pᵢ / 2 MiB) × 2 MiB + marginBytes ≤ free

where pᵢ are the per-slot per-buffer byte strides (4 buffers per MoE layer).

Land it in ONE implementation — `allocSlots` calls `capSlots`, not a copy — with the gate pointed
at the shipping path and a mutation check. And correct, in the same change, wherever it was written
down that `cuda/slotcap_test.go` corroborates production sizing: it corroborates a parallel copy.

**Landed and verified on hardware through the shipping auto-cap path**, no manual slot count:
requesting 128 on the real 26B logs `capping to 33` and generates 4 tokens, where it previously
logged `capping to 34` and generated 0. Free after `allocSlots` is 450,494,464 B — byte-identical to
A7's manually-set 33-slot run, so the search lands exactly where the measurement said.

Two withdrawn claims went with it. `cuda/slotcap_test.go` said it "corroborates the sizing" (it
corroborated the *copying*; the answer was wrong) and that the agreement placed the discrepancy
"downstream of the sizing decision" (the sizing decision **was** the defect). The gate now asserts
properties rather than a remembered number: search ≠ division at the 26B configuration, monotonicity
in n, the 4-quanta step at 33→34, and that no returned count leaves n+1 also fitting.

**README retraction, done in the same change — revert-includes-claims.** The README said the slot
count was "a manual workaround for a safety net that is not holding". That was accurate and stopped
being accurate the moment this landed. It is not deleted: it now records what the old behaviour was,
names both costs the cap was missing with their measured figures, and gives a reader who hit the
failure a way to tell whether their build has the fix (`capping to 33` has it, `34` does not).

**Mutation deltas, re-derived.** The margin mutation was first documented as 38, carried over from
the raw-sum derivation without re-deriving under the rounding form; the gate caught it. The real
answer is **37** — 33 → 37, not 34 → 38. Same delta, different endpoints, and only one is real.

**A9 · The deferred fixed cost paid after the cap is computed** — `linux`, **CLOSED — cause
established, after one reopen**

Ran 2026-08-12, **before A5 landed**, so no cap override was needed and 34 was reachable.

**Answer: `moe_route`'s first launch demands 289,013,760 B (275.6 MiB) of free VRAM, and retains
138,412,032 B (132.0 MiB) of it.**

| quantity | bytes | MiB |
|---|---|---|
| highest observed FAIL | 286,916,608 | 273.6 |
| **lowest observed PASS (the demand)** | **289,013,760** | **275.6** |
| residual after a successful launch | 138,412,032 | 132.0 |

Peak demand is **2.09× the residual**, and the ~150 MiB difference is transient and **still
unnamed** — the one loose end this item leaves.

**It closed once too early.** The first answer was the 132 MiB residual, taken as the cause. It is
not: free before the failing launch was 198,836,224 B, which *exceeds* 138,412,032 by 60,424,192 B.
The reservation fit and the launch failed anyway. The tell was in the numbers already recorded —
free after the failure (265,945,088) was 67,108,864 B **above** the pre-attempt level, and an unwind
cannot return more than was taken, so something that pre-existed the attempt had been released. The
demand was measured directly rather than inferred: balloon the device to a chosen free level, launch
`moe_route`, bisect, one fresh context per trial.

**The arithmetic closes now:**

- 26B at 34 slots: 198,836,224 free against 275.6 MiB demand — short by **90,177,536 B**. Fails.
- post-trim 265,945,088 — still short by **23,068,672 B**, which is why trimming did not save it.
- 26B at 33 slots: 450,494,464 free — clear by **161,480,704 B**. (A7 still runs; see below.)

**Capacity, not contiguity.** Three identical repeats only exclude run-to-run noise, since a
deterministic balloon produces a deterministic layout. The discriminating control is balloon *shape*:
re-run filling with many 2 MiB blocks instead of a few large ones — same free bytes, very different
arrangement — and the threshold is **identical to the byte**. (Contiguity was refuted earlier in this
campaign against a different observation; that refutation was about slot buffers and did not carry
here, so it was re-tested rather than reused.)

**Enumerated, not sampled** (`TestKernelLocalMemoryCensus`): `LOCAL_SIZE_BYTES` for all **37 entry
points across all 12 embedded modules**. Three declare per-thread local memory — `moe_route` at
**4416** B/thread, `rope_kv` and `rope_kv_batched` at 32 each. Two kernels was a sample and the
sibling-drift class is about exactly that.

**All three figures are pinned**, not asserted non-zero — a gate that only checks "a reservation
exists" lets a future `MOE_MAX_E` change double a hidden cost while staying green, which is the
exercised-but-never-triggered shape inside the gate written for this finding. Mutation-checked: each
pin perturbed by one byte, each fails red.

**A9-FIX · The fix is ORDERING, not a bigger margin** — `linux`, **DONE `0103b49`**

Adding 275.6 MiB (or 132, or any measured figure) to `slotMarginBytes` is the correction-term mistake
relocated from A5 to A9. It buries a named consumer inside an unnamed constant, and the next deferred
cost — a new kernel with per-thread scratch, a driver that reserves differently — reopens it silently.

**Structural fix: pay the deferred reservation BEFORE taking the free reading that sizes the cache.**
Force the flagged kernels to launch once during `BuildResident`, ahead of `allocSlots` at
`cuda/backend.go:836`. The cap is then correct **by construction**, because the free reading it is
computed from already includes every fixed cost that will ever be paid.

**Known-unbounded, recorded rather than filed as a defect.** Module load is **resident-zero and
transient-nonzero**: `CompileLibrary` + `NewComputePipeline` cost 0 B by both instruments at 7.6 GB
free, but `cuModuleLoadData` *fails* with `CUDA_ERROR_OUT_OF_MEMORY` under memory pressure — measured
incidentally when a test's module setup was placed behind a balloon. The transient is unbounded and
unmeasured. A9-FIX's ordering argument is unaffected, because forcing happens while ~3.8 GB is free;
what nothing currently prevents is a *later* module load paying that transient under pressure.

**On iterating the census rather than naming `moe_route` — how this actually landed, and why.**
Mechanical iteration turned out not to be available: forcing a reservation requires *launching*, and
launching requires valid arguments, which is per-kernel knowledge. A zero-block launch would have
avoided that (no thread runs, so no argument is dereferenced) and was tested — **rejected by aikit's
geometry validation**, `invalid launch geometry (grid 0x1x1)`.

So the fix forces `moe_route` **by name**, which is sound for a reason that is itself measured: the
backing store is **shared and sized by the largest kernel**, so forcing the maximum forces the pool
for every kernel. The naming is kept honest by moving the assumption into a check —
`TestKernelLocalMemoryCensus` enumerates every entry point in every embedded module and **fails,
naming `cuda/backend.go`**, if any kernel declares more per-thread scratch than `moe_route`. That is
the enumerate-the-members remedy applied where enumeration cannot be mechanical: the *selection* is
checked even though the *launch* is hand-written.

**Measured result.** Cap moves 33 → 31 and the decrement from first launch to last launch goes from
138,412,032 B to **0** — nothing is taken after the sizing decision, which is the whole property. Free
after `allocSlots` rises to 501,415,936 B; 4 tokens generate as before. **The trade is two slots**, and
that is the point: the margin no longer silently absorbs 132 MiB it was never sized for, so 384 MiB
now means 384 MiB.

**A9-MARGIN · Re-derive `slotMarginBytes` now that it covers only what it names** — `linux`,
**ANSWERED by A10: do not lower it. 151,191,552 B of the margin is a driver floor, not slack.**

Three runs on the real 26B with A9-FIX in place, varying only the margin:

| margin | cap granted | outcome | leftover after `allocSlots` |
|---|---|---|---|
| 384 MiB (shipped) | 31 | 4 tokens | 501,415,936 |
| 128 MiB | **33** | **4 tokens** | 312,672,256 |
| 32 MiB | 34 | **allocation FAILS**, declines to CPU | — |

So the two slots A9-FIX cost **are recoverable** at a 128 MiB margin, on this card, at this free
level. **But do not take that as the recommendation**, because the reason 34 fails is not the one the
margin models — see the next item. The margin is currently doing a job nobody specified, and lowering
it to 128 MiB would work here for a reason that is not understood. Measure the *servability*
constraint first.

What this run did establish: with the reservation paid up front, the decrement after `allocSlots` is
**0 at every cap tested**, so post-sizing consumption is genuinely nil and the margin is not covering
launch growth. Its remaining job is whatever the next item names.

**A10 · The ~150 MiB driver allocation floor** — `linux`, **THE OPEN CUDA ITEM — and now SCHEDULED,
not merely interesting: D3b is blocked on it.**

Raising the expert-cache default above `topK` requires `cuda/backend.go`'s stated precondition,
"fixing the margin FIRST" — read as *derived rather than asserted*. **A10 is 151,191,552 B of the
402,653,184 B margin, 37% of it, unattributed**, and no derivation can be written while more than a
third of what the margin covers is unexplained. So A10 is on D3b's critical path.

**FIRST DISCRIMINATOR RUN 2026-08-12 — the floor is NOT local-memory backing.**

Launched `shared_gate_combine` — zero declared per-thread scratch — alone, from a freshly loaded
module, in the balloon harness:

    allocation floor                151,191,552
    zero-local kernel LAUNCHES at   153,288,704   (one 2 MiB quantum above the floor)
    moe_route demand                289,013,760   = floor + 137,822,208
    moe_route residual              138,412,032   (differs by 589,824, the A9-RESID drift)

**The bracket check refused to search**, because the low end *succeeded* — which is the answer. A
kernel with no local memory launches at the lowest free the balloon can reach, so **the floor is not
a launch demand at all**: it is the allocator's own limit, and `moe_route`'s 289 MiB is
`floor + its backing store`. Local-memory backing is fully accounted for by the residual, and the
floor sits underneath everything, kernel-independent.

**Candidate space halved without proposing a mechanism.** Ruled out: anything scaling with a kernel's
declared local memory. Remaining: a per-context or per-device allocator reserve.

**MODEL TESTED, 2026-08-12 — the reporting gap is CONFIRMED, exactly.**

> `cuMemGetInfo` reports **151,191,552 B more free than is allocatable, by anyone**.
> `usable = reported_free − 151,191,552`.

Cross-checked **with no launch and no bisection**: allocate directly in a fresh context until even a
1 MiB request fails, and compare what was obtained against what was reported at the start.

    reported free at start   7,665,287,168
    total obtained           7,514,095,616
    SHORTFALL                  151,191,552   ← the floor, to the byte

No kernel involved, so this is not about launches at all. It also reproduces the 34-slot failure
directly: usable at 198,836,224 reported is **47,644,672**, against `moe_route`'s 137,822,208 backing
store — short by 90 MiB, which is the figure A9 measured.

**The CONTEXT discriminator did not resolve, and is blocked by the API.** With the child holding
everything down to the floor, the parent process **cannot create a context at all** —
`cuDevicePrimaryCtxRetain: CUDA_ERROR_OUT_OF_MEMORY` at 151,191,552 B reported free. That is a real
finding (the floor is not available for context setup either) but it means the arm cannot measure
what it was built for. The in-process arm is blocked too: gocudrv exposes only primary-context
retain, not `cuCtxCreate`.

**RESOLVED 2026-08-12 — the reserve is PER-CONTEXT, and the derivation survives anyway because
goinfer has exactly one.**

The first attempt failed because the child drained to the floor and left the parent nothing for
context setup. Giving it room — child balloons to ~300 MiB reported free and holds, parent reads free
via `nvidia-smi` (no context needed), retains the primary context, reads again:

    child holds, reporting          312,672,256 B free
    parent BEFORE its context       313,524,224 B free
    parent AFTER  its context       205,717,504 B free
    DELTA — what a 2nd context cost 107,806,720 B  (102.8 MiB)

**Per-process: a second context pays again.** So `slotMarginBytes` is **not a device constant** in
general — N contexts cost roughly N × this.

**A10 IS NOW FULLY DECOMPOSED — nothing is unattributed.** The gap splits exactly into a
**once-per-device** portion and a **per-context** one, and the split is measured rather than fitted:

| quantity | bytes | MiB |
|---|---|---|
| gap in a single process (its first context) | 151,191,552 | 144.1875 |
| marginal cost, **2nd** context | **106,954,752** | **102.0000** |
| marginal cost, **3rd** context | **106,954,752** | **identical, to the byte** |
| device-once portion (gap − per-context) | **44,236,800** | 42.1875 |

    44,236,800 + 106,954,752 = 151,191,552   EXACT

Two independent additional contexts cost the same 102.00 MiB, so the per-context term is a constant
and the residue is the once-per-device setup. **`usable = reported_free − 44,236,800 − 102 MiB × contexts`**,
which for goinfer's single context is the 151,191,552 already in use.

**A measurement defect was fixed to get here, and it mattered.** The first run reported the marginal
cost as 107,806,720 B — it took `pre` from `nvidia-smi` and `post` from `cuMemGetInfo`, so the delta
silently carried ~832 KiB of *instrument disagreement* as if it were context cost. That is the
851,968 B discrepancy visible in the earlier record. Reading both sides with the same instrument gives
**exactly 102.00 MiB**, and the decomposition only closes with the corrected figure. Same shape as the
measurement-shape class: the number was real and the comparison was not like-for-like.

**Why the derivation holds for goinfer regardless: it creates exactly ONE context, and cannot create
a second.** `cuda/backend.go:463` is the only production call of `CreateSystemDefaultDevice`, aikit
retains the **primary** context (`cuDevicePrimaryCtxRetain`, refcounted per device per process, so
repeat calls do not make a second), and **`cuCtxCreate` is not bound by gocudrv at all**. The
single-context premise is therefore enforced by the dependency, not merely by current usage.

**Original entry follows.**

**A10 (original) · The ~150 MiB driver allocation floor** — `linux`, **THE OPEN CUDA ITEM.** Mechanism measured,
cause unattributed.

**Status: measured, not explained.** ~150 MiB that `cuMemGetInfo` reports as free and `cuMemAlloc`
will not hand out — 150,601,728 B at `MOE_MAX_E=512`, 151,191,552 B at 256, i.e. **constant across
the one parameter tested**, within the known 589,824 B baseline drift. What it *is* remains
unattributed: driver reserve, allocator bookkeeping, or something else. It is not fragmentation
(refused at any request size down to 1 MiB) and not capacity (free was 2.71× the request).

**Two figures are RETIRED, and why matters.** A9 twice recorded a "~150 MiB transient, still unnamed"
and a "peak is 2.09× the residual" ratio. Both dissolve: **demand = floor + residual**, exact at
`MOE_MAX_E=256` and off by exactly the baseline drift at 512. So the transient was never transient —
it is this floor — and the 2.09× was never a property of anything, being `(floor + residual)/residual`,
which reads **3.77×** at 256 for the same system. A ratio between two quantities that scale
differently is not a constant of the thing.

The original finding follows.

**Original: the total fitting does not imply the allocations succeed.**

**The mechanism.** `cuMemGetInfo` reports **151,191,552 B (144.2 MiB) more free than `cuMemAlloc`
will hand out** — at *any* request size down to 1 MiB. Measured directly by draining the device in
shrinking blocks (`TestAllocFloor`, seconds, no model):

    1024 MiB blocks exhausted -> free 1,222,836,224
     ...
      32 MiB blocks exhausted -> free   182,648,832
      16 MiB blocks exhausted -> free   165,871,616
       8 MiB blocks exhausted -> free   157,483,008
       4 MiB blocks exhausted -> free   153,288,704
       2 MiB blocks exhausted -> free   151,191,552
       1 MiB blocks exhausted -> free   151,191,552   <- FLOOR

**The ladder reproduces both 26B failures exactly.** The 32 MiB rung exhausts at **182,648,832** —
the precise free figure at which the group-by-group order refused a 67 MB request. The 4–8 MiB rungs
bracket 155,385,856, where largest-first refused a 4.2 MB one. Nothing about either failure is
mysterious once the floor is named.

**The rule, and it fits every observation:** *leftover after `allocSlots` must exceed the floor.*

| cap | leftover | vs floor 151,191,552 | outcome |
|---|---|---|---|
| 31 (shipped) | 501,415,936 | clear | works |
| 33 | 312,672,256 | clear | works |
| 34 | 61,014,016 | **below** | fails mid-allocation |

**This retires a figure A9-MARGIN nearly recommended.** A 128 MiB margin is **134,217,728 — below the
floor**. The cap-33 run under it worked only because *that cap's leftover* happened to be 312 MiB;
the margin itself was unsafe. That was luck, and it is now something a test can see:
`TestAllocFloor` asserts `slotMarginBytes ≥ floor` (shipped 402,653,184, clear by 251,461,632).
Mutation-checked at 128 MiB → red.

**The ordering hypothesis was REFUTED, and the change was kept anyway on different grounds.**
Largest-first was pre-registered to complete at 34 slots. It did not — it failed on a 4,212,736 B
request with 155,385,856 B free, a ratio of **36.88**, which no contiguity account survives. It *is*
kept, because measured on its own merit it drains **27 MiB further** before hitting the floor at zero
cost. The code comment says so rather than carrying the refuted rationale.

**A9-MARGIN is unblocked and its answer is: do not lower it.** The margin's job is now decomposed —
151,191,552 B of it is the unallocatable floor and the remaining 251,461,632 B is genuine headroom.
Any reduction has to stay above the floor, which leaves far less room than the "recover two slots"
framing suggested.

**A9-RESID · The 589,824 B is baseline variance, not reservation variance** — `linux`, **CLOSED —
BUT THE MECHANISM WAS WRONG; see A11 (2026-08-12)**

> The verdict "baseline variance" was harmless in effect and wrong in cause. 589,824 B is the amount
> by which `demand = floor + residual` failed to close at `MOE_MAX_E=512`, and the demand now measures
> the closed form to the byte with both components unchanged. It was a **failure to close**, read as
> drift. Merged into A11, which is the resolved record.

The launch-configuration branch is **refuted**. The reservation is **138,412,032 B at every
configuration tested** — nE ∈ {1, 8, 128, 512}, k ∈ {1, 2, 8}, a 512× span in nE — which is what a
compile-time property should do, and confirms the driver sizes the backing store from the kernel's
declared footprint rather than from anything passed at launch.

So the 576 KiB is the other branch: **the pre-launch free-VRAM baseline itself moves**. It reproduced
directly — the same harness reported free before the first `moe_route` as 7,662,600,192 in one build
and 7,663,190,016 in another, **a difference of exactly 589,824 B**, with the reservation identical in
both.

**Caveat worth carrying.** Every figure in A1/A2/A5/A7 is anchored to a pre-`allocSlots` free of
3,847,880,704 B, and that anchor is now known to drift by ~576 KiB. That is well under the 2,097,152 B
quantum, so it can only change a cap decision when a requirement lands within 576 KiB of the
free-minus-margin boundary — not the case at any figure recorded here, but it is why the cap should
never be quoted as a property of the card alone.

**Why the ordering fix is better than a margin bump, stated so nobody later "simplifies" it into one.**
Peak demand is 289,013,760 and residual is 138,412,032 — a ratio of **2.09×**. Forcing early pays the
275.6 MiB *peak* while ~3.8 GB is still free, and the free reading taken afterwards then sees only the
132 MiB *residual*. **One reordering covers both quantities, and neither has to be represented by a
constant.** A margin bump would have to be sized against the peak, permanently, while the peak is
transient — it would reserve 275.6 MiB forever to cover something that is only briefly needed.

**A9-SPEC · Specialize `MOE_MAX_E` at JIT time** — `linux`, **CLOSED — not worth doing, on
measurement.**

**Measured basis for closing.** The allocation floor is **150,601,728 B at `MOE_MAX_E=512` and
151,191,552 B at 256** — invariant within 589,824 B, which is exactly the A9-RESID baseline drift. At
256 the floor is already **74% of total demand** (151,191,552 of 205,717,504). Driving the residual
all the way to zero would still leave the floor, so the reclaim is **bounded near one slot** — and
the measured 512→256 step already buys exactly one (cap 31 → 32).

Against that: a second specialized module, selection logic keyed on expert count at load, and a
dependency on the pinned 12.6.85 NVRTC path for every future rebuild. **Closed on the numbers, not
on preference.** Reopen only if the floor changes or a device shows a materially different ratio.

**The frozen-artifact decision is not part of this item.** The standing constraint is that frozen
artifacts are not regenerated and new kernels get their own files — so the shape is a **second,
specialized module alongside `moe.ptx`, selected by expert count at load**, never a rebuild of the
audited artifact. Recorded so it is not re-litigated as a freeze exception.

**Measured, 2026-08-12** — `moe.cu` compiled at `MOE_MAX_E=256` to a scratch PTX through the pinned
12.6.85 NVRTC (`cuda/testdata/moe.ptx` untouched), then read in the balloon harness:

| | MOE_MAX_E=512 (shipped) | MOE_MAX_E=256 | saved |
|---|---|---|---|
| residual reservation | 138,412,032 | **54,525,952** | 83,886,080 |
| launch demand | 289,013,760 | **205,717,504** | 83,296,256 |
| ratio (residual) | — | **0.394** | not 0.5 |

**0.394, not a halving** — which settles A9-MULT by measurement rather than by refuting a derivation.

**And it names the transient.** The two reductions are nearly identical, because
**demand = allocation floor + residual**:

    MOE_MAX_E=256:  151,191,552 + 54,525,952 = 205,717,504  — EXACT
    MOE_MAX_E=512:  151,191,552 + 138,412,032 = 289,603,584  — measured 289,013,760, off by 589,824,
                    which is exactly the baseline drift A9-RESID measured

So the "~150 MiB transient, still unnamed" that A9 recorded twice **is the A10 allocation floor**, and
the "peak is 2.09× the residual" ratio was never a property of anything — it is
`(floor + residual) / residual`, and at 256 it reads 3.77×. Both figures are retired.

**What the reclaim actually buys: ONE slot.** With A9-FIX the residual is charged before sizing, so
free before `allocSlots` rises from 3,709,468,672 to 3,793,354,752 — and the cap moves **31 → 32**.
83.9 MB is *less than one slot* (30 layers × 3,345,408 = 100,362,240 raw), so this is a boundary
effect, not a proportional win. Worth knowing before anyone budgets a second module for it.

**Not extrapolable to 128**, and now moot: 0.394 at one halving does not predict the next (the
derivation A9-MULT withdrew), and the floor caps the payoff regardless. The harness keeps
`GOINFER_MOE_PTX_FILE`, so re-measuring is a two-minute job if the basis for closing ever changes.

**A9-MULT · The halving was DERIVED and is now withdrawn** — `linux`, **CLOSED, refuted**

"`MOE_MAX_E` 256 → 512 doubled the cost from ~66 to 132 MiB" assumed the backing store is linear in
per-thread bytes with a constant occupancy multiplier. Checked: `moe_route` declares **4416**
B/thread (not the 4096 that "two `float[512]`" implies), and 4416 × 40 SMs × 1024 threads/SM =
180,879,360 B ≠ the measured 138,412,032 (**ratio 0.7652**). The occupancy factor is **not**
`SMs × maxThreadsPerSM`, so proportionality in local-bytes is unverified and the halving does not
follow.

**A second, independent reason.** The residual is **exactly 66 quanta** — 138,412,032 / 2,097,152 = 66
— so it passes through the same 2 MiB rounding A1 closed on. A quantity that is both occupancy-scaled
by an unknown factor *and* quantum-rounded cannot be halved by halving its input, even if the
occupancy factor were linear. Two independent reasons, which is why A9-SPEC's reclaim has to be
**measured** rather than predicted.

Withdrawn rather than restated. The replacement is a measurement at a lower `MOE_MAX_E`, which
A9-SPEC needs anyway.

**The named mechanism was wrong.** moePTX's *module* memory is **0 B** — at `CompileLibrary`, at
`NewComputePipeline`, and at the first launch of a module kernel that declares no scratch — with both
instruments agreeing, so it is a real zero and not a blind spot. A9's *shape* (a deferred fixed cost
invisible to the cap) is confirmed; the thing paying it is local memory, not code.

Gated by `TestMoERouteFirstLaunchReservation` (`cuda/moe_route_reservation_test.go`, seconds, no
fixture), which asserts `shared_gate_combine` reserves 0 and `moe_route` reserves more than 0 — so a
future change to `MOE_MAX_E` cannot silently move a 132 MiB fixed cost.

**Price of the router cap, recorded not proposed.** Raising `MOE_MAX_E` 256 → 512 doubled this
reservation from ~66 MiB to 132 MiB. That halving is **derived from the form, not measured**. It is
written down as the VRAM price of the cap, not as an argument to change it.

**The forcing mechanism that did not fire.** `CUDA_MODULE_LOADING=EAGER` was the intended way to pay
the load early. Readings are **byte-identical with and without it**, so it does not engage on this
driver and path — and the 26B run made under it forced nothing. Its null was uninformative, and would
have been read as "module load excluded" had the cheap control not been run. **A forcing mechanism
has to be shown to fire before a null from it means anything.**

**What this leaves for A5 — corrected, because the earlier statement is now checkable and false.**
This entry previously recorded A5 as *necessary but not sufficient*, on the reasoning that the
rounding fix alone would not have prevented the failure. With the demand measured, **it would have**:
the corrected cap picks 33, which leaves 450,494,464 B against a demand of 289,013,760 B. **A5 alone
avoids this failure.**

What A5 does not avoid is the **class**. It works only because `slotMarginBytes` (402,653,184)
happens to exceed the peak demand (289,013,760) — a relationship **nobody chose, nothing checked, and
`MOE_MAX_E` has already moved once**. That is a stronger reason to keep A9-FIX than the one written
here before, not a weaker one: the fix is not needed to make 33 work, it is needed so that the next
`MOE_MAX_E`, the next kernel with per-thread scratch, or the next driver does not silently reintroduce
a cap whose forward cannot run.

The relationship is now pinned (`slotMarginBytes ≥ measured peak demand`, clear by 113,639,424 B) so
it is at least checked rather than merely true. **`max`, not `Σ`** — launching the whole census gives
a threshold and residual identical to `moe_route` alone, to the byte, so the driver shares one backing
store sized by the largest kernel.

**The regime is part of that claim.** It was measured with the census launched **sequentially in one
context**, which is what goinfer does: batch-1, single stream, one resident model. Under concurrent
residency on separate streams there is no reason the bound stays `max` — and the assertion would then
be **wrong without failing**, the worse of the two ways to be wrong. Recorded next to the claim, and
in the gate's own failure message, the way the measured-quantities rule requires. Concurrent streams
or multi-model residency reopens it.

Historical framing, kept because the trigger was rewritten twice:

**Measured, 2026-08-12 — the pre-launch probe.** Free VRAM read immediately *before* every
`cuLaunchKernel` of the token, at 34 slots:

| reading | value |
|---|---|
| free after `allocSlots` | 198,836,224 (= predicted, to the byte) |
| free before each of the 20 launches, `fRms` … `fRouterF32` | 198,836,224, **constant** |
| free before the failing `fRoute` | **198,836,224** |
| free reported by `describeLaunchErr`, after the failure | 265,945,088 |

So **nothing is consumed between launches**, and the "64 MiB released" was an artifact of probe
position: the block appears only after the failed attempt unwinds. Settled — do not carry it as an
observation.

**Two supersessions this produced.** First, the earlier ~100 MiB threshold was already wrong (the
closed form predicts 198,836,224, so a large reading is the expected case). Second, **the decrement
trigger that replaced it is blind to the thing A9 is about.** It read free after `allocSlots` → free
at first launch → free at failing launch, expecting a deferred module load to appear as a gap. All
gaps came back 0 — but under the driver's default `CUDA_MODULE_LOADING=LAZY`, `moePTX` materialises
*during* the launch that fails, which is after the last pre-launch reading and before the
post-failure one. **No difference of those three readings can contain it.** The zero is the
instrument's blind spot, not a result.

A9 therefore runs on its own merits, and it is the only instrument that can see the cost at all.

Rationale: `fRoute` is the first kernel launched out of `moePTX` (`ropeKV` comes from `gmod`,
`fAttn` from the glue module), so a lazily-deferred module load is attributable to it exactly as a
first-launch would be. The cap is computed from a free-VRAM reading taken **before** that load. That
cost is invisible to before/after readings around `allocSlots`, and invisible to a between-slot-count
delta, because it does not scale with slots.

It is **additive with the rounding shortfall, not an alternative to it**: rounding eats into the
headroom the 384 MiB margin was sized to provide, and the module load then spends from what remains.

**Mechanism, now located precisely.** `CompileLibrary(moePTX)` runs at `cuda/backend.go:591`;
`allocSlots` runs at `cuda/backend.go:836`. Under lazy loading the module's *device* memory is not
taken at 591 — it is taken at the first launch of one of its kernels, which is `fRoute`, long after
the cap was computed from the free reading at 793. Corroborating: the failed attempt released
exactly 2^26 B while unwinding, which reads as a driver-side code/constant block rather than as
application scratch.

Experiment: force `moePTX` to load while free VRAM is still at its full ~3.8 GB, then re-run at 34
slots. The cheapest forcing is `CUDA_MODULE_LOADING=EAGER` in the environment — driver-level, read at
context creation, and **read-only on the allocation path**, so it changes when the cost is paid
without changing any goinfer code. Branches, pre-registered:

- `fRoute` launches after the forced load → module load was the mechanism; the fix shape is to size
  the cache **after** deferred fixed costs are paid, not before.
- `fRoute` still fails → module load excluded; candidate list reopens one entry shorter.
- the forced load itself fails → same finding, relocated to where it is visible. That is a result.

**Outcome against those branches: none of them, because the forcing mechanism never fired.** The
question was settled instead by measuring each step directly on a fresh context, which needed no
model and took under a second — the mechanism question was never model-dependent, and trying to
answer it inside a five-minute 26B load is what made it look expensive.

Read-only on the allocation path. The reordering is an **experiment first and a fix only after it
answers**.

**Sequencing constraint, honoured.** A9 reproduced at 34 slots and A5 fixes the cap to 33, which
makes 34 unreachable. A9 ran **before A5 landed**, so no override was needed. Recorded because a run
at the new cap would simply pass and look like confirmation, leaving no trace of the loss.

**P1 · KV re-gather and V re-transpose on every decode token** — `decoder/forwardn.go:378`

Estimated **~10–15% of per-token traffic at 4k+ context**, on all mainstream CPU families — the
largest single item in the group. Frozen core, and it needs a new aikit row-pitch API, so it is the
**v1.0-unfreeze headline** rather than something to slip in.

**P2 · Scalar `int8→f32` widen on the LM head** — **DONE, landed via the ordinary aikit release
cadence.** aikit `linalg/quant.go:136` (`q8Span`).

**Resolved 2026-08-15.** aikit shipped the exact fix this entry specified — `dequantRowInt8(deq, bq,
1.0)`, the scale-1.0 route below, verbatim — as `2f0c65f perf(linalg): SIMD widen in q8Span — ~2×
faster q8 LM head, bit-identical (P2)`, first in aikit `v1.18.0`. goinfer bumped `v1.17.1 → v1.19.0` (`fb8e26b`, goldens-refreshed at
`88ac2cd`), picking it up. **This entry's own analysis is why the
shipped fix is correct**, not incidentally: it worked out that the naive substitution
(`DequantizeRowsInt8Into` with the real scale) is a *silent* numerics change, and that passing `scale
= 1.0` is the one substitution that is bit-identical by construction — and that is precisely the
route aikit's commit took, asserted with its own bit-identity test
(`TestQ8Span_bitIdenticalToScalarWiden`, serial/parallel/SIMD-tail/prefill, every output float
compared by raw bits).

**goinfer-side proof, not just trust in the upstream claim:** the bump's `deps_hash` refresh ran the
forward goldens on **both** architectures — 19 f32 argmax+cosine goldens green on arm64 (the
FMA-contracting arch, where a reassociation would show first) and the prior box run on amd64 — no
argmax/cosine breach anywhere. Consistent with, but independent confirmation of, aikit's own
byte-comparison.

**The "leave it UNRELEASED, planned for v1.0" framing below is superseded by events, not reversed.**
It predates aikit actually resuming ordinary releases (v1.17.0 onward) and goinfer bumping through
them in the normal course — exactly what E6 (closed 2026-08-12) already settled: a release needs a
reason a consumer can receive, and this one had two (P2 itself, plus the bit-identical encoder GELU
fix riding the same bump). No decision reversed; the precondition just arrived sooner than the note
assumed.

**Still owed — the goinfer-side magnitude, not the aikit-side one.** aikit's own perfgate measured
**Δ ≈ −50% on the widen** (its own microbenchmark, LM-head shapes, M1 Pro) — real, but internal to
`q8Span`. What is NOT yet measured: goinfer's own end-to-end decode/LM-head tok/s delta from the bump
(P9/P10-style A/B, box + Mac). That is the number worth banking before the win goes in a release note.

**Original analysis, kept — it is the record of why the fix had to be exactly this shape:**

**The bit-identity condition, checked in source rather than assumed.** It splits in two, and the
half that matters is the one the original wording did not cover:

- *The widen kernel itself is exact.* `dequantI8AVX2` (VPMOVSXBD → VCVTDQ2PS → VMULPS) and
  `dequantI8NEON` (SXTL/SXTL2 → SCVTF → FMUL) both compute `float32(q[i]) * scale` elementwise, with
  no reduction and no reassociation freedom. `int8 → float32` is exact for all 256 values.
- **But the shipped call site does not apply the scale per element.** `q8Span` widens *without* the
  scale — `deq[k] = float32(bq[k])` — and applies it **after** the dot:
  `dst[i,j] = dotF32(a_i, deq_j) · s_j`. So the naive substitution changes

      dot(a, widen(q)) · s        one rounding of the scale, at the end
      dot(a, widen(q) · s)        one rounding PER ELEMENT

  which are mathematically equal and **not bit-equal**. Swapping in `DequantizeRowsInt8Into` with the
  real scale is a silent numerics change, exactly the kind that reaches a release looking like a pure
  speedup.

**The route that IS bit-identical: pass `scale = 1.0`.** Multiplication by 1.0 is exact in IEEE-754
for every finite value (and preserves ±0, inf, NaN), so `float32(q[k]) * 1.0` equals `float32(q[k])`
bit for bit on both kernels, and the scale stays where `q8Span` already applies it. Then the
structural argument holds and no parity run is needed.

Mechanics: `dequantRowInt8` is unexported and `DequantizeRowsInt8Into` is the whole-matrix form
taking a per-row `scales` slice, so this needs either an exported per-row entry or a ones-filled
slice. The `len(q) < 8` (amd64) / `< 16` (arm64) and `!hasAVX2` fallbacks all route to
`dequantRowInt8Scalar`, which is the same expression — no additional argument needed there.

**The magnitude is still an ESTIMATE and should be measured before the E6 decision.** "Several
ms/token at large vocab" was a verifier's reading. The package comment measures the same widen at
~190 ms per CodeRankEmbed forward for 113 M elements (~1.7 ns/element), and an LM head at Gemma's
262,144 × 2,560 would be 671 M elements — two orders of magnitude larger, which suggests the LM head
does **not** go through `q8Span` on the paths that matter. Establish which path the LM head actually
takes before quoting a number, and measure it there.

**P3 · Gemma final-logit softcap, serial O(vocab) `tanh` on the sampling path** — **DONE `4c26a58`**

Measured rather than estimated: the loop costs **1.43 ms/sampled token** at Gemma's 262,144 vocab and
**640 µs** parallelised — a **2.3×** on the loop, saving ~0.85 ms/token.

**The 10–30% estimate needed qualifying, not correcting.** 0.85 ms is ~28% of a 3 ms decode step and
**under 1% of the 26B's ~80 ms**. The share depends entirely on which model you run, so the loop
figure is what is recorded — it is the part that does not.

Greedy decoding does not pay it at all (`ForwardArgmax` reduces on-device and reads back 4 B), which
confirms the audit's "sampling only".

The threshold is measured, and the small end is a **loss**: 8,192 elements parallelise at 0.95×.
Hence `softcapParallelMin = 32768` rather than an unconditional fan-out.

Bit-identity is **structural** — each output element is a pure function of the input at the same
index, so there is no accumulation order to perturb. Gated at exact equality across sizes straddling
the threshold and GOMAXPROCS ∈ {1, 3, 16}, with lengths that do not divide evenly.

**Two of five siblings fixed** — see B6. The other three are frozen or on hold.

**P4 · Metal RoPE dispatched twice per layer — DONE, MEASURED NET-ZERO. Do not re-queue as a win.**

Grid-merge (2→1 dispatch/layer) is bit-identical and already implemented on branch `metal-rope-merge`
(`d682315`; snapshot-golden byte-exact) — **but that branch is not on origin and the commit resolves
in no clone here, so this claim is unverifiable from any machine but the mac. Push it or restate the
claim.** The audit re-surfaced this as "estimated a few %" **not knowing
that branch existed** — a measurement that wasn't composed into the queue (the class this file exists to
prevent). Dispatch census (2026-08-12) measured `rope` = 56/token = exactly 2/layer, so the merge
removes 28/token = **8.3% of the 338 dispatches/token**. But re-A/B'd on the current binary
(`TestZZ_metalDepthBench`, qwen2.5-coder-1.5b W4A8, M1 Pro): before 59.7/49.1/28.4/18.4 vs after
61.0/46.5/26.9/18.4 tok/s at 128/512/2048/4000 — **net-zero, within noise**. 8.3% fewer dispatches, 0%
tok/s. Correct and harmless; kept on the branch as a measured record, not merged (no speedup to bank).
See ollama-chase §A2-Metal.

**P5 · Metal `quant_vec` fused into the o-proj GEMV — PREDICTED NET-ZERO (do not build standalone)**

Dispatch census (2026-08-12): exactly **one** `quant_vec` dispatch/layer = 28/token (the o-proj input
quant; the other GEMVs already fuse theirs — so the swiglu half the "~5–6%" estimate worried about is
not a `quant_vec` dispatch and is out of scope). Fusing it removes 28/token = **8.3% of 338** — the
**same magnitude and mechanism as P4** (one small per-layer dispatch), and **P4 measured net-zero**. So
P5 is predicted net-zero by direct analogy; the fusion is more invasive than P4's merge, so it is not
worth a standalone build for a tok/s win. Only reconsider inside a **megakernel collapse** (many
dispatches at once), which is the actual Metal-decode lever (with int4 unpack / bandwidth). If ever
built, A/B it — do not assume the estimate.

**P6 · `moeMLP` allocates ~7–8 MB/token** — **DONE `eea7f29`** (`decoder/mlp.go:82`)

By skipping the `decodeScratch` invariant its dense sibling honours. **See B6.**

**PRICED (2026-08-12) — the freeze is a cost, not a prohibition, and the cost is 6 seconds.**
`decoder/mlp.go` is in the `core` shared set and `decoder/weightmat.go` in `quant`, and **all 23
families use both**, so an exception re-stales the entire matrix. But the sanctioned instrument is
`scripts/refresh_parity_hashes.sh` — the goldens-gated refresh, precedent **`ecc5af2`** (default-off
diagnostic hooks: a core-file change that is non-numeric by construction, refreshed behind the
goldens) — **not**
`scripts/parity_sweep.sh`'s T3 oracle sweep, because these are allocation changes rather than arithmetic.

Measured on `linux-62gb`: **19 goldens pass, 11 skip, 0 fail, 6.09 s wall.** One machine, no model
zoo, no HF venv. (18 of the 23 manifest rows name `linux-62gb`; only `gemma4` names
`macbook-arm64`, and that is its *oracle*, not its golden — `TestGemma4MoE_forwardParity` ran here.)

**Coverage is good for P6.** Nine MoE goldens actually RAN: `TestGemma4MoE_forwardParity`,
`TestGemma4MoEKV_forwardParity`, `TestGemma4MoEUnified_forwardParity`, `TestMixtral_forwardParity`,
`TestGlm4Moe_textParity`, `TestQwen35_forwardParity`, `TestDeepseek_textParity`,
`TestKimi_textParity`, `TestLlama4_textParity`. `TestQwen2Moe_forwardParity` skipped.

**Verdict: P6 can land now under the exception.** Do not refresh the hash without running the
script — it refuses on any golden failure and on a vacuous all-skipped run, which is the whole point.

**P7 · W4A8 allocates a fresh `Workspace` per projection per token** — **DONE `91f359f`**, verified by the int4 goldens

**RESOLVED BY READING THE SIBLING — and the answer is neither branch as posed.** No concurrency
argument is needed; the tree already contains one.

**Concurrent decode streams DO exist**, and W8A8 has **no latent race**. `decodeScratch`'s own doc
settles it: *"One lives on each KVCache — a cache is one generation stream, so the buffers are never
shared concurrently."* The Workspace W8A8 reuses (`ws *linalg.Workspace`, `decoder/scratch.go:38`)
lives inside that per-stream struct, so W8A8's "fix" was never a *shared* Workspace — it is a
**per-stream** one, race-free by the same property that makes every other scratch buffer safe.

**So the per-call Workspace comment is accurate and irrelevant to the fix.** `matmul` — the free
function, for callers with no scratch — keeps a per-call Workspace for W8A8 *and* W4A8 alike
(`decoder/weightmat.go`, the same per-call-Workspace pattern twice). The divergence is elsewhere: `matmulInto`
special-cases `isW8A8(w)` and falls through to `matmul` for everything else, so W4A8 never reaches
the per-stream Workspace even though its six call sites already pass one.

**Verdict: P7 is a straightforward divergence repair.** Route W4A8 through
`linalg.MatmulBTW4A8Into(ws, ...)` in `matmulInto`, exactly as W8A8 does. Race-free by the same
argument, no new one required.

**THE FREEZE IS NOT P7'S BLOCKER, and waiting for the unfreeze is not a route to it.** The goldens'
numeric protection is **f32-only**; P7 is an **int4** path. Lifting `6edd1ca` adds no coverage
whatsoever to W4A8, so P7 would be **just as blocked at v1.0** as it is today. It is blocked on Q1(c)
— authoring int4 goldens — and on nothing else.

**Landed once Q1(c) existed** — `91f359f`. `matmulInto` now dispatches on *"does this weight have an
Into form that takes a Workspace"* rather than on `isW8A8`. All **23 int4 rows pass** across 16
architectures; before `1d0d1ed` nothing in the tree could have told a correct W4A8 change from a
broken one, and the goldens-gated refresh would have gone green either way. That is the whole
argument for Q1(c), demonstrated on its first customer.

**Historical: blocked ONLY by Q1.** The goldens give no numeric proof on this path — every golden that runs is
f32, and W4A8 is precisely the path being changed — so the 6-second refresh would be **vacuous
exactly where it matters**. P7 lands when Q1 gives int4 a golden, or behind a real T3 quant gate.

**See B6** — and note the "sibling" framing was loose in a way the audit did not capture: the pair is
not W8A8-fixed / W4A8-unfixed, it is `matmulInto` covering one quantization and silently delegating
the rest.

**P9 · aikit v1.17.0 cost ~3% of DECODE throughput on this shape — CLOSED 2026-08-12 by aikit
v1.17.1** — `linux`

**Four statements, kept separate on purpose. Collapsing them would make the record claim more than
the work did.**

1. **The A/B measured a decode regression.** v1.16.0 against v1.17.0, interleaved, pre-registered
   2.0% floor: **−2.96%**, above the floor, per-visit medians not overlapping.
2. **aikit v1.17.1 fixed it.** The same instrument, the same floor, re-run with the expectation
   written down first: **+0.43%**, branch 1, **flat**.

   **SCOPE OF THAT "FIXED", added 2026-08-12 — it says less than it appears to.** +0.43% is a
   **benchmark-level** number, and the changed int8 code is only part of what `BenchmarkDecode`
   spends time in. So what was confirmed is that **the benchmark-level regression is gone**, not that
   the changed code is unchanged in cost. A residual effect inside the int8 path smaller than
   `0.43% ÷ (that path's share of decode runtime)` is entirely consistent with this result.

   **THE SHARE IS NOW MEASURED (2026-08-12), and it is small: 6.48%.** `BenchmarkDecode` was
   profiled — `linalg.MatmulBTW8A8Into` is **6.48%** of decode runtime (`w8a8Span` 5.22%,
   `dotI8AVX2` 5.13% flat), on the v1.17.1 build. The changed int8 code is a **sixteenth** of what
   this benchmark spends time in, so benchmark-level figures divide by ~0.065 to become statements
   about that code:

   | benchmark-level | ÷ 6.48% → within the int8 path |
   |---|---|
   | v1.17.0 regression **−2.96%** | **≈ −46%** |
   | v1.17.1 result **+0.43%** (median) | ≈ +6.6% |
   | v1.17.1 bootstrap 95% CI **[−2.52%, +3.73%]** | **[−39%, +58%]** |
   | the 2.0% floor | **≈ 31%** |

   **What that does to statement 2, plainly: the flat verdict is much weaker than it looks.** A
   residual of up to **~31%** inside the int8 path would have been *undetectable* by that A/B, and
   the bootstrap interval on the measured delta spans **−39% to +58%** within the path. So
   "v1.17.1 fixed it" means **the benchmark-level regression is gone** — nothing more. It is *not*
   evidence that the int8 path returned to its v1.16.0 cost, and it never was.

   **A corroboration worth recording, because it is independent.** The v1.17.0 regression converts
   to **≈ −46% within the int8 path**, derived end-to-end from goinfer's benchmark and a profile
   divisor. aikit measured that kernel directly at **+49% slower** in its worst (serial) case. Two
   unrelated methods, ~3 points apart — which raises confidence in the divisor and in the −2.96%
   alike.

   **Caveats on the divisor, stated with it.** Measured **under a profiler** and applied to
   **unprofiled** runs, so it is rough. And only the **v1.17.1** build is profiled — the v1.17.0
   build's share would likely be *larger* (a slower kernel takes a bigger slice), which would make
   the −46% an *over*-estimate; 6.48% is the conservative choice for the *flat* claim, which is
   where it matters most here.

   **And the flat verdict needs its delta and uncertainty printed beside it**, for the same reason:
   a floor is a practical-significance threshold, not a detection limit. "Flat" means *no effect
   exceeding the declared threshold*, never *no difference*. 17 of 36 pairwise comparisons separate, where
   18 is exactly none. `w8a8Span`'s executable body in v1.17.1 is byte-identical to v1.16.0.
3. **The locus was INFERRED, not measured.** "The int8 kernel at M=1" was recorded as inference at
   the time, explicitly labelled, with an ablation named as the thing that would settle it. The
   ablation was never run here. Upstream's revert later confirmed the inference was right — **and
   that does not retroactively make it evidence.** This measurement established a direction and a
   magnitude; it never located a cause. Anyone reading this entry as "the A/B found the int8 kernel"
   has upgraded a guess to a finding, which is the error the labelling existed to prevent.
4. **The upstream report produced a fix in a patch release the same day.** aikit v1.17.1 reverts the
   eight-column span, and its commit message adopts the method — *"interleaved with a pre-registered
   2% floor and warm-up discard — a better methodology than the one that shipped the regression"* —
   and states the mechanism this A/B could not: the two forms walk memory differently, so the
   eight-column kernel wins when B is cache-resident and loses when B is streamed. Both production
   callers stream.

**That last point is the case for the methodology, and it is written down for the next time a
careful A/B looks expensive.** The regression shipped from a real measurement taken at ONE shape.
What caught it was not a better benchmark but a *disciplined* one — interleaved rather than
sequential, floor fixed before the data, warm-up discard defined in advance, and the limits stated
rather than the result rounded. The first attempt at this number, two runs separated in time, was
worthless and would have been reported as −4% had it not been checked. **The extra ~40 minutes of
machine time is the entire reason a regression reached a patch release instead of a user.**

**Session drift makes the point concretely:** this box ran ~0.93–0.97 tok/s during the v1.17.0 A/B
and ~0.98–1.03 during the v1.17.1 one — a **5% shift, larger than either effect under test**. Any
before/after comparison spanning them would have been dominated by whichever session it straddled.

**STILL OWED — DECODE ONLY.** Every number here is decode. `linalg/matmul_blocked.go` is **unchanged
in v1.17.1**, so v1.17.0's f32 blocked-matmul rework is still live and **unmeasured in both
versions**. That path is a prefill shape this instrument barely exercises. **A prefill measurement
gates cutting a goinfer tag that carries this bump** — a release characterizing one phase while
silently carrying an unmeasured change to another is a claim by omission.

Full records: `docs/measurements/aikit-v1.17.0-decode-ab.md` and `-v1.17.1-decode-ab.md`, each
carrying its pre-registration, its raw samples, and its own weaknesses.

<details><summary>The original v1.17.0 finding, as recorded before the fix</summary>

Not a product claim and deliberately not in the CHANGELOG: an engineering finding, recorded with its
method and its limits so it is not lost. The bump (`f33fcaf`) is the **only** compiled-code change
between the arms, so the effect is attributable to it.

**Result: −2.96%**, against a noise floor of **2.0% pre-registered before the comparison ran**.

| arm | median | mean | sd | min | max |
|---|---|---|---|---|---|
| pre (`aikit v1.16.0`) | 0.9662 | 0.9674 | 0.0051 | 0.9621 | 0.9745 |
| post (`aikit v1.17.0`) | 0.9380 | 0.9364 | 0.0220 | 0.8988 | 0.9631 |

**Method**, because the first attempt at this number was confounded and the design is the finding as
much as the figure. `BenchmarkDecode`, DeepSeek-V2-Lite-Chat-Q4_K_M at `Quant: "int8int8"` (W8A8),
`-benchtime 30x`, batch=1 greedy, Ryzen 7 3700X / GOMAXPROCS 16. Both arms in detached worktrees,
`GOWORK=off`, no `replace`, aikit from the module cache. **Interleaved pre/post/pre/post in one
session** — not two runs separated in time, which is what made the first attempt worthless. The
**first sample of every visit is discarded** as warm-up, defined from an independent 8-sample
characterization and applied identically to both arms. 6 retained per arm.

**Why the direction survives the noise.** The post arm's sd is 4× the pre arm's, concentrated
entirely in one visit (per-visit sd 0.0055 then 0.0343), and one post sample reaches into the pre
range. But the **per-visit means do not overlap** — pre {0.9658, 0.9690} against post {0.9351,
0.9378} — and 34 of 36 pairwise comparisons put post below pre. The direction is solid; treat the
**magnitude as ~3% ± a point**, not as 2.96%.

**Locus NOT isolated — this is the open part.** v1.17.0 brings two new kernels onto goinfer's paths
(see `f33fcaf`): a new AVX2 int8 routine behind `w8a8Span`, and a reworked inner loop in the blocked
f32 matmul. This benchmark is int8int8 at M=1, so the int8 kernel is the *likely* locus and the f32
blocked path is barely exercised — **that is inference, not measurement.** An ablation would pin it;
nobody has run one. Do not repeat the inference as a finding.

**Scope.** One model, one quantization, batch=1, one box. It says nothing about prefill, where the
changed f32 blocked path actually lives and where the upstream optimisation is presumably aimed — a
decode regression is consistent with a prefill win. Measure that before concluding the change is bad
rather than mis-shaped for this workload.

*Action:* report upstream with the method above. Not urgent — 3% of decode on one shape, against a
bump whose bit-identity is gated and green.

*(That action was taken. It produced aikit v1.17.1 the same day — see statement 4 above.)*

</details>

**P8 · `sampleChunked` allocates a full-vocab `[]float64` and rebuilds the goroutine pool twice per
sampled token** — `decoder/sampler_chunked.go:188`. **TRIED AND REVERTED — the allocation removal
costs 5–6% throughput.**

**Not frozen** (checked, not assumed): `decoder/sampler_chunked.go` and `decoder/sampler.go` are
absent from `testdata/parity_manifest.json`'s 21 files, so this does not re-stale any family's
`deps_hash`. Sampling is not forward numerics. (`decoder/mlp.go` and `decoder/weightmat.go` — P6 and
P7 — **are** in it, confirming those two are genuinely blocked.)

**The change:** hang the full-vocabulary `exp()` scratch off the `Sampler` and reuse it, rather than
`make([]float64, vocab)` per draw. Safe by the type's existing contract — a `Sampler` holds a
`*rand.Rand` and appends to `history` without a lock, so it was never usable across goroutines.

**Measured, `BenchmarkFilterNew262k`, 5 runs of 400 iterations each:**

| | ns/op median | B/op |
|---|---|---|
| before | 6,344,898 | 2,150,668 |
| after | 6,682,883 | **58,732** |

Allocation drops **97%** and throughput drops **5.3% median / 6.3% min**, with the two distributions
**not overlapping** (before max 6,391,328 < after min 6,649,446). So this is not noise.

**Reverted.** P8 was filed as a jitter reducer, and paying 5–6% of throughput for it is a bad trade
against jitter nobody has measured. **No mechanism proposed** — the obvious guesses (page-fault
behaviour on fresh spans, aliasing or bounds-check effects from a field-derived slice) are exactly
the premature-mechanism shape, and none was tested.

**Two discriminators run, both NEGATIVE.** Medians, `BenchmarkFilterNew262k`, 4 runs × 400 iterations:

| variant | GOGC default | GOGC=off |
|---|---|---|
| baseline (`make` per call) | 6,221,779 | 6,215,477 |
| pooled, inline arg | 6,690,158 | — |
| pooled + hoisted local | 6,585,193 | 6,700,647 |

- **(a) codegen — NOT the cause.** Hoisting the field-derived slice into a local at function entry
  recovers ~1.6 of ~7 points. The gap does not close, so **no systematic grep for the pattern is
  warranted** — that follow-up was conditional on (a) closing it, and it did not.
- **(b) GC — EXCLUDED.** The gap survives `GOGC=off` and is slightly *wider* there (ratio 1.078
  against 1.058).

**Cause unidentified, and no mechanism is proposed.** What remains untested is memory/page behaviour,
which is where the guesses point and precisely why they are not written down as findings.

**What would make this landable:** measure the jitter P8 exists to reduce, so the trade has two
numbers rather than one. Until then the allocation stays.

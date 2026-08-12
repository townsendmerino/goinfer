# Work queue — the shared, claimable list

> **Why this file exists.** The queue used to live in conversation, where only the top of it gets
> restated each turn and everything below silently sinks. Three items aged out that way — the Metal
> consumer window, the out-of-tree consumer audit, the drain fix's CUDA verification — none through
> carelessness. And two boxes pulling from the same unstated queue independently built two
> mechanisms for running the heavy tier, because neither could see the other's progress.
>
> That makes the conversational queue an instance of the class this fortnight has been cataloguing:
> an artifact that exists and is not composed into any decision. A check that cannot fail.
>
> **This file is the queue.** If it is not written here, it is not queued.

## How to use it

- **Claim before starting.** Move the item to `In flight` and put your box and the date on it.
  A claim is what stops the other box duplicating it.
- **Release on finish** — move to `Done` with the commit, or back to `Queued` with what you learned.
- **Never delete an item to tidy up.** Strike it with a reason, so "we decided not to" is
  distinguishable from "it sank".
- **Add the whole item, not a title.** Enough that whoever picks it up does not have to reconstruct
  the context from a transcript they may not have.

Boxes: `linux` (nvidia-rtx2070s, CUDA) · `mac` (Apple Silicon, Metal).

## In flight

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

Ordered roughly by priority within each group. Each item carries enough context to be picked up
cold. Where something is believed done but unconfirmed, it says so — **verify before striking**.

### A. Open investigation

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

**A3 · Make the launch OOM say what it is** — `linux`, **DONE `e42e83e`**

The one failure that has resisted a day of investigation also produces the least informative
message: a raw `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY` with nothing tying it to the cache
setting the user chose. The decline floor added in `7c91ccc` does not catch it — that fires below
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
down that `slotcap_test.go` corroborates production sizing: it corroborates a parallel copy.

**Landed and verified on hardware through the shipping auto-cap path**, no manual slot count:
requesting 128 on the real 26B logs `capping to 33` and generates 4 tokens, where it previously
logged `capping to 34` and generated 0. Free after `allocSlots` is 450,494,464 B — byte-identical to
A7's manually-set 33-slot run, so the search lands exactly where the measurement said.

Two withdrawn claims went with it. `slotcap_test.go` said it "corroborates the sizing" (it
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
`cuda/backend.go:793`. The cap is then correct **by construction**, because the free reading it is
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

**A10 · The ~150 MiB driver allocation floor** — `linux`, **THE OPEN CUDA ITEM.** Mechanism measured,
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

**A9-RESID · The 589,824 B is baseline variance, not reservation variance** — `linux`, **CLOSED**

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
`allocSlots` runs at `cuda/backend.go:793`. Under lazy loading the module's *device* memory is not
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

**B0 · Repo-hygiene group must run what CI runs** — `linux`, **DONE `0c54e35`**

`scripts/ci_checks.py` parses `.github/workflows/ci.yml` and emits the hygiene-class steps; group 5
runs them. **Derived, not duplicated** — a check CI gains appears in the gate with no edit to the
gate. 21 steps across 7 jobs; 13 run here and pass, 8 are darwin-only and are a **counted skip**
naming the platform.

The old block was a strict subset of CI: no staticcheck at all, `vet` without the
`goinfer_testhooks` tag and over narrower packages, no build, no module-boundary guard.

**The environment turned out to be part of the check**, found rather than reasoned. CI's root job has
no `go.work`, so the module-boundary guard sees the root module graph in isolation; this box has a
committed `go.work` that unions every submodule, so the guard reported a **false red** on its first
run, naming `cuda`, `gpu` and `webgpu` as leaks. Derived fix: a job with a `workspace` step runs with
one, a job without runs `GOWORK=off`. **Reproducing the command without reproducing the environment
is not reproducing the check** — worth carrying to any other "run what CI runs" work.

Mutation-checked both directions: a gofmt violation turns root gofmt and staticcheck red; breaking
the derivation's selector makes it **refuse** ("derived ZERO hygiene steps") rather than report an
empty set as a pass. The second is the one that matters — a derivation that degrades to nothing
looks exactly like a clean run.

**B0a · A guard that cannot find its tool must fail, not skip** — `linux`, **AUDITED — no live
instance in the repo. Closed with the residual risk named.**

The shape: a check ran as `command -v staticcheck >/dev/null && staticcheck ... | head`. The binary
exists but is not on `PATH`, so `command -v` failed, the `&&` short-circuited, and the whole check
evaluated to nothing while the surrounding output looked exactly like a clean run. It was reported as
"clean".

**Audit result: the repo does not do this.** Three `command -v` uses, all in `scripts/gpu_gate.sh`,
all `nvidia-smi` **backend detection** rather than tool-guarding a check — and each emits a counted
verdict on the absent path (group 0: `skip "clean-GPU check (no nvidia-smi; ...)"`). Absence there is
a real condition about the machine, not a missing instrument.

**And the class is structurally prevented**, which is the better answer than "we checked". Group 6
reconciles **emitted** verdicts against a **declared** set, so a group that dies or short-circuits
without emitting fails the gate — silence is detectable by construction, not by remembering.

**B0's new group 5 is correct by the same standard**, mutation-checked two ways: PyYAML unavailable →
`ci_checks.py` exits 2 with a message and group 5 **fails**; the script missing entirely → non-zero
and empty output, group 5 **fails**. Neither degrades to the old hand-written list, which would have
looked like a pass.

**Residual risk, named because it is the one that actually bit:** the instance was an **ad-hoc shell
command typed at a prompt**, not repo code. No gate polices that. The mitigation is the habit the
gate exists to replace — run `gpu_gate.sh` rather than hand-rolling the check — which is exactly what
B0 makes worth doing.

**B5 · `RELEASING.md` must reference `QUEUE.md`** — either box

A file nothing reads is accurate today and inert the first week nobody opens it — the pattern this
queue was written to fix, applied to itself. A tag is the natural moment to review what is
outstanding, and it is a checkpoint that already happens. Cheap; do it before it is needed.

**B2 · Gate reconciliation — one entry point** — `linux`

Two mechanisms now exist for running the heavy tier: `gpu_gate.sh` group 2c (linux) and
`heavy_gate.sh` (`8fecfad`, mac).

Resolve to one: **`gpu_gate.sh` always declares the heavy group.** When not requested it emits a
counted skip with its reason and the verdict line carries it. Fast runs stay fast; no run silently
omits the tier. `heavy_gate.sh` becomes the implementation group 2c invokes, or it goes. Two files
is fine; two entry points isn't, because **the verdict has to come from one place**.

**B3 · Re-tier by cost** — `linux`

`GOINFER_HEAVY_TESTS` gates "needs a real model" and is used as "slow".
`TestSplitKV_bitIdentical` asserts bit-identity at 2048 context in 13 seconds behind two flags,
while 26B streaming runs 5m16s behind the same one.

Rule: **anything asserting a claim the README makes runs by default.** Census is gathered — 26
heavy-gated tests, with `TestSplitKV_bitIdentical`, `TestPrefillDivergenceRate` and
`TestArgmaxTieBreak` all backing published claims. Report the resulting tier membership so the
split is reviewable.

**B4 · Label or drop `stash@{0}`** — **REOPENED. Absent on `linux-62gb`, UNSEARCHED on
`macbook-arm64`.**

`git stash list` is empty in all four repos here. That is a result about **this box**, and closing
the item on it would repeat the exact distinction the SHA lint learned this turn: a search that could
not have seen the object does not report its absence. `stash@{0}` may only ever have existed on the
mac. **In the mac batch below.**

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

1. **Push `metal-rope-merge`** so `d682315` resolves from anywhere and P4's "already implemented,
   snapshot-golden byte-exact" becomes checkable. It does not need merging to be safe.
2. **B4's stash check** — `git stash list` in all four repos; the stash is absent here and unsearched
   there.
3. **C3, the Metal consumer window** — the largest completely uncovered surface, and the
   highest-priority of the three items that sank before this campaign started.

**Still outstanding, and it needs the mac:** `metal-rope-merge` carrying `d682315`. It is not on
origin and resolves in no clone here, so **P4's "already implemented, snapshot-golden byte-exact" is
unverifiable from any machine but that one**. Pushing the branch is enough — it does not need merging
to make the claim checkable.

**B4 (original) · Label or drop `stash@{0}`** — superseded

"item2 unload-close fix + tests (wip)", `admin.go` +32. Almost certainly adds `Close()` to the
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
| `decoder/forwardn.go:502` | unchanged — `decoder/` is under the `6edd1ca` numerics freeze |
| `decoder/model.go:731` | unchanged — same freeze |
| `metal/model.go:827` | unchanged — Metal is on hold for core-numerics surfaces |

The three unchanged members are a **deliberate** partial fix, not an oversight, and they are the
reason this row exists: had P3 been taken at face value and only `cuda/resident.go` parallelised,
even the second CUDA site would have drifted from it. Adopting `applySoftcap` (or its equivalent) at
the remaining three is the work that closes the row, and it unblocks with the freeze.

**Enumerate the members; do not name one.** A test that names one member is exactly what the passing
sibling already had — it reproduces the class rather than closing it. Where enumeration cannot be
mechanical, the invariant's own comment carries the full set, so the next fix is written by someone
who has been told the set exists.

### C. Verification surfaces never exercised

**C1 · Drain fix — CUDA verification** — `linux`

Prompt already written. The admin-unload drain (`588052b`) is verified on Metal: unload freed
325 MB and reported `freed:true`, against 95 MB before. CUDA is the arm that can't run there, and
it's the backend where `Close` does the most — pinned host memory, mapped expert stacks, CUDA
graphs, then `ReleaseObjects` routed through the pinned executor via `reqCh`/`ackCh`. That teardown
meeting the drain is the untested interaction.

Four parts: VRAM reclaim across load/generate/unload/reload; the `preamblePark` regression test
under `-tags goinfer_testhooks` against CUDA; the straggler case if adapters are loadable; and
`--unload-drain-wait`'s 5s default under a real generation.

**C2 · Out-of-tree consumer audit against v0.11.0** — fresh session, **no repo access**

Prompt already written. More valuable now than when drafted, because the README has since acquired
many specific provenanced claims — deployment size, JIT timings, the depth curve, the configuration
sweep, request-body caps, unload semantics. The audit's Tier 1 is a claim-by-claim check from
outside, which is the only instrument that tests claim-discipline rule 7 ("a claim nobody can
reproduce from the public documents is not shipped") **from the position the rule is about**.

Must run blind: no clone, no source, no test suite. Carry the known-findings list so nothing is
rediscovered.

**C3 · Metal consumer window** — `mac`

The largest completely uncovered surface. Nothing has tested Metal from outside. Claims attached:
cgo-free with no Xcode, 73.6 tok/s, 0.96×/0.74× against Ollama-Metal, bit-identity within machine
and OS.

Sharpened by two things found since: `TestMetalSnapshotGolden` is a §4 gate that **cannot fail** (it
drives `Forward`/`ForwardArgmax`, which apply no embed scale, where production applies √hidden), and
the Metal device suite **doesn't run in GitHub CI at all** — the runner's paravirtual objc layer
SIGSEGVs inside purego. So Metal's entire device coverage is one manual box run, behind an
unfalsifiable gate.

The tautological-gate shape was found on CUDA today (four graph tests comparing graphs-on against
graphs-off without asserting graphs were admitted). **The same shape is plausibly live on Metal and
nothing would say so.**

**C4 · Soak testing** — either box

Nothing has run longer than minutes. The G1 memory-plateau finding rested on **75 seconds**. Memory
growth, KV cache reuse, session accumulation, fd leaks, thermal behaviour over hours are all
unobserved.

### D. Structural work, sequenced

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
error accumulator stays intact and `prefill.go` and tests share the wrappers.

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

Commit 1 changes no call sites and must be provably inert. Then migrate per file (`resident.go` 36,
`prefill.go` 11, testhooks 1) with a trace comparison at each step.

State the **641 → 0** figure in the commit message with its limit: zero counts cross-name
transpositions the type system prevents; passing a wrong *value* of the right kind still compiles.
The failure moves from an invisible positional slip to a legible mis-assertion at the call site.
**Do not write "eliminates transposition bugs".**

**D3 · The parked flag-pair** — `linux`, **unblocked by the proof requirement; belongs in the
release if its rebase is clean**

**Does it complete the MoE-cache story the release headlines? YES.** `c8b65ba` adds
`--moe-cache-experts` / `--moe-cache-slots` to `serve`, and the README instructs the **env vars in
three places** — including the very section this release rewrites around the cap fix. Shipping the
release without D3 means rewriting that section again next version, for the same feature.

**But its branch predates the fix it completes.** `flag-pair-moe-cache` is based on `7ccec1e` — the
slot-default commit that was **reverted** — and it touches `cuda/backend.go` and `decoder/model.go`,
both of which A5 (`6091e7a`) and A9-FIX (`0103b49`) changed substantially. So this is a **rebase and
re-verify**, not a merge.

**OUT OF THE RELEASE. Do not attempt the rebase yet — the first question is not a merge question.**

D3 was designed **while the cap computed the wrong value**. A5 fixed the cap. So before anything:
**does the flag pair still have a reason?**

- **If the flags exist to work around a cap that could not size the cache correctly** — that reason
  is **gone**, and shipping them would document a control whose justification was removed. The item
  **closes** rather than rebases.
- **If they exist for legitimate explicit override** — a smaller cache than the correct cap, chosen
  deliberately — they stay. But the **defaults and the docs were written against the old behaviour**
  and both need re-deriving against the corrected cap.

**A clean rebase would not distinguish those two.** Read the design, not the diff. Scheduled after G2.

**D3 (original) · blocked on the freeze** — superseded

`flag-pair-moe-cache` (`f6bbf7c`) carries `--moe-cache-experts` and `--moe-cache-slots` as CLI
flags. The `Options` fields and accessors touch `decoder/model.go` and `gguf.go`, which re-stales 19
families' `deps_hash`. `BRANCH-NOTE.md` records the pickup steps and the instruction that matters:
**run the goldens, do not refresh `deps_hash` to quiet the gate**.

Precedent exists for a goldens-gated refresh (`9e5f8fa`, where a metadata field addition re-staled
`weights.go` and the refresh ran 19 goldens). It was deliberately not spent on ergonomics.

### E. Release and claims

**E1 · v1.0 gate as written criteria** — `linux`

Decisions already taken: the parity backfill lands as **v0.13.0** (moved from v0.12.0, which this campaign took — decided 2026-08-12), not v1.0. v1.0 gets its own gate
requiring parity coverage complete, the verification machinery sound, the loader and
architecture-descriptor surface **actually frozen** (the docs still say it may change), and a clean
out-of-tree audit against the release candidate.

Write that as a checklist so 1.0 is a decision against criteria rather than a feeling.

**E2 · The four per-family demotion judgments** — `linux`

`gpt2`, `granitemoehybrid`, `kimi_k2`, `nemotron_h` carry `validated_at: null` and are the same four
the `deps_hash` tripwire does not enforce — so 19/23 tracks both the backfill's progress and the
tripwire's coverage, and clearing it closes both.

Rule: every family claimed as supported at v1.0 has a current T3 row; families that can't get one go
experimental. **Honesty test per family — would you move it to experimental if no release were
pending?** Structural reasons qualify (no reference, fixture size, licence). "Unfinished" does not;
demoting unfinished work to clear a release hollows out the tier permanently.

**E3 · Freeze re-declaration** — `linux`, **inventory taken and the condition drafted; see below**

**THE FREEZE-BLOCKED INVENTORY, read rather than grepped** (21 frozen paths, from
`testdata/parity_manifest.json`'s `shared_sets`):

| column | item | what blocks it |
|---|---|---|
| **freeze-only** | **D3** the parked flag-pair | `Options` fields touch `model.go` + `gguf.go`; re-stales 19 families |
| **freeze-only** | **G2** `go fix` modernizers | re-stales the manifest wholesale |
| freeze **plus other** | **P1** KV re-gather / V re-transpose | freeze **+** a new aikit row-pitch API **+** E6's deferred aikit release |

Everything else touching a frozen path has **landed** (P6 `eea7f29`, P7 `91f359f`) or only references
those paths as instances (B6, P8 — `sampler_chunked.go` is not in the manifest).

**So the freeze-only column is TWO items — and that is the answer to what an unfreeze buys.** It is
smaller than it looks, because both are landable *today* under the goldens exception, exactly as P6
and P7 were: the cost is a ~33-golden run, not a blocked queue.

**THE UNFREEZE CONDITION, drafted as a capability rather than a version number:**

> The core unfreezes when a change to a frozen path receives numeric proof across the **loader** and
> **quantization** axes it can affect, demonstrated by a gate that **prints its composition**.

**What remains unmet: nothing.** Checked against the axes, not against a summary:

| axis | release gate | goldens refresh (the freeze-exception path) |
|---|---|---|
| quantization | f32, int4, int8, int8int8 | f32, int4, int8, int8int8 |
| loader | safetensors, gguf | safetensors, gguf |

Both print their composition (`scripts/sweep_composition.py`, and the refresh's own
"33 passed / 14 quantized" line). **The loader axis was the open question and it is covered** — but
only since this turn, and only because the GGUF parity gates entered the selector: before `f9d5d07`
the refresh was safetensors-only on loader as well as f32-only on quant.

**THE FREEZE, RE-DECLARED AS A PROOF REQUIREMENT.** It has functioned as one all day — every
frozen-path change that landed ran the goldens, and none was refused:

> **Changes to paths covered by `testdata/parity_manifest.json` require a goldens run whose axis
> composition is printed with the result. No version gate, no per-change exception.**

**Decider: Francis. Declared 2026-08-12.** Recorded with an author because a rule with none drifts
back into habit.

**Justifying inventory:** the freeze-only column is **D3** and **G2**, both landable under this rule
today; **P1** is blocked on the aikit row-pitch API and E6 independently. **Lifting the freeze as a
freeze buys nothing the rule does not.**

**THE AXES, and why ARCHITECTURE is excluded — stated, not left silent.** The condition names
**loader** and **quantization**. It does not name architecture, and the reason is measured rather
than assumed: arm64 contracts `x*y+z` at **85 decoder sites** where amd64's baseline contracts none,
and the FMA campaign measured **114,431× minimum headroom with no argmax flip**. A separate arm64
run is therefore very likely unnecessary, and 18 of 23 manifest rows are `linux-62gb` anyway.

**The exception, in the same breath:** that headroom was measured **for the code as written**. A
change that **rewrites expressions** rather than allocations puts it back in scope, because the
measurement does not survive the expressions changing. **G2 is exactly that change class** — which is
why it gets the check below rather than a wave-through.

The `6edd1ca` freeze remains in force; tagging on top of it touches no core numerics and does not
lift it. But it needs re-declaring in a **live document** with scope, an explicit lift condition,
and who decides — rather than being reconstructed from a commit several tags back.

Enforced scope, now quantified: **19 of 23 families, `decoder/` surface only, zero GPU coverage.**
No `cuda/` file appears in the manifest at all.

And answer, rather than leave as an absence: **should `cuda/` files be in the parity manifest**, or
are the resident parity gates the right home for that guarantee with the manifest deliberately
CPU-only? Note that until B2/`gpu_gate.sh` ran the parity gates, GPU forward numerics had no
enforced signal in the release gate — not a staleness tripwire, not a parity assertion.

**E4 · `scripts/bench_compare.sh` — fix or retire** — `linux`, **status unconfirmed**

It measures goinfer with in-process Go benchmarks and never drives the peer, which is what made the
476/268 headline divide a kernel throughput by an end-to-end one. The README's false-rigor sentence
is gone, but **if the committed artifact still measures the two sides differently the gap reopens
the next time it runs.** Either make it produce a defensible server-to-server comparison, or remove
it and record that peer figures are measured manually with the procedure written down.

**E5 · Promo drafts** — Francis / Claude

Blocked on nothing now. They need **rebuilding, not editing**: written for v0.9.0, quoting withdrawn
476/268-era figures and the pre-fix peer table, carrying the 26B claim without its configuration,
and predating the `top_k` guidance and the §5 bit-identity correction. Claude holds them and will
rebuild against current numbers on request.

**E6 · aikit release** — `linux` or `mac`

**Deliberately not cut.** `be049df` (aikit: *gpu(gemv): explicit `__fmaf_rn` in the quantized GEMV —
the bit-identity contraction rule*, 2026-08-04, in six tags `gpu/v0.25.0`…`gpu/v0.27.0`) — its FMA fix
is already released (contained in `gpu/v0.25.0`
onward, and goinfer requires `gpu/v0.27.0`), and the unreleased diff is two test files plus
comment-only edits with byte-identical PTX. The rule recorded: **a release needs a reason a consumer
can receive**; test coverage, lint rules and CI are properties of the repository, not of the
artifact. The three gates and the first-ever GPU CI job ride with v1.0, where they are a line item
rather than the whole changelog.

Also open there, deliberately: branch protection is not enabled and `gpu-kernels` is advisory.
`scripts/gpu_gate.sh` plus a `RELEASING.md` gate ritual is the enforcement instead. Revisit at v1.0.

### F. Audit backlog

**F1 · §4 gates — five still open** — `linux`

G-03 closed today (`buildMatrix` env-pinning, via the `GOINFER_GEMMA4_RESIDENT` flip). Remaining:

- **G-01** — `TestResidentAdmission_matrix` tautological
- **G-02** — Metal snapshot golden drives `Forward`/`ForwardArgmax`, which apply no embed scale
- **G-04** — `metal/model.go:590`, `case "slots"` doesn't assign `r.residencyBufs`
- **G-05** — tokenizer/chat tests probe `/home/francis/models/...` with no committed fixture
- **G-06** — hardcoded developer-home paths across many files; no `GOINFER_MODELS_DIR`

G-06 is now partly subsumed by B1's registry, since the paths and the env surface are the same
problem seen twice.

**F2 · §2/§3 open criticals** — `linux`

Roughly eleven remain. The ones with the sharpest consequences:

- **C-08** — `cuda/resident.go:244-256`, `_ = gpu.Upload(...)` in `up32`/`upu32`/`upu16`;
  `r.setupErr` declared and read but never assigned, so a failed upload yields `ok=true` over
  zeroed weights
- **C-14** — CUDA argmax reduce has no index tie-break. Now has a funded reason: routing `top_k=1`
  to `ArgmaxEquivalent` recovers 13–18% and is blocked on it, and v0.10.3 made ascending-token-id a
  written contract the device side doesn't honour
- **C-05 / C-06** — gemma-4 stride assumption on snapshot restore, and unvalidated tensor shapes
  driving writes into config-sized scratch
- **C-21 / C-22** — embeddings has no batch cap and is un-queued; shutdown takes an unconditional
  lock after `Shutdown` and swallows a second signal
- **C-30 / C-31** — no mutex in the paging paths; `internal/giw/bundle.go:105` `make([]byte, u32)`
  with no file-size bound

§5 (23 major) and §6 (24 minor) have **never been verified at all** — presumed open pending a
targeted pass.

**F3 · G-01 class — confirm the sub-shapes landed** — `linux`, **status unconfirmed**

The class entry in `parity-coverage-policy.md` should carry **exercised but never triggered** —
`allocSlots` runs in every MoE test and caps in none of them, so a safety branch reads as fully
covered by every measure the project has. Recognition test: **does any test reach the branch**, not
just the function.

### G. New capability, scoped but not started

**G1 · LFM2.5-2.6B as an experimental family** — `linux`

Scoping prompt written. A fifth sequence-mixing family: interleaved gated short-convolution blocks
and GQA, `layer_types` controlling the pattern, `conv_L_cache` 3, LayerNorm QK-norm (not RMSNorm),
FFN dim computed rather than stated. The conv layers carry a rolling conv state instead of a KV
cache.

The estimate turns on two questions: whether Mamba-2's causal depthwise `conv1d` is factored out or
inlined, and whether the cache abstraction already carries mixed per-layer state types
(Granite-4.0-H and Nemotron-H suggest it may). Also unestablished: **whether LFM2.5 is
architecturally the same as LFM2** — the transformers docs cover only LFM2.

Blast radius matters: anything touching shared `decoder/` core re-stales all 19 enforced families.
Answer that before estimating.

**G2 · Items from the Go-for-AI tooling inventory** — either box

- **PGO** — absent from both repos. goinfer's default build is the pure-Go CPU path and this is a
  performance project; 2–7% is typical. Gate it on the parity goldens, since PGO changes inlining
  and inlining could shift Go's permitted FMA fusion.
- **govulncheck** — VERIFY FIRST: goinfer already runs it in CI and it is green (confirmed 2026-08-12), so this is stale for goinfer. aikit may still lack it; the entry should end up saying which rather than being struck entire. Originally: absent from both. For a project whose pitch is one static binary you `scp` and
  run offline, a reachability-filtered vulnerability statement is part of the deployment claim.
- **Fuzz corpora** — sixteen fuzz targets across the two repos, three committed corpus directories.
  A crasher found once and not committed is found again next year. The audit's hostile-input
  findings should each be seeds.
- ~~**Execution tracing**~~ — **DONE (2026-08-11).** `go tool trace` on `BenchmarkDecode` (0.5B,
  M1 Pro) resolved it: the "~8 ms host cost" / "71% `pthread_cond`" is an **idle-M sampling artifact**,
  not a recoverable cost — serial (zero fork/join) ties parallel in tok/s, the trace's real
  scheduler-wake tax is ~1%/token, and pprof's `pthread_cond` samples are parked idle workers between
  dispatches (a CPU profiler counts them, a wall-clock trace shows them idle). The right tool
  dissolved the question. Confirms the Phase-3b pool-null-result. Writeup: perf-campaign.md
  "Profiling coda". (Lesson: for park/wake questions use `-pprof=sync`/`-pprof=sched` from `trace`,
  not pprof CPU, which can't tell critical-path stall from an idle parked M.)
- **`go fix` modernizers** — one deterministic pass, reviewed as a diff. **CLEARED FOR THE amd64
  GOLDENS RUN ALONE; no arm64 read needed.** Checked before running, in an isolated `git worktree` so
  the real tree could not be touched:

  **21 of the 22 registered analyzers are numerics-inert by construction** — API/idiom migration
  (`any`, `fmtappendf`, `mapsloop`, `newexpr`, `omitzero`, `reflecttypefor`, `slices*`, `stditerators`,
  `strings*`, `testingcontext`, `waitgroup`, `inline`), loop/scope forms (`forvar`, `rangeint`), build
  directives (`buildtag`, `plusbuild`), and one diagnostic-only (`hostport`). None rewrites an
  arithmetic expression.

  **`minmax` is the one that could**, and it is the reason to check rather than assume: it replaces
  `if a > b { m = a } else { m = b }` with `max(a, b)`, and Go's builtins **propagate NaN** where the
  if/else form does not — a real behaviour change in a float path. Its candidates in `decoder/` are
  **7, and every one is integer** dimension or index arithmetic:

      ge := min(gs+group, cols)                                  end := min(g+int4GroupSize, len(row))
      sc := max(moe.SharedIntermediateDim, moe.IntermediateDim)  b := min(32, n)          (x2)
      window := max(len(access)/8, 1)                            workers := min(GOMAXPROCS(0), numChunks)

  **Censused across G2's ACTUAL scope**, not just `decoder/` — 9 candidates, **all integer, zero
  float**:

  | package | candidates | float | integer |
  |---|---|---|---|
  | `decoder` | 7 | 0 | 7 |
  | `cuda` | 2 | 0 | 2 — `softcap.go`, worker count and chunk bounds |
  | `gpu` | 0 | — | — |
  | `metal` | 0 | — | — |
  | aikit | 0 | — | — |

  **No float `min`/`max` anywhere, and none of the 85 contraction sites is touched.** The headroom
  measurement survives and G2 needs no scope narrowing.

  **WHAT CLEARED G2 WAS SOURCE ANALYSIS, NOT THE GOLDENS RUN — and the distinction is load-bearing.**
  A float `minmax` rewrite differs from the if/else form only on **NaN**, and NaN paths trigger on
  degenerate inputs while goldens run normal ones. Such a rewrite would have landed **green** and sat
  dormant until a real NaN arrived. That is exercised-but-never-triggered, in the one change class the
  proof requirement above does **not** cover — the requirement proves numerics for the inputs the
  goldens carry, and this class changes behaviour only outside them.

  So do not let "goldens green" later read as the authorization for G2. **The authorization is the
  census**, and it must be re-run if the analyzer set changes.

- **D3** has no expression-rewriting exposure at all (it adds `Options` fields and accessors), so it
  proceeds on the goldens run alone.

### P. Audit findings, 2026-08-12 — nine survived adversarial verification

Eight are below. The **ninth is the Metal `ResidentGreedy` gap**, filed under Struck rather than here
because it is measured net-negative and therefore not work — the count is stated so its absence from
this list reads as a decision rather than as a dropped item.

**Every figure below is a verifier's ESTIMATE, not a measurement.** Written with that word attached
deliberately: these came from reading code, not from running it. Any figure later measured **moves
to the measured-quantities table** in `parity-coverage-policy.md` with machine, method and date, and
stops being an estimate here.

**P1 · KV re-gather and V re-transpose on every decode token** — `decoder/forwardn.go:378`

Estimated **~10–15% of per-token traffic at 4k+ context**, on all mainstream CPU families — the
largest single item in the group. Frozen core, and it needs a new aikit row-pitch API, so it is the
**v1.0-unfreeze headline** rather than something to slip in.

**P2 · Scalar `int8→f32` widen on the LM head** — aikit `linalg/quant.go:113`, **condition VERIFIED,
and it does not hold as a drop-in**

Not frozen, so the work is unblocked. **And no decision gets reversed: merge into aikit `main` and
leave it UNRELEASED.** The `require` bump is already planned for v1.0, so the win lands on the
schedule E6 already chose — E6 defers the *release*, not the *work*, and banking verified work in
`main` costs nothing and reverses nothing. Recorded this way so it is not re-litigated as an E6
exception every time it surfaces.

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
`scripts/refresh_parity_hashes.sh` — the goldens-gated refresh, precedent `9e5f8fa` — **not**
`parity_sweep.sh`'s T3 oracle sweep, because these are allocation changes rather than arithmetic.

Measured on `linux-62gb`: **19 goldens pass, 11 skip, 0 fail, 6.09 s wall.** One machine, no model
zoo, no HF venv. (18 of the 23 manifest rows name `linux-62gb`; only `gemma4` names
`macbook-arm64`, and that is its *oracle*, not its golden — `TestGemma4MoE_forwardParity` ran here.)

**Coverage is good for P6.** Nine MoE goldens actually RAN: `TestGemma4MoE_forwardParity`,
`TestGemma4MoEKV_forwardParity`, `TestGemma4MoEUnified_forwardParity`, `TestMixtral_forwardParity`,
`TestGlm4Moe_textParity`, `TestQwen35_forwardParity`, `TestDeepseek_textParity`,
`TestKimi_textParity`, `TestLlama4_textParity`. `TestQwen2Moe_forwardParity` skipped.

**Verdict: P6 can land now under the exception.** Do not refresh the hash without running the
script — it refuses on any golden failure and on a vacuous all-skipped run, which is the whole point.

**R1 · The refresh script's history — two corrections to the record** — `linux`, **CLOSED, both
answered from the log**

**Correction 1: "the refresh script had never been usable" is WRONG.** I wrote that in `eea7f29`'s
commit body. The log says otherwise — **nine commits carry its goldens proof**, with counts rising as
fixtures were added:

| date | commit | goldens |
|---|---|---|
| 2026-07-26 | `2e91607` | 14 passed |
| 2026-07-28 | `9624dd9` | 14 passed |
| 2026-08-01 | `ecc5af2` | 119 passed |
| 2026-08-02 | `e58ac8a` | 17 passed |
| 2026-08-02 | `1f6dbe0` | 4 passed |
| 2026-08-03 | `2922468` | 17 passed |
| 2026-08-04 | `7cc2f0d` | 17 passed |
| 2026-08-09 09:58 | `ed81e13` | 19 passed |
| 2026-08-09 15:10 | `ca29d6c` | goldens=19 |

**It worked, was used, and broke on 2026-08-09 at 20:59** — `6edd1ca` introduced `method: null`,
which the writer could not round-trip. `eea7f29` (2026-08-12) is the **first refresh attempted after
that**, three days later, and it aborted. The abort was the guard working on its first real
opportunity, not a guard that had never let anything through.

**Correction 2: the precedent cited in this queue was the wrong commit.** `9e5f8fa` was described
here as "a metadata field addition re-staled `weights.go` and the refresh ran 19 goldens". It is
`fix(quant): reject --quant that conflicts with a prequant .giw at startup` and **touches the manifest
not at all** — its five files are `CHANGELOG.md`, `giwquant_test.go`, `serialize.go` and two `main.go`s.
The real precedents are the nine above. I repeated the wrong SHA several times from this file without
opening it.

**Second-order check: a mangled write DID land, and lived ~7 weeks.** The HTML-escaping defect was not
caught by the guard, because it predates the script and arrived through `go test -run ParityManifest
-update`:

    2026-06-20  82b39cc  \u003e appears (1)
    2026-07-26  93eb7d4  (2)
    2026-07-28  99b3f95  (1)
    2026-08-09  6edd1ca  0 — cleaned

So the answer to "did the guard hold from the start" is **no**: escaped sequences were in the tree
from 2026-06-20 to 2026-08-09. They are cosmetic — a `>` inside a `reference` string — and changed no
hash or verdict, but the claim "a clean result means the guard held" would have been false.
`method: ""` never landed (checked; 4 `null`s today, 0 empty strings).

**What is durable:** the writer is now faithful (`SetEscapeHTML(false)`, `Method` as `RawMessage`), so
neither defect can recur through either route.

**Q2 · The GGUF-quant cross-gate gap — CLOSED, and it was unplumbed too** — `linux`, `bd08936`→

The cross-gate check showed `parity_sweep.sh` covering the GGUF quant formats while the goldens
refresh did not. **(a) Exposure: a LAG, not a hole.** `parity_sweep.sh` is not in CI — it is
release-only, run by hand on the box (`RELEASING.md` §C1). So the formats are covered at release and
**not between releases**, which is exactly when a frozen-core edit gets only the goldens refresh.

**(b) Both routes priced before choosing, and route B turned out unnecessary:**

| route | cost |
|---|---|
| extend the goldens selector to the existing GGUF gates | **26.8 s**, 11 gates, no new fixtures |
| author GGUF-quant goldens for those 11 rows | unnecessary — the gates already exist and already pass |

Same shape as Q1(b): **unplumbed, not missing.** The gates were simply outside `GOLDEN_RE`. Adding
`^TestGGUF_.*_parity$` took the refresh from **19 passed / 0 quantized** at the start of this campaign
to **33 passed / 14 quantized**, and the cross-gate check now reports *"the two gates span the same
quantizations."*

One bug fixed in the cross-gate check itself: it compared a composite label (`int4/int8`, from a file
driving two quantizations) against atomic ones and reported a difference that was purely notational —
a permanent false positive in the check built to make real differences visible. Both sides are
atomised now.

**Q1 · The forward goldens prove f32 ONLY — no quantized path has a golden that runs** — `linux`,
**NEW. G-01 at the largest scale it has appeared.**

int4 is the documented default quantization. **Zero goldens drive it.** And the hole is wider than
that: of the 19 goldens that actually RAN in the 2026-08-12 refresh, **every one is f32**.

| quantization | golden files | did any RUN? |
|---|---|---|
| f32 (explicit or default) | 24 | **19 ran** |
| `int8int8` (W8A8) | 3 — `gemma4_parity`, `gemma4_12b_parity`, `mellum2_parity` | **all 3 SKIPPED** |
| `int8` (weight-only Q8) | 1 — `gptoss_real` | not matched by the goldens regexp at all |
| **`int4` / W4A8** | **0** | — |

So `refresh_parity_hashes.sh` — the sanctioned freeze-exception path, and the thing that makes a
core edit auditable — **proves f32 numerics and nothing else**. A change that is bit-identical in f32
and wrong in int4 passes it in 6 seconds.

**Retroactive scope, and this is the part to act on.** Any claim of the form *"the parity suite
covers X"* is scoped to **the quantizations the goldens drive**, which today is f32. Every place such
a claim is written down needs that scope added — `docs/parity-coverage-policy.md`'s tier table,
`RELEASING.md`'s §C1, the README's support matrix, and the P6 commit body (which states it already).

**And the freeze protects what the goldens check.** The `6edd1ca` numerics freeze over `decoder/` is
enforced by `deps_hash` staleness, whose release valve is this goldens run. Where the goldens are
silent — every quantized path — the freeze is a *procedural* barrier with no numeric proof behind it.
That is not an argument for lifting it; it is an argument for knowing what it is.

**WHY THIS OUTRANKS THE REST OF THE QUEUE — sequencing, not enthusiasm.**

**P1 is the v1.0 headline and lives in the frozen core.** The numeric proof available when that core
unfreezes was **f32-only**. So lifting the freeze did not buy the ability to verify the work the
freeze defers — and the shortfall **would not have announced itself**, because the goldens would pass.
An f32-green refresh over an int4 regression is a passing gate, not a silent one; nothing in the
output distinguishes them.

That makes Q1(c) a **prerequisite for the v1.0 core work**, not a parallel item, and it belongs ahead
of the E-group release gate for that reason rather than because it is interesting. **Done
2026-08-12 (`1d0d1ed`)**: 23 fixtures across 16 architectures, so the prerequisite is now met for
int4 specifically.

**RUN WHAT EXISTS FIRST — and most of it was UNPLUMBED, not missing.** Done 2026-08-12, `23b2ee7`:

- **(b) the three `int8int8` goldens** skipped for one liftable reason, the same for all three:
  `GOINFER_HEAVY_TESTS` unset. **Two of the three pass here in ~70 s** (gemma4, mellum2). The refresh
  now enables heavy by default. The third (gemma4-12B) skips on a genuinely absent GGUF — an asset
  question, not a plumbing one.
- **(a) the `int8` golden did NOT turn out to be a selector bug.** `TestGptOssReal_logitParity` **does**
  match the regexp. It is invisible because `gptoss_real_test.go` is behind `//go:build realckpt`,
  which the refresh does not pass — and with the tag it still skips for a missing GGUF. **Two gates,
  either sufficient.** A one-line regexp change would have bought nothing.

**Non-f32 rows after (a) and (b): 2** (21 passed, 2 quantized). The distinction the ordering was meant
to test comes out clearly: **int8 was unplumbed** (one env var), **int4 is genuinely missing**, and the
gpt-oss int8 row is **asset-blocked behind a build tag**.

The refresh now also prints the **quantization breakdown**, because "19 passed" and "21 passed" read
identically to a human and that is precisely how this stayed invisible through nine prior refreshes.

**(c) int4 goldens — DONE `1d0d1ed`.** Scope measured *before* authoring and stated as a target: int4
has no divisibility constraint (`nGroups` is a ceiling divide), so eligibility was never the limit —
fixture availability was. **Target: 23 fixtures / 16 architectures. Delivered: 23 / 16.**

The goldens compare **int4 output against recorded int4 output**, not int4 against f32 within a
tolerance. A tolerance band against f32 measures quantizer loss — a real question with its own gate
on the policy's quant axis — and would read as "int4 is covered" while proving nothing about whether
the W4A8 path still computes what it computed yesterday. Only the self-comparison catches a
regression in the path the freeze protects and P7 will change.

Fixtures are **enumerated** from `testdata/` rather than listed by name, so a new family is picked up
without editing the gate, and a run comparing **zero** fixtures **fails** rather than passing.
Mutation-checked by perturbing the quantizer itself (`int4GroupSize` 32 → 64 → red).

Recorded **absences**, not gaps: `gpt_oss` (MXFP4-prequant, rejects a conflicting `--quant` by
design), `siglip_vision_model` (an encoder), `gpt2` / `mellum` / `qwen2` / `qwen3` (no tiny
safetensors fixture), `qwen2_moe` and `gemma4-dense-scaled-{24,48,64}` (incomplete fixture dirs).

**Refresh now reports 22 passed / 3 quantized**, against 19 passed / 0 quantized when this began.

**Also record with P6's 6.09 s price: cheap and thorough are different properties.** 6.09 s buys 19
passes and 11 skips. The skips are not free — they are the coverage this item is about.

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
(`weightmat.go:202` and `:219`, the same pattern twice). The divergence is elsewhere: `matmulInto`
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

## Struck — decided against, kept so the decision is visible

- ~~**Default `top_k`**~~ — truncating the distribution changes which tokens are reachable, which is
  a silent substitution of something other than what was asked. Document it; do not default it.
- ~~**Change the global `--quant` default**~~ — CPU inverts the CUDA quant ordering, so a single
  global default cannot be right for both, and the evidence is one model on one box, never
  reproduced at 1.5B.
- ~~**Force cross-architecture float agreement**~~ — explicit `math.FMA` everywhere is a software
  fallback on amd64 that costs the SIMD performance the CPU backend exists for. Scoped in the policy
  instead.
- ~~**Slab restructure for expert slots**~~ — the control produced the reverse of fragmentation's
  prediction: a fresh heap with ~10 large allocations had *worse* contiguity (32–64 MiB) than the
  slot-loaded heap (96–128 MiB) at the same free figure.
- ~~**aikit branch protection**~~ — required checks force PR-only merges, which is friction against
  a threat model aikit doesn't have. The gate ritual is the enforcement. Revisit at v1.0.
- ~~**Metal `ResidentGreedy` gap**~~ — measured **net-negative**. Kept here rather than under group
  P because it is not work. The 2026-08-12 audit reached the same conclusion **independently**, from
  code, without access to the measurement — recorded as a corroboration of that audit's calibration,
  which is the only reason the entry is worth keeping at all.

## Done

_(append with commit sha and date)_

## Sequencing — release BEFORE G2

**Revised, and D3 is OUT of the release — no rebase attempted.**

> **cut the release → G2 → D3 design read → B1, B2 → mac batch**

The README change in this release is a **retraction**: the workaround language goes away because the
cap holds. D3, if it survives its design read, is an *addition to adjacent text later* — not the same
edit made twice, which was the argument for including it.

A repo-wide mechanical diff immediately before a tag costs bisectability and reasoning room and buys
the modernizers nothing. G2 is not urgent and never was; it is cleared, which is different from being
next.

## Draft: contents of the next release

**Not a version number** — that is a separate call. This is what has accumulated since
`demo/agent/v0.11.0` (93 commits) that a user would notice, and **none of it depends on the freeze
decision**.

### The headline: the 26B expert cache sizes itself correctly

The defect that opened this campaign was live in the product and is fixed. On an 8 GB card the
runtime auto-capped the MoE expert cache to **34 slots/layer, which allocates and then cannot
launch** — the forward produced **zero tokens**.

- **A5 (`6091e7a`)** — the cap is a **search over the granularity form**, not a division. The driver
  charges each of four buffers per layer its own whole 2 MiB quantum, so the requirement is a step
  function; at 34 all four tip at once, putting it 203,816,960 B over free. Verified through the
  shipping auto-cap path: `capping to 34` → 0 tokens becomes `capping to 33` → coherent output.
- **A9-FIX (`0103b49`)** — the deferred first-launch reservation (`moe_route` takes 138,412,032 B of
  local memory the first time it runs) is now paid **before** the free reading that sizes the cache,
  so the cap is correct by construction rather than covered by a margin. Costs two slots, and that is
  the point: 384 MiB now means 384 MiB.
- **A3 (`e42e83e`)** — a launch OOM now names the kernel and **both** the requested and effective
  slot counts, instead of a bare `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY`.
- **README** — the manual-workaround section is retracted and replaced with what the cap now
  accounts for, plus a version test (`capping to 33` has the fix, `34` does not).

### Performance, all bit-identical

- **P3 (`4c26a58`)** — Gemma's final-logit softcap parallelised: **1.43 ms → 640 µs** per sampled
  token at 262,144 vocab. Sampling path only; greedy never paid it.
- **P6 (`eea7f29`)** — MoE experts share one gate/up buffer pair per token instead of one per expert:
  **16 allocations → 2** at top-k 8.
- **P7 (`91f359f`)** — W4A8 reaches the per-stream `Workspace` it was silently excluded from, ending
  a fresh allocation per projection per token.

### Verification a user can check

- **int4 forward goldens** (`1d0d1ed`) — 23 fixtures across 16 architectures. int4 is the documented
  default quantization and **nothing gated it** before this.
- The goldens refresh went from **19 passed / 0 quantized** to **33 passed / 14 quantized**, and now
  prints its composition rather than a bare count.

### Known-unfixed, disclosed

- **A10** — a ~150 MiB driver allocation floor: memory `cuMemGetInfo` reports as free and
  `cuMemAlloc` will not hand out, at any request size down to 1 MiB. Measured, unattributed. It is
  why the margin cannot simply be lowered to recover the two slots.

<!-- sha-lint: allow d682315 UNPUSHED — Metal branch `metal-rope-merge`, mac-local; not on origin and not in any clone here. Owner: whoever cited it. P4's "already implemented, snapshot-golden byte-exact" rests on a commit only that machine can see; push the branch or the claim stays unverifiable from anywhere else. Flagged 2026-08-12 -->

<!-- SHA-INDEX: generated by scripts/queue_sha_lint.py --update; do not edit by hand -->

## SHA index

Generated. Every commit id cited above, with the subject it resolved to at the time
of generation. Regenerate with `scripts/queue_sha_lint.py --update`.

| sha | subject |
|---|---|
| `0103b49` | fix(cuda): pay the deferred reservation before sizing the cache (A9-FIX) |
| `0c54e35` | fix(gate): repo hygiene runs what CI runs, derived from ci.yml (B0) |
| `1d0d1ed` | test(decoder): int4 forward goldens — 23 fixtures, 16 architectures (Q1c) |
| `1f6dbe0` | fix(parity,fmt): gofmt the threshold sweep + refresh deps_hash after comment-only core edits |
| `23b2ee7` | fix(parity): the goldens refresh runs quantized goldens, and reports the split |
| `2e91607` | test: refresh parity deps_hash — non-numeric core-file drift (un-reds main) |
| `4c26a58` | perf(cuda): parallelise the Gemma final-logit softcap, bit-identical (P3) |
| `588052b` | serve: drain in-flight requests before freeing an unloaded model (fixes the leak safely) |
| `6091e7a` | fix(cuda): size the expert cache by SEARCH over the granularity form (A5) |
| `6edd1ca` | parity: make "validated" MEAN T3 — method-tier gate + honest experimental tier (D2, pre-freeze) |
| `7c91ccc` | cuda+docs: decline floor, slot-cap gate, driver allocation facts, and seven rules |
| `7cc2f0d` | fix(parity,ci): refresh deps_hash after 38061b1's pread-staging core plumbing (non-numeric) |
| `7ccec1e` | fix(cuda): the expert cache sizes itself — topK was the worst possible default |
| `82b39cc` | docs(parity): document qwen3_5_moe's int8-vs-bf16 movement (v0.8.0 §1 — gate-backed pass) |
| `8fecfad` | ci: heavy_gate.sh — a runner for the real-checkpoint tier that no CI job executes |
| `91f359f` | fix(decoder): matmulInto dispatches on the property, not on W8A8 (P7) |
| `93eb7d4` | feat(decoder): gpt-oss real-model path — batched-prefill fix + real gates |
| `9624dd9` | chore(parity): refresh deps_hash for aikit v1.12.0 (goldens-proven non-numeric) |
| `98936cf` | test(goldens): strengthen mamba-2 + deltanet parity fixtures (kill identity weights) |
| `99b3f95` | chore(deps): pin aikit v1.12.0 — gpt-oss MXFP4 reproducible on main |
| `9e5f8fa` | fix(quant): reject --quant that conflicts with a prequant .giw at startup (T1-7) |
| `bd08936` | fix(gate): cannot-search is not not-found; cross-gate composition; B7 sweep |
| `be049df` | [aikit] gpu(gemv): explicit __fmaf_rn in the quantized GEMV — the bit-identity contraction rule |
| `c8b65ba` | feat(serve): --moe-cache-experts / --moe-cache-slots — PARKED on the freeze |
| `ca29d6c` | cuda: resident context cap becomes configuration-derived (-ctx), VRAM-checked at load |
| `cc238c6` | cleanup: consolidate GINFER_ env vars to GOINFER_ + add env-var registry |
| `e42e83e` | fix(cuda): name the kernel and both slot counts when a launch runs out of memory |
| `e58ac8a` | fix(parity): refresh deps_hash after f340d4e's guarded int4-scale seam — non-numeric, validated_at preserved |
| `ecc5af2` | chore(parity): refresh deps_hash after default-off diagnostic hooks (non-numeric) |
| `ed81e13` | P1: route top_k=1 to the on-device greedy fast path |
| `eea7f29` | perf(decoder): one gate/up pair per token in MoE, not one per expert (P6) |
| `f6bbf7c` | feat(serve): --moe-cache-experts / --moe-cache-slots (decisions 2+3) — HELD, trips the parity manifest |
| `f9d5d07` | feat(decoder): dispatch census (B6); close the GGUF-quant gap; reopen B4 |

<!-- /SHA-INDEX -->

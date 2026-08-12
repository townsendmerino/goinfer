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
**MEASURED; blocked on A10, which named the mechanism**

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

**A10 · The total fitting does not imply the allocations succeed** — `linux`, **ANSWERED: it is
servability, not capacity. Fix candidate identified, not implemented.**

At cap 34 with a 32 MiB margin, `allocSlots` fails mid-sequence. The per-allocation probe
(`GOINFER_A10_PROBE=1`, recording only) names it exactly:

    [a10] alloc #112:   67403776 B, free before  304283648
    [a10] alloc #113:    8425472 B, free before  235077632
    [a10] alloc #114:   33701888 B, free before  224591872
    [a10] alloc #115:    4212736 B, free before  188940288
    [a10] alloc #116:   67403776 B, free before  182648832   <- REFUSED
    [a10] alloc #116 FAILED: requested 67403776 B, free immediately before = 182648832 B
          (174.2 MiB), ratio free/request = 2.71 — cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY

**The model is not wrong — 182,648,832 B is the figure derived from the closed form BEFORE the run,
to the byte.** Free was **2.71×** the request and the driver refused. So the constraint is
**per-allocation servability**: the remaining 174.2 MiB exists only as fragments smaller than the
64.3 MiB wanted.

**This does not re-open the struck fragmentation item, and it does not contradict it.** That control
compared heaps at *low* occupancy and found a fresh heap had *worse* contiguity than the slot-loaded
one — a valid result at that pressure, and it says nothing about the tail of a sequence that has just
consumed 3.5 GB in 116 allocations. The refutation was scoped and this is outside its scope.

**Where it bites.** #116 is the **last MoE layer's `expGU.W`**, the largest buffer in the per-layer
group (34 × 1,982,464). The sequence is 30 repetitions of {64.3, 8.0, 32.1, 4.0} MiB, so the biggest
request in each group is issued into a heap the previous group has already carved up, and the last
one is issued into the most carved-up heap of all.

**Fix candidate: allocate largest-first across ALL layers, not group-by-group.** Issue all 30
`expGU.W` while the heap is least fragmented, then all `expDown.W`, then the two scale buffers. Total
bytes are unchanged, so `capSlots` needs no change and the granularity form still holds; only the
issue order moves. Pre-registered test: at cap 34 with a 32 MiB margin, the current order fails at
#116 and largest-first should complete — and if it does not, the ordering hypothesis is wrong and the
remedy is a slack term rather than a permutation.

*(The slab restructure remains struck. This is not that: it is a reordering of the same allocations,
not one big buffer sub-allocated, and it costs nothing if it works.)*

**Until it is fixed, the shipped configuration is safe by margin.** The 384 MiB margin keeps the cap
at 31, where the tail allocation has ~500 MiB behind it, and a failure declines cleanly to the
staged/CPU path rather than crashing. **A9-MARGIN stays blocked on this** — lowering the margin to
128 MiB is what moves the cap into the region where servability starts to bind.

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

**A9-SPEC · Specialize `MOE_MAX_E` at JIT time** — `linux`, real reclaim, measured not estimated

The per-thread arrays are `float[MOE_MAX_E] × 2` and `MOE_MAX_E` is compile-time — but goinfer JITs
through NVRTC, so it can be specialized to the loaded model's **actual expert count** rather than the
worst case. A 128-expert model currently pays the 512-expert reservation.

Attach the measured figures, not an estimate: `moe_route` declares **4416 B/thread**, retains
**138,412,032 B**, and demands **289,013,760 B** at first launch. What the reclaim actually is must be
**measured by building at a lower `MOE_MAX_E`** — see A9-MULT below for why it cannot be derived.
Note the PTX freeze: regeneration is reproducible only via the pinned `nvidia-cuda-nvrtc-cu12==12.6.85`
venv, so this is a deliberate exercise rather than an incidental rebuild.

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

**B0 · Repo-hygiene group must run what CI runs** — `linux`

CI went red on `staticcheck -tags cuda` (ST1005) and stayed red for three commits. The local
sequence — gofmt, go build, go test — and CI's check set are **different, and nothing declares the
relationship**, so they drift on the next change to either. Adding staticcheck to one person's
habits fixes the instance, not the class.

`gpu_gate.sh` group 5 has the identical gap: it runs `gofmt` and `go vet`, not `staticcheck`. The
gate is supposed to be what you run *instead of* remembering, and on this axis it is a subset of CI
without saying so — so running the gate would not have caught this either.

Fix: the repo-hygiene group runs what CI runs, **derived rather than duplicated** if practical, so
the next check CI gains does not reopen the gap.

**B0a · A guard that cannot find its tool must fail, not skip** — `linux`, the guard shape

B0 is the CI side; this is the shape one level down, and it happened *inside* the session written to
close B0. A check ran as `command -v staticcheck >/dev/null && staticcheck ... | head`. The binary
exists but is not on `PATH`, so `command -v` failed, the `&&` short-circuited, the whole check
evaluated to nothing, and the surrounding output looked exactly like a clean run. It was reported as
"clean". Re-running it directly, with CI's own invocation, was clean — so the conclusion survived and
**the guard did not**.

This is an absence-of-signal instance in its guard form: a missing tool is indistinguishable from a
passing check. Remedy, either: **the guard fails when its tool is absent**, or **the skip is recorded
in the census** so a not-run check is visibly not-run. A silent third state is what makes it dangerous
— it was found last and mentioned last, which is where things sink.

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

**B4 · Label or drop `stash@{0}`** — `linux`

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

**D3 · The parked flag-pair** — `linux`, blocked on the freeze

`flag-pair-moe-cache` (`f6bbf7c`) carries `--moe-cache-experts` and `--moe-cache-slots` as CLI
flags. The `Options` fields and accessors touch `decoder/model.go` and `gguf.go`, which re-stales 19
families' `deps_hash`. `BRANCH-NOTE.md` records the pickup steps and the instruction that matters:
**run the goldens, do not refresh `deps_hash` to quiet the gate**.

Precedent exists for a goldens-gated refresh (`9e5f8fa`, where a metadata field addition re-staled
`weights.go` and the refresh ran 19 goldens). It was deliberately not spent on ergonomics.

### E. Release and claims

**E1 · v1.0 gate as written criteria** — `linux`

Decisions already taken: the parity backfill lands as **v0.12.0**, not v1.0. v1.0 gets its own gate
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

**E3 · Freeze re-declaration** — `linux`

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

**Deliberately not cut.** `be049df`'s FMA fix is already released (contained in `gpu/v0.25.0`
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
- **`go fix` modernizers** — one deterministic pass across ~20 adapters written over months,
  reviewed as a diff. **After the freeze**; it re-stales the manifest wholesale.

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

Not frozen, so the work is unblocked — but shipping it **reverses E6** (aikit release deferred to
v1.0), which must be an explicit decision rather than one arrived at by landing the patch. **Not
landed.**

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

**P4 · Metal RoPE dispatched twice per layer** — `metal/model.go:1301`

One grid-merged dispatch is bit-identical. Estimated a few %.

**P5 · Metal `quant_vec` fused into the o-proj GEMV** — estimated ~5–6% of dispatches per token.
The swiglu half is **not** a clean fusion; do not bundle them.

**P6 · `moeMLP` allocates ~7–8 MB/token** — `decoder/mlp.go:82`

By skipping the `decodeScratch` invariant its dense sibling honours. Frozen core. **See B6** — this
is a sibling-drift instance, and fixing it without enumerating the pair leaves the class open.

**P7 · W4A8 allocates a fresh `Workspace` per projection per token** — `decoder/weightmat.go:202`

The W8A8 sibling was fixed; this one was not. Frozen core. **See B6.**

**P8 · `sampleChunked` allocates a full-vocab `[]float64` and rebuilds the goroutine pool twice per
sampled token** — `decoder/sampler_chunked.go:188`. A jitter reducer rather than a throughput win.

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

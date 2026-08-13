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

**A12 · The CUDA heavy tier does not fit in 8 GB in one process — the gate cannot pass here** —
`linux`, **open, found 2026-08-12**

`scripts/gpu_gate.sh` is red, and after A11 and the two leak fixes the remaining failures are **all
VRAM exhaustion presenting as something else**:

| test | in-suite | alone |
|---|---|---|
| `TestAttention_HeadDimWidths` | FAIL — `attention cosine 0.000000 vs CPU ref` | **PASS**, cosine ≈ 1.0 |
| `TestSplitKV_bitIdentical_gemma3` | FAIL | **PASS** — bit-identical at depth 1536 |
| `TestMoERouteFirstLaunchReservation` | FAIL (0.13 s) | **PASS** |
| `TestResidentLaunchVRAMProbe` | FAIL (248 s) | VRAM-sensitive by construction |

**`cosine 0.000000` is the signature the script's own header names**: *"parallel packages contend for
VRAM and the failures come back as bogus numerics ('cosine 0.000000') rather than 'you are out of
memory'."* The other symptom is `panic: interface conversion: decoder.ResidentForward is nil` — a
`BuildResident` that could not allocate. Neither is a numerics defect.

**It is CUMULATIVE, not a single leaker.** Two genuine leaks were found and fixed (`5ece205`:
`TestAllocFloor`, `TestA10ReportingGap` — both drained the device and never released). Fixing them
stopped the tier aborting at 91 s and let it run its full ~26 minutes, which is what exposed these.
Beyond that: `TestMoERouteDemandThreshold` + `TestMoERouteFirstLaunchReservation` pass as a pair; the
pressure builds across the whole tier rather than at one site.

**And there is no systematic missing-`Close`.** Audited: 75 `Load` calls against 175 `defer …Close()`
across the CUDA tests; only three small files load without any (`cuda/gemma_bos_build_test.go`,
`cuda/graphs_safe_test.go`, `cuda/mustalloc_test.go`). Worth tidying, but not the cause.

**So this reads as a capacity limit of the 8 GB card for the full heavy tier in one process, not a
defect in the tree.** Every test passes on its own; the tier as a whole does not fit.

**Options, none taken yet — this is a decision, not a fix:**

1. **Run each heavy test in its own process.** Bounds VRAM by construction, since exit reclaims
   everything. Costs process-start time per test and needs the runner reworked. This is the one that
   actually removes the class.
2. **Tidy the three no-`Close` files and re-measure.** Cheap, and it may or may not be enough — say
   which before running it, or it becomes a fishing expedition.
3. **Declare the heavy tier out of scope for an 8 GB box** and record the gate as environment-limited,
   with the per-test isolation runs as the evidence. Honest, but it weakens the gate.

**The tag is blocked on this.** Not because numerics are wrong — they are demonstrably right test by
test — but because `scripts/gpu_gate.sh` says *do not tag*, and a gate that is overridden by the person it
was written to constrain is not a gate. B8 applies here too: the gate cannot presently distinguish
"failed" from "could not evaluate", and these four are the second kind.

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
`scripts/ci_checks.py` exits 2 with a message and group 5 **fails**; the script missing entirely → non-zero
and empty output, group 5 **fails**. Neither degrades to the old hand-written list, which would have
looked like a pass.

**Residual risk, named because it is the one that actually bit:** the instance was an **ad-hoc shell
command typed at a prompt**, not repo code. No gate polices that. The mitigation is the habit the
gate exists to replace — run `scripts/gpu_gate.sh` rather than hand-rolling the check — which is exactly what
B0 makes worth doing.

**B5 · `RELEASING.md` must reference `QUEUE.md`** — **DONE (2026-08-12).**

A file nothing reads is accurate today and inert the first week nobody opens it — the pattern this
queue was written to fix, applied to itself. A tag is the natural moment to review what is
outstanding, and it is a checkpoint that already happens. **Landed:** `RELEASING.md` now has a
"Queue-gated follow-ups — consult QUEUE.md at each tag" section (after the GitHub Release step),
whose **first concrete customer is C3** (Metal consumer window fires on a release carrying an aikit
bump). The abstract "reference QUEUE.md" and its first real trigger landed together, so the reference
is not itself an inert line.

**B2 · Gate reconciliation — one entry point** — `linux`

Two mechanisms now exist for running the heavy tier: `scripts/gpu_gate.sh` group 2c (linux) and
`scripts/heavy_gate.sh` (`8fecfad`, mac).

Resolve to one: **`scripts/gpu_gate.sh` always declares the heavy group.** When not requested it emits a
counted skip with its reason and the verdict line carries it. Fast runs stay fast; no run silently
omits the tier. `scripts/heavy_gate.sh` becomes the implementation group 2c invokes, or it goes. Two files
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

- `cuda/backend.go:836` — cited as `allocSlots`'s call site; now points at a bare `//` (A9-FIX
  inserted the warm-up above it).
- `cuda/resident.go:244` — cited for audit C-08's `_ = gpu.Upload`; now a comment about backend locals.
- two citations were **unresolvable**, because they omitted the repo — an aikit `linalg/quant.go`
  line and a bare `decoder/weightmat.go` one. Both
  are aikit paths written as if they were local ones; the SHA lint learned this distinction for
  commits and the same gap exists for paths.

**FIXED in the same change as the lint that found them** (`scripts/queue_citation_lint.py`), because
a lint landing red on its first run is a lint nobody adopts. The stale `allocSlots` call-site
line was corrected (it had drifted when A9-FIX inserted the warm-up above it);
the two bare `decoder/weightmat.go` / `decoder/mlp.go` references repo-qualified or de-numbered; `linalg/quant.go:113` resolves in
aikit once the lint searches the sibling set.

**And one turned out not to be a line drift at all.** `cuda/resident.go:244` was cited for audit
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

**DEFERRED BY CHOICE (2026-08-12) — auto-pickup, trigger pinned. Not sunk: deferral is the decision.**

- **Trigger = the next goinfer RELEASE TAG THAT CARRIES AN AIKIT BUMP.** No version floor: the
  condition is the *bump*, and a version number standing in for it is a literal that drifts. **Same
  shape corrected twice in one day** — "the aikit **v1.17.0** bump" was loosened to *any* aikit bump
  this morning when v1.17.1 shipped hours later, and "**≥ v0.13.0**" is the same substitution one
  level up (it happens to fire correctly for v0.13.0, so this is hygiene, not a fix). B7's constant
  class. `aikit` and/or
  `aikit/gpu` increased against the previous release tag, whatever the version. **Was written as "the
  aikit v1.17.0 bump" and that literal drifted within hours**: aikit shipped v1.17.1 the same day, so
  the trigger named a version `main` was no longer on. RELEASING.md's copy was already generic, which
  is the only reason the trigger still fires — the two carriers disagreed and only the durable one was
  right. This is B7's constant shape (a literal restating a value maintained in five `go.mod` files),
  and it is now stated the same way in both places. It is *not*
  the bump commit on `main`. C3 is a public-view consumer evaluation: an external consumer `go get`s a
  released tag, so evaluating main-HEAD-with-the-bump would certify a state **nobody installs**. This
  is the (b) reading, chosen deliberately over the faster (a): "post-bump tag" here means a **release**
  tag, not the bump commit. (v0.12.0 shipped 2026-08-12, so v0.13.0 may be days out — hence the bound.)
- **Bound = 2026-08-26 (14 days).** If no qualifying release tag by then, run C3 anyway against the
  **latest published goinfer tag** and **record the exact dependency set it evaluated** (resolved
  `aikit` + `aikit/gpu` versions), flagged as the bounded fallback. A consumer window against a
  slightly stale set beats one that never runs — an auto-pickup with no bound is indistinguishable
  from forgetting. **The bound is a date with no in-repo reader, so it is carried EXTERNALLY:** Francis
  is arranging a persistent 2026-08-26 reminder from the Cowork side. Do **not** assume the cron or a
  session covers the bound — the cron expires at 7 days, well before it.
- **Why deferred, not dropped:** the attached claims (73.6 tok/s, cgo-free/no-Xcode, 0.96×/0.74× vs
  Ollama-Metal, bit-identity) are version-sensitive and all originate in `aikit/gpu`; running mid-bump
  documents a set superseded within hours and forces a re-run. This surface **sank once already** and
  was first in its batch precisely to prevent that.
- **Carriers, in order of durability:** (1) **`RELEASING.md`** carries the trigger as a release-process
  line — *"if this release carries an aikit bump, C3 runs on macbook-arm64 against this tag; see
  QUEUE.md C3"* — read by whoever cuts the tag, **at the moment it fires**, surviving every session
  ending. This is the actual carrier (and B5's first concrete customer, landed with it). (2) a
  **session-scoped** cron (daily, 7-day cap) runs C3 the moment a qualifying tag appears *while this
  session lives* — a bonus accelerator, not the guarantee. (3) **this entry** is the record. The
  fragile half — needing a session to outlive the wait — is retired: the release process carries it.

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

**D3 · Promote the expert-cache env vars to CLI flags** — `linux`, **re-derived `2d28358`; ready to
rebase**

**This entry's own description was wrong at the source, not stale.** It read as a "parked flag-pair"
with a workaround premise. `BRANCH-NOTE.md` says what it is: **an API-surface promotion** of
env-var-only controls to CLI flags, wired `decoder.Options` → `Model` accessor → CUDA backend,
following the `KVPrecision` pattern rather than adding more `os.Getenv` to the backend. The entry was
mine and it mischaracterised the branch from the beginning — a status sweep would never have caught
it, because the status was right.

**Does it complete the MoE-cache story the release headlines? YES.** `c8b65ba` adds
`--moe-cache-experts` / `--moe-cache-slots` to `serve`, and the README instructs the **env vars in
three places** — including the very section this release rewrites around the cap fix. Shipping the
release without D3 means rewriting that section again next version, for the same feature.

**But its branch predates the fix it completes.** `flag-pair-moe-cache` is based on `7ccec1e` — the
slot-default commit that was **reverted** — and it touches `cuda/backend.go` and `decoder/model.go`,
both of which A5 (`6091e7a`) and A9-FIX (`0103b49`) changed substantially. So this is a **rebase and
re-verify**, not a merge.

**DESIGN READ DONE 2026-08-12 — the reason SURVIVES, but the entry was resting on a stale premise.**

Read from `BRANCH-NOTE.md`, not the diff. The stated intent is *"the env-var-only expert-cache
controls promoted to real CLI flags, wired `decoder.Options` → `Model` accessor → CUDA backend,
following the `KVPrecision` pattern rather than adding more `os.Getenv` to the backend"*. That is an
**API-surface change**, not a workaround.

**And the branch note itself draws the line the question asks about**: *"the user-visible half of this
work — the slot default, which was costing ~3× decode rate — was CUDA-only and landed on `main` at
`7ccec1e`"*. The workaround-shaped part was always a **separate change**. It landed, was reverted, and
was redone correctly as A5. **A5 changed what the default computes; it did not remove a user's reason
to override it.** So the flag pair is the second branch: legitimate explicit override, and it stays.

**What needs re-deriving, per that branch:**

1. **The flag's documented meaning.** Written when the cap could not be trusted, so it read as *how
   you get a working cache*. With A5 it means *request no more than N*, and the cap may still lower
   it — the log line says which.
2. **`BRANCH-NOTE.md`'s own rebase guidance is stale.** It says to expect one conflict in
   `cuda/backend.go` and to *"keep `main`'s comment"* — but that hunk has changed three times since
   (`7ccec1e` reverted at `97ee663`, then A5 `6091e7a`, then A9-FIX `0103b49`). The instruction now
   points at text that no longer exists.
3. **Its freeze paragraph is superseded** — it waits on a lift; the freeze is now a proof requirement
   and the goldens run is the authorisation.

`decoder/model.go` and `decoder/gguf.go` were untouched by A5/A9-FIX/P6/P7, so only `cuda/backend.go`
should conflict.

**REBASED 2026-08-12 (`2d28358`), not yet merged.** One conflict, in `cuda/backend.go`, and it was
not the plumbing conflict the old note predicted:

**the branch defaults the slot request to `nE` — "ask for all, auto-cap" — which is `7ccec1e`'s
reverted behaviour.** Resolved by keeping the accessor (the flag promotion) and **preserving main's
`topK` default**: unset changes nothing. Raising the default is a **separate decision** and main's own
comment states its precondition — *"fixing the margin FIRST and proving it on the 26B"* — which A5
(`6091e7a`) and A7 have now met. It belongs in a change about defaults, not one promoting env vars to
flags. **Queued below as D3b.**

Flag help rewritten for the corrected cap: `--moe-cache-slots` now documents itself as *request at
most this many*, notes that the runtime lowers it and logs what it chose, and no longer describes a
"deliberately greedy" default that is not taken.

**The goldens run is still owed, on a fixture-bearing checkout** — see the finding below. Merging is
the next action.

**D3b · Should the expert-cache default rise above `topK`?** — `linux`, **unblocked, now a real question**

`topK` degenerates to re-fetching every routed expert every token (~5 tok/s against ~17 at 38 slots on
the 26B). It was set there because the cap could not be trusted; A5 fixed the cap and A7 proved it on
the 26B, which is exactly the precondition `cuda/backend.go`'s own comment names. The candidate
default is "request all, let the corrected cap decide" — which on this card lands at 31.

Separate from D3 on purpose: D3 changes the *surface*, this changes *what happens to someone who sets
nothing*. Needs the 26B run and a hit-rate figure at the chosen cap before it lands.

**PRECONDITION READ, 2026-08-12 — and it is NOT met. D3b waits on A10.**

`cuda/backend.go`'s comment says *"raising it again requires fixing the margin FIRST and proving it on
the 26B"*. Two readings are available and the comment's own complaint chooses between them: it faults
the margin for being *"a flat `marginBytes = 384 MB` **described as** covering the greedy-argmax
readback + driver overhead — per-token costs — **while what it must actually leave room for** is
everything the forward allocates AFTER it runs"*.

That is an objection to the margin **not being derived from what it must cover**. So "fixing the
margin" means **deriving** it, not checking it — and the second reading is the intended one:

> **margin derived rather than asserted → A10 blocks D3b, and it waits on the floor being attributed.**

**What we have instead**, and it is genuinely better than when the comment was written, just not the
thing it asks for:

- `slotMarginBytes` is still **402,653,184 by assertion** — the same flat constant.
- **A9-FIX removed the largest unaccounted consumer from the margin's job**, by paying the deferred
  first-launch reservation *before* sizing rather than expecting the margin to absorb it. The
  concrete failure the comment names — capped to 34, then `CUDA_ERROR_OUT_OF_MEMORY` — cannot recur
  that way.
- A gate asserts `slotMarginBytes ≥ measured peak demand` (402,653,184 ≥ 289,013,760, clear by
  113,639,424). **That is a check, not a derivation**: it confirms the constant is big enough today
  on this card, and would confirm it just as happily if the constant had been picked by coin flip.
- **A10's 151,191,552 B floor is unexplained and sits inside that margin** — 37% of it. A derivation
  cannot be written while more than a third of what the margin covers is unattributed.

**THE BLOCKER NOW HAS A ROUTE — read this before treating D3b as indefinitely stalled.** A candidate
derivation exists:

> **margin ≥ reporting gap + peak transient** = 151,191,552 + 137,822,208 = **289,013,760**

against the shipped 402,653,184, which clears it by 113,639,424.

**BASIS — read this rather than the number alone.** The derivation rests on **single-context scoping**,
not on a per-device result. The reserve was measured as **per-context** (a second context costs
107,806,720 B), so the margin is not a device constant in general. It is a constant *for goinfer*
because goinfer creates exactly one context and **cannot create a second** — `cuCtxCreate` is not
bound by gocudrv, and aikit uses only the refcounted primary context.

**Stated precondition: revisit if goinfer ever creates a second CUDA context**, or if a dependency
gains `cuCtxCreate` and something uses it. Until then D3b's blocker is **resolved on that basis**, and
and A10 is now **fully decomposed**, so there is no longer a residue to worry about: the gap is
44,236,800 B once per device plus 106,954,752 B per context, summing exactly. The derivation used the
*measured gap* and did not depend on the decomposition — it is simply better founded for having it.

**So: D3b is unblocked as a question and blocked as a change.** The precondition's second half
("proving it on the 26B") is met — A7 did that. The first half is not. Recorded here so the next
person does not re-decide it; **reopen when A10 is attributed**, not before.

**Historical framing: out of the release, and the first question was not a merge question.**

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

`flag-pair-moe-cache` (`bacc04c`) carries `--moe-cache-experts` and `--moe-cache-slots` as CLI
flags. The `Options` fields and accessors touch `decoder/model.go` and `decoder/gguf.go`, which re-stales 19
families' `deps_hash`. `BRANCH-NOTE.md` records the pickup steps and the instruction that matters:
**run the goldens, do not refresh `deps_hash` to quiet the gate**.

Precedent exists for a goldens-gated refresh on exactly this shape: **`ca29d6c`**, where making the
resident context cap configuration-derived added `Options` plumbing to **`decoder/model.go` and
`decoder/gguf.go` — the same two files this branch touches** — and refreshed behind 19 goldens.
(This line previously cited `9e5f8fa`, which touches the manifest not at all; the "re-staled
`decoder/weights.go`" detail was fabricated with it — none of the nine real refreshes touches that
file.) It was deliberately not spent on ergonomics.

### E. Release and claims

**E1 · v1.0 gate as written criteria** — `linux`

**The parity backfill lands as `v0.14.0`** — moved from `v0.13.0`, **decided 2026-08-12 by
Francis**, which is the **second** move of this reservation and the history is kept deliberately:

| target | moved because |
|---|---|
| `v0.12.0` | that number was taken by the CUDA expert-cache campaign |
| `v0.13.0` | *(superseded, 2026-08-12)* |
| **`v0.14.0`** | **current** — v0.13.0 is being cut for the aikit bump + D3's flag promotion |

**The reason, recorded because it is the useful part.** `v0.13.0` is the honest number for what it
carries: **D3's `--moe-cache-experts` / `--moe-cache-slots` promotion is new user-visible CLI
surface**, and a minor is what that content warrants. The backfill reservation had been attached to
a **number** rather than to a **plan** — and reservations attach to plans. Moving the reservation is
a **smaller correction than numbering a release to satisfy a bookkeeping artifact**, which is what
holding v0.13.0 for the backfill would have been.

*(Same shape as B7's constant class, one level up: a plan pinned to a literal drifts when the
literal is needed for something else. The remedy there was to derive the value; here it is to attach
the reservation to the work rather than to the number.)*

**E2's obligation is unchanged in substance — only its target release moves.** The four families
still carry `validated_at: null`; see E2.

v1.0 gets its own gate
requiring parity coverage complete, the verification machinery sound, the loader and
architecture-descriptor surface **actually frozen** (the docs still say it may change), and a clean
out-of-tree audit against the release candidate.

Write that as a checklist so 1.0 is a decision against criteria rather than a feeling.

**The v1.0 gate checklist (draft — the point of E1):**
- [ ] Parity coverage complete (E2's four `validated_at: null` families resolved: T3 or demoted to experimental).
- [ ] Verification machinery sound (the gates run and can fail; skip census clean at the freeze).
- [ ] Loader + architecture-descriptor surface **actually frozen** (docs stop saying it may change).
- [ ] Clean out-of-tree audit against the release candidate (C-group consumer window).
- [ ] **The repo contains no Python** — all analysis in Go tests; shell minimized to process
  orchestration. **Decided 2026-08-12 by Francis.** Inventory, ranking, acceptance criteria and the
  reference-tensor carve-out are in **E7**. (The reference-tensor / `pin_*` generation is *excluded*
  from this line — blocked on Francis's torch-replacement research; see E7 item 7.)

**E2 · The four per-family demotion judgments** — `linux`

`gpt2`, `granitemoehybrid`, `kimi_k2`, `nemotron_h` carry `validated_at: null` and are the same four
the `deps_hash` tripwire does not enforce — so 19/23 tracks both the backfill's progress and the
tripwire's coverage, and clearing it closes both.

**Retargeted to `v0.14.0`** (2026-08-12, with E1's reservation — substance unchanged, target only).

Rule: every family claimed as supported at v1.0 has a current T3 row; families that can't get one go
experimental. **Honesty test per family — would you move it to experimental if no release were
pending?** Structural reasons qualify (no reference, fixture size, licence). "Unfinished" does not;
demoting unfinished work to clear a release hollows out the tier permanently.

**E3 · Freeze re-declaration** — `linux`, **inventory taken and the condition drafted; see below**

**THE FREEZE-BLOCKED INVENTORY, read rather than grepped** (21 frozen paths, from
`testdata/parity_manifest.json`'s `shared_sets`):

| column | item | what blocks it |
|---|---|---|
| **freeze-only** | **D3** the parked flag-pair | `Options` fields touch `decoder/model.go` + `decoder/gguf.go`; re-stales 19 families |
| **freeze-only** | **G2** `go fix` modernizers | re-stales the manifest wholesale |
| freeze **plus other** | **P1** KV re-gather / V re-transpose | freeze **+** a new aikit row-pitch API **+** E6's deferred aikit release |

Everything else touching a frozen path has **landed** (P6 `eea7f29`, P7 `91f359f`) or only references
those paths as instances (B6, P8 — `decoder/sampler_chunked.go` is not in the manifest).

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
CPU-only? Note that until B2/`scripts/gpu_gate.sh` ran the parity gates, GPU forward numerics had no
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

**E6 · aikit release** — `linux` or `mac` — **SUPERSEDED BY EVENTS 2026-08-12, and the deferral was
right on its own terms**

aikit cut `v1.17.0` / `gpu/v0.28.0` (`ada417e`), and goinfer is on it (`f33fcaf`). E6 is closed by the
release happening, not by the argument below being withdrawn — and the release **satisfied E6's own
criterion** rather than overriding it. The reason a consumer can receive is `linalg.MatmulBTW8A8Pre`
gaining 8-column blocking: goinfer never calls it directly, but `MatmulBTW8A8Into` now delegates to
it, so every W8A8 decode matmul goes through changed code. That is a consumer-visible change. The
gates and CI that E6 declined to release *for* rode along as line items, which is exactly the shape
E6 predicted for them.

**The FMA fix was never the pending part, and the bump re-confirmed it.** `be049df` is an ancestor of
`gpu/v0.27.0`, so goinfer had already been running that PTX; across `gpu/v0.27.0..gpu/v0.28.0` the
quantized GEMV PTX is **byte-identical** and `gpu/gemv_quant.cu` changes by three lines, all comment.
E6's read of its own diff was accurate.

**A measurement trap found while checking this, worth more than the closure.** `git diff
v1.16.0..v1.17.0 -- gpu/` reports the quantized GEMV PTX changed by 72 lines. It is a **misleading
comparison**, and it looks authoritative. `gpu/` is a nested module with its **own tag series**, and
the two series do not track: `v1.17.0` and `gpu/v0.28.0` are the same commit, but `v1.16.0`
(`a79303e`) and `gpu/v0.27.0` (`4642b7c`) are not. Diffing a nested module across the *parent's* tags
therefore spans commits the consumer already had — here it re-reports `be049df`, weeks-old and
already shipped, as if the pending bump introduced it. **Diff a nested module across ITS OWN tags.**
This is the Environment class from the measurement-shape list (docs/parity-coverage-policy.md) with a
new instance: not where it ran, but which boundary it was measured across. It was caught only because
72 changed lines contradicted a claim already written down — an expectation, not an instrument. There
is no gate for this; the countermeasure is the rule in bold.

The original reasoning, kept because the rule in it outlives the decision:

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
- **Keep (orchestration):** `scripts/parity_sweep.sh`, `scripts/gpu_gate.sh`, `scripts/heavy_gate.sh`, `cuda/build_ptx.sh`, and the two demo asset-build scripts under `demo/chat/` and `demo/agent/`. **Move (reads tree + decides):** the deciding half of `scripts/refresh_parity_hashes.sh` and `scripts/mutation_check.sh` are candidates once the Go tooling exists; `scripts/bench_compare.sh` folds into the bench-peer Go successor.
- **Rule audit (the four rules):**
  - **Rule 3 (`set -euo pipefail`):** `scripts/refresh_parity_hashes.sh` and the three build scripts compliant. **Violations:** `scripts/bench_compare.sh` (`set -u` only) and `scripts/mutation_check.sh` (`set -u` only) — add `pipefail`. The gate scripts (`scripts/gpu_gate.sh`, `scripts/heavy_gate.sh` are `set -u`; `scripts/parity_sweep.sh` is `set -uo pipefail`) **omit `-e` DELIBERATELY** — they run N families/packages and tally, and `-e` would abort on the first failure and lose the tally. The real requirement there is per-command rc / `PIPESTATUS` capture, which they already do; **do not blanket-add `-e` to a tallying gate.** Document the reason inline so it is not "fixed" into a regression.
  - **Rules 1/2/4:** no violations found in a targeted pass. The scripts show awareness — `scripts/mutation_check.sh`'s header explicitly records fixing the `command -v staticcheck && staticcheck` (rule 1+4) anti-pattern; `scripts/gpu_gate.sh`'s `command -v nvidia-smi` is backend *detection*, not a skipped check; the `grep -c … || true` counts guard the pipe's exit correctly. A full line-by-line pass on the keep-set is deferred with the migration.

**Item 7 — OUT OF SCOPE of E7's migration, owner named:** the pin-fixture and reference-tensor generation (the 57 torch/HF scripts). **Francis owns it.** Do not start it in parallel and do not design the tooling migration around a guess at what he finds.

**Replacement research: DELIVERED as a scoping plan — `docs/task-oracle-refforward.md`.** The design questions are resolved there (verdict: buildable; a pure-Go `oracle/` submodule with its own `go.mod`, independent safetensors reader + f64 math, anchored against HF once per architecture, emitting the existing golden schema — Python shrinks from ~50 per-model generators to a handful of per-architecture anchor runs, not to zero). It is a **plan, not a start**: still Francis's go/no-go to fund, and gated behind v0.13.0 (§C1 + CUDA gate) like the rest of E7. Its Phase 0 (cluster the 57 by shared forward-math to size the real kernel surface) is the first work if funded — the E7 inventory counted the scripts but did not cluster them by math.

**E8 · One Go gate-runner over `go test -json` — collapse the tallying shell + census Python** — `linux`/`mac`, **PLAN DRAFTED; not started; after v0.13.0 (§C1 + CUDA gate), same freeze as E7.** Decided 2026-08-12 by Francis.

Distinct from E7 (which migrates scripts one-for-one): **E8 recognizes that six scripts are one program.** The three tallying shell gates (`scripts/parity_sweep.sh`, `scripts/gpu_gate.sh`, `scripts/heavy_gate.sh`) and the three census Python scripts (`scripts/skip_census.py`, `scripts/sweep_composition.py`, `scripts/selector_coverage.py`) all run `go test -json` across a *package × family × quant × tag* matrix, tally PASS/SKIP/FAIL (SKIPs bucketed by reason), and decide — differing only in matrix and decision. **One runner (`cmd/gate`) + committed configs subsumes all six** (~6 scripts → 1 runner + configs).

Why Go is *strictly better* here, not just same-language — it dissolves the item-6 audit's footguns by construction: the deliberate-omit-`-e` tally tension vanishes (Go has no abort-on-error — capture `rc`, append, never lose the count); `PIPESTATUS` capture is direct via `os/exec`; the `command -v tool && tool` silent-skip (the one `scripts/mutation_check.sh` records fixing) can't recur (`exec.LookPath` miss is an explicit error). Backend/asset detection → SKIP-with-reason in Go, where it can't fail open.

**Stays shell (pure glue, per the decides-vs-orchestrates line):** `cuda/build_ptx.sh`, the two `demo/*` asset-build scripts, env-then-run-one-command wrappers. The runner **shells out to** `go test -json` (stdlib `os/exec`) — orchestrates it, doesn't reimplement it; **no new main-`go.mod` dependency.**

**Acceptance = E7's a–d verbatim** (agree-before-swap incl. the tally-integrity mutation case; delete the script in the same commit; the skip-reason scope line survives). **Sequencing note:** do **not** harden shell you're about to delete — the item-6 `pipefail` fixes for `scripts/bench_compare.sh` and `scripts/mutation_check.sh` are moot (both are on the migration list); add `pipefail` only to the surviving glue shells.

**Full plan: `docs/task-gate-runner.md`** (matrix-config shape, core loop, hardware detection, order-within-E8, not-in-scope). E8 and E7's census migrations converge — the three census scripts fold in as configs as each matching gate lands, rather than being migrated twice.

### F. Audit backlog

**F1 · §4 gates — SWEPT 2026-08-12: all five were already FIXED** — `linux`, **CLOSED**

Every entry here was stale, the same shape as C-08 and five times over. Swept against the tree with a
content-keyed citation added to each, so the next sweep is the lint rather than a person:

| gate | state | anchor |
|---|---|---|
| G-01 `TestResidentAdmission_matrix` tautological | **fixed** — compares against a reviewed golden and errors on any family missing a row | `decoder/features_test.go:146` |
| G-02 Metal snapshot golden applies no embed scale | **fixed** — `Forward`/`ForwardArgmax` apply the arch scale, with a named regression gate | `metal/snapshot_golden_test.go:77` |
| G-03 `buildMatrix` env-pinning | already closed | — |
| G-04 `case "slots"` doesn't assign `residencyBufs` | **fixed** — the switch populates `pinned` and it is assigned after | `metal/model.go:728` |
| G-05 tokenizer/chat tests probe a developer home | **fixed** — `GOINFER_MODELS_DIR`, defaulting to `$HOME/models` | `decoder/modelsdir_test.go:13` |
| G-06 hardcoded developer-home paths | **substantially fixed** — same mechanism; residue is a literal `/home/francis/models` reached only when `os.UserHomeDir()` *fails*, in the four per-package `modelsdir` test helpers | `decoder/modelsdir_test.go:13` |

G-06's residue is the only thing left and it is a last-resort fallback, not a probe path. Recorded
rather than struck, because "substantially fixed" is a different state from "fixed".

**F2 · §2/§3 criticals — SWEPT 2026-08-12** — `linux`, **most were already fixed; two lack an anchor**

| finding | state | anchor |
|---|---|---|
| C-05 gemma-4 stride on snapshot restore | **fixed**, with a gate | `decoder/kvsnapshot_gemma4_test.go:10` |
| C-06 unvalidated tensor shapes | **fixed**, break-it-first gate | `decoder/serialize_shapecheck_test.go:15` |
| C-08 `_ = gpu.Upload` over zeroed weights | **fixed** — `recordUpload` → `setupErr` → graceful decline | `cuda/resident.go:397` |
| C-14 CUDA argmax has no index tie-break | **fixed** at `c6600fc`, gated | `cuda/argmax_tiebreak_test.go:19` |
| C-31 `make([]byte, u32)` unbounded | **fixed** — bounded against the remaining file size before the allocation | `internal/giw/bundle.go:114` |
| C-21 embeddings batch cap, un-queued | **fixed** — `checkEmbedInputBounds` caps the input count, gated at the boundary and at +1; the un-queued half is a *documented deliberate decision*, not an omission. The body-cap tests are a different concern (bytes, not count) — covered-by-something-else, which is why they did not answer this | `internal/serveapp/embeddings.go:26` |
| C-22 shutdown lock, swallowed second signal | **fixed**, with a named gate — the checkpoint cannot block forever on a busy model, and a second Ctrl-C always kills | `internal/serveapp/main.go:432` |
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
  | `cuda` | 2 | 0 | 2 — `cuda/softcap.go`, worker count and chunk bounds |
  | `gpu` | 0 | — | — |
  | `metal` | 0 | — | — |
  | aikit | 0 | — | — |

  **No float `min`/`max` anywhere, and none of the 85 contraction sites is touched.** The headroom
  measurement survives and G2 needs no scope narrowing.

  **`slicessort`'s NaN axis, answered rather than left unasked.** Tie-order was the first question and
  it is not the only one: `slices.Sort` uses `cmp.Less`, which *defines* NaN placement, where a bare
  `<` does not — the same shape as `minmax`, one analyzer over, and the tie-order answer does not
  cover it. Its single site sorts `[]ResidentFeature`, and `ResidentFeature` is a **`string`** type.
  **Strings cannot carry NaN, so the question is moot** — recorded so it is answered.

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

**STILL OPEN after aikit `v1.17.0` — checked, not assumed (2026-08-12).** The v1.17.0 bump did land
8-column blocking in `MatmulBTW8A8Pre`, which is close enough to this entry to look like it. It is
not: `q8Span` is **unchanged** between `v1.16.0` and `v1.17.0`, still widening with
`deq[k] = float32(bq[k])` and still applying the scale after the dot. The blocking landed in a
different function, so neither the ~1.7 ns/element cost nor the scale-placement hazard below has
moved. Nothing here is superseded; the entry stands as written.

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
here as "a metadata field addition re-staled `decoder/weights.go` and the refresh ran 19 goldens". It is
`fix(quant): reject --quant that conflicts with a prequant .giw at startup` and **touches the manifest
not at all** — its five files are `CHANGELOG.md`, `decoder/giwquant_test.go`, `decoder/serialize.go` and two `an aikit benchmarks entrypoint`s.
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

**R2 · The refresh `arch=` stamp worked on its first real use — `mac`, CLOSED.** Added `2026-08-12`
(`a163150`) because last time the machine that ran a refresh was **unrecoverable from the record** —
the `2e8dfb6` arm64-vs-amd64 question could only be inferred. The `2026-08-13` arm64 gate run is the
first refresh since, and its record answers the question **directly**: proof block and trailer both
read `arch=arm64`. The record now says which arch ran the goldens instead of leaving it to inference.

**Q2 · The GGUF-quant cross-gate gap — CLOSED, and it was unplumbed too** — `linux`, `bd08936`→

The cross-gate check showed `scripts/parity_sweep.sh` covering the GGUF quant formats while the goldens
refresh did not. **(a) Exposure: a LAG, not a hole.** `scripts/parity_sweep.sh` is not in CI — it is
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

> **The "14 quantized" composition figure, resolved by enumeration rather than by authority
> (2026-08-12).** Two classifiers disagreed — an ad-hoc name grep said **7**, the refresh script's
> said **14** — and 14 had already propagated into commit bodies and into the proof requirement.
> Adopting it because it was the script's would have been a tiebreak by authority, so both were
> tested instead.
>
> **7 was structurally incapable of being right**, for two independent reasons. Five of the fourteen
> carry no quantization token in their NAME at all: `TestGemma4_logitParity` and
> `TestMellum2_logitParity` set it in the test body (`Options{Quant: "int8int8"}`), and
> `TestGGUF_gemma3/qwen2/qwen3_parity` set it in the **fixture filename** the test loads. No
> name-based match can see either. (The other two misses, `Q2_K` and `Q3_K_M`, were a plain gap in
> the ad-hoc pattern, which listed `q4|q5|q6|q8` — a bug rather than a structural limit, but it lands
> in the same place.)
>
> **The script's classifier cannot double-count.** `grep -c` counts matching LINES; every top-level
> result is one line; subtest lines are indented and excluded by its `^--- PASS:` anchor. Measured on
> the captured run: 33 top-level PASS lines, **0** indented ones, no duplicate names among the 14.
>
> And it does not misclassify — all fourteen drive a genuinely quantized path:
>
> | gate | quantization | set where |
> |---|---|---|
> | `TestGemma4_logitParity` | int8×int8 | test body |
> | `TestMellum2_logitParity` | int8×int8 | helper body |
> | `TestInt4_forwardParity` | int4 group-wise | test body |
> | `TestGGUF_Q2_K_parity` | Q2_K (+Q3_K/Q4_K/Q6_K mix-ins) | fixture |
> | `TestGGUF_Q3_K_M_parity` | Q3_K (+Q4_K/Q6_K) | fixture |
> | `TestGGUF_Q4_0_parity` | Q4_0 | fixture |
> | `TestGGUF_Q4_K_M_parity` | Q4_K (+Q6_K) | fixture |
> | `TestGGUF_Q4_K_S_parity` | Q4_K_S | fixture |
> | `TestGGUF_Q5_K_M_parity` | Q5_K (+Q6_K) | fixture |
> | `TestGGUF_Q6_K_parity` | Q6_K | fixture |
> | `TestGGUF_Q8_0_parity` | Q8_0 (tinyllama) | fixture |
> | `TestGGUF_gemma3_parity` | Q8_0 (gemma-3-270m) | fixture |
> | `TestGGUF_qwen2_parity` | Q8_0 (Qwen2.5-0.5B) | fixture |
> | `TestGGUF_qwen3_parity` | Q8_0 (Qwen3-1.7B) | fixture |
>
> **So 14 stands, and every commit body citing it is correct.** The reason is now recorded, which is
> the point: the figure is load-bearing in the proof requirement, and "the script said so" is not a
> reason. Note what the table also shows — **11 of the 14 take their quantization from a fixture**,
> so any future classifier that reads test names will undercount for the same structural reason.

int4 is the documented default quantization. **Zero goldens drive it.** And the hole is wider than
that: of the 19 goldens that actually RAN in the 2026-08-12 refresh, **every one is f32**.

| quantization | golden files | did any RUN? |
|---|---|---|
| f32 (explicit or default) | 24 | **19 ran** |
| `int8int8` (W8A8) | 3 — `gemma4_parity`, `gemma4_12b_parity`, `mellum2_parity` | **all 3 SKIPPED** |
| `int8` (weight-only Q8) | 1 — `gptoss_real` | not matched by the goldens regexp at all |
| **`int4` / W4A8** | **0** | — |

So `scripts/refresh_parity_hashes.sh` — the sanctioned freeze-exception path, and the thing that makes a
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

**RUN WHAT EXISTS FIRST — and most of it was UNPLUMBED, not missing.** Done 2026-08-12, `a6c5b57`:

- **(b) the three `int8int8` goldens** skipped for one liftable reason, the same for all three:
  `GOINFER_HEAVY_TESTS` unset. **Two of the three pass here in ~70 s** (gemma4, mellum2). The refresh
  now enables heavy by default. The third (gemma4-12B) skips on a genuinely absent GGUF — an asset
  question, not a plumbing one.
- **(a) the `int8` golden did NOT turn out to be a selector bug.** `TestGptOssReal_logitParity` **does**
  match the regexp. It is invisible because `decoder/gptoss_real_test.go` is behind `//go:build realckpt`,
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


## Freshness sweep — C, D, E, G (2026-08-12)

F was **fifteen for fifteen already fixed**, because it was seeded from `docs/completed/`. These
groups were seeded from conversation, and **the rate is much lower**, which is the useful result:

| entry | state | evidence |
|---|---|---|
| C1 drain fix — CUDA verification | **open** | no CUDA unload/drain test found |
| C2 out-of-tree consumer audit | **open** | needs a fresh no-repo session by design |
| C3 Metal consumer window | **open** | mac batch |
| C4 soak testing | **open** | `internal/serveapp/fuzz_test.go` and `internal/serveapp/chaos_test.go` exist; neither is an hours-long soak |
| D1 trace tap + launch-site table | **open** | no coverage table in `docs/` |
| D2 launch-wrapper commit 1 | **open** | no `cuda/internal/gen` |
| D3 parked flag-pair | **open**, design read done above | — |
| E1 v1.0 gate as written criteria | **open** | prose item, no tree anchor |
| E2 four per-family demotions | **open** | manifest still lists `gpt2`, `granitemoehybrid`, `kimi_k2`, `nemotron_h` as `pending` |
| E3 freeze re-declaration | **DONE `cda8cfe`** | re-declared as a proof requirement, with decider and date |
| E4 `scripts/bench_compare.sh` fix or retire | **FIXED** | it now opens with *"goinfer's OWN numbers only. NOT a peer comparison"* and points at `scripts/bench_peer.py`, which drives both sides |
| E5 promo drafts | **unverifiable** | held in conversation, nothing in the tree to check |
| E6 aikit release | **CLOSED 2026-08-12** — superseded by events, not by reversal | aikit cut `v1.17.0`/`gpu/v0.28.0` (`ada417e`); goinfer is on it (`f33fcaf`). The release met E6's own "a reason a consumer can receive" test |
| G1 LFM2.5 family | **open** | no LFM2 code in the tree |
| G2 `go fix` modernizers | **DONE `3d6ae1e`** | — |

**Rate: 1 of 13 previously-open entries was silently already fixed (E4), against F's 15 of 15.**
Two more (E3, G2) were closed by this campaign and were already recorded.

**That difference is the finding.** F was seeded from a *filed audit* — work done elsewhere, reported
once, never propagated back. C/D/E/G were seeded from *conversation*, where the person who did the
work was the person holding the list. **The burial folder is what produced the 15/15, not the passage
of time.** So the sweep paid for itself once, on the group that came from a document, and should not
be assumed to pay again on groups that did not.

E5 is recorded as **unverifiable** rather than open: nothing in the tree can confirm or deny it, which
is a different state and should read as one.


## Description sweep (2026-08-12) — does each entry match its source?

The status sweep found 1 of 13. **D3 shows description can be wrong while status is right, and
description is what someone acts on.** So: for every open entry with a source outside conversation —
a branch, a commit, an audit line, a script — does the entry describe it correctly?

*(A goldens run in a fresh `git worktree` is fixture-less: the same commit proved 33 goldens in the
main checkout and 7 in the worktree, because the fixture checkpoints are gitignored. The refresh
script now says so when skips outnumber passes — `scripts/refresh_parity_hashes.sh`. Found by running
D3's refresh in the rebase worktree and getting `goldens=7`.)*

| entry | source | description matches? |
|---|---|---|
| D3 | branch `flag-pair-moe-cache` + `BRANCH-NOTE.md` | **NO — corrected `2d28358`.** Called a "parked flag-pair" on a workaround premise; it is an API-surface promotion following `KVPrecision` |
| B4 | a stash that does not exist here | **unverifiable** — the description is all that survives, and it names a file that resolves nowhere |
| C1 | `588052b` (the drain fix) | matches — Metal-verified, CUDA arm untested |
| D2 | design recorded in-entry, no branch | matches; no external source to drift from |
| E2 | `testdata/parity_manifest.json` | matches — the four families are still `pending` |
| E4 | `scripts/bench_compare.sh` | **stale** — the entry says "status unconfirmed, may still measure the two sides differently"; the script now refuses that use and points at `scripts/bench_peer.py`. Corrected in the status sweep above |
| E6 | aikit tree + `gpu/v0.27.0` | **now stale by design** — the tag it pinned is superseded by `gpu/v0.28.0`. Re-checked at the bump: `be049df` is an ancestor of `gpu/v0.27.0`, and across `gpu/v0.27.0..gpu/v0.28.0` the quantized GEMV PTX is byte-identical |
| G1 | `docs/scoping-lfm2.md` | matches |
| P1, P2, P4, P5, P8 | audit lines + the cited source | match; each carries a measured figure or an explicit ESTIMATE label |

**Split: 9 entries had an external source and were checkable; 2 of those 9 were wrong (D3, E4).
4 entries — C2, C3, E1, E5 — have no source outside conversation and are recorded as unverifiable
rather than checked.**

**THE TRIGGER, not a cadence.** An entry's description **and the specific details inside it** — counts,
file names, measurements that no lint covers — are re-read against their source **at the moment the
item is picked up for work** — nothing schedules this and nothing lints it. That is when the
description matters, when someone is already loading the context anyway, and it is exactly what caught
D3: the read happened because the item was next, not because a sweep came due. The cost falls at the
only point where the drift would have changed what someone did.

**That rate (2 of 9) is higher than the status sweep's (1 of 13), and the two are not the same
population.** A description drifts silently because nothing re-reads it against its source; a status
drifts only when work lands elsewhere. The queue's SHA and path citations are now linted, but **no
lint reads an entry's prose against the branch note or audit line it describes** — that remains a
person's job, and this sweep is its baseline.

## Sequencing

**D3 (loaded and bounded) → the mac batch as one session → B1, B2.**

Within the mac batch, **C3 goes FIRST**, not last: it is the largest completely uncovered surface and
it sank once already. Batching it behind two chores is precisely how that happened. Then
`metal-rope-merge`'s push, then B4's stash check.

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

<!-- sha-lint: allow c8b65ba UNPUSHED — the PRE-REBASE D3 flag-pair commit, on the local-only branch `flag-pair-moe-cache`; never pushed, so CI cannot resolve it (it failed there 2026-08-12). DELIBERATELY NOT re-pointed to its rebased successor `bacc04c` (on main via the D3 merge): the two have DIFFERENT patch-ids, and the passage citing this one is historical — it describes the branch as it stood BEFORE the rebase ("its branch predates the fix it completes"), which `bacc04c` no longer illustrates. Re-pointing would keep the lint green by making the surrounding sentence false. Allowlisted rather than laundered. Flagged 2026-08-12 -->
<!-- sha-lint: allow d682315 UNPUSHED — Metal branch `metal-rope-merge`, mac-local; not on origin and not in any clone here. Owner: whoever cited it. P4's "already implemented, snapshot-golden byte-exact" rests on a commit only that machine can see; push the branch or the claim stays unverifiable from anywhere else. Flagged 2026-08-12 -->

<!-- SHA-INDEX: generated by scripts/queue_citation_lint.py --update; do not edit by hand -->

## SHA index

Generated. Every commit id cited above, with the subject it resolved to at the time
of generation. Regenerate with `scripts/queue_citation_lint.py --update`.

| sha | subject |
|---|---|
| `0103b49` | fix(cuda): pay the deferred reservation before sizing the cache (A9-FIX) |
| `0c54e35` | fix(gate): repo hygiene runs what CI runs, derived from ci.yml (B0) |
| `1d0d1ed` | test(decoder): int4 forward goldens — 23 fixtures, 16 architectures (Q1c) |
| `1f6dbe0` | fix(parity,fmt): gofmt the threshold sweep + refresh deps_hash after comment-only core edits |
| `a6c5b57` | fix(parity): the goldens refresh runs quantized goldens, and reports the split |
| `2e91607` | test: refresh parity deps_hash — non-numeric core-file drift (un-reds main) |
| `3d6ae1e` | chore: go fix modernizers, one deterministic pass (G2) |
| `4c26a58` | perf(cuda): parallelise the Gemma final-logit softcap, bit-identical (P3) |
| `588052b` | serve: drain in-flight requests before freeing an unloaded model (fixes the leak safely) |
| `6091e7a` | fix(cuda): size the expert cache by SEARCH over the granularity form (A5) |
| `6edd1ca` | parity: make "validated" MEAN T3 — method-tier gate + honest experimental tier (D2, pre-freeze) |
| `a15a394` | cuda+docs: decline floor, slot-cap gate, driver allocation facts, and seven rules |
| `7cc2f0d` | fix(parity,ci): refresh deps_hash after 38061b1's pread-staging core plumbing (non-numeric) |
| `7ccec1e` | fix(cuda): the expert cache sizes itself — topK was the worst possible default |
| `82b39cc` | docs(parity): document qwen3_5_moe's int8-vs-bf16 movement (v0.8.0 §1 — gate-backed pass) |
| `8fecfad` | ci: scripts/heavy_gate.sh — a runner for the real-checkpoint tier that no CI job executes |
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
| `bacc04c` | feat(serve): --moe-cache-experts / --moe-cache-slots (decisions 2+3) — HELD, trips the parity manifest |
| `f9d5d07` | feat(decoder): dispatch census (B6); close the GGUF-quant gap; reopen B4 |

<!-- /SHA-INDEX -->

<!-- CITATION-INDEX: generated by scripts/queue_citation_lint.py --update; do not edit by hand -->

## SHA index

Generated. Every commit id cited above, with the subject it resolved to at the time
of generation. Regenerate with `scripts/queue_sha_lint.py --update`.

| sha | subject |
|---|---|
| `0103b49` | fix(cuda): pay the deferred reservation before sizing the cache (A9-FIX) |
| `0c54e35` | fix(gate): repo hygiene runs what CI runs, derived from ci.yml (B0) |
| `1d0d1ed` | test(decoder): int4 forward goldens — 23 fixtures, 16 architectures (Q1c) |
| `1f6dbe0` | fix(parity,fmt): gofmt the threshold sweep + refresh deps_hash after comment-only core edits |
| `2d28358` | docs(branch-note): re-derive against the corrected cap (D3 design read) |
| `2e8dfb6` | fix(parity): goldens-gated deps_hash refresh for the aikit v1.17.1 bump |
| `2e91607` | test: refresh parity deps_hash — non-numeric core-file drift (un-reds main) |
| `38061b1` | perf(gemma4-paging): pread expert nibbles straight into the slot buffers |
| `3d6ae1e` | chore: go fix modernizers, one deterministic pass (G2) |
| `4642b7c` | [aikit] gpu(metal): Device.MaxThreadgroupMemoryLength() — tile-memory limit (goinfer M-11) |
| `4c26a58` | perf(cuda): parallelise the Gemma final-logit softcap, bit-identical (P3) |
| `53a96f6` | docs: P10 DISCHARGED — prefill +4.49%; P9's divisor measured and it weakens the flat verdict |
| `588052b` | serve: drain in-flight requests before freeing an unloaded model (fixes the leak safely) |
| `5ece205` | fix(cuda): two VRAM-draining tests never freed — they failed a later test as "a forward moved" |
| `6091e7a` | fix(cuda): size the expert cache by SEARCH over the granularity form (A5) |
| `6edd1ca` | parity: make "validated" MEAN T3 — method-tier gate + honest experimental tier (D2, pre-freeze) |
| `7cc2f0d` | fix(parity,ci): refresh deps_hash after 38061b1's pread-staging core plumbing (non-numeric) |
| `7ccec1e` | fix(cuda): the expert cache sizes itself — topK was the worst possible default |
| `82b39cc` | docs(parity): document qwen3_5_moe's int8-vs-bf16 movement (v0.8.0 §1 — gate-backed pass) |
| `8fecfad` | ci: heavy_gate.sh — a runner for the real-checkpoint tier that no CI job executes |
| `91f359f` | fix(decoder): matmulInto dispatches on the property, not on W8A8 (P7) |
| `93eb7d4` | feat(decoder): gpt-oss real-model path — batched-prefill fix + real gates |
| `9624dd9` | chore(parity): refresh deps_hash for aikit v1.12.0 (goldens-proven non-numeric) |
| `97ee663` | revert(cuda): the expert-cache default back to topK — it broke both real-26B gates |
| `98936cf` | test(goldens): strengthen mamba-2 + deltanet parity fixtures (kill identity weights) |
| `99b3f95` | chore(deps): pin aikit v1.12.0 — gpt-oss MXFP4 reproducible on main |
| `9e5f8fa` | fix(quant): reject --quant that conflicts with a prequant .giw at startup (T1-7) |
| `a15a394` | cuda+docs: decline floor, slot-cap gate, driver allocation facts, and seven rules |
| `a163150` | fix(release): the parity refresh's arch is load-bearing and was unrecorded — stamp it, correct the "Mac tool" error |
| `a6c5b57` | fix(parity): the goldens refresh runs quantized goldens, and reports the split |
| `a79303e` | [aikit] release: prepare v1.16.0 — mmap SpanCache eviction-policy knob |
| `ada417e` | [aikit] scripts: ptx-repro is n/a on darwin, keyed on the PLATFORM not on NVRTC's absence |
| `bacc04c` | feat(serve): --moe-cache-experts / --moe-cache-slots — PARKED on the freeze |
| `bd08936` | fix(gate): cannot-search is not not-found; cross-gate composition; B7 sweep |
| `be049df` | [aikit] gpu(gemv): explicit __fmaf_rn in the quantized GEMV — the bit-identity contraction rule |
| `c6600fc` | audit C-14: argmax index tie-break (match CPU lowest-index on exact ties) |
| `ca29d6c` | cuda: resident context cap becomes configuration-derived (-ctx), VRAM-checked at load |
| `cc238c6` | cleanup: consolidate GINFER_ env vars to GOINFER_ + add env-var registry |
| `cda8cfe` | docs: re-declare the freeze as a proof requirement; clear G2 for amd64 alone |
| `e42e83e` | fix(cuda): name the kernel and both slot counts when a launch runs out of memory |
| `e58ac8a` | fix(parity): refresh deps_hash after f340d4e's guarded int4-scale seam — non-numeric, validated_at preserved |
| `ecc5af2` | chore(parity): refresh deps_hash after default-off diagnostic hooks (non-numeric) |
| `ed81e13` | P1: route top_k=1 to the on-device greedy fast path |
| `eea7f29` | perf(decoder): one gate/up pair per token in MoE, not one per expert (P6) |
| `f33fcaf` | chore(deps): aikit v1.16.0 -> v1.17.0, aikit/gpu v0.27.0 -> v0.28.0 |
| `f340d4e` | metal(9c Step 4): argmax-primary gate + f16-scale confound diagnostic (finding recorded) |
| `f8c4777` | docs(queue): arm64 f32 goldens gate — PARTLY discharged on arm64, 8/19 f32 green, 0 hash delta |
| `f9d5d07` | feat(decoder): dispatch census (B6); close the GGUF-quant gap; reopen B4 |

## Path index

Generated. Every `file:line` cited above, the repo it resolved in, and the trimmed
content of that line. A line that MOVED is reported with its new number; content that
has VANISHED is red, because the citation then claims something the file no longer
supports.

| doc \| path:line | repo | line content |
|---|---|---|
| `docs/QUEUE.md|cuda/argmax_tiebreak_test.go:19` | goinfer | `func TestArgmaxTieBreak(t *testing.T) {` |
| `docs/QUEUE.md|cuda/backend.go:463` | goinfer | `if r.dev, e = CreateSystemDefaultDevice(); e != nil {` |
| `docs/QUEUE.md|cuda/backend.go:591` | goinfer | `if v, err := strconv.Atoi(os.Getenv("GOINFER_SPLITKV_MIN_KEYS")); err == nil && v >= 0 {` |
| `docs/QUEUE.md|cuda/backend.go:836` | goinfer | `// Synchronise: the reservation must be a fact before free VRAM is read, and an async` |
| `docs/QUEUE.md|cuda/resident.go:244` | goinfer | `// backend.go locals; the per-layer KV cache and UploadKV read r.layers[l].kvDim.` |
| `docs/QUEUE.md|cuda/resident.go:397` | goinfer | `func (r *cudaResident) recordUpload(e error) {` |
| `docs/QUEUE.md|decoder/features_test.go:146` | goinfer | `want, ok := admissionGolden[name]` |
| `docs/QUEUE.md|decoder/forwardn.go:378` | goinfer | `for kvh := range nKV {` |
| `docs/QUEUE.md|decoder/forwardn.go:502` | goinfer | `logits[j] = sc * float32(math.Tanh(float64(val/sc)))` |
| `docs/QUEUE.md|decoder/kvsnapshot_gemma4_test.go:10` | goinfer | `func TestSnapshot_refusesNonUniformKVWidth_C05(t *testing.T) {` |
| `docs/QUEUE.md|decoder/layerpaging.go:42` | goinfer | `// mu guards the mutable paging state below (audit C-30). The pager lives on *Model, sha` |
| `docs/QUEUE.md|decoder/mlp.go:82` | goinfer | `func moeMLP(h []float32, lw *LayerWeights, arch *Architecture, be Backend, pager *expert` |
| `docs/QUEUE.md|decoder/model.go:731` | goinfer | `anchor: func (m *Model) ForwardSubCapture(id int, cache *KVCache) (attn, mlp, ctx, mlpPr` |
| `docs/QUEUE.md|decoder/modelsdir_test.go:13` | goinfer | `root := os.Getenv("GOINFER_MODELS_DIR")` |
| `docs/QUEUE.md|decoder/sampler_chunked.go:188` | goinfer | `return drawChunked(e, sums, z, r)` |
| `docs/QUEUE.md|decoder/scratch.go:38` | goinfer | `ws        *linalg.Workspace // W8A8 activation-quant scratch (zero-alloc Into/Batch)` |
| `docs/QUEUE.md|decoder/serialize_shapecheck_test.go:15` | goinfer | `func TestValidateShapes_catchesArchMismatch(t *testing.T) {` |
| `docs/QUEUE.md|internal/giw/bundle.go:114` | goinfer | `if avail := fi.Size() - (tokOff + 4); tokLen > avail {` |
| `docs/QUEUE.md|internal/serveapp/embeddings.go:26` | goinfer | `// Embedding request bounds (audit C-21). /v1/embeddings is deliberately un-queued (the ` |
| `docs/QUEUE.md|internal/serveapp/main.go:432` | goinfer | `// write, so a long stream is unaffected. WriteTimeout stays 0: SSE responses are long-l` |
| `docs/QUEUE.md|linalg/quant.go:113` | aikit | `for k := range K {` |
| `docs/QUEUE.md|metal/model.go:728` | goinfer | `r.residencyBufs = pinned` |
| `docs/QUEUE.md|metal/model.go:827` | goinfer | `r.logitsHost[j] = sc * float32(math.Tanh(float64(v/sc)))` |
| `docs/QUEUE.md|metal/snapshot_golden_test.go:77` | goinfer | `func TestMetalEmbedScale_forwardMatchesForwardEmb(t *testing.T) {` |
| `docs/benchmarks.md|cuda/resident.go:28` | goinfer | `anchor: var (` |
| `docs/cuda-megakernel-spec.md|gpu/attention.go:14` | goinfer | `// uses f64 accumulation; the GPU f32 — cosine ~1.0, not bit-exact).` |
| `docs/cuda-megakernel-spec.md|gpu/decoderunner.go:730` | goinfer | `// moeExpert records one indexed sparse-expert GEMV: dst[n] = expert[idx[slot]]·aq` |
| `docs/cuda-megakernel-spec.md|gpu/decoderunner.go:835` | goinfer | `// relu²→int8 → down + residual into xd. The other kinds fall through to the mixer.` |
| `docs/cuda-megakernel-spec.md|gpu/forward_parity_test.go:36` | goinfer | `func TestWebGPU_forwardParity(t *testing.T) {` |
| `docs/cuda-megakernel-spec.md|gpu/gemv.go:41` | goinfer | `@compute @workgroup_size(64)` |
| `docs/gpu-residency-coverage.md|decoder/registry.go:135` | goinfer | `IntermediateDim:   cfg.IntermediateDim,` |
| `docs/how-inference-works.md|decoder/attention.go:104` | goinfer | `if !arch.LearnedPosEmbed && !arch.isNoPELayer(layer) {` |
| `docs/how-inference-works.md|decoder/attention.go:144` | goinfer | `cache.Append(layer, k, v)` |
| `docs/how-inference-works.md|decoder/attention.go:59` | goinfer | `nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim` |
| `docs/how-inference-works.md|decoder/kvcache.go:126` | goinfer | `subCapture bool` |
| `docs/how-inference-works.md|decoder/kvcache.go:20` | goinfer | `func quantizeHeads(src []float32, q []int8, scales []float32, nKV, headDim int) {` |
| `docs/how-inference-works.md|decoder/model.go:545` | goinfer | `anchor: func (m *Model) runLayers(id int, cache *KVCache) ([]float32, error) {` |
| `docs/how-inference-works.md|decoder/model.go:586` | goinfer | `anchor: func (m *Model) runLayersFromEmbed(h []float32, cache *KVCache) ([]float32, erro` |
| `docs/how-inference-works.md|decoder/registry.go:19` | goinfer | `var registry = map[string]archAdapter{` |
| `docs/how-inference-works.md|decoder/sampler.go:109` | goinfer | `// can never silently diverge. They are separate predicates, not one widened one, so tha` |
| `docs/how-inference-works.md|decoder/sampler.go:116` | goinfer | `// though a temperature is set — the `top_k=1` shape. It is TRUE at any temperature, whi` |
| `docs/how-inference-works.md|decoder/sampler.go:118` | goinfer | `// distribution restricted to ONE token is deterministic regardless of that token's prob` |
| `docs/how-inference-works.md|decoder/session.go:71` | goinfer | `// stale history. Callers must skip it (and reconcile) for an empty prompt, so a rejecte` |
| `docs/ideas-weight-memory.md|decoder/mlp.go:69` | goinfer | `anchor: func mlp(h, out []float32, lw *LayerWeights, arch *Architecture, be Backend, scr` |
| `docs/internal/recon-qwen35-gguf.md|decoder/gguf.go:541` | goinfer | `anchor: func ggufGptOssConfig(g *embed.GGUFFile) (*Config, error) {` |
| `docs/multimodal.md|decoder/config.go:466` | goinfer | `case c.MoeIntermediateSize <= 0:` |
| `docs/multimodal.md|decoder/gguf_qwen35.go:77` | goinfer | `anchor: func ggufQwen35Config(g *embed.GGUFFile) (*Config, error) {` |
| `docs/multimodal.md|decoder/weights.go:344` | goinfer | `const shardIndexFile = "model.safetensors.index.json"` |
| `docs/ollama-chase.md|cuda/resident.go:1066` | goinfer | `// All of it runs ON the executor thread — that thread made the context current — and th` |
| `docs/ollama-chase.md|cuda/resident.go:340` | goinfer | `g4x1, g4x2, g4rn Buffer` |
| `docs/ollama-chase.md|cuda/resident.go:41` | goinfer | `anchor: const ctxCapMarginBytes = 384 << 20` |
| `docs/ollama-chase.md|cuda/resident.go:583` | goinfer | `// declined to the staged/CPU path upstream.` |
| `docs/ollama-chase.md|decoder/forwardn.go:378` | goinfer | `for kvh := range nKV {` |
| `docs/ollama-chase.md|decoder/mlp.go:82` | goinfer | `func moeMLP(h []float32, lw *LayerWeights, arch *Architecture, be Backend, pager *expert` |
| `docs/ollama-chase.md|decoder/model.go:825` | goinfer | `// logits. On the batched archs this runs the layers at M=len in one pass (each` |
| `docs/ollama-chase.md|decoder/model.go:973` | goinfer | `// sample. Identical to the logits path — guarded by ArgmaxEquivalent/GreedyEquivalent.` |
| `docs/ollama-chase.md|decoder/residency.go:677` | goinfer | `return false, "sequential — this backend has no batched prefill (per-token resident forw` |
| `docs/ollama-chase.md|decoder/weightmat.go:202` | goinfer | `var ws linalg.Workspace` |
| `docs/parity-coverage-policy.md|cuda/resident.go:910` | goinfer | `// always been allocated without one, and a hard failure here would regress every driver` |
| `docs/parity-coverage-policy.md|linalg/dot.go:25` | aikit | `sum += a[k] * b[k]` |
| `docs/plan-cpubrrr-steal-and-bindings.md|decoder/registry.go:46` | goinfer | `"gpt_oss":             gptOssArchitecture,     // gpt-oss (20b/120b): sparse MoE + per-h` |
| `docs/plan-cpubrrr-steal-and-bindings.md|linalg/quant.go:327` | aikit | `for i := range M {` |
| `docs/prompts/metal-close-leak-check.md|metal/backend.go:139` | goinfer | `return nil, fmt.Errorf("metal: batched prefill declined — not bit-identical to decode (5` |
| `docs/prompts/metal-close-leak-check.md|metal/model.go:350` | goinfer | `// expert weights, buffer OOM — model.go/moe.go/gemma4_moe.go) into the error this signa` |
| `docs/prompts/metal-close-leak-check.md|metal/model.go:46` | goinfer | `// same-op kernel inherit it. A byte-exact fixture for such an op MUST use context > the` |
| `docs/scoping-lfm2.md|decoder/arch.go:156` | goinfer | `type nemotronParams struct {` |
| `docs/scoping-lfm2.md|decoder/attention.go:94` | goinfer | `if arch.QKNorm {` |
| `docs/scoping-lfm2.md|decoder/config.go:627` | goinfer | `case c.UseQKNorm:` |
| `docs/scoping-lfm2.md|decoder/deltanet.go:99` | goinfer | `// 1. Projection + depthwise causal conv (+ SiLU). Taps t-K+1..t: the last K-1` |
| `docs/scoping-lfm2.md|decoder/forward_qwen35.go:30` | goinfer | `if arch.isLinearLayer(l) {` |
| `docs/scoping-lfm2.md|decoder/kvcache.go:50` | goinfer | `type KVCache struct {` |
| `docs/scoping-lfm2.md|decoder/mamba2.go:89` | goinfer | `// 2. Depthwise causal conv over xBC (+ bias, + SiLU). Taps t-K+1..t: the last` |
| `docs/scoping-lfm2.md|decoder/mamba2_chunked.go:60` | goinfer | `// Depthwise causal conv over xBC (+bias, +SiLU), then split into x/B/C.` |
| `docs/scoping-lfm2.md|decoder/rmsnorm.go:49` | goinfer | `func layerNorm(x, weight, bias []float32, rows, dim int, eps float64) {` |
| `docs/task-admin-unload-drain.md|decoder/speculative.go:123` | goinfer | `// staged CPU cache. Draft is a separate Model with its own claim.` |
| `docs/task-admin-unload-drain.md|internal/serveapp/admin.go:122` | goinfer | `s.regMu.Lock()` |
| `docs/task-admin-unload-drain.md|internal/serveapp/admin.go:95` | goinfer | `anchor: func (s *server) handleAdminLoad(w http.ResponseWriter, r *http.Request) {` |
| `docs/task-admin-unload-drain.md|internal/serveapp/anthropic.go:406` | goinfer | `s.withModelAnthropic(w, req.Model, func(lm *loadedModel) { s.serveMessagesWith(w, r, req` |
| `docs/task-admin-unload-drain.md|internal/serveapp/anthropic.go:535` | goinfer | `anchor: func (s *server) handleCountTokens(w http.ResponseWriter, r *http.Request) {` |
| `docs/task-admin-unload-drain.md|internal/serveapp/helpers.go:78` | goinfer | `func limitInflight(sem chan struct{}, h http.HandlerFunc) http.HandlerFunc {` |
| `docs/task-admin-unload-drain.md|internal/serveapp/main.go:480` | goinfer | `if cfg.sessionDir != "" && cfg.kvSessions > 0 {` |
| `docs/task-admin-unload-drain.md|internal/serveapp/main.go:537` | goinfer | `anchor: func newServer(cfg config) (*server, error) {` |
| `docs/task-admin-unload-drain.md|internal/serveapp/main.go:707` | goinfer | `anchor: func (s *server) loadQwenVisionTower(dir string, int8Tower bool) error {` |
| `docs/task-admin-unload-drain.md|internal/serveapp/openai.go:153` | goinfer | `func (lm *loadedModel) tryEnter() bool {` |
| `docs/task-admin-unload-drain.md|internal/serveapp/openai.go:166` | goinfer | `func (lm *loadedModel) enter(w http.ResponseWriter) bool {` |
| `docs/task-admin-unload-drain.md|internal/serveapp/openai.go:213` | goinfer | `// the safe default). Only Matryoshka-trained models may be sliced; see resolveDimension` |
| `docs/task-admin-unload-drain.md|internal/serveapp/openai.go:464` | goinfer | `if len(imgs) > 0 {` |
| `docs/task-admin-unload-drain.md|internal/serveapp/openai.go:489` | goinfer | `anchor: func (s *server) serveChatText(w http.ResponseWriter, r *http.Request, req chatR` |
| `docs/task-admin-unload-drain.md|internal/serveapp/openai.go:542` | goinfer | `if req.Logprobs {` |
| `docs/task-admin-unload-drain.md|internal/serveapp/responses.go:86` | goinfer | `s.withModel(w, req.Model, func(lm *loadedModel) { s.serveResponsesWith(w, r, req, lm) })` |
| `docs/task-admin-unload-drain.md|internal/serveapp/sessions.go:25` | goinfer | `model   *decoder.Model` |
| `docs/task-admin-unload-drain.md|internal/serveapp/tools.go:19` | goinfer | `s.withModel(w, req.Model, func(lm *loadedModel) { s.serveChatToolsWith(w, r, req, lm) })` |
| `docs/task-admin-unload-drain.md|internal/serveapp/vision_serve.go:119` | goinfer | `s.withModel(w, req.Model, func(lm *loadedModel) { s.serveVisionChatWith(w, r, req, imgs,` |
| `docs/task-admin-unload-drain.md|metal/model.go:972` | goinfer | `// gigabytes; Close reclaims them", which makes calling Close from handleAdminUnload loo` |
| `docs/task-decode-attention-fa.md|cuda/attn_batched_test.go:28` | goinfer | `func TestAttnBatched_bitIdentical(t *testing.T) {` |
| `docs/task-decode-attention-fa.md|gpu/mla.go:27` | goinfer | `// One workgroup per query head, online (FlashAttention-style) softmax over keys so no` |
| `docs/task-decode-attention-fa.md|gpu/mla_test.go:366` | goinfer | `if cos < 0.9999 {` |
| `docs/task-gpu-batched-prefill.md|decoder/residency.go:54` | goinfer | `// ResidentGreedy is an optional capability on a ResidentForward: compute the token's gr` |
| `docs/task-mla-cuda-residency.md|cuda/backend.go:89` | goinfer | `// Without this check the failure is silent — the feature is dropped and the logits are` |
| `docs/task-mla-cuda-residency.md|decoder/arch.go:173` | goinfer | `QLoRARank      int  // q_a_proj bottleneck width; 0 ⇒ direct q_proj (no q-LoRA)` |
| `docs/task-mla-cuda-residency.md|decoder/features.go:126` | goinfer | `// shared taxonomy, so the hardware matrix matches admission.` |
| `docs/task-mla-cuda-residency.md|decoder/features.go:330` | goinfer | `anchor: var residentBackendFeatures = map[string]map[ResidentFeature]bool{` |
| `docs/task-mla-cuda-residency.md|decoder/features.go:48` | goinfer | `FeatMoE               ResidentFeature = "moe"                 // sparse mixture-of-exper` |
| `docs/task-mla-cuda-residency.md|decoder/forward_deepseek.go:188` | goinfer | `func (m *Model) mlaAttentionAbsorb(n []float32, lw *LayerWeights, arch *Architecture, ca` |
| `docs/task-mla-cuda-residency.md|gpu/decoderunner.go:776` | goinfer | `add(c.mlaStorePipeline, bind(c.mlaStoreLayout, kvDown, normW, invFreq, latCache, mlaStor` |
| `docs/task-mla-cuda-residency.md|gpu/mla.go:26` | goinfer | `UNKEYABLE` |
| `docs/task-model-family-deepseek-v4-kimi-k3.md|decoder/arch.go:173` | goinfer | `QLoRARank      int  // q_a_proj bottleneck width; 0 ⇒ direct q_proj (no q-LoRA)` |
| `docs/task-model-family-deepseek-v4-kimi-k3.md|decoder/deltanet.go:70` | goinfer | `anchor: func softplusf(x float32) float32 {` |
| `docs/task-model-family-deepseek-v4-kimi-k3.md|decoder/forward_deepseek.go:89` | goinfer | `invFreq := arch.ropeInvFreq(layer)` |
| `docs/task-model-family-deepseek-v4-kimi-k3.md|decoder/weights.go:1032` | goinfer | `func streamExperts(t embed.Tensor, nExpert, rows, cols int, quant quantMode) ([]linalg.W` |
| `docs/task-moe-streaming.md|decoder/forwardn.go:14` | goinfer | `// MoE FFN itself stays per-row (router picks different experts per token).` |
| `docs/task-moe-streaming.md|decoder/forwardn.go:228` | goinfer | `// Sequential: add the attention residual, then re-norm the updated stream for the MLP.` |
| `docs/task-moe-streaming.md|decoder/mlp.go:81` | goinfer | `// Only the chosen experts are evaluated — the point of MoE.` |
| `docs/task-moe-streaming.md|decoder/moepaging.go:15` | goinfer | `// only K·L per token; the router's top-k selection is the demand signal. The` |
| `docs/task-moe-streaming.md|decoder/moepaging_test.go:11` | goinfer | `// it with the frequency-aware policy (TestSpanCache_evictsLeastRecentWithPolicy),` |
| `docs/task-moe-streaming.md|decoder/residency.go:130` | goinfer | `return m.residentProjsInt4()` |

## Bare file index

Generated. Every file referenced WITHOUT a line number, and the repo it resolves in.
Existence only — there is no line to key content against, which is recorded rather
than papered over.

| file | repo |
|---|---|
| `cuda/allocgran_test.go` | goinfer |
| `cuda/backend.go` | goinfer |
| `cuda/build_ptx.sh` | goinfer |
| `cuda/gemma_bos_build_test.go` | goinfer |
| `cuda/graphs_safe_test.go` | goinfer |
| `cuda/moe_route_demand_test.go` | goinfer |
| `cuda/moe_route_reservation_test.go` | goinfer |
| `cuda/mustalloc_test.go` | goinfer |
| `cuda/nvrtc_compile.py` | goinfer |
| `cuda/prefill.go` | goinfer |
| `cuda/resident.go` | goinfer |
| `cuda/slotcap_test.go` | goinfer |
| `cuda/softcap.go` | goinfer |
| `decoder/gguf.go` | goinfer |
| `decoder/giwquant_test.go` | goinfer |
| `decoder/gptoss_real_test.go` | goinfer |
| `decoder/layerpaging.go` | goinfer |
| `decoder/mlp.go` | goinfer |
| `decoder/model.go` | goinfer |
| `decoder/moepaging.go` | goinfer |
| `decoder/sampler.go` | goinfer |
| `decoder/sampler_chunked.go` | goinfer |
| `decoder/serialize.go` | goinfer |
| `decoder/weightmat.go` | goinfer |
| `decoder/weights.go` | goinfer |
| `internal/serveapp/chaos_test.go` | goinfer |
| `internal/serveapp/fuzz_test.go` | goinfer |
| `linalg/matmul_blocked.go` | aikit |
| `linalg/quant.go` | aikit |
| `scripts/bench_compare.sh` | goinfer |
| `scripts/bench_peer.py` | goinfer |
| `scripts/bench_prompts_calibrate.py` | goinfer |
| `scripts/chatml_tiny_fixture.py` | goinfer |
| `scripts/ci_checks.py` | goinfer |
| `scripts/diff_gemma4_12b.py` | goinfer |
| `scripts/gpu_gate.sh` | aikit |
| `scripts/heavy_gate.sh` | goinfer |
| `scripts/mutation_check.sh` | goinfer |
| `scripts/parity_sweep.sh` | goinfer |
| `scripts/queue_citation_lint.py` | goinfer |
| `scripts/refresh_parity_hashes.sh` | goinfer |
| `scripts/selector_coverage.py` | goinfer |
| `scripts/skip_census.py` | goinfer |
| `scripts/sweep_composition.py` | goinfer |

<!-- /CITATION-INDEX -->

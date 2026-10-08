# Task: one class of audit finding as a mechanical gate — options × paths, fail closed (2026-10)

> **Status 2026-10-08: step 1 DONE — BUILD, with a thin margin.** 71 of 261 counted Critical and Major findings are
> R or B (27.2%), more than three distinct options or families among them; every registered sensitivity clears 25%
> (narrowest 26.7%). Record: [`audit-classes-2026-10-08.md`](../measurements/audit-classes-2026-10-08.md). Its §3
> put one scoping choice to the owner: options are 22 of the 71, families 20, limits 19 and state kinds 10. **Decided
> 2026-10-08: kinds of state join options as the grid's columns; limits stay out (§4.0).** Step 2 not started.

## 1. The class, and why it keeps coming back

The audits so far have produced several hundred findings, read one at a time. One shape recurs: **an option or a
family does not register with every guard or path that needs to know about it**, so a combination nobody wired is
admitted and runs wrong, usually silently. The two Criticals of `docs/audit-metal-2026-09-30.md`, fixed in `de1c7f17`,
are this shape exactly:

- **A-C01:** Metal's `prefillOK` was derived from the model's *features* (`MissingResidentFeatures(prefillFeatures)`)
  and never looked at the *load option* `-kv i8`. Batched prefill wrote half-precision K/V into an int8 cache.
- **F-C02:** the exact prefill attention kernel holds 4096 scores, and nothing checked the context a load was given
  against that bound when the fused kernel could not run.

Further candidates, named in the suggestion that prompted this task and not yet checked against their records: prefix
reuse without a recurrent-state exclusion (2026-09-03), Bailing Hybrid's state missing from every recurrent predicate
(2026-09-10), and resident LoRA and prefix reuse being blind to adapters.

**Half of the machinery exists.** The decoder already declares, per backend, which model features a resident path
implements (`decoder/features.go`: `ResidentFeature`, `ResidentBackendFeatures`, `Model.MissingResidentFeatures`),
pins the result in an admission golden (`decoder/features_test.go`: `TestResidentAdmission_registryCovered`,
`TestResidentAdmission_matrix`), generates `docs/hardware-matrix.md` from it, and checks every family at one
chokepoint (`Architecture.validateResolved`, `decoder/registry.go`). That covers **model features × backends**. It
does not cover **load options × paths**, which is where A-C01 fell.

## 2. Step 1 — test the hypothesis (desk work, about a day)

Classify every Critical and Major finding in:

- `docs/audit-metal-2026-09-30.md`, `docs/audit-metal-2026-09-12.md`, `docs/audit-2026-09-10.md`
- `docs/completed/audit-2026-09-10.md`, `docs/completed/audit-2026-09-02.md`, `docs/completed/audit-2026-08-05.md`

into these classes, one per finding, with the finding's ID and a one-line reason:

| class | meaning |
|---|---|
| **R** (registration) | an option, family or state existed, and a guard, predicate or path that needed to know about it did not |
| **B** (bound) | a hard limit (an array size, a context cap, a byte budget) that a reachable input exceeds |
| **N** (numerics) | a computation is wrong for an input the path does handle |
| **G** (gate) | a test, gate or claim that does not check what it says |
| **O** (other) | none of the above |

A finding in two classes is counted in the first that fits in the order above. For each R and B finding, also record
the **option or family** involved and the **path** that missed it. That list sizes step 2's matrix from what has
actually recurred.

Deliverable: `docs/measurements/audit-classes-<date>.md`, the per-finding table and the counts.

### 2.1 Counting rules — fixed 2026-10-08, before any finding was classified

The classes and §3's thresholds were fixed on 2026-10-01. Reading the six audits showed three things they leave open:
which tiers count as "Critical and Major" when an audit files gates, correctness or performance as tiers of their own;
where the class boundaries fall; and how to treat a finding filed twice. These rules settle them. They were written and
committed before the first finding was classified, and the verdict is read under them alone.

**Which entries count.** Every entry an audit files in a tier above Minor: release blockers, Critical, "gates that
cannot fail", Major (the Sep 12 Metal audit's "Performance — Major" included), and correctness. Gates count because
the critical bar the 08-05 and 09-02 audits state names "a gate that passes when it should fail", and the 09-30 Metal
register files its gates as Major. Not counted: Minor; performance *observations* filed as their own tier (`P-xx`);
checked-and-correct, landscape and open-question sections; carry-forward rows. One ID is one finding (the 09-02 G-10
and 09-10 G-13 "smaller instances" entries count once each).

| audit | counted from | entries |
|---|---|---|
| 2026-08-05 | original entries B-01–B-14, C-01–C-31, G-01–G-06, M-01–M-23 (not the consolidated-dispositions re-listing) | 74 |
| 2026-09-02 | C-01–C-12, G-01–G-10, M-01–M-35 | 57 |
| 2026-09-10 | `docs/completed/audit-2026-09-10.md`: C-01–C-09, G-01–G-13, M-01–M-58. The live `docs/audit-2026-09-10.md` now holds only Minor findings and adds none | 80 |
| Metal 2026-09-12 | M-01–M-11 and M-13–M-16 (no M-12 entry exists; the summary's "M-12" points at M-13's subject), C-01–C-06, G-01–G-10 | 31 |
| Metal 2026-09-30 | the §2 register; A-C01 ≡ F-C01 counted once; the §12 carry-forward rows of Sep 12 IDs not counted again | 23 |
| **total** | | **265** |

**Where the classes divide.** Tested in the order R, B, N, G, O; the first that fits wins.

- **R** — the defect is in product code, and an option value, family or kind of state that exists is not accounted
  for by a guard, predicate, dispatch, serializer or lifecycle path. The fix is a registration: add it to a list,
  predicate or switch, or give it a named decline.
- **B** — a specific limit in product code (an array or threadgroup size, a kernel capacity, a context or position
  cap, a byte budget) that a reachable input exceeds with nothing checking it. The fix is a bound check or a decline.
  Unbounded resource use where the code holds no limit at all is **O**, not B.
- **N** — a path that does handle the input computes it wrong: arithmetic, indexing, rounding, layout.
- **G** — the defect is in a test, gate, CI job, or a claim in docs or comments. A test whose hand-written family list
  is incomplete is G, and is flagged as an *R-shaped gate*.
- **O** — everything else: performance, API shape, release mechanics, leaks, races and panics not caused by an R or B
  miss, request-validation and security items, resource exhaustion.

**Withdrawn entries.** An entry its own audit later records as not a defect is listed with the quote and left out of
both the count and the total.

**Sensitivities, reported beside the verdict and never substituted for it:** (a) without the performance findings;
(b) with R-shaped gates counted as R; (c) with a defect re-filed by a later audit (the 09-12 Metal entries marked
"prior audit … open") counted once.

**Procedure.** Two independent classification passes over the same 265 entries, the second blind to the first;
disagreements are settled by re-reading the entry, and the agreement rate is reported.

## 3. Pre-registration

- **Build step 2** if R and B together are **at least 25%** of the Critical and Major findings classified, and at
  least three distinct options or families appear among them.
- **Park** if R + B is under 15%: the class is real but not common enough to justify a registry, and the audits keep
  catching it by hand.
- **Ambiguous (15–25%):** write the counts up, build step 2 only for the options that recur at least twice, and say
  so.

The threshold and the classes are fixed now, before any finding is classified.

**Result (2026-10-08): build.** R + B = 71 of 261 = 27.2% under §2.1's rules; the two blind passes alone read 31.0%,
and a full review in both directions moved nine R calls out and none in. Sensitivities: 29.9% without performance
findings, 29.9% with R-shaped gates as R, 26.7% with re-filed defects counted once, 25.6% at the extreme floor of
dropping every entry that cites an earlier one. Per-finding table and the recurrence lists that size step 2:
[`audit-classes-2026-10-08.md`](../measurements/audit-classes-2026-10-08.md).

## 4. Step 2 — fail-closed admission for options × paths (if step 1 clears)

### 4.0 Scope decision, 2026-10-08 (owner): kinds of state become columns

Step 1 found options are 22 of the 71 R/B findings and kinds of state 10, with the state misses concentrated on
lifecycle paths (truncate, reset, snapshot, reuse, rollback). Owner decision: **the grid's columns are the recurring
options AND the kinds of state; limits stay out of step 2.** What that changes in the bullets below:

- **Columns, options:** the five that recurred in step 1 — LoRA adapter, quantisation mode, prefill-lever selection
  (`--cpu-fast-attention`, the fused/MMA levers, `--exact-prefill`), paged MoE slots, KV precision — plus every other
  `decoder.Options` field, classified by the registration test as before.
- **Columns, state:** one per kind `KVCache` carries — the four recurrent kinds `hasRecurrentState()` names (DeltaNet,
  short-conv, Mamba-2, KDA), sliding-window rings, int8 KV scales, MLA latents, image blocks with m-RoPE positions,
  captured hidden states (EAGLE), and adapter identity (the LoRA runtime and which adapter built a reused prefix).
  `decoder/kvsnapshot.go` already checks MLA latents beside `hasRecurrentState()` rather than through it — one predicate
  does not cover every path, which is the case for a column per kind.
- **Rows:** the execution paths §4 names, with batched/fast prefill admission and the speculative paths first (19 of
  step 1's 71 between them), plus the lifecycle paths a state column meets: `TruncateTo`, recurrent reset, session
  snapshot and `LoadSession`, resident prefix reuse, and speculative rollback.
- **A second registration test**, beside the `decoder.Options` one: reflection over `KVCache`'s fields fails when a
  state field lands that a lifecycle row has not classified. Bailing Hybrid's KDA state (09-10 C-03) and LFM2's conv
  window (09-02 C-02) are the two misses it would have caught at the commit that added them.
- **Limits are out:** the 19 B findings (threadgroup memory, fixed score arrays, head-dim and alignment caps, context
  caps) are a bound declared by each kernel and checked at its dispatch site, a different mechanism. Not filed here;
  the record's §2 lists them for whoever picks that up.

### 4.1 Stage 1 — kinds of state × lifecycle paths: BUILT 2026-10-08

- **The grid** is `decoder/cachestate.go`: 14 kinds of state (every `KVCache` field is one of them, or is geometry,
  a counter or scratch, with a reason) × 5 lifecycle paths (partial rewind, full reset, snapshot, session prefix
  reuse, speculative rollback), each cell a handling and why it is right.
- **Read, not documented.** `Session.Snapshot` refuses, and a partial `TruncateTo` reports inexact, for every kind the
  grid marks so (`holdsStateHandled`), replacing the hand-listed `hasRecurrentState() || mlaLatent` and recurrent-only
  checks.
- **What the grid changed in behaviour.** (1) A library `Session` that switches LoRA adapter on a warm prefix now
  prefills cold; before, it reused K/V built under the previous adapter (`UseAdapter`'s comment left it to the caller).
  serve was not exposed: it keeps one session LRU per adapter in its own snapshot namespace. (2) Image blocks and m-RoPE
  positions are refused by `Snapshot` and make a partial rewind inexact. Neither is reachable today (VL generations
  use fresh caches); the cells were undeclared, and now fail closed.
- **One cell is caller-bound, by declaration:** adapter × snapshot. The blob does not record the adapter; a restored
  session adopts the first one bound (serve rebinds inside a per-adapter fingerprint namespace). Making the blob carry
  it is a format-version bump, left for the owner.
- **Tests** (`decoder/cachestate_test.go`): the registration test over `KVCache`'s fields; every cell declared and
  within what its path can do; `holdsState` knows every kind; the grid's recurrent row equals `hasRecurrentState()`;
  rewind, snapshot and speculative-rollback cells checked on every own-forward family's real cache and by properties
  that do not consult the grid; the adapter switch through `rewindForReuse`/`reconcile`. Seven planted defects (an
  unclassified field; a snapshot cell set to kept, and to persisted; image blocks declared exact on rewind; the adapter
  check removed; KDA dropped from `hasRecurrentState`; KDA dropped from `holdsState`) each turned its test red.
- **Stage 2** (options × paths, with the admission chokepoints) is next. The Metal admission site (`metal/backend.go`
  `prefillOK`) waits until the in-flight multimodal work there is committed.

- **One chokepoint per path.** Every execution path (sequential decode, batched prefill, MC3 batched decode,
  resident, paged MoE, fast/exact prefill, speculative verify) asks one function whether it admits this model *with
  these load options*, the way `prefillOK` asks about features today. Each path declares which options it supports;
  an option it does not name **declines**, so a new option fails closed instead of running wrong.
- **A registration test.** A reflection-based test over `decoder.Options` fails when a field lands that no path has
  classified, as `TestOptions_everyFlagReachesOptions` (`internal/loadflags/loadflags_test.go`) already does for the
  CLI flags. A new path fails the same test until it declares its options.
- **"Tested" means driven through the real caller.** A supported cell names a test that runs that option through that
  path from the entry point a user calls. CLAUDE.md's rule applies: a unit test that supplies its own calling
  convention vouches for itself, not for the system, which is how A-C01 and F-C02 survived. A declined cell names the
  decline message the user sees.
- **Generated like the hardware matrix.** The option × path × backend table is generated and committed, so a review
  reads it instead of rediscovering it. At about 8 options × 7 paths × 4 backends it is roughly 200 cells; most start
  declined and move to supported only with a test.
- **Pattern to copy:** A-C01's fix (`!r.kvI8` in `prefillOK`, a named decline in `PrefillPath`/`PrefillLast`, and
  `TestPrefill_declinesInt8KV` red without it) is one cell of this matrix done by hand.

## 5. Not in scope

Rewriting existing guards that already work; performance; the HTTP layer's own option handling (`serve`'s
per-model overrides end in a `decoder.Options`, so they are covered once the options are).

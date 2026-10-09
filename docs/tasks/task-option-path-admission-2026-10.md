# Task: one class of audit finding as a mechanical gate — options × paths, fail closed (2026-10)

> **Status 2026-10-08: step 1 DONE — BUILD, with a thin margin.** 71 of 261 counted Critical and Major findings are
> R or B (27.2%), more than three distinct options or families among them; every registered sensitivity clears 25%
> (narrowest 26.7%). Record: [`audit-classes-2026-10-08.md`](../measurements/audit-classes-2026-10-08.md). Its §3
> put one scoping choice to the owner: options are 22 of the 71, families 20, limits 19 and state kinds 10. **Decided
> 2026-10-08: kinds of state join options as the grid's columns; limits stay out (§4.0).** Step 2 built the same day
> as two registries (§4.1 state × lifecycle, behaviour-changing where a cell was unsafe; §4.2 options × paths, a
> ratchet with no behaviour change, owner's call): [`option-state-grid.md`](../option-state-grid.md). Open: 9 option
> cells admitted untested (57 at build; 48 moved out the same day), each held open by a finding in §4.3.

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
- **Adapter × snapshot is persisted (owner decision 2026-10-08).** Snapshot format v3 records the adapter's name;
  `LoadSession` rebinds the model's adapter of that name, and refuses the blob when the model has not loaded it. A
  restored session bound to a different adapter goes cold through the reuse check; serve's `bindAdapter` now also
  clears a base LRU's binding, so a mismatch there goes cold too. v2 blobs are refused by the version guard: a one-time
  cold prefill per stored session, since snapshots are a regenerable cache. The name is the identity, so an adapter
  reloaded under the same name with different weights across a restart is not caught. Four planted defects (restore
  drops the adapter, writer drops it, an unloaded adapter accepted, `kvAdapter` not restored) each turned a test red.
  Driven end to end by two more: `TestSession_snapshotAdapterContinuation` (a real compute-time adapter through the
  forward; the restored cache must match a never-snapshotted session's, compared on K/V because the synthetic model's
  greedy tokens are the same under every adapter) and `TestSessionLRU_restoreBindsTheLRUsAdapter` (serve's
  `save`/`load`; red with the old `bindAdapter`). Dropping the adapter on restore is caught by the first only: serve
  rebinds the LRU's adapter itself, so at that level it is redundant.
- **Tests** (`decoder/cachestate_test.go`): the registration test over `KVCache`'s fields; every cell declared and
  within what its path can do; `holdsState` knows every kind; the grid's recurrent row equals `hasRecurrentState()`;
  rewind, snapshot and speculative-rollback cells checked on every own-forward family's real cache and by properties
  that do not consult the grid; the adapter switch through `rewindForReuse`/`reconcile`. Seven planted defects (an
  unclassified field; a snapshot cell set to kept, and to persisted; image blocks declared exact on rewind; the adapter
  check removed; KDA dropped from `hasRecurrentState`; KDA dropped from `holdsState`) each turned its test red.

### 4.2 Stage 2 — load options × execution paths: BUILT 2026-10-08, as a ratchet

**Owner decision 2026-10-08: no behaviour change in this stage.** A combination that runs today with no test that
drives it stays admitted and is declared *admitted, untested*; the count of such cells may only fall. This replaces
the bullets below on one point: cells do not "start declined" — they start honestly labelled.

- **The grid** is `decoder/optiongrid.go`: every `decoder.Options` field × 7 paths (CPU decode, CPU batched prefill,
  CPU batched decode, GPU resident decode, GPU resident prefill, speculative verify, session reuse and snapshot). Each
  cell is *tested* (names a test), *declined* (names the function that refuses it, and a test where one exists),
  *admitted, untested*, or *n/a* (with a reason, quoting the option's own doc comment where it has one). Nine fields are
  load-only (read while loading, by no path afterwards), each with a reason. Generated page:
  [`docs/option-state-grid.md`](../option-state-grid.md), beside the cache-state grid.
- **"Tested" is strict.** The test must set the option through `Options` and run the path. Applying that cut five
  claims on the way in: `TestInt4_forwardParity` runs per token (decode, not batched prefill); `TestKVI8_genParity` and
  `TestKVI8_batchedPrefill` set the internal `kvI8` themselves; the Metal int8-KV parity tests drive kernels;
  `TestMC5_prefillChunkInvariance` chunks `PrefillLast` itself and never sets `ResidentPrefillChunk`. Each of those
  shows the path works, not that the option reaches it.
- **Where it stands: 9 cells admitted untested, 46 tested, 8 declined** (KVPrecision at Metal's `PrefillPath`, Quant
  int4 at `SpecDecodeConflict`, KVQuant at `cpuBatchCacheEligible`, the three MoE paging options at
  `cpuBatchModelEligible`, EmbedInt4 on both resident paths at Metal's `int8Buf`). 57 at build; 48 moved out on
  2026-10-08. The CPU columns' tests run in CI; the GPU columns' (§4.3) run on a Mac only. First round, ten cells:
  - `TestOptionPath_cpuBatchedPrefill` (CPU batched prefill × Quant, KVQuant, ActQuantGroup, ExactPrefill, EmbedInt4):
    a Session's batched prefill must leave the same K/V, bit for bit, and pick the same first token as the per-token
    prefill on the same model.
  - `TestOptionPath_specVerify` (speculative verify × ActQuantGroup, CPUBatchDecode, EmbedInt4, ExactPrefill,
    KVQuant): Session n-gram speculative decoding must emit plain greedy's tokens and leave the same K/V, with the
    production n-gram drafter and with a drafter built so every block is partly rejected (on this model the n-gram
    drafter's blocks are mostly accepted, so it alone never exercised the rollback). Tokens alone were too coarse:
    greedy picks on a model this small do not move under a small numeric error.
  - Each case first checks that its option took hold on the loaded model. Planted defects, each red: the speculative
    rollback skipped; every M>1 W8A8 matmul ignoring ActQuantGroup (K/V off by 0.008); batched prefill ignoring the
    int8 cache; Load dropping KVQuant.
  - Observed, not a defect: a speculative block accepted past maxTokens stays committed, so the session ends a few
    positions past plain decode, holding greedy tokens it never emitted (2 here). Its tokens and cache agree, it
    snapshots, and the next turn reuses the shared prefix.

  - Second round, eight cells. `TestOptionPath_cpuBatchedDecode` (CPU batched decode × Quant, ActQuantGroup,
    EmbedInt4): concurrent generations with CPUBatchDecode on must each match the same generation alone on a
    batching-off model, tokens and K/V, with a control that batched steps of two or more ran. `TestOptionPath_sessionSnapshot`
    (session reuse and snapshot × Quant, CPUBatchDecode): a session snapshotted after one turn, restored and continued
    must reuse its whole stored prefix and match an uninterrupted session, tokens and K/V. KVQuant × CPU batched decode
    is a decline, not a gap: `cpuBatchCacheEligible` keeps int8 caches out of the batcher, and
    `TestCPUBatch_ineligibleCachesBypass` already held it. KVQuant and EmbedInt4 × CPU decode are booked to
    `TestOptionPath_cpuBatchedPrefill`, whose sequential reference is the decode path; each was booked only after a
    defect planted in the decode-only code (the int8 attention branch; the int4 embedding lookup) turned it red.
    Other planted defects, each red: every M>1 W8A8 matmul ignoring ActQuantGroup (batched decode); the batched
    step's fused int4 q‖k‖v given the wrong group; a restore dropping its last stored position.
  - Third round, the MoE paging options (StreamWeights, WeightCacheBytes, MoEPager; 12 cells). The committed
    mixtral-tiny cannot exercise them: its int8 experts (8 KB) do not survive page rounding, so Load builds no pager.
    `ogMoEFixture` writes a synthetic Mixtral (4 experts of 64 KB per int8 projection) and `ogGIW` serializes it to
    the .giw these options page from. `TestOptionPath_moePaging`: on CPU decode, CPU batched prefill and speculative
    verify (both drafters), a paged load must emit the tokens and leave the K/V, bit for bit, of the same path on the
    same .giw fully resident, under StreamWeights at the auto budget, WeightCacheBytes 1, and MoEPager mmap and pool at
    1 byte. Effect checks: a pager was built; the requested mode was built; at 1 byte the pager's budget is below the
    auto budget's and it evicted (evictions alone do not show the budget took hold: the pread pool rounds its slots
    and evicts even at the auto budget, which a planted defect exposed). CPU batched decode is a decline:
    `cpuBatchModelEligible` refuses MoE (and a dense StreamWeights load through its layer pager);
    `TestOptionPath_moePagingDeclinesCPUBatch` checks the refusal and that four concurrent generations, each
    re-faulting experts the others evict, match the same generations run alone on the resident model. Race-detector
    clean. Planted defects, each red: the pool reading every expert one byte off; the pool keeping an evicted
    expert's slot mapping; the mmap pager never touching (budget unenforced); Load ignoring WeightCacheBytes; the
    batcher admitting MoE. StreamWeights on a dense .giw streams layers instead (layerPager), which llama-tiny is
    too small to engage; that branch stays with TestLayerPaging_bitExact, which needs a downloaded GGUF.
  - Fourth round: the GPU-resident columns on Metal (18 cells), below in §4.3.
- **What fails closed:** a new `Options` field until it is classified on every path
  (`TestOptionGrid_everyOptionClassified`); a cell naming a test or decline that does not exist in any module
  (`TestOptionGrid_cellsCarryEvidence`); a rise in the untested count, or a fall not booked into the ceiling
  (`TestOptionGrid_ratchet`); a stale generated page (`TestOptionStateGrid_fresh`). Four planted defects (a new
  field; a missing test name; a tested cell dropped to untested; a hand edit to the page) each turned its test red.
- **Not yet done, and why.** The §4 "one chokepoint per path asks one function" refactor would change admission code,
  including Metal's `PrefillPath` in files the multimodal work has open; under the no-behaviour-change decision it is
  better done cell by cell, each with the test that moves it from untested to tested. New *paths* are not detected
  automatically; adding one means adding it to `ogPaths`, which then fails every option until each is classified.

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

### 4.3 The GPU-resident columns on Metal (2026-10-08), and three findings

`metal/optiongrid_metal_test.go` (darwin, `goinfer_testhooks`) drives each option through `decoder.Options` on the
committed llama-tiny at int8int8 (Metal runs int8 or int4 weights; an f32 load stays on the CPU), checks the option
took hold and the model went Metal-resident, and uses the generation's reported log-probabilities (chosen token and
top five) as a numeric fingerprint through the public entry points.

- `TestOptionPathMetal_neutralOptions`: ResidentContext and ResidentKVSlots size the KV allocation and may not change
  a number; on resident decode, resident prefill, speculative verify and session reuse they match the baseline
  exactly (8 cells).
- `TestOptionPathMetal_quant`: Quant on resident decode (teacher-forced logits, every position within cosine 0.999 of
  the CPU at the same Quant) and resident prefill (the CPU's first token, log-probabilities within 0.02) (2 cells).
- `TestOptionPathMetal_kvPrecision`: KVPrecision f16 equals Metal's default exactly (Metal keeps resident KV at f16
  whatever is asked; `ResidentKVPrecision` says so, the Options doc's "f32 default" does not hold on Metal); i8 decode
  within cosine 0.999 of f16; speculative verify equal to plain decoding; session reuse within 0.02 of a cold run.
  Resident prefix reuse on Metal is not bit-identical to a cold prefill even at f16 (about 0.004 here), and
  `resident_reuse.go` does not claim it is (3 cells).
- `TestOptionPathMetal_exactPrefill`: the prefill equals a knob-forced sequential prefill exactly and differs from
  the batched f16 one (1 cell).
- `TestOptionPathMetal_prefillChunk`: under MC3 with another generation decoding, a 96-token prompt prefilled in
  32-token chunks matches the same prompt prefilled whole; the batcher's prefill-pass count shows the chunks ran
  (1 cell). Speculative verify × ResidentPrefillChunk is n/a: chunking is `mc3Prefill`'s, and the speculative paths
  claim the resident exclusively and prefill whole.
- `TestOptionPathMetal_embedInt4Declines`: int4 with EmbedInt4 is declined at Metal's `int8Buf` and its output is the
  CPU's exactly (2 declined cells).
- Planted defects, each red: ResidentContext leaking into KV precision; the resident speculative rollback keeping a
  rejected position; Load dropping KVPrecision, ExactPrefill or EmbedInt4; a chunk boundary skipping a position; the
  resident embedding scaled 5% (decode); the resident prefill's positions shifted by one (prefill). Two first attempts
  were harmless and replaced: re-prefilling a chunk's last position rewrites identical values, and a uniform
  embedding scale is mostly removed by the RMS norm before the prefill's first token.
- **Metal only.** The grid has no backend axis, and finding 1 shows a cell can hold on CUDA and not on Metal. CUDA's
  resident columns are proven where CUDA tests exist (ActQuantGroup) and not otherwise.

**Findings, behaviour unchanged pending the owner:**

1. **ActQuantGroup is silently ignored by Metal.** Nothing in `metal/` reads it, and for a family without the
   activation hazard `residentAdmission` admits the load: `Quant: "int8int8", ActQuantGroup: 32` on llama-tiny runs
   Metal-resident with log-probabilities identical, bit for bit, to the same load without the option. The Options
   doc says resident backends other than CUDA decline to the CPU when it is set. Reachable from a library caller;
   the CLIs set the group only for hazard families, which Metal does decline (to the CPU, not resident, although
   `internal/modelload`'s `activationSafeQuant` comment says Metal runs them resident at per-32). Fix candidates: a
   named decline in `residentAdmission` when the group is set on Metal or WebGPU, or the doc and comment corrected.
   ActQuantGroup × resident prefill stays untested.
2. **MoECacheExperts on Metal is not bit-identical to fully resident.** The doc's claim cites CUDA tests. On
   mixtral-tiny at int4 with fewer slots than experts (2 or 3 of 8), decode-only log-probabilities differ from the
   fully resident model by about 0.004; with slots for all experts they are identical. With int8 experts Metal's
   build panics ("experts are not int4-direct") and the recover turns it into a decline, rather than a named refusal.
   The eight MoECacheExperts and MoECacheSlots resident cells stay untested.
3. **ResidentPrefillChunk below Metal's batched-prefill floor breaks chunk invariance.** With two KV slots the floor
   is 32 tokens; at chunk 8 or 16 a prompt prefilled in chunks while another generation decodes differs from the same
   prompt prefilled whole by 0.003-0.004 in log-probability, so the reply depends on whether someone else was
   decoding, which `mc3_batch.go`'s comment rules out. At chunk 32 and above the results are identical. serve's
   default (512) is unaffected; a library caller choosing a small chunk is not. A fix candidate: raise the chunk to
   the floor, or decline below it.

## 5. Not in scope

Rewriting existing guards that already work; performance; the HTTP layer's own option handling (`serve`'s
per-model overrides end in a `decoder.Options`, so they are covered once the options are).

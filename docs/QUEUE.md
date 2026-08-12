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
instruction does: **8 slots / 0% hit / ~5 tok/s · 16 / 57.3% / 11.33 · 38 / 81.6% / 16.98**. Note
that at the default of 8 the cache is **inert** — the routed set exactly fills it and nothing
survives to the next token.

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

Why it matters: 105 variables, exactly one of which anything has ever set. Six have no prefix at all
(`ZZBASE`, `GEMMA3_4B`, `G4_TRACE`, `NOISE_FLOOR_CKPT`, `ROUTER_CAPTURE_OUT`, `GIW_BIG`) and are
only findable *as a class* if something enumerates `os.Getenv` mechanically. A markdown table
maintained by intention drifts on the first variable someone adds.

**A3 · Make the launch OOM say what it is** — `linux`, NOT blocked on A1

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

**A7 · Confirm the corrected cap by run** — `linux`, after A5

33 should succeed; 34 should fail. **Pre-registered: free after `allocSlots` at 33 slots is
450,494,464 B exactly.** State it before running, not after.

33 clears the margin by only 47,841,280 B, so treat a **failure at 33** as information about the
margin rather than about the formula — the closed form predicting the requirement correctly and the
margin being too thin to absorb what follows are different findings with different fixes.

**A8 · Is `fRoute` the first launch?** — `linux`, **CLOSED**

`fRoute` is **not** the first launch of the token — `ropeKV` (from `gmod`) and `fAttn` (from the
glue module) precede it. But it is plausibly the first launch out of **`moePTX`**, so a lazily
deferred module load is attributable to it exactly as a first-launch would be. A9 stands.

**A5 · The corrected cap must be a SEARCH, not a division** — `linux`, design fixed in advance

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

**A9 · Force `moePTX` to load BEFORE `allocSlots`, then re-run at 34** — `linux`, **TRIGGERED**

Not conditional. What survives every instrument question about the 34-slot failure is that free VRAM
at the moment of the attempt was **at least 189.6 MiB**, and a one-block routing kernel cannot want
that. So this is **not exhaustion**, and A5 is necessary without being the whole remedy.

**Read the decrements, not the absolute value.** The earlier threshold ("run only if free at the
failing launch exceeds ~100 MiB") was written before the post-allocation headroom was known and
reads the wrong way round: the closed form predicts **198,836,224 B free after `allocSlots` at 34
slots**, so a reading above 100 MiB is the *expected* case, not a signal. The two informative gaps:

- free after `allocSlots` → free at first launch — what `gmod` and the glue module cost.
- free at first launch → free at failing launch — what the successful launches cost.

The only outcome that would make A9 unnecessary is free after `allocSlots` coming back **far below**
198,836,224 — which would mean the closed form is wrong and the cache took more than predicted. That
reopens A1; it does not close A9.

Rationale: `fRoute` is the first kernel launched out of `moePTX` (`ropeKV` comes from `gmod`,
`fAttn` from the glue module), so a lazily-deferred module load is attributable to it exactly as a
first-launch would be. The cap is computed from a free-VRAM reading taken **before** that load. That
cost is invisible to before/after readings around `allocSlots`, and invisible to a between-slot-count
delta, because it does not scale with slots.

It is **additive with the rounding shortfall, not an alternative to it**: rounding eats into the
headroom the 384 MiB margin was sized to provide, and the module load then spends from what remains.

Experiment: force `moePTX` to load while free VRAM is still at its full ~3.8 GB, then re-run at 34
slots. Branches, pre-registered:

- `fRoute` launches after the forced load → module load was the mechanism; the fix shape is to size
  the cache **after** deferred fixed costs are paid, not before.
- `fRoute` still fails → module load excluded; candidate list reopens one entry shorter.
- the forced load itself fails → same finding, relocated to where it is visible. That is a result.

Read-only on the allocation path. The reordering is an **experiment first and a fix only after it
answers**.

**Sequencing constraint — this is the part that can be lost silently.** A9 reproduces at 34 slots,
and A5 fixes the cap to 33, which makes 34 unreachable. **Run A9 before A5 lands, or with an
explicit cap override — and record which was used.** A run at the new cap of 33 would simply pass
and look like confirmation, so the loss would leave no trace.

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
GEMV int8 / int4**; **`capSlots` and its inline copy in `allocSlots`**; **SIMD / scalar widen**.

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

**P2 · Scalar `int8→f32` widen on the LM head** — aikit `linalg/quant.go:113`

A SIMD widen sits in the same package. Estimated **several ms/token at large vocab**. Not frozen,
so the work is unblocked — but shipping it **reverses E6** (aikit release deferred to v1.0), which
must be an explicit decision rather than one arrived at by landing the patch.

Bit-identity is **structural** if the SIMD path is purely elementwise: `int8→f32` widening is exact,
and a per-element scale is a single multiply with no reordering freedom. Verify that condition holds
and it needs no parity argument at all.

**P3 · Gemma final-logit softcap, serial O(vocab) `tanh` on the sampling path** — `cuda/resident.go:1561`

Estimated **10–30%**, Gemma with sampling only. A host parallel-for is bit-identical by construction.
Not frozen. **Queued for work now.** Measure with the sampling configuration recorded on **both**
sides, same method both sides — this is the rule the 476/268 headline broke.

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

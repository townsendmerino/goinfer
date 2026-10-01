# Task: one class of audit finding as a mechanical gate — options × paths, fail closed (2026-10)

> **Status: SCOPED 2026-10-01, not started.** Step 1 is desk work and decides whether step 2 is built (§3). To run
> after v0.20.0 is tagged, beside `task-library-surface-review-2026-10.md`, since both feed the v1.0 freeze and the
> review may change which options exist.

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

## 3. Pre-registration

- **Build step 2** if R and B together are **at least 25%** of the Critical and Major findings classified, and at
  least three distinct options or families appear among them.
- **Park** if R + B is under 15%: the class is real but not common enough to justify a registry, and the audits keep
  catching it by hand.
- **Ambiguous (15–25%):** write the counts up, build step 2 only for the options that recur at least twice, and say
  so.

The threshold and the classes are fixed now, before any finding is classified.

## 4. Step 2 — fail-closed admission for options × paths (if step 1 clears)

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

# Parity staleness gate — stop diagnostic seams from tripping every `core` family

> **Status:** planning. **Not urgent** — the current resolution ("refresh `deps_hash`
> with forward goldens green") is correct every time; this is friction, not a hole.
> **Trigger:** promote Phase 2 when capture-seam additions to `core` become common
> enough that the refresh is recurring toil. Recorded now so the next person starts
> from the design, not a re-derivation.

## The problem (grounded)

`freshDepsHash` (`decoder/parity_manifest_test.go`) is a **whole-file content hash** over
a family's `uses` shared sets + `own` files. Diagnostic capture seams have to live in
`core` forward files — the `subCapture` field (`kvcache.go`), the `if cache.subCapture`
hooks and `ForwardSubCapture` (`model.go`) — because a struct field and inline hooks
**cannot** move to an excluded file. So every capture-seam addition changes the `core`
bytes and re-stales **every family with `uses: [core]`** (the gate reports the first
alphabetically — `deepseek_v2` this time). The change is provably non-numeric (guarded,
default-off, read-only; forward goldens stay green), so the policy's
[*"a `deps_hash` refresh is not a re-validation"* exception](parity-coverage-policy.md)
applies and a bare refresh is correct. But it recurs on every future seam.

## The invariant that governs every option (read first)

The staleness gate is a **correctness gate**: it keys on `deps_hash` so a parity record
that predates a numeric change **cannot** pass. Its conservatism is the feature — a
whole-file hash never misses a numeric change. Therefore:

> **The fix may reduce false positives (non-numeric trips) but must add ZERO
> false-negative risk.** A false negative here = a stale parity record passes = the exact
> silent-wrong-output class the whole parity system exists to prevent. Friction is
> cheap; a missed numeric change is not.

Every option below is judged against that asymmetry.

## Options, weighed against the invariant

**A — Symbol-level / AST hash (hash the numeric functions, not the file bytes).**
Precise, but the *sound* version requires hashing the **verified transitive call-graph
from the forward entry point** — miss one transitively-touched symbol (a new struct field
the forward later reads, a helper it calls) and you get a false negative. That's a lot of
machinery, couples the gate to Go's AST across toolchain versions, and its failure mode is
exactly the silent-wrong hole. **Reject as the hashing mechanism.** (Keep only as a
documented "only if precision ever truly matters, and only with a proven call-graph
closure" — a high bar it currently doesn't clear.)

**B — Carve-out / strip guarded-diagnostic blocks from the hash** (`//parity:ignore`, or
auto-strip `if cache.subCapture { … }`). Tempting and simple, but soundness rests on the
carved block being **guaranteed** non-numeric — and a mismarked block, or a `tap` that
mutates a slice it was handed, is a false negative. That's soundness-**by-convention**,
which reintroduces the risk the gate exists to remove. **Reject as the hashing mechanism.**

**C — Stabilize the capture surface (architectural; the durable reducer).** Don't make the
hash smarter — make `core` **stop churning**. Replace the ad-hoc `subCapture` seams with a
**single generic tap**: one nil-able field + a `tap(id TapID, layer int, data []float32)`
dispatch, added to `core` **once**, general enough to cover the tap points this session
already proved useful (per-layer hidden, per-layer KV, sub-layer contributions,
pre-final-norm residual, per-channel). All diagnostic **logic** — `ForwardSubCapture`'s
body, the confirmer tests, the target curve, `LayerKVForTest` — moves to **excluded**
files that implement/consume the tap. After the one-time addition, a future hunt adds a
`TapID` constant + one guarded line **at most**, not a new seam. Preserves the whole-file
hash (fully sound); moves the churn out of `core`. Honest limit: a genuinely new tap point
still trips the hash **once** — but that trip is now a one-liner, handled trivially by D.

**D — Sound, scripted, goldens-gated refresh (operational; the immediate friction-kill).**
Operationalize the policy exception so the correct resolution is one command, not manual
reasoning + a 20-hash edit. `scripts/refresh_parity_hashes.sh`:
1. **Precondition (enforced):** the affected families' forward goldens are green in this
   run — refuse to refresh otherwise. This *mechanizes* "provably non-numeric": no green
   goldens, no refresh.
2. Refresh only `deps_hash` (preserve `validated_at`/metrics/dates — same as the manual
   fix just done).
3. Record the proof: the goldens-green commit + a one-line reason into the commit / a
   refresh log, so the exception is **auditable**, never a silent re-hash.
Zero soundness change (the hash is untouched); it just makes the recurring refresh cheap
and abuse-resistant.

## Recommendation

**Do D now; do C when the trigger fires; reject A and B as hashing mechanisms.** D removes
the friction immediately at zero soundness cost; C removes the *cause* (core churn)
durably while keeping the hash airtight; A/B buy precision by trading away the one property
the gate cannot lose. The whole-file hash stays the mechanism — conservative on purpose.

## Phased plan

- **Phase 0 — done.** The `deps_hash` refresh landed (goldens green, numerics unchanged).
- **Phase 1 — D, DONE (`scripts/refresh_parity_hashes.sh`).** Goldens-green gate → refresh
  `deps_hash` only → assert nothing but `deps_hash` moved → print an auditable proof block.
  Refuses (exit 1) on any failed golden OR a vacuous all-skip. Break-it-first verified: a
  corrupted `argmax` golden → RED standalone → script refuses exit 1; happy path here runs 14
  goldens green (dense/MoE/MLA/SSM/Gemma) and refreshes cleanly. No code-path change.
- **Phase 2 — C, the real fix (trigger: recurrence).** Design the `TapID` set from this
  session's seams; add the single generic `tap` dispatch + field to `core` once; move all
  capture logic to excluded files (`*_capture_test.go` / an excluded `capture` file);
  delete the ad-hoc `subCapture` hooks. Land it as a **pure refactor** — the one-time
  `core` trip is a Phase-1 refresh, and every family's forward goldens must stay green
  (bit-identical) or the refactor changed behavior and stops.
- **Rejected, recorded:** A (symbol hash) and B (carve-out) — with the soundness reasoning
  above, so they aren't re-litigated.

## Verification (break-it-first, per the gate-audit discipline)

The fix must itself be gated — a change to how the gate decides staleness is exactly where
a silent hole would hide:

1. **Churn-elimination is real (C):** adding a new `TapID` + tracer *implementation in an
   excluded file* changes **no** `deps_hash`. (Assert freshness stays green.)
2. **Soundness preserved (C + always):** a genuine numeric edit to a `core` forward file
   **still trips** `TestParityManifest_fresh`. Mutate one line of real math, confirm RED,
   revert — the same break-it-first proof the Metal/CUDA gates now carry.
3. **Exception can't be abused (D):** `refresh_parity_hashes.sh` **refuses** to refresh
   when a forward golden is red. Force a red golden, confirm the script exits non-zero.

## One-line principle

Keep the hash conservative (whole-file, never misses a numeric change); attack the
friction **operationally** (a sound goldens-gated refresh) and by **moving the churn out
of `core`** (one stable tap surface) — never by making the gate less able to fail.

---

## Phase-2 trigger fired at ~27 — classification and verdict: **DON'T build C**

The self-announcing trigger fired. Before designing option C, the ~27 past refreshes were
classified. The verdict revises the Phase-2 plan above: **do not build C.** The evidence:

### 1. The counter is defective — fix this first
The trigger counts `git log --grep='^Parity-Deps-Refresh:'` = **28**. But only **17 actually
changed `deps_hash`.** The other 11 carry the trailer for **dependency bumps** (`1b90b5c`,
`50eaa89`), cuda/MoE feature work (`6f3b084`, `4b63ed4`, `2f51449`, `424645c`, `2b7e78a`),
docs (`6c995fe`, `c77e952`, `f93bda1`), and the script's own creation commit (`7c2c74a`). The
trailer is **overloaded**: "Deps" reads as both `deps_hash` and *dependency*. So the trigger
fired on **~65% inflation** (28 vs 17).

Fix: split the trailer (`Deps-Hash-Refresh:` vs `Dependency-Bump:`), or count the real diff
(`git show $h -- testdata/parity_manifest.json | grep -c '^+.*deps_hash'`).

**Re-run with correct counting:** 17 real refreshes — and of those, only **2** are the
inert-diagnostic-seam churn C exists to remove (bucket (a) below). The trigger's threshold
(`prior >= 2`) is itself miscalibrated: it should count *seam* refreshes against a real
recurrence bar, not all trailers ≥ 2. Two seam-refreshes across the whole project history is
not recurrence. **Under corrected counting + the correct metric, the track does not fire on
its own premise** — fix the counter and it may close on its own.

### 2. Classification of the 17 real refreshes

- **(a) inert diagnostic seam** (guarded, default-off no-op — the `f340d4e` archetype; pure
  tax, the only thing C removes): **2** — `e58ac8a` (f340d4e int4-f16-scale seam),
  `ecc5af2` (default-off diagnostic hooks).
- **(b) real numeric change** (the alarm working): **1** clear — `625303e` (gemma4 RoPE base
  fix) — plus 6 *planned family additions* (`9c04ea3`/`9e83043`/`51ea350`/`6eea5fa` gemma4-moe,
  `93eb7d4`/`9a03293` gpt-oss) that are legitimate new-family validations, not the alarm
  catching a regression.
- **(c) non-numeric shared-file edit / hash-scope** (a real edit to a hashed file that did not
  change validated numerics but restaled everyone): **8** — `1f6dbe0` (comments), `63c5d88`
  (docs/provenance), `b9b215d` + `199d4da` (decode *parallelism thresholds*), `2e91607`
  (drift), `9624dd9` (aikit *version string* in the hash), `96269ca` (int4-head capability;
  default stays int8 for validated safetensors families), `7b01208` (pager `unsafe.Pointer`
  type change inside `mlp.go`).

(a) is **~12%**; (c) **dominates**. Per the standing decision rule, a high-(c) fraction points
at hash *scope*, not seams.

### 3. Verdict: don't build C
C addresses **2 of 17** while *adding* the hazard the design section itself names: a tap
surface exists so changes "here" don't re-stale families, which makes it by construction a
channel through which a genuine numeric change could enter **without re-staling** — the same
failure as refreshing a hash without goldens, but automated and invisible. Constraint #4
(provably-free hot-path indirection) is a real, permanent cost weighed against a **12% lever**.
Not worth it.

### 4. Scope narrowing is **not** the obviously-safe alternative
The high-(c) fraction tempts "narrow the hashed file set / hash a declared numeric surface."
Recorded reasons it is *not* obviously safe — it inherits the **same four constraints as C**:
- **Parallelism thresholds can change numerics.** `199d4da` was byte-identical only because
  those kernels reduce **order-stably** — a *kernel* property, not a *threshold* property. A
  hash that excluded thresholds would silently miss a threshold change that reorders a
  non-stable reduction. The whole-file hash catches it today precisely because it's coarse.
- **The aikit version string is a coarse proxy** for dependency numerics. Removing it from the
  hash loses that detection entirely unless replaced by a real content hash of the aikit
  surface — which is strictly more work, not less.
- **`7b01208` is plumbing inside a numerically-relevant file** (`mlp.go`). File-granularity
  narrowing cannot reach it; you'd need sub-file (symbol/AST) granularity, and then the
  symbol-selection *is* a declared surface — a channel for silent under-staling, same hazard
  class as C.

"Hash a declared numeric surface" makes the **declaration** the channel. Same four constraints.

### 5. Close
**Keep paying the (small, ~2-seam) tax.** Fix the counter (split/rescope the trailer + count
the real diff). Treat scope narrowing as **scoped but unfunded**, pending the corrected
trigger. Reassess only if the corrected *seam* count crosses a real recurrence bar — not the
inflated trailer count that fired this one.

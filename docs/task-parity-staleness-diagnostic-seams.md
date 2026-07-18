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
- **Phase 1 — D, small (do next time it trips, or now).** Write
  `scripts/refresh_parity_hashes.sh`: goldens-green gate → `go test ./decoder -run
  ParityManifest -update` → record the proof. One script; no code-path change.
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

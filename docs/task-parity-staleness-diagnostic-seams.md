# Plan: stop diagnostic seams from tripping the parity staleness gate

> **Audience:** implementation plan under `parity-coverage-policy.md` (§"A
> `deps_hash` refresh is not a re-validation"). One recurring friction point, three
> ranked fixes, and the one invariant any fix must preserve. Low priority — the
> current workaround is correct, just manual and slightly risky. First triggered
> 2026-07-17 (6b88e9e).

## The problem (one paragraph)

The staleness gate (`TestParityManifest_fresh`) hashes the **whole-file bytes** of
each family's dependency set (`freshDepsHash`, `decoder/parity_manifest_test.go`).
The `core` set includes `decoder/model.go` and `decoder/kvcache.go` — the forward
loop and the KV cache. But those same files also carry **diagnostic seams**:
`ForwardCapture`/`ForwardSubCapture`, `LayerKVForTest`, the `subCapture` struct
fields, and the inline `if cache.subCapture { … }` hooks in the forward loop. Every
one of those is provably non-numeric (guarded, default-off, byte-identical forward
output — the goldens prove it), yet adding or editing one changes the file bytes,
so **every validated family that `uses: core` goes stale at once**. The Gemma hunt
added four such seams and tripped the gate on all ~20 validated families (reported
as the first, `deepseek_v2`).

## Why the current workaround is imperfect

The policy's exception (b) — "provably non-numeric" — lets us `-update` the hashes
instead of re-validating the whole checkpoint matrix (which would be absurd for a
test-only seam). That is correct, but it has two costs:

1. **Friction.** Every future diagnostic-seam edit repeats the dance: refresh, prove
   non-numeric, justify in the commit.
2. **Integrity risk.** "Refresh `deps_hash`, keep `validated_at`" is *also* the exact
   shape of the forbidden gate-silencing move. The only thing separating a legitimate
   refresh from silencing a real numeric regression is human judgement + "the goldens
   are green." That judgement is unreviewable after the fact — a future refresh could
   wave through a genuine change under the same justification.

## Fixes, ranked by leverage

### 1 — Marker-region carve-out (recommended)

Teach `freshDepsHash` to strip lines between `//parity:ignore-start` and
`//parity:ignore-end` before hashing. Wrap the diagnostic seams (the struct fields,
the inline hooks, and — if not moved per #2 — the methods) in those markers. Then a
seam edit **cannot** change the hash, so the gate stops false-positiving, and the
carve-out is **greppable and reviewable** (`git grep parity:ignore`) — unlike a bare
`-update`, a reviewer can see exactly what was excluded and object if numeric code
sneaks inside a marker. Preserves the gate's integrity while removing the friction.
Small change to one function + a test that a marked region genuinely doesn't affect
the hash.

### 2 — Move what can move out of `core`

`ForwardCapture`/`ForwardSubCapture`/`LayerKVForTest` are methods; relocate them to a
new `decoder/capture.go` that appears in **no** family's dep set. That removes the
method *bodies* from the hash for free. It does **not** solve the problem alone — the
struct fields (must stay in the `KVCache` declaration) and the inline forward-loop
hooks (must stay in `model.go`) remain in `core` — so pair it with #1 for the
residual. Cheap, and it shrinks the marked surface #1 has to cover.

### 3 — Symbol-level hash (most principled, most work)

Hash the AST of only the numeric-relevant declarations, skipping test-only exported
seams entirely. Correct in principle but heavy: needs a Go parser pass, a rule for
"numeric-relevant," and it makes the hash logic itself something that can be wrong.
Not worth it for the volume of churn we actually have.

## The invariant any fix must preserve

Whatever changes, the gate must **still catch a real numeric change to the forward**.
The carve-out (#1) must cover *only* the guarded diagnostic code, never a line the
forward math depends on — enforced by review of the `parity:ignore` regions and by
keeping the per-family **goldens** as the independent numeric ground truth. The
staleness gate is a cheap proxy; the goldens are the real proof. Do not let a
convenience fix erode either.

## Scope

Low priority. Do #1 (+ optionally #2) the next time a diagnostic seam is added to a
`core` file and the friction actually recurs; until then the documented
refresh-with-goldens-green workaround holds.

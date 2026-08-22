# aikit task: fix `MatmulBTQ8`'s stale docstring (says scalar, code is SIMD since P2)

> **STATUS: OBSOLETE — the docstring was fixed upstream.** Verified 2026-08-21 against the aikit
> version goinfer requires (v1.21.0): `linalg/matmul_blocked_q8.go` no longer describes a scalar
> widen. Its one remaining "scalar" mention (line 150) is an accurate reference to the N%8≠0
> correctness tail, not the stale claim. Nothing to send.


## Context

Not a perf task — the perf work here is already DONE. This is a one-comment doc fix,
found while cross-checking a 9-item cross-repo performance audit against the actual
current code (goinfer session, 2026-08-19): item #2 of that audit claimed "scalar
int8→f32 widen on the LM head every token... a bit-identical SIMD widen exists in the
same package" as a still-open finding. It isn't open. `2f0c65f` ("perf(linalg): SIMD
widen in q8Span — ~2× faster q8 LM head, bit-identical (P2)") already fixed exactly
this, months before the audit ran. `q8Span` (`linalg/quant.go`) calls
`dequantRowInt8` — the AVX2/NEON SIMD widen — not a scalar loop.

The audit's false-positive traces to a real bug in this file, though: `MatmulBTQ8`'s
own doc comment was never updated after `2f0c65f` landed, and it still describes the
OLD scalar behavior. An automated cross-repo scan (or a future person) reading only
the doc comment — not `q8Span`'s own, correct, up-to-date comment three lines below
it — will conclude the widen is still scalar and re-file this as a finding. That is
what happened here.

## The fix

`linalg/quant.go`, `MatmulBTQ8`'s doc comment currently reads (paraphrased from the
line that's wrong): *"Only the cheap int8→f32 widen stays scalar; the
multiply-accumulate is vectorized."* That sentence is stale. `q8Span`'s own comment,
correctly, already says: *"SIMD int8→f32 widen (was a scalar convert loop the
compiler does not vectorize; at M=1 on the LM head it is ~68% of this function — P2).
BIT-IDENTICAL by construction..."*

Update `MatmulBTQ8`'s comment to match reality — the widen is SIMD via
`dequantRowInt8`, not scalar; only the row-scale multiply-and-write-back stays
scalar (that's the actual remaining scalar part, if the sentence is worth keeping in
some form rather than just removing the claim). Point at `2f0c65f` / P2 for anyone
who lands on the wrapper's comment first, the way `q8Span`'s own comment already
does, so the two comments agree instead of one silently outdating the other.

## Scope

Doc-only. No behavior change, no test change expected (the SIMD widen's correctness
is already covered by `TestQ8Span_bitIdenticalToScalarWiden` per `2f0c65f`'s own
commit message — confirm it's still green, don't re-derive it).

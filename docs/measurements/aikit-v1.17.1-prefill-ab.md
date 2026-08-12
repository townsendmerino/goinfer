# A/B pre-registration — aikit v1.16.0 vs v1.17.1, PREFILL

**Written before any prefill sample exists.** No prefill measurement has been taken on this box for
either version.

## Why this one is the critical path

It gates two things at once, and the second is the expensive one:

1. **A goinfer tag cannot honestly be cut without it.** `linalg/matmul_blocked.go` is unchanged
   between v1.17.0 and v1.17.1, so v1.17.0's f32 blocked-matmul rework — `dot8ColsInto` replacing
   `Dot8x4`'s 32-partial round trip, plus `blockRows3x4` ahead of the single-row path on amd64 — is
   **live in both versions and measured in neither.** Decode is measured and flat; prefill is the
   phase that rework actually lives on. A release characterizing one phase while silently carrying
   an unmeasured change to another is a claim by omission.
2. **The next tag fires C3**, the Metal consumer window — the largest completely uncovered surface
   in the project, which has already **sunk once**. C3's trigger is any release tag carrying an aikit
   bump, and this bump qualifies.

So a single unmeasured phase is holding both a tag and the largest uncovered surface. That is what
makes this the critical path rather than a nice-to-have.

## Harness — the same discipline as the decode A/B, for the same reason

Interleaved **a/b/a/b in one session on one box**. Two worktrees at the same goinfer commit with only
`go.mod` differing, so the goinfer source is identical between arms and the aikit version is the only
variable. `GOWORK=off`, no `replace`, aikit from the module cache.

**Comparison: v1.16.0 against v1.17.1**, because `matmul_blocked.go` is identical between v1.17.0 and
v1.17.1 — the rework is live in both, so v1.16.0 is the only baseline that predates it.

**Warm-up discard: the first sample of each visit**, carried over from the recorded calibration
rather than re-derived.

**Floor: 2.0%** of the pooled mean, pre-registered here, before the first sample.

**Statistic: median of the retained samples per arm.**

**The 5% session-drift figure applies here too** (`decoder/decode_bench_test.go`): a sequential
before/after would be dominated by it. Interleaving is not optional.

## Branches, fixed now — including the flat one

1. **Within ±2.0%** → **flat.** The f32 blocked-matmul rework is **below this instrument's noise
   floor on prefill**. That is the recorded answer and it discharges the obligation — it is a
   result, not an absence of one, and it does *not* become "prefill is unmeasured" in the release
   notes.
2. **v1.17.1 slower by ≥2.0%** → a prefill regression carried by the rework, still live upstream.
   Reported upstream with the same discipline as the decode one: direction, magnitude, method, and
   an explicit statement of what was not isolated.
3. **v1.17.1 faster by ≥2.0%** → the rework does what it was aimed at. **Scoped exactly as a loss
   would be** — one model, one prompt length, one box, prefill only — and it does **not** enter
   CHANGELOG, docs or release notes on this evidence alone.

## Known weakness, stated in advance

`BenchmarkDecode` is a decode harness; prefill needs a batched-forward measurement, so the instrument
is **not** the one already calibrated. The 2.0% floor is inherited from the decode calibration and is
therefore an *assumption* here, not a measurement. **If the prefill instrument's own spread turns out
to exceed 2.0%, the floor is wrong and the honest move is to re-derive it from a characterization run
and say so** — not to keep the inherited floor because it was written down first.

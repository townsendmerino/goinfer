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
2. **A tag numbered v0.13.0 or above fires C3**, the Metal consumer window — the largest completely
   uncovered surface in the project, which has already **sunk once**.

**CONDITIONAL, and I stated it too strongly first.** C3's trigger is *"the next goinfer RELEASE TAG
(≥ v0.13.0) that carries an AIKIT BUMP"* — the version floor is part of it. A **v0.12.1** patch
carrying this bump would **not** fire C3, and would not owe E1's parity backfill either (E1 reserves
that for v0.13.0). So what P10 gates depends on a numbering decision nobody has taken yet:

| next tag | P10 owed? | C3 fires? | E1 backfill owed? |
|---|---|---|---|
| `v0.12.1` (patch) | **yes** — the f32 rework ships either way | no | no |
| `v0.13.0` | **yes** | **yes** | **yes** |

P10 is owed under both, which is why it is the critical path regardless. The rest is a decision.

## STEP 0 — the instrument must first be shown CAPABLE, then CHARACTERIZED

Two prerequisites, both before the first comparison sample, and the first was nearly missed.

**0a. Capability. The obvious instrument cannot see the change.** `BenchmarkPrefillLong` calls
`loadBenchModel()`, which hardcodes `Options{Quant: "int8int8"}`. Traced through goinfer:

    int8 weights -> matmulInto -> linalg.MatmulBTW8A8Into        (never reaches blockedFill)
    f32  weights -> matmulInto -> matmul -> linalg.MatmulBT -> blockedFill   <- the reworked code

So an int8int8 prefill run exercises the f32 blocked matmul **not at all**, and would have produced
a confident flat result measuring nothing. That is exercised-but-never-triggered, inside the
pre-registration written to prevent it. The instrument therefore needs:

- **f32 weights** (not int8int8, not int4) — otherwise `blockedFill` is never entered;
- **a `canBatchN` architecture** — `prefillLogits` routes MoE and Gemma4 down the sequential
  per-token path, so DeepSeek-V2-Lite (the decode A/B's model) would run at M=1 and miss the batched
  shape the rework targets. A dense model (e.g. Qwen2.5-Coder-1.5B) is required, and
  `b.Logf("canBatchN(%d)=%v")` must be **asserted true in the run log**, not assumed.

**Capability is proven before calibration**: one run of each arm, confirming `canBatchN=true` and a
non-trivial wall time at f32. If the instrument cannot be made to enter `blockedFill`, the honest
outcome is "no instrument exists for this yet", not a flat result.

**0b. Characterization, as its own artifact.** A calibration pass on the prefill instrument —
**same shape as the 8-sample decode one**: repeated samples in one session on one arm, recording the
warm-up form (is the first sample cold, and by how much), the discard count that follows from it, and
the spread (sd, relative sd, min/max).

**This is recorded as instrument calibration in its own right, before any comparison sample exists.**
The 2.0% floor is inherited from the *decode* instrument and is an assumption here. Either it
survives contact with the prefill characterization, or a measured floor replaces it — and **either
way the number predates the first comparison sample.**

**Why this cannot be folded into the A/B run.** Deriving the floor from the comparison data is
deriving the threshold from the data it judges. Separating the 8-sample pass is exactly what kept the
decode floor honest, and "re-derive the floor if the spread exceeds it" quietly gives that back if
the re-derivation reads the A/B's own samples. Characterize first, fix the floor, then compare.

## Harness — the same discipline as the decode A/B, for the same reason

Interleaved **a/b/a/b in one session on one box**. Two worktrees at the same goinfer commit with only
`go.mod` differing, so the goinfer source is identical between arms and the aikit version is the only
variable. `GOWORK=off`, no `replace`, aikit from the module cache.

**Comparison: v1.16.0 against v1.17.1**, because `matmul_blocked.go` is identical between v1.17.0 and
v1.17.1 — the rework is live in both, so v1.16.0 is the only baseline that predates it.

**Warm-up discard: from the STEP 0b characterization**, not carried over from the decode one — a
prefill run is a different shape (one long batched forward per iteration rather than many small
ones) and there is no reason its warm-up form should match.

**Floor: 2.0% provisionally, confirmed or replaced by 0b before the first comparison sample.**

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

The floor is inherited until 0b runs. That is now a **sequenced prerequisite** rather than a caveat:
the earlier wording ("re-derive the floor if the spread exceeds it") was right in intent and wrong in
mechanism, because a re-derivation performed on the A/B's own samples derives the threshold from the
data it judges. 0b fixes the number first, from separate data.

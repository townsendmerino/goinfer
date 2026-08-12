# Task (SPEC — implementation deferred to the v1.0.1 batch): Metal runtime self-test at backend init

> **Status:** design only. No code lands from this doc before v1.0 (the release plan requires zero
> code delta on `main` between the v0.11.0 and v1.0 tags). Implementation queues to **v1.0.1**,
> alongside the Metal dispatch removals (audit #4/#5). Motivated by dmikey's A1111 Metal write-up
> (the field-codegen-drift risk) — the same source as the commit/sync census (ollama-chase §A2-Metal).

## The gap this closes

goinfer compiles its MSL **on the user's machine at load** (cgo-free, `gpu.CompileLibrary` at
`languageVersion ≥ 3.1`). The one absolute Metal correctness gate is `TestMetalSnapshotGolden`
(`metal/snapshot_golden_test.go`) — a byte-compare of resident-path logits against a committed
`sha256` golden. That golden is **machine-pinned by construction**: it was produced on a specific
GPU + macOS + toolchain, and byte-identity is only expected on that combination.

So a macOS update that shifts the Metal compiler's codegen — a different fast-math contraction, a
changed reduction order, an FP16 intermediate where there was FP32 — is caught **on the dev box**
(the golden goes red) and by **nothing in the field**. The snapshot golden cannot ship as a runtime
check: a byte-compare on arbitrary hardware would false-positive on every machine that isn't the one
that pinned it. We have an absolute gate that is structurally undeployable, and a field failure mode
it cannot see.

## The design constraint that resolves the obvious objection

*"Ship the golden and compare at startup"* does not work — see above; hardware-legitimate FP
differences are indistinguishable from a codegen regression under byte-compare. The correct shape is
**not** absolute-vs-stored; it is **relative, at init: Metal vs the pure-Go CPU reference**, on a
tiny fixed input, judged with the **parity-gate tolerances** (`cosMaxAbs`: cosine + max-abs), which
are hardware-agnostic by construction. The CPU path is already in the binary (it is the decline
fallback), it is deterministic, and it is the same reference every T-tier Metal parity test uses. A
tolerance-based Metal-vs-CPU check is legitimate on *any* GPU; a byte-check is legitimate on exactly
one.

This is the mirror of the snapshot-golden reasoning in ollama-chase §A2-Metal: an *absolute* stored
reference is the only thing that catches "both arms moved together" on the dev box; a *relative*
CPU-reference check is the only thing that ships to the field. They are complementary, not
redundant — keep both.

## Spec

### Probe contents

One fixed, tiny, deterministic vector pass through the **numerically riskiest kernels** — the ones
whose output is codegen-sensitive and whose breakage is silent:

- **Attention** — the softmax reduction (reduction *width* is part of the bit-identity contract; a
  256-wide vs 128-wide tree sums the denominator in a different order) and the FP16-scale MMA seam
  (the root of the Metal MMA NaN and the batched-prefill divergence). Drive it past **both reduction
  widths** (≥ 256 keys) so a width-coupled codegen change is in range.
- **The norm sandwich** — `rmsnorm_quant` (the norm→int8-quant seam) and, on Gemma-admitting builds,
  the GELU-tanh path (the tanh-overflow-to-NaN site clamped in `glu_act`). These are the two spots
  where a changed intermediate precision has flipped output to NaN before.

Input is a committed fixed vector (a few KB, deterministic — same discipline as the tiny goldens),
not a model asset, so the probe has no external dependency and runs before any checkpoint is loaded.
The probe reuses the resident kernels themselves, not reimplementations, so it exercises exactly what
the field will run.

### Tolerance choice and why

Use the **existing parity-gate tolerances**, not new ones: **cosine ≥ 0.999** and a max-abs bound in
the band the Metal-vs-CPU gates already accept (the FP16-scale gap puts legitimate divergence at
~1e-2 on activations, ~1e-6 on reduction order — see `gemma4_router_parity_test.go`). Reusing the
shipped thresholds means the probe accepts exactly what the parity suite already certified as
correct and rejects exactly what it would reject — no second, driftable tolerance to maintain. The
probe must **not** invent a tighter bound (it would decline healthy machines) or a looser one (it
would pass a real codegen regression) — it inherits the gate's calibration.

### Startup cost budget

**~ms, hard.** One tiny fixed-vector pass through a handful of kernels is a single small command
buffer (one commit + one sync — the census confirms that is the floor). Budget: **< 5 ms** added to
backend init, measured. If it cannot be met, the probe is too big — shrink the vector, not the
tolerance. It runs **once**, at `BuildResident`, never per token.

### What "decline" looks like

Reuse the existing decline path (`metal/backend.go`: a failed resident build already recovers →
`ok=false` → falls back to the shared-SIMD CPU kernels, printing `[metal] declined — …`). On probe
mismatch:

- **Name what mismatched** — which kernel, cosine/max-abs observed vs threshold — in the warning, so
  a field report is actionable ("`rmsnorm_quant` cos 0.981 < 0.999 on macOS 26.x"), not just
  "Metal declined".
- **`serve`** — log the decline at WARN to stderr/structured log, continue on CPU, and surface it in
  the health/introspection surface (a served process should be queryable for "why am I on CPU?").
  Never crash a long-running server over a startup probe.
- **`chat`** — a single human-readable stderr line ("Metal self-test failed (<kernel>, <numbers>);
  running on CPU") and proceed. Decline-never-crash, field edition — the same ethos as the
  batched-prefill and BuildResident declines.

### The test that proves the probe can fail (the vacuous-gate rule)

Per `parity-coverage-policy.md` §"Falsifiable: prove the gate red before trusting it green" — a gate
is **vacuous until seen to fail on a real defect**. The probe ships with a **break-it-first** test:
perturb a shader (e.g. change the softmax reduction width, or drop the GELU-tanh clamp) and assert
the probe **declines** — and, restored, that it **passes**. Without this the probe could be a
tolerance so loose it accepts anything, which reads as "Metal verified" while testing nothing — the
exact failure this whole class is about.

## Open questions (for whoever implements in v1.0.1)

1. **Probe granularity vs the decline unit.** If only the Gemma GELU path mismatches, should the
   backend decline *entirely* to CPU, or decline *only Gemma families* and keep Metal for dense
   archs? Per-family decline is more surgical but multiplies the probe's surface. Default proposal:
   whole-backend decline (simple, safe), revisit if it proves over-broad in the field.
2. **Fixed vector provenance.** Generate the committed input from a seeded RNG in a `pin_*.py`
   (auditable, regenerable) vs hand-pick — lean to seeded, matching the tiny-golden generators.
3. **Interaction with the snapshot golden.** The probe and the dev-box golden should share the fixed
   input so a field decline and a dev-box red point at the same computation. Confirm the snapshot
   golden's sequence can double as the probe vector without weakening either.
4. **Cost of the CPU reference at init.** The CPU pass over the tiny vector must also be ~ms; confirm
   the shared-SIMD kernels don't pull in a heavy one-time init (quant tables) that blows the budget.

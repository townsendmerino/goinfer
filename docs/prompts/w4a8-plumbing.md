# Task (aikit + goinfer): W4A8 plumbing — ship the harness-winning layout to production CPU decode

> **For:** Claude Code, in `~/tmcode/aikit` + `~/tmcode/goinfer`. Written 2026-08-24, after the
> item-3+4 harness phase recorded GO. Read `docs/task-w4a8-neon-bandwidth.md` (goinfer) first —
> Gate 0, both probes, the items-1+2 negative, the harness grid, and the GO line are all there.
> **The layout is settled and is not this phase's to redesign: split-half nibbles (item 3) +
> 4-row interleave (item 4), signed centering, one accumulator per real row** — single-call cold
> 42.4-42.7 GMAC/s, 6-worker aggregate 1.40-1.41x over baseline, reproduced. This phase wires it
> in: aikit production kernel entry point, goinfer arm64 load-time repack, dispatch, gates,
> end-to-end numbers, release. The `.giw` format kind stays parked per its own sequencing rule.

## Step 0 — pin the inputs

1. **Commit the harness artifacts in aikit** (winner kernel, repack function, oracles, tests —
   still unwired). Everything below references committed code, not a working tree.
2. **Quiet-box `bench_peer` baseline cells, if not already recorded in the campaign doc**
   (the harness brief's Step 0 — verify whether it ran before the pivot consumed the session):
   0.5B + 1.5B, depth 128, `-quant int4` AND `-quant int8int8`, 2+ runs, idle box, tagged
   versions. These are the *before* cells every end-to-end claim below compares against.

## The pieces, in order

**1. aikit: production entry point.** Promote the winner to a real exported API (naming per
`linalg` conventions — the M>1 story decides whether it's a new `MatmulBTW4A8x4Into` beside the
existing entry or a layout-tagged variant). Wire the parallel fan-out the same way
`MatmulBTW4A8Into` fans out today; the 6-worker profile is already measured. Additive only —
no existing signature changes.

**2. goinfer: arm64-only load-time repack.** The GPU-precedent pattern from the campaign doc's
Gate-1 correction: on the CPU load path, `GOARCH == arm64` only, repack canonical → split-half
4-row-interleaved into a second byte array the new kernel consumes. The canonical packer,
`.giw` reader, scalar oracle, and amd64 are untouched. Record the two costs the format
follow-on says this phase must measure: **load-time delta** (0.5B and 1.5B, cold and warm —
the prequant fast-load win is the thing being spent) and **resident-memory delta** (the
repacked copy is heap, not page-cache-backed mmap; decide and document whether the canonical
mmap alias is dropped after repack or kept). These two numbers are what the parked `.giw`-kind
decision will eventually be made from — measure them like they matter.

**3. Dispatch, with two mandatory carve-outs.**
- **Paged-MoE tensors keep the old kernel.** `expertPager` reads experts directly off the
  read-only `.giw` mapping in canonical layout — there is no load-time repack for paged
  tensors by construction. Dispatch on the tensor's actual layout, never on architecture alone.
- **Row-count and group-count tails.** The 4-row interleave needs an answer for rows % 4 ≠ 0
  and every nGroups residue — whatever the harness layout chose (tail rows via the old kernel,
  or padded repack), test every residue explicitly.

**4. Numerics — this is the phase's real risk section, read it twice.** The new kernel is
rel-err-clean against the scalar oracle but is NOT bit-identical to the old kernel: split-half
consumes groups j and j+16 from one load and the 2-lane accumulation combines at the end, so
per-output summation order changed. Consequences to handle deliberately, not discover:
- **decode == prefill == verify must still hold.** Speculative decoding requires the M=1 and
  M=K paths to produce identical logits. That is preserved iff the new kernel's per-output
  reduction order is **M-independent** — same weight-group order for output (i,j) no matter
  how many activation rows ride along. Verify this property in the kernel by construction,
  then prove it with `TestForwardN_matchesSequential` and `TestSpeculativeGreedyParity` run
  with the new dispatch active. If the kernel is M=1-only and forwardN takes a different path,
  that identity breaks — not acceptable; make both route through the same per-output order.
- **Committed goldens.** Enumerate which CPU W4A8 parity gates compare exact vs tolerance
  before wiring anything. Exact goldens will shift — that's the expected cost, handled the
  cosine-re-gate-plus-regold way (the quant-gate shape the f16-scales task and the format
  follow-on both describe), regenerated once, with the change and its justification recorded.
  A silent golden update with no cosine check is not the house pattern; a failing gate
  "fixed" by regolding without the quality check is worse.
- **Quality re-gate.** The W4A8 cosine/quality sweep re-run once with the new kernel wired —
  reordered f32 accumulation should be quality-neutral, but "should" is measured here, per
  this campaign's record on plausible-sounding claims.

**5. End-to-end, against Step 0's cells.** `bench_peer`, quiet box, tagged state: 0.5B + 1.5B
int4, depth 128. Also record the real-decode aggregate rate (bytes/token ÷ matmul-stub time,
the probe-2 method with the corrected 1.05 GB/token accounting) so the real-vs-isolated
parallel-efficiency ratio (~60% last measured) gets a fresh data point — if it moved, that's
the next campaign's opening fact; the harness brief's 196-matrix rotation cell remains the
cheap diagnostic if it moved the wrong way.

**Projection to hold the result against** (Amdahl, from the harness aggregate — state it in
the doc before measuring): W4A8 is ~78% of matmul bytes (the int8-pinned LM head is
untouched); 43.6 ms × (0.78/1.40 + 0.22) ≈ 34 ms matmul + ~2.8 attention + ~0.4 other ≈ 37
ms/token → **~27 tok/s at 1.5B (~0.40x ollama)**, with upside toward ~31 if the isolated
parallel efficiency carries over better than the historical 60%. If the measurement lands well
under the projection band, the real-vs-isolated gap is where to look first — do not silently
accept a shortfall the way the projection method (validated to 0.03 tok/s in A1) says
shouldn't exist.

**6. Release, last.** aikit bump (v1.26.0) through the full `RELEASING.md` ritual, gpu pins,
goinfer onto the real tag, override removed, T3 re-run, manifest refresh — the same chain the
A1 release already walked. Final published numbers measured against tags on a quiet box; the
A1 close-out's load-caveat lesson applies verbatim.

## Acceptance

- 1.5B int4 end-to-end within or above the stated projection band (~27 tok/s), against Step
  0's own quiet-box baseline — plus the 0.5B cell recorded.
- Spec/forwardN identity gates green with the new dispatch active; golden shifts (if any)
  regolded once through the cosine re-gate with the record written.
- Load-time and resident-memory deltas measured and recorded per model size — these feed the
  parked `.giw` decision.
- Paged-MoE fallback proven by test, not asserted.
- All results appended to `docs/task-w4a8-neon-bandwidth.md`; a negative or shortfall recorded
  at the campaign's usual standard is a complete deliverable.

## Not in scope

- The `.giw` format kind — parked; this phase only produces the two numbers its decision needs.
- The uncentered-Σact retry (deferred, kernel-only, no weight bytes — its own small task if
  ever), i8mm/M2+, VNNI/amd64, f16 scales, 3-bit, the LM-head int8-pinning question.
- Prefill-side W4A8 work beyond what M-independence requires.
- Re-tuning attention, thresholds, or anything the closed campaigns own.

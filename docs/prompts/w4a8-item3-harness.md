# Task (aikit + goinfer): W4A8 item 3+4 — the repacked-layout kernel, harness phase only

> **For:** Claude Code, in `~/tmcode/aikit` (kernel + harness work lands there) with sibling
> `~/tmcode/goinfer` (baseline cells + doc updates). Written 2026-08-23, after the attention
> campaign (A1) closed. Read `docs/task-w4a8-neon-bandwidth.md` (goinfer) first — it carries
> Gate 0, the items-1+2 negative, both probe results, and the format follow-on's sequencing
> rule. **This brief funds the HARNESS phase only: repacked weights exist in benchmark code,
> kernels are real but unwired, and the deliverable is a measured go/no-go on funding the
> production plumbing.** No `WeightMat` changes, no `.giw` changes, no dispatch wiring, no
> touching the canonical packer in `quant.go` — the campaign doc's Gate-1 correction explains
> why the canonical layout is load-bearing and must not move.

## Why this is the next unit of work (updated context since the doc was written)

A1 cut decode attention to ~2.8 ms/token, so the 1.5B decode profile is now **~93% weight
matmul** (43.6 ms of ~46.4 ms). The W4A8 doc's bandwidth-class scenario now projects ~64 tok/s
(~0.93x ollama) instead of the 0.84x it was scoped under. The one measured negative so far
(items 1+2, decoupled Σact pass, 0.972x) explicitly does not touch this work — and Gate 0's
issue-limited finding (probe ratio 1.11) is direct evidence *for* an instruction-removing
repack. The unpack tax to the no-unpack reference is measured: `dotW4A8FoldSDOT` 24.62 GMAC/s
cold vs `dotI8SDOT` 49.23 — **1.97x on the table**.

## Step 0 — quiet-box baseline cells (required before any kernel work)

On an otherwise-idle box (no other sessions, no VS Code — the campaign has two recorded
incidents of load-contaminated numbers), `bench_peer` method, 2+ runs each, all against the
tagged versions now on main (aikit v1.25.0):

- 0.5B and 1.5B, depth 128, `-quant int4` — these become this campaign's *before* cells, and
  they certify (or correct) the A1 close-out's load-caveated 40.71/21.52 tok/s figures.
- The same two cells at `-quant int8int8` — the post-A1 projection is ~40 tok/s at 1.5B
  (~0.59x ollama); a confirmed number here upgrades the doc's zero-cost-guidance item from
  projection to measurement.

Record all of it in `docs/task-w4a8-neon-bandwidth.md` before proceeding.

## The design work — one layout decision, explored as a small measured grid

The harness repacks canonical-format weights into candidate layouts in benchmark code only
(repack function + a repack→unpack roundtrip test against the canonical layout). Kernels are
written against the candidates and measured cold and hot. The grid — keep it to these axes,
every cell is cheap once the harness exists:

1. **Split-half nibble layout** (item 3's core): one 16-byte load feeds two dot products —
   low nibbles are block j, high nibbles are block j+16 — one AND + one SHR total, both
   results consumed. This attacks the ~57% unpack share directly.
2. **× row interleave: 1-row vs 4-row** (item 4): the 4-row variant computes 4 output rows
   per pass, reusing activation registers and keeping the DotProd pipes fed (ggml's
   Q4_0_4x4-style repack — a large measured win for llama.cpp on this hardware class).
3. **× centering: signed (current convention) vs unsigned + Σact correction** — this is the
   items-1+2 *sanctioned retry*: the negative closed only the decoupled-pass shape; inside a
   repacked kernel whose unpack prologue has shrunk, the correction may fit in the freed
   issue budget. Σact comes precomputed per group (the harness can just compute it; the
   `MatmulBTW4A8Into` calling-convention question is a plumbing-phase concern, not yours).

Not in the grid: i8mm/SMMLA (M1 baseline is DotProd-only — follow-on), VNNI/amd64 (no
hardware), any scale-format change (f16 scales belong to the format follow-on).

Correctness bar per variant: rel-err ≤1e-3 vs `dotW4A8Scalar` across random + stress shapes
and every nGroups residue of the interleave width — the same bar the items-1+2 kernel met.
Where a variant changes the algebra (unsigned+Σact), prove the rearrangement exact against
the oracle first, the way `dotW4A8UncenteredScalar` did.

## Measurement protocol

- Quiet box, order-alternated best-of-3, cold AND hot, per the Gate 0 ops-per-byte method
  (`w4a8_opsperbyte_bench_arm64_test.go` is the shape to extend).
- Real shapes from the 1.5B GGUF metadata: gate/up [8960,1536], down [1536,8960], and one
  qkv shape — not the borrowed 27B shape (probe 2's lesson).
- **Report GMAC/s and GB/s together, with the bytes-per-MAC conversion stated explicitly**
  (~0.625 B/MAC for W4A8 nibbles+scales). Probe 2's "parallelization anomaly" was a
  byte-estimate artifact; don't create the conditions for another one.
- The winning variant additionally gets: (a) the 6-worker parallel measurement at the real
  shape through the same fan-out `MatmulBTW4A8Into` uses (current baseline: 40.58 GB/s at 6
  workers, 46% efficiency), and (b) a 196-distinct-matrix rotation cell, since probe 2 left
  the real-decode-vs-isolated 60% gap "real but partially unexplained" — one cell here either
  implicates the many-matrices pattern or rules it out for free.

## Go/no-go for funding the plumbing phase — named numbers, decided in the doc

**GO** if the winning variant, single-call cold at the real gate/up shape, reaches **≥40
GMAC/s** (≥1.6x over the production kernel's 24.62 — i.e. most of the 1.97x tax closed), AND
the 6-worker aggregate at the real shape beats the current 40.58 GB/s baseline by **≥1.4x**.
Project end-to-end tok/s via Amdahl from the measured aggregate (matmul bytes ~1.05 GB/token
at 1.5B, attention ~2.8 ms, other ~0.4 ms) — the A1 campaign validated this projection method
to 0.03 tok/s, so state the projected number with confidence and let it carry the funding
argument.

**NO-GO** if the grid tops out below that: document the best variant, the mechanism that
capped it, and the projected end-to-end ceiling, per the perf-dead-ends convention — in which
case `-quant int8int8` guidance remains the shipped answer for Apple Silicon CPU decode and
the campaign records that as its outcome. A documented no-go here is a complete, successful
deliverable; do not stretch measurements to clear the bar.

What GO funds (NOT this brief's scope, listed so the boundary is unambiguous): the GPU-style
arm64 load-time repack in goinfer's `WeightMat`, per the campaign doc's Gate-1 correction;
dispatch wiring; parity gates; end-to-end `bench_peer`; and only after all of that, the `.giw`
format follow-on per its own sequencing rule.

## Disposition of artifacts

Winning kernel + repack function + tests: committed in aikit (unwired — additive, no dispatch
changes, no existing-API changes). Losing variants: results recorded in the campaign doc;
code deleted or parked per aikit's own convention for measured negatives. All numbers,
including Step 0's baselines, appended to `docs/task-w4a8-neon-bandwidth.md` in the style of
its existing result sections, ending with the explicit GO/NO-GO line and, if GO, a short list
of what the plumbing phase needs (calling-convention change for Σact if that variant won,
per-worker anything, etc.) so the next brief starts from a list rather than a re-read.

## Not in scope — do not drift into these

- Production wiring of any kind — harness only, this phase.
- The `.giw`/format work — parked behind its own sequencing rule in the campaign doc.
- i8mm, VNNI, amd64 ports, f16 scales, 3-bit anything.
- Re-litigating the decoupled Σact pass (closed) or the accumulator-chain fix (closed, amd64).
- Attention, sampler, norms — that campaign is closed (A2/A3 disposition recorded).
- The LM-head int8-pinning question (~0.23 GB/token) — real, separate, quality-gated; leave it
  to its own future scoping.

# Task (goinfer): 35B-A3B paged decode — diagnose the ~1160 ms/token, fix nothing

> **For:** Claude Code, in `~/tmcode/goinfer`, on the 16 GB M1 Pro. Written 2026-08-24, from
> Zeno Phase 0 Part A's close (`2dee492`): real Qwen3.5-35B-A3B, 22 GB `.giw`, expert
> demand-paging at ~2.4-2.7 GB resident, coherent decode at **~0.86 tok/s**. The same box
> pages gemma4 26B-A4B — smaller total, similar active — at **16.98 tok/s** (81.6% hit rate,
> `docs/completed/gemma4-resident-scope.md`). A ~20x gap that model size doesn't explain.
> **This task is diagnosis only.** The deliverable is a ranked cost table; fixes are the next
> campaign's business. Value is independent of Zeno — Qwen3.6-35B-A3B is a roadmap target
> (`docs/qwen3_5_moe.md`), and whatever this finds is that campaign's opening fact.

## Cheap reads first — before any stub work

1. **Pager statistics for the run itself**: hit rate, faults/token, bytes faulted/token,
   slot count — whatever `expertPager` already counts (gemma4's doc reported hit rate and
   slots, so the counters exist). Compare directly to gemma4's 81.6%.
2. **Warm vs cold**: a second run immediately after the first — the 22 GB file can't fully
   fit page cache on this box, but the delta bounds how much is I/O.
3. **Disk-read rate during decode** (Activity Monitor grade is fine): bytes/s ÷ tok/s =
   faulted bytes/token, cross-checking the pager's own numbers.
4. **Verify the LM head dispatch is the new W8A8 path** for this config — the
   dispatch-inertness lesson applies: the observable is the measured rate, not the code read.

State the hypothesis math before measuring (house discipline): ~3B active params ≈ ~1.5 GB
touched/token at int4 if every touched expert misses; at NVMe-class read rates a plausible
miss pattern costs low hundreds of ms/token — which does NOT reach 1160 ms on its own unless
the hit rate is very low. So expect at least one large non-I/O component.

## The stub split — fourth outing of the component method

Same env-gated-stub approach as the A1 and LM-head diagnostics, buckets chosen for this
family's structure:

- **Expert paging + expert matmuls** (the paged path runs the canonical W4A8 kernel by the
  plumbing carve-out — expected, not a finding on its own; separate I/O wait from compute if
  the pager's accounting allows).
- **The f32-scratch handicap — the one Phase 0 brief item still open** (`docs/queue-performance.md`
  ~line 97: this family runs f32 scratch regardless of `Options.Quant`, "parity-first...
  never revisited"). Size it at the real model's shapes. If expert weights are being widened
  to f32 per use, that alone could dominate — this is the lead suspect for the non-I/O
  component.
- **DeltaNet layers vs attention layers** — hybrid family; time them separately. The A1 wins
  apply only to the attention layers; DeltaNet's path has never been on any perf campaign.
- **LM head** (should be fast post-fix; verify, don't assume).
- **Norms/sampler/streaming residual** (expected noise; confirm).

Percentages must sum to ~100% of the measured ms/token — the A0 splits held to within 2%,
same bar here.

## Deliverable

A ranked cost table (bucket, ms/token, %, method) appended to the Phase 0 record in
`docs/task-zeno-compare.md`, with: the pager stats alongside gemma4's for comparison, the
f32-scratch item formally sized (closing the Phase 0 brief's last open item), and a closing
recommendation naming the top one or two levers with rough sizes — scoping input for the next
campaign, not work performed here. Quiet box; `b.Run` shapes, never nested
`testing.Benchmark`; order-alternated where applicable. A surprising answer (e.g. "it's
mostly I/O after all") recorded plainly is a complete deliverable.

## Not in scope

Fixing anything found; Zeno install or Part B (separate, gated on Francis); prefill; Phase 1
benchmarking; touching gemma4's paged path; any aikit changes.

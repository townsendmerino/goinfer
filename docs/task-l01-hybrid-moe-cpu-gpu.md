# Task: L-01 — hybrid CPU/GPU expert execution for the 8 GB CUDA MoE path (design pass, 2026-09-10)

> **Status: SCOPING ONLY, no code.** This is the "dedicated future pass" the audit
> (`docs/audit-2026-09-02.md` L-01) called for, done as a design pass per explicit instruction —
> size q⋆ from the already-captured trace, work out the split/merge design, re-check the
> speculation-antagonism risk, write it up, stop before any CUDA. **The headline finding
> reverses the audit's optimistic framing: at today's measured numbers, computing even one
> missed expert on the CPU (using the existing kernel as-is) is slower than this layer's own
> DMA-fetch-everything baseline. The naive form of L-01 is not obviously a win — it needs either
> a genuinely lower-latency CPU expert path (a bigger prerequisite than scoped) or a much
> narrower trigger condition than "route every miss to CPU."** This is not a rejection of L-01;
> it is why the audit said "dedicated future pass" rather than "next PR."

## 0. What L-01 is, verbatim from the audit

`docs/audit-2026-09-02.md` L-01: *"per layer, the m experts missing from the device cache are
split — q⋆ ≈ m·(B_PCIe/B_host) fetched into slots and run on the GPU, the rest computed on the
CPU from the pinned host copy with the existing W4A8/W8A8 NEON/AVX2 expert kernels, partial sums
merged exactly."* Rationale: `cuda/resident.go`'s C′ path already holds every expert pinned on
the host, already has the CPU expert kernels (P18 made them expert-major), and already pays a
host-visible routing readback per layer — so the architectural objection recorded in
`docs/task-freetoken-techniques.md` Lead 5 is smaller than when written. The lever: with misses
on CPU, token time becomes `max(GPU hits, CPU misses)` instead of `hits + DMA`, and G32 already
found the DMA bandwidth-saturated at line rate — exactly the case where moving work off the bus
beats overlapping it.

**Pre-registered decision rule** (unchanged, restated for this pass): Qwen3.6-35B-A3B int4 on
the 8 GB card, paired and interleaved against the shipped C′ path and against `off`; fund at
**≥1.3× end-to-end**, park below **1.15×**, 1.15–1.3× ambiguous → second mechanism. "The number
to beat is 39.3" — FreeToken's own published tok/s on an RTX 4060 Laptop (a different card, per
`docs/task-freetoken-techniques.md:7`), an external sanity check, not the primary bar.

## 1. What the trace already answers (G33, 2026-08-28 — no new hardware run needed)

`docs/measurements/g33-routing-trace.json` (2730 decisions = 30 MoE layers × 91 positions,
gemma-4-26b, 30 slots) + `scripts/g33_replay.py`, validation gate passed exactly (offline replay
reproduces the measured 16611/5229 hits/misses to the token).

- **In steady state, ~90% of misses are evictions (capacity), not cold/unpredictable.** This
  directly supports L-01's premise — the reason speculative *prefetch* was closed (G33 RESULT)
  is a DIFFERENT reason than why hybrid *compute* stays viable: prefetch fails because it can
  only attack the small cold floor; compute-on-CPU doesn't care whether a miss was predictable,
  it just needs the miss to happen.
- **m (missed experts per layer per token) ≈ 1.915 on average at 30 slots**
  (5229 misses / 2730 decisions, topK=8 requested per decision).
- **DMA cost is an exact law: 346 µs per miss**, cross-validated three ways in this pass (G31's
  original figure, G33's 19.9 ms/token at 30 slots = 1.915 × 30 layers × 0.346 ms, and G34's
  serial baseline 18.56 tok/s = 1/(32.1+1.9+19.9 ms)) — I did not re-measure this; I re-derived
  it from existing numbers and it lines up exactly, which is itself a useful cross-check that
  G31–G34's arithmetic is internally consistent.
- **"Perfect overlap" (hide DMA behind existing GPU compute, no CPU involved) is separately
  already closed as unattainable** (`docs/QUEUE.md` G32 section, "Perfect overlap is
  unattainable — routing for layer L+1 is not known until layer L computes"). This matters for
  scoping L-01: the simpler, cheaper lever is not sitting there unexploited waiting to be taken
  instead of L-01 — it hit a hard serial-dependency wall. L-01 is a genuinely different
  mechanism (WHO computes a miss, not WHEN it's known), so it isn't just reinventing overlap.

## 2. New measurement this pass: gemma-4-26B CPU-only decode rate, real hardware

Used `fit -measure` (`internal/fitcmd/fit.go`, shipped this session, Phase 4) exactly as built —
no new code for this. `~/models/gemma4-26b-int4.giw`, CPU-only build (`demo/chat`), nobara-pc
(16 cores, 62 GB RAM, model fully resident — no paging, unlike the Mac's catastrophic case for
this same model class):

```
(measure) cpu    2.1 tok/s over 32/32 decode steps (includes a 64-token prompt prefill)
```

**Derived B_host (rough, NOT an isolated microbenchmark):** 1/2.1 tok/s ≈ 476 ms/token total;
gemma-4-26b has 30 MoE layers × topK=8 = 240 expert-FFN computes per token. Dividing (ignoring
attention/router/shared-expert share, which is real but a minority of FLOPs for a MoE model)
gives **≈1.98 ms per expert, at full machine width** — i.e. this already reflects whatever
parallelism the CPU kernel gets, not a single-core number. **This estimate needs a real isolated
microbenchmark before it's trusted for a funding decision** (see §5) — it's good enough to
motivate or kill the naive design, not good enough to size a shipped q⋆.

## 3. The finding that reverses the framing: today's CPU expert path has no spare capacity to give

Read `decoder/mlp.go`'s `moeMLP` (the CPU MoE decode path) directly, not assumed:

- **The expert loop is sequential across experts, by design and by comment**: *"The experts run
  sequentially, so k pairs were never simultaneously live"* — the 2-buffer (not 2k-buffer)
  scratch reuse actively depends on this and would need restructuring (per-goroutine scratch) to
  parallelize across experts.
- **But each expert's OWN matmul already parallelizes across cores** — `swiGLUExpert` → `matmul`
  → aikit's `linalg.MatmulBT`, which calls `parallelCols(M*N*K, N, ...)`: a work-stealing pool
  across output columns, gated on a MAC-count threshold. At decode shape (M=1, gemma-4-26b's
  expert intermediate dim), this MAC count is well above any reasonable threshold, so one
  expert's own compute already claims most/all 16 cores.

**Consequence: there is little genuinely idle CPU capacity for a second, concurrently-computed
expert to exploit "for free."** Adding cross-expert parallelism (compute two missed experts on
different cores simultaneously) means dividing the SAME 16-core pool between them, not finding
spare cores — a throughput/latency trade that needs its own measurement, not an assumption.

**The per-LAYER arithmetic (the real constraint, since layers are strictly sequential — layer
L+1 needs layer L's fully-merged output) is unfavorable at today's numbers:**

| quantity | value | source |
|---|---|---|
| per-layer GPU compute (hits) | ≈1.07 ms | 32.1 ms ÷ 30 layers (G31) |
| per-layer DMA cost at m≈1.915 | ≈0.66 ms | 1.915 × 0.346 ms (G33) |
| **one CPU expert, full-width, sequential** | **≈1.98 ms** | this pass, §2 |

**One CPU-computed expert already costs more than that entire layer's current total (GPU
compute + DMA for ALL its misses) combined (1.07+0.66=1.73 ms < 1.98 ms).** At the AVERAGE miss
count (1.915), routing even a single miss to CPU — with the kernel as it exists today — likely
makes that layer slower, not faster, because CPU compute doesn't overlap with same-layer GPU
compute the way the mechanism's name suggests: the layer's output needs BOTH paths done and
merged before the next layer can start, and 1.98 ms already exceeds what GPU alone was going to
cost.

**This is a rough estimate from a derived B_host, not a rejection — it is the reason to measure
before building**, exactly the discipline `CLAUDE.md`'s "measurement discipline" section asks
for ("A kernel win is not an end-to-end win until Amdahl has been paid" — here, the opposite
risk: a plausible-sounding mechanism that a derived estimate says may not even be a kernel win).

## 4. What would have to be true for L-01 to work anyway

Three ways the numbers above could still support a real win, each naming what it would need:

1. **B_host is much lower than 1.98 ms in an isolated, single-expert measurement.** My estimate
   folds in attention/router/shared-expert time as if it were expert compute, which
   over-counts. If those are, say, 30% of the 476 ms/token, real per-expert cost could be closer
   to ~1.4 ms — still above the 0.66 ms per-layer DMA baseline, not a game-changer, but the gap
   matters for exactly how narrow a trigger condition is needed. **First concrete step: an
   isolated Go benchmark of `swiGLUExpert`/`moeMLP` at gemma-4-26b's exact expert geometry,
   synthetic weights (same pattern as `metal/attention_prefill_bench_test.go` used real
   architecture geometry with random data) — no real checkpoint load, seconds not minutes.**
2. **A genuinely faster, lower-latency single-expert CPU kernel** (different from today's
   `parallelCols`-parallel-but-still-~2ms path) — e.g. one tuned for minimum latency at M=1
   rather than the throughput-oriented blocked kernel the decode path shares with everything
   else. This is real, scoped kernel work, not a wiring change, and was not budgeted by the
   audit's one-paragraph mechanism description.
3. **A narrower trigger than "every miss splits":** only route to CPU when a layer's m exceeds
   some threshold where GPU-side DMA queueing/backpressure already dominates (bursty misses,
   not the steady 1.915 average) — meaning most tokens never touch the CPU path at all, and it
   only helps the tail. This changes q⋆'s formula from a smooth ratio to a threshold gate, and
   needs the trace's own miss-count DISTRIBUTION (not just its mean) to size — `g33-routing-trace.json`
   already has this; it was not pulled in this pass and is the next thing to read off it if this
   direction is pursued.

## 5. q⋆ sizing: what the formula needs and what's still missing

`q⋆ ≈ m·(B_PCIe/B_host)` is the low-traffic-CPU-share approximation of the balance point
`q⋆ = m·B_PCIe/(B_host+B_PCIe)` (time-to-fetch q⋆ on GPU ≈ time-to-compute the rest on CPU,
so the two paths, run concurrently, finish together). With this pass's numbers
(B_PCIe⁻¹≈0.346 ms/expert, B_host⁻¹≈1.98 ms/expert, both as PER-EXPERT COSTS not rates — so
B_PCIe/B_host in the formula's own terms is (1/0.346)/(1/1.98) ≈ 5.72):

- q⋆ ≈ m × 5.72/(1+5.72) ≈ 0.85m — i.e. **the balance point already says ~85% of misses should
  stay on GPU/DMA, only ~15% go to CPU** — consistent with §3's per-layer arithmetic (CPU is the
  slower per-unit path here, so the balance naturally leans away from it), and a much more
  modest mechanism than "split roughly half."
- This is NOT the same as §3's per-layer finding that even ONE CPU expert may lose outright —
  the balance-point formula assumes the two paths run perfectly concurrently and finish
  together; §3's finding is that CPU's OWN latency for one expert already exceeds the layer's
  total current budget, i.e. concurrency does not save it because there's nothing on the GPU
  side happening at the same time worth waiting out. **These two views need to be reconciled
  with a real measurement, not two different approximations talking past each other** — which is
  exactly why this pass stops here rather than picking one and building against it.

## 6. Partial-sum merge — the "exactly" in the audit's mechanism

Not designed in detail this pass (gated on §4/§5 resolving first), but named so it isn't a
surprise later: `moeMLP`'s existing weighted-sum accumulation (`out[i] += w * expOut[i]` per
expert, §3 above) already merges an arbitrary subset of experts into one output — GPU-computed
and CPU-computed expert outputs would merge the SAME way, order-independent for f32 addition
modulo reassociation (the repo's own `MatmulBTAcc64` precedent shows where that has mattered:
router-adjacent sums that can flip a near-tie). Merging is not the hard part; getting the
GPU-side partial output back from CUDA to combine with the CPU-side partial output without a
second host round trip (defeating the whole point) is — this needs its own design once the
above is resolved, not before.

## 7. The speculation-antagonism risk — not re-verified this pass

`docs/task-freetoken-techniques.md` §"Pre-registered risk" names four gates that currently make
the collision moot (nothing large enough to need expert streaming is ever allowed to speculate
on CUDA today): MoE batched-verify decline, recurrent-rollback refusal, windowed-rollback
refusal, MLA having no CUDA resident path. **I did not re-confirm all four still exist in
current code this pass** (a `grep` for one exact phrase came back empty, which may mean the
comment wording moved, not that the gate is gone — worth a direct check, not a re-derivation,
before any L-01 code, since L-01 would be the first feature to make expert-streaming and
speculation coexist).

## 8. Recommended next step

**Not CUDA code.** In order:

1. The isolated CPU expert-kernel microbenchmark (§4.1) — cheap, synthetic, answers whether
   B_host≈1.98ms is real or an artifact of folding in attention/router cost.
2. Pull the miss-count DISTRIBUTION (not just mean) from `g33-routing-trace.json` — already
   captured, no new hardware run — to check whether a threshold-gated design (§4.3) has a real
   target (a meaningful tail of high-m layers) or whether misses are evenly spread (in which
   case a threshold gate helps nobody and the mechanism needs #2 from §4 instead).
3. Re-confirm the four speculation gates (§7) directly against current `cuda/*.go`.
4. Only then: a scoped kernel-design pass for whichever of §4's three paths the above supports,
   with its own pre-registered, paired-and-interleaved measurement plan on the real Qwen3.6-35B-A3B
   cell per §0's decision rule.

## Sources

`docs/audit-2026-09-02.md` L-01 (mechanism, decision rule, disposition) · `docs/task-freetoken-techniques.md`
Lead 5 (architecture, antagonism risk) · `docs/QUEUE.md` G31–G34 (DMA cost law, overlap ceiling,
miss classification, block-verify's own infra-gap finding) · `decoder/mlp.go` (`moeMLP`,
`swiGLUExpert` — the existing sequential CPU expert loop) · aikit `linalg.MatmulBT`/`parallelCols`
(the existing per-expert core-width parallelism) · `internal/fitcmd/fit.go` (`-measure`, used
as-is for §2's new number) · `docs/measurements/g33-routing-trace.json` (not re-read for its
distribution in this pass — named in §8 as the next pull).

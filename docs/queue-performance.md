# Performance queue

> **EMPTY as of 2026-08-31.** Every item that was here is closed, refuted, withdrawn, or has moved
> to the track that owns it. This file is live — new performance work is filed here — it just has
> nothing in it right now.
>
> **The closed record is [`docs/completed/queue-performance.md`](./completed/queue-performance.md)**
> (2,245 lines, G15–G24 · P1–P16 · A1–A11 and the A9-\* series). It moved rather than being deleted
> because the reasoning is the point, negative results included. Per `docs/README.md`'s archival
> rule, `completed/` is not scanned by the citation lint, so archiving it also retired its citations
> from the live gate — that is deliberate, not an oversight. Pages that linked to
> `docs/queue-performance.md` still resolve here.

## What is open

- **P19 · Fused (FlashAttention-style) tiled attention — STEP 0 ANSWERED, item stays OPEN.**
  *(Filed as P19, not P17: this queue already has a P17. The item arrived carrying that number
  from outside the repo; renumbered here rather than shadowing an existing entry.)*

  **Step 0 asked whether G20's tile loop fuses the softmax or materializes per-tile scores, and
  said the item closes at zero cost if already fused. It is NOT fused.** The tile is over QUERY
  ROWS only — `scores` is `tile × nKeys`, the full key extent per row — and `attendBatchedHeads`
  forbids the fused schedule explicitly: *"No key-dimension split happens here and none may: that
  would re-associate the softmax denominator and the AV fold, the exact thing acc64 exists to
  prevent."* G20's own stated purpose is *"The point is memory, not speed"* — bounding scratch so
  the worker pool can still fan out. So the N-wide score row makes three trips through memory per
  tile, which is the traffic a fused schedule removes. **Written down because nobody knew.**

  **The prize, measured rather than assumed** (`docs/measurements/cuda-prefill-attention-share-2026-09-01.md`).
  Attention's share of CUDA prefill, dense 1.5B int4, RTX 2070 SUPER:

  | K | 128 | 512 | 2048 | 3900 |
  |---|---|---|---|---|
  | attention % | 5.3% | 14.3% | **39.0%** | **55.0%** |

  K=2048 reproduces the 2026-08-04 figure of 39% exactly, across a driver/distro re-anchor — the
  instrument validated itself before the new cell was read. Amdahl at K=3900: a perfect fusion caps
  at **2.22×**, a plausible 2.5× on the attention term gives **~1.49×**.

  **THE DECIDING VARIABLE IS MODEL CLASS, NOT BACKEND**, and getting that wrong nearly killed the
  item. A first pricing used the 2026-09-01 full-model profile's **17.4%** attention share and
  concluded a 1.21× ceiling — but that is **Mellum2, a MoE**, whose expert FFN takes 42.1%. Dense
  models at depth run ~70% (Mac CPU acc64, K=8192) and 55% (CUDA, K=3900). Attention dominates at
  depth on dense on BOTH backends; the 17.4% is a property of the model, not of prefill.

  **It coheres with the CUDA prefill re-anchor.** That measured goinfer's marginal cost per prefill
  token RISING with depth (0.377 → 0.932 ms/tok) while Ollama's stays FLAT (0.064 → 0.063). A
  rising marginal cost is the O(K²) attention term; a flat one is what a tiled/fused attention
  produces. Three independent measurements today point at the same mechanism.

  **Cost, unchanged from the filing:** the running-max rescale re-associates, so a fused path is
  NOT bit-identical — same category as `--cpu-fast-attention`, breaking the same A1 guarantees
  (spec-decode verify == sequential greedy, decode == prefill).

  **Two notes on the item's own text, now stale.** Its confound warning — *"the f32 gather branch
  is single-threaded by construction (shared mutable kh/vt state), so any end-to-end A/B that moves
  both precision and fusion at once measures the parallelism confound"* — was fixed on 2026-09-01
  (A3 fan-out: each worker gathers into its own buffers). And its sequencing condition, "A3 first",
  is satisfied: A3's kernel measurement and its end-to-end result both landed.

  **Not yet known, and what decides funding:** no fused kernel exists, so the 2.22× is an Amdahl
  ceiling from a measured share, not a measured win — the exact distinction this repo retracted
  twice on 2026-09-01. Next step is a prototype fused kernel measured against the materialized one
  at matched precision, so the fusion win is attributable separately from A3's.

- **P18 · Expert-major MoE prefill batching — REOPENED 2026-09-01, and the reason it was parked
  does not apply to it.** Scoped, not funded: the next step is a cost measurement, not a rewrite.

  **Why it is back.** The full-model K=8192 profile
  (`docs/measurements/mellum2-fullmodel-profile-RESULT.md`) moved the target after A3 collapsed
  attention: `moeMLP` is now **42.1%** of prefill against attention's 17.4%, and inside it
  `swiGLUExpert` is **93.1%** — so the expert weight matmuls are **~39% of prefill and the largest
  single bucket**. `routeExperts` is 1.7% of `moeMLP`; routing is not the cost and never was.

  **Why the existing verdict does not close it.** `task-moe-streaming.md` Lever 4 is parked on
  "expert-major MoE prefill batching is NOT a compute lever", whose ceiling was `uniform` (every
  row picks the same experts) vs `varied`. **Both arms call `moeMLP` per row at M=1**, so that
  experiment varies *which weights are touched* — the bandwidth/locality axis — and holds the
  M=1→M=N axis fixed. Measured 2026-09-01 at real Mellum2 expert shapes with int4 weights and
  locality already perfect (i.e. under the parked ceiling's own condition): **1.55× at N=8 rising
  to 2.13× at N=256**
  (`docs/measurements/moe-expert-batching-m1-vs-mn-2026-09-01.md`). Real N is ~10³ rows per expert
  per layer at K=8192.

  **What is NOT yet known, and is the whole item.** The gather/scatter cost of collecting an
  expert's scattered rows is unmeasured, and it is the most likely reason the real win lands below
  2.13×. Uneven routing puts small groups at the left of that table (1.55×), not the right. And
  the microbenchmark is cache-warm on three matrices while the real forward ran with swap growing
  to 18 GB — a benchmark that omits the real pressure can promote a mechanism as easily as
  exonerate one ([[synthetic-reproduces-shape-not-pressure]] is about the same trap in the other
  direction).

  **No end-to-end speedup is claimed.** Multiplying 39% by 2.13× is the projection this repo
  retracted twice on 2026-09-01. **Decision rule, pre-registered here:** measure permutation cost
  against batching win on the real forward at K≥4096; **fund if the net is ≥15% end-to-end,
  park if <8%, and treat 8–15% as ambiguous → parked pending a second mechanism.**

  Structural note for whoever picks it up: `canBatchN`'s comment states the per-row design as a
  deliberate constraint ("the MoE FFN itself stays per-row"), so this is a restructuring of the
  prefill loop, not a kernel swap.

An observation rather than a work item — nothing below is claimed as a bottleneck to go fix.

- **P17 · `TestSamplingThroughputGate` has ~1% headroom on arm64 and fails under suite load** —
  **DIAGNOSED AND FIXED 2026-08-31. The bar was NOT moved.**

  **The question was: is full-vocab selection genuinely load-sensitive, or was the gate set with no
  headroom on this machine class? Neither, quite.** The gate is a RATIO, and both its level and its
  variance come from the DENOMINATOR (temp-only) — which is not the thing under test. Measured, three
  consecutive isolated runs at V=262144:

  | | temp-only (denominator) | temp+top_p (numerator) | ratio |
  |---|---|---|---|
  | x86 | 1.356 / 1.694 / 1.994 ms — **47% spread** | 6.54 / 6.72 / 6.93 ms — 6% | 4.83 / 3.97 / 3.48 |
  | arm64 | 558.7 / 564.6 / 558.6 µs — 1% | 2.682 / 2.685 / 2.680 ms — **0.2%** | 4.80 / 4.76 / 4.80 |

  The numerator — the "full-vocab selection" the gate claims to protect — is the most stable
  quantity in the measurement, on both machines. The ratio moves because the baseline moves, which
  is what the gate's own comment already warned it does ("a gate whose baseline moves is measuring
  two things at once", written when P2b made that denominator 4.7x faster and the bound had to be
  raised).

  **Fix: best-of-3 instead of a single `testing.Benchmark` mean.** A mean tracks jitter upward — it
  has a floor but no ceiling — and the quantity is floored, so the minimum is the right estimator.
  Under full-suite load, the exact condition that produced the original failure: **5.13x FAIL →
  4.82x PASS.** x86 run-to-run spread cut from 1.35x to 0.39x.

  **A prediction I made and the measurement refuted**, recorded because the correction is the
  useful part: I expected best-of-N to make the ratio machine-independent (predicted x86 4.83 vs
  arm64 4.80). It does not — x86 settles near 4.0 and arm64 near 4.83, a real 0.7x gap on the same
  code. The comment in the test said the wrong thing for one commit and now says the measured one.

  **Residual, NOT fixed:** the bound is effectively set by whichever machine runs hottest on this
  ratio (arm64, ~3% margin), and the denominator remains an optimization target — any future
  speedup to temp-only re-tightens this gate for everyone. The structural answer is to gate the
  NUMERATOR, which is stable to 0.2%; the reason that was not done here is that an absolute
  numerator bound is machine-dependent (2.7 ms arm64 vs 6.5 ms x86), which is the problem the ratio
  was chosen to avoid. Filing that trade rather than resolving it.

  Measured while landing G1, on a tree whose diff contains **no sampler file** — the benchmarked
  code is byte-identical to `main`, so this is not a regression from that work:

  | context | V=262144 temp+top_p ÷ temp-only | vs gate 5.0x |
  |---|---|---|
  | inside the full `./decoder/` suite | **5.13x** | **FAIL** |
  | isolated, `-count=1`, four runs | 4.81 / 4.95 / 4.79 / 4.97 | pass, by 0.6-4% |

  So the gate passes alone and fails under concurrent load, with a mean margin under 3%. Two
  traps worth naming before anyone picks this up:

  - **`go test` CACHES a passing result.** The first three "repeats" here returned byte-identical
    timings (566997 / 2759883 ns/op) because only the first actually ran. Any repeat-measurement
    of this gate needs `-count=1`, or it measures nothing and looks stable while doing it.
  - **Do NOT raise the bar to 5.5x to make this green.** Re-baselining because a number moved is
    how a regression gets blessed; a bar moves only with a mechanism. The open question is whether
    full-vocab selection is genuinely load-sensitive (a cache-pressure story) or whether the gate
    was set with no headroom on this machine class in the first place. Answer that first.

## Where the last items went

| item | disposition |
|---|---|
| **P16** · re-anchor to Nobara 44 / driver `595.91.07` | **DONE 2026-08-31.** Six legs were already re-anchored on 2026-08-27; the seventh (the v0.11.0 qualification) was **retired** rather than re-measured — its numbers were §B6/§B7 by code-identity, and both are current. |
| **P14** · the CPU gap is kernel arithmetic | **DONE.** Items 1+2 refuted (centering), item 3 built, wired, measured (+2.10% decode) and **parked default-OFF** against its pre-registered 4% bar. |
| **A2** · 26B documentation correction | **DONE 2026-08-31**, published as `docs/benchmarks.md` §B4.2. |
| **A10** · the ~150 MiB allocation floor | **CLOSED 2026-08-12** — fully decomposed, nothing unattributed. |
| **D3b** · the expert-cache default | **SHIPPED 2026-08-20** (`8f3c5e7`). Lives in `docs/queue-release.md`. |
| **P10** · DSpark / DFlash block drafters | **MOVED to the speculation track** — see below. Not finished; not performance-queue debt. |
| **P15** · DFlash 2 | **MOVED to the speculation track** — see below. |

## P10 and P15 moved rather than closed, and the distinction matters

They are **not done**, and nothing here should be read as saying they are. They left this queue
because they were never really performance-queue items: the rest of this queue was *find a
bottleneck, measure it, fix or refute it*, and those two are an ongoing research program with
pre-registered kill-gates.

Their substance already lived in [`docs/spec/08-dspark-dflash.md`](./spec/08-dspark-dflash.md)
(~25.5k words — the kill-gates, the increment log, the licensing correction, the Metal verdict);
the entries here were a second, thinner copy that could drift from it. The spec track owns them,
[`docs/spec/README.md`](./spec/README.md) indexes them, and
[`docs/spec/experiments.md`](./spec/experiments.md) is the dated run log.

**Open state, carried over so it is not lost in the move:**

- **P10 · increment 4.** Kill-gates 1 and 2 cleared 2026-08-15 (6.78 tok/verify against a ≥3.0
  bar). Remaining: gate 3 — end-to-end **≥1.3× vs plain resident decode on ≥1 GPU backend** — and
  gate 4's mixed-workload width router. The **Metal leg is measured not-ready**: ~1.13× ceiling
  even at `draft_ms=0`, and `PrefillLast` is not bit-identical, so the lossless contract cannot be
  met there. `gpt-oss` is blocked on a missing harmony chat template, not on the seam.
- **P15 · DFlash 2.** Filed 2026-08-20, **gates before code**, not started. Sequenced to land
  **before** P10's gate-4 width router — doing gate 4 first would mean redoing it.

## Filing a new item

Read `docs/completed/queue-performance.md` first — it is 2,245 lines of what was already tried,
including the negative results, and several of its entries exist because something was rebuilt that
had already been measured and rejected. Then follow the same discipline the archive does: state the
mechanism, pre-register the decision rule and its ambiguous band, include the do-nothing arm, and
record a negative with the same care as a win.

**One recurring defect that archive documents four times over, worth knowing before you add
anything:** an entry gets its resolution *appended* while the stale conclusion is left standing in
verdict position, so the item reads as open long after it closed. A2 (a pre-registration answered
four days earlier), D3b (shipped eleven days earlier), A10 (resolved, header still said OPEN) and
P16 (a stale-list four items out of date) all failed this way. **When you close something, correct
the sentence a scanner stops at — not only the body.**

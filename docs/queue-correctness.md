# Correctness queue

Parity, numerics, goldens, quantization, model families. Anything whose success criterion is **agreement with a reference** — a cosine, an argmax match, a golden. If the question is *does it compute the right thing*, it belongs here.

> **One of four queues.** The work list is split by *success criterion*, not by component:
> [performance](queue-performance.md) · [correctness](queue-correctness.md) ·
> [engineering](queue-engineering.md) · [release](queue-release.md).
> [`QUEUE.md`](QUEUE.md) is the index over all four and holds the cross-cutting sweeps.
>
> **Task docs are NOT queues.** `docs/task-*.md` are *design records* — why a thing is built as it
> is — and they are cited from 88 code comments. A queue entry cannot carry that, so the task docs
> stay put and the queues hold only the open work.
>
> Entries keep the section they were filed under (`In flight`, `Queued`, …) and their original IDs,
> so a citation to an ID still finds it.


## Queued

> **ELEVEN closed entries are archived** in
> [`docs/completed/queue-correctness.md`](completed/queue-correctness.md) — G4, G5, G6, G9, G10,
> G11 and Q2 (2026-08-31, first pass), then **G7, G1, Q3 and Q1** (2026-08-31, second pass, when
> the last of them closed).
>
> **What is below is the open work, and it is one PARKED item.** G8 is not queued-and-waiting: it
> is blocked on hardware that does not exist here (V4-Flash is 0.16 TB against 62 GB of host RAM),
> so nothing in this queue is startable today. That is a real state, not an empty-queue
> formality — read G8's entry for what would unpark it.

**G8 · DeepSeek V4-Flash as a new family** — `any`, **PARKED — and the reason has CHANGED
(2026-08-31). It is no longer "lowest priority"; it is UNVALIDATABLE on any hardware here.**

> **Two things were checked before treating Q3 as the unblock, and both move the item.**
>
> **1. The fp8 prerequisite is only HALF lifted.** Q3 shipped e4m3 with blockwise **f32** scales
> read from `*.weight_scale_inv`. V4-Flash's config declares **`scale_fmt: "ue8m0"`** — an
> exponent-only 8-bit scale encoding — and `grep` for `ue8m0`/`scale_fmt` across `decoder/`
> returns NOTHING. So "Q3 unblocks G8" is true of the element format and unproven of the scale
> format. Anyone picking this up should confirm what V4 actually stores before assuming the
> reader covers it.
>
> **2. The decisive blocker is now HARDWARE, and it is not close.** V4-Flash is **0.16 TB** — the
> SMALLEST real member (V4-Pro 0.86 TB, Kimi-K3 1.56 TB). Against 62 GB of host RAM on the box,
> 8 GB of VRAM, and 16 GB unified on the Mac, there is no configuration in which a whole V4
> forward runs here. Expert streaming does not rescue it either: `--moe-cache-experts` needs the
> weights resident in HOST memory, which is the 62 GB that 160 GB does not fit in.
>
> **WHY THAT IS DISQUALIFYING RATHER THAN INCONVENIENT — G7 just paid for this lesson.** G7's
> gate was ONE real forward, and running it found THREE silent defects (a slot-indexed bias
> table, a never-applied per-expert bias, an unloaded router kernel) that `./cuda/...` at 100
> PASS could not see, because every one was a term the WIRING dropped rather than a kernel
> computing it wrongly. G8 is EIGHT new primitives — DSA sparse attention over a learned Indexer,
> strided KV compression, sliding-window + attention sink, grouped low-rank output projection,
> hash routing, `sqrtsoftplus` scoring, hyper-connections, clamped SwiGLU. Building that with no
> possible end-to-end run reproduces G7's failure mode across eight times the surface, with no
> way to detect it. Kernel-level gates would all be green.
>
> **So the honest state is not "later, it is low value" but "not yet, the proof is unreachable".**
> The strategic case in the original entry stands — DSA is where V3.2/GLM-5.1/V4 are converging,
> and building the compressor path once plausibly buys several frontier releases. What changed is
> that the entry read as a priority call when it is really a hardware precondition, and a reader
> deciding what to do next should see the difference.
>
> **What would unpark it,** in order of cost: a machine with ≥256 GB host RAM (V4-Flash then fits
> for a CPU-reference run, which is the arm the gate actually needs); or a released sub-100B
> member of the family; or upstream publishing a small DSA reference the primitives could be
> validated against piecewise — the last being the only one that does not require new hardware,
> and the one worth watching for.

**Original entry follows.**

**G8 (original) · DeepSeek V4-Flash as a new family — blocked on fp8 support, post-1.0** — `any`.

Scoping already done: `docs/completed/task-model-family-deepseek-v4-kimi-k3.md`'s Phase 0 verdict.
**Not** a `deepseekArchitecture` alias — eight new primitives (DSA sparse attention over a learned
Indexer, strided KV compression, sliding-window + attention sink, grouped low-rank output
projection, hash routing, `sqrtsoftplus` router scoring, hyper-connections, clamped SwiGLU).
**Hard prerequisite, not a subtask:** V4-Flash ships fp8 e4m3 blockwise-quantized weights and
**there is no fp8 support anywhere in the tree today** — file/estimate the fp8 reader as its own
piece of work before scoping the primitive additions. MIT license, DeepSeek's brand pulls the
whole local community, and native sparse attention is where the field (V3.2, GLM-5.1, V4) is
converging — building the DSA/compressor path once plausibly buys the next several Chinese
frontier releases, which is the strategic case for filing this now even though it's not a
near-term ship. Lowest priority of the five items filed alongside this one (`G4`-`G7`).

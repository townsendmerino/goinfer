# Decode-residency campaign — landscape & verdicts

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


The one-page synthesis of the GPU-decode work: what shipped, what's a labeled dead end, and the
methodology that makes both trustworthy. Read this first; each thread is detailed in its own doc
(see the map at the bottom). RTX 2070 SUPER (8 GB) unless noted.

## The arc

The campaign opened on a thesis — *"`dot4I8Packed`/DP4A is the cheap, high-certainty TTFT win"* —
that **measurement demolished** (bandwidth-bound; ~0 gain). What it produced instead, by refusing
to ship anything that didn't survive a control, was a different and larger win: a **default-on
resident SSM/Mamba decode engine** for a hybrid family we actually serve. The pivot was the reframe
that overturned an "intractable" verdict — **decode is a bounded per-token recurrence, not the
prefill scan** — so the Mamba state slots onto the resident DecodeRunner like a KV cache.

## The win — Nemotron-H resident, DEFAULT-on at int4

Near-lossless, ~10× over CPU, fits 8 GB. The headline metrics (9B-v2, int4 resident vs f32 CPU):

| metric | value |
|---|---|
| perplexity | **1.677** vs f32 **1.695** (within noise) |
| mean KL(f32‖int4) | **0.058** |
| greedy agreement | 92.5% — but **top-2 agreement 99.6%** |
| disagreements | **100% benign** (median f32 margin 0.069 vs 0.953 at agreements; 16/17 picks f32's exact #2; **zero** confident-token errors) |
| speed | 77.2 ms/tok = 12.9 tok/s (~13× CPU) |

The KL (0.058, ~16× smaller than granite's 0.93) certifies it under **sampling**, not just greedy —
the full distribution tracks f32, so the default flip is good for real serving. The eligibility flip
is guarded: default-on only when projections loaded int4; int8 (unmeasured on 8 GB) stays opt-in
behind `GOINFER_SSM_RESIDENT`; every other family is byte-identical. Detail: `nemotron-resident.md`.

## The three-family scorecard

| family | verdict | why |
|---|---|---|
| **Nemotron-H** | **DEFAULT-on (int4)** — the win | dense squared-ReLU hybrid, no router → quantizes cleanly (above) |
| **Granite-4.0-H** | ported, **opt-in greedy-only** | its deep all-MoE stack hits a real int8 cliff (next section), proven fundamental, not a bug |
| **Llama-4** | ports cleanly, **doesn't fit 8 GB** | needs ≥12 GB; the resident path itself is correct |

## Granite's int8 wall — a real cliff, fully diagnosed (not a bug)

Granite int8 resident degrades hard (66% agreement / KL 0.93 / 2.4× perplexity) and **stays
opt-in**. This was chased to ground through four *rejected* root-cause guesses, each killed by a
control, before the truth survived:

1. ~~int8 mamba projections too lossy~~ — refuted: int8 mamba on CPU = 97.3%.
2. ~~f64-vs-f32 SSM compute + router sensitivity~~ — refuted: f32 SSM on CPU = 100%.
3. ~~the GPU mamba kernels~~ — refuted (Phase A): the conv/ssm/gatedNorm kernels are **bit-correct**
   on real inputs, in isolation *and* in the full plan (cosine 1.000000). The "D3 vs resident"
   delta that implicated them was a **confound** — D3 ran attn/MoE through the staged tiled GEMM
   while the resident ran its own W8A8 GEMVs.
4. ~~activation precision / accumulation~~ — refuted: W8A16 (f32 activations) = 62.6%, no recovery;
   the resident is **invariant to every precision/accumulation knob** (int8/f16/W8A16 all ~63–66%).

**Verdict — fundamental and precision-invariant:** chaotic f32-reduction-order perturbations across
the deep stack, which granite's **64-expert top-6 router turns into discrete expert-selection
flips** (a cliff). Nemotron has no router, so the *same* perturbations stay smooth — which is
exactly why Nemotron is default-on and Granite isn't. The GPU has no f64 and can't match the
reference's reduction order, so there is no cheap fix. Detail: `ssm-int8-quality.md`.

## Labeled dead ends (don't re-fund)

| lever | verdict | evidence |
|---|---|---|
| `dot4I8Packed` / DP4A TTFT | **bandwidth-bound, ~0 gain** | the opening thesis; measurement demolished it (`gpu-assessment.md`) |
| kernel fusion (dispatch-count cut) | **bounded** | Incr1 rope+k+v +1.5%, Incr2 bias→GEMV epilogue +2.3% (real Qwen2.5), Incr3 qk-norm fold **rejected** −2.7% (occupancy). Lesson: dispatch-count cut is necessary not sufficient — keep launch geometry (`decode-fusion-next.md`) |
| **go-webgpu** (goffi, zero-CGO) binding | **Go-1.26 callback-ABI crash** at `RequestAdapter` — the REAL binding blocker (its own `examples/compute` SIGABRTs); but it's the **goffi FFI**, not wgpu-native v29 | blocked (`gpu-gowebgpu-migration-assessment.md`) |
| ~~wgpu-native **v29 = decode slowdown**~~ | **REFUTED** — measured directly via the `oliverbestmann/webgpu` CGO v29 fork: gemv compute v29 0.64 ms ≈ v22 0.62 ms (~4%), per-dispatch record **identical** (1.1 µs both), 256-dispatch decode-shaped total +3.5% (the fork's GC layer). The earlier "−23% v29 decode penalty" does NOT reproduce | stay on cogentcore — but for *zero churn*, not a v29 perf cliff (`gpu-gowebgpu-migration-assessment.md`) |
| granite f16 mamba / router-island / W8A16 | **all refuted** | see the wall section + `ssm-int8-quality.md` |

## Why the win is trustworthy (methodology)

The negatives are *final* and the win is *certified* for the same reason: **capture-and-diff
controls**. The pattern that worked repeatedly — capture the resident's REAL per-op I/O and diff it
against the CPU reference, rather than theorize — is what caught the D3 confound, exonerated the GPU
mamba kernels, and characterized the Nemotron disagreements as benign (margin-correlated, not
arbitrary-threshold). Four wrong root causes were refused until one survived a control; that's what
lets "default-on" mean default-on and "dead end" mean dead end.

## Open items (neither needs chasing now)

- **Mellum2 tok/s** — unmeasured; blocked behind a windowed-attention deadlock (a real pre-existing
  bug to clear first). Prompt is ready when wanted.
- **Nemotron int8-default** — only matters on ≥12 GB; int8 quality is ≥ int4 (nemotron is not at a
  precision-invariant wall), so the flip there is a measurement formality.

## Doc map

- Reframe + SSM engine scope/build — `ssm-residency-scope.md`, `ssm-residency-build.md`
- Granite int8 wall (full investigation) — `ssm-int8-quality.md`
- Nemotron-H port + default-on flip — `nemotron-resident.md`
- Resident coverage ladder + family triage — `gpu-residency-coverage.md`, `residency-port-triage.md`
- Dead ends (dot4 / fusion / binding / go-webgpu) — `gpu-assessment.md`,
  `gpu-next-levers-assessment.md`, `decode-fusion-next.md`, `gpu-gowebgpu-migration-assessment.md`

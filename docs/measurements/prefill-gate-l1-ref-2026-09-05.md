# L1 §3.1 re-run — Metal batched prefill AND sequential decode, both vs a CPU f32-activation reference (2026-09-05)

**Verdict: neither model ships on the decision set, and the failures do not share a cause.**
`measurements/prefill-gate-l1-2026-09-05.md` scored Metal's fast (f16-activation) prefill against
its own exact (int8-per-row-activation) prefill and found a real distance between them, but §3.1
found that comparison couldn't say which arm was closer to the truth — both are quantisations. This
re-run adds a third arm: a CPU reference with f32 activations, and scores both Metal arms against
it, paired per prompt. The pre-registered prediction ("fast closer than exact on every cell, margin
larger on D7") does not hold as stated. What holds instead: **fast's mean KL divergence from the
reference is lower than exact's in every one of the five cells measured** — by that continuous
measure fast is never worse and often meaningfully closer — but the two discrete pass/fail criteria
(hard-flip count, teacher-forced agreement within tolerance) each fail on one cell, and the two
failures are at opposite ends of the K range on opposite models, on different criteria. Metal's
default stays sequential; `--metal-fast-prefill` stays the disclosed opt-in; Phase 2 does not
proceed from this doc's plan, but not because fast was shown to be worse — it wasn't.

## Provenance

- **Machine:** MacBook Pro, Apple M1 Pro, macOS 26.6.2 (Darwin 25.6.0 arm64).
- **goinfer:** `29d40c2` (code) / `556523a` (perf fix, CPU reference generator) — both on `main`.
  Phase A ran under `556523a`; Phase B (the Metal scoring) under the same commit, no further code
  changes between the two phases.
- **Thermal note: NOT instrument-read, and worse than the first run's gap, not better.**
  `sudo powermetrics` needs an interactive password this session doesn't have, so there is no SMC
  sample at all this time (the first run at least ran on a box the operator attested was otherwise
  idle). This run's Phase A (CPU, ~60 min, sustained 500-700% across up to 8 cores) and Phase B
  (Metal, ~44 min) both ran while the machine was in **normal interactive use** — the operator
  confirmed this mid-run. That should not affect the CORRECTNESS numbers below (agreement, KL,
  hard-flip counts are deterministic given the same inputs; nothing here is a race with the
  operator's own processes), but it means the WALL-CLOCK durations reported are not clean speed
  measurements and should not be quoted as such.
- **Checkpoints, both from `~/models`:** S = `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (1.5B), D7 =
  `qwen2.5-7b-instruct-q4_k_m.gguf` (7B).
- **Reference (Phase A):** CPU backend, `Options{Backend:"cpu"}`. S at `Quant:""` (f32 weights, f32
  activations, ~6GB). D7 at `Quant:"int8"` (weight-only per-row int8, f32 activations, ~7.6GB) —
  D7's f32 weights would be ~28GB and do not fit; this was the pre-registered fallback and it
  worked (no load failure). `GOINFER_CPU_FAST_ATTENTION=0` forced throughout, so the reference used
  the exact f64-accumulating attention kernel, never the f32-fast default.
- **Metal arms (Phase B):** `Options{Backend:"metal", Quant:"int4"}`, both S and D7. Exact =
  sequential `Forward` per token (today's shipped default). Fast = batched `PrefillLast`
  (`GOINFER_METAL_BATCHED_PREFILL=1`, i.e. what `--metal-fast-prefill` sets).
- **Sampling:** none anywhere in the harness — every position scored by argmax over full logits.
- **Tests:** `decoder/prefill_ref_gen_test.go` (`TestPrefillGateReference`, Phase A) and
  `metal/prefill_gate_ref_test.go` (`TestPrefillGateVsReference`, Phase B), both
  `GOINFER_HEAVY_TESTS=1`.
- **Raw logs** (this directory): `prefill-ref-gen-2026-09-05.log` (Phase A, complete) and
  `prefill-gate-vs-ref-2026-09-05.log` (Phase B, complete). The 50 reference binaries themselves
  (`~/goinfer-logs/prefill-ref/*.bin`, 1.8GB) are NOT in the repo, per the brief — private,
  per-machine, per-session artifacts only Phase B needs.

## Method

Same ten real-prose prompts as the first run (`decoder.PrefillGateProseFiles`, moved there from
`metal/` specifically so both phases score the identical set), same K's: decision set {256, 1024}
on both models, S's K=3900 as a confirmation cell run after the decision (not gating).

**Phase A** built the reference once per (model, K, prompt): the CPU's own batched prefill
(`prefillLogits`, exposed for test as `PrefillLogitsForTest` — weights streamed once across all K
positions, ~1.7-2x faster than a naive per-token loop, bit-identical to it) for the seed logits,
then 64 more greedy CPU steps for the reference continuation and that continuation's own logits.
Run with 8-way concurrency across prompts (one goroutine per CPU core, each with its own
`*KVCache`) — verified safe with an inline preflight (N workers given the identical prompt, must
return bit-identical seed logits) before trusting it with real compute, since a CPU forward's
scratch state living on the cache rather than the shared model is a design fact, not something to
assume without checking. Both preflights (S and D7) passed.

**Phase B** ran both Metal arms per (model, K, prompt): EXACT sequential over the K-token prompt,
then FAST one batched `PrefillLast` call, both teacher-forced through 64 more positions on the
SAME reference-supplied tokens (neither arm's own output) so a divergence at position i can't
cascade into i+1 and so neither arm is privileged as "the tokens being predicted." Each arm scored
against the reference's seed logits and per-position continuation logits with the existing
`decoder/fidelity_testhook.go` scorer: `NearTieArgmaxForTest` (3%-near-tie hard-flip rule, same bar
CUDA decode is already held to against CPU), `TeacherForcedTop1AgreementForTest`, `KLDivergenceForTest`.

**Gate, pre-registered in §3 (as amended by §3.1):** fast ships for a (model, K) DECISION cell iff
all three:
（a）fast's hard-flip count vs the reference ≤ exact's, over the same 640 continuation positions
（b）fast's mean teacher-forced agreement ≥ exact's mean − 1.0 percentage point, AND fast ≥ exact on
≥ half the prompts (paired, not just the cell mean)
（c）fast's mean continuation KL(reference ‖ arm) ≤ 1.1 × exact's mean

A model ships only if every decision-set cell ships. S's K=3900 is scored and reported the same way
but is confirmation-only, per the brief — it doesn't veto or approve the decision on its own.

## Results

| model | K | exact meanAgree | fast meanAgree | exact HF/640 | fast HF/640 | exact meanKL | fast meanKL | critA (HF) | critB (agree) | critC (KL) | cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S | 256 | 92.8% | 93.3% | 4 | 4 | 0.0375 | 0.0335 | ✓ | ✓ | ✓ | **SHIPS** |
| S | 1024 | 90.3% | 88.9% | 8 | 8 | 0.0442 | 0.0432 | ✓ | ✗ (−1.4pt) | ✓ | **DOES NOT SHIP** |
| S | 3900 (confirm.) | 92.3% | 92.0% | 9 | 7 | 0.0507 | 0.0476 | ✓ | ✓ | ✓ | SHIPS |
| D7 | 256 | 87.8% | 87.0% | 14 | 17 | 0.0795 | 0.0747 | ✗ (17>14) | ✓ | ✓ | **DOES NOT SHIP** |
| D7 | 1024 | 87.0% | 88.8% | 12 | 11 | 0.0857 | 0.0802 | ✓ | ✓ | ✓ | SHIPS |

**S's decision-set verdict (K ∈ {256, 1024}): DOES NOT SHIP** — K=1024 fails criterion B by a
narrow margin (fast trails exact's agreement by 1.4 points, just past the 1.0-point tolerance).
K=256 and the K=3900 confirmation cell both ship cleanly.

**D7's decision-set verdict (K ∈ {256, 1024}): DOES NOT SHIP** — K=256 fails criterion A (fast's
17 hard-flips exceed exact's 14, out of 640 continuation positions). K=1024 ships cleanly, and by a
larger margin than S's K=1024 miss (fast actually beats exact on agreement there, +1.8pt).

**The one consistent signal across all five cells, decision and confirmation alike: fast's mean KL
divergence from the reference is lower than exact's, every time**, by margins from 4% (D7 K=1024:
0.0802 vs 0.0857) to 11% (S K=256: 0.0335 vs 0.0375) to 15% (S K=1024: 0.0432 vs 0.0442, S K=3900:
0.0476 vs 0.0507, D7 K=256: 0.0747 vs 0.0795). Criterion C (KL) never fails, at any cell, for
either model. Read alongside the two failing cells being narrow, single-criterion misses on
opposite ends of the K range for opposite models, this does not look like "fast is worse" in any
direction one can generalize — it looks like real cell-to-cell noise around a boundary where the
two arms are genuinely close, with a continuous measure (KL) favoring fast throughout and two
discrete pass/fail thresholds occasionally landing fast just on the wrong side by a small margin.

### Why this differs from the pre-registered prediction

§3.1 predicted "fast closer than exact on every cell, margin larger on D7," with the fallback
"if the prediction fails on D7 but not S, the batched attention or the f16 weight dequant is the
suspect, and the next probe is per-layer, not per-model." That fallback's premise (a clean
per-model split) doesn't match what happened: the prediction failed once on EACH model, not
uniformly on one, and the KL numbers it was reasoned from turned out to support fast rather than
contradict it. The per-layer follow-up as originally scoped (isolate whether attention or weight
dequant is responsible) is not motivated by this data — there is no clean model-level split for it
to explain. What the data actually asks for, if anyone wants to chase it further, is a per-PROMPT
question instead: are S's K=1024 misses and D7's K=256 misses concentrated on the same kind of
content (the six prompts don't obviously share a theme by inspection of the raw log), or is this
just sampling noise around a 3-4 percentage-point true gap that a 10-prompt cell can land on either
side of. Ten prompts was the pre-registered sample size; this run doesn't argue for changing it,
just notes that it's a real limit on how finely the two narrow misses can be trusted.

## The first Metal W4A8 fidelity number the tree has (for task-peer-benchmarks.md §4, not acted on here)

Exact-vs-reference (the "exact meanAgree" / "exact meanKL" columns above) is, on its own, the first
measurement anywhere in this tree of how far Metal's SHIPPED default prefill path sits from a true
f32-activation reference: 92.8%/90.3%/92.3% teacher-forced agreement for S at K=256/1024/3900, and
87.8%/87.0% for D7 at K=256/1024. This is exactly the number `task-peer-benchmarks.md` §4's
fidelity column has been scoping (a "teacher-forced agreement with an fp16 [here, f32] reference"
per that doc's own language) — the scorer that produces it
(`decoder/fidelity_testhook.go`'s `TeacherForcedTop1AgreementForTest`/`KLDivergenceForTest`) was
built for this gate specifically so it would be reusable there. Filed here as a finding, per the
brief: not wired into that doc's matrix, not acted on further in this session.

## What broke, and what didn't, this time

- **The concurrency question got asked before it got trusted, not after.** Before spending an hour
  of real compute on parallel CPU reference generation, an inline preflight (N goroutines, one
  identical prompt, independent caches) required bit-identical results across all workers. Both
  models' preflights passed on the first try. This is the same discipline the buffer-aliasing bug
  in the first run should have used from the start — verify a concurrency/aliasing assumption
  empirically, in the test itself, before the run that would make discovering it late expensive.
- **The `launchctl submit` auto-restart trap did not recur.** Every long job this run appended
  `; launchctl remove <label>` to its own command (now recorded in `CLAUDE.md`), and every log
  survived intact to be read after the process exited.
- **A live cross-session git collision did recur** — a stale `.git/index.lock` from what appears to
  be the other session's own interrupted `git commit`, in this same physical checkout. Investigated
  (`lsof`/`ps` showed no process actually holding it) before removing, per this repo's own git
  safety rule; `docs/QUEUE.md`'s in-flight edits from that session were left untouched.

# Prompt: measure the Metal verify curve for P10 (run on the M1 Pro)

Paste the block below into a Claude Code session on the MacBook. Everything above the line is
context for whoever is dispatching it; the prompt itself is self-contained.

**Why this exists.** P10's block-drafter speedup is currently a projection built from three
constants measured on the CUDA box (RTX 2070 SUPER). Acceptance — the other half — is measured
and is backend-independent, so it transfers to Metal unchanged. The three timing constants do
not. This task measures them, and nothing else.

---

In this repo (`goinfer`, branch `main`), I need three timing constants measured on **Metal /
Apple Silicon**. This is a measurement task, not a build task — do not implement a drafter.

## Background, in brief

P10 is speculative decoding with a pretrained *block* drafter (DFlash): the drafter proposes a
whole block of tokens, the target verifies them in one batched forward, and every token the
target's own argmax disagrees with is discarded (so it is lossless). See
`docs/spec/08-dspark-dflash.md`.

The speedup is

```
speedup = decode_ms x (1 + accepted) / (draft_ms + verify_ms(k))
verify_ms(k) = W + C*k          # batched M=k forward: one weight read, k rows
```

**Acceptance is already measured and transfers to Metal unchanged** — it is a property of the
drafter's distribution against the target's, i.e. numerics, not hardware. On Qwen3-4B, code
suite, int8, greedy, at each verify width `k` (mean accepted tokens per round):

| k | 16 | 12 | 10 | 8 | 7 | 6 | 4 |
|---|---|---|---|---|---|---|---|
| accepted | 5.10 | 5.01 | 4.64 | 4.24 | 3.97 | 3.41 | 2.45 |

**What is NOT measured on Metal is `decode_ms`, `W`, `C`, and `draft_ms`.** Those are the whole
calculation. On CUDA they are 11.12 / 8.77 / 2.35 / 6.6 ms, giving 1.74x on code at k=7.

## What I want

1. **The batched-verify curve `T(M) = W + C*M`** on a resident Metal target, at a realistic KV
   depth, for M in {1, 2, 4, 6, 8, 10, 12, 16}. Fit W and C. `T(1)` is `decode_ms`.
   - The CUDA analogue is `cuda/spec_verify_ceiling_test.go` (`TestSpecVerifyCeiling`), which
     times one batched `PrefillLast(M=k)` against k sequential `Forward` calls. Read it first
     and mirror its method on the `gpu/` (Metal) side. `gpu/` already has resident decode
     benchmarks (`decode_realmodel_bench_test.go`, `decoderunner_w4a8_bench_test.go`) to model
     the harness on.
   - Use a real model, not a tiny fixture. Which one is your call given what fits — say which
     you used and its quantization.

2. **`draft_ms`** — the cost of one DFlash block draft on Metal. The drafter is 5 transformer
   layers plus an `fc` over the concatenated tap hidden states; `decoder/dflash.go` has the
   forward and `BenchmarkDFlashTrunk` in `decoder/dflash_accept_test.go` (build tag `realckpt`)
   times it on CPU. If the trunk is CPU-only on the Mac today, measure it there and say so —
   a CPU draft against a Metal target is a real configuration and its cost is the number that
   matters.

3. **The composed speedup and the optimal k**, using the acceptance table above with your
   measured constants. Report the optimum and the curve either side of it.

## Two things to be careful about

**Do not reason from memory bandwidth.** The intuitive argument is that unified memory has less
bandwidth, so the fixed weight-read term `W` is a bigger share, so batched verify amortizes more
and Metal should look *better*. That argument is unsound here: goinfer decode runs at **6-10% of
peak bandwidth**, so it is not bandwidth-bound and `W` is not a bandwidth term — it is dispatch
and launch overhead. Measure it; do not scale the CUDA number.

**Projections in this repo have a track record of not surviving.** CUDA graphs projected
1.4-1.7x and measured **1.01x**, because CPU dispatch overlaps GPU compute at real model size in
a way the small-model estimate missed. The draft term is exactly that class of cost. If your
measured numbers disagree with the CUDA-derived expectation, the measurement is right.

## Deliverable

The four constants, the model and quantization you measured on, the fitted curve, and the
composed speedup with its optimal k. A short markdown summary appended to
`docs/spec/08-dspark-dflash.md` under a "Metal" heading, plus the benchmark committed so it can
be re-run. If Metal turns out not to amortize batched verify at all, that is a complete and
useful answer — say so plainly rather than finding a configuration that looks better.

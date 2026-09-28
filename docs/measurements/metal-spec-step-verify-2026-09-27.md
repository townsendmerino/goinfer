# `--spec ngram` on Metal, verified on the MC3 step kernels: 2.08× plain decode on copy-heavy traffic and 1.07× on chat (1.5B), every reply identical (2026-09-27)

MC4's spec item in [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md), "Metal spec verify on the
step kernels". This morning [`spec-vs-batching-metal-2026-09-27.md`](spec-vs-batching-metal-2026-09-27.md) measured
n-gram speculation on Metal at 0.98× plain decode even on copy-heavy prompts. Its verify (`ForwardN`) is
bit-identical, but each extra row costs 0.71–0.96 of a token. MC3's step kernels run 8 rows bit-identically for
~1.8–2.7 tokens, so they became the verify.

**Result: both deciding gates and all three served gates pass. It ships.**

| 1 client, greedy | copy (verbatim code) | chat (W7) |
|---|---:|---:|
| **1.5B, spec-new ÷ plain (graded)** | **2.082×** (2.072–2.084) | **1.068×** (1.066–1.317) |
| 1.5B, spec-old ÷ plain | 0.977× | 0.956× |
| 7B, spec-new ÷ plain (reported) | 1.846× | 1.014× |
| 7B, spec-old ÷ plain | 0.992× | 0.970× |

Every reply equals plain decode's, on both models and both workloads.

## 1. The deciding gates (exploratory, registered in `8bb80b1f`)

**Gate 1: identity.** `TestMC3Verify_sameSlotRowsBitIdentical` runs a step whose rows are one sequence's
consecutive positions on one KV slot, against production decoding the same tokens one after another on a twin slot.
- **0 logits and 0 K/V elements differ** in all 7 cases, on the 1.5B and the 7B: M = 2, 4 and 8 at depths 128 and
  2048, and a run of 8 straddling `attnFADepthFloor`, where one step mixes per-head and `attention_fa` rows.
- Logs: [`gates-1.5b.log`](metal-spec-step-verify-2026-09-27/gates-1.5b.log),
  [`gates-7b.log`](metal-spec-step-verify-2026-09-27/gates-7b.log).

**Gate 2: cost.** `TestMC3Verify_rowCost` times the step against a production token, as (step ÷ token − 1) ÷ (M − 1)
per extra row. The bar was ≤ 0.30 at M = 8 up to depth 2048.

| | extra row at M = 8, depth 128 / 512 / 2048 | step at M = 2 ÷ a token |
|---|---:|---:|
| 1.5B | 0.108 / 0.133 / 0.193 | 1.55–1.61× |
| 7B | 0.157 / 0.172 / 0.235 | 1.76–1.79× |

- The curve has a fixed overhead of ~0.2–0.3 of a token even at one row, then a nearly flat slope.
- So one draft token rarely pays, and eight nearly always do.
- Logs: [`cost-1.5b.log`](metal-spec-step-verify-2026-09-27/cost-1.5b.log), [`cost-7b.log`](metal-spec-step-verify-2026-09-27/cost-7b.log).

## 2. The build (`1e153876`)

- **`forwardMulti` accepts one slot on several rows,** but only as one sequence's consecutive positions in row order.
  Every row's K/V is stored before its attention runs, layer by layer. Two sequences on one slot are still refused.
- **`metalResident.PrefillLastNArgmax`** is the argmax-only verify the n-gram loop already prefers when a resident
  has one. Rows run on the step kernels in pieces of at most 8, and each argmax is read in place.
  - A one-row piece is production's own `Forward`, so a round with no draft costs what plain decode does.
  - Both argmax helpers keep the first strict maximum, so ties break as plain decode breaks them.
- **A cost-curve depth controller.** `AdaptiveDepth.Cost` holds the verify's cost by row count, and `Depth` picks
  the depth with the most expected tokens per unit cost, Σᵢ₌₀..d αⁱ ÷ Cost[d+1]. Plain decode scores 1.
  - The linear θ rule drafts while αᵏ > θ, so on this curve it would draft single tokens at chat-like acceptance and
    pay ~1.8 steps for ~1.4 tokens (`TestAdaptiveDepth_costCurve`).
  - Probes run every 64 idle rounds, not 16.
  - The curve comes from the resident (`decoder.VerifyCostReporter`): the conservative end, the 7B at depth 2048.
- **The curve applies to greedy requests only.** A sampled request still verifies through the full-logits `ForwardN`
  (`TestGenerateNgramSpeculativeAdaptive_costOnlyWhenGreedy`).

**Correctness before timing:**
- `TestSpecNgram_multiTurnMatchesPlain`: two chat turns identical to plain decode, and identical K/V after turn 0.
- `TestSpecNgram_copyOnStepVerify`: 192 tokens identical; 27 rounds on the 1.5B (171 of 175 drafts accepted) and 25
  on the 7B.
- Both hold on the 1.5B and the 7B. The tagged Metal suite ran 183 pass, 0 fail; the decoder suite ok; the parity
  manifest is fresh.
- Logs: [`identity-build-1.5b.log`](metal-spec-step-verify-2026-09-27/identity-build-1.5b.log),
  [`identity-build-7b.log`](metal-spec-step-verify-2026-09-27/identity-build-7b.log),
  [`metal-suite-verify.log`](metal-spec-step-verify-2026-09-27/metal-suite-verify.log).

## 3. The served grading

As registered (`8bb80b1f`):
- M1 Pro 16 GB, macOS 26.6.2.
- *plain* and *spec-new* ran on `serve-metal` at `1e153876` (plain's path is unchanged by it). *spec-old* ran at
  `8bb80b1f`, the build's parent.
  - The registration said "one binary", which a *spec-old* arm cannot satisfy; the parent binary is what it means.
- 1 client, greedy. Workloads:
  - the spec record's copy workload (`scripts/bench_spec_copy.py`, ~960-token sections, 256 tokens, 2 requests);
  - W7 chat (6 turns × 128 tokens, `--fixed-nonce`).
- 3 rounds with the arm order rotated (plain specold specnew / specnew plain specold / specold specnew plain),
  idle-gated per cell.
- The 1.5B is graded and the 7B reported. 2026-09-27 17:26–17:47 PDT (00:26–00:47 UTC).

Raw: [`copy-15b.json`](metal-spec-step-verify-2026-09-27/copy-15b.json), [`chat-15b.json`](metal-spec-step-verify-2026-09-27/chat-15b.json),
[`copy-7b.json`](metal-spec-step-verify-2026-09-27/copy-7b.json), [`chat-7b.json`](metal-spec-step-verify-2026-09-27/chat-7b.json),
[`run-served.log`](metal-spec-step-verify-2026-09-27/run-served.log), [`servers.log`](metal-spec-step-verify-2026-09-27/servers.log),
[`run-served.sh`](metal-spec-step-verify-2026-09-27/run-served.sh). Gates by [`gates-served.py`](metal-spec-step-verify-2026-09-27/gates-served.py),
output in [`gates-served-output.txt`](metal-spec-step-verify-2026-09-27/gates-served-output.txt).

| gate (pre-registered) | result |
|---|---|
| 1. every reply equals plain's, every turn (hard) | **pass**: copy 6/6 requests and chat 18/18 turns, on both models and both spec arms |
| 2. copy, 1.5B: spec-new ÷ plain, median ≥ 1.25× | **pass: 2.082×** |
| 3. chat, 1.5B: spec-new ÷ plain ≥ 0.97× (hard) | **pass: 1.068×** |

**Decision, by the registered rule:** all pass, so it ships. `server.md`'s "leave it off on Metal" is rewritten.

**Reading.**
- **Copy-heavy traffic doubles on the 1.5B (109 against 52.7 tok/s)** and rises 1.85× on the 7B. That is in line
  with CUDA's 2.14× on a verbatim-copy prompt.
- **Chat gains too, modestly (1.07× / 1.01×), and never loses.** The cost-curve controller drafts only where
  acceptance can pay the step's fixed overhead. The old verify cost chat 3–4%.
- The 1.5B's round-1 plain chat cell (61.5 tok/s against 76 in rounds 2 and 3) is the median's outlier, recorded and
  not explained. The median of three is unaffected: rounds 2 and 3 alone read 1.066× and 1.068×.
- **Under load nothing changes.** `--spec` still takes the resident exclusively, so with several clients a spec
  server gives up MC3's batching (this morning's L = 0.61× / 0.49×).
  - So `--spec ngram` is now the right choice for one user or an agent, and still the wrong one for shared serving.
  - That makes MC4's "speculate when alone, batch under load" worth building, by the spec measurement's own
    registered rule: S > 1.05× on both workloads, with L < 1.
- P10's block drafters (`docs/spec/08-dspark-dflash.md`) have their Metal verify now: bit-identical, at ~0.11–0.24 of
  a token per extra row. Their drafter port is its own item.

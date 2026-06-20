# Speculative decoding — experiment scoreboard

> Live tracker. Update inline as runs land. Companion to the [design index](./README.md).
> Cut 2026-06-20.

**How to read this.** Acceptance (`α̅`) and correctness are machine-independent —
measure once. Speed (`tok/s`, `c`, `B`, γ\*) is backend-specific — measure on **every
target you ship to** (see [00-core §7](./00-core.md), [06](./06-acceptance-analysis.md)).
That's why every row carries a **Machine** column.

**Status legend:** ⬜ todo · 🟡 running · ✅ benched (gate green + numbers in) · 🔴 failed gate / regressed · ⏸ blocked

**Machines:** `mac` = Apple Silicon (Metal / NEON, on-device target) · `lx` = Linux box (Vulkan WebGPU, 8 GB discrete GPU — bulk-grind workhorse + Vulkan speed target). Add more as needed.
> Division of labor: run gates + heavy sweeps + 05 head on `lx` (GPU); re-measure speed on `mac` (acceptance/correctness transfer, speed does not). 8 GB bounds two-model/head setups on `lx` — 01/02 add no model memory, so prefer them there first.

**Lossless invariant:** every run must reproduce the non-speculative baseline (greedy bit-exact; sampled in-distribution). A 🔴 in *Lossless* blocks everything else for that row.

---

## 0. Foundation gates — verify + rollback per family ([00-core §6](./00-core.md))

Prerequisite. Nothing below is valid until the family's rollback is correctness-gated.
SSM / linear-attention need the state-checkpoint path; softmax/MLA just truncate.

| Family | Rollback kind | Model | Greedy bit-exact | Sampled in-dist | Status |
|---|---|---|---|---|---|
| softmax / GQA | truncate KV | qwen2.5-coder-1.5b | ✅ (`TestSpeculativeGreedyParity`, `TestSpeculativeResident_parity`) | ⬜ | 🟡 greedy done; sampled-rejection path new |
| MLA | truncate latent KV | deepseek-v2-lite / kimi | ⬜ | ⬜ | ⬜ |
| Mamba-2 (SSM) | **state checkpoint** | nemotron-h / granite-4.0-h | ⬜ | ⬜ | ⬜ |
| Gated DeltaNet (linear) | **state checkpoint** | (gated-linear model) | ⬜ | ⬜ | ⬜ |

---

## 1. Spoke scoreboard — acceptance + speed

Rows are (spoke × its home workload). `α̅` = mean per-position acceptance.
`tok/v` = committed tokens per verify. `speedup` = tok/s vs non-spec baseline **on that Machine**.

### 01 — Grammar-fused ([doc](./01-grammar-fused.md)) · home: structured / tool-call

| Model | Workload | Machine | Lossless | α̅ | tok/v | speedup | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-1.5b | structured | mac | ⬜ | – | – | – | ⬜ | |
| qwen2.5-coder-1.5b | structured | lx | ⬜ | – | – | – | ⬜ | |
| gemma-4-E2B | structured | mac | ⬜ | – | – | – | ⬜ | |

### 02 — Cache / n-gram ([doc](./02-cache-ngram.md)) · home: codeedit / rag / agent

> **Landed (lx):** `decoder.NgramDrafter` (prompt-lookup) + `GenerateNgramSpeculative`
> (single-model greedy, rides the existing `ForwardN`/`TruncateTo` verify+rollback) +
> `SpecTrace`/`TraceCollector` instrumentation + the `TestNgramSpecHarness` measurement
> harness. Lossless gate **green** everywhere (`TestNgramSpeculativeGreedyParity` K∈{1,4,8};
> harness asserts == plain greedy per workload). First real numbers below, on
> qwen2.5-coder-0.5b q4 (loaded int8int8), K=8, CPU, 128 tok. Machine-independent
> metrics (α̅, tok/v) hold across backends; the CPU wall-clock speedup does not (the
> batched `ForwardN` amortizes the weight stream — a much bigger win on GPU/`mac`).
> **First §06 dataset dumped**: 308 traced positions → `/tmp/spec_ngram_traces.jsonl`
> (`GINFER_SPECTRACE_OUT`), schema matches §06 §2.

| Model | Workload | Machine | Lossless | α̅ | acc | tok/v | speedup (CPU) | Status |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-0.5b | parity (novel) | lx | ✅ | – | – | – | – | ✅ lossless; no repetition to accept |
| qwen2.5-coder-0.5b | rag-copy | lx | ✅ | 0.973 | 0.967 | 7.11 | **1.32×** | ✅ |
| qwen2.5-coder-0.5b | agent-json | lx | ✅ | 0.955 | 0.933 | 8.53 | **1.43×** | ✅ |
| qwen2.5-coder-0.5b | codeedit | lx | ✅ | 0.738 | 0.352 | 1.86 | 0.81× 🔴 | ✅ lossless, but K=8 too deep → net slowdown |

> **Finding (orders the backlog):** copy-heavy workloads (rag/agent) already win
> ~1.3–1.4× *on CPU* at fixed K=8. `codeedit` is a **slowdown** — α̅=0.74 but realized
> acc=0.35, so the K=8 chain breaks early and the wide `ForwardN` isn't paid back. This
> is the textbook case for **[04 adaptive depth](./04-adaptive-depth.md)**: depth should
> collapse where α̂ is low. The α̅≫acc gap (0.74 vs 0.35) also shows many `codeedit`
> drafts are strong-but-not-top — value that **sampled** rejection (not greedy) would
> capture. Both are now data-backed, not guesses.

| Model | Workload | Machine | Lossless | α̅ | tok/v | speedup | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-1.5b | codeedit | mac | ⬜ | – | – | – | ⬜ | match-len curve |
| qwen2.5-coder-1.5b | rag | mac | ⬜ | – | – | – | ⬜ | |
| qwen2.5-coder-1.5b | agent | lx | ⬜ | – | – | – | ⬜ | warm-KV reuse |

### 04 — Adaptive depth ([doc](./04-adaptive-depth.md)) · vs fixed K=1,2,4,8

| Model | Workload | Machine | Lossless | α̅ | tok/v | speedup | best fixed-K | Status |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-1.5b | codeedit | mac | ⬜ | – | – | – | – | ⬜ |
| qwen2.5-coder-1.5b | chat | mac | ⬜ | – | – | – | – | ⬜ |

### 03 — Router + trees ([doc](./03-router-tree.md)) · grammar+ngram(+head) fused

| Model | Workload | Machine | Lossless | α̅ | tok/v | speedup | build step | Status |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-1.5b | agent | mac | ⬜ | – | – | – | priority→tree→α̂ | ⬜ |
| qwen2.5-coder-1.5b | structured | mac | ⬜ | – | – | – | priority | ⬜ |

### 05 — EAGLE-3 head ([doc](./05-eagle3-head.md)) · home: chat / reasoning

| Model | Workload | Machine | Lossless | α̅ | tok/v | speedup | head source | Status |
|---|---|---|---|---|---|---|---|---|
| (target w/ available head) | reasoning | lx | ⬜ | – | – | – | import? / train? | ⏸ |

---

## 2. Acceptance analysis tracker ([06](./06-acceptance-analysis.md))

Per (model × workload mix). `α̂` is the calibrated predictor; ship-bar = monotone reliability + ECE < ~0.03.

| Model | Workload mix | Traces dumped | α̂ fitted | AUC (held-out) | ECE | Calibrated | q_top1-only AUC | Status |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-1.5b | code+rag+agent | ⬜ | ⬜ | – | – | ⬜ | – | ⬜ |
| qwen2.5-coder-1.5b | chat+reasoning | ⬜ | ⬜ | – | – | ⬜ | – | ⬜ |

---

## 3. Cross-machine speed (because speed isn't portable)

Same (spoke, model, workload) on each target. Confirms γ\* / depth thresholds per box.
Note: `B` (verify width) differs sharply — the `lx` GPU affords a much wider "free"
batch than `mac` CPU SIMD, so the **optimal tree shape (03) and depth (04) differ by
backend**, not just tok/s. Tune the speculation policy per machine.

| Spoke | Model | Workload | Machine | c (draft:target) | B (verify width) | tok/s | speedup | Status |
|---|---|---|---|---|---|---|---|---|
| 02 | qwen2.5-coder-1.5b | codeedit | mac | – | – | – | – | ⬜ |
| 02 | qwen2.5-coder-1.5b | codeedit | lx | – | – | – | – | ⬜ |

---

## 4. Next up — driven by fixable-mass ([06 §7](./06-acceptance-analysis.md))

Re-rank after each analysis pass; build the spoke with the most fixable acceptance mass.

1. ⬜ Foundation gates green for softmax/GQA + MLA (unblocks 01/02).
2. ⬜ 01 grammar-fused on `structured` (cheapest win, goinfer's headline workload).
3. ⬜ 02 n-gram on `codeedit`/`agent` (cheap, generates the first α̂ dataset).
4. ⬜ First α̂ fit + calibration → unblock 04.
5. ⬜ SSM / linear rollback gates (correctness, differentiator).
6. ⬜ 03 router once 2+ sources exist.
7. ⬜ 05 head (decide import vs train first).

---

## Run log (append-only)

| Date | SHA | Machine | What | Result |
|---|---|---|---|---|
| 2026-06-20 | (wip) | lx | 02 n-gram drafter + single-model greedy verify (`spec_ngram.go`) | ✅ lossless gate green (K=1,4,8); `TestNgramDrafter` unit green |
| 2026-06-20 | (wip) | lx | SpecTrace + `TestNgramSpecHarness` measurement harness | ✅ rag 1.32× / agent 1.43× / codeedit 0.81× (CPU, K=8); α̅ 0.74–0.97; 308-row §06 dataset dumped. codeedit slowdown → motivates 04 adaptive depth |

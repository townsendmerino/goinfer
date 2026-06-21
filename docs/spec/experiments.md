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
| softmax / GQA | truncate KV | qwen2.5-coder-1.5b | ✅ (`TestSpeculativeGreedyParity`, `TestSpeculativeResident_parity`) | ✅ (`TestSpecStepLossless` kernel + `TestNgramSampledFirstTokenMatchesPlain`) | ✅ greedy bit-exact + sampled in-dist (point-mass q rejection) |
| MLA | truncate latent KV | deepseek-v2-lite | ✅ (`TestNgramSpecMLA_parity`, real V2-Lite) | ✅ (model-free `TestSpecStepLossless` kernel + same truncate rollback) | ✅ latent KV truncates correctly; spec lossless |
| Mamba-2 (SSM) | **state checkpoint** | nemotron-h / granite-4.0-h | 🛡 guarded-out | 🛡 guarded-out | 🛡 `specRollbackSafe`=false ⇒ spec REFUSED, falls back to plain (`TestSpecRollbackSafetyGuard`). TruncateTo can't roll back the rolling state. Checkpoint/restore not yet built |
| Gated DeltaNet (linear) | **state checkpoint** | qwen3_5_moe | 🛡 guarded-out | 🛡 guarded-out | 🛡 same — guarded-out, no silent-bug risk; checkpoint/restore not yet built |

---

## 1. Spoke scoreboard — acceptance + speed

Rows are (spoke × its home workload). `α̅` = mean per-position acceptance.
`tok/v` = committed tokens per verify. `speedup` = tok/s vs non-spec baseline **on that Machine**.

### 01 — Grammar-fused ([doc](./01-grammar-fused.md)) · home: structured / tool-call

> **Landed (lx):** `constrain.Masker.ForcedRun(max)` + `Grammar.Clone()` (json/schema/
> tool) — non-mutating forced-run extractor, `TestForcedRun` model-free gate.
> **Finding:** goinfer's grammars permit optional whitespace at every structural
> boundary, so strict single-token forcing fires only INSIDE fixed literals (keys,
> enum/const), not at the scaffolding. Inc-2 go/no-go = forced-fraction on real
> tool/schema outputs with a real BPE tokenizer before building the masked-verify
> integration. **Inc-2 measured (`TestGrammarForcedFraction`): strict token-forcing
> = 0% (BPE prefix-ambiguity), but BYTE-level forcing = 44–74% (weather 54 / record
> 44 / enum-heavy 74). GO, but the drafter must be byte-level (forced byte-run →
> canonical retokenize → propose; acceptance <1 via tokenization, verify-lossless).**
> Inc 3 = byte-level `ForcedBytesRun` + masked-verify integration.

| Model | Workload | Machine | Lossless | α̅ | tok/v | speedup | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-1.5b | structured | mac | ⬜ | – | – | – | ⬜ | |
| qwen2.5-coder-0.5b | structured | lx | ✅ `TestGrammarSpecParity` | – | 1.13 | – | ✅ lossless (==constrained greedy); acc 0.40, forced runs fire |
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
>
> **Resolved by 04 (below):** the adaptive controller fixed the `codeedit` slowdown
> (0.79×→0.94×) *and raised* the copy-heavy wins (rag 1.34×→1.55×, agent 1.46×→1.59×)
> by trimming wasteful depth.
>
> **Sampled re-measure (`TestNgramSampledHarness`) — the α̅≫acc lead was a metric
> artifact, and sampling HURTS the spoke.** `acc=Accepted/Drafted` charged the
> untested post-reject tail; the new `EvalAcceptanceRate` (accepted/evaluated)
> confirms **eval-acc ≈ α̅ everywhere** (rag sampled 0.690≈0.684, codeedit greedy
> 0.753≈0.738) — i.e. the rejection math is exactly right, no hidden mass to capture.
> The real finding: under temperature (0.7 + top-p 0.95) **tok/v collapses** (rag
> 7.11→1.80, agent 8.53→7.11) because a *correct* verbatim copy run that greedy
> commits whole is accepted only token-by-token w.p. p(x)<1 under sampling, so the
> chain breaks ~every 1/(1−eval-acc) tokens. **The n-gram win is largely a greedy
> phenomenon**; recovering it under sampling needs tree/multi-candidate verify (03)
> or temperature-aware structural handling — future work, now data-backed.

| Model | Workload | Machine | Lossless | α̅ | tok/v | speedup | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-1.5b | codeedit | mac | ⬜ | – | – | – | ⬜ | match-len curve |
| qwen2.5-coder-1.5b | rag | mac | ⬜ | – | – | – | ⬜ | |
| qwen2.5-coder-1.5b | agent | lx | ⬜ | – | – | – | ⬜ | warm-KV reuse |

### 04 — Adaptive depth ([doc](./04-adaptive-depth.md)) · vs fixed K=8

> **Landed (lx):** `decoder.AdaptiveDepth` + `GenerateNgramSpeculativeAdaptive`. Drives
> per-round depth from a running EMA of realized acceptance: `D = floor(ln Θ / ln α)`
> clamped to `[0, K]`, Θ = marginal verify-node cost (~0.5 on this batched-CPU path),
> with a periodic probe to climb back out of D=0. Lossless gate green
> (`TestNgramAdaptiveGreedyParity`, exercises the D=0/probe path). qwen2.5-coder-0.5b
> q4, K(max)=8, CPU, 128 tok, vs **same-stream fixed K=8** and plain greedy:

| Model | Workload | Machine | Lossless | adaptive speedup | fixed-K=8 | tok/v (ada) | final α |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder-0.5b | codeedit | lx | ✅ | **0.94×** (was 0.79×) | 0.79× | 1.58 | 0.566 |
| qwen2.5-coder-0.5b | rag-copy | lx | ✅ | **1.55×** | 1.34× | 7.11 | 0.995 |
| qwen2.5-coder-0.5b | agent-json | lx | ✅ | **1.59×** | 1.46× | 6.74 | 0.989 |

> Adaptive dominates fixed K=8 on every workload — it removes the over-draft tax
> (codeedit) and trims wasteful rounds on the wins. A full Θ-sweep + the K∈{1,2,4}
> baselines + the `mac` re-measure remain.
>
> **GPU-resident re-measure (lx, RTX 2070 SUPER, qwen2.5-coder-0.5b webgpu,
> `TestNgramSpecResident_throughput`) — the win does NOT transfer to GPU yet:**
>
> | workload | plain tok/s | fixed K=8 | adaptive | tok/v (fixed) |
> |---|---|---|---|---|
> | codeedit | 78.2 | 0.60× 🔴 | 0.90× | 1.98 |
> | rag-copy | 105.7 | 1.00× | 1.03× | 7.62 |
> | agent-json | 115.6 | 0.96× | 0.96× | 8.42 |
>
> **Why (reverses the earlier "GPU should win bigger" guess):** the resident
> `ForwardN` is **not** an M=K batched GEMM. `gpu/decoderunner.go:runBatch` records K
> separate **M=1** runners into one command pass with a single Submit/Poll — it
> amortizes only the sync, not the per-token weight stream (each runner re-reads all
> weights). So committing 7.6 tok/verify still costs ~7.6 forwards of GPU compute →
> ~1.0×. This is the **same Stage-A wall the draft-model Lever 2 hit**
> ([memory: GPU spec-decode Lever 2]); the n-gram win is real only where `ForwardN`
> truly batches the weight read — the **CPU batched-arch path** (rag 1.55× above).
> Unlocking GPU needs **Stage B: a real M=K GEMM verify** (one weight stream across
> all K positions). Adaptive depth is what keeps GPU *safe* meanwhile (codeedit
> 0.60×→0.90×); note Θ should be **higher** (more conservative) on GPU, not lower —
> deep verify isn't cheaper there.

### 03 — Router + trees ([doc](./03-router-tree.md)) · grammar+ngram(+head) fused

| Model | Workload | Machine | Lossless | α̅ | tok/v | speedup | build step | Status |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder-0.5b | agent-loop (repeat tool-call) | lx | ✅ `TestGrammarRouterSpec` | – | 5.67 | – | confidence (inc 3) | ✅ confidence-router 5.67 (inc-1 priority 4.25, grammar-only 1.13), lossless |
| qwen2.5-coder-0.5b | structured (prose→schema) | lx | ✅ | – | 1.13 | – | priority | ✅ router=grammar-only (n-gram can't echo prose context) |
| qwen2.5-coder-1.5b | agent | mac | ⬜ | – | – | – | tree→α̂ | ⬜ |

### 05 — EAGLE-3 head ([doc](./05-eagle3-head.md)) · home: chat / reasoning

| Model | Workload | Machine | Lossless | α̅ | tok/v | speedup | head source | Status |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B + AngelSlim/Qwen3-1.7B_eagle3 | novel text | lx | ✅ `TestEagleSpecParity` (==greedy) | – | 1.60 | – | IMPORT, SHIPPED | ✅ end-to-end LOSSLESS EAGLE spec decode; 1.60 tok/verify. Modest (greedy single-chain); trees+sampled → published ~3-4× is future |

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
**Headline so far: speed is NOT portable here, and not in the direction expected.**
The CPU batched-arch `forwardN` is a true M-batched GEMM (one weight read for the
whole verify) → n-gram spec wins (rag 1.55×). The GPU resident `ForwardN` is K
M=1 dispatches sharing one sync (`runBatch`, Stage A) → weights re-read per token →
~1.0× even at tok/v 7.6. So today the spoke wins on **CPU**, breaks even on **GPU**;
the GPU win is gated on Stage B (M=K GEMM verify). Adaptive depth (04) is the per-box
safety: it prevents the fixed-K GPU slowdown on low-acceptance streams.

| Spoke | Model | Workload | Machine | mode | tok/v | tok/s | speedup | Status |
|---|---|---|---|---|---|---|---|---|
| 02 | qwen2.5-coder-0.5b | rag-copy | lx-cpu | adaptive | 7.11 | – | **1.55×** | ✅ |
| 02 | qwen2.5-coder-0.5b | rag-copy | lx-gpu | fixed K=8 | 7.62 | 105.9 | 1.00× | 🔴 Stage-A wall |
| 02 | qwen2.5-coder-0.5b | agent-json | lx-cpu | adaptive | 6.74 | – | **1.59×** | ✅ |
| 02 | qwen2.5-coder-0.5b | agent-json | lx-gpu | adaptive | 7.27 | 111.5 | 0.96× | 🔴 Stage-A wall |
| 02 | qwen2.5-coder-0.5b | codeedit | lx-gpu | adaptive | 1.29 | 70.3 | 0.90× | adaptive averts 0.60× fixed |
| 02 | qwen2.5-coder-1.5b | codeedit | mac | – | – | – | – | ⬜ |

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
| 2026-06-20 | (wip) | lx | 04 `AdaptiveDepth` + `GenerateNgramSpeculativeAdaptive` | ✅ codeedit 0.79×→0.94× (slowdown fixed), rag 1.34×→1.55×, agent 1.46×→1.59×; lossless (`TestNgramAdaptiveGreedyParity`). Adaptive dominates fixed K=8 everywhere |
| 2026-06-20 | (wip) | lx-gpu | GPU-resident re-measure (`TestNgramSpecResident_throughput`) | 🔴 win does NOT transfer: rag 1.00×, agent 0.96×, codeedit 0.90× (adaptive). Cause: resident `ForwardN`=K M=1 dispatches/1 sync (Stage-A wall), weights re-read per token. Parity green. Needs Stage B (M=K GEMM verify). Reverses "GPU wins bigger" guess |
| 2026-06-20 | (wip) | lx | Sampled rejection sampling (`spec_sample.go`) | ✅ lossless under temperature + top-k/p/min-p: point-mass q ⇒ accept p(x), residual correction. `TestSpecStepLossless` (model-free, emitted dist==p exactly), first-token==plain, penalties/bias rejected. softmax/GQA sampled-in-dist gate now green |
| 2026-06-20 | (wip) | lx | Session integration + `cmd/serve --spec ngram` | ✅ `Session.GenerateNgramSpeculative[Adaptive]` reuses warm-KV prefix (`genNgramInto`, mirrors `generateInto`); serve flag routes through it, auto-falls-back per-request for penalties/bias/tools. `TestSessionNgramSpecParity` (incl. 2-turn prefix reuse) + serve e2e diff == plain (lossless). Ships the CPU win to users |
| 2026-06-20 | (wip) | lx | Penalty/bias threading for sampled (`distVectorHist`) | ✅ sampled spec now lossless WITH repetition/presence/frequency penalties + logit bias: per-position history threaded (prompt+committed+cur+draft[:i]). `TestDistVectorHistMatchesSampler` (model-free: distVectorHist == real sampler draw dist over 300k). Only constrained/tool decoding (LogitProcessor) still falls back. Serve now speculates on penalty-using OpenAI traffic |
| 2026-06-20 | (wip) | lx | Sampled re-measure + `SpecStats.Evaluated`/`EvalAcceptanceRate` | 🔎 eval-acc ≈ α̅ everywhere (rejection math confirmed; the α̅≫acc "gap" was a Drafted-tail artifact). Real finding: sampling HURTS tok/v (rag 7.11→1.80) — probabilistic rejection shreds verbatim copy runs greedy commits whole. n-gram win is largely greedy; sampled recovery needs trees (03). `TestNgramSampledHarness` |
| 2026-06-20 | (wip) | lx | Stage B scoping (docs/spec/07) | 🔎 De-risked: the M=K W8A8 GEMM kernel ALREADY exists + is parity-gated (`gpu/gemm.go` `MatmulW8A8Tiled`/`BatchTiled`, `TestTiled_parity`), used by staged prefill but NOT the resident decode runner. Stage B = wire it into the resident verify (replace `runBatch`'s K M=1 GEMVs), not a new kernel. Remaining work = hot-path runner surgery; design + increments in 07 |
| 2026-06-20 | (wip) | lx-gpu | Stage B increment 1: `DecodeTokenFusedBatched` (dense W8A8) | ✅ batched M=K verify forward — projections as ONE tiled GEMM (weights streamed once), per-row rms/rope/attn/swiglu/residual, all-rows-KV-write-then-all-rows-attend. `TestDecodeTokenFusedBatched_parity` BIT-EXACT vs M sequential DecodeTokenFused (cosine 1.0, maxAbs 0.0, 5 rows incl. intra-block causal). Next: increment 2 — wire into residentDecoder.ForwardN + throughput re-measure |
| 2026-06-20 | (wip) | lx-gpu | Stage B increment 2: batched-vs-per-row microbench | 🔴 KILL-GATE: batched **0.88×** (slower) vs per-row×M at qwen2.5-0.5b dims, M=8, 24L (`TestDecodeTokenFusedBatched_microbench`). The 16×16 tiled GEMM is the PREFILL kernel — wastes half its M rows at M=8; per-row GEMV is already bandwidth-optimal. Existing kernel can't deliver Stage B; production wiring deferred. **Inc 3 = thin-M kernel (workgroup-per-column multi-row GEMV, weight read once, M accumulators)** then re-measure |
| 2026-06-20 | (wip) | lx-gpu | Stage B increment 3: thin-M kernel (`gemmRowW8A8`) | 🔴 **Stage B NO-GO.** Thin-M multi-row GEMV (one workgroup/column, weight read once, M accumulators) — bit-exact parity holds; microbench 0.88×→**0.98×** (kernel choice mattered, still ≤1×). Projection-batching saving cancelled by per-row attn/rms/swiglu + multi-row M× ALU/load + gather-scatter glue; worse vs real `runBatch`. Even with free n-gram draft + right kernel, GPU verify ~break-even at M=8 → CONFIRMS prior deferral. CPU `--spec ngram` win stands; resident verify stays Stage A. Spike code kept + gated (see 07 conclusion) |
| 2026-06-20 | (wip) | lx | Foundation: spec rollback-safety guard + MLA gate | ✅ `specRollbackSafe` REFUSES recurrent families (Mamba-2 granite/nemotron, DeltaNet qwen3_5_moe) — closes a silent rollback bug (`--spec` was unguarded); caller falls back to plain. `TestSpecRollbackSafetyGuard` (model-free). MLA gate flipped GREEN: `TestNgramSpecMLA_parity` bit-exact on real DeepSeek-V2-Lite (truncate latent rollback correct). softmax/GQA + MLA now spec-safe; SSM/DeltaNet need checkpoint/restore (unbuilt)
| 2026-06-20 | (wip) | lx | 01 grammar-fused inc 1: `Masker.ForcedRun` + `Grammar.Clone` | ✅ forced-run extractor (non-mutating, via Clone) + Clone on all 3 grammars; `TestForcedRun` model-free. Finding: grammars allow optional ws everywhere ⇒ strict forcing fires only inside fixed literals (keys/enum), not scaffolding. Inc 2 = forced-fraction on real BPE tokenizer (go/no-go) before masked-verify integration |
| 2026-06-20 | (wip) | lx | 01 grammar-fused inc 2: forced-fraction (go/no-go) | 🔎 `TestGrammarForcedFraction` real BPE: strict TOKEN-forcing 0% (prefix-ambiguity defeats it), BYTE-forcing ceiling 44–74%. GO but redesign byte-level: forced byte-run→canonical retokenize→propose (acceptance<1 via tokenization freedom, verify-lossless). Inc 3 = byte-level drafter + masked-verify integration |
| 2026-06-20 | (wip) | lx | 01 grammar-fused inc 3: byte-level drafter + masked verify | ✅ END-TO-END: `ForcedBytesRun` + `GrammarDrafter` + `GenerateGrammarSpeculative` (CPU greedy). Masked verify with grammar clone rolled over accepted prefix; grammar advances only over emitted tokens (no rewind). `TestGrammarSpecParity` token-identical to constrained Generate; acc 0.40, 1.13 tok/round on weather JSON (forced runs fire, lossless). Modest on tiny JSON, scales w/ structural fraction. Fixed: tpos wasn't advancing (KV corruption). Follow-ups: sampled/resident, serve wiring |
| 2026-06-20 | (wip) | lx | 01 grammar-fused inc 4: density characterization | 🔎 `TestGrammarSpecHarness` tok/round by schema density: free-string 1.10 / mixed 1.11 / enum-heavy(2-choice) 1.05 / fixed-keys(single-value) 1.45. Real lossless zero-draft win but MODEST (~1.05–1.45 tok/round), best only when keys+values fully fixed; multi-choice enums force less (choice breaks the run). Wire into serve only if cheap; not worth heavy investment |
| 2026-06-20 | (wip) | lx | 03 router inc 1: priority router (grammar+ngram) | ✅ `RouterDrafter` + pluggable-drafter masked verify. Lossless (`TestGrammarRouterSpec` == constrained Generate). FIRST fusion win: agent-loop (repeat tool-call) 4.25 tok/round (acc 0.82) vs 1.13 grammar-only; generic prose→schema 1.13 (n-gram can't echo prose context). Inc 2 = tree verify |
| 2026-06-21 | (wip) | lx | 03 router inc 2: tree verification (go/no-go) | 🔴 DEFERRED (kill-gate). `TestTreeUpside`: 17.6% positions tree-recoverable on agent-loop, BUT it's priority-order suboptimality not a missing tree — grammar 2/5 correct (tokenization) vs n-gram 15/15; n-gram-FIRST 15/17 vs grammar-first 12/17 captures ALL of it by reordering. Tree's unique residual value ~0. Tree-mask forward (hardest build: non-contiguous attn mask × CPU/resident/int8/ring) NOT worth it. Inc 3 (α̂/confidence source selection) captures the win cheaply |
| 2026-06-21 | (wip) | lx | 03 router inc 3: confidence-driven source selection | ✅ `RouterDrafter` takes most-confident source (n-gram match-len vs grammar const `grammarConf`). Captures the inc-2 tree upside cheaply: agent-loop 4.25→**5.67 tok/round** (acc 0.94), generic prose→schema 1.13 (NO regression). Lossless (`TestGrammarRouterSpec`, both cases). Tree-mask forward correctly skipped. Follow-up: swap heuristic for §06 α̂ |
| 2026-06-21 | (wip) | lx | 05 EAGLE-3 feasibility (research) | 🟡 IMPORT decision: AngelSlim/Qwen3-4B_eagle3 (1-layer, hidden 2560, draft-vocab 32000, t2d/d2t; license Apache/MIT/CC-BY ✓ attribution). Base Qwen3 dense = goinfer-runnable; general head accelerates code too. No qwen2.5-coder head exists; code heads only for Qwen3-Coder-Next (DeltaNet, guarded-out, FP8). Reframe: head=Drafter ⇒ lossless verify protects correctness, imperfect head only lowers acceptance. Risk=protocol (fused layers/fusion/autoregress/t2d-d2t) lives in vLLM/SGLang/SpecForge code. Plan: seam→loader→forward→draft→measure. Multi-session
| 2026-06-21 | (wip) | lx | 05 inc 1: hidden-state seam (`Model.ForwardCapture`) | ✅ read-only export of residual stream after configurable layers (EAGLE-3 fuses low/mid/high). `TestForwardCaptureSeam`: logits byte-identical to forward + last-layer capture reproduces logits via final-norm+head (exact). Zero overhead when unused. Generic decode path (special families return an error). Next: head loader |
| 2026-06-21 | (wip) | lx | 05 inc 2: EAGLE head loader (`LoadEagleHead`) | ✅ loads AngelSlim/Qwen3-1.7B_eagle3 (converted .bin→f32 safetensors): fc[2048,6144]=3·hidden fusion, attn q/k/v from 2·hidden, lm_head[32000] draft-vocab, d2t[32000]→target. `TestLoadEagleHead` gates shapes. Base qwen3-1.7b-q8_0.gguf local. Next: head forward (inc 3) |
| 2026-06-21 | (wip) | lx | 05 inc 4 + tuning: autoregressive draft, bf16 base, layer sweep | 🟢 pipeline complete + working: context-prefill lifted accepted-len 0.26→0.64 (~1.64 tok/verify greedy). bf16 base NO change vs q8 (0.64 vs 0.62) → not precision; layer agreement plateaus 38-41% (not layers). Gap to ~3-4× = EAGLE uses TREES + sampled-accept vs my greedy single-chain. Inc5: lossless `GenerateEagleSpeculative` (needs forwardN-capture) + trees |
| 2026-06-21 | (wip) | lx | 05 inc 5: `GenerateEagleSpeculative` (end-to-end LOSSLESS) | ✅ forwardN hidden-capture (batched seam) + head-as-drafter verify loop. `TestEagleSpecParity`: token-identical to greedy, 1.60 tok/verify. The full EAGLE-3 pipeline (seam→loader→forward→context-prefill→autoregress→batched-verify→commit) works lossless in pure Go. Higher acceptance = trees+sampled (future) |
| 2026-06-21 | (wip) | lx | 05 trees+sampled — kill-gate measurement | 🔎 Measured EAGLE per-position α=1−TV(p,q) BEFORE building trees: sampled DOUBLES accept (greedy 0.13→α 0.26-0.29) — real lever. BUT α plateaus ~0.3 (peak {1,12,26}=0.316) far below published ~0.8, and it's NOT base precision (bf16 0.29 vs q8 0.26) NOR capture layers (sweep plateaus ~0.3). Bottleneck = head↔target distribution match, a subtle FORWARD-PROTOCOL detail needing exact Python-reference parity to fix. Trees+sampled build on α; gated on raising it first. `TestEagleAlpha` |

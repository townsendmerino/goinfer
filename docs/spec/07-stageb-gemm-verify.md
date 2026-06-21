# 07 — Stage B: M=K GEMM verify on the resident path

> Status: **design / not built**. Cut 2026-06-20. Prereq for any GPU speculative
> win (the CPU win already ships via `--spec ngram`). Scope sharpened by a codebase
> finding below — Stage B is **runner wiring, not a new kernel**.

## Why

The GPU re-measure ([experiments §3](./experiments.md)) showed n-gram speculation
flat at ~1.0× on the resident path even at tok/v 7.6, because the resident verify
re-streams the weights once **per draft row**:

- `gpu/decoderunner.go:runBatch` records K separate **M=1** runners into one command
  pass with a single Submit/Poll — it amortizes the *sync*, not the per-token weight
  stream. Each of the K rows re-reads all model weights through the per-token
  `gemvW8A8` GEMV. So committing K tokens costs ~K× the GPU compute of one → no win.

The fix (Leviathan/§00-core §2 economics): make the verify's **projection** matmuls
M=K GEMMs so each weight streams **once** across all K verify rows. Attention / RoPE
/ KV-store touch no model weights and stay per-row.

## The finding that sharpens scope: the kernel already exists

The batched W8A8 GEMM primitive is **already written and parity-gated** — it is the
staged-**prefill** kernel, not yet used on the resident decode path:

- `gpu/gemm.go`: `matmulTiledW8A8ShaderWGSL` (16×16 shared-memory tiled int8 GEMM,
  math identical to the GEMV), `MatmulW8A8Tiled(aq, aScales, rm, M)`, and
  `BatchTiled(aq, aScales, M, rms)` (shared-activation qkv / gate-up, one submit).
- Gated: `TestTiled_parity`, `TestBatchTiled_parity` (bit-parity vs the integer ref).
- Used only in `gpu/backend.go` (the staged/prefill GPU path), **not** in the
  resident `DecodeRunner`.

So Stage B does **not** need a new kernel (a fresh `gemmW8A8` would duplicate this).
It needs to route the resident verify's projections through the existing tiled GEMM.
This removes the kernel-correctness risk; the remaining risk is all in the runner.

## Scope — what changes, what doesn't

Per the original Lever-2 assessment ("projection-GEMM only; attention/RoPE/kv-store
touch no model weights and stay per-row, so NO new M=K attention shader is needed"):

| Resident verify component | Today (Stage A) | Stage B |
|---|---|---|
| q/k/v/o, gate/up/down, lm_head projections | K × M=1 `gemvW8A8` | **1 × M=K tiled GEMM** (weights streamed once) |
| RoPE, QK·softmax·V, KV append | per-row | per-row (unchanged) |
| residual / norm / activation | per-row elementwise | per-row (unchanged) |

The verify is a hybrid: batched projections feeding per-row attention feeding batched
projections. The K rows of a draft block are at consecutive positions, so each row's
attention reads its own causal prefix from the (shared) resident KV.

## Build increments (each independently parity-gated)

1. **Resident batched-projection seam.** A path that, given K stacked activation
   rows, runs a layer's projections via `BatchTiled` into K output rows, with the
   per-row attention in between. Start with **one dense arch** (qwen2 — the bench
   model) behind a flag; leave `runBatch` (Stage A) as the default and the fallback.
2. **Parity gate.** Resident Stage-B verify output == resident Stage-A verify output,
   bit-exact (greedy) on the existing spec prompts — reuses `TestSpeculativeResident_parity`
   shape. This is the hard gate; a wrong row/position mapping is a silent bug.
3. **Throughput re-measure.** `TestNgramSpecResident_throughput` with Stage B on:
   target ≥1.3× on rag/agent (the ceiling the Stage-A wall blocked). If it misses,
   the win was overestimated — report and stop.
4. **Per-arch rollout.** Extend past dense to the families the resident path covers
   (GQA, MLA, MoE, SSM…). Each is a separate parity gate. MoE/SSM verify may not be
   worth it; decide per-arch with numbers.

## Increment results

- **Inc 1 (done, `b7e873b`):** `DecodeTokenFusedBatched` — batched M=K verify forward,
  projections via the existing 16×16 tiled GEMM. **Bit-exact** vs M sequential
  (`TestDecodeTokenFusedBatched_parity`, cosine 1.0 / maxAbs 0.0). Correctness proven.
- **Inc 2 (done — KILL-GATE FIRED):** microbench at qwen2.5-0.5b dims, M=8, 24 layers:
  batched **0.88×** (314 ms) vs per-row×M (278 ms) — *slower*. And the batched side is
  *flattered* (1 submit vs M sequential submits), so vs the real `runBatch` (also 1
  submit) it is worse still. Cause = the §risks "thin-M" item: the **16×16 tiled GEMM
  wastes half its M rows at M=8** (it is the *prefill* kernel, M≫16), and the per-row
  GEMV is already bandwidth-optimal. **The existing kernel cannot deliver Stage B.**
  Production wiring into `ForwardN` is therefore NOT worth doing with this kernel.
- **Inc 3 (done — thin-M kernel built, still no win):** `gpu/gemm_rows.go`
  `gemmRowW8A8` — one workgroup per output column, each weight vec4 read ONCE and
  accumulated against all M rows (M register accumulators, no wasted tile dim).
  Bit-exact (the batched parity gate still passes). Microbench: batched
  **0.98×** (was 0.88× with the tiled kernel) — the kernel choice mattered, but it is
  **still ≤ 1×**. The projection-batching saving (weights 1× vs M×) is cancelled by
  (a) per-row attention/rms/swiglu/quant — unchanged by batching, (b) the multi-row
  kernel's M× ALU per weight load (compute-bound, not bandwidth-bound at M=8), and
  (c) the gather/scatter copies. Versus the real `runBatch` (1 submit) it is worse.

## Conclusion (2026-06-20): Stage B is marginal — NO-GO, do not wire into production

Even with a **free** n-gram draft (removing the draft cost that sank the original
Lever-2 Stage B) **and** a thin-M kernel built specifically for the verify block,
the batched verify is ~break-even (0.98×) at M=8 on the 2070S. The resident decode
at these dims is not dominated by projection weight-streaming to a degree that
batching K≈8 rows recovers, once the per-row attention/elementwise and the
quant/gather/scatter glue are counted. This **confirms and strengthens the prior
deferral**: the GPU speculative win is not there. The CPU win (shipped via
`--spec ngram`) stands; the resident verify stays Stage A (`runBatch`).

The Inc-1/3 code (`DecodeTokenFusedBatched`, `gemmRowW8A8`) is kept as the gated
Stage-B spike — correct and reusable if a future, much-larger model (projection-
bound, big hidden/inter) or a non-square thin-M tile changes the arithmetic. Don't
fund the hot-path multi-arch runner surgery on this evidence. Possible (unfunded)
levers if revisited: quantize-into-combined-buffer + offset-bind reads to delete the
gather/scatter; measure at larger M and larger hidden; profile bandwidth vs ALU.



## Large-dim re-measure (2026-06-21): amortization IS there for BIG models + SHORT drafts

Ran the unfunded lever the conclusion flagged ("measure at larger M and larger hidden")
— `gpu/decodetoken_largedim_bench_test.go` (`TestDecodeTokenFusedBatched_largedim`),
batched-vs-per-row×M on the 2070S, synthetic layers (L small to fit 8 GB):

Dim ladder @ M=8: qwen0.5b **0.79–0.86×** · llama7b **0.98–1.09×** · llama13b
**1.02–1.04×** (MHA: 40 KV heads ⇒ per-row attention dilutes the projection win) ·
llama70b-layer **1.18–1.37×**. So the bandwidth-bound amortization **does appear** as
hidden/inter grow — Stage B is NOT dead for large models.

M-sweep @ 70B dims (h=8192, inter=28672): **M=4 1.58× · M=8 1.28× · M=16 1.18× ·
M=32 1.13× · M=48 1.12×**. The amortization is **best at SMALL M and DECAYS with M** —
the multi-row kernel is compute-bound (M× ALU per weight load), so wide batches lose
the win. This is the key strategic flip:

- **Trees are the WRONG structure for Stage B.** A big EAGLE tree (M=32–48 nodes) gets
  only ~1.1× — the per-node ALU swamps the weight-reuse saving. Trees help CPU
  tok/verify but kill GPU verify amortization.
- **Short LINEAR drafts (K≈4) are right.** At 70B, M=4 = 1.58×. With a GPU-resident
  drafter (head draft cost negligible vs a 70B target decode) and chat-regime
  acceptance (~2 of 4), end-to-end ≈ (a+1)/(M/1.58): a=2 → ~1.2×, a=3 → ~1.58×. A
  real, modest win.

**Refined verdict:** Stage B is **conditional-GO for large (≈70B-class) models with a
short, GPU-resident drafter**, NO-GO for small models and for trees. The remaining
blockers are real: (1) a 70B model needs a >8 GB GPU to run resident (the test box
can't — measured on synthetic single layers), (2) the GPU-resident drafter must make
the draft ~free, (3) the high-risk multi-arch runner surgery still stands. So: not dead,
but it only pays in a regime this hardware can't host — park it behind "have a big GPU +
a large model we want to serve fast", then it's worth the build.

## Risks

- **Hot-path, multi-arch surgery** in the resident `DecodeRunner` (its fused `steps`
  model) — the highest-risk change in the program. Gate hard, arch by arch, keep
  Stage A as the default fallback until each arch's gate is green.
- **Attention interleave**: the K rows must each attend to the correct causal prefix;
  an off-by-one in the position/row mapping is a silent distribution bug, not a crash.
- **Tile efficiency at small M**: K≈8 is a thin GEMM (M=8 vs N=thousands); the 16×16
  tile wastes half its M dimension. Measure — a non-square tile (e.g. 8×N) may help,
  but only after correctness.
- **Marginal-payoff honesty**: the Lever-2 kill-gate ethos applies — if Stage B + free
  n-gram draft still misses 1.3× broadly on real prompts, stop and say so.

## Validation plan

- Correctness: Stage-B resident verify ≡ Stage-A resident verify ≡ plain greedy,
  bit-exact, per arch (greedy) + sampled in-distribution.
- Speed: `TestNgramSpecResident_throughput` Stage-A vs Stage-B, per workload, on `lx`
  (Vulkan) and re-measured on `mac` (Metal) — `B`/tile efficiency differ by backend.

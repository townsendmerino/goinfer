# Metal decode decomposition — where a decode token's time goes, by depth (S0, 2026-09-25)

**Question.** Metal greedy decode is 0.58–0.61× Ollama's at depth 3900 (`peer-claim-2026-09-25.md` cell g) and
~1.17× slower per token at depth 128 on the 1.5B / 7B. Before designing anything: which kernels carry the depth term,
and which carry the fixed per-token cost?

**Answer.** Two separate problems, cleanly separable:
- **Depth is the attention kernel.** Its in-sequence cost grows 1.89 / 2.11 / 6.30 ms per 1,000 keys (0.5B / 1.5B /
  7B) — 36–61% of the token at 3900 — reading the KV cache at an effective **~6–13 GB/s** on a ~200 GB/s machine.
- **The fixed cost is the GEMVs** — 10.4 ms per token on the 1.5B, 41.8 ms on the 7B (more than Ollama's whole 7B
  token, 39.2 ms), running the weights at ~62–99 GB/s. The pure dispatch floor is small: 0.65–1.1 ms, 2–12%.

## Provenance

M1 Pro 16 GB, macOS 26.6.2; goinfer `78503526` plus this record's test file and the context-pin fix (uncommitted at
run time; no production file differs). W4A8 decode path, `attention_fa` on (default, engages at ≥ 1536 keys for head
dim 128). Models from `~/models`: qwen2.5-coder-1.5b and -0.5b q4_k_m (`.gguf`, via their sidecars), qwen2.5-7b
q4_k_m from its `.int4.metal.giw` sidecar (aliased). Resident context pinned to 4096. 2026-09-25 16:43:47–16:48:25
local; idle at start. Raw: [`metal-decode-decomp-2026-09-25/run.log`](metal-decode-decomp-2026-09-25/run.log).

## Method — no replica

`TestMetalDecodeDecomp` (`metal/decode_decomp_test.go`) times the **production** `Forward` — its own command buffer's
GPU timestamps — while one category's pipelines are swapped for a no-op kernel (same dispatches, same grids; only
that category's work disappears). A category's in-sequence work = full − full-with-it-no-op'd; the dispatch floor =
the token with every pipeline no-op'd. The KV cache is filled to each depth by `PrefillLast`, then decode repeats at
that position, so the depth is constant. Arms interleave back to back (5 reps × 20 tokens each), and after the no-op
arms a decode step must reproduce the reference logits bit for bit (it did at every depth). **The categories plus the
floor sum to 98.0–101.5% of the full token in all nine cells** — additive.

## Result (ms per token, medians; share of the token)

| | 1.5B @128 | @2048 | @3900 | 7B @128 | @2048 | @3900 | 0.5B @128 | @2048 | @3900 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **attention** | 0.67 (5%) | 4.98 (29%) | **8.62 (41%)** | 0.78 (2%) | 13.47 (23%) | **24.55 (36%)** | 0.48 (9%) | 4.09 (46%) | **7.61 (61%)** |
| GEMV gate/up | 4.66 | 4.68 | 4.70 | 19.42 | 19.41 | 19.22 | 1.25 | 1.37 | 1.42 |
| GEMV down (+res) | 3.06 | 3.15 | 3.14 | 14.19 | 14.21 | 14.35 | 0.87 | 0.87 | 0.93 |
| GEMV qkv (+bias) | 0.69 | 0.69 | 0.67 | 2.72 | 2.73 | 2.74 | 0.23 | 0.23 | 0.24 |
| GEMV o (+res) | 0.55 | 0.55 | 0.57 | 2.16 | 2.17 | 2.28 | 0.20 | 0.15 | 0.19 |
| LM head (int8) | 1.33 | 1.32 | 1.34 | 3.21 | 3.25 | 3.38 | 0.75 | 0.63 | 0.69 |
| norm+quant | 0.49 | 0.51 | 0.49 | 0.76 | 0.84 | 0.88 | 0.29 | 0.33 | 0.34 |
| rope, kv store, act/ctx quant | 0.63 | 0.62 | 0.71 | 1.01 | 0.95 | 1.18 | 0.42 | 0.39 | 0.41 |
| dispatch floor | 0.86 | 0.89 | 0.85 | 1.06 | 1.11 | 1.12 | 0.65 | 0.63 | 0.64 |
| **full token** | **12.90** | **17.21** | **20.91** | **45.51** | **58.18** | **68.65** | **5.22** | **8.87** | **12.40** |

Full-token spreads 0.2–1.4%. Two cells have one noisy paired value each (7B norm+quant and the small-kernel group at
128 and 3900, single reps near or below zero); their medians are stable.

## What it implies

- **Attention is the depth term, and it is the kernel, not memory bandwidth.** Per 1,000 keys it costs 2.11 ms on
  the 1.5B against Ollama's whole-token depth slope of 0.37 (cell g) — 5.7× — and reads KV at ~13 GB/s. The per-key
  figures agree with cell g's end-to-end marginals (2.07 / 6.35 / 1.89 ms), measured independently. The 0.5B never
  reaches `attention_fa` (head dim 64) and is worst per byte (~6 GB/s) on the old per-query-head kernel.
- **Attention alone nearly closes the 1.5B at depth; the 7B needs both.** Projected, not measured: at Ollama's
  per-key cost, goinfer's 1.5B token at 3900 would be ~13.8 ms (≈ 0.95× Ollama's 13.06), and the 7B ~45.5 ms (≈ 0.93×
  of 42.7) — because the 7B's GEMVs alone (41.8 ms) exceed Ollama's whole token at 128.
- **The short-context gap is GEMV efficiency, not dispatch.** At 128 the GEMVs are 80% of the 1.5B token and 92% of
  the 7B's; the no-op dispatch floor is ≤ 1.1 ms. (That floor is launch cost only; the dependency serialization
  between real kernels is inside each category's work, not separated here.)

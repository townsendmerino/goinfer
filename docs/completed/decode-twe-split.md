# Decode TWrite/TEncode/TSync split — §2 (widen eligibility) vs §3 (spine fusion)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


Measurement task on RTX 2070 SUPER / Vulkan / cogentcore wgpu, `-tags gpu`. Drives the
DecodeRunner's `TWrite`/`TEncode`/`TSync` instrumentation on **real resident decode**
(`decoder.Load` → `ResidentForwardForTest` → `Forward` per token) and the staged fallback
(`GOINFER_NO_RESIDENCY=1`) to size both funding levers. No production decode change; harness
is `gpu/decode_twe_split_test.go` (resident split + RunN) and `gpu/decode_staged_prize_test.go`
(§2 prize). Run with `-run TestDecodeTWE_split` / `-run TestDecodeStaged_prize`.

`TWrite` = per-token input + pos-uniform uploads. `TEncode` = command record (the alloc-free
535-dispatch plan: 535× cgo SetPipeline/SetBindGroup/Dispatch). `TSync` = Submit + MapAsync +
blocking Poll = CPU-blocked-on-GPU wall.

## Resident path — per-token split (400 steady-state tokens, median + p10/p90)

| model | TWrite | TEncode | TSync | total (median) | best (throttle-free) |
|---|---|---|---|---|---|
| Qwen2.5-coder-1.5B | 20 µs (0%) | 905 µs (7%) | 12 441 µs (**95%**) | 13 070 µs / 76 tok/s | 8 393 µs / **119 tok/s** |
| Qwen3-1.7B | 20 µs (0%) | 1 015 µs (7%) | 13 161 µs (**92%**) | 14 243 µs / 70 tok/s | 8 446 µs / **118 tok/s** |

spread (Qwen2.5): TSync p10 8 382 → p90 16 856 µs. The CPU parts (TWrite+TEncode ≈ 0.93 ms)
are rock-steady token-to-token; **only TSync swings 2×** → the spread is sustained-load GPU
clock throttling over a 400-token run, not CPU. Best-case (throttle-free) per-token ≈ 8.4 ms.

**Device-timestamp GPU portion:** the resident device isn't created with the timestamp-query
feature, so the harness can't timestamp it directly (degrades gracefully). The true-GPU split
for this exact shape/binding was measured last session with timestamp queries
(`docs/perf-dot4-report.md`): full plan **9.46 ms GPU**, CPU poll/submit only **~0.4 ms**. So
**TSync ≈ GPU time** — the resident path is GPU-bound, not poll-bound.

## Run() vs batched RunN (ForwardN) — fence amortization (all min/throttle-free)

| model | single Run | ForwardN K=8 | ForwardN K=16 |
|---|---|---|---|
| Qwen2.5-1.5B | 8 393 µs/tok | 8 460 µs/tok | 8 075 µs/tok |
| Qwen3-1.7B | 8 446 µs/tok | 8 424 µs/tok | 8 589 µs/tok |

**RunN gives ~0 per-token amortization** (≤4%). The single Submit+Poll fence is already cheap
(~0.4 ms measured), so batching K tokens into one fence saves almost nothing — consistent with
GPU-bound. (RunN's real use is speculative *verify*, not decode throughput.)

## §2 prize — staged fallback vs resident (same model, clean delta)

| path | ms/token | tok/s | resident? |
|---|---|---|---|
| Qwen2.5-1.5B **resident** | 9.68 | 103 | yes |
| Qwen2.5-1.5B **staged** (`NO_RESIDENCY`) | 28.60 | 35 | no |
| Gemma-4-E2B (softcap → forced staged) | 61.54 | 16 | no |

**§2 prize = 18.9 ms/token recoverable (66% of staged; 3× speedup)** on the *same* model — far
above the ~1 ms/token the background doc estimated. The staged path pays not just the ~330
bind-group/uniform/scratch recreations per token but **per-matmul Submit+Poll fences** (~330
round-trips/token vs the resident path's one). That fence cost is the bulk of the 19 ms.

## Read-out

- **§3 ceiling (spine fusion / RunN) for already-resident models: < ~1 ms/token, largely
  spent.** The resident path is GPU-bound (TSync 92–95% ≈ GPU time; poll ~0.4 ms). RunN
  amortizes nothing (fence already cheap). The only CPU headroom is TEncode (~0.9 ms, the 535×
  cgo dispatch record) and it isn't amortizable (every token re-records). The GPU portion is
  gemv ~4.1 ms (irreducible bandwidth) + glue ~3.4 ms; glue fusion Increments 1 (+1.5%) & 2
  (+2.3%) already shipped, and the qk-norm fold **regressed** (occupancy, `decode-fusion-next.md`).
  Realistic remaining §3 ≈ 0.3–0.8 ms/token, bounded by the WGSL matmul-geometry wall.

- **§2 prize (widen eligibility): ~19 ms/token (3×) per newly-eligible family.** Applies to the
  still-staged families — Gemma-2/3/4 (logit/attn softcap), and any remaining own-forward/hybrid
  archs. Most mainstream models are already resident (the C-lever ladder: Qwen2/2.5/3, Llama,
  Mistral, Mixtral, GLM, DeepSeek/Kimi MLA, Mellum, Phi). The biggest concrete unlock is
  **softcap residency (the C8 lever — a bounded kernel change)**, which moves Gemma off the
  61.5 ms/token staged path toward resident speed.

## Funding recommendation

**Fund §2 (widen resident eligibility), starting with softcap/Gemma (C8).** It is worth
~**19 ms/token (3×)** for every family it moves onto the resident path, versus **< 1 ms/token**
of bounded, mostly-spent headroom from more §3 spine fusion on already-resident models.

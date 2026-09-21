# CUDA prefill of prompts over 512 tokens ran 7 of 8 chunks on the exact kernels: found while starting R5, fixed

Found on 2026-09-21 measuring the baseline for R5 (`docs/tasks/red-october.md`, the prefill attention tile). `TestPrefillDecomp` at K=3900 read **GEMM 2.22 s / attention 2.69 s / glue 0.10 s = 5.01 s**, against R5's stated baseline
of 531 / 805 / 57 ms (1.39 s). At K=512 the same test matched the record exactly (68 / 12.8 ms), and K=1024 already read 5.5x the K=512 GEMM time for 2x the tokens. RTX 2070 SUPER, driver `595.91.07`, dense 1.5B int4, idle box.

## The mechanism (checked in code, then by launch counting)

`prefillChunked` splits a prompt into passes of at most 512 rows; every pass but the last runs `tailKVOnly`. Audit item M-09/M-10/M-11 (`docs/audit-2026-09-10.md`) made `prefillCore` set
`r.forceExactKernels = tail != tailLastLogits`, to keep speculative-verify tails (`tailAllLogits/Argmax`) and the embedding tail (`tailHiddenLast`) on the decode-identical exact kernels. `tailKVOnly` is not
`tailLastLogits`, so **every non-final chunk of an ordinary prompt was demoted to `gemv_w4a8_rn` and `attn_batched`** — the slow paths the L2/L3 campaign replaced (`attn_fused`, `gemm_w4a8_mma`). The bug window is the
audit fix (2026-09-10) to this commit, which **includes v0.19.0**: any CUDA prompt over 512 tokens prefilled 2-3.5x slower than `docs/benchmarks.md` says, and the headline prefill row (1.9-3.2x behind Ollama, measured 09-05..09-08)
described a path the shipped code no longer took for those prompts. The gate that should have said so is `TestPrefillChunked_bitIdentical` (heavy, not in CI): **it was red at HEAD**, both chunk widths, all 151,936 logits
differing — chunked prefill no longer matched a single pass, because chunks 0..n-2 were exact and the single pass was fast.

A second, smaller defect sat beside it: the fast-prefill floor (512) is judged on `passPromptLen = startPos + M`, the prompt SO FAR, so a first chunk narrower than 512 (the OOM-halving path, or `GOINFER_PREFILL_CHUNK`)
ran exact while its later chunks ran fast — mixed numerics inside one prompt, contradicting the field's own comment ("a property of the PROMPT, not of the chunk").

## The fix (`cuda/prefill.go`, `cuda/resident.go`)

`prefillChunked` records two facts for the duration of a multi-chunk prefill: `chunkOrdinary` (the FINAL tail is the ordinary `tailLastLogits`, so its non-final passes take the kernels the final pass takes) and
`chunkPromptLen` (the whole prompt's length, so the floor is judged on the prompt). `forceExactKernels` is now `tail != tailLastLogits && !(tail == tailKVOnly && chunkOrdinary)`. **A HiddenLast (embedding) prefill leaves
`chunkOrdinary` false, so every one of its chunks stays exact**; verify tails are never chunked and unaffected. Two counters (`fastAttnLaunches`, `fastGemmLaunches`) let a test prove which kernels ran.

## Evidence

- `TestPrefillChunked_fastKernelsOnEveryChunk` (new): a 1300-row prompt (3 chunks) launches `attn_fused` 84 = 3 x 28 layers and `gemm_w4a8_mma` 588 = 3 x 196 times (single pass: 28 and 196); a 3-chunk `HiddenLast` launches **0** of either.
  **Mutation-checked:** with the pre-fix rule it fails (28 and 196 launches: only the final chunk was fast).
- `TestPrefillChunked_bitIdentical`: **red at HEAD, green now** at chunk 256 and 300 (0 of 151,936 seed logits differ from the single pass, greedy continuation identical). This also shows chunked-fast == single-pass-fast bit for bit, so the fast path's
  fidelity evidence (gated on the fast kernels, single pass) applies to the chunked production path.
- `TestAttnBatched_bitIdentical`, `TestAttnFused_vsExact/vsF16Reference/attendsKeysBeforeStartPos`, `TestPrefillLastNArgmax_matchesPerRow`, `TestHiddenLastResidentParityCUDA`, `TestPrefillLast_e2e` (int4, int8int8), `TestPrefillLast_gemma3`, `TestPrefillLast_qwen3`,
  all `TestPrefillPath_*`: green.
- **Fidelity gate for the fast path in its production (chunked, default 512) shape** (`TestPrefillGateVsReferenceCUDA`, S = qwen2.5-coder-1.5b, K=3900, prompt set A, the existing CPU f32/f64 references; log `prefill-chunk-demotion-gate-S-K3900-2026-09-21.log`):
  exact meanAgree 79.69% HF 73/640 meanKL 0.48806; **fast 80.16%, HF 69/640, meanKL 0.49021 (1.004x)**; (a)(b)(c) all met — CELL SHIPS. (This is the confirmation cell, on set A, which has been scored before; the D7 K=3900 cell has no reference on disk and was skipped.)

## Effect (`TestPrefillDecomp`, best of 3, default chunk 512)

| K | before (GEMM / attn / glue = catSum) | after | speedup |
|---|---|---|---:|
| 512 | 68.3 / 12.8 / 10.6 = 91.6 ms | 68.2 / 12.8 / 10.6 = 91.6 ms | 1.00x (one chunk: unaffected) |
| 1024 | 375.6 / 86.5 / 24.4 = 486.6 ms | 136.1 / 46.0 / 22.1 = 204.3 ms | **2.38x** |
| 2048 | 997 / 566 / 51 = 1614 ms | 272 / 210 / 45 = 527 ms | **3.06x** |
| 3072 | 1615 / 1455 / 78 = 3148 ms | 409 / 488 / 70 = 967 ms | **3.26x** |
| 3900 | 2217 / 2694 / 101 = 5012 ms | **529 / 804 / 90 = 1423 ms** | **3.52x** |

After the fix K=3900 reads 529 / 804 ms — R5's stated baseline (531 / 805) reproduced; attention is again 56.5% of prefill. (With the chunk raised to 4096 = a single pass, K=3900 is 452 / 595 / 62 = 1110 ms: chunking itself
costs ~28% over a single pass, an unmeasured-mechanism remainder: each chunk re-streams the weights and attends a growing prefix.)

## Not established / follow-ups

No served TTFT vs Ollama measurement after the fix yet (`scripts/bench_peer_prefill.py`); `docs/benchmarks.md`'s prefill row is unchanged and is the pre-audit number the fix restores, not a new one. Other callers of `prefillChunked` with a non-ordinary final tail
(`HiddenLast`) intentionally keep the exact path; a chunk-size sweep (is 512 still the right default now that all chunks are fast; the 28% single-pass advantage) was not run.

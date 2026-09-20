# R7 — device-side top-K for filtered sampled decode (CUDA), 2026-09-20

**Verdict, against the band registered in `docs/tasks/red-october.md` R7 (≥0.90 ships, 0.80–0.90 parked,
<0.80 killed; sampled ÷ greedy, same binary, same session, CUDA 0.5B):** the **top_p** cell went
**0.657 → 0.957** and **SHIPS**. `top_k` (0.961) and `min_p` (0.971) clear it too. **Temperature-only
sampling (T=1.0, no truncation) is not served and stays at 0.739** — see "What is not served".

Predecessor record: [`sampled-topk-baseline-2026-09-20.md`](sampled-topk-baseline-2026-09-20.md) (step 0).
Raw logs: [`sampled-topk-2026-09-20.log`](sampled-topk-2026-09-20.log).

## Provenance

| | |
|---|---|
| machine | `nobara-pc`: Ryzen 7 3700X, RTX 2070 SUPER 8 GB, NVIDIA driver **595.91.07**, Nobara 44, no other GPU or CPU work started by this session during any timed run (runs were sequential; box idle before the first, load average not re-read before the final one) |
| checkpoint | `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`, `--quant int4`, `Backend: cuda`, resident (local NVMe path, not `/srv/models`) |
| code | goinfer `main` @ `12777baa` (`feat(cuda): device-side top-K ...`); Go 1.27.0; `topk.ptx` built at the ambient NVRTC **12.9.86** (a new isolated module — no audited PTX regenerated) |
| instrument | `TestSampledDecodeLadder` (`cuda/sampled_decode_ladder_test.go`): 15 interleaved rounds, arm order rotated per round, one discarded warm-up per arm, 77-token narrative-prose prompt (in the source), ≤160 generated tokens, decode rate excludes prefill, no run dropped for early EOS. Paired ratio = sampled ÷ greedy **within a round**. |
| sampling | explicit per arm, fixed seeds; **do-nothing arm included**: the same top_p config with `GOINFER_NO_TOPK_FASTPATH=1` in the same session |
| thermal | not logged (57 s run); GPU idle at 639 MiB before the run |

## Result (final code, n=15)

| arm | decode tok/s (median) | paired ratio to greedy (median) | paired SD |
|---|---:|---:|---:|
| greedy | 340.3 | 1.000 | — |
| T=1.0, no truncation | 251.4 | **0.739** | 0.011 |
| T=0.8 + top_p 0.95, **top-K off** | 224.5 | **0.657** | 0.013 |
| T=0.8 + top_p 0.95 | 326.0 | **0.957** | 0.016 |
| T=0.7 + top_k 40 | 326.4 | **0.961** | 0.017 |
| T=1.0 + min_p 0.05 | 329.3 | **0.971** | 0.017 |

Fallback rate on the ladder's top_p arm: 13 of 2,355 steps (0.55%). Top-K-off ÷ greedy read 0.664, 0.691
and 0.657 in three sessions (step 0 read 0.651): the ratio drifts ~±0.03 between sessions, which is why only
same-session pairs are quoted and why the band was decided on the do-nothing arm run beside the treatment.

## What changed, and where the time was

Step 0 (`sampled-topk-baseline-2026-09-20.md`) attributed the sampled-decode gap: forward is 3.0–3.2 ms in
**every** arm (greedy included), the full-row D2H costs only ~0.1–0.2 ms, and **host sampling is ~85%** of
the gap (0.92–0.97 ms/token temperature-only, 1.29–1.33 ms with top_p). That contradicts the premise on
which `docs/completed/plan-still-slow.md` P3 was banked ("what P3 addresses is a small fraction of a
now-smaller gap"): after P2b the host term for a filtered sampler is still ~1 ms/token.

The device path: `topk_select` returns the K=256 (K=1024 for ≥200k vocab) best `(logit, id)` pairs ordered
(logit desc, id asc) plus the full-vocabulary Z when top_p needs it; the sampler runs the *same*
`finishFilter` the host path runs over those K, and falls back to a full-row read for that token when the K
cannot prove they hold the retained set (min-p threshold not crossed inside K; top-p nucleus not reached
inside K). No randomness is consumed before a fallback, so it is invisible in the stream.

Three things went wrong on the way, recorded because two of them produced wrong numbers for a while:

1. **The kernel's first timing was of a stale row.** A launch-only hook reads whatever `r.logits` holds; after
   the correctness loop that was tie-heavy leftover data, which sent the kernel down its slow ordered path and
   reported ~320 µs. The instrument now uploads a real row first. (An `ncu` source-level profile of the
   wrong launch is what exposed it: the hottest instruction was the *ordered-gather loop header*.)
2. **A histogram-based radix select was too slow, and `__match_any_sync` is a poor warp-aggregation tool on
   Turing.** The one clean `ncu` reading of the radix kernel (warp-aggregated histogram via `match_any`,
   normal logits, V=152k) was **222 µs**, with `ncu` stall sampling putting the hottest instructions on
   `BREV` — how `match.any` is microcoded (a loop over distinct values). The plain-atomic variant was never
   measured cleanly on a normal row (its early timings ran on the stale row of item 1), so this record does
   **not** claim `match_any` beat or lost to plain atomics; the reading that logit keys concentrate into a few
   dozen bins, which serialises shared-memory atomics, is a hypothesis consistent with the profile, not a
   measured cause. What was measured: ~222 µs was too slow for a ~300 µs total budget, so the shipped kernel
   does not histogram the row. It samples it at a stride, picks a threshold that leaves a few thousand entries
   above it, compacts them with a plain atomic append and bitonic-sorts them — **102 µs** on the GPU (`ncu`,
   three launches, 102.2–102.5 µs) — with the exact radix select kept as the estimate-free fallback for short
   vocabularies, K close to V and rows whose estimate misses.
3. **`math.Exp` is ~18× slower in the burst right after the GPU sync** (a tight loop of 256 ordinary calls
   took 90–100 µs there, ~5 µs in a benchmark). The first version summed all K exps to verify the nucleus and
   then took them again: 512 exps, **190 µs per token** measured in the loop, which was the whole gap between
   top_p (0.895 in the first ladder) and top_k (0.960). It now walks the candidates in row order and stops
   when the cumulative mass reaches the cut, so only the nucleus is exponentiated.

## Correctness gates (all on the final code)

| gate | result |
|---|---|
| `TestTopKSelect_matchesReference` (kernel) | 272 (vocab × shape × K) rows identical to "sort by (logit desc, id asc)": normal, quantized, ~20-value ties straddling the K boundary, all-equal, ±0 mixed, ascending/descending, all-negative; K ∈ {1,2,32,256,1000,1024,V}; V ∈ {151936, 50000, 5000, 1024, 300, 60}. Z within 1e-6 of an f64 sum. |
| `TestSampleFromTopK_matchesFullPath` (sampler) | 102,336 draws, **0 mismatches** against the full-row sampler on identical RNG streams (4 vocab sizes × 6 row shapes × 4 temperatures × 12 filter combinations); 86.5% served, the rest fell back |
| `TestSampledTopKStreamIdentity` (end to end) | **12 of 12** (3 checkpoints × 4 configs) token streams identical to the full-row path, up to 1,000 tokens each: qwen2.5-coder-0.5b, gemma3-1b (262k vocab, K=1024), llama-3.2-1b. Fallbacks: 4 of 999 (0.40%) on llama top_p, 0 elsewhere. The test fails vacuously-passing runs (served+fallbacks == 0). |
| `TestSampleFromTopK_deviceRoundedZ` | 0 of 5,000 draws differ under a 1e-6 relative error in Z (the device's Z differs from the host's chunked f64 sum by rounding, far below that) |
| `TestKernelFMALint`, `_coversEmbeddedPTX`, `TestKernelLocalMemoryCensus` | green: `topk.cu` is in the FMA-linted set (the Z accumulation uses explicit `__fmul_rn`/`__fadd_rn`), and the kernel declares **0** bytes of per-thread local memory (a `shifts[3]`/`widths[3]` pair had put 24 B/thread there; the census caught it) |

**Mutation checks.** Each was applied by hand, the named test observed red, then restored:
min-p verification removed → `matchesFullPath` red (flat row, min-p 0.05); nucleus verification removed →
red (top_p 0.999); eligibility switched back to `penaltiesActive` (see below) → `ineligibleConfigs` red.
A fourth, deleting a trailing-probability-tie loop after the nucleus cut, **survived every test**; working it
through showed it equivalent (two distinct float32 logits do not give the same float64 exp, so equal
probabilities imply equal logits, which the row already orders by ascending id), so the loop was deleted,
not kept.

**A real bug the gates caught:** eligibility first used `penaltiesActive()`, which is false until the sampler
has history — so a configured repeat penalty looked inactive at loop start and would have switched on
mid-stream after the device path had stopped reading the full row. It now uses `penaltiesConfigured()`.

## What is not served, and the disclosed residuals

- **Temperature-only sampling** (`T=1.0`, no `top_k`/`top_p`/`min_p` — the OpenAI default) is not served, and
  cannot be by a top-K: it draws by inverse CDF in vocabulary *index* order over a full-V normalisation, and
  P2b already refuted a truncated-tail shortcut. It stays at **0.739**. Serving it would need a different,
  non-stream-identical device sampler (an owner decision, not made here).
- **Also on the full-row path:** logit bias, repetition/presence/frequency penalties, logprobs, a
  `LogitProcessor` (constrained decoding), and any family whose logits are transformed on the host after
  readback (final-logit softcap: Gemma 2/4; a logit scale other than 1: Cohere).
- **Seed contract, top_p only.** The device's Z is an f32 exp-sum reduced in f64, not `chunkedZ`'s f64 sum, so
  it differs by rounding. A draw can move only if the cumulative mass lands within that rounding of the
  cut. Observed: 0 divergences in 12 real streams and 0/5,000 under a 1e-6 error; expected ~1e-7 per token.
  It is **CUDA-only**: Metal, WebGPU and CPU keep the host Z, so the same seed can differ *between backends*
  at such a boundary event — the cost P3's banking note named, now measured small rather than assumed.
- **The band was decided on one checkpoint** (the 0.5B, as registered). gemma3-1b and llama-3.2-1b were
  identity-tested but not speed-benchmarked; `benchmarks.md` §B5.1's peer rows (goinfer vs Ollama) predate
  this change and are **not re-measured** here — they are stale for the top_p / top_k cells.
- The archived ladder log's "top-K served/fallbacks" counters on the *greedy* row include the other arms'
  warm-ups (a bookkeeping bug in the first run of the instrument, fixed in `12777baa`); the other rows are
  unaffected.

## Reproduce

```sh
# kernel + sampler + end-to-end + ladder (needs the checkpoints in ~/models; ~4 min, box idle)
GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run 'TestTopKSelect|TestSampledTopKStreamIdentity|TestSampledDecodeLadder' -v ./cuda/
go test -run TestSampleFromTopK -v ./decoder/
# A/B on any run: GOINFER_NO_TOPK_FASTPATH=1
```

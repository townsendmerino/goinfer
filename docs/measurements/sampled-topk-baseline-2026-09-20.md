# R7 step 0 — sampled-decode baseline and attribution (CUDA, 0.5B), 2026-09-20

Instrument: `TestSampledDecodeLadder` (`cuda/sampled_decode_ladder_test.go`), raw log in
[`sampled-topk-baseline-2026-09-20.log`](sampled-topk-baseline-2026-09-20.log).

**Provenance.** Machine `nobara-pc` (Ryzen 7 3700X, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07, box idle,
load average 0.18). Checkpoint `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`, `--quant int4`,
`Backend: cuda`, resident. Code at `main` @ `3fa3ce73` plus the harness. Go 1.27.x. Prompt: 77 tokens of
narrative prose (in the test source), up to 160 generated tokens; decode rate excludes prefill. 15 rounds,
the three arms interleaved with a rotating start, one warm-up per arm discarded, no run dropped for early
EOS. Same binary, same session; no peer involved. Thermal: not logged (27 s run).

| arm | decode tok/s (median, n=15) | sd | paired ratio to greedy (median) | paired sd |
|---|---:|---:|---:|---:|
| greedy | 341.6 | 4.3 | 1.000 | — |
| T=1.0 | 248.3 | 1.8 | **0.729** | 0.010 |
| T=0.8, top_p 0.95 | 222.4 | 6.0 | **0.651** | 0.021 |

This reproduces `benchmarks.md` §B5.1's sampled cliff (0.70 and ~0.66 there). R7's band on this cell:
≥0.90 ships, 0.80–0.90 parked, <0.80 killed; both arms are below the kill line today.

**Attribution** (`GOINFER_DECODE_TIMING=1`, per token, 3 reps): forward 3.0–3.2 ms in **every** arm, greedy
included, so the full-logits D2H costs ~0.1–0.2 ms at most. Host `sample`: **0.92–0.97 ms** temperature-only
(`sampleChunked`, full-V) and **1.29–1.33 ms** with top_p (`topFilterLogits` + the full-V Z pass). Host
sampling is ~85% of the ~1.1–1.6 ms sampled-decode gap.

**What that constrains (recorded before any build).**
- A device top-K return removes the D2H and shrinks host work for the *filtered* paths; ceiling for
  T=0.8+p0.95 is ~3.2 ms/token ≈ 0.93–0.95 of greedy.
- The temperature-only path draws by inverse-CDF in vocabulary index order over a full-V normalisation
  (`sampler.go`, P2b), and P2b already refuted a truncated-tail shortcut (the remainder bound needs an
  ~11.2-nat gap; real logits give 5.29 at K=32). A top-K return cannot reproduce that stream or that
  distribution. `optFwd` is gated off above T=0.2, so it does not help here either.
- top_p needs the full-V normaliser Z for its cutoff; a device-side Z differs from the host's chunked f64 sum
  by ULPs, the same class of given-seed shift P2b already accepted for the host regrouping.

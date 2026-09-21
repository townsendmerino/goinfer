# Speculative decoding and the flash-decode lane (R6 follow-up): what is at stake, and the options

**Not a gate, not a result on a shipped change.** A measurement of what speculation delivers on CUDA at depth today, where its rounds
spend GPU time, and what that says about making the lane (`GOINFER_CUDA_FLASH_DECODE`) coexist with it. Scripts and the raw kernel
window are in `spec-decode-lane-2026-09-21/`.

## The refusal today

Every speculative loop that verifies on the CUDA resident (n-gram `genNgram`, two-model `GenerateSpeculative`, block drafters
`NewBlockSpec`, and `serve --spec ngram` at startup) calls `Model.SpecDecodeConflict()`, which asks the resident's
`DecodeVerifyDivergence()`. With the lane on that returns an error, because decode (M=1) would use the lane's online-softmax tree
while verify (M>1) uses the exact `attn_batched` tree, so a verified position would not score as a decoded one and speculation would
stop being token-identical to plain greedy. Confirmed live: `serve --spec ngram` with `GOINFER_CUDA_FLASH_DECODE=16` exits at startup with
that message.

## What speculation is worth (1.5B, qwen2.5-coder-1.5b q4_K_M, ~2100-token copy-heavy code prompt, greedy, 192 tokens, fresh serve per arm, idle box, 3 runs each)

| arm | tok/s | vs plain exact |
|---|---:|---:|
| plain, exact attention | 152.1 | 1.00x |
| plain, lane | 203.6 | 1.34x |
| **exact + `--spec ngram`** | **324.8** | **2.14x** |
| lane + `--spec ngram` | refuses to start | — |

Outputs were identical across exact / lane / exact+spec on this prompt (one prompt, 192 tokens: no evidence either way about near-ties).
This is the BEST case for n-gram drafting (the prompt asks for a verbatim rewrite of text in context); it is an upper end, not typical traffic.

## Where a speculative round's GPU time goes (ncu, `gpu__time_duration`, a 6,000-launch window of the same run, 5,063 kernels, 314.9 ms)

| kernel | launches | share | avg |
|---|---:|---:|---:|
| `attn_batched` (verify attention, exact tree) | 372 | **47.1%** | 399 us |
| `gemv_w4a8_rn` (weight GEMVs) | 2600 | 27.7% | 33.6 us |
| `gemv_w8a8_fwd` (LM head) | 117 | **20.9%** | 563 us |
| everything else | | 4.3% | |

372 attention launches over 28 layers is ~13 verify rounds; 117 LM-head launches is ~9 per round, i.e. the verify runs the LM head once
per row. These are counts inferred from one window, not a per-round trace.

**Reading:** in a speculative round at depth, attention is the largest single term, larger than in plain decode (where the lane already
removed most of it), and it is running the slow exact tree. A verify attention that shared K/V reads across the batch's rows would move
that term the most: K+V per layer at this depth is ~4 MB, ~10 us of DRAM time, against 399 us measured. Nothing measured here says how
much of that a bit-compatible kernel could recover.

## Options

- **A. Speculation forces the exact tree (small).** When a speculative loop is active, decode and verify both use the exact attention
  (a per-generation scope on the resident), so losslessness holds and the startup refusal goes away. The lane serves plain requests. Cost:
  spec users get no lane (they keep today's 2.14x); a server's greedy output would differ from a lane-only server at near-ties only when
  speculation is on, so "turning speculation on changes no tokens" would no longer hold across that boundary. Gate: the existing spec
  lossless parity tests with the lane env set.
- **B. A multi-row lane so verify rows are bit-identical to the M=1 lane (large).** Design constraint: the rows' chunk partition must be a
  function of the row's own position, so batch rows can share K/V reads only if their partitions coincide; that needs the partition
  quantised to a coarse grid (e.g. multiples of 128 keys at S=16) with a per-row-group fallback when a batch straddles a boundary. Masked keys
  are exactly neutral in the online softmax (alpha = exp(0) = 1, p = 0), so shared processing is bit-identical to per-row processing. Gates: cross-M
  bit-identity (lane M=1 vs multi-row), speculative output identical to plain lane, then fidelity. **Upper-bound arithmetic, not a measurement:**
  if verify attention fell from ~47% to a few percent, a round would be ~1.6-1.8x faster, i.e. spec+lane on the order of 3.5x plain exact
  on this copy-heavy prompt. Kernel-design risk: accumulators for rows x GQA group x dims overflow registers, so rows must be spread over warps.
- **C. Side finding, not the lane:** the LM head runs per verify row (21% of a round). Batching it is a separate lever with its own
  bit-identity constraint.

## Not established

One model, one prompt, one depth; spec acceptance was not recorded here; the ncu window is one run; nothing above was run with the lane
inside a verify. Option B's benefit is arithmetic from a profile.

## Option A: implemented (2026-09-21)

`decoder.ExactAttentionScoper` (`decoder/spec_verify_guard.go`) is implemented by the CUDA resident (`EnterExactAttention`, a counted atomic scope);
the three speculative entry points (`genNgram`, `GenerateSpeculative`, `BlockSpec.GenerateStream`) enter it synchronously before their goroutine starts and
release it in the goroutine's defer. `DecodeVerifyDivergence` no longer errors for the lane (the V-sum spike, which has no scope to enter, still does), so
`serve --spec ngram` / `--drafter` start with the lane env set. Gate: `TestFlashDecodeSpeculativeScope` (real 1.5B, copy-heavy prompt): no conflict reported;
a plain lane generation launches the lane; a speculative generation launches it **0** times over 11 verify rounds and equals plain EXACT greedy token for token
(96 tokens); the lane resumes afterwards; a cancelled speculative stream leaves the scope at 0. **Mutation-checked:** with the scope's increment removed the test
fails ("the lane launched 28 times inside a speculative generation"). Served, same prompt, 3 runs: exact 151.5, lane 202.6-203.2, exact+spec 323.5-325.9,
**lane+spec 326.6-328.1** (formerly a startup refusal) — equal to exact+spec within noise, as the design says it must be: speculation gets no lane speed until option B.
Not covered: the two-model and block-drafter paths are wrapped identically but only the n-gram path has a lane-on test; concurrent generations on one Model are not
possible (single-tenant KV), so the scope is per-Model rather than per-request by construction.

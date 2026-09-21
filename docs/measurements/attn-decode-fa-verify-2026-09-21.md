# Multi-row flash-decode lane for speculative verify (option B): SHIPS by the pre-registered rule

Pre-registration: `attn-decode-fa-verify-PREREGISTERED.md` (committed before any multi-row kernel existed, not edited). Code: `fa_partial_rows_*` /
`fa_combine_rows` in `cuda/decode_fa.cu`, routing in `cuda/prefill.go` (`verifyLaneFrom`) and `cuda/flash_decode.go`. Harness:
`scripts/bench_spec_lane.py`. Raw: `attn-decode-fa-verify-served-{1p5b,d7}.{json,log}`.

## Correctness gates (all held)

- **G1, cross-M bit-identity** (`TestFlashDecodeRowsBitIdentical`): every row of a verify batch equals the M=1 lane at that position under
  `math.Float32bits`, on random K/V at hd64/G7 (0.5B), hd128/G6 (1.5B), hd256/G4 (gemma3-1b), hd128/G7 (D7); S in {2,4,8,16}; M in {1,2,3,5,9,16};
  windows 0 and 300; batches placed to cross a `per` change (843 per geometry span more than one run). **52,000,000+ elements, 0 differing bits.**
  Mutation-checked: swapping the block-to-warp assignment fails on the first batch (a 1-ulp difference).
- **G2, speculative losslessness on the lane's tree** (`TestFlashDecodeForwardNBitIdentical`, `TestFlashDecodeSpecVerifyLane`): a batched verify pass's
  logits (`ForwardN`) equal M sequential M=1 lane decode steps bit for bit at floors {0, P+3, P+9, none} (the middle two are crossed inside the batch) for M in
  {1,2,5,9,16,20} (20 exercises chunking); the lane's logits differ from the exact path's somewhere in the same setup, so the equality is not the exact path against
  itself. Speculative output equals plain lane greedy over 112 tokens at floor 0 and when the generation crosses the floor mid-stream, with hundreds of multi-row
  launches counted inside verify rounds. Mutation-checked: making the floor rule ignore the floor fails at floor 2051.
- **G3, M=1 unchanged:** the M=1 kernels and combine are not edited (`TestFlashDecodeVsF64` unchanged and green).
- **G4, option A retained:** `GOINFER_CUDA_FLASH_DECODE_VERIFY=0` restores the exact-attention scope and is tested (speculative == plain exact, 0 lane launches).

## Speed (served, greedy, `serve --spec ngram`, fresh serve per arm, alternating order, 3 pairs x 2 reps, idle box, 192 tokens)

Prompt: the first N characters of `decoder/model.go` followed by "rewrite the file above EXACTLY" — a **copy-heavy, best-case prompt for n-gram drafting**.
`prompt_tokens` are the engine's usage counts. A = exact + `--spec ngram` (option A behaviour), B = lane (`GOINFER_CUDA_FLASH_DECODE=16`) + `--spec ngram`.

| model, prompt tokens | A: exact+spec | B: lane+spec | **B / A** (median of 3 pairs) | A/A floor | plain exact | plain lane |
|---|---:|---:|---:|---:|---:|---:|
| 1.5B, 1838 | 355 | 356 | 1.005 | 1.005 | 165.3 | 165.7 |
| **1.5B, 3998** | 257 | 383 | **1.488** | 0.998 | 122.4 | 192.9 |
| 1.5B, 7848 | 166 | 317 | **1.913** | 1.005 | 84.6 | 169.8 |
| **D7, 3998** | 104.9 | 153.0 | **1.458** | 1.000 | 51.4 | 66.1 |
| **D7, 7848** | 70.2 | 131.5 | **1.871** | 0.996 | 39.9 | 61.1 |

Every pair's ratio: 1.5B@3998 1.488/1.500/1.480, @7848 1.910/1.918/1.913; D7@3998 1.458/1.459/1.458, @7848 1.871/1.869/1.880. A/A floors are within 0.5%.

**Pre-registered rule:** ships if R >= 1.15 on the 1.5B at ~3900 AND on D7 at ~7500, with no cell below 0.98. **R = 1.488 and 1.871, the smallest cell is 1.005: SHIPS**
(offered opt-in; it is the lane's cost to speculation that is removed). The 1838-token cell is a no-op by design (the prompt plus generation sits at or under the
2048-key floor, so both arms run the exact path), and reads the A/A floor.

Against plain exact decoding the combined effect at ~7850 tokens is **3.75x on the 1.5B (316.6 vs 84.6)** and **3.30x on D7 (131.5 vs 39.9)**, on this best-case prompt.

## What this does not establish

- **Prompt dependence.** Acceptance was NOT recorded, and this prompt is the best case for n-gram drafting. On traffic where drafts rarely hit, speculation itself gives
  little and so does this. No non-copy prompt was measured here.
- Only the n-gram path was served-measured; the two-model and block-drafter paths call the same `ForwardN`/`PrefillLastNArgmax` verify (routed the same way), but
  have no lane-on losslessness test of their own.
- Two models on one card, one prompt family, two depths; not gemma3-1b/0.5B; no peer comparison for speculative decoding.
- The LM head (21% of a verify round in the earlier profile) and the weight GEMVs are untouched; this removed the attention term only.
- The lane remains opt-in and fidelity-gated on two cells (`attn-decode-fa-fidelity-2026-09-20.md`); nothing here changes that.

## An incident worth keeping

The first push of this change turned CI red: `TestPTX_matchesSourcesAndBindings` requires every PTX entry to be declared textually in its `.cu`, and the multi-row
kernels were macro-generated. The guard is right and the kernels are now spelled out (PTX byte-identical); the failure came from running the lint tests with a name
filter that did not include that one. The full CI cuda command (`-short`, both tags) is what to run.

# SigLIP (Gemma 3) int8 vs f32, and the `visionmetal` "real-size defect" withdrawn (2026-10-07)

**A record, not a gate.** It answers two questions:

1. Is aikit's `gpu/visionmetal` (the Metal SigLIP tower) wrong at Gemma 3's real size, as S3's G-S3a reading suggested?
2. How far is SigLIP's int8 (W8A8) tower from its f32 tower?

The second is the measurement `vision-tower-int8-fidelity-2026-10-02.md` did not make for SigLIP. That record kept Gemma 3
on an int8 tower under `--backend cuda|webgpu` because the resident tower needs int8, but it measured only the other
towers.

## Answer

- **There is no `visionmetal` defect.** The Metal int8 tower is as far from the f32 tower as aikit's own CPU int8 tower
  is. G-S3a's red (worst token 0.180-0.461) compared two int8 towers, which go wrong on the same tokens and then disagree
  with each other.
  - The "int8 Metal vs int8 CPU" rows below reproduce G-S3a's figures exactly: 0.181, 0.338, 0.180, 0.461.
  - The task doc's S3 entry is corrected to say so.
- **SigLIP's int8 tower is much further from f32 than the other towers' int8 towers.** Across the four images:
  - mean per-token cosine 0.932-0.987;
  - worst token 0.012-0.172;
  - relative L2 0.16-0.52.

  The 2026-10-02 record has Qwen2.5-VL, Qwen3.5 and Gemma 4 at relative L2 0.14-0.31.

  Serve still forces this tower for Gemma 3 under `--backend cuda|webgpu` (`towerInt8` in `internal/serveapp/main.go`).
  How much reaches the decoder's text was **not** measured here.

## What was measured

The tower is `vision.LoadEncoder(~/models/gemma-3-4b-it, quant)`, the call serve makes. Three arms, on each image:

| arm | tower | where |
|---|---|---|
| f32 | `quant=false` | CPU |
| int8 CPU | `quant=true` | CPU |
| int8 Metal | `quant=true` with `EnableResident()` | Metal, through `gpu/visionmetal` |

- Every arm takes the same preprocessed pixels: `vision.Preprocess(..., vision.Gemma3())`, 896x896.
- The output compared is `last_hidden_state`, 4,096 tokens x 1152.
- Images: the four F2a uses, `testdata/{gemma3_preprocess_image,qwen25vl_preprocess_image,glm_ocr/formula,glm_ocr/table}.png`.
- Test source: `siglip-int8-fidelity-2026-10-07/siglip_int8_record_test.go.txt`, a throwaway test run in aikit's
  `gpu/visionmetal` and not in either tree. Raw output: `siglip-int8-fidelity-2026-10-07/record.log`.

## Provenance

- **Machine:** MacBook, Apple M1 Pro (arm64), macOS 26 (Darwin 25.6.0), on 2026-10-07, 16:04-16:11 PDT.
- **Builds:** aikit at `5db4ce6` (local branch `s2-tower-exports`); goinfer `main` at `75b18eb6` (only its test images
  were used).
- **Checkpoint:** `~/models/gemma-3-4b-it` (local disk, the bench set).
- **Runs:** deterministic, with no sampling. One run per row.
- **Timing:** nothing was timed. The seconds in the log are wall time per image for all three arms together and are not
  quotable.

## Result

| image | arm | per-token cosine, mean | worst | relative L2 |
|---|---|---|---|---|
| gemma3_preprocess_image | int8 CPU vs f32 CPU | 0.93484 | 0.06172 | 0.5084 |
| gemma3_preprocess_image | int8 Metal vs f32 CPU | 0.93487 | 0.03450 | 0.5008 |
| gemma3_preprocess_image | int8 Metal vs int8 CPU | 0.99006 | 0.18129 | 0.2064 |
| qwen25vl_preprocess_image | int8 CPU vs f32 CPU | 0.93391 | 0.01182 | 0.5231 |
| qwen25vl_preprocess_image | int8 Metal vs f32 CPU | 0.93248 | 0.02706 | 0.5218 |
| qwen25vl_preprocess_image | int8 Metal vs int8 CPU | 0.98996 | 0.33782 | 0.1843 |
| glm_ocr/formula | int8 CPU vs f32 CPU | 0.97907 | 0.17155 | 0.2275 |
| glm_ocr/formula | int8 Metal vs f32 CPU | 0.97818 | 0.09285 | 0.2553 |
| glm_ocr/formula | int8 Metal vs int8 CPU | 0.98264 | 0.18043 | 0.2479 |
| glm_ocr/table | int8 CPU vs f32 CPU | 0.98711 | 0.11535 | 0.1632 |
| glm_ocr/table | int8 Metal vs f32 CPU | 0.98659 | 0.11001 | 0.1892 |
| glm_ocr/table | int8 Metal vs int8 CPU | 0.99095 | 0.46114 | 0.1511 |

## How the defect hypothesis was ruled out first

These are exploratory checks; the table above is what decides.

1. **The fused register GEMMs.** `GEMMW8A8Bias` and `GEMMW8A8BiasAdd` had tests only on CUDA. A new Metal test checks them
   at SigLIP's shapes, ragged N = 4304 included, against the plain register kernel plus a host add, and they pass. It is
   committed to aikit as `gpu/metal_vit_fused_test.go`.
2. **The geometry.** A synthetic tower at so400m's exact geometry (hidden 1152, 16 heads of 72, MLP 4304, 4,096 patches,
   one layer, random weights; `mk.py`) gives int8 Metal vs int8 CPU at cosine 0.99999, worst token 0.99997. Shape alone
   breaks nothing.
3. **The depth.** The real tower cut to its first N layers (`cut.py`) gives int8 Metal vs int8 CPU, worst token, by
   depth (`synth_probe_test.go.txt`):

   | layers | 1 | 4 | 8 | 14 | 20 | 27 |
   |---|---|---|---|---|---|---|
   | worst token | 0.99997 | 0.99935 | 0.99839 | 0.98773 | 0.92983 | 0.57638 |

   The divergence grows smoothly with depth. A broken kernel would show up at the first layer.
4. **Both int8 towers against f32.** On the synthetic-pixel input (`synth_probe_test.go.txt`) at full depth, the worst
   token is 0.219 for the int8 CPU tower and 0.218 for the int8 Metal tower. Both are equally far from f32, so int8
   itself is what diverges.

Why the tiny-fixture test could not see this: at 2 layers and 16 patches, the two int8 towers agree to 1.0. The
divergence is a property of the real weights at depth.

## Not covered

- What the decoder writes from int8 against f32 SigLIP features (an end-to-end comparison).
- CUDA's SigLIP tower (`cuda/vision_encoder.go`, int8 by construction) and WebGPU's. They were not run here, and on this
  evidence they inherit the int8 loss whatever their kernels do.
- Whether a different activation quantization (per-channel smoothing, or keeping fc2's input at f16) would make an int8
  SigLIP tower acceptable.

## What it changes

- **The `visionmetal` defect report to aikit is withdrawn.** It was never sent.
- S3's rebuilt Metal SigLIP tower is f32 and passed G-S3a at real size, so serve on Metal is unaffected.
- `-vision-quant`'s help text said int8 is "cosine ~0.999". It now cites the two records.
- **Open, for the owner:** serve's `towerInt8` still forces this int8 tower for Gemma 3 under `--backend cuda|webgpu`. Those
  backends are nobara's to measure, end to end, before anything changes.

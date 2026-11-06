# MEM11-bf16 on CUDA: the bf16 drafter's upload (nobara-sb01)

`docs/measurements/mem11-2026-10/README.md` left one thing open: "the CUDA drafter upload with a bf16 drafter (the packer change) ... runs only on nobara". Every CUDA drafter test
loads the f32 conversion (`GOINFER_DFLASH_F32`); none attaches the bf16 checkpoint to the device.

## MEM11-CUDA-G1, tier 1, registered 2026-11-06 04:34:05 PST before it runs

- **Instrument:** `TestResidentDrafter_bf16AttachMatchesF32` (`backend/cuda/drafter_test.go`, `cuda goinfer_testhooks`, `GOINFER_HEAVY_TESTS=1`): the Qwen3-4B target resident on the card, the f32 conversion attached as the reference, then the bf16 checkpoint (`~/models/qwen3-4b-dflash`, the file the conversion was made from, registered as `GOINFER_DFLASH_BF16`) attached held and opened to stream. The buffers compared, one by one, by bytes: `fc`, both trunk norms, each layer's seven matrices (weights and scales) and four norms. The comparison is the helper the existing streamed-against-held test now shares.
- **Rule:** PASS if both subtests (`held`, `opened`) pass with every buffer equal and neither is skipped; FAIL on any differing buffer. Widening bf16 to f32 is exact, so the quantized device bytes should be equal; a difference is a finding about the bf16 path, not noise.
- **Plant:** the test must go red when the bf16 widening is wrong: one run with the widening in `decoder/dflash.go:stRows` perturbed (not committed), expected red.
- **Cost:** one test binary on the card under the exclusive timing lock, about 2 to 4 minutes; no other card use.

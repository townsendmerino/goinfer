# MEM11-bf16 on CUDA: the bf16 drafter's upload (nobara-sb01)

`docs/measurements/mem11-2026-10/README.md` left one thing open: "the CUDA drafter upload with a bf16 drafter (the packer change) ... runs only on nobara". Every CUDA drafter test
loads the f32 conversion (`GOINFER_DFLASH_F32`); none attaches the bf16 checkpoint to the device.

## MEM11-CUDA-G1, tier 1, registered 2026-11-06 04:34:05 PST before it runs

- **Instrument:** `TestResidentDrafter_bf16AttachMatchesF32` (`backend/cuda/drafter_test.go`, `cuda goinfer_testhooks`, `GOINFER_HEAVY_TESTS=1`): the Qwen3-4B target resident on the card, the f32 conversion attached as the reference, then the bf16 checkpoint (`~/models/qwen3-4b-dflash`, the file the conversion was made from, registered as `GOINFER_DFLASH_BF16`) attached held and opened to stream. The buffers compared, one by one, by bytes: `fc`, both trunk norms, each layer's seven matrices (weights and scales) and four norms. The comparison is the helper the existing streamed-against-held test now shares.
- **Rule:** PASS if both subtests (`held`, `opened`) pass with every buffer equal and neither is skipped; FAIL on any differing buffer. Widening bf16 to f32 is exact, so the quantized device bytes should be equal; a difference is a finding about the bf16 path, not noise.
- **Plant:** the test must go red when the bf16 widening is wrong: one run with the widening in `decoder/dflash.go:stRows` perturbed (not committed), expected red.
- **Cost:** one test binary on the card under the exclusive timing lock, about 2 to 4 minutes; no other card use.

## MEM11-CUDA-G1 read, 2026-11-06 04:34 to 04:36 PST: PASS

`g1-green.log`, `g1-plant-red.log`; the exclusive timing lock, no other card user, 4B target resident on the 2070 SUPER.

- **Green:** `TestResidentDrafter_bf16AttachMatchesF32` passes, 15.4 s; `held` and `opened` each equal the f32 conversion's device bytes in every buffer. `TestResidentDrafter_streamedAttachMatchesHeld`, which now shares the comparison helper, still passes (19.0 s).
- **Plant:** with the bf16 widening in `decoder/dflash.go:stRows` set to `<< 16 | 0x00400000` (a mantissa bit forced on; reverted, not committed), `opened` fails with 72 differing buffers and `held` stays green. The test can go red, and the two attach routes are separate.
- **What it is not:** a fidelity or memory reading. It says the bf16 drafter's device image equals the f32 file's; the host-side saving is MEM11-bf16's own record.

## Found on the way: the asset registry cannot be found from `backend/cuda`

`decoder/assets.go:repoRoot` is the constant `".."` ("every package that uses this helper is one level below the checkout root"). The GPU backends moved to `backend/<name>/` on 2026-11-03, two levels down, so from `backend/cuda` the registry opens `../testdata/assets.json` and fails: **every `decoder.AssetPathForTest` call in `backend/` skips** ("asset registry unreadable"), with `GOINFER_HEAVY_TESTS=1` set, as `ok`. First run of this gate showed it for both drafter tests (`g1-green.log` before it was overwritten; the skip text was `asset GOINFER_DFLASH_F32: asset registry unreadable: open ../testdata/assets.json: no such file or directory`). The readings above were taken with a temporary symlink `backend/testdata` to `../testdata`, removed afterwards (0 untracked paths left). Ten files in `backend/cuda` and others call `AssetPathForTest`. **No fix is in this PR**: the fix is in `decoder/assets.go` (walk up to the registry), which stales the parity manifest; it is a separate change for the coordinator to order.

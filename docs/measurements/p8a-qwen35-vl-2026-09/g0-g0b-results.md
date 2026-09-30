# P8a G0 / G0b results — 2026-09-30

Bars: `preregistration.md` (incl. amendment A1). Machine `nobara-pc` (amd64, CPU f32). HF reference
`transformers 5.15.0`, `torch 2.12.0+cpu`, f32. Checkpoint `~/models/qwen3.5-0.8b` (local NVMe). aikit at commit
`f119c63` (local, NOT pushed or tagged — see "Release state" below). Day run: every command here took seconds.

## G0 — tower vs HF, real Qwen3.5-0.8B tower weights, random pixel_values (seed 1), grids (1,16,16) (1,12,20) (1,32,8) (1,2,4)

Golden written outside the repo: `~/models/qwen35vl_tower_golden/qwen35vl_tower_golden.json.gz` (22.9 MB) by
`aikit/scripts/oracle/pin_qwen35_vision.py --real ~/models/qwen3.5-0.8b` (rotary `inv_freq` asserted finite and
equal to the expected table first, per the persistent=False-buffer trap). Test: `go test ./vision/ -run
TestQwen3VisionEncoder_realParity -v` in aikit.

| stage | worst-row cosine | max\|diff\| | relative (bar 2e-5) | verdict |
|---|---|---|---|---|
| S1 patch_embed + pos_embed | 1.000000000 | 1.05e-05 | 2.65e-06 | PASS |
| S2 last block output | 0.999999997 | 4.32e-02 | 1.96e-05 | PASS — at 98% of the bar |
| S3 merged features | 0.999999999 | 6.08e-05 | 1.66e-05 | PASS |

Bit-exact sub-gates (bilinear tap indices, weights ≤ 1 ulp, rotary position ids) PASS for all four grids.
S2's relative figure is the one to watch: its max\|golden\| is large (outlier activations ≈ 2.2e3), so a 4e-2
absolute difference is 2e-5 relative. It passed by 2%; that is not slack to spend, and a 9B/35B tower run
should be expected to land in the park band more easily.

**A1 mutation evidence.** The tiny fixture initially passed a merger with GELU-tanh in place of GELU-erf on
cosine alone. Mutations run against the Go tower:

| mutation | tiny (bar 5e-6) | real (bar 2e-5) |
|---|---|---|
| drop patch-embed bias | FAIL, S1 cosine 0.987 | not run |
| interpolation `align_corners=False` | FAIL, tap weights | not run |
| merger erf → tanh | FAIL, S3 relative 1.72e-5 | FAIL, S3 relative 3.24e-4 (16× the bar) |

Pre-registered condition met: the erf→tanh mutant breaches the bar on the REAL tower, so the real gate is
credited with detecting a wrong merger activation. (Un-amplified, the tiny fixture read 5e-7 either way; the
×8 merger-fc1 scale in the pin script is what makes it sensitive.)

## G0b — preprocessing

`go test ./multimodal -run 'Qwen3Preprocess|LoadQwen3' -v` (goinfer): PASS ×3.
- pixel_values, 64×96 image → 24 patches × 1536: **36,864 / 36,864 bit-identical** to the numpy transcription of
  HF's torchvision-backend path. The UNFUSED normalize `(x/255 − mean)/std` differs in 15,512 of the same values
  (111 of the 256 byte values differ at mean = std = 0.5), so the bar is a property of the fused form
  `(x − 255·mean)/(255·std)` and `QwenPreprocessConfig.FusedNormalize` is required for Qwen3.5.
- `LoadQwen3PreprocessConfig` on the real 0.8B dir: 65536 / 16777216 / 16 / 2 / 2, mean = std = 0.5.
  The Qwen2.5-VL loader on the same file was shown to keep its own budget (the trap), the new one refuses a file
  without `size.*`, a `min_pixels`-only file, missing geometry, missing mean/std, and zero std.
- Not covered (disclosed in the pre-registration): images needing a resize. HF's torchvision backend resizes
  uint8 with bicubic `antialias=True`; goinfer's `qwenBicubicU8` is PIL-style (a = −0.5), which is
  tolerance-matched to PIL, not to torchvision, and a = −0.75 vs −0.5 differ. No PIL/torchvision in the venvs
  here. Every G2/G4 image is grid-aligned; a photograph from a user is not.

## Release state (deviation from the brief, flagged)

The brief says to tag aikit and bump all five pins in one commit. The aikit release ritual includes `perfgate`
(an interleaved same-session run, well over the 10-minute by-day ceiling in CLAUDE.md § Run budget) and a CI
wait before the tag. Not done today. The new tower is in aikit commit `f119c63`, LOCAL; goinfer builds against
it through `go.work` only. **Until aikit is tagged and the pins are bumped, no goinfer commit that uses
`vision.Qwen3VisionEncoder` may be pushed** — a `GOWORK=off` root build resolves aikit v1.50.2 from the proxy
and fails on the missing symbol (the known between-releases failure, CLAUDE.md § Working in the tree).

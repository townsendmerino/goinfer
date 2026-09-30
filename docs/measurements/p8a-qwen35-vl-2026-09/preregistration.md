# P8a — Qwen3.5+ images, end to end: gate pre-registration

Written 2026-09-30, at tree `0a3f0a7f` (+ this directory), BEFORE any tower, seam or loader code exists.
Context and Phase 0 evidence: `docs/multimodal.md` § P8 "RE-SCOPED 2026-09-30". No threshold below is to be
loosened after a result is seen; a bar moves only with a mechanism, in a new dated amendment at the bottom.

Machine: `nobara-pc` (amd64, CPU f32 path; the brief's Mac is not this session's box). HF reference:
`transformers 5.15.0`, `torch 2.12.0+cpu`, f32, `~/g4venv`. Checkpoint: `Qwen/Qwen3.5-0.8B` at
`~/models/qwen3.5-0.8b` (local NVMe, not `/srv/models`), 9B for the second leg of G2 (not on this box —
needs a `models-pull`/download first). Every checkpoint and image path a gate reads is under `~/models`.

## What is bit-identical, what is a named fidelity gate

Default bar is bit-identity. The tower cannot be bit-identical to torch (matmul reduction order, `erf`,
`tanh`, `cos`/`sin` implementations), so G0/G1 are NAMED fidelity gates with the numbers below; everything
that is integer or a pure copy is bit-exact and asserted so: patchify order/pixel_values, grid, placeholder
count, m-RoPE positions, delta, the text-path logits (G3).

## G0 — tower features vs HF (aikit; new encoder, `Qwen3_5VisionEncoder`-shaped, old one untouched)

Real 0.8B tower weights (bf16 → f32), random `pixel_values` (seed fixed in the pin script), HF
`Qwen3_5VisionModel` at f32. Grids `(t,h,w)` in patch units: **(1,16,16) square, (1,12,20) non-square,
(1,32,8) tall, (1,2,4) tiny** (the last exercises the `n−1` interpolation edge).
Stages, each compared per ROW (row = one patch token pre-merge, one merged token post-merge):

| stage | quantity | bar |
|---|---|---|
| S1 | patch_embed + interpolated pos_embed | per-row cosine ≥ 0.999999 |
| S2 | last block output (pre-merger) | per-row cosine ≥ 0.9999 |
| S3 | merged features `[h·w/4, out_hidden]` | per-row cosine ≥ 0.9999 |

Also recorded, not gated: max|diff| per stage. **Ambiguous → parked band:** any row with cosine in
[0.999, bar) on S2/S3 parks the gate (no ship, no pass) and gets a per-block diff before anything is
changed. A row below 0.999 is a FAIL.
Bit-exact sub-gates (integer/copy, no tolerance): the bilinear tap indices for all four grids equal HF's
`get_vision_interpolation_indices_and_weights` output (indices exact, weights ≤ 1 ulp f32), the rotary
position ids equal `get_vision_position_ids` exactly.
Then tag aikit and bump the pin in all five modules in one commit (`RELEASING.md` version alignment).

## G0b — preprocessing (goinfer `multimodal`)

For a grid-aligned image (both sides a multiple of 32, ≥ 65536 px, so `smart_resize` and the resize are
identities): `pixel_values` and `grid_thw` are **bit-exact** to the numpy patchify transcribed from
`Qwen2VLImageProcessor.patchify` (temporal frame duplicated, order (block-row, block-col, merge-row,
merge-col | channel, temporal, py, px)), mean = std = 0.5. `LoadQwenPreprocessConfig` on the 0.8B dir must
return min 65536, max 16777216, patch 16, merge 2 — the old loader read `min_pixels`/`max_pixels` and
silently defaulted; this is asserted, not assumed. NOT covered, disclosed: images that need a resize
(bicubic, torchvision `antialias=True` vs PIL vs our `qwenBicubicU8`) — all G2/G4 images are grid-aligned.
No PIL/torchvision in the HF venvs here; a resize-fidelity gate is a follow-up needing them installed.

## G1 — tiny fixture, real tower + decoder shape (`scripts/pin_qwen35_vl_tiny.py`)

A tiny `qwen3_5` checkpoint with a vision tower of real structure (biased Conv3d, learned 8×8 pos grid,
2D rotary, LayerNorm, gelu-tanh MLP, erf merger, `deepstack_visual_indexes: []`) and the 3:1 hybrid text
decoder with interleaved m-RoPE `[11,11,10]`-shaped section and partial rotary. One image + text prefix and
suffix; greedy 8-token continuation. Goldens: `image_features`, `position_ids`, `last_logits`, continuation.
Bars: `position_ids` and `mrope_delta` **exact**; `image_features` per-row cosine ≥ 0.9999; end-to-end
`last_logits` cosine ≥ 0.9999 (the text-only tiny goldens already hold 0.9999; the Qwen2.5-VL image gate's
looser 0.99 is NOT adopted); **argmax exact and all 8 continuation tokens exact.**

## G3 — text path unchanged (with G1)

`runLayersQwen35` becomes a wrapper of `runLayersQwen35FromEmbed`; the rotary call moves from `applyRoPE` to
`ropeAt`. Bar: raw f32 logits bits identical to the pre-change tree at every prompt position and 12 greedy
steps, 2 prompts, on `testdata/qwen3_5-tiny`, `testdata/qwen3_5_moe-tiny` and the real 0.8B.
Baseline: `g3-baseline-0a3f0a7f.json` (this directory), written at `0a3f0a7f` by
`TestQwen35_textIdentityHashes` (`GOINFER_G3_WRITE`), checked after the change with `GOINFER_G3_CHECK`.
The instrument was shown deterministic (same-tree re-run byte-identical, all 6 lists) and able to go red (a
mutated hash fails) before any decoder edit. Plus: every existing qwen35 forward golden test green under
`-v` with `--- PASS` (not `ok` on skips).

## G2 — real 0.8B, then 9B, vs HF

Three grid-aligned images, generated procedurally by the pin script and written as PNG (pure zlib) and as
the exact pixel array (sizes 256×256, 384×256, 320×512 → 64/96/160 image tokens), each with the model's
real chat template and a fixed question. HF f32 greedy, 32 new tokens; goinfer CPU f32 greedy, 32 new
tokens. **Bar: all 32 tokens identical on all 3 images.** Ambiguous → parked band: a divergence at a step
where HF's own top-1 − top-2 logit gap is < 0.02 is classed NEAR-TIE and PARKS the gate (it is neither a
pass nor a defect) and is resolved by a teacher-forced pass (feed HF's tokens, compare per-step argmax and
the gap); any divergence with gap ≥ 0.02 is a FAIL. Stage-isolated before the end-to-end run: image_features
per-row cosine ≥ 0.9999, positions exact. 9B repeats the same rule (needs the 9B safetensors; the run is a
night-queue item: ~19 GB bf16 download, HF f32 reference ~36 GB RAM — estimate before launching).
After G2: regenerate `capability-matrix.json` by its generator; do not hand-edit.

## G4 — serve, and the reuse decision

(a) An OpenAI `image_url` (data URI, PNG) content part through `serve` on the 0.8B returns G2's 32 tokens
for each of the 3 images (temperature 0). (b) The same request twice returns identical tokens and
identical `usage.prompt_tokens` (cold determinism). (c) The reuse decision function, tested directly and
through its caller `GenerateQwenVL`'s branch selection: on a recurrent family an image claim is honoured
ONLY under exact extension AND every committed image block has a same-nonzero-hash claim → returns
`len(resIDs)`; changed hash, missing claim, edited prefix, identical resend, shorter resend → 0. And
`GenerateQwenVL` on a recurrent family with a `ResidentMRoPE` resident must NOT take the UploadKV bridge
(test asserts it via a fake resident whose `UploadKV` fails the test if called). **The end-to-end
"reused == cold, bitwise, for an image turn" is NOT gated here**: with CPU decode the resident cache is
never populated by an image turn, so it would pass vacuously. It moves to the resident-image step and is
named there. (Disclosed narrowing of the brief's G4 wording — see the P8 record.)

## Out of scope for this pre-registration

Video; >1 image per turn beyond what `GenerateQwenVL` already handles; DeepStack; resident image
decode/prefill for the hybrid; the MoE checkpoint (`Qwen3.6-35B-A3B`); P8b; image-resize fidelity.

## Amendments

**A1 — 2026-09-30, after the tiny G0 fixture ran green, before any real-tower G0 number was seen.**
Mechanism: the tiny G0 gate (per-row cosine only) PASSED a deliberately wrong tower whose merger used
GELU-tanh instead of GELU-erf (this is llama.cpp's actual substitution, see the P8 prior-art record). Cosine
cannot resolve a ~1e-4 activation difference. Mutations run against the Go tower on the tiny fixture: drop
the patch bias → FAIL (S1 cos 0.987); align_corners=False interpolation → FAIL (tap weights); merger
erf→tanh → PASS (blind). Fix, two parts, both fixed now:
1. The tiny fixture's merger `fc1` is scaled ×8 (tiny only) so its pre-activations reach |x|≈3. Measured on
   that fixture: clean S3 max|diff|/max|golden| = 4.6e-7; erf→tanh mutant = 1.7e-5 (37× larger).
2. A second criterion beside cosine, **relative max|diff| = max|got−golden| / max|golden|**: tiny gate
   S1, S2, S3 ≤ **5e-6** (≈10× the clean run, ≈3× below the mutant). Real-tower gate S1, S2, S3 ≤ **2e-5**,
   park band [2e-5, 1e-4), FAIL ≥ 1e-4 — and the real gate is only credited with detecting a wrong merger
   activation if the erf→tanh mutation is RUN against the real tower and shown to breach 2e-5; otherwise
   it is declared blind to that defect and only the tiny gate carries it.
Nothing else in G0 changes.

**A2 — 2026-09-30, after the G1 tiny fixture first ran green, before any real-checkpoint image number.**
Mechanism: with `rope_theta = 1e7` (the released value) the tiny fixture PASSED two deliberately wrong
decoders — the contiguous m-RoPE layout in place of the interleaved one, and a dropped `mropeDelta` on
decode — because at positions 3..9 every frequency but the first rotates by ~1e-2 rad or less, so the logits
move by less than the 0.9999 cosine bar. A gate that passes the defect it exists for is not a gate. The tiny
fixture's `rope_theta` is now 10 (tiny only; the layout and delta logic are theta-independent; the real
checkpoints keep 1e7). Bars UNCHANGED (position_ids/delta exact; features cosine ≥ 0.9999; logits cosine
≥ 0.9999; argmax and all 8 tokens exact). Mutations against the re-pinned fixture: scalar RoPE → FAIL (cosine
0.9946); contiguous layout → FAIL (0.99977); `mropeDelta` = 0 → FAIL (continuation diverges at step 5);
resident branch not refused → FAIL (`UploadKV` runs). Consequence for G2: the real checkpoints have theta 1e7,
so G2's image runs are LESS sensitive to a rotary-layout error than this fixture; G2 (exact tokens over 32
steps on 3 images) is not evidence about the layout, G1 is.


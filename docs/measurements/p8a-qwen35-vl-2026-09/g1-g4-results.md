# P8a G1 / G2 / G3 / G4 results — 2026-09-30

Bars: `preregistration.md` incl. amendments A1 (tower relative max|diff|) and A2 (tiny fixture rope_theta).
Machine `nobara-pc` (amd64, CPU f32, no GPU backend used). HF reference `transformers 5.15.0`, `torch 2.12.0+cpu`,
f32. Checkpoint `~/models/qwen3.5-0.8b` (local NVMe). goinfer at `f4e56538` + the uncommitted serve wiring; aikit at
`f119c63` (local, untagged — see "Release state" in `g0-g0b-results.md`). Logs: `~/goinfer-logs/p8a/`.
All wall times below are from runs that overlapped other work on the box: **exploratory, not quotable as speed.**

## G3 — the text path is byte-identical (PASS)

`TestQwen35_textIdentityHashes`: SHA-256 of the raw f32 logits bits at every prompt position and 12 greedy steps, two
prompts, on `testdata/qwen3_5-tiny`, `testdata/qwen3_5_moe-tiny` and the real 0.8B. Baseline written at `0a3f0a7f`
BEFORE the first decoder edit (`g3-baseline-0a3f0a7f.json`), checked after the seam landed (`runLayersQwen35` →
wrapper over `runLayersQwen35FromEmbed`; rotary through `ropeAt`; `MRopeSection` set from `rope_parameters`; a
recurrent guard in `GenerateQwenVL`; `prefillLogitsQwenVL` dispatch): **6 of 6 lists, 132 of 132 logit vectors
byte-identical.** The instrument was shown deterministic (same-tree re-run) and able to fail (mutated hash).
The existing qwen35 / qwen2.5-VL / resident-reuse / registry / gemma VL tests: 101 ran and passed, 3 skipped (missing
assets or the env this test needs), 0 failed.

## G1 — tiny Qwen3.5 image checkpoint vs HF (PASS)

`scripts/pin_qwen35_vl_tiny.py` (real tower structure incl. non-zero biases and merger ×8, 3:1 Gated-DeltaNet
hybrid, head_dim 64 / partial 0.25 / mrope_section [3,3,2] interleaved, theta 10 per A2). Bars all met: `position_ids`
and `rope_delta` (−3) EXACT against HF's `get_rope_index`; image_features per-row cosine ≥ 0.9999 (tower on the golden
pixel_values, then the decoder); last-position logit cosine 1.000000000, argmax 174 = HF; all 8 continuation tokens
exact (`[174 179 26 59 179 26 59 7]`). Every HF continuation step has top1−top2 gap ≥ 0.049.
**Mutation record (against the re-pinned fixture):** scalar RoPE in place of `ropeAt` → FAIL (cosine 0.9946);
contiguous m-RoPE layout → FAIL (cosine 0.99977); `mropeDelta` = 0 on decode → FAIL (continuation diverges);
resident branch not refused → FAIL (`UploadKV` ran, 1 call — only the single full-attention layer has KV, the three
DeltaNet layers are skipped, which is the zero-state hazard itself). Before A2 (theta 1e7) the layout and delta mutants
PASSED.

## G2 — real Qwen3.5-0.8B, 3 images, 32 greedy tokens vs HF f32 (PASS)

Images (procedural, grid-aligned): A 256×256 (64 image tokens), B 384×256 (96), C 320×512 (160). Through goinfer's own
`QwenPreprocess` → aikit `Qwen3VisionEncoder` → `GenerateQwenVL` (the entry point serve uses), HF's input ids.

| image | pixel_values | image_features worst-row cosine | tokens identical | smallest HF top1−top2 gap in the 32 steps |
|---|---|---|---|---|
| A | bit-exact | 0.999999996 | 32 / 32 | 0.007 (below the 0.02 near-tie line) |
| B | bit-exact | 0.999999996 | 32 / 32 | 0.129 |
| C | bit-exact | 0.999999934 | 32 / 32 | 0.017 (below the 0.02 near-tie line) |

Run twice (once on the original goldens, once after `pin_qwen35_vl_real.py` was re-run to add the `serve_` variant):
both PASS. HF reads the images as "a colorful, abstract digital artwork … geometric shapes … gradient background",
i.e. the reference path sees the picture.
**Sensitivity, measured:** at the released rope_theta 1e7, G2 is a WEAK detector of rotary errors. Scalar RoPE in place
of m-RoPE and the contiguous layout each left 2 of 3 images identical and PARKED the third on a near-tie (steps 3 and
22, gaps 0.017 and 0.007). So a wrong rotary would surface here only as a parked near-tie; G1 (theta 10) is the evidence
for the layout, G2 for the tower + splice + long decode.
**Not done: the 9B leg.** Its safetensors are not on this box (`~/models` holds only the GGUF). Prepared as a night job:
`run-g2-9b.sh` (pinned revision, ~19 GB download, HF f32 reference ~36 GB RAM, goinfer f32 ~36 GB, est. 90–120 min),
test `TestQwen35VLReal_G2_9B`, assets registered. Not queued — see the report.

## G4 — serve (PASS, with the reuse arm narrowed as pre-registered)

`TestServe_qwen35Image_G4`, real 0.8B through `newServer` → `POST /v1/chat/completions` with an OpenAI `image_url`
data-URI part, temperature 0, `max_tokens` 32:
(a) serve's prompt ids equal HF's ids built from Qwen3.5's own template **minus its empty think block** (see finding
below), on all three images; the reply text equals HF's 32 greedy tokens decoded, on all three; (b) `usage`
prompt/completion tokens equal the id counts; (c) the same request twice returns identical text. The tower loaded on
the first image, not at startup (asserted). 303 s for 3 images × (1 prompt build + 2 requests), overlapping other work.
The gate went red on its first run — serve's ids were 4 tokens short of HF's — which is what found the finding below.
**Reuse arm:** `TestResidentReuseLen_recurrentImageClaims` (12 rows, three branches mutation-checked) and
`TestGenerateQwenVL_recurrentTakesNoResidentBranch`. The end-to-end "reused == cold bitwise for an image turn" is NOT
gated: on CPU decode no image turn populates the resident cache, so it would pass vacuously. It belongs to the
resident-image step.

## Findings, in order of how much they matter

1. **Serve's Qwen3.5 text prompts differ from the model's own template.** Qwen3.5's `chat_template.jinja` appends
   `<think>\n\n</think>\n\n` after the generation prompt unless `enable_thinking` is true (default: thinking OFF).
   goinfer renders every Qwen family through its generic ChatML renderer (`chat/templates.go`), which emits nothing
   there — right for Qwen3 (whose default is thinking ON), wrong for Qwen3.5. So a text-only Qwen3.5 request through
   serve puts the model in a state HF's default does not, on every turn, not just image turns. NOT changed here: doing
   so alters every Qwen3.5 text turn, which P8a's scope promise ("the image work must not change a text turn") ruled out. *(Corrected 2026-09-30: this was worded as "which G3 says must not move", but G3's test hashes logits for fixed token ids and never touches the renderer — a default change cannot fail it. See `docs/tasks/task-qwen35-think-prompt-2026-09.md` § What flipping the default changes.)* Pinned as a named 4-token delta in G4. Needs an
   owner decision.
2. **A resident executor would have silently served zero-state image decode.** Every executor implements
   `ResidentMRoPE`; the CPU-prefill → `UploadKV` bridge skips KV-less DeltaNet layers. Guarded and tested.
3. **`LoadQwenPreprocessConfig` silently mis-reads a Qwen3.5 config** (defaults to Qwen2.5-VL's pixel budget). New
   loader refuses instead; the old one is untouched.
4. **A pixel-exact preprocessing claim needs HF's fused normalize** (`(x − 255·mean)/(255·std)`); the obvious form
   differs in 111 of 256 byte values.
5. **Two of my own gates were blind until I mutated the code** (A1: erf vs tanh merger; A2: rotary layout and decode
   delta at theta 1e7). Both were fixed in the fixture and pre-registered as amendments; G2 at real theta remains weak
   to rotary errors (above).
6. **Image speed on the CPU path is poor and unmeasured.** Sequential hybrid prefill; serve caps an image at 1024
   merged tokens (`qwen3MaxImageTokens`) because the checkpoint's own ceiling is 16384. A build-time constant.

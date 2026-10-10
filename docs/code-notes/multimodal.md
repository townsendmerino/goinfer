# multimodal: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `multimodal`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## DecodeWAV

Moved from `multimodal/audio.go` (the comment above `DecodeWAV`) on 2026-10-10.

```text
DecodeWAV reads a RIFF WAV of 16-bit PCM, mono, at 16 kHz into samples in [-1, 1] (s/32768). Any other format,
channel count or rate is refused: serve does not resample yet. DecodeWAVAnyRate does, but G-S5e
(docs/tasks/task-multimodal-support-2026-10.md) read FAIL against scipy's resampler, so it is not wired in; the
owner decides.
```

Note added when this moved (2026-10-10): "not wired in; the owner decides" is no longer true. G-S5e passed with option (b) after this
comment was written (`docs/tasks/task-multimodal-support-2026-10.md`, the S5 follow-up), and `internal/serveapp/vision_serve.go` calls
`DecodeWAVAnyRate` for chat audio. The code comment now says what each function accepts and nothing more.

## Gemma3PromptBlock

Moved from `multimodal/gemma3_block.go` (the comment above `Gemma3PromptBlock`) on 2026-10-10.

```text
Gemma3PromptBlock is the text a Gemma 3 image turn actually splices into the prompt: n soft
tokens wrapped in Gemma3ImageBlock, itself wrapped in "\n\n" on both sides (M-38,
audit-2026-09-10) — Gemma 3's own processor (processing_gemma3.py, verified against the real
transformers source) does exactly this: f"\n\n{boi_token}{image_tokens}{eoi_token}\n\n". Gemma's
SPM vocab has distinct \n\n/\n\n\n pieces, so the id stream around the sentinel differs from
HF/llama.cpp's without the wrapping. Shared by internal/serveapp and demo/agent so both build
the identical shape.
```

## Gemma4TowerAccelerator

Moved from `multimodal/gemma4_tower.go` (the comment above `Gemma4TowerAccelerator`) on 2026-10-10.

```text
Gemma 4 vision tower accelerators (docs/multimodal.md, "Finishing this doc", F2). A backend module (goinfer/metal)
registers a device tower built from aikit's export (vision.Gemma4Encoder.Weights) here, and every Gemma 4 tower
consumer can use it: Gemma 4's own image input in serve and EmbeddingGemma 2's image embeddings. The tower runs the
patch embed and the encoder layers; the pool and projection after them stay aikit's (Gemma4Encoder.FinishHidden).
```

## glm_ocr.go.header

Moved from `multimodal/glm_ocr.go` (the comment above `glm_ocr.go.header`) on 2026-10-10.

```text
GLM-OCR (zai-org/GLM-OCR, model_type glm_ocr) image plumbing: the prompt block, the task prompts and the
preprocessing config. The pixel path itself is QwenPreprocess unchanged — GLM-OCR's Glm46VImageProcessor is the
same smart-resize + CLIP-normalize + merge-block patchify the Qwen towers take (docs/tasks/task-glm-ocr-2026-10.md
O0), so only the config differs.
```

## GlmOcrPrompts

Moved from `multimodal/glm_ocr.go` (the comment above `GlmOcrPrompts`) on 2026-10-10.

```text
The task prompts the model card lists. The user turn is the image followed by exactly one of these (the card's three;
Ollama's page also names "Figure Recognition:", which the card does not). The information-extraction prompt is a JSON
template, not one of these (task O5).
```

## GlmOcrExtractionInstruction

Moved from `multimodal/glm_ocr.go` (the comment above `GlmOcrExtractionInstruction`) on 2026-10-10.

```text
GlmOcrExtractionInstruction opens the model's information-extraction prompt (the model card's own, verbatim: "output
the information in the image in the following JSON format"). It is followed by a newline and a JSON TEMPLATE, an
object whose values are empty strings, NOT a JSON Schema (O0, docs/tasks/task-glm-ocr-2026-10.md); build the template
with constrain.TemplateFromSchema / TemplateFromStruct so the prompt and the grammar come from one source.
```

## GlmOcrExtractionText

Moved from `multimodal/glm_ocr.go` (the comment above `GlmOcrExtractionText`) on 2026-10-10.

```text
GlmOcrExtractionText applies the O5 rule for a request that asks for schema-bound output ON AN IMAGE: the template
prompt REPLACES the text the user sent only when that text is empty (nothing but whitespace) or is exactly one of the
three bare task prompts (a client that always sends "Text Recognition:" next to a schema means "read the image into
this schema", and the task prompt would contradict the grammar). Any other text is the user's own prompt (their own
extraction prompt, a question, a different instruction) and is used UNCHANGED: the grammar still constrains the
reply, but the user owns what the model is told. replaced reports which happened.
```

## LoadGlmOcrPreprocessConfig

Moved from `multimodal/glm_ocr.go` (the comment above `LoadGlmOcrPreprocessConfig`) on 2026-10-10.

```text
LoadGlmOcrPreprocessConfig reads a GLM-OCR checkpoint's preprocessor_config.json into the QwenPreprocessConfig that
QwenPreprocess takes.

THE PIXEL BOUNDS ARE HALVED, ON PURPOSE. The file says size.shortest_edge 12544 and size.longest_edge 9633792, and
LoadQwen3PreprocessConfig would pass them through as MinPixels/MaxPixels. But Glm46VImageProcessor calls smart_resize
with num_frames = temporal_patch_size = 2 and tests t_bar·h_bar·w_bar against the bounds, so they bound TWO frames of a
still image: one frame's budget is bound / temporal_patch_size (6,272 .. 4,816,896 px, i.e. at most 24,576 patches and
6,144 image tokens, not the 12,000 the unhalved bound would allow). Passing the file's values unhalved silently doubles
the budget and gives a larger grid than HF's for any image over ~4.8 MP (task doc §1 correction, O0). The division is by
the file's own temporal_patch_size, not a literal 2. TestLoadGlmOcrPreprocessConfig_halvesPixelBounds pins this and
fails if the unhalved values come through.

Everything else is LoadQwen3PreprocessConfig's: patch 14, merge 2, temporal 2, CLIP mean/std, and the required-field
refusals (a file missing the size keys is an error, not a default).
```

## HashImageBytes

Moved from `multimodal/imagehash.go` (the comment above `HashImageBytes`) on 2026-10-10.

```text
HashImageBytes returns a content hash of raw image bytes, for P9(a)'s resident image-block
reuse check (decoder.GenerateVL/GenerateQwenVL's imgHash parameter): a plain non-cryptographic
64-bit hash is enough at agent-loop scale — there is no adversarial-input concern, only "is
this the same image byte-for-byte as the one already sitting in the resident KV cache."
```

## pixtralMergerPositionMajor

Moved from `multimodal/pixtral.go` (the comment above `pixtralMergerPositionMajor`) on 2026-10-10.

```text
pixtralMergerPositionMajor, when set, lays each merge unit out position-major (the 4 patches' vectors one after the
other) instead of HF's channel-major unfold: S10's planted defect for the merger.
```

## LoadProjector.normW

Moved from `multimodal/projector.go` (the comment above `LoadProjector.normW`) on 2026-10-10.

```text
Validate the norm tensor's length like the projection tensor below: Forward indexes normW by
vision-hidden channel, so a checkpoint whose config says visionHidden=N but ships a shorter
mm_soft_emb_norm.weight would load cleanly and then panic "index out of range" inside the HTTP
handler on the first image request, killing the process (audit C-23).
```

## Projector.Forward

Moved from `multimodal/projector.go` (the comment above `Projector.Forward`) on 2026-10-10.

```text
A config with mm_tokens_per_image larger than the patch grid makes kernel 0, so the pool
divides by kernel*kernel == 0 and every embedding is silently NaN. Fail loudly instead (N-19).
```

## qwen_preprocess.go.header

Moved from `multimodal/qwen_preprocess.go` (the comment above `qwen_preprocess.go.header`) on 2026-10-10.

```text
Qwen2.5-VL image preprocessing: image bytes -> pre-flattened pixel_values
[n_patches, channels*temporal*patch*patch] + grid_thw (t,h,w in patch units),
matching HF Qwen2VLImageProcessor. The aikit Qwen vision encoder consumes this
directly (it does no preprocessing — see docs/prompts/aikit-qwen25vl-vit.md).

Three stages: smart_resize (round H,W to a multiple of patch*merge within the
pixel budget), resize+rescale+CLIP-normalize, and the spatial-merge patchify
rearrange. The patchify order is HF-exact: patches sequence (block-row,
block-col, merge-row, merge-col), each patch's values (channel, temporal,
patch-row, patch-col). The resize is BICUBIC (qwenBicubicU8, called at the resize site) and is
a no-op when the image is already grid-aligned, so a pre-sized image preprocesses bit-exactly
(TestQwenPreprocess_exact). It is tolerance-matched to PIL rather than bit-exact, because the
coefficients here are float where PIL's are fixed-point.

N-34: this said "the resize here is bilinear" and described PIL-bicubic parity as a future
refinement, after the bicubic path had already landed.
```

Note added when this moved (2026-10-10): `docs/prompts/aikit-qwen25vl-vit.md`, cited above, does not exist in the tree; the pointer was dropped
from the code comment. The "N-34" paragraph recorded a correction already made (the comment had said the resize was bilinear).

## QwenPreprocessConfig.FusedNormalize

Moved from `multimodal/qwen_preprocess.go` (the comment above `QwenPreprocessConfig.FusedNormalize`) on 2026-10-10.

```text
FusedNormalize selects HF's torchvision-backend arithmetic, (x - mean*255)/(std*255) on the
0..255 value, instead of (x/255 - mean)/std. The two differ in the last bit for 111 of the
256 byte values at mean = std = 0.5 (measured), so a "bit-exact vs HF" claim needs the
fused form. Set by LoadQwen3PreprocessConfig; the Qwen2.5-VL loader leaves it false, so that
path's pixel_values are unchanged.
```

## qwen35FamilyPreprocessorJSON

Moved from `multimodal/qwen_preprocess.go` (the comment above `qwen35FamilyPreprocessorJSON`) on 2026-10-10.

```text
qwen35FamilyPreprocessorJSON is the preprocessor_config.json every Qwen3.5/3.6 checkpoint ships, byte for byte (the
same MD5 on Qwen3.5-0.8B, Qwen3.5-9B and Qwen3.6-35B-A3B, read 2026-10-06). A GGUF mmproj carries no pixel budget
(docs/multimodal.md P8b, F5 Phase 0), so a tower loaded from one takes the family's own processor config from here.
```

## qwenMaxInputPixels

Moved from `multimodal/qwen_preprocess.go` (the comment above `qwenMaxInputPixels`) on 2026-10-10.

```text
qwenMaxInputPixels bounds the raw (pre-resize) image area QwenPreprocess will allocate for,
independent of cfg.MaxPixels (which caps only the resized output). ~33.5 MP: above any real
camera image, below the memory a decompression bomb would demand (audit M-15).
```

## QwenPreprocess.bomb

Moved from `multimodal/qwen_preprocess.go` (the comment above `QwenPreprocess.bomb`) on 2026-10-10.

```text
Peek the declared dimensions from the header BEFORE decoding pixels.
image.DecodeConfig only parses the header (PNG's IHDR chunk / JPEG's SOF
marker) — unlike image.Decode, it never allocates or fills a pixel buffer.
A decompression bomb (a few KB of highly-compressible data declaring a huge
canvas) must be rejected here: image.Decode itself allocates the full
decoded pixel buffer (e.g. ~1.2 GB for a 10000×10000 PNG) as part of
decoding, before any check on img.Bounds() could run (audit M-15 originally
only guarded the later qwenExtractRGB/qwenBicubicU8 allocations, missing
this earlier and larger one).
```

## qwenSmartResize.aspect

Moved from `multimodal/qwen_preprocess.go` (the comment above `qwenSmartResize.aspect`) on 2026-10-10.

```text
N-81 (docs/audit-2026-09-10.md, re-verified 2026-09-16, no action needed): HF's
smart_resize raises ValueError past aspect ratio 200 instead of returning a
resized shape at all; this function has no error return (int, int) only, and
changing that ripples through every caller for what upstream itself treats as a
malformed-input rejection, not a resizing decision. max(factor, …) below is the
deliberate choice already made here: an extreme ratio floors a dimension to 0 →
empty pixel_values with no error, so flooring to one cell trades an HF-parity
crash for a degraded-but-non-crashing grid on an input real models don't produce.
Kept as documented, acknowledged behavior rather than added scope this batch.
```

## qwenExtractRGB

Moved from `multimodal/qwen_preprocess.go` (the comment above `qwenExtractRGB`) on 2026-10-10.

```text
qwenExtractRGB reads img into a [h*w*3] 0..255 channel-last float buffer. The
>>8 recovers the 8-bit channel value exactly for an 8-bit (PNG/JPEG) source.

P-19: the generic img.At(x,y).RGBA() path dispatches through the image.Image
and color.Color interfaces per pixel — measured ~1-2s per image at the vision
preprocessing cap. *image.YCbCr (Go's decoded-JPEG format — the common case
for real photos) and *image.RGBA get a direct fast path reading the backing
buffer instead. Both are PROVABLY bit-identical to the generic path, not just
close: image.RGBA.At() computes uint32(pix)*0x101, and (pix*0x101)>>8 == pix
exactly for any pix in [0,255] (pix*257 = pix*256+pix, and the >>8 term
vanishes since pix<256) — so reading Pix directly and skipping the >>8
round-trip gives the same result. image.YCbCr.At() calls color.YCbCrToRGB
then wraps it in color.RGBA, so calling color.YCbCrToRGB directly and
skipping the same round-trip is exactly the same identity. Every other
concrete type (NRGBA's alpha premultiplication makes a naive buffer read
WRONG for a transparent pixel, so it is deliberately not fast-pathed) falls
through to the generic loop unchanged.
```

## resample.go.header

Moved from `multimodal/resample.go` (the comment above `resample.go.header`) on 2026-10-10.

```text
Audio towers take 16 kHz mono (S5's follow-up, docs/tasks/task-multimodal-support-2026-10.md, G-S5e). A WAV at another rate
or with several channels is brought to that here: the channels averaged per frame (librosa's to_mono convention), then
resampled with scipy.signal.resample_poly's own filter, built the same way (owner decision 2026-10-07, G-S5e option (b)):
up and down reduced by their gcd, a firwin low-pass at 1/max(up, down) of the upsampled Nyquist with unit DC gain, a Kaiser
window of beta 5.0 over 20·max(up, down)+1 taps, centred and scaled by up, zero padding, ceil(n·up/down) outputs.
TestResample_matchesResamplePoly holds it to scipy's output (testdata/resample_poly_golden.json).
```

## splice.go.header

Moved from `multimodal/splice.go` (the comment above `splice.go.header`) on 2026-10-10.

```text
This is the one image-block splice every vision caller shares (goinfer-serve's vision routes, goinfer-chat --image, the
examples). It lived in internal/serveapp until O5 gave it a second and third caller; serveapp's spliceImageBlock is now a
one-line delegate and keeps its tests.
```

## SpliceImageBlock

Moved from `multimodal/splice.go` (the comment above `SpliceImageBlock`) on 2026-10-10.

```text
SpliceImageBlock re-tags the image block as its own Special segment, splitting the content
segment that contains it.

The block does NOT start its segment: the template renders the role prefix and the message body
into one non-special span, so a Gemma-4 user turn arrives as "user\n<start_of_image>…<end_of_image>\nhello".
The first cut of this looked for the block as a PREFIX, found it nowhere, and would have refused
every vision request — caught by the test, which is why the test drives this function rather than
re-implementing its logic beside it.
```

## SpliceImageBlock.lastOccurrence

Moved from `multimodal/splice.go` (the comment inside `SpliceImageBlock`) on 2026-10-10.

```text
V-19 (docs/review-2026-09-04.md): search from the END and splice the LAST occurrence, not
the first. The block is always appended to the CURRENT (last) user turn, which renders
last; an EARLIER turn that happens to contain the same literal text — a user asking what
the sentinel means, say, as ordinary words — must not be mistaken for it. Splicing the
wrong occurrence reopens the exact special-token-forging class M-22 closed, just for this
sentinel instead of a role marker: the real image tokens stay unspliced plain text (and
fail downstream with a misleading "template mismatch"), while unrelated earlier text gets
tagged Special and parsed as sentinels it was never meant to be.
```

## voxtral.go.header

Moved from `multimodal/voxtral.go` (the comment above `voxtral.go.header`) on 2026-10-10.

```text
Voxtral Mini's audio constants (docs/tasks/task-multimodal-support-2026-10.md, S14.4b, desk read 2026-10-09).
```

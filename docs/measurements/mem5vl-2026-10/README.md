# MEM5-VL: chunked image prefills, 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, "MEM5-VL" in §4 MEM5, each route's registered before its code. Machine: MacBook
Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2. Tier 0 only so far; the tier-1 peaks are for the morning, by day.

## Which prefills hold the whole prompt

On the CPU, a text prompt on a `manualPos` family (Gemma 4, the Qwen3.5 hybrids, LFM2, Granite-H, Nemotron-H, MLA, KDA) never runs a
whole-prompt pass. `canBatchN` is false for every one of their tiny fixtures (checked on 14 fixtures at f32 and int4), so they prefill
a token at a time. The whole-prompt passes `KVCache.prefillChunkable` left single are the image and audio prefills. They fall into
four routes, ordered by families served: (1) the shared batched route, (2) Gemma 4's bidirectional image prefill, (3) Qwen3.5's image
prefill, (4) audio. LFM2-VL and Gemma 4's causal image prefill already run one row at a time.

## Route 1: the shared batched route (V1-G)

**The build.** The chunk loop G40 gave `prefillLogits` is now `prefillPassesN`. The text prefill calls it, and so do the image
prefills of `prefillHiddenVLSpans` (Gemma 3, Pixtral/Ministral 3) and `prefillLogitsQwenVLSpans` (Qwen2.5-VL, Qwen3-VL dense and MoE,
GLM-OCR). Each pass embeds its own rows and splices the image features that fall in them (`spliceImageSpansRange`), so the whole
prompt's embeddings are no longer held either. `KVCache.prefillCut` moves a cut out of a bidirectional image block: back to the
block's start, or forward to its end when the block starts the pass. `prefillChunkable` no longer excludes image blocks, m-RoPE
positions or DeepStack rows: every pass reads them at absolute positions (`mropePos[seqPos]`, `attendHi(pos)`, DeepStack by
`startPos`). A sliding-window cache under fast attention keeps the single pass (MEM5-ring, G43). That covers Gemma 3's long image
prompts whenever fast attention is on.

| gate | reading | result |
|---|---|---|
| V1-G1 identity | `TestMEM5VL_sharedRouteChunked`: a 60-token prompt with a 12-row image span (longer than a 7-row chunk, crossing the 16-row cut) and a 4-row one, at chunk 7 and 16, at f32 and int4, on gemma3-vl-tiny (bidirectional), ministral3-tiny (causal), qwen25vl-tiny and glm-ocr-tiny (m-RoPE), qwen3vl-tiny and qwen3vlmoe-tiny (m-RoPE and DeepStack): every position's rows, the last logits and four decode steps after equal one pass's in every bit, in 9 passes at 7 rows and 4 to 5 at 16 | **PASS** (12 cells) |
| V1-G2 bytes | `TestMEM5VL_sharedRouteBytes`, a 200-row prompt with a 20-row first span, 16-row chunks: the pass's row-sized buffers 563,200 B in one pass → 56,320 B on gemma3-vl-tiny (the 20-row block's pass) and 45,056 B on qwen25vl-tiny (16 rows; its image rows are causal), each exactly the single pass's scaled to its longest pass; the key-sized attention buffers (`alk`, `alv`, sized by the whole prefill's keys as G40 amended) equal in both, 51,200 B on Gemma 3; every chunk after the first allocates at most 86 KB (bar 1 MiB) | **PASS** |
| V1-G3 unchanged | the decoder tests matching the image routes, the G40 chunk gates and the prefill tests: ok in 947 s | **PASS** |
| V1-G4 peak, tier 1 | Qwen2.5-VL-3B with one image and about 8K text tokens, main against the branch | open: by day after 07:00 |

**Plant** (the cut rule off): red. gemma3-vl-tiny differs in 3,520 rows and 1,305 logits at both chunk sizes.

## Route 2: Gemma 4's bidirectional image prefill (V2-G)

**The build.** `prefillLogitsGemma4VLBidirectionalSpans` runs the prompt in passes of `prefillChunkRows`. Each pass embeds and splices
its own rows and is cut outside the image blocks (`cutOutsideBlocks`, route 1's rule over the spans). It reserves the prefill's KV
after the first pass, as the text path does. `runLayersGemma4FromEmbedN` reads a row's image membership at its absolute position, so
a pass may start after earlier ones. Its attention is the exact single-row kernel over absolute key ranges, so MEM5-ring's
sliding-window question does not arise. Gemma 4's causal image prefill already ran one token at a time and is unchanged.

| gate | reading | result |
|---|---|---|
| V2-G1 identity | `TestMEM5VL_gemma4BidirChunked`, gemma4-vl-bidir-tiny, route 1's two-span prompt at chunk 7 and 16, f32 and int4: the last logits and four decode steps after equal one pass's in every bit, in 9 and 5 passes | **PASS** |
| V2-G2 bytes | `TestMEM5VL_gemma4BidirBytes`, 200 rows with a 20-row block, 16-row chunks: one pass allocates 17.27 MB; the largest chunked pass 2.06 MB (bound 2.61 MB: 20 rows' share, the attention scratch at 200 keys, 64 KiB) | **PASS** |
| V2-G1 and V2-G2 where they run | gemma4-vl-bidir-tiny is gitignored (`scripts/pin_gemma4_vl_bidir_tiny.py` makes it), so CI skips both tests, as it skips the existing bidirectional tests; they ran on the Mac with the fixture present | note |
| V2-G3 unchanged | `TestGemma4VLBidir_textParity`, `TestGemma4VLBidir_imageParity` (HF goldens), `TestTwoImages_gemma4Bidir` | **PASS** |
| V2-G4 peak, tier 1 | a bidirectional Gemma 4 (26B or 31B class), one image and about 8K tokens | open: no such checkpoint in the Mac's bench set |

**V2-G2's instrument, corrected after its first reading.** The first reading put the first pass at 5.87 MB, against a 3.0 MB bound.
A memory profile showed it was the cache: `KVCache.Append` allocating each layer's KV on its first write (about 4 MB), and in an
earlier reading the cache's own creation, which the test made after its baseline. Both are the cache's storage, not the pass's
working set, which is what the gate bounds. The instrument now creates the cache before the baseline and leaves out of each pass's
reading the KV arrays it reallocated. The bar did not move.

**Plant** (the cut rule off): red. A pass ending inside the block reads keys past the end of the cache (a slice-bounds panic).

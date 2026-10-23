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

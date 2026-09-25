# gemma4 GGUF transcode streams — S2's last family (2026-09-24)

`docs/tasks/task-never-swap-2026-09.md` S2. Raw data: `transcode-streaming-gemma4-2026-09-24/`.

## What changed

`buildWeightsFromGGUF`'s gemma4 branch drives the per-layer sink like every other family, and
`needsResidentSerialize` — the list that routed families around streaming — is deleted with its last entry. The
"fused PLE/MoE tail" that kept gemma4 on a resident build does not exist in this format: its model-level PLE inputs
(`per_layer_token_embd` / `per_layer_model_proj` / `per_layer_proj_norm`, plus `FFNPerLayer`) are part of the HEAD
(`writeHeadGlobals`, written before any layer), and every other gemma4-specific field — PLE gate/proj/norm, the
layer scalar, the KV-shared and K=V flags, the whole 26B-A4B MoE branch — is in each layer's own record and read from
`blk.{i}.*` only. Three details made it work:

1. the streaming writer is given the architecture (`sink.arch`), which gates the gemma4 head extras and per-layer
   record — the resident path set it via `writeBundle`, the streaming path never did;
2. gemma4 writes its head inside its own branch, after the PLE globals are loaded (the generic head is written before
   any family branch runs);
3. after the head is written, the model-level globals are released (the 26B-A4B's 262k-vocab int8 embedding alone is
   ~0.74 GB), and a streaming build constructs a layer's experts in parallel (`stackedExperts`, `sink != nil` only —
   the resident path already runs one layer per core and is unchanged).

## Gates

- **Byte identity, fixtures** (`decoder/gguf_gemma4_stream_test.go`): synthetic gemma4 GGUFs in two shapes — "26b"
  (parallel dense+MoE FFN, a global layer with no `attn_v`) and "e2b" (PLE, a KV-shared tail layer, per-layer FFN widths),
  both with a sliding/global pattern of different head dims and KV-head counts — streamed vs the former resident path,
  at int4 / int8int8 / f32 × default / metal target: **12/12 byte-identical** outside the quant label (which a streamed
  body records as `""` by design). Mutation-checked: without `sink.arch` all 12 fail (the streamed body is 38% short);
  with the head written before the PLE loads, all 6 e2b cases fail. A pre-cancelled context writes nothing.
  `TestStreamableFamilyClosures_onlyReadPerLayerTensors` now covers `loadG4` and passes (no model-level read inside it).
- **Byte identity, the real checkpoint** — the 26B-A4B Q4_K_M `.gguf`, `-quant int4 -target metal`, on nobara: the
  streamed bundle equals the resident one in every byte of the weights body before the label and **all 16,190,667,177
  bytes after it**, and in the tokenizer section; the files differ only by the label (16 B).
- **Peak RSS ≤ 1.5 GB above baseline** (RssAnon sampled every 200 ms — anonymous memory; RSS proper also counts the
  mmap'd `.gguf` pages, which are file-backed and reclaimable):

  | 26B-A4B transcode | wall | CPU | anonymous peak | RSS peak |
  |---|---|---|---|---|
  | resident (HEAD `8f452a7e`) | 1:15.6 | 577% | 18.14 GB | 34.8 GB |
  | streaming, first cut (sequential, globals held) | 6:07 | 93% | 2.58 GB (+2.35) — **over the bar** | 17.2 GB |
  | **streaming, shipped** | **1:49.6** | 409% | **1.61 GB (+1.41)** | 14.7 GB |

  The shipped peak is at t=10 s, while the embedding table is built for the head (identical work in the resident
  path); once the globals are released the layer phase runs at 0.6–1.25 GB. **Met, narrowly** (+1.41 against +1.5).
- **Wall time: 45% slower than resident** (1:50 vs 1:16). The brief's Measure section expected no slowdown ("the work
  is the same, reordered"); the reorder costs the across-layer parallelism, and parallel experts recover most but not
  all of it. Recorded, not chased: this is a one-time transcode, and the resident path is not available on the machine
  that needs streaming.

## S1's gate — the sidecar vs the direct load — blocked by a direct-load bug, then FIXED and PASSED

On nobara (CPU), 16 greedy tokens of "The capital of France is", `-quant int4 -ctx 1024` (the default ctx's KV made the
fit guard refuse the direct load; it was not bypassed): the streamed sidecar generates
`' Theer.\n\n<|channel>thought\n<channel|>The statement is **incorrect**. The capital'`; the **direct** `.gguf` load
generates **16 `<pad>` tokens** — and so does a serve built from HEAD `8f452a7e`, before any of this change (the direct
load never passes through the streaming path). So a direct CPU load of the gemma4-26B GGUF is broken on `main`
independently of S2; the sidecar path is not.

**Cause and fix (same day).** `loadG4` built each layer's `gemma4MoEWeights` — which copies `l.LayerScalar` into its
own `layerScalar` — *before* it assigned `l.LayerScalar` (1, or `blk.{i}.layer_output_scale.weight`). The MoE forward
multiplies the whole layer's output by that copy (`out = (h + comb) * layerScalar`), so every MoE layer of a
directly loaded 26B-A4B was scaled by 0 and the argmax fell to token 0, `<pad>`. The `.giw` reader and the safetensors
loader already read the scalar first, which is why the sidecar was fine; Metal's resident build copies the same field,
so a direct GGUF load there was broken the same way (rarely hit: darwin defaults to the sidecar). Fixed by reading the
scalar before the MoE branch. `TestGemma4GGUF_moeLayerScalarMatchesLayer` pins it on the synthetic 26B-shaped GGUF —
red on the old code (every layer's MoE copy 0 against a layer scalar of 34–99), green after. Existing sidecars are
unaffected (the writer serializes `l.LayerScalar`, never the copy) and need no rebuild.

**S1's gate, run after the fix** (`s1gate-after-fix/`, nobara, `-quant int4 -ctx 1024`): serve's default path — the
`.gguf` transcoded to its `cpu-amd64` sidecar on first use, **streamed** (S2), 1:00 — against `-direct-load`, 3 prompts ×
64 tokens greedy: **3/3 byte-identical** (prompt 1 stops at its own EOS after 22 tokens). **PASS.**

## Not done

A transcode on the Mac itself (the source `.gguf` is 16.8 GB and the Mac has ~19 GB free, so source + output does not fit
until something moves); three-run RSS averaging (n=1 per arm).

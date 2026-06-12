# Task (goinfer): ring-buffer KV storage for sliding-window layers (lossless)

> **For:** `~/tmcode/goinfer`, CPU path (`decoder/kvcache.go`). Companion to
> `task-cpu-kv-quant.md` — the two compose (quantized ring slots) but ship
> independently. **This one is LOSSLESS**: outputs are already provably
> invariant to evicted keys (`TestMellum2_slidingWindowEviction`,
> `gemma_window_golden`), so the gate is **bit-exact**, not argmax+cosine.
> Arguably it should land *before* quantization: free memory with zero
> precision argument beats cheap memory with one.

## Problem

"Sliding-window eviction" in goinfer today is **masking, not eviction**.
`WindowStart` bounds what attention *reads*; `Append` (kvcache.go:59) stores
every position on every layer forever. A local layer with window 1024 at 32k
context stores 32k positions and reads 1024 — **31/32 of its cache is dead
weight that can never influence any future token** (that invariance is exactly
what the existing window tests pin).

Who pays, at 32k context:

| family | pattern | window | live fraction of KV | wasted |
|---|---|---|---|---|
| Gemma 3 / Gemma 4 dense | 5 local : 1 global | 512–1024 | 1/6 + (5/6)·(W/32k) ≈ **19%** | **~81%** |
| Mellum2 | 3 local : 1 full | 1024 | ≈ 27% | ~73% |
| Mistral (all-sliding cfgs) | all local | 4096 | W/ctx = 12.5% | ~87% |
| full-attention families (Qwen2/3, Llama) | — | — | 100% | 0 (not their fix) |

For the Gemma E-models — the README's headline demo — this is a **~5×
long-context KV reduction with zero quality cost**. llama.cpp shipped the
same idea as its SWA cache; goinfer has the masking half already proven and
just keeps the storage.

## Design

### Ring storage on local layers only

Global (and KV-shared, and DeltaNet) layers keep today's append-forever
arrays. A local layer allocates a fixed ring of `W` slots
(`window × layerKVDim`):

```go
// per local layer, replacing the unbounded append:
ring     []float32 // [W * layerKVDim], slot = pos % W
ringBase int       // oldest absolute position still resident
```

- `Append`: write into slot `pos % W`, bump `ringBase = max(0, pos-W+1)`.
  O(kvDim), no copy, no allocation after warmup — strictly cheaper than
  today's `append` (which reallocs on growth).
- Read path: `attendQuery` / `attendBatchedHeads` already iterate
  `s ∈ [WindowStart(pos), pos]`; the only change is index mapping
  `s → (s % W) * stride` instead of `s * stride`. `WindowStart ≥ ringBase`
  holds by construction, so every read hits a live slot.
- Per-layer geometry (Gemma 4 per-layer widths, zero-width KV-shared layers)
  already forces per-layer stride derivation; the ring follows the same rule.

### The two seams that need care

1. **`TruncateTo` (spec-decode rollback + prefix-reuse rewind).** A rewind to
   `p` is exact iff every local layer still holds `[max(0,p−W+1), p)` — i.e.
   `p ≥ ringBase` … `p+W` window intact. Mechanically: rewind is allowed iff
   `p ≥ maxOverLocalLayers(ringBase)`; otherwise `TruncateTo` reports
   unrewindable (new boolean return or error) and the caller cold-prefills.
   - Speculative rollback: draft lengths ≪ W ⇒ always safe. Property test.
   - Session prefix reuse (`cmd/serve/sessions.go`): the LRU picks the longest
     shared prefix; if that prefix is deeper than the ring remembers, fall
     back to full prefill — today's behavior for a cache miss, just a new
     reason for it. Net: reuse keeps working for the common
     "append-to-conversation" case (rewind depth 0 or tiny).
2. **`.giw-kv` snapshots** (kvsnapshot.go): persist `ringBase` + the live
   window per local layer instead of the full history (snapshot v2 field —
   coordinate the version bump with task-cpu-kv-quant Inc 3 so the format
   bumps once, not twice). Restored sessions get the same rewind rule.

Image-block (multimodal) prefill: bidirectional blocks attend within
`[start,end)`; a block wider than W on a local layer would be a correctness
trap — but that can't happen, because the *read* mask already can't see past
W today (`attendHi` caps at the block, `WindowStart` at the window). Eviction
changes storage, not visibility. Assert it anyway in the property test.

### Knob

None needed — it's lossless and strictly better. (Escape hatch behind a
build-time or Options bool for one release if paranoia wins; delete after.)

## Increments

1. ✅ **DONE (commit 4a9c2b5).** Ring storage + decode/prefill index mapping,
   full-attention families untouched; gemma4 + qwen3_5_moe deferred (per-layer
   widths / linear attention — their own forward). **The doc's "just map s→s%W"
   is decode-only**: batched prefill writes all K then reads, so a prompt wider
   than W (the Mellum2 golden is 1441 > 1024) would evict in-batch history a
   naive ring still needs. Resolution: decode dense reads the ring directly
   (`s%W`); batched prefill + MoE decode DEFER the ring write, assemble a
   contiguous `[base, startPos+K)` window (ring history + the K new rows), attend
   with a base offset (`s−base`), then commit. `attendBatchedHeads` gained a
   `(keys,vals,base)` arg — global layers pass `cache.Keys/0` (bit-exact).
   Gates (green): real-model Mellum2 logit + window parity (MoE+sliding, both
   paths, argmax exact / cosine 0.998); three model-free bit-exact gates
   (`decoder/ring_test.go`: decode 4×W wrap, batched K=19≫W=4, MoE K=1
   multi-step) vs append-forever; local-layer bytes == `W·stride` at any context.
2. **TruncateTo rewind rule + spec-decode/session property tests**; sessions
   fall back gracefully. Gate: spec-decode suite green under rings; a
   deeper-than-ring rewind triggers cold prefill, never wrong output.
3. **Snapshot v2 windowed persistence** (with the quant doc's version bump).

## Estimated impact

| metric | today | with rings |
|---|---|---|
| KV RAM, Gemma-class @ 32k | 100% | **~19% (≈5.2×), bit-exact** |
| KV RAM, Mellum2 @ 32k | 100% | ~27% (3.7×) |
| KV RAM, Qwen2/Llama (full attn) | — | unchanged (use kv-quant) |
| stacked with int8 KV (both docs) | — | **~20× on Gemma local layers** |
| decode tok/s | — | neutral-to-slightly-better (no realloc/copy growth; same bytes read) |
| `.giw-kv` Gemma session size | O(ctx) | **O(W) on 5/6 layers** |
| quality | — | **zero change — bit-exact gate** |
| prefix-reuse | exact, any depth | exact; rewinds deeper than W cold-prefill (rare) |

## Effort

~3–5 days total (Inc 1 is mostly index-mapping discipline; Inc 2 is where the
thinking already happened above). Risk: low-medium — the invariance is
already pinned by goldens; the rewind seam is the only behavioral change.

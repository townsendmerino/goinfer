# Scoping: HySparse2 (Xiaomi MiMo-V3 architecture)

> **Status:** scoping only. Nothing here is committed work. Drafted 2026-09-23, the day Xiaomi
> published the architecture, following a Francis ↔ Cowork conversation. Nothing runnable
> shipped: this is a paper plus announcements, **no weights and no modeling code**. Every
> architecture fact below comes from the arXiv HTML read through a summarizer, not from
> the PDF directly. Per the claim-discipline rule in `docs/parity-coverage-policy.md`, re-pin
> each one against the PDF (and later against `config.json`) before it feeds a design decision.
> The rows marked **(unverified)** are the ones the summary left ambiguous.

**Verdict:** park until a trigger fires (see "Pickup triggers"). Once weights exist, this
is **medium work on CPU and large on the resident backends**. Most of the model runs on
substrate goinfer already has: MoE, sliding-window attention, partial RoPE, and
cross-layer KV sharing (Gemma 4 E-models, CPU-only). Two ideas are new: the two-level KV
sharing, and token-level top-k sparse attention that reuses another layer's cache and
selection. The design lines up with the current gap: its whole point is cheaper prefill and a
smaller long-context KV cache, and those are the two areas where `docs/tasks/red-october.md`
says goinfer is furthest behind. The practical blocker is size. The published configurations
are 80B-A3B and 290B-A8B, which on the 16 GB Mac means streaming-only from day one.

## What was published, 2026-09-23

| | value | source |
|---|---|---|
| Artifact | paper only, "HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing" | arXiv 2609.26368 |
| Weights | **none released**; MiMo-V3 announced as "will use" this architecture | arXiv; Luo Fuli announcement (Sina); HF XiaomiMiMo org has V2.x only |
| Configurations | 80B-A3B MoE (headline results) and 290B-A8B (ablations) | arXiv; Pandaily |
| Headline | ~5.02× lower prefill FLOPs at 1M tokens vs Hybrid SWA (MiMo-V2.6's architecture) | arXiv; Pandaily; HuggingNews |
| KV at 1M tokens (FP8 KV) | ~2.69 GB vs 6.72 GB (HySparse v1) vs 12.09 GB (Hybrid SWA) | arXiv |
| Structure | self-decoder (full attention + SWA) followed by cross-decoder (full attention + sparse attention); 5 full-attention layers per decoder | arXiv |
| Layer count | "49" per decoder or total. **(unverified)** The summary was ambiguous; pin from the PDF | arXiv |
| SWA | 128-token window, partial RoPE (64 rotary dims, base 10,000) | arXiv |
| Sparse selection | token-level (not block-level): 128 forced most-recent tokens + top-1,024 by score; scores come from the owning full-attention layer ("oracle selection"), **no separate indexer module** | arXiv |
| Baseline (MiMo-V2 Hybrid SWA) | 9 full-attention layers, GQA 64/4, 128-token SWA elsewhere | arXiv |

## The mechanism, in engine terms

**Outer level: KV bridging (the prefill win).** The cross-decoder's full-attention layers
don't project K/V from their own input. Each one applies its own K/V projection to the
*input hidden state of a matching self-decoder full-attention layer*. So once the
self-decoder has run over a prompt, every KV entry the cross-decoder will ever read can be
computed without running the cross-decoder. Prefill = self-decoder over all K tokens +
cheap projections. The cross-decoder runs only for positions that emit logits, which for
plain chat is the last one. This has the same shape as YOCO (You Only Cache Once).

**Inner level: KV reuse.** Within each hybrid block of the cross-decoder, the sparse layers
keep no cache of their own. They read the block's full-attention layer's KV **and its
selection indices**.

**Selection.** The full-attention layer produces attention scores anyway. The top-1,024 of
them (plus the forced 128 most recent) become the index set its block's sparse layers attend
to. No learned indexer is involved, which removes the component that made Qwen3.8-Next's QSA
(`docs/scoping-qwen38-flash-next.md`) and DeepSeek's sparse attention expensive to bring up.

## Substrate map

| Piece | goinfer today | New work | Size |
|---|---|---|---|
| MoE FFN (A3B) | many families; CPU expert pager, Metal/CUDA expert streaming | config mapping; router variant TBD from `config.json` | S |
| SWA, 128 window | yes: `Architecture.SlidingWindow` + `isGlobalLayer` (decoder/arch.go), KV priced at the window in `kvBytesForCtx` | per-layer pattern from config | S |
| Partial RoPE 64 dims | yes (Gemma 4 path, CPU + resident) | none expected | S |
| Cross-layer KV reuse, same decoder | **CPU only**: Gemma 4 E-model `SharedKVLayers` / `kvSrc` in decoder/forward_gemma4.go. `FeatGemma4EModel` (decoder/features.go) makes every resident backend decline | generalize `kvSrc` from "last N layers" to an explicit per-layer source map | M |
| KV bridging (cross-decoder K/V from self-decoder hidden states) | nothing | keep self-decoder hidden states at bridge points during prefill (or project immediately and drop them); new cache-ownership model | M |
| Prefill early exit | nothing: every family runs every layer per prompt token | the prefill driver stops after the self-decoder and runs bridge projections instead | M, and it's the whole reason for the design |
| Token-level top-k selection from a full layer's scores | nothing. Decode attention computes scores but discards them | score retention + top-k per (head or KV group, **unverified**) per query | M on CPU |
| Gathered sparse attention (1,152 scattered positions) | nothing: attention always reads a contiguous range | CPU gather loop; CUDA/Metal gather kernels + a top-k kernel | M CPU, **L** GPU |
| FP8 KV | no: CPU KV is f32/f16/i8 (`KVQuant`, `KVPrecision` in decoder/model.go); `decoder/fp8.go` is weights only | not needed. The KV-size claim holds at i8 or f16 too, just larger | — |

## Where it will bite

1. **Cache-shape predicates.** Everything that sizes, reuses or rewinds the KV cache assumes
   one cache per attention layer:
   - the fit guard (`kvBytesPerPosition` in decoder/fitguard.go, `kvBytesPerPositionAllLayers`
     in decoder/fitplan.go)
   - resident reuse (`residentReuseLen` in decoder/resident_reuse.go)
   - the CUDA pricer (`kvBytesForCap` in cuda/resident.go)

   HySparse2 breaks that twice: bridged layers own a cache but never run over the prompt, and
   sparse layers own none. This is the class of bug the Sep 10 audit found (Bailing KDA state
   missing from every recurrent predicate). The fix is the pattern `hasRecurrentState` already
   adopted after C-03 (decoder/kvcache.go + decoder/recurrent_census_test.go): make a census test
   ask the struct, rather than hand-listing a new kind at each site. **Write that census
   before the forward, not after.**
2. **Selection indices are cache state.** If decode re-selects every token, indices are
   derived and nothing is stored. If selection is cached or reused across steps (the way
   Qwen3.8-Next's MTP reuses QSA block indices), indices become state that prefix reuse and
   rewind must invalidate. Pin which one it is from the reference code.
3. **Prefill early exit versus the prefill paths.** CUDA fast prefill (`attn_fused.cu`,
   `gemm_w4a8_mma.cu`), Metal batched prefill and CPU batched prefill all assume
   "all layers × all tokens." The early exit belongs in the driver, not in each backend.
   Otherwise it's three implementations of one idea.
4. **Oracle.** No HF modeling code exists yet. Until it does there are no golden pins, and a
   synthetic-tiny bring-up can only check self-consistency (sparse path with k ≥ context ==
   dense path over the same KV), not fidelity.

## Hardware reality

At int4, 80B-A3B is about 40 GB of weights. On the M1 Pro 16 GB it's paged-expert territory,
which is exactly where `docs/tasks/task-never-swap-2026-09.md` (S0–S6) is still settling
swap behaviour. It doesn't fit resident on the 8 GB card. The 3B-active decode cost is in the
same class as Qwen3.6-35B-A3B, so throughput would be DMA-bound, like M26/M35 today (P20).
The design's KV and prefill wins land fully only at long context, and long context is where
goinfer currently loses most to Ollama. The fit is real, but only on hardware that can hold
the experts.

## Pickup triggers

Start when **any** fires:

1. **A carrier at goinfer's target sizes**: a HySparse2 checkpoint at or under ~30B total, or
   an A3B small enough to run resident on the 8 GB card with streaming.
2. **Weights + HF modeling code for MiMo-V3 at any size.** The modeling code is the
   reference that pins items 1–2 above and unlocks a synthetic-tiny bring-up with real pins.
3. **Another lab adopts KV bridging / oracle-selection sparse attention** in a model goinfer
   wants anyway. At that point the substrate pays for itself twice.

**Not a trigger:** the paper alone, a 290B release, or third-party GGUF quants without upstream
modeling code.

## Plan shape once triggered (CPU first)

- **H0 Pin.** Re-read the PDF + `config.json` + the modeling code. Settle: the layer count;
  whether selection is per head, per KV group or shared; the re-selection cadence; which
  self-decoder hidden state feeds each bridge; router details. Update this doc's table in
  place.
- **H1 Census first.** Add the per-layer KV-ownership map (owns / borrows-from-L /
  bridged-from-L) and a census test that fails when any sizing/reuse/rewind site ignores it.
  Generalize Gemma 4's `kvSrc` onto it, so the E-model path becomes the first user and is
  regression-tested for free.
- **H2 Dense-equivalent forward (CPU).** Bridging + reuse with selection disabled
  (k = context). Gate: bit-identical to a naive all-layers forward on a synthetic tiny.
- **H3 Selection (CPU).** Top-k + forced window + gathered attention. Gate: k ≥ context equals
  H2 exactly; then golden pins once the oracle route exists.
- **H4 Prefill early exit** in the driver. Gate: last-token logits identical to H3; measure
  prefill time vs the H3 path at K = 512/4k/32k against a projection band derived beforehand
  (the paper's 5× is at 1M tokens and won't hold at 4k).
- **H5 Resident backends.** Metal first (main machine), then CUDA: top-k + gather kernels.
  Declare a new `ResidentFeature` so backends decline until each lands, the way
  `FeatGemma4EModel` does today.

## Not doing

- No speculative work before a trigger. The architecture may change between paper and
  checkpoint.
- No FP8 KV cache just to match the paper's numbers.
- MiMo-V2.x (Hybrid SWA) is **not** a stepping stone worth building on its own: its open
  checkpoints are far above goinfer's target sizes, and its only new piece (9 full + SWA) is
  already covered by existing substrate.

## Sources

- [arXiv 2609.26368 — HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing](https://arxiv.org/abs/2609.26368) ([HTML](https://arxiv.org/html/2609.26368))
- [Pandaily — Xiaomi details HySparse2 for MiMo-V3](https://pandaily.com/xiaomi-hysparse2-mimo-v3-two-level-kv-sharing-prefill)
- [HuggingNews — MiMo-V3 prefill 5.02×](https://huggingnews.com/ai/update-xiaomi-cuts-mimo-v3-prefill-compute-502x-with-hysparse2-architect-1fc708bb)
- [Luo Fuli announcement (Sina)](https://www.sina.cn/weibo/detail/5346582186688756.html)
- [XiaomiMiMo on Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-V2.5) (V2.x only as of this date)
- `docs/scoping-qwen38-flash-next.md` (sibling scoping doc; same trigger discipline)

<!-- doc-reviewed: 2026-09-23 -->

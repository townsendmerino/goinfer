# Gemma 4 E2B Tier 2 candidate — blocked on a real safetensors loader gap, not measured

**Why this exists.** `docs/prompts/mac-demo-chat-apple-silicon-numbers.md` item 2 asked for a
`BenchmarkDecode`-equivalent number on `google/gemma-4-E2B-it`, having already downloaded it to
`~/models/gemma-4-E2B-unq` (a raw HF safetensors checkpoint, not a `.gguf`). Loading it fails, and
the cause is not the asset-registry directory-vs-file mismatch the prompt anticipated.

## The actual failure

```
decoder.Load("~/models/gemma-4-E2B-unq", Options{Quant: "int8int8"})
  -> safetensors: tensor "model.language_model.layers.15.self_attn.k_proj.weight" not found
```

## Root cause

The checkpoint's `config.json` (`text_config`) has:

```
"model_type": "gemma4_text", "num_hidden_layers": 35, "num_kv_shared_layers": 20
```

`num_kv_shared_layers` is a REAL, already-modeled config field
(`decoder/config.go:287` `SharedKVLayers`, read into `Architecture.gemma4.SharedKVLayers` at
`decoder/registry.go:371`) — the last N layers of a Gemma-4 E-model carry no `k_proj`/`v_proj` at
all and reuse an earlier layer's KV cache. Confirmed directly against the checkpoint: layer 15 (of
35, with `SharedKVLayers=20` meaning layers 15–34 are shared) has `q_proj`/`o_proj`/`q_norm` but
genuinely no `k_proj`/`v_proj`/`k_norm` tensors on disk — this is not a corrupt download.

**The GGUF loader already handles this** (`decoder/gguf.go:2412-2365`: computes
`firstShared := arch.NumLayers - g4.SharedKVLayers` and sets `l.KVShared`/skips k/v loading past
that point). **The safetensors loader does not** — `grep SharedKVLayers decoder/weights.go` finds
nothing; the safetensors path loads every layer's k/v unconditionally and fails the moment it hits
a shared layer.

This is a real, scoped architecture-support gap, not a benchmark-harness problem — the fix is
porting the GGUF path's `firstShared`/`KVShared` logic into `buildGemma4Weights` (or wherever the
generic gemma4 safetensors loader lives), then validating against an HF oracle the way every other
family in `docs/capability-matrix.md` is. That row's "full-oracle 100.0%/0.99128" for Gemma 4
does NOT cover this: the matrix's MoE column ("dense ‖ sparse, no-shared") describes MoE
shared-*experts*, unrelated to this attention KV-*layer*-sharing knob — it is not evidence this
checkpoint shape was ever oracle-tested.

## Verdict for the prompt's actual question

**Not measured, not "does it clear the bar."** Gemma 4 E2B cannot be evaluated as a Tier 2
candidate via the safetensors checkpoint until this loader gap is closed. Options for whoever picks
this up: (a) port the GGUF path's shared-KV handling into the safetensors loader (the real fix,
needed regardless of this benchmark), or (b) benchmark against a `.gguf` build of E2B instead if a
non-shared-KV or GGUF-converted asset is available (the repo already has
`~/models/gemma-4-E2B_q4_0-it.gguf`, but that's pre-quantized to q4_0, not the f32/bf16 source
`BenchmarkDecode`'s int8int8 path expects — a fresh f32/bf16 GGUF export would be needed).

Declined rather than worked around: patching around a missing k/v tensor in a throwaway benchmark
harness would either crash later (KVCache reads for a "shared" layer expect the sharing wiring to
exist) or silently produce wrong numbers — worse than reporting no number at all.

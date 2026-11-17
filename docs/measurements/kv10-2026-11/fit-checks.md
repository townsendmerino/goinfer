# KV10: fit checks, 2026-11-16 (the server's own plan, `serve --backend cuda --kv-sessions 1`, 8 GB RTX 2070 SUPER)

Each load was started with the depth as `--ctx` and the KV precision as `--kv`, and the banner read; nothing was generated. A load the plan declines says so in its first line. Banner lines are verbatim (cut at 260 characters).

## qwen2.5-coder-1.5b-instruct-q4_k_m.gguf, --ctx 131072, --kv f16

```
  context: 32768 tokens (model maximum; --ctx 131072 is above it) · KV f16 (requested f16, lossy)
  fit: 32768-token cap needs 0.9 GB KV (weights 0.9 GB, budget 41.9 GB, 41.0 GB left for a request's own prefill)
  KV plan: 1 conversation x 32768 positions (--ctx 131072)
```

## qwen2.5-coder-1.5b-instruct-q4_k_m.gguf, --ctx 65536, --kv f32

```
  context: 32768 tokens (model maximum; --ctx 65536 is above it) · KV f32 (requested f32)
  fit: 32768-token cap needs 1.8 GB KV (weights 0.9 GB, budget 41.9 GB, 40.2 GB left for a request's own prefill)
  KV plan: 1 conversation x 32768 positions (--ctx 65536)
```

## qwen2.5-coder-1.5b-instruct-q4_k_m.gguf, --ctx 131072, --kv f32

```
  context: 32768 tokens (model maximum; --ctx 131072 is above it) · KV f32 (requested f32)
  fit: 32768-token cap needs 1.8 GB KV (weights 0.9 GB, budget 41.9 GB, 40.2 GB left for a request's own prefill)
  KV plan: 1 conversation x 32768 positions (--ctx 131072)
```

## lfm25-2.6b, --ctx 131072, --kv f16

```
load model (/home/francis/models/lfm25-2.6b.int4.e4h.cuda.v17.giw): --kv f16 cannot be delivered on the cpu KV cache: this family's own forward loop keeps an f32 CPU cache until phase 4
```

## lfm25-2.6b, --ctx 131072, --kv f32

```
  context: 131072 tokens (model maximum) · KV f32 (requested f32) — the CPU path has no --ctx cap: KV is allocated per request, so this is a ceiling, not memory held; a GPU-resident load caps it with --ctx, but this model declined one (see decode path)
  fit: 131072-token cap needs 4.0 GB KV (weights 1.9 GB, budget 41.8 GB, 37.8 GB left for a request's own prefill) — a ceiling, reached only by a request that fills the window
```

## qwen3.5-0.8b, --ctx 131072, --kv f16

```
  context: 131072 tokens (--ctx; model maximum 262144) · KV f16 (requested f16, lossy)
  fit: 131072-token cap needs 1.5 GB KV (weights 0.4 GB, budget 41.9 GB, 40.4 GB left for a request's own prefill)
  KV plan: 1 conversation x 131072 positions (--ctx 131072; 0.7 GB held back for the vision tower)
```

## qwen3.5-0.8b, --ctx 131072, --kv f32

```
  context: 131072 tokens (--ctx; model maximum 262144) · KV f32 (requested f32)
  fit: 131072-token cap needs 3.0 GB KV (weights 0.4 GB, budget 41.9 GB, 38.9 GB left for a request's own prefill)
  KV plan: 1 conversation x 131072 positions (--ctx 131072; 0.7 GB held back for the vision tower)
```

## smollm3-3b, --ctx 65536, --kv f16

```
[resident] BuildResident declined (ok=false err=cuda: resident context 65536 positions needs 9.81 GB of KV (146.3 KB/position across 36 layers) but only 5.61 GB is free on the device beside the weights (plus 262 MB reserved for driver and decode scratch) — l
  context: 65536 tokens (model maximum) · KV f16 (requested f16, lossy) — the CPU path has no --ctx cap: KV is allocated per request, so this is a ceiling, not memory held; a GPU-resident load caps it with --ctx, but this model declined one (see decode path
  fit: 65536-token cap needs 4.5 GB KV (weights 1.6 GB, budget 41.9 GB, 37.4 GB left for a request's own prefill) — a ceiling, reached only by a request that fills the window
```

## qwen2.5-vl-3b, --ctx 65536, --kv f16

```
[resident] BuildResident declined (ok=false err=cuda: resident context 65536 positions needs 2.57 GB of KV (38.3 KB/position across 36 layers) plus 3.46 GB reserved for a companion attach (--drafter) but only 5.58 GB is free on the device beside the weights (p
  context: 128000 tokens (model maximum) · KV f16 (requested f16, lossy) — the CPU path has no --ctx cap: KV is allocated per request, so this is a ceiling, not memory held; a GPU-resident load caps it with --ctx, but this model declined one (see decode pat
  fit: 65536-token cap needs 2.2 GB KV (weights 1.6 GB, budget 41.9 GB, 39.7 GB left for a request's own prefill)
```

## qwen2.5-vl-3b, --ctx 65536, --kv f32

```
[resident] BuildResident declined (ok=false err=cuda: resident context 65536 positions needs 4.98 GB of KV (74.3 KB/position across 36 layers) plus 3.46 GB reserved for a companion attach (--drafter) but only 5.58 GB is free on the device beside the weights (p
  context: 128000 tokens (model maximum) · KV f32 (requested f32) — the CPU path has no --ctx cap: KV is allocated per request, so this is a ceiling, not memory held; a GPU-resident load caps it with --ctx, but this model declined one (see decode path)
  fit: 65536-token cap needs 4.5 GB KV (weights 1.6 GB, budget 41.9 GB, 37.4 GB left for a request's own prefill)
```


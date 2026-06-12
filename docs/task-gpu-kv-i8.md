# Task (goinfer): int8 GPU KV cache (4× VRAM KV; ~64k context on 8 GB)

> **For:** `~/tmcode/goinfer/gpu` (`-tags gpu`, the RTX box). The third rung of
> the ladder f32 → f16 (`--kv f16`, landed 2026-06-10, `task-gpu-f16-kv.md`)
> → **int8**. Same knob, same gate shape, one more halving. Lossy ⇒ opt-in;
> f32 default stays bit-exact, f16 stays as shipped.

## Problem / opportunity

f16 KV made 32k fit the 8 GB card by halving the f32 cache (32k f16 ≈ 16k f32
≈ 1.88 GB, peak 6912 MiB). int8 halves it again:

| 7B int4 resident + KV | KV bytes | peak est. | context on 8 GB |
|---|---|---|---|
| 16k f32 (v1 cap) | 1.88 GB | 6926 MiB | 16k |
| 32k f16 (shipped) | 1.88 GB | 6912 MiB | 32k |
| **32k int8** | **0.94 GB** | **~5950 MiB** | 32k + ~1 GB freed headroom |
| **64k int8** | **1.88 GB** | **~6912 MiB** | **64k — same envelope as 32k f16** |

Two ways to spend it: **64k context** (long-doc / agent-transcript story), or
~1 GB freed VRAM at 32k (bigger resident model class on small cards). It also
finishes the unmeasured f16 story: the long-context attention kernel is
KV-read-bound, and int8 halves the bytes f16 left on the table.

## Building blocks (all shipped)

- **GPU-side int8 infrastructure exists end-to-end**: the W8A8 resident path
  already quantizes activations *on device* and dots int8×int8 in WGSL
  (`gpu/quant.go` — `runQuant`, `ResidentW8A8` with `bq` + `bScales`
  bindings). KV int8 reuses the same store-quantized/read-with-scale idiom.
- The three KV kernels are isolated (`ropeStore`, `vStore`, `attnShaderWGSL`)
  and already branch once for the f16 pack — int8 is a second arm of the same
  branch, not a new seam.
- Scale storage: per (position, KV-head) f32 scales — the CPU doc's
  granularity (`task-cpu-kv-quant.md` §Design), <1% overhead, and one scheme
  shared across CPU and GPU keeps snapshots/parity reasoning uniform.
- Gate machinery: `TestKVCacheF16_parity` is the template (long-context
  cosine + argmax with the 3%-near-tie rule, real-HW, software-adapter-skip).

## Design notes (the deltas vs f16)

- Cache buffer: `array<u32>`, 4 int8/word (vs 2 f16/word); scales in a small
  side buffer indexed `[pos*nKV + head]`. No `shader-f16`/dp4a feature
  needed — unpack is shifts+masks, scores multiply by `qScale·kScale` in f32.
  (When `dot4I8Packed` ever lands in the binding, the attention dot gets the
  same DP4A boost the prefill item is gated on — free future upside.)
- `ropeStore`/`vStore` compute the sub-row absmax → scale → quantize at
  write. One position × one head per invocation already; the reduction is
  over `headDim` (64–128) — trivial in-workgroup.
- Quality expectation: KV int8 per-head ≫ f16's 10-bit mantissa in dynamic
  range terms for post-norm K/V; the CPU doc argues the same bar
  (argmax-preserved, cosine ≥0.999 target, accept ≥0.99 with near-tie rule).
  If keys (RoPE outliers) miss the bar, fall back to **int8 keys per-channel /
  values per-token** (KIVI split) before dropping to f16-keys+int8-values.
- Residency fit math is static (`Size:` fields in decodelayer.go) — the cap
  bump to 64k is arithmetic, gated by a real allocation test like
  `TestKVCacheF16_fit`.

## Increments

1. **int8 KV storage + the three kernels**, knob `--kv i8` joining
   `f16|f32` (`Options.KVPrecision`). Gate: int8-KV vs f32-KV decode at ≥8k
   keys on real HW — argmax preserved (3% rule), cosine ≥0.99 (target 0.999);
   f32 and f16 paths unchanged (existing parities green).
2. **64k cap when int8** + fit gate: 7B int4 + 64k int8 KV on the 8 GB card,
   no OOM, peak recorded (~6.9 GiB expected). Record the long-context decode
   tok/s delta vs f16 at equal context — this finally measures the
   "KV-read-bound" speedup the f16 doc left open, with 4× instead of 2× of
   lever.

## Estimated impact

| metric | f16 (shipped) | int8 |
|---|---|---|
| KV VRAM @ 32k, 7B | 1.88 GB | **0.94 GB** |
| max context, 7B int4 on 8 GB | 32k | **~64k** |
| freed headroom @ 32k | — | ~1 GB (next model class up, or batch room) |
| long-ctx decode tok/s | unmeasured (predicted >1× at 16k+) | predicted **1.2–1.8× vs f32 at 16k+** (half f16's bytes; measure, don't claim) |
| short-ctx decode | 0.99× measured | ~same (weight-stream-bound) |
| quality | cosine 0.99868 @ 8k | ≥0.99 gate; KIVI split as fallback |

## Effort / when

~3–4 days on the RTX box (kernel branch + knob + two gates), low-medium risk.
**Trigger:** same as f16's was — a real >32k workload, or marketing the 8 GB
story harder. Sequence *after* the CPU int8 cache lands so the granularity
and gate bars arrive pre-validated.
